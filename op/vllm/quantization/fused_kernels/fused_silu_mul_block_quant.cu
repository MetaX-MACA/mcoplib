// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../dispatch_utils.h"
#include "quant_conversions.cuh"
#include "../w8a8/fp8/common.cuh"

// Output dispatch for swiglu_step_and_mul_per_block_quant: fp8, int8, and the
// normalized bf16 path (scale = amax, out in [-1, 1]).
#define SWIGLU_MCOP_CASE_OUT(...)                     \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
  VLLM_DISPATCH_CASE_QUANT_TYPES(__VA_ARGS__)

#define SWIGLU_MCOP_DISPATCH_OUT(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, SWIGLU_MCOP_CASE_OUT(__VA_ARGS__))

namespace vllm {

typedef __NATIVE_VECTOR__(4, _Float16) swiglu_v4f16;


static __device__ __forceinline__ int8_t float_to_int8_rn_local(float const x) {
  int32_t dst;
  dst = __float2int_rn(x);
  dst = min(dst, 127);
  dst = max(dst, -127);
  return reinterpret_cast<const int8_t&>(dst);
}

// sigmoid(x) = 1/(1+exp(-x)). The MACA hardware exp path (__builtin_expf,
// translated to the native EXP instruction) is measurably faster than any
// hand-written ALU polynomial/rational approximation (guide pitfall #6) —
// A/B sweep measured 608 GB/s (expf) vs 483 GB/s (rational). Accuracy of the
// native expf is ~1ulp, easily clearing the fp8/int8/bf16 thresholds.
static __device__ __forceinline__ float fast_sigmoid(float const x) {
  return __builtin_mxc_rcpf(1.0f + __builtin_expf(-x));
}

template <typename scalar_out_t>
static __device__ __forceinline__ float swiglu_qmax() {
  if constexpr (std::is_same_v<scalar_out_t, at::BFloat16>) {
    return 1.0f;
  } else {
    return quant_type_max_v<scalar_out_t>;
  }
}

template <typename scalar_out_t>
static __device__ __forceinline__ float swiglu_min_scale() {
  if constexpr (std::is_same_v<scalar_out_t, at::BFloat16>) {
    return 1.1920928955078125e-7f;
  } else {
    return min_scaling_factor<scalar_out_t>::val();
  }
}

// ---------------------------------------------------------------------------
// swiglu_step_and_mul_per_block_quant
//   result = min(SiLU_alpha(gate), limit) * (clamp(up, -limit, limit) + beta)
//   SiLU_alpha(x) = x * sigmoid(alpha * x)
//
// Fast path (hidden % group_size == 0): flattened (token, group) dispatch,
// 16 elements per thread from four 128-bit loads, absmax reduced over the
// GROUP_SIZE/16 lanes of each group with XOR shuffles (offsets < 16, so the
// whole reduction lives inside one native 16-lane shuffle subgroup on MACA —
// no shared memory, no barriers), per-lane redundant scale computation, and
// quantization straight from registers (SREG). Scales are written by lane 0.
// ---------------------------------------------------------------------------
template <typename scalar_t, typename scalar_out_t, bool is_scale_transposed,
          int32_t GROUP_SIZE, bool HAS_SCALE_UB, int32_t NUM_THREADS = 256,
          int32_t ELEMS = 32>
__global__ __launch_bounds__(NUM_THREADS) void swiglu_step_fast_kernel(
    scalar_out_t* __restrict__ out,   // [M, H]
    float* __restrict__ scales,       // [M, G] or [G, M]
    scalar_t const* __restrict__ input,  // [M, 2H]
    float const* __restrict__ scale_ub,  // optional scalar upper bound
    float const limit, float const alpha, float const beta,
    int32_t const hidden_size, int32_t const groups_per_row,
    int64_t const total_groups, int64_t const num_tokens) {
  // ELEMS elements per thread from ELEMS/4 128-bit loads. LANES =
  // GROUP_SIZE/ELEMS controls the shuffle-reduction depth. int8/fp8 use
  // ELEMS=32 (deeper coalescing, one fewer reduction step); bf16 uses ELEMS=16
  // because its 2-byte stores are store-BW-bound and the wider variant spills.
  constexpr int LANES = GROUP_SIZE / ELEMS;
  constexpr int GROUPS_PER_BLOCK = NUM_THREADS / LANES;
  constexpr int NVEC = ELEMS / 4;           // number of float4 loads / lane
  float const QMAX = swiglu_qmax<scalar_out_t>();

  int const tid = threadIdx.x;
  int const lane = tid & (LANES - 1);
  int64_t gidx = (int64_t)blockIdx.x * GROUPS_PER_BLOCK + tid / LANES;
  bool const active = gidx < total_groups;
  if (!active) gidx = total_groups - 1;
  int64_t const token = gidx / groups_per_row;
  int32_t const group = (int32_t)(gidx % groups_per_row);

  int64_t const base = token * (int64_t)hidden_size * 2 +
                       (int64_t)group * GROUP_SIZE + lane * ELEMS;
  scalar_t const* pg = input + base;
  scalar_t const* pu = input + base + hidden_size;

  float4 vg[NVEC];
  float4 vu[NVEC];
#pragma unroll
  for (int k = 0; k < NVEC; k++) {
    vg[k] = *reinterpret_cast<float4 const*>(pg + k * 8);
    vu[k] = *reinterpret_cast<float4 const*>(pu + k * 8);
  }
  scalar_t const* gp = reinterpret_cast<scalar_t const*>(vg);
  scalar_t const* upp = reinterpret_cast<scalar_t const*>(vu);

  float reg[ELEMS];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < ELEMS; j++) {
    float const gate = (float)gp[j];
    float const up = (float)upp[j];
    float const act = gate * fast_sigmoid(alpha * gate);
    float const g = fminf(act, limit);
    float const u = fminf(fmaxf(up, -limit), limit) + beta;
    float const r = g * u;
    reg[j] = r;
    amax = fmaxf(amax, fabsf(r));
  }

#pragma unroll
  for (int off = LANES / 2; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffffffffffULL, amax, off, 16));

  // Compute scale and its reciprocal once per logical group. The source lane
  // is relative to the 16-lane shuffle segment because multiple logical
  // groups can share one segment when LANES is 2, 4, or 8.
  constexpr uint64_t FULL_MASK = 0xffffffffffffffffULL;
  int const subgroup_lane = tid & 15;
  int const group_lane0 = subgroup_lane & ~(LANES - 1);
  float scale = 0.0f;
  float inv = 0.0f;
  if (lane == 0) {
    scale = amax / QMAX;
    if constexpr (HAS_SCALE_UB) scale = fminf(scale, *scale_ub);
    scale = fmaxf(scale, swiglu_min_scale<scalar_out_t>());
    inv = 1.0f / scale;
  }
  scale = __shfl_sync(FULL_MASK, scale, group_lane0, 16);
  inv = __shfl_sync(FULL_MASK, inv, group_lane0, 16);
  if (lane == 0 && active) {
    float* sp = is_scale_transposed
                    ? scales + (int64_t)group * num_tokens + token
                    : scales + (int64_t)token * groups_per_row + group;
    *sp = scale;
  }
  if (!active) return;
  scalar_out_t* op = out + token * (int64_t)hidden_size +
                     (int64_t)group * GROUP_SIZE + lane * ELEMS;
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    alignas(16) int8_t q[ELEMS];
#pragma unroll
    for (int j = 0; j < ELEMS; j++) q[j] = float_to_int8_rn_local(reg[j] * inv);
#pragma unroll
    for (int k = 0; k < ELEMS / 16; k++)
      __stcg(reinterpret_cast<uint4*>(op + k * 16),
             *reinterpret_cast<uint4 const*>(q + k * 16));
  } else if constexpr (std::is_same_v<scalar_out_t, at::BFloat16>) {
    alignas(16) at::BFloat16 q[ELEMS];
#pragma unroll
    for (int j = 0; j < ELEMS; j++) {
      float r = reg[j] * inv;
      q[j] = static_cast<at::BFloat16>(fminf(fmaxf(r, -1.0f), 1.0f));
    }
#pragma unroll
    for (int k = 0; k < ELEMS / 8; k++)
      __stcg(reinterpret_cast<uint4*>(op + k * 8),
             *reinterpret_cast<uint4 const*>(q + k * 8));
  } else {
    using swiglu_v4f32 = float __attribute__((ext_vector_type(4)));
    alignas(16) uint32_t packed[ELEMS / 4];
#pragma unroll
    for (int j = 0; j < ELEMS; j += 4) {
      swiglu_v4f32 pk;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        float r = reg[j + t] * inv;
        pk[t] = fminf(fmaxf(r, -QMAX), QMAX);
      }
      packed[j / 4] = __builtin_mxc_cvt_pk4_f32tof8(pk);
    }
#pragma unroll
    for (int k = 0; k < ELEMS / 16; k++)
      __stcg(reinterpret_cast<uint4*>(op + k * 16),
             *reinterpret_cast<uint4 const*>(packed + k * 4));
  }
}

// ---------------------------------------------------------------------------
// Generic path
template <typename scalar_t, typename scalar_out_t, bool is_scale_transposed,
          int32_t GROUP_SIZE>
__global__ void swiglu_step_generic_kernel(
    scalar_out_t* __restrict__ out, float* __restrict__ scales,
    scalar_t const* __restrict__ input, float const* scale_ub,
    float const limit, float const alpha, float const beta,
    int32_t const hidden_size, int64_t const num_tokens) {
  int const token_idx = blockIdx.x;
  int const group_idx = blockIdx.y;
  int const tid = threadIdx.x;
  int const elem = group_idx * GROUP_SIZE + tid;
  bool const valid = elem < hidden_size;
  float const QMAX = swiglu_qmax<scalar_out_t>();

  __shared__ float shared_max[GROUP_SIZE];
  __shared__ float shared_scale;

  float result = 0.0f;
  if (valid) {
    int64_t const off =
        token_idx * (int64_t)hidden_size * 2 + elem;
    float const gate = (float)input[off];
    float const up = (float)input[off + hidden_size];
    float const act =
        gate * fast_sigmoid(alpha * gate);
    result = fminf(act, limit) * (fminf(fmaxf(up, -limit), limit) + beta);
  }
  shared_max[tid] = valid ? fabsf(result) : 0.0f;
  __syncthreads();
#pragma unroll
  for (int stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
    __syncthreads();
  }
  if (tid == 0) {
    float scale = shared_max[0] / QMAX;
    if (scale_ub != nullptr) scale = fminf(scale, *scale_ub);
    scale = fmaxf(scale, swiglu_min_scale<scalar_out_t>());
    int const num_groups = gridDim.y;
    float* sp =
        is_scale_transposed
            ? scales + (int64_t)group_idx * num_tokens + token_idx
            : scales + (int64_t)token_idx * num_groups + group_idx;
    *sp = scale;
    shared_scale = scale;
  }
  __syncthreads();
  if (valid) {
    if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
      out[token_idx * (int64_t)hidden_size + elem] =
          float_to_int8_rn_local(result / shared_scale);
    } else if constexpr (std::is_same_v<scalar_out_t, at::BFloat16>) {
      out[token_idx * (int64_t)hidden_size + elem] =
          static_cast<at::BFloat16>(
              fminf(fmaxf(result / shared_scale, -1.0f), 1.0f));
    } else {
      out[token_idx * (int64_t)hidden_size + elem] =
          ScaledQuant<scalar_out_t, false>::quant_fn(result, shared_scale);
    }
  }
}

// Logic: one thread block per (token, group) pair

template <typename scalar_t, typename scalar_out_t, bool is_scale_transposed,
          int32_t group_size>
__global__ void silu_and_mul_per_block_quant_kernel(
    scalar_out_t* __restrict__ out,  // Output: [num_tokens, hidden_size] in
                                     // FP8/INT8
    float* __restrict__ scales,      // Output: [num_tokens, hidden_size /
                                 // group_size] or [hidden_size / group_size,
                                 // num_tokens]
    scalar_t const* __restrict__ input,  // Input: [num_tokens, hidden_size * 2]
    float const* scale_ub,               // Optional scale upper bound
    int32_t const hidden_size  // Output hidden size (input is 2x this)
) {
  static_assert((group_size & (group_size - 1)) == 0,
                "group_size must be a power of 2 for correct reduction");

  // Grid: (num_tokens, num_groups)
  int const token_idx = blockIdx.x;
  int const group_idx = blockIdx.y;
  int const tid = threadIdx.x;  // tid in [0, group_size)
  int const num_tokens = gridDim.x;

  // Input layout: [gate || up] concatenated along last dimension
  int const input_stride = hidden_size * 2;
  int const group_start = group_idx * group_size;

  // Pointers to this token's data
  scalar_t const* token_input_gate =
      input + token_idx * input_stride + group_start;
  scalar_t const* token_input_up = token_input_gate + hidden_size;
  scalar_out_t* token_output = out + token_idx * hidden_size + group_start;

  // Scale pointer for this group
  int const num_groups = gridDim.y;
  float* group_scale_ptr = is_scale_transposed
                               ? scales + group_idx * num_tokens + token_idx
                               : scales + token_idx * num_groups + group_idx;

  // Shared memory for reduction (compile-time sized)
  __shared__ float shared_max[group_size];

  // Step 1: Each thread loads one element, computes SiLU, stores in register
  float gate = static_cast<float>(token_input_gate[tid]);
  float up = static_cast<float>(token_input_up[tid]);

  // Compute SiLU(gate) * up
  float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
  float silu_gate = gate * sigmoid_gate;
  float result = silu_gate * up;  // Keep in register

  // Step 2: Reduce to find group max
  shared_max[tid] = fabsf(result);
  __syncthreads();

// Power-of-2 reduction (group_size guaranteed to be power of 2)
#pragma unroll
  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }

  // Step 3: Compute scale (thread 0), broadcast via shared memory
  if (tid == 0) {
    float group_max = shared_max[0];

    float const quant_range = quant_type_max_v<scalar_out_t>;
    float group_scale = group_max / quant_range;

    // Apply scale upper bound if provided
    if (scale_ub != nullptr) {
      group_scale = fminf(group_scale, *scale_ub);
    }

    // Use minimum safe scaling factor
    group_scale = fmaxf(group_scale, swiglu_min_scale<scalar_out_t>());

    // Store scale to global memory
    *group_scale_ptr = group_scale;

    // Reuse shared_max[0] to broadcast scale
    shared_max[0] = group_scale;
  }
  __syncthreads();

  float group_scale = shared_max[0];

  // Step 4: Quantize and write output
  token_output[tid] =
      vllm::ScaledQuant<scalar_out_t, false>::quant_fn(result, group_scale);
}

}  // namespace vllm

void silu_and_mul_per_block_quant(torch::Tensor& out,
                                  torch::Tensor const& input,
                                  torch::Tensor& scales, int64_t group_size,
                                  std::optional<torch::Tensor> scale_ub,
                                  bool is_scale_transposed) {
  static c10::ScalarType kFp8Type = is_fp8_ocp()
                                        ? c10::ScalarType::Float8_e4m3fn
                                        : c10::ScalarType::Float8_e4m3fnuz;

  TORCH_CHECK(out.dtype() == kFp8Type || out.dtype() == torch::kInt8);
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous());
  TORCH_CHECK(
      input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16,
      "Input must be FP16 or BF16");
  TORCH_CHECK(scales.dtype() == torch::kFloat32, "Scales must be FP32");
  TORCH_CHECK(group_size == 128 || group_size == 64,
              "Unsupported group size: ", group_size);

  if (scale_ub.has_value()) {
    TORCH_CHECK(out.dtype() == kFp8Type);
  }

  int32_t hidden_size = out.size(-1);
  auto num_tokens = input.size(0);
  int32_t num_groups = hidden_size / group_size;

  TORCH_CHECK(input.size(-1) == hidden_size * 2,
              "input last dim must be 2x output hidden_size");
  TORCH_CHECK(hidden_size % group_size == 0,
              "hidden_size must be divisible by group_size");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(num_tokens, num_groups);
  dim3 block(group_size);

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_block_quant", [&] {
        using scalar_in_t = scalar_t;

        VLLM_DISPATCH_QUANT_TYPES(
            out.scalar_type(), "silu_and_mul_per_block_quant", [&] {
              using scalar_out_t = scalar_t;

              VLLM_DISPATCH_GROUP_SIZE(group_size, gs, [&] {
                VLLM_DISPATCH_BOOL(is_scale_transposed, transpose_scale, [&] {
                  vllm::silu_and_mul_per_block_quant_kernel<
                      scalar_in_t, scalar_out_t, transpose_scale, gs>
                      <<<grid, block, 0, stream>>>(
                          out.data_ptr<scalar_out_t>(),
                          scales.data_ptr<float>(),
                          input.data_ptr<scalar_in_t>(),
                          scale_ub.has_value() ? scale_ub->data_ptr<float>()
                                               : nullptr,
                          hidden_size);
                });
              });
            });
      });
}

void swiglu_step_and_mul_per_block_quant(
    torch::Tensor& out,  // [M, H], FP8/INT8
    torch::Tensor const& input,  // [M, 2H], FP16/BF16
    double limit,
    torch::Tensor& scales,  // FP32, [M, G] or [G, M]
    int64_t group_size,
    std::optional<torch::Tensor> scale_ub, bool is_scale_transposed,
    double alpha, double beta) {
  static c10::ScalarType kFp8Type = is_fp8_ocp()
                                        ? c10::ScalarType::Float8_e4m3fn
                                        : c10::ScalarType::Float8_e4m3fnuz;

  TORCH_CHECK(out.dtype() == kFp8Type || out.dtype() == torch::kInt8 ||
              out.dtype() == torch::kBFloat16);
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous());
  TORCH_CHECK(input.dtype() == torch::kFloat16 ||
                  input.dtype() == torch::kBFloat16,
              "Input must be FP16 or BF16");
  TORCH_CHECK(scales.dtype() == torch::kFloat32, "Scales must be FP32");
  TORCH_CHECK(group_size == 128 || group_size == 64,
              "Unsupported group size: ", group_size);
  TORCH_CHECK(std::isfinite(limit) && limit > 0.0,
              "limit must be finite and positive");
  if (scale_ub.has_value()) {
    TORCH_CHECK(out.dtype() == kFp8Type);
    TORCH_CHECK(scale_ub->is_contiguous());
    TORCH_CHECK(scale_ub->numel() == 1, "scale_ub must be a scalar");
  }

  int32_t const hidden_size = out.size(-1);
  int64_t const num_tokens = input.size(0);
  int32_t const num_groups = (hidden_size + group_size - 1) / group_size;

  TORCH_CHECK(input.size(-1) == hidden_size * 2,
              "input last dim must be 2x output hidden_size");
  if (is_scale_transposed) {
    TORCH_CHECK(scales.size(0) == num_groups && scales.size(1) == num_tokens,
                "scales must be [num_groups, num_tokens] when transposed");
  } else {
    TORCH_CHECK(scales.size(0) == num_tokens && scales.size(1) == num_groups,
                "scales must be [num_tokens, num_groups]");
  }

  if (num_tokens == 0 || hidden_size == 0) return;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // The fast kernel issues 16-byte vector loads/stores. Shape divisibility
  // alone does not guarantee that arbitrary contiguous storage is aligned.
  bool const vector_aligned =
      (reinterpret_cast<uintptr_t>(input.data_ptr()) & 0xF) == 0 &&
      (reinterpret_cast<uintptr_t>(out.data_ptr()) & 0xF) == 0;
  bool const aligned = (hidden_size % group_size == 0) && vector_aligned;

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "swiglu_step_and_mul_per_block_quant", [&] {
        using scalar_in_t = scalar_t;

        SWIGLU_MCOP_DISPATCH_OUT(
            out.scalar_type(), "swiglu_step_and_mul_per_block_quant", [&] {
              using scalar_out_t = scalar_t;
              float const l = (float)limit;
              float const a = (float)alpha;
              float const b = (float)beta;
              float const* ub_ptr =
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr;

              if (aligned) {
                VLLM_DISPATCH_BOOL(is_scale_transposed, transpose_scale, [&] {
                  VLLM_DISPATCH_BOOL(scale_ub.has_value(), has_ub, [&] {
                    VLLM_DISPATCH_GROUP_SIZE(group_size, gs, [&] {
                      constexpr int NT = 256;
                      // bf16 stores 2 bytes/elem and is store-BW bound: the
                      // 16-elem variant is faster and avoids register spills.
                      // int8/fp8 benefit from the 32-elem variant (one fewer
                      // reduction step + deeper coalescing).
                      constexpr int ELEMS =
                          std::is_same_v<scalar_out_t, at::BFloat16> ? 16 : 32;
                      constexpr int LANES = gs / ELEMS;
                      constexpr int GROUPS_PER_BLOCK = NT / LANES;
                      int64_t const total_groups =
                          (int64_t)num_tokens * num_groups;
                      int const grid =
                          (int)((total_groups + GROUPS_PER_BLOCK - 1) /
                                GROUPS_PER_BLOCK);
                      vllm::swiglu_step_fast_kernel<scalar_in_t, scalar_out_t,
                                              transpose_scale, gs, has_ub, NT,
                                              ELEMS>
                          <<<grid, NT, 0, stream>>>(
                              out.data_ptr<scalar_out_t>(),
                              scales.data_ptr<float>(),
                              input.data_ptr<scalar_in_t>(), ub_ptr, l, a, b,
                              hidden_size, num_groups, total_groups,
                              num_tokens);
                    });
                  });
                });
              } else {
                VLLM_DISPATCH_BOOL(is_scale_transposed, transpose_scale, [&] {
                  VLLM_DISPATCH_GROUP_SIZE(group_size, gs, [&] {
                    dim3 const grid(num_tokens, num_groups);
                    vllm::swiglu_step_generic_kernel<scalar_in_t, scalar_out_t,
                                               transpose_scale, gs>
                        <<<grid, gs, 0, stream>>>(
                            out.data_ptr<scalar_out_t>(),
                            scales.data_ptr<float>(),
                            input.data_ptr<scalar_in_t>(), ub_ptr, l, a, b,
                            hidden_size, num_tokens);
                  });
                });
              }
            });
      });
}