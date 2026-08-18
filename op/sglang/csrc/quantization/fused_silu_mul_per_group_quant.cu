// 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
// Reserved.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <torch/all.h>

#include "../dispatch_utils.h"

#ifndef USE_ROCM
  #include <cub/cub.cuh>
  #include <cub/util_type.cuh>
  #include <c10/util/Float8_e4m3fn.h>
  #define MAYBE_HOST_DEVICE C10_HOST_DEVICE
#else
  #include <ATen/hip/HIPContext.h>
  #include <c10/util/Float8_e4m3fn.h>
  #include <c10/util/Float8_e4m3fnuz.h>
  #include <hipcub/hipcub.hpp>
  #include <hipcub/util_type.hpp>
  // ROCm doesn't seem to need C10_HOST_DEVICE for static constexpr
  #define MAYBE_HOST_DEVICE
#endif


namespace vllm {
static constexpr int kSiluMulGroupSize = 128;
static constexpr int kWarpSize = 64;

template <typename t,
          typename = std::enable_if_t<std::is_same_v<t, c10::Float8_e4m3fn> ||
                                      std::is_same_v<t, c10::Float8_e4m3fnuz> ||
                                      std::is_same_v<t, int8_t>>>
struct quant_type_max {
  static constexpr t val() { return std::numeric_limits<t>::max(); }
};

// using the default max value from pytorch (240.0 0x7f) will cause accuracy
// issues when running dynamic quantization. here use 224.0 0x7e for rocm.
template <>
struct quant_type_max<c10::Float8_e4m3fnuz> {
  static constexpr c10::Float8_e4m3fnuz val() {
    return c10::Float8_e4m3fnuz(0x7e, c10::Float8_e4m3fnuz::from_bits());
  }
};

template <typename T>
MAYBE_HOST_DEVICE static constexpr T quant_type_max_v =
    quant_type_max<T>::val();

static __forceinline__ __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate
  dst = std::clamp(dst, i8_min, i8_max);
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  //uint32_t dst;
  //asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  //return reinterpret_cast<const int8_t&>(dst);
  int32_t dst;
  dst = __float2int_rn(x);
  dst = min(dst, 127);
  dst = max(dst, -128);
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

template <typename fp8_type>
static __device__ __forceinline__ fp8_type float_to_fp8(float const x) {
  float const r =
      fmax(-quant_type_max_v<fp8_type>, fmin(x, quant_type_max_v<fp8_type>));
  return static_cast<fp8_type>(r);
}

template <typename quant_type_t, bool is_scale_inverted, typename enable = void>
struct ScaledQuant;

template <typename quant_type_t, bool is_scale_inverted>
struct ScaledQuant<
    quant_type_t, is_scale_inverted,
    typename std::enable_if_t<std::is_same_v<quant_type_t, int8_t>>> {
  static __device__ __forceinline__ quant_type_t quant_fn(float const x,
                                                          float const scale) {
    if constexpr (is_scale_inverted) {
      return float_to_int8_rn(x * scale);
    } else {
      return float_to_int8_rn(x / scale);
    }
  }
};

template <typename quant_type_t, bool is_scale_inverted>
struct ScaledQuant<quant_type_t, is_scale_inverted,
                   typename std::enable_if_t<
                       std::is_same_v<quant_type_t, c10::Float8_e4m3fn> ||
                       std::is_same_v<quant_type_t, c10::Float8_e4m3fnuz>>> {
  static __device__ __forceinline__ quant_type_t quant_fn(float const x,
                                                          float const scale) {
    if constexpr (is_scale_inverted) {
      return float_to_fp8<quant_type_t>(x * scale);
    } else {
      return float_to_fp8<quant_type_t>(x / scale);
    }
  }
};

template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedArray {
  T data[N];
};

static inline int64_t ceil_div(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

static inline bool is_aligned(const void* ptr, uintptr_t alignment) {
  return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

template <typename T, bool kHasLimit>
__device__ __forceinline__ float silu_mul_value(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    int64_t idx,
    float swiglu_limit) {
  const float gate_v = static_cast<float>(gate[idx]);
  const float up_v = static_cast<float>(up[idx]);

  const float silu =
      gate_v * __builtin_mxc_rcpf(1.0f + __builtin_expf(-gate_v));

  float v = silu * up_v;

  if constexpr (kHasLimit) {
    v = fmaxf(-swiglu_limit, fminf(v, swiglu_limit));
  }

  return v;
}

template <typename quant_t>
__device__ __forceinline__ float quant_qmax() {
  if constexpr (std::is_same_v<quant_t, int8_t>) {
    return 127.0f;
  } else {
    return 448.0f;
  }
}

template <typename quant_t>
__device__ __forceinline__ float quant_min_absmax() {
  if constexpr (std::is_same_v<quant_t, int8_t>) {
    return 127.0f * std::numeric_limits<float>::epsilon();
  } else {
    return 1.0f / 512.0f;
  }
}

template <typename quant_t>
__device__ __forceinline__ quant_t do_quant(float x, float inv_scale) {
  return ScaledQuant<quant_t, true>::quant_fn(x, inv_scale);
}

// ============================================================
// default kernel
// 1 block = 1 token + 1 group
// blockDim = 128
// ============================================================
template <typename input_t, typename quant_t, bool kHasLimit>
__global__ void fused_silu_mul_per_group_quant_default_kernel(
    quant_t* __restrict__ out,
    float* __restrict__ scales,
    const input_t* __restrict__ input,
    int64_t hidden,
    float swiglu_limit) {
  constexpr int GROUP = kSiluMulGroupSize;

  const int group_id = blockIdx.x;
  const int token_id = blockIdx.y;
  const int tid = threadIdx.x;

  const int64_t col = static_cast<int64_t>(group_id) * GROUP + tid;

  const int64_t input_row =
      static_cast<int64_t>(token_id) * hidden * 2;
  const int64_t output_row =
      static_cast<int64_t>(token_id) * hidden;

  const input_t* gate = input + input_row;
  const input_t* up = gate + hidden;

  float val = 0.0f;
  float abs_val = 0.0f;

  if (col < hidden) {
    val = silu_mul_value<input_t, kHasLimit>(gate, up, col, swiglu_limit);
    abs_val = fabsf(val);
  }

  __shared__ float smem[GROUP];

  smem[tid] = abs_val;
  __syncthreads();

#pragma unroll
  for (int stride = GROUP >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }

  const float qmax = quant_qmax<quant_t>();
  const float absmax = fmaxf(smem[0], quant_min_absmax<quant_t>());
  const float scale = absmax / qmax;
  const float inv_scale = qmax * __builtin_mxc_rcpf(absmax);

  const int64_t groups = hidden / GROUP;

  if (tid == 0) {
    scales[static_cast<int64_t>(token_id) * groups + group_id] = scale;
  }

  if (col < hidden) {
    out[output_row + col] = do_quant<quant_t>(val, inv_scale);
  }
}

// ============================================================
// vec kernel
//
// VEC=2:
//   subgroup = 64 lanes
//   1 warp = 1 group
//
// VEC=4:
//   subgroup = 32 lanes
//   1 warp = 2 groups
//
// VEC=8:
//   subgroup = 16 lanes
//   1 warp = 4 groups
//
// 每个线程向量化读取 gate/up:
//   gate_vec = *(AlignedArray<input_t, VEC>*)
//   up_vec   = *(AlignedArray<input_t, VEC>*)
// ============================================================
template <typename input_t, typename quant_t, int VEC, bool kHasLimit>
__global__ void fused_silu_mul_per_group_quant_vec_kernel(
    quant_t* __restrict__ out,
    float* __restrict__ scales,
    const input_t* __restrict__ input,
    int64_t hidden,
    int64_t groups,
    float swiglu_limit) {
  static_assert(VEC == 2 || VEC == 4 || VEC == 8);

  constexpr int GROUP = kSiluMulGroupSize;
  constexpr int SUBGROUP_LANES = GROUP / VEC;
  constexpr int GROUPS_PER_WARP = kWarpSize / SUBGROUP_LANES;
  constexpr unsigned long long FULL_MASK = 0xffffffffffffffffULL;

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = tid & (kWarpSize - 1);

  const int subgroup_id = lane / SUBGROUP_LANES;
  const int subgroup_lane = lane & (SUBGROUP_LANES - 1);

  const int warps_per_block = blockDim.x / kWarpSize;

  const int64_t warp_tile =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id;

  const int64_t group_id =
      warp_tile * GROUPS_PER_WARP + subgroup_id;

  if (group_id >= groups) {
    return;
  }

  const int token_id = blockIdx.y;

  const int64_t group_start = group_id * GROUP;
  const int64_t col = group_start + subgroup_lane * VEC;

  const int64_t input_row =
      static_cast<int64_t>(token_id) * hidden * 2;
  const int64_t output_row =
      static_cast<int64_t>(token_id) * hidden;

  const input_t* gate = input + input_row;
  const input_t* up = gate + hidden;

  using InVec = AlignedArray<input_t, VEC>;

  const InVec gate_vec =
      *reinterpret_cast<const InVec*>(gate + col);
  const InVec up_vec =
      *reinterpret_cast<const InVec*>(up + col);

  float vals[VEC];
  float local_absmax = 0.0f;

#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    const float gate_v = static_cast<float>(gate_vec.data[i]);
    const float up_v = static_cast<float>(up_vec.data[i]);

    const float silu =
        gate_v * __builtin_mxc_rcpf(1.0f + __builtin_expf(-gate_v));

    float v = silu * up_v;

    if constexpr (kHasLimit) {
      v = fmaxf(-swiglu_limit, fminf(v, swiglu_limit));
    }

    vals[i] = v;
    local_absmax = fmaxf(local_absmax, fabsf(v));
  }

#pragma unroll
  for (int offset = SUBGROUP_LANES >> 1; offset > 0; offset >>= 1) {
    const float other =
        __shfl_xor_sync(FULL_MASK, local_absmax, offset, SUBGROUP_LANES);
    local_absmax = fmaxf(local_absmax, other);
  }

  const float qmax = quant_qmax<quant_t>();
  const float absmax = fmaxf(local_absmax, quant_min_absmax<quant_t>());
  const float scale = absmax / qmax;
  const float inv_scale = qmax * __builtin_mxc_rcpf(absmax);

  if (subgroup_lane == 0) {
    scales[static_cast<int64_t>(token_id) * groups + group_id] = scale;
  }

  using OutVec = AlignedArray<quant_t, VEC>;
  OutVec out_vec;

#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    out_vec.data[i] = do_quant<quant_t>(vals[i], inv_scale);
  }

  *reinterpret_cast<OutVec*>(out + output_row + col) = out_vec;
}

// ============================================================
// launchers
// ============================================================
template <typename input_t, typename quant_t, bool kHasLimit>
void launch_fused_silu_mul_per_group_quant_default(
    quant_t* out,
    float* scales,
    const input_t* input,
    int64_t tokens,
    int64_t hidden,
    float swiglu_limit,
    cudaStream_t stream) {
  constexpr int GROUP = kSiluMulGroupSize;

  const int64_t groups = hidden / GROUP;

  dim3 grid(groups, tokens);
  dim3 block(GROUP);

  fused_silu_mul_per_group_quant_default_kernel<input_t, quant_t, kHasLimit>
      <<<grid, block, 0, stream>>>(out, scales, input, hidden, swiglu_limit);
}

template <typename input_t, typename quant_t, int VEC, int BLOCK_THREADS,
          bool kHasLimit>
void launch_fused_silu_mul_per_group_quant_vec(
    quant_t* out,
    float* scales,
    const input_t* input,
    int64_t tokens,
    int64_t hidden,
    float swiglu_limit,
    cudaStream_t stream) {
  static_assert(VEC == 2 || VEC == 4 || VEC == 8);
  static_assert(BLOCK_THREADS == 64 ||
                BLOCK_THREADS == 128 ||
                BLOCK_THREADS == 256 ||
                BLOCK_THREADS == 512);

  constexpr int GROUP = kSiluMulGroupSize;
  constexpr int SUBGROUP_LANES = GROUP / VEC;
  constexpr int GROUPS_PER_WARP = kWarpSize / SUBGROUP_LANES;
  constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / kWarpSize;

  const int64_t groups = hidden / GROUP;

  const int64_t warp_tiles = ceil_div(groups, GROUPS_PER_WARP);
  const int64_t grid_x = ceil_div(warp_tiles, WARPS_PER_BLOCK);

  dim3 grid(grid_x, tokens);
  dim3 block(BLOCK_THREADS);

  fused_silu_mul_per_group_quant_vec_kernel<input_t, quant_t, VEC, kHasLimit>
      <<<grid, block, 0, stream>>>(out, scales, input, hidden, groups,
                                    swiglu_limit);
}

// ============================================================
// dispatch
// ============================================================
template <typename input_t, typename quant_t, bool kHasLimit>
void dispatch_fused_silu_mul_per_group_quant(
    quant_t* out,
    float* scales,
    const input_t* input,
    int64_t tokens,
    int64_t hidden,
    bool can_vec8,
    bool can_vec4,
    bool can_vec2,
    float swiglu_limit,
    cudaStream_t stream) {
  if (can_vec8) {
    if (hidden == 128) {
      launch_fused_silu_mul_per_group_quant_default<input_t, quant_t, kHasLimit>(
          out, scales, input, tokens, hidden, swiglu_limit, stream);
    } else if (hidden <= 512) {
      launch_fused_silu_mul_per_group_quant_vec<input_t, quant_t, 8, 64, kHasLimit>(
          out, scales, input, tokens, hidden, swiglu_limit, stream);
    } else if (hidden <= 1024) {
      launch_fused_silu_mul_per_group_quant_vec<input_t, quant_t, 8, 128, kHasLimit>(
          out, scales, input, tokens, hidden, swiglu_limit, stream);
    } else if (hidden <= 2048) {
      launch_fused_silu_mul_per_group_quant_vec<input_t, quant_t, 8, 256, kHasLimit>(
          out, scales, input, tokens, hidden, swiglu_limit, stream);
    } else {
      launch_fused_silu_mul_per_group_quant_vec<input_t, quant_t, 8, 512, kHasLimit>(
          out, scales, input, tokens, hidden, swiglu_limit, stream);
    }
    return;
  }

  if (can_vec4) {
    if (hidden == 128) {
      launch_fused_silu_mul_per_group_quant_default<input_t, quant_t, kHasLimit>(
          out, scales, input, tokens, hidden, swiglu_limit, stream);
    } else if (hidden <= 512) {
      launch_fused_silu_mul_per_group_quant_vec<input_t, quant_t, 4, 64, kHasLimit>(
          out, scales, input, tokens, hidden, swiglu_limit, stream);
    } else if (hidden <= 1024) {
      launch_fused_silu_mul_per_group_quant_vec<input_t, quant_t, 4, 128, kHasLimit>(
          out, scales, input, tokens, hidden, swiglu_limit, stream);
    } else if (hidden <= 2048) {
      launch_fused_silu_mul_per_group_quant_vec<input_t, quant_t, 4, 256, kHasLimit>(
          out, scales, input, tokens, hidden, swiglu_limit, stream);
    } else {
      launch_fused_silu_mul_per_group_quant_vec<input_t, quant_t, 4, 512, kHasLimit>(
          out, scales, input, tokens, hidden, swiglu_limit, stream);
    }
    return;
  }

  if (can_vec2) {
    if (hidden == 128) {
      launch_fused_silu_mul_per_group_quant_default<input_t, quant_t, kHasLimit>(
          out, scales, input, tokens, hidden, swiglu_limit, stream);
    } else if (hidden <= 512) {
      launch_fused_silu_mul_per_group_quant_vec<input_t, quant_t, 2, 64, kHasLimit>(
          out, scales, input, tokens, hidden, swiglu_limit, stream);
    } else if (hidden <= 1024) {
      launch_fused_silu_mul_per_group_quant_vec<input_t, quant_t, 2, 128, kHasLimit>(
          out, scales, input, tokens, hidden, swiglu_limit, stream);
    } else if (hidden <= 2048) {
      launch_fused_silu_mul_per_group_quant_vec<input_t, quant_t, 2, 256, kHasLimit>(
          out, scales, input, tokens, hidden, swiglu_limit, stream);
    } else {
      launch_fused_silu_mul_per_group_quant_vec<input_t, quant_t, 2, 512, kHasLimit>(
          out, scales, input, tokens, hidden, swiglu_limit, stream);
    }
    return;
  }

  launch_fused_silu_mul_per_group_quant_default<input_t, quant_t, kHasLimit>(
      out, scales, input, tokens, hidden, swiglu_limit, stream);
}

} // namespace vllm

// Note: only support group size 128
//
// Optional swiglu_limit: when provided, the SiLU(gate) * up output is
// symmetrically clamped to [-swiglu_limit, +swiglu_limit] BEFORE the per-group
// absmax reduction and quantization. This bounds the dynamic range fed into the
// quantizer so that outliers in the activation do not blow up the per-group
// scale and destroy resolution for the rest of the group. When omitted, the
// kernel skips the clamp entirely (equivalent to an infinite limit).
void fused_silu_mul_per_group_quant(
    torch::Tensor& out,
    torch::Tensor& scales,
    const torch::Tensor& input,
    c10::optional<double> _swiglu_limit) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");

  TORCH_CHECK(input.device() == out.device(),
              "input and out must be on the same device");
  TORCH_CHECK(input.device() == scales.device(),
              "input and scales must be on the same device");

  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");

  TORCH_CHECK(input.dim() >= 2,
              "input must have at least 2 dimensions, got ",
              input.dim());

  TORCH_CHECK(input.numel() > 0, "input must be non-empty");

  const int64_t input_hidden = input.size(-1);

  TORCH_CHECK(input_hidden > 0,
              "input last dimension must be positive, got ",
              input_hidden);

  TORCH_CHECK((input_hidden % 2) == 0,
              "input last dimension must be even because it contains "
              "[gate, up], got ",
              input_hidden);

  const int64_t hidden = input_hidden / 2;

  TORCH_CHECK(hidden > 0,
              "hidden_size must be positive, got ",
              hidden);

  TORCH_CHECK((hidden % vllm::kSiluMulGroupSize) == 0,
              "hidden_size must be divisible by 128, got ",
              hidden);

  const int64_t tokens = input.numel() / input_hidden;
  const int64_t groups = hidden / vllm::kSiluMulGroupSize;

  TORCH_CHECK(out.dim() == input.dim(),
              "out dim must equal input dim, got out.dim=",
              out.dim(),
              ", input.dim=",
              input.dim());

  for (int64_t i = 0; i < input.dim() - 1; ++i) {
    TORCH_CHECK(out.size(i) == input.size(i),
                "out shape must match input shape except last dimension, "
                "mismatch at dim ",
                i,
                ": out.size(",
                i,
                ")=",
                out.size(i),
                ", input.size(",
                i,
                ")=",
                input.size(i));
  }

  TORCH_CHECK(out.size(-1) == hidden,
              "out last dimension must be input.size(-1) / 2, got ",
              out.size(-1),
              ", expected ",
              hidden);

  TORCH_CHECK(scales.dim() == 2,
              "scales must be 2D with shape [num_tokens, hidden_size / 128], "
              "got dim=",
              scales.dim());

  TORCH_CHECK(scales.size(0) == tokens,
              "scales.size(0) must equal num_tokens, got ",
              scales.size(0),
              ", expected ",
              tokens);

  TORCH_CHECK(scales.size(1) == groups,
              "scales.size(1) must equal hidden_size / 128, got ",
              scales.size(1),
              ", expected ",
              groups);

  TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
              "scales dtype must be float32, got ",
              scales.scalar_type());

  TORCH_CHECK(input.scalar_type() == torch::kFloat16 ||
                  input.scalar_type() == torch::kBFloat16 ||
                  input.scalar_type() == torch::kFloat32,
              "input dtype must be float16, bfloat16 or float32, got ",
              input.scalar_type());

  const bool is_int8_out = out.scalar_type() == torch::kInt8;
  const bool is_fp8_out = out.dtype() == torch::kFloat8_e4m3fn;

  TORCH_CHECK(is_int8_out || is_fp8_out,
              "out dtype must be int8 or float8_e4m3fn, got ",
              out.dtype());

  // Safety CHECK on swiglu_limit per the explicit requirement.
  // When provided, it must be a strictly positive, finite scalar. A non-positive
  // or non-finite value would either collapse all activations to 0 (clamp to a
  // degenerate range) or poison the FP pipeline (NaN/Inf), making the output
  // meaningless. Default of 10.0 is applied when the caller omits the arg.
  constexpr double kDefaultSwigluLimit = 10.0;
  float swiglu_limit = 0.0f;
  bool use_limit = false;
  if (_swiglu_limit.has_value()) {
    const double v = *_swiglu_limit;
    TORCH_CHECK(std::isfinite(v),
                "swiglu_limit must be finite (got ", v, ")");
    TORCH_CHECK(v > 0.0,
                "swiglu_limit must be strictly positive (got ", v, ")");
    swiglu_limit = static_cast<float>(v);
    use_limit = true;
  } else {
    swiglu_limit = static_cast<float>(kDefaultSwigluLimit);
    use_limit = true;
  }

  const size_t input_elem_size = input.element_size();
  const size_t out_elem_size = out.element_size();

  const void* input_ptr = input.data_ptr();
  const void* out_ptr = out.data_ptr();

  const bool can_vec8 =
      (hidden % 8 == 0) &&
      vllm::is_aligned(input_ptr, input_elem_size * 8) &&
      vllm::is_aligned(out_ptr, out_elem_size * 8);

  const bool can_vec4 =
      (hidden % 4 == 0) &&
      vllm::is_aligned(input_ptr, input_elem_size * 4) &&
      vllm::is_aligned(out_ptr, out_elem_size * 4);

  const bool can_vec2 =
      (hidden % 2 == 0) &&
      vllm::is_aligned(input_ptr, input_elem_size * 2) &&
      vllm::is_aligned(out_ptr, out_elem_size * 2);

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (is_fp8_out) {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "fused_silu_mul_per_group_quant", [&] {
          if (use_limit) {
            vllm::dispatch_fused_silu_mul_per_group_quant<
                scalar_t,
                c10::Float8_e4m3fn,
                true>(
                reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr()),
                scales.data_ptr<float>(),
                input.data_ptr<scalar_t>(),
                tokens,
                hidden,
                can_vec8,
                can_vec4,
                can_vec2,
                swiglu_limit,
                stream);
          } else {
            vllm::dispatch_fused_silu_mul_per_group_quant<
                scalar_t,
                c10::Float8_e4m3fn,
                false>(
                reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr()),
                scales.data_ptr<float>(),
                input.data_ptr<scalar_t>(),
                tokens,
                hidden,
                can_vec8,
                can_vec4,
                can_vec2,
                swiglu_limit,
                stream);
          }
        });
  } else {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "fused_silu_mul_per_group_quant", [&] {
          if (use_limit) {
            vllm::dispatch_fused_silu_mul_per_group_quant<scalar_t, int8_t, true>(
                out.data_ptr<int8_t>(),
                scales.data_ptr<float>(),
                input.data_ptr<scalar_t>(),
                tokens,
                hidden,
                can_vec8,
                can_vec4,
                can_vec2,
                swiglu_limit,
                stream);
          } else {
            vllm::dispatch_fused_silu_mul_per_group_quant<scalar_t, int8_t, false>(
                out.data_ptr<int8_t>(),
                scales.data_ptr<float>(),
                input.data_ptr<scalar_t>(),
                tokens,
                hidden,
                can_vec8,
                can_vec4,
                can_vec2,
                swiglu_limit,
                stream);
          }
        });
  }
}
