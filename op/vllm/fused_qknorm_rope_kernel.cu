/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <cuda_runtime.h>
#include <type_traits>

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "type_convert.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <optional>
#include <tuple>
#include <vector>


#define CHECK_TYPE(x, st)                                              \
  TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), \
              ", while ", st, " is expected")
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_TH_CUDA(x);    \
  CHECK_CONTIGUOUS(x)

#ifdef USE_ROCM
  #define FINAL_MASK 0xffffffffffffffffULL

  #if defined(HIP_VERSION) && HIP_VERSION < 70000000
// On ROCm versions before 7.0, __syncwarp isn't defined. The below
// implementation is copy/pasted from the implementation in ROCm 7.0
__device__ inline void __syncwarp() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
  #endif
#else
  #define FINAL_MASK 0xffffffff
#endif

namespace tensorrt_llm::common {
template <typename T, int num>
struct packed_as;
// Specialization for packed_as used in this kernel.
template <>
struct packed_as<uint, 1> {
  using type = uint;
};

template <>
struct packed_as<uint, 2> {
  using type = uint2;
};

template <>
struct packed_as<uint, 4> {
  using type = uint4;
};

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
inline __device__ __host__ T divUp(T m, T n) {
  return (m + n - 1) / n;
}

}  // namespace tensorrt_llm::common

namespace tensorrt_llm::kernels {
// NOTE(zhuhaoran): This kernel is adapted from TensorRT-LLM implementation,
// with added support for passing the cos_sin_cache as an input.
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu

// Perform per-head QK Norm and RoPE in a single kernel.
// scalar_t_in: data type of QKV and RMSNorm weights
// scalar_t_cache: data type of cos/sin cache
// head_dim: the dimension of each head
// interleave: interleave=!is_neox.
template <typename scalar_t_in, typename scalar_t_cache, int head_dim,
          bool interleave>
__global__ void fusedQKNormRopeKernel(
    void* qkv_void,                  // Combined QKV tensor
    int const num_heads_q,           // Number of query heads
    int const num_heads_k,           // Number of key heads
    int const num_heads_v,           // Number of value heads
    float const eps,                 // Epsilon for RMS normalization
    void const* q_weight_void,       // RMSNorm weights for query
    void const* k_weight_void,       // RMSNorm weights for key
    void const* cos_sin_cache_void,  // Pre-computed cos/sin cache
    int64_t const* position_ids,     // Position IDs for RoPE
    int const num_tokens,            // Number of tokens
    int const rotary_dim             // Dimension for RoPE
) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr ((std::is_same_v<scalar_t_in, c10::BFloat16>) ||
                std::is_same_v<scalar_t_cache, c10::BFloat16>) {
    return;
  } else {
#endif

    using Converter = vllm::_typeConvert<scalar_t_in>;
    static_assert(Converter::exists,
                  "Input QKV data type is not supported for this CUDA "
                  "architecture or toolkit version.");
    using T_in = typename Converter::hip_type;
    using T2_in = typename Converter::packed_hip_type;

    using CacheConverter = vllm::_typeConvert<scalar_t_cache>;
    static_assert(CacheConverter::exists,
                  "Cache data type is not supported for this CUDA architecture "
                  "or toolkit version.");
    using T_cache = typename CacheConverter::hip_type;

    T_in* qkv = reinterpret_cast<T_in*>(qkv_void);
    T_in const* q_weight = reinterpret_cast<T_in const*>(q_weight_void);
    T_in const* k_weight = reinterpret_cast<T_in const*>(k_weight_void);
    T_cache const* cos_sin_cache =
        reinterpret_cast<T_cache const*>(cos_sin_cache_void);

    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;

    // Calculate global warp index to determine which head/token this warp
    // processes
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    // Total number of attention heads (Q and K)
    int const total_qk_heads = num_heads_q + num_heads_k;

    // Determine which token and head type (Q or K) this warp processes
    int const tokenIdx = globalWarpIdx / total_qk_heads;
    int const localHeadIdx = globalWarpIdx % total_qk_heads;

    // Skip if this warp is assigned beyond the number of tokens
    if (tokenIdx >= num_tokens) return;

    bool const isQ = localHeadIdx < num_heads_q;
    int const headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;

    int const num_heads = num_heads_q + num_heads_k + num_heads_v;

    static_assert(head_dim % (32 * 2) == 0,
                  "head_dim must be divisible by 64 (each warp processes one "
                  "head, and each thread gets even number of "
                  "elements)");
    constexpr int numElemsPerThread = head_dim / 32;
    float elements[numElemsPerThread];
    constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
    static_assert(elemSizeBytes % 4 == 0,
                  "numSizeBytes must be a multiple of 4");
    constexpr int vecSize =
        elemSizeBytes /
        4;  // Use packed_as<uint, vecSize> to perform loading/saving.
    using vec_T = typename tensorrt_llm::common::packed_as<uint, vecSize>::type;

    int offsetWarp;  // Offset for the warp
    if (isQ) {
      // Q segment: token offset + head offset within Q segment
      offsetWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
    } else {
      // K segment: token offset + entire Q segment + head offset within K
      // segment
      offsetWarp = tokenIdx * num_heads * head_dim + num_heads_q * head_dim +
                   headIdx * head_dim;
    }
    int offsetThread = offsetWarp + laneId * numElemsPerThread;

    // Sum of squares for RMSNorm
    float sumOfSquares = 0.0f;

    // Load.
    {
      vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[offsetThread]);
      constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
#pragma unroll
      for (int i = 0; i < num_packed_elems; i++) {
        // Interpret the generic vector chunk as the specific packed type
        T2_in packed_val = *(reinterpret_cast<T2_in*>(&vec) + i);
        // Convert to float2 for computation
        float2 vals = Converter::convert(packed_val);
        sumOfSquares += vals.x * vals.x;
        sumOfSquares += vals.y * vals.y;

        elements[2 * i] = vals.x;
        elements[2 * i + 1] = vals.y;
      }
    }

    // Reduce sum across warp using the utility function
    sumOfSquares = tensorrt_llm::common::warpReduceSum(sumOfSquares);

    // Compute RMS normalization factor
    float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

    // Normalize elements
#pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      int dim = laneId * numElemsPerThread + i;
      float weight = isQ ? Converter::convert(q_weight[dim])
                         : Converter::convert(k_weight[dim]);
      elements[i] *= rms_rcp * weight;
    }

    // Apply RoPE to normalized elements
    float elements2[numElemsPerThread];  // Additional buffer required for RoPE.

    int64_t pos_id = position_ids[tokenIdx];

    // Calculate cache pointer for this position - similar to
    // pos_encoding_kernels.cu
    T_cache const* cache_ptr = cos_sin_cache + pos_id * rotary_dim;
    int const embed_dim = rotary_dim / 2;
    T_cache const* cos_ptr = cache_ptr;
    T_cache const* sin_ptr = cache_ptr + embed_dim;
    int const rotary_lanes = rotary_dim / numElemsPerThread;  // rotary range
    if (laneId < rotary_lanes) {
      if constexpr (interleave) {
        // Perform interleaving. Use pre-computed cos/sin values.
#pragma unroll
        for (int i = 0; i < numElemsPerThread / 2; ++i) {
          int const idx0 = 2 * i;
          int const idx1 = 2 * i + 1;
          // Global dimension index in the head
          int const dim_idx = laneId * numElemsPerThread + idx0;

          float const val0 = elements[idx0];
          float const val1 = elements[idx1];

          int const half_dim = dim_idx / 2;
          float const cos_val =
              CacheConverter::convert(VLLM_LDG(cos_ptr + half_dim));
          float const sin_val =
              CacheConverter::convert(VLLM_LDG(sin_ptr + half_dim));

          elements[idx0] = val0 * cos_val - val1 * sin_val;
          elements[idx1] = val0 * sin_val + val1 * cos_val;
        }
      } else {
        // Before data exchange with in warp, we need to sync.
        __syncwarp();
        int pairOffset = (rotary_dim / 2) / numElemsPerThread;
        // Get the data from the other half of the warp. Use pre-computed
        // cos/sin values.
#pragma unroll
        for (int i = 0; i < numElemsPerThread; i++) {
          elements2[i] = __shfl_xor_sync(FINAL_MASK, elements[i], pairOffset);

          if (laneId < pairOffset) {
            elements2[i] = -elements2[i];
          }
          int dim_idx = laneId * numElemsPerThread + i;

          dim_idx = (dim_idx * 2) % rotary_dim;
          int half_dim = dim_idx / 2;
          float cos_val = CacheConverter::convert(VLLM_LDG(cos_ptr + half_dim));
          float sin_val = CacheConverter::convert(VLLM_LDG(sin_ptr + half_dim));

          elements[i] = elements[i] * cos_val + elements2[i] * sin_val;
        }
        // __shfl_xor_sync does not provide memfence. Need to sync again.
        __syncwarp();
      }
    }
    // Store.
    {
      vec_T vec;
      constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
#pragma unroll
      for (int i = 0; i < num_packed_elems; i++) {
        // Convert from float2 back to the specific packed type
        T2_in packed_val = Converter::convert(
            make_float2(elements[2 * i], elements[2 * i + 1]));
        // Place it into the generic vector
        *(reinterpret_cast<T2_in*>(&vec) + i) = packed_val;
      }
      *reinterpret_cast<vec_T*>(&qkv[offsetThread]) = vec;
    }

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

// Borrowed from
// https://github.com/flashinfer-ai/flashinfer/blob/8125d079a43e9a0ba463a4ed1b639cefd084cec9/include/flashinfer/pos_enc.cuh#L568
#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    const bool INTERLEAVE = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool INTERLEAVE = false;                       \
    __VA_ARGS__                                          \
  }

template <typename scalar_t_in, typename scalar_t_cache>
void launchFusedQKNormRope(void* qkv, int const num_tokens,
                           int const num_heads_q, int const num_heads_k,
                           int const num_heads_v, int const head_dim,
                           int const rotary_dim, float const eps,
                           void const* q_weight, void const* k_weight,
                           void const* cos_sin_cache, bool const interleave,
                           int64_t const* position_ids, cudaStream_t stream) {
  constexpr int blockSize = 256;

  int const warpsPerBlock = blockSize / 32;
  int const totalQKHeads = num_heads_q + num_heads_k;
  int const totalWarps = num_tokens * totalQKHeads;

  int const gridSize = common::divUp(totalWarps, warpsPerBlock);
  dim3 gridDim(gridSize);
  dim3 blockDim(blockSize);

  switch (head_dim) {
    case 64:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormRopeKernel<scalar_t_in, scalar_t_cache, 64, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(
                qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight,
                k_weight, cos_sin_cache, position_ids, num_tokens, rotary_dim);
      });
      break;
    case 128:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormRopeKernel<scalar_t_in, scalar_t_cache, 128, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(
                qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight,
                k_weight, cos_sin_cache, position_ids, num_tokens, rotary_dim);
      });
      break;
    case 256:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fusedQKNormRopeKernel<scalar_t_in, scalar_t_cache, 256, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(
                qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight,
                k_weight, cos_sin_cache, position_ids, num_tokens, rotary_dim);
      });
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported head dimension for fusedQKNormRope: ", head_dim);
  }
}
}  // namespace tensorrt_llm::kernels

void fused_qk_norm_rope(
    torch::Tensor& qkv,       // Combined QKV tensor [num_tokens,
                              // (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int64_t num_heads_q,      // Number of query heads
    int64_t num_heads_k,      // Number of key heads
    int64_t num_heads_v,      // Number of value heads
    int64_t head_dim,         // Dimension per head
    double eps,               // Epsilon for RMS normalization
    torch::Tensor& q_weight,  // RMSNorm weights for query [head_dim]
    torch::Tensor& k_weight,  // RMSNorm weights for key [head_dim]
    torch::Tensor& cos_sin_cache,  // Cos/sin cache [max_position, head_dim]
    bool is_neox,                  // Whether RoPE is applied in Neox style
    torch::Tensor& position_ids    // Position IDs for RoPE [num_tokens]
) {
  // Input validation
  CHECK_INPUT(qkv);
  CHECK_INPUT(position_ids);
  CHECK_INPUT(q_weight);
  CHECK_INPUT(k_weight);
  CHECK_INPUT(cos_sin_cache);
  CHECK_TYPE(position_ids, torch::kInt64);

  TORCH_CHECK(qkv.dim() == 2,
              "QKV tensor must be 2D: [num_tokens, "
              "(num_heads_q+num_heads_k+num_heads_v)*head_dim]");
  TORCH_CHECK(position_ids.dim() == 1, "Position IDs must be 1D: [num_tokens]");
  TORCH_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
  TORCH_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
  TORCH_CHECK(cos_sin_cache.dim() == 2,
              "Cos/sin cache must be 2D: [max_position, head_dim]");
  TORCH_CHECK(q_weight.size(0) == head_dim,
              "Query weights size must match head dimension");
  TORCH_CHECK(k_weight.size(0) == head_dim,
              "Key weights size must match head dimension");

  TORCH_CHECK(cos_sin_cache.size(1) % 2 == 0, "rotary_dim must be even");
  TORCH_CHECK(cos_sin_cache.size(1) <= head_dim,
              "rotary_dim must be less than or equal to head_dim");

  TORCH_CHECK(qkv.scalar_type() == q_weight.scalar_type() &&
                  qkv.scalar_type() == k_weight.scalar_type(),
              "qkv, q_weight and k_weight must have the same dtype");

  int64_t num_tokens = qkv.size(0);
  TORCH_CHECK(position_ids.size(0) == num_tokens,
              "Number of tokens in position_ids must match QKV");

  int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(
      qkv.size(1) == total_heads * head_dim,
      "QKV tensor size must match total number of heads and head dimension");

  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  VLLM_DISPATCH_HALF_TYPES(qkv.scalar_type(), "fused_qk_norm_rope_kernel", [&] {
    using qkv_scalar_t = scalar_t;
    VLLM_DISPATCH_FLOATING_TYPES(
        cos_sin_cache.scalar_type(), "fused_qk_norm_rope_kernel", [&] {
          using cache_scalar_t = scalar_t;
          tensorrt_llm::kernels::launchFusedQKNormRope<qkv_scalar_t,
                                                       cache_scalar_t>(
              qkv.data_ptr(), static_cast<int>(num_tokens),
              static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
              static_cast<int>(num_heads_v), static_cast<int>(head_dim),
              static_cast<int>(cos_sin_cache.size(1)), static_cast<float>(eps),
              q_weight.data_ptr(), k_weight.data_ptr(),
              cos_sin_cache.data_ptr(), !is_neox,
              reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
              stream);
        });
  });
}

namespace step5_fused_qk_norm_rope_impl {

constexpr int kThreadsPerBlock = 128;
constexpr int kThreadsPerHead = 4;  // head bytes/thread: 32B@64 64B@128 96B@192 128B@256
constexpr int kVecElems = 8;        // 8 bf16 = one 16B vector
constexpr int kUnitsPerBlock = kThreadsPerBlock / kThreadsPerHead;
// Grids at or below this block count are latency-bound (few waves on 32
// SMs): use the smem-staged fast kernel. Above it, the free-running chain
// kernel wins. 160 blocks = 5/SM (~one full occupancy wave).
constexpr int kStageMaxBlocks = 160;

struct Step5Params {
  const __nv_bfloat16* __restrict__ qkv;
  __nv_bfloat16* __restrict__ q_out;
  __nv_bfloat16* __restrict__ k_out;
  __nv_bfloat16* __restrict__ v_out;
  const void* __restrict__ q_weight;  // fp32 or bf16 [head_dim]
  const void* __restrict__ k_weight;
  const void* __restrict__ cos;  // bf16 (fast) or fp32 (generic) [max_pos, rp]
  const void* __restrict__ sin;
  const void* __restrict__ positions;  // int32 or int64 [tokens]
  int64_t qkv_row_stride;              // element strides
  int64_t q_row_stride;
  int64_t k_row_stride;
  int64_t v_row_stride;
  int64_t cos_row_stride;
  int64_t sin_row_stride;
  int num_q_heads;
  int num_kv_heads;
  int head_dim;
  int rotary_pairs;
  int heads_per_token;
  int num_tokens;
  int total_units;  // num_tokens * heads_per_token
  float eps;
  float norm_weight_bias;
};

// Butterfly all-reduce inside a <=16-lane subgroup (offsets < 16 stay on the
// native fast shuffle path on C600-U, warpSize=64 -> 64-bit mask).
template <int Width>
__device__ __forceinline__ float subgroup_reduce_sum(float v) {
#pragma unroll
  for (int off = Width / 2; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffffffffffULL, v, off, Width);
  }
  return v;
}

template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return static_cast<float>(v);
}

// bf16-round-tripped normalized value: y = bf16_round(x * inv_rms * wb).
__device__ __forceinline__ float bf16_rounded_y(const __nv_bfloat16* x, int d,
                                                float inv_rms, float wb) {
  return __bfloat162float(
      __float2bfloat16_rn(__bfloat162float(x[d]) * inv_rms * wb));
}

// Rotate one pair of 8-element vectors (NeoX): a holds dims [8v, 8v+8) of the
// first rotary half, b holds the partner dims [8v+rp, 8v+rp+8). cs/sn are the
// cos/sin values for pair indices [8v, 8v+8), packed as 8 bf16.
__device__ __forceinline__ void rotate_vec_pair(uint4& a, uint4& b,
                                                const uint4& cs,
                                                const uint4& sn) {
  unsigned int* aw = reinterpret_cast<unsigned int*>(&a);
  unsigned int* bw = reinterpret_cast<unsigned int*>(&b);
  const unsigned int* cw = reinterpret_cast<const unsigned int*>(&cs);
  const unsigned int* sw = reinterpret_cast<const unsigned int*>(&sn);
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const float2 y0 =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(aw + w));
    const float2 y1 =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(bw + w));
    const float2 cf =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(cw + w));
    const float2 sf =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(sw + w));
    const float2 r0 =
        make_float2(y0.x * cf.x - y1.x * sf.x, y0.y * cf.y - y1.y * sf.y);
    const float2 r1 =
        make_float2(y0.x * sf.x + y1.x * cf.x, y0.y * sf.y + y1.y * cf.y);
    *reinterpret_cast<__nv_bfloat162*>(aw + w) = __float22bfloat162_rn(r0);
    *reinterpret_cast<__nv_bfloat162*>(bw + w) = __float22bfloat162_rn(r1);
  }
}

// Load 8 consecutive weights starting at element offset e0 (multiple of 8)
// and convert to fp32. WeightT is float (32B) or __nv_bfloat16 (16B).
template <typename WeightT>
__device__ __forceinline__ void load_weight_vec(
    const void* __restrict__ w_ptr, int e0, float (&w)[8]) {
  const WeightT* w_base = reinterpret_cast<const WeightT*>(w_ptr) + e0;
  if constexpr (sizeof(WeightT) == 4) {
    const uint4* w4 = reinterpret_cast<const uint4*>(w_base);
    const uint4 lo = w4[0];
    const uint4 hi = w4[1];
    const float* f = reinterpret_cast<const float*>(&lo);
    w[0] = f[0];
    w[1] = f[1];
    w[2] = f[2];
    w[3] = f[3];
    f = reinterpret_cast<const float*>(&hi);
    w[4] = f[0];
    w[5] = f[1];
    w[6] = f[2];
    w[7] = f[3];
  } else {
    const uint4 packed = *reinterpret_cast<const uint4*>(w_base);
    const unsigned int* pw = reinterpret_cast<const unsigned int*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f =
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(pw + i));
      w[2 * i] = f.x;
      w[2 * i + 1] = f.y;
    }
  }
}

// ── Fast kernel, smem-staged variant (small grids) ─────────────────────────
template <int kHeadDim, int kRp, typename PosT, typename WeightT>
__global__ __launch_bounds__(kThreadsPerBlock) void step5_fast_kernel_staged(
    const Step5Params p) {
  constexpr int kVecsPerHead = kHeadDim / kVecElems;
  constexpr int kVPT = kVecsPerHead / kThreadsPerHead;  // vectors per thread
  constexpr int kPPT = kRp / 8 / kThreadsPerHead;       // rope pairs/thread
  constexpr int kRpVecs = kRp / 8;                      // row size in 16B vecs
  // 32 units/block; heads_per_token >= 3 (host-checked) -> <= 12 tokens.
  constexpr int kMaxTok = (kUnitsPerBlock + 2) / 3 + 1;

  const int tid = static_cast<int>(threadIdx.x);
  const int unit = blockIdx.x * kUnitsPerBlock + (tid >> 2);
  const int lane = tid & (kThreadsPerHead - 1);
  const bool active = unit < p.total_units;
  const int token = active ? unit / p.heads_per_token : 0;
  const int local_head = unit - token * p.heads_per_token;  // Q...K...V order

  const bool is_q = local_head < p.num_q_heads;
  const bool is_v = local_head >= p.num_q_heads + p.num_kv_heads;

  // ── stage this block's cos/sin rows into shared memory (kRp > 0) ─────────
  __shared__ uint4 s_cs[kRp > 0 ? kMaxTok : 1][kRp > 0 ? kRpVecs : 1];
  __shared__ uint4 s_sn[kRp > 0 ? kMaxTok : 1][kRp > 0 ? kRpVecs : 1];
  const int token_lo = (blockIdx.x * kUnitsPerBlock) / p.heads_per_token;
  if constexpr (kRp > 0) {
    int token_hi =
        (blockIdx.x * kUnitsPerBlock + kUnitsPerBlock - 1) / p.heads_per_token;
    if (token_hi > p.num_tokens - 1) token_hi = p.num_tokens - 1;
    const int per_tok = 2 * kRpVecs;
    const int n_stage = (token_hi - token_lo + 1) * per_tok;
    const PosT* pos_arr = reinterpret_cast<const PosT*>(p.positions);
    for (int i = tid; i < n_stage; i += kThreadsPerBlock) {
      const int t = i / per_tok;
      const int rem = i - t * per_tok;
      const bool is_sin = rem >= kRpVecs;
      const int v = rem - (is_sin ? kRpVecs : 0);
      const int64_t pos = static_cast<int64_t>(pos_arr[token_lo + t]);
      const __nv_bfloat16* base = reinterpret_cast<const __nv_bfloat16*>(
          is_sin ? p.sin : p.cos);
      const int64_t rstride =
          is_sin ? p.sin_row_stride : p.cos_row_stride;
      const uint4 val = *reinterpret_cast<const uint4*>(
          base + pos * rstride + v * kVecElems);
      if (is_sin) {
        s_sn[t][v] = val;
      } else {
        s_cs[t][v] = val;
      }
    }
  }

  // ── per-unit work before the barrier ─────────────────────────────────────
  uint4 vin[kVPT];
  float ss = 0.0f;
  if (active) {
    const __nv_bfloat16* head_in =
        p.qkv + token * p.qkv_row_stride + local_head * kHeadDim;
    __nv_bfloat16* out =
        is_q ? p.q_out + token * p.q_row_stride + local_head * kHeadDim
             : (is_v ? p.v_out + token * p.v_row_stride +
                           (local_head - p.num_q_heads - p.num_kv_heads) *
                               kHeadDim
                     : p.k_out + token * p.k_row_stride +
                           (local_head - p.num_q_heads) * kHeadDim);
    const uint4* in4 = reinterpret_cast<const uint4*>(head_in);
    uint4* out4 = reinterpret_cast<uint4*>(out);

    if (is_v) {
      // Bit-exact V copy, 16B vectors.
#pragma unroll
      for (int r = 0; r < kVPT; ++r) {
        out4[lane + r * kThreadsPerHead] = in4[lane + r * kThreadsPerHead];
      }
    } else {
      // Q/K: load, accumulate sum of squares.
#pragma unroll
      for (int r = 0; r < kVPT; ++r) {
        vin[r] = in4[lane + r * kThreadsPerHead];
        const unsigned int* vw =
            reinterpret_cast<const unsigned int*>(&vin[r]);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const float2 f = __bfloat1622float2(
              *reinterpret_cast<const __nv_bfloat162*>(vw + w));
          ss += f.x * f.x + f.y * f.y;
        }
      }
      ss = subgroup_reduce_sum<kThreadsPerHead>(ss);
      const float inv_rms =
          __builtin_mxc_rcpf(sqrtf(ss / static_cast<float>(kHeadDim) + p.eps));

      // Normalize + round to bf16 (round-trip contract before RoPE), in
      // place into vin[] so only one kVPT-vector payload is ever live.
      const void* w_ptr = is_q ? p.q_weight : p.k_weight;
#pragma unroll
      for (int r = 0; r < kVPT; ++r) {
        float w[8];
        load_weight_vec<WeightT>(w_ptr, (lane + r * kThreadsPerHead) * kVecElems,
                                 w);
        unsigned int* vw = reinterpret_cast<unsigned int*>(&vin[r]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 x = __bfloat1622float2(
              *reinterpret_cast<const __nv_bfloat162*>(vw + i));
          const float2 yv = make_float2(
              x.x * inv_rms * (w[2 * i] + p.norm_weight_bias),
              x.y * inv_rms * (w[2 * i + 1] + p.norm_weight_bias));
          *reinterpret_cast<__nv_bfloat162*>(vw + i) = __float22bfloat162_rn(yv);
        }
      }
    }
  }

  // ── one barrier: staged rows visible to every thread (kRp > 0 only) ──────
  if constexpr (kRp > 0) __syncthreads();

  // ── NeoX partial RoPE from smem + store (Q/K only) ───────────────────────
  if (active && !is_v) {
    __nv_bfloat16* out =
        is_q ? p.q_out + token * p.q_row_stride + local_head * kHeadDim
             : p.k_out + token * p.k_row_stride +
                   (local_head - p.num_q_heads) * kHeadDim;
    uint4* out4 = reinterpret_cast<uint4*>(out);
    if constexpr (kPPT > 0) {
      const int t = token - token_lo;
#pragma unroll
      for (int i = 0; i < kPPT; ++i) {
        const int v0 = lane + i * kThreadsPerHead;  // first-half vector idx
        const uint4 c = s_cs[t][v0];
        const uint4 s = s_sn[t][v0];
        rotate_vec_pair(vin[i], vin[i + kPPT], c, s);
      }
    }
#pragma unroll
    for (int r = 0; r < kVPT; ++r) {
      out4[lane + r * kThreadsPerHead] = vin[r];
    }
  }
}

// ── Fast kernel, chain variant (large grids) ───────────────────────────────
// Same math and mapping as the staged variant, but every Q/K thread runs its
// own cos/sin loads (pos issued FIRST, cos/sin right behind the head loads)
// with no barrier: at >= ~5 waves of blocks the scheduler covers the rope
// dependency chain as well as a barrier would, without the phase alignment.
template <int kHeadDim, int kRp, typename PosT, typename WeightT>
__global__ __launch_bounds__(kThreadsPerBlock) void step5_fast_kernel(
    const Step5Params p) {
  constexpr int kVecsPerHead = kHeadDim / kVecElems;
  constexpr int kVPT = kVecsPerHead / kThreadsPerHead;  // vectors per thread
  constexpr int kPPT = kRp / 8 / kThreadsPerHead;       // rope pairs/thread

  const int unit =
      blockIdx.x * kUnitsPerBlock + (static_cast<int>(threadIdx.x) >> 2);
  if (unit >= p.total_units) return;
  const int lane = static_cast<int>(threadIdx.x) & (kThreadsPerHead - 1);
  const int token = unit / p.heads_per_token;
  const int local_head = unit - token * p.heads_per_token;  // Q...K...V order

  const bool is_q = local_head < p.num_q_heads;
  const bool is_v = local_head >= p.num_q_heads + p.num_kv_heads;

  const __nv_bfloat16* head_in =
      p.qkv + token * p.qkv_row_stride + local_head * kHeadDim;
  __nv_bfloat16* out =
      is_q ? p.q_out + token * p.q_row_stride + local_head * kHeadDim
           : (is_v ? p.v_out + token * p.v_row_stride +
                         (local_head - p.num_q_heads - p.num_kv_heads) *
                             kHeadDim
                   : p.k_out + token * p.k_row_stride +
                         (local_head - p.num_q_heads) * kHeadDim);

  const uint4* in4 = reinterpret_cast<const uint4*>(head_in);
  uint4* out4 = reinterpret_cast<uint4*>(out);

  if (is_v) {
    // Bit-exact V copy, 16B vectors.
#pragma unroll
    for (int r = 0; r < kVPT; ++r) {
      out4[lane + r * kThreadsPerHead] = in4[lane + r * kThreadsPerHead];
    }
    return;
  }

  // ── positions FIRST: its (uncached) latency sits on the rope critical ────
  // path (pos -> cos/sin address -> cos/sin load -> rotate -> store), so the
  // load is issued before the head data and overlaps with everything below.
  const int64_t pos =
      kRp > 0 ? static_cast<int64_t>(
                    reinterpret_cast<const PosT*>(p.positions)[token])
              : 0;

  // ── Q/K: load, accumulate sum of squares ─────────────────────────────────
  uint4 vin[kVPT];
  float ss = 0.0f;
#pragma unroll
  for (int r = 0; r < kVPT; ++r) {
    vin[r] = in4[lane + r * kThreadsPerHead];
    const unsigned int* vw = reinterpret_cast<const unsigned int*>(&vin[r]);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const float2 f = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(vw + w));
      ss += f.x * f.x + f.y * f.y;
    }
  }

  // ── issue cos/sin loads right behind the head loads (pos already in ──────
  // flight): their latency hides behind the reduction + normalize ALU instead
  // of stalling the kernel tail. Vector (lane + 4i) pairs with vector
  // (lane + 4i + 4*kPPT) -- the intra-thread partner-vector property.
  uint4 cs[kPPT > 0 ? kPPT : 1];
  uint4 sn[kPPT > 0 ? kPPT : 1];
  if constexpr (kPPT > 0) {
    const __nv_bfloat16* cos_row =
        reinterpret_cast<const __nv_bfloat16*>(p.cos) + pos * p.cos_row_stride;
    const __nv_bfloat16* sin_row =
        reinterpret_cast<const __nv_bfloat16*>(p.sin) + pos * p.sin_row_stride;
#pragma unroll
    for (int i = 0; i < kPPT; ++i) {
      const int v0 = lane + i * kThreadsPerHead;  // first-half vector index
      cs[i] = *reinterpret_cast<const uint4*>(cos_row + v0 * kVecElems);
      sn[i] = *reinterpret_cast<const uint4*>(sin_row + v0 * kVecElems);
    }
  }

  ss = subgroup_reduce_sum<kThreadsPerHead>(ss);
  const float inv_rms =
      __builtin_mxc_rcpf(sqrtf(ss / static_cast<float>(kHeadDim) + p.eps));

  // ── normalize + round to bf16 (round-trip contract before RoPE) ─────────
  // Written in place into vin[] (elementwise read-then-write) so only one
  // kVPT-vector payload is ever live -- halves the data register footprint.
  const void* w_ptr = is_q ? p.q_weight : p.k_weight;
#pragma unroll
  for (int r = 0; r < kVPT; ++r) {
    float w[8];
    load_weight_vec<WeightT>(w_ptr, (lane + r * kThreadsPerHead) * kVecElems,
                             w);
    unsigned int* vw = reinterpret_cast<unsigned int*>(&vin[r]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 x = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(vw + i));
      const float2 yv = make_float2(
          x.x * inv_rms * (w[2 * i] + p.norm_weight_bias),
          x.y * inv_rms * (w[2 * i + 1] + p.norm_weight_bias));
      *reinterpret_cast<__nv_bfloat162*>(vw + i) = __float22bfloat162_rn(yv);
    }
  }

  // ── NeoX partial RoPE (intra-thread exchange, cos/sin already loaded) ────
  if constexpr (kPPT > 0) {
#pragma unroll
    for (int i = 0; i < kPPT; ++i) {
      rotate_vec_pair(vin[i], vin[i + kPPT], cs[i], sn[i]);
    }
  }

  // ── store ────────────────────────────────────────────────────────────────
#pragma unroll
  for (int r = 0; r < kVPT; ++r) {
    out4[lane + r * kThreadsPerHead] = vin[r];
  }
}

// ── Generic fallback kernel ────────────────────────────────────────────────
// One thread per output element. Handles any head_dim, any rotary_pairs, any
// 2B alignment, fp32 or bf16 cos/sin. Never on the production hot path.
template <typename PosT, typename WeightT, typename CosT>
__global__ void step5_generic_kernel(const Step5Params p) {
  const int hd = p.head_dim;
  const int64_t qk_heads = p.num_q_heads + p.num_kv_heads;
  const int64_t per_qk_row = qk_heads * hd;
  const int64_t qk_total = static_cast<int64_t>(p.num_tokens) * per_qk_row;
  const int64_t v_total =
      static_cast<int64_t>(p.num_tokens) * p.num_kv_heads * hd;
  const int64_t gid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= qk_total + v_total) return;

  if (gid < qk_total) {
    // Q or K element.
    const int64_t token = gid / per_qk_row;
    const int64_t rem = gid - token * per_qk_row;
    const int head = static_cast<int>(rem / hd);  // packed QK head index
    const int dim = static_cast<int>(rem - static_cast<int64_t>(head) * hd);
    const bool is_q = head < p.num_q_heads;

    const __nv_bfloat16* x = p.qkv + token * p.qkv_row_stride + head * hd;
    float ss = 0.0f;
    for (int j = 0; j < hd; ++j) {
      const float v = __bfloat162float(x[j]);
      ss += v * v;
    }
    const float inv_rms = rsqrtf(ss / static_cast<float>(hd) + p.eps);
    const WeightT* w =
        reinterpret_cast<const WeightT*>(is_q ? p.q_weight : p.k_weight);

    __nv_bfloat16* out =
        is_q ? p.q_out + token * p.q_row_stride + head * hd
             : p.k_out + token * p.k_row_stride +
                   (head - p.num_q_heads) * hd;

    const int rp = p.rotary_pairs;
    if (rp > 0 && dim < 2 * rp) {
      const int64_t pos = reinterpret_cast<const PosT*>(p.positions)[token];
      const int pidx = dim < rp ? dim : dim - rp;          // cos/sin index
      const int partner = dim < rp ? dim + rp : dim - rp;  // NeoX partner
      const CosT* crow =
          reinterpret_cast<const CosT*>(p.cos) + pos * p.cos_row_stride;
      const CosT* srow =
          reinterpret_cast<const CosT*>(p.sin) + pos * p.sin_row_stride;
      const float y0 =
          bf16_rounded_y(x, dim, inv_rms,
                         to_float<WeightT>(w[dim]) + p.norm_weight_bias);
      const float y1 =
          bf16_rounded_y(x, partner, inv_rms,
                         to_float<WeightT>(w[partner]) + p.norm_weight_bias);
      const float c = to_float<CosT>(crow[pidx]);
      const float s = to_float<CosT>(srow[pidx]);
      // y0 = y[dim]; y1 = y[partner]. NeoX pair (p, p+rp):
      //   out[p]    = y[p]*cos[p]   - y[p+rp]*sin[p]
      //   out[p+rp] = y[p]*sin[p]   + y[p+rp]*cos[p]
      const float r =
          (dim < rp) ? (y0 * c - y1 * s) : (y1 * s + y0 * c);
      out[dim] = __float2bfloat16_rn(r);
    } else {
      out[dim] = __float2bfloat16_rn(
          bf16_rounded_y(x, dim, inv_rms,
                         to_float<WeightT>(w[dim]) + p.norm_weight_bias));
    }
  } else {
    // V: bit-exact copy.
    const int64_t vg = gid - qk_total;
    const int64_t per_v_row = static_cast<int64_t>(p.num_kv_heads) * hd;
    const int64_t token = vg / per_v_row;
    const int64_t col = vg - token * per_v_row;
    const __nv_bfloat16* src = p.qkv + token * p.qkv_row_stride +
                               (p.num_q_heads + p.num_kv_heads) * hd + col;
    p.v_out[token * p.v_row_stride + col] = *src;
  }
}

inline void launch_generic(const Step5Params& p, bool pos_is_i32, bool w_is_f32,
                           bool cos_is_bf16, cudaStream_t stream);

template <int kHeadDim, int kRp>
void launch_fast_rp(const Step5Params& p, bool pos_is_i32, bool w_is_f32,
                    cudaStream_t stream) {
  // Instantiate for every (kHeadDim, kRp) pair the switch below mentions, but
  // only launch the fast kernels when the pair is geometrically valid
  // (2*kRp <= kHeadDim); otherwise fall back to the generic kernel.
  if constexpr (2 * kRp <= kHeadDim) {
    const int grid =
        (p.total_units + kUnitsPerBlock - 1) / kUnitsPerBlock;
#define STEP5_LAUNCH(POST, WT)                                         \
  do {                                                                 \
    if (grid <= kStageMaxBlocks) {                                     \
      step5_fast_kernel_staged<kHeadDim, kRp, POST, WT>                \
          <<<grid, kThreadsPerBlock, 0, stream>>>(p);                  \
    } else {                                                           \
      step5_fast_kernel<kHeadDim, kRp, POST, WT>                       \
          <<<grid, kThreadsPerBlock, 0, stream>>>(p);                  \
    }                                                                  \
  } while (0)
    if (pos_is_i32) {
      if (w_is_f32) {
        STEP5_LAUNCH(int32_t, float);
      } else {
        STEP5_LAUNCH(int32_t, __nv_bfloat16);
      }
    } else {
      if (w_is_f32) {
        STEP5_LAUNCH(int64_t, float);
      } else {
        STEP5_LAUNCH(int64_t, __nv_bfloat16);
      }
    }
#undef STEP5_LAUNCH
  } else {
    launch_generic(p, pos_is_i32, w_is_f32, /*cos_is_bf16=*/true, stream);
  }
}

// kRp is a template parameter so the rope path fully unrolls and the cos/sin
// registers are sized exactly. rp values outside {0,32,64,96,128} (e.g. the
// rp=48 generic test case) go to the generic kernel, as before.
template <int kHeadDim>
void launch_fast(const Step5Params& p, int rotary_pairs, bool pos_is_i32,
                 bool w_is_f32, cudaStream_t stream) {
  switch (rotary_pairs) {
    case 0:
      launch_fast_rp<kHeadDim, 0>(p, pos_is_i32, w_is_f32, stream);
      break;
    case 32:
      launch_fast_rp<kHeadDim, 32>(p, pos_is_i32, w_is_f32, stream);
      break;
    case 64:
      launch_fast_rp<kHeadDim, 64>(p, pos_is_i32, w_is_f32, stream);
      break;
    case 96:
      launch_fast_rp<kHeadDim, 96>(p, pos_is_i32, w_is_f32, stream);
      break;
    case 128:
      launch_fast_rp<kHeadDim, 128>(p, pos_is_i32, w_is_f32, stream);
      break;
    default:
      launch_generic(p, pos_is_i32, w_is_f32, /*cos_is_bf16=*/true, stream);
      break;
  }
}

inline void launch_generic(const Step5Params& p, bool pos_is_i32, bool w_is_f32,
                           bool cos_is_bf16, cudaStream_t stream) {
  const int64_t total =
      static_cast<int64_t>(p.num_tokens) *
      ((p.num_q_heads + 2 * p.num_kv_heads) * p.head_dim);
  const int grid = static_cast<int>((total + 255) / 256);
  if (pos_is_i32) {
    if (w_is_f32) {
      if (cos_is_bf16) {
        step5_generic_kernel<int32_t, float, __nv_bfloat16>
            <<<grid, 256, 0, stream>>>(p);
      } else {
        step5_generic_kernel<int32_t, float, float>
            <<<grid, 256, 0, stream>>>(p);
      }
    } else {
      if (cos_is_bf16) {
        step5_generic_kernel<int32_t, __nv_bfloat16, __nv_bfloat16>
            <<<grid, 256, 0, stream>>>(p);
      } else {
        step5_generic_kernel<int32_t, __nv_bfloat16, float>
            <<<grid, 256, 0, stream>>>(p);
      }
    }
  } else {
    if (w_is_f32) {
      if (cos_is_bf16) {
        step5_generic_kernel<int64_t, float, __nv_bfloat16>
            <<<grid, 256, 0, stream>>>(p);
      } else {
        step5_generic_kernel<int64_t, float, float>
            <<<grid, 256, 0, stream>>>(p);
      }
    } else {
      if (cos_is_bf16) {
        step5_generic_kernel<int64_t, __nv_bfloat16, __nv_bfloat16>
            <<<grid, 256, 0, stream>>>(p);
      } else {
        step5_generic_kernel<int64_t, __nv_bfloat16, float>
            <<<grid, 256, 0, stream>>>(p);
      }
    }
  }
}

inline bool is_16b_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

}  // namespace step5_fused_qk_norm_rope_impl

// Fused RMSNorm(Q/K) + NeoX partial RoPE(Q/K) + V copy, out-of-place on a
// packed QKV projection. See the block comment above for the math contract.
// Returns {q_out, k_out, v_out}: the caller-provided tensors when all three
// are given, freshly allocated bf16 tensors otherwise.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
step5_fused_qk_norm_rope(
    const torch::Tensor& qkv, const torch::Tensor& q_weight_in,
    const torch::Tensor& k_weight_in, const torch::Tensor& cos,
    const torch::Tensor& sin, const torch::Tensor& positions_in,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t rotary_pairs, double eps, double norm_weight_bias,
    const std::optional<torch::Tensor>& q_out_opt,
    const std::optional<torch::Tensor>& k_out_opt,
    const std::optional<torch::Tensor>& v_out_opt) {
  using namespace step5_fused_qk_norm_rope_impl;

  // ── output contract: all three or none ───────────────────────────────────
  const bool has_q = q_out_opt.has_value();
  const bool has_k = k_out_opt.has_value();
  const bool has_v = v_out_opt.has_value();
  TORCH_CHECK(has_q == has_k && has_k == has_v,
              "step5_fused_qk_norm_rope: q_out, k_out, v_out must be all "
              "given or all omitted");

  // ── input validation ─────────────────────────────────────────────────────
  TORCH_CHECK(qkv.is_cuda(), "step5: qkv must be a CUDA tensor");
  TORCH_CHECK(qkv.dim() == 2, "step5: qkv must be 2D [tokens, width]");
  TORCH_CHECK(qkv.scalar_type() == torch::kBFloat16,
              "step5: qkv must be bf16");
  TORCH_CHECK(qkv.stride(-1) == 1,
              "step5: qkv must be contiguous in the last dimension");
  TORCH_CHECK(num_q_heads >= 0 && num_kv_heads >= 0 && head_dim > 0,
              "step5: invalid head counts / head_dim");
  TORCH_CHECK(rotary_pairs >= 0 && 2 * rotary_pairs <= head_dim,
              "step5: need 0 <= 2*rotary_pairs <= head_dim, got ",
              rotary_pairs, " and ", head_dim);

  const int64_t tokens = qkv.size(0);
  const int64_t q_width = num_q_heads * head_dim;
  const int64_t kv_width = num_kv_heads * head_dim;
  const int64_t packed_width = q_width + 2 * kv_width;
  TORCH_CHECK(qkv.size(1) >= packed_width,
              "step5: qkv width must be >= ", packed_width, ", got ",
              qkv.size(1));

  TORCH_CHECK(q_weight_in.is_cuda() && k_weight_in.is_cuda(),
              "step5: weights must be CUDA tensors");
  TORCH_CHECK(q_weight_in.numel() == head_dim &&
                  k_weight_in.numel() == head_dim,
              "step5: q_weight/k_weight must have head_dim elements");
  TORCH_CHECK(q_weight_in.scalar_type() == k_weight_in.scalar_type(),
              "step5: q_weight and k_weight dtypes must match");
  const bool w_is_f32 = q_weight_in.scalar_type() == torch::kFloat32;
  TORCH_CHECK(w_is_f32 || q_weight_in.scalar_type() == torch::kBFloat16,
              "step5: weights must be fp32 or bf16");

  TORCH_CHECK(cos.is_cuda() && sin.is_cuda(),
              "step5: cos/sin must be CUDA tensors");
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2,
              "step5: cos/sin must be 2D [max_pos, rotary_pairs]");
  TORCH_CHECK(cos.scalar_type() == sin.scalar_type(),
              "step5: cos and sin dtypes must match");
  const bool cos_is_bf16 = cos.scalar_type() == torch::kBFloat16;
  TORCH_CHECK(cos_is_bf16 || cos.scalar_type() == torch::kFloat32,
              "step5: cos/sin must be bf16 or fp32");
  TORCH_CHECK(cos.stride(-1) == 1 && sin.stride(-1) == 1,
              "step5: cos/sin must be contiguous in the last dimension");
  TORCH_CHECK(cos.size(1) >= rotary_pairs && sin.size(1) >= rotary_pairs,
              "step5: cos/sin width must be >= rotary_pairs");
  // NOTE: positions values must lie in [0, cos.size(0)) - checked by the
  // caller; a device-side content check would break CUDA graph capture.

  TORCH_CHECK(positions_in.is_cuda(), "step5: positions must be a CUDA tensor");
  TORCH_CHECK(positions_in.numel() == tokens,
              "step5: positions must have num_tokens elements");
  const bool pos_is_i32 = positions_in.scalar_type() == torch::kInt32;
  TORCH_CHECK(pos_is_i32 || positions_in.scalar_type() == torch::kInt64,
              "step5: positions must be int32 or int64");

  // Normalize layouts (no-ops for the production contiguous inputs; copies
  // are on-stream and therefore CUDA-graph capture safe).
  const torch::Tensor q_weight = q_weight_in.contiguous();
  const torch::Tensor k_weight = k_weight_in.contiguous();
  const torch::Tensor positions = positions_in.reshape(-1).contiguous();

  // ── outputs: validate or allocate ────────────────────────────────────────
  torch::Tensor q_out, k_out, v_out;
  if (has_q) {
    q_out = *q_out_opt;
    k_out = *k_out_opt;
    v_out = *v_out_opt;
    TORCH_CHECK(q_out.is_cuda() && k_out.is_cuda() && v_out.is_cuda(),
                "step5: outputs must be CUDA tensors");
    TORCH_CHECK(q_out.scalar_type() == torch::kBFloat16 &&
                    k_out.scalar_type() == torch::kBFloat16 &&
                    v_out.scalar_type() == torch::kBFloat16,
                "step5: outputs must be bf16");
    TORCH_CHECK(q_out.dim() == 2 && q_out.size(0) == tokens &&
                    q_out.size(1) == q_width,
                "step5: q_out must have shape [", tokens, ", ", q_width, "]");
    TORCH_CHECK(k_out.dim() == 2 && k_out.size(0) == tokens &&
                    k_out.size(1) == kv_width,
                "step5: k_out must have shape [", tokens, ", ", kv_width, "]");
    TORCH_CHECK(v_out.dim() == 2 && v_out.size(0) == tokens &&
                    v_out.size(1) == kv_width,
                "step5: v_out must have shape [", tokens, ", ", kv_width, "]");
    TORCH_CHECK(q_out.stride(-1) == 1 && k_out.stride(-1) == 1 &&
                    v_out.stride(-1) == 1,
                "step5: outputs must be contiguous in the last dimension");
  } else {
    q_out = qkv.new_empty({tokens, q_width});
    k_out = qkv.new_empty({tokens, kv_width});
    v_out = qkv.new_empty({tokens, kv_width});
  }
  if (tokens == 0) {
    return {q_out, k_out, v_out};
  }

  // ── fill launch params ───────────────────────────────────────────────────
  Step5Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.q_out = reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr());
  p.k_out = reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr());
  p.v_out = reinterpret_cast<__nv_bfloat16*>(v_out.data_ptr());
  p.q_weight = q_weight.data_ptr();
  p.k_weight = k_weight.data_ptr();
  p.cos = cos.data_ptr();
  p.sin = sin.data_ptr();
  p.positions = positions.data_ptr();
  p.qkv_row_stride = qkv.stride(0);
  p.q_row_stride = q_out.stride(0);
  p.k_row_stride = k_out.stride(0);
  p.v_row_stride = v_out.stride(0);
  p.cos_row_stride = cos.stride(0);
  p.sin_row_stride = sin.stride(0);
  p.num_q_heads = static_cast<int>(num_q_heads);
  p.num_kv_heads = static_cast<int>(num_kv_heads);
  p.head_dim = static_cast<int>(head_dim);
  p.rotary_pairs = static_cast<int>(rotary_pairs);
  p.heads_per_token = static_cast<int>(num_q_heads + 2 * num_kv_heads);
  p.num_tokens = static_cast<int>(tokens);
  p.total_units = static_cast<int>(tokens * p.heads_per_token);
  p.eps = static_cast<float>(eps);
  p.norm_weight_bias = static_cast<float>(norm_weight_bias);

  auto stream = at::cuda::getCurrentCUDAStream(qkv.device().index());

  // ── fast vs generic dispatch (metadata only: CUDA graph safe) ───────────
  const int hd = p.head_dim;
  const int rp = p.rotary_pairs;
  const bool fast_shape =
      p.heads_per_token >= 3 &&  // smem staging bound: <= 12 tokens/block
      (hd == 64 || hd == 128 || hd == 192 || hd == 256) &&
      (rp == 0 || rp % 32 == 0);
  // With <=1 token the row strides are never multiplied by a token index.
  const bool strides_ok =
      tokens <= 1 ||
      (p.qkv_row_stride % 8 == 0 && p.q_row_stride % 8 == 0 &&
       p.k_row_stride % 8 == 0 && p.v_row_stride % 8 == 0);
  const bool rope_tables_ok =
      rp == 0 ||
      (is_16b_aligned(p.cos) && is_16b_aligned(p.sin) &&
       p.cos_row_stride % 8 == 0 && p.sin_row_stride % 8 == 0);
  const bool fast_align =
      is_16b_aligned(p.qkv) && is_16b_aligned(p.q_out) &&
      is_16b_aligned(p.k_out) && is_16b_aligned(p.v_out) &&
      is_16b_aligned(p.q_weight) && is_16b_aligned(p.k_weight) && strides_ok &&
      rope_tables_ok;

  if (fast_shape && fast_align && cos_is_bf16) {
    if (hd == 64) {
      launch_fast<64>(p, rp, pos_is_i32, w_is_f32, stream);
    } else if (hd == 128) {
      launch_fast<128>(p, rp, pos_is_i32, w_is_f32, stream);
    } else if (hd == 192) {
      launch_fast<192>(p, rp, pos_is_i32, w_is_f32, stream);
    } else {
      launch_fast<256>(p, rp, pos_is_i32, w_is_f32, stream);
    }
  } else {
    launch_generic(p, pos_is_i32, w_is_f32, cos_is_bf16, stream);
  }

  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "step5_fused_qk_norm_rope launch failed: ",
              cudaGetErrorString(err));
  return {q_out, k_out, v_out};
}
