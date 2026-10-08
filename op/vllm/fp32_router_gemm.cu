// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Router GEMM: activation(T) x weight(fp32) -> fp32, H=3072, E=256, M<=32.
// Supports bf16 or fp32 activation; weight is always fp32.
// Extremely Optimized for Metax C500 (sm_80, 104 SMs, 1.55 TB/s).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <stdexcept>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// ---------------------------------------------------------------------------
// 128-bit Vectorized Load Helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ void load_weight_8(float const* __restrict__ ptr, float* dst) {
  float4 v0 = *reinterpret_cast<float4 const*>(ptr);
  float4 v1 = *reinterpret_cast<float4 const*>(ptr + 4);
  dst[0] = v0.x; dst[1] = v0.y; dst[2] = v0.z; dst[3] = v0.w;
  dst[4] = v1.x; dst[5] = v1.y; dst[6] = v1.z; dst[7] = v1.w;
}

__device__ __forceinline__ void load_activation_fp32_8(float const* __restrict__ ptr, float* dst) {
  float4 v0 = *reinterpret_cast<float4 const*>(ptr);
  float4 v1 = *reinterpret_cast<float4 const*>(ptr + 4);
  dst[0] = v0.x; dst[1] = v0.y; dst[2] = v0.z; dst[3] = v0.w;
  dst[4] = v1.x; dst[5] = v1.y; dst[6] = v1.z; dst[7] = v1.w;
}

__device__ __forceinline__ void load_activation_bf16_8(__nv_bfloat16 const* __restrict__ ptr, float* dst) {
  uint4 v = *reinterpret_cast<uint4 const*>(ptr);
  __nv_bfloat162 const* bf16x2_ptr = reinterpret_cast<__nv_bfloat162 const*>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 f2 = __bfloat1622float2(bf16x2_ptr[i]);
    dst[i * 2] = f2.x;
    dst[i * 2 + 1] = f2.y;
  }
}

// ---------------------------------------------------------------------------
// SIMD-16 Reduction for Metax C500 micro-architecture
// ---------------------------------------------------------------------------

__device__ __forceinline__ float simd16_reduce_sum(float val) {
  val += __shfl_down_sync_16(0xffffffffffffffff, val, 8);
  val += __shfl_down_sync_16(0xffffffffffffffff, val, 4);
  val += __shfl_down_sync_16(0xffffffffffffffff, val, 2);
  val += __shfl_down_sync_16(0xffffffffffffffff, val, 1);
  return val;
}

// ---------------------------------------------------------------------------
// BF16 Activation Path - K-Loop Eliminated (One-Wave Execution)
// ---------------------------------------------------------------------------

template <int kBlockSize, int kNumTokens, int kEPB, int kNumExperts, int kHiddenDim, int kTGroups = 1>
__global__ void fp32_router_gemm_kernel_bf16(
    float* __restrict__ out, __nv_bfloat16 const* __restrict__ mat_a,
    float const* __restrict__ mat_b) {
  
  constexpr int VPT = 8;
  constexpr int kSimd16Groups = kBlockSize / 16;
  constexpr int kMG = kNumTokens / kTGroups;

  static_assert(kNumTokens % kTGroups == 0);
  static_assert(kBlockSize % 16 == 0);
  static_assert(kSimd16Groups <= 32);
  
  int const e_base = blockIdx.x * kEPB;
  int const tid = threadIdx.x % kBlockSize;
  int const tgroup = threadIdx.x / kBlockSize;
  int const m0 = tgroup * kMG;

  int const simd16_lane = tid & 15;
  int const simd16_group = tid >> 4; // Max 23 for BlockSize=384

  float acc[kMG][kEPB] = {};

  int const k_base = tid * VPT;
  float b_float[kEPB][8];

  // M-loop completely unrolled for high density FFMA
#pragma unroll
  for (int m_idx = 0; m_idx < kMG; m_idx++) {
    float a_float[8];
    load_activation_bf16_8(mat_a + (size_t)(m0 + m_idx) * kHiddenDim + k_base, a_float);
#pragma unroll
    for (int e = 0; e < kEPB; e++) {
#pragma unroll
      for (int k = 0; k < 8; k++) {
        acc[m_idx][e] += a_float[k] * b_float[e][k];
      }
    }
  }

  // 24 SIMD-16 groups per token group for kBlockSize=384.
  __shared__ float sm_reduction[kNumTokens][kEPB][kSimd16Groups];

#pragma unroll
  for (int m = 0; m < kMG; m++) {
#pragma unroll
    for (int e = 0; e < kEPB; e++) {
      float sum = simd16_reduce_sum(acc[m][e]);

      if (simd16_lane == 0) {
        sm_reduction[m0 + m][e][simd16_group] = sum;
      }
    }
  }

  __syncthreads();

  if (tid < 32) {
#pragma unroll
    for (int m = 0; m < kMG; m++) {
#pragma unroll
      for (int e = 0; e < kEPB; e++) {
        float val = (tid < kSimd16Groups) ? sm_reduction[m0 + m][e][tid] : 0.0f;
        val = simd16_reduce_sum(val);
        if (simd16_lane == 0) {
          sm_reduction[m0 + m][e][simd16_group] = val;
        }
      }
    }
  }

  __syncthreads();

  // Final scalar add and global memory commit
  for (int idx = tid; idx < kMG * kEPB; idx += kBlockSize) {
    int const m = idx / kEPB;
    int const e = idx % kEPB;

    float final_sum = sm_reduction[m0 + m][e][0] + sm_reduction[m0 + m][e][1];

    out[(m0 + m) * kNumExperts + e_base + e] = final_sum;
  }
}

// ---------------------------------------------------------------------------
// FP32 Activation Path - K-Loop Eliminated (One-Wave Execution)
// ---------------------------------------------------------------------------

template <int kBlockSize, int kNumTokens, int kEPB, int kNumExperts, int kHiddenDim, int kTGroups = 1>
__global__ void fp32_router_gemm_kernel_fp32(
    float* __restrict__ out, float const* __restrict__ mat_a,
    float const* __restrict__ mat_b) {
  
  constexpr int VPT = 8;
  constexpr int kSimd16Groups = kBlockSize / 16;
  constexpr int kMG = kNumTokens / kTGroups;

  static_assert(kBlockSize % 16 == 0);
  static_assert(kSimd16Groups == 24);
  static_assert(kNumTokens % kTGroups == 0);

  int const e_base = blockIdx.x * kEPB;
  int const tid = threadIdx.x % kBlockSize;
  int const token_group = threadIdx.x / kBlockSize;
  int const m0 = token_group * kMG;

  int const simd16_lane = tid & 15;
  int const simd16_group = tid >> 4;

  float acc[kMG][kEPB] = {};

  int const k_base = tid * VPT;
  float b_float[kEPB][8];

#pragma unroll
  for (int e = 0; e < kEPB; e++) {
    load_weight_8(mat_b + (e_base + e) * kHiddenDim + k_base, b_float[e]);
  }

#pragma unroll
  for (int m_idx = 0; m_idx < kMG; m_idx++) {
    float a_float[8];
    load_activation_fp32_8(mat_a + (size_t)(m0 + m_idx) * kHiddenDim + k_base, a_float);

#pragma unroll
    for (int e = 0; e < kEPB; e++) {
#pragma unroll
      for (int k = 0; k < 8; k++) {
        acc[m_idx][e] += a_float[k] * b_float[e][k];
      }
    }
  }

  __shared__ float sm_reduction[kNumTokens][kEPB][kSimd16Groups];

#pragma unroll
  for (int m = 0; m < kMG; m++) {
#pragma unroll
    for (int e = 0; e < kEPB; e++) {
      float sum = simd16_reduce_sum(acc[m][e]);
      if (simd16_lane == 0) {
        sm_reduction[m0 + m][e][simd16_group] = sum;
      }
    }
  }

  __syncthreads();

  if (tid < 32) {
#pragma unroll
    for (int m = 0; m < kMG; m++) {
#pragma unroll
      for (int e = 0; e < kEPB; e++) {
        float val = (tid < kSimd16Groups) ? sm_reduction[m0 + m][e][tid] : 0.0f;
        val = simd16_reduce_sum(val);
        if (simd16_lane == 0) {
          sm_reduction[m0 + m][e][simd16_group] = val;
        }
      }
    }
  }

  __syncthreads();

  for (int idx = tid; idx < kMG * kEPB; idx += kBlockSize) {
    int const m = idx / kEPB;
    int const e = idx % kEPB;
    float final_sum = sm_reduction[m0 + m][e][0] + sm_reduction[m0 + m][e][1];
    out[(m0 + m) * kNumExperts + e_base + e] = final_sum;
  }
}

// ---------------------------------------------------------------------------
// C500 Tuned Launcher
// ---------------------------------------------------------------------------

template <typename InputT, int kBlockSize, int kEPB, int kNumTokens, int kNumExperts, int kHiddenDim, int kTGroups = 1>
static void launchFp32RouterGemm(float* output, InputT const* mat_a,
                                 float const* mat_b, cudaStream_t stream) {
  static_assert(kNumExperts % kEPB == 0);

  constexpr int kGridSize = kNumExperts / kEPB;
  constexpr int kBlockDim = kBlockSize * kTGroups;

  if constexpr (std::is_same_v<InputT, __nv_bfloat16>) {
    fp32_router_gemm_kernel_bf16<kBlockSize, kNumTokens, kEPB, kNumExperts,
                                  kHiddenDim, kTGroups>
        <<<kGridSize, kBlockDim, 0, stream>>>(output, mat_a, mat_b);
  } else {
    fp32_router_gemm_kernel_fp32<kBlockSize, kNumTokens, kEPB, kNumExperts,
                                 kHiddenDim, kTGroups>
        <<<kGridSize, kBlockDim, 0, stream>>>(output, mat_a, mat_b);
  }
}

template <typename InputT, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeFp32RouterGemm(float* output, InputT const* mat_a,
                          float const* mat_b, cudaStream_t stream) {
  // C500 baseline configuration.
  constexpr int kBlockSize = 384;

  if constexpr (std::is_same_v<InputT, __nv_bfloat16> &&
                kNumExperts == 256 && kHiddenDim == 6144) {
    if constexpr (kNumTokens >= 16 && kNumTokens % 2 == 0) {
      launchFp32RouterGemm<InputT, 192, 2, kNumTokens, kNumExperts, kHiddenDim,
                           2>(output, mat_a, mat_b, stream);
    } else {
      launchFp32RouterGemm<InputT, kBlockSize, 2, kNumTokens, kNumExperts,
                           kHiddenDim>(output, mat_a, mat_b, stream);
    }
  } else if constexpr (std::is_same_v<InputT, __nv_bfloat16> &&
                       kNumExperts == 128 && kHiddenDim == 6144) {
    if constexpr (kNumTokens >= 12 && kNumTokens % 2 == 0) {
      launchFp32RouterGemm<InputT, 192, 1, kNumTokens, kNumExperts, kHiddenDim,
                           2>(output, mat_a, mat_b, stream);
    } else if constexpr (kNumTokens >= 6 && kNumTokens % 2 == 0) {
      launchFp32RouterGemm<InputT, kBlockSize, 1, kNumTokens, kNumExperts,
                           kHiddenDim, 2>(output, mat_a, mat_b, stream);
    } else {
      launchFp32RouterGemm<InputT, kBlockSize, 1, kNumTokens, kNumExperts,
                           kHiddenDim>(output, mat_a, mat_b, stream);
    }
  } else if constexpr (std::is_same_v<InputT, __nv_bfloat16> &&
                       kNumExperts == 256 && kHiddenDim == 3072) {
    if constexpr (kNumTokens >= 14 && kNumTokens % 2 == 0) {
      launchFp32RouterGemm<InputT, 192, 2, kNumTokens, kNumExperts, kHiddenDim,
                           2>(output, mat_a, mat_b, stream);
    } else if constexpr (kNumTokens >= 8 && kNumTokens <= 12 &&
                         kNumTokens % 2 == 0) {
      launchFp32RouterGemm<InputT, 192, 1, kNumTokens, kNumExperts, kHiddenDim,
                           2>(output, mat_a, mat_b, stream);
    } else {
      launchFp32RouterGemm<InputT, kBlockSize, 1, kNumTokens, kNumExperts,
                           kHiddenDim>(output, mat_a, mat_b, stream);
    }
  } else {
    launchFp32RouterGemm<InputT, kBlockSize, 1, kNumTokens, kNumExperts,
                         kHiddenDim>(output, mat_a, mat_b, stream);
  }
}

// ---------------------------------------------------------------------------
// Explicit instantiations: M=1..32, for both input types, for the supported
// (E, H) pairs:  (256, 3072) [MiniMax-M2/M2.5],  (128, 6144) [MiniMax-M3]
// and  (256, 6144) [GLM-5.2].
// ---------------------------------------------------------------------------

#define INSTANTIATE(T, M, E, H)                                    \
  template void invokeFp32RouterGemm<T, M, E, H>(float*, T const*, \
                                                 float const*, cudaStream_t);

#define INSTANTIATE_ALL(T, E, H) \
  INSTANTIATE(T, 1, E, H)        \
  INSTANTIATE(T, 2, E, H)        \
  INSTANTIATE(T, 3, E, H)        \
  INSTANTIATE(T, 4, E, H)        \
  INSTANTIATE(T, 5, E, H)        \
  INSTANTIATE(T, 6, E, H)        \
  INSTANTIATE(T, 7, E, H)        \
  INSTANTIATE(T, 8, E, H)        \
  INSTANTIATE(T, 9, E, H)        \
  INSTANTIATE(T, 10, E, H)       \
  INSTANTIATE(T, 11, E, H)       \
  INSTANTIATE(T, 12, E, H)       \
  INSTANTIATE(T, 13, E, H)       \
  INSTANTIATE(T, 14, E, H)       \
  INSTANTIATE(T, 15, E, H)       \
  INSTANTIATE(T, 16, E, H)       \
  INSTANTIATE(T, 17, E, H)       \
  INSTANTIATE(T, 18, E, H)       \
  INSTANTIATE(T, 19, E, H)       \
  INSTANTIATE(T, 20, E, H)       \
  INSTANTIATE(T, 21, E, H)       \
  INSTANTIATE(T, 22, E, H)       \
  INSTANTIATE(T, 23, E, H)       \
  INSTANTIATE(T, 24, E, H)       \
  INSTANTIATE(T, 25, E, H)       \
  INSTANTIATE(T, 26, E, H)       \
  INSTANTIATE(T, 27, E, H)       \
  INSTANTIATE(T, 28, E, H)       \
  INSTANTIATE(T, 29, E, H)       \
  INSTANTIATE(T, 30, E, H)       \
  INSTANTIATE(T, 31, E, H)       \
  INSTANTIATE(T, 32, E, H)

INSTANTIATE_ALL(float, 256, 3072)
INSTANTIATE_ALL(__nv_bfloat16, 256, 3072)
INSTANTIATE_ALL(float, 128, 6144)
INSTANTIATE_ALL(__nv_bfloat16, 128, 6144)
INSTANTIATE_ALL(float, 256, 6144)
INSTANTIATE_ALL(__nv_bfloat16, 256, 6144)

#undef INSTANTIATE_ALL
#undef INSTANTIATE