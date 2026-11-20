// H2 candidate: split the head reduction over K, then finalize gates/collapse.
// Initially instantiate Split20 only for C500 HC4/H5120 BF16 N=1/2/3/4.
#pragma once
#include "fused_hc_head_h5120.cuh"

namespace mhc_head_h5120 {
__device__ __forceinline__ void load_head_bf16x8(
    const bfloat16_t* src, float (&values)[8]) {
  const uint4 packed = *reinterpret_cast<const uint4*>(src);
  const auto* pairs = reinterpret_cast<const __maca_bfloat162*>(&packed);
  #pragma unroll
  for (int p = 0; p < 4; ++p) {
    const float2 v = __bfloat1622float2(pairs[p]);
    values[2 * p] = v.x;
    values[2 * p + 1] = v.y;
  }
}

template <int Splits>
__global__ void __launch_bounds__(128, 1) hc_head_h5120_splitk_partial_kernel(
    const bfloat16_t* __restrict__ x, const float* __restrict__ fn,
    float* __restrict__ partials) {
#if defined(__MACA_ARCH__) && \
    (__MACA_ARCH__ != 1000) && (__MACA_ARCH__ != 1008)
  __trap();
#else
  static_assert(Splits > 0 && 20480 % Splits == 0);
  constexpr int Chunk = 20480 / Splits;
  static_assert(Chunk % 8 == 0);
  __shared__ float scratch[5 * 128];
  const int tid = threadIdx.x;
  const int split = blockIdx.y;
  const int token = blockIdx.x;
  float sumsq = 0.0f;
  float mixes[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  #pragma unroll
  for (int tile = 0; tile < (Chunk + 1023) / 1024; ++tile) {
    const int local = tile * 1024 + tid * 8;
    if (local < Chunk) {
      const int k = split * Chunk + local;
      float values[8];
      load_head_bf16x8(x + static_cast<int64_t>(token) * 20480 + k, values);
      #pragma unroll
      for (int j = 0; j < 8; ++j) sumsq += values[j] * values[j];
      #pragma unroll
      for (int m = 0; m < 4; ++m) {
        float weights[8];
        const float* wp = fn + m * 20480 + k;
        *reinterpret_cast<float4*>(weights) = *reinterpret_cast<const float4*>(wp);
        *reinterpret_cast<float4*>(weights + 4) = *reinterpret_cast<const float4*>(wp + 4);
        #pragma unroll
        for (int j = 0; j < 8; ++j) mixes[m] += values[j] * weights[j];
      }
    }
  }
  reduce_head_statistics(sumsq, mixes, scratch);
  if (tid == 0) {
    float* dst = partials + (static_cast<int64_t>(token) * Splits + split) * 5;
    dst[0] = sumsq;
    #pragma unroll
    for (int m = 0; m < 4; ++m) dst[m + 1] = mixes[m];
  }
#endif
}

template <int Splits>
__global__ void __launch_bounds__(128, 1) hc_head_h5120_splitk_collapse_kernel(
    const bfloat16_t* __restrict__ x, const float* __restrict__ partials,
    const float* __restrict__ scale, const float* __restrict__ base,
    bfloat16_t* __restrict__ y, float norm_eps, float hc_eps) {
#if defined(__MACA_ARCH__) && \
    (__MACA_ARCH__ != 1000) && (__MACA_ARCH__ != 1008)
  __trap();
#else
  __shared__ float inverse_rms;
  __shared__ float gates[4];
  const int tid = threadIdx.x;
  const int token = blockIdx.x;
  const int h = blockIdx.y * 1024 + tid * 8;
  float statistic = 0.0f;
  if (tid < 5) {
    const float* src = partials + static_cast<int64_t>(token) * Splits * 5 + tid;
    #pragma unroll
    for (int s = 0; s < Splits; ++s) statistic += src[s * 5];
    if (tid == 0) inverse_rms = rsqrtf(statistic / 20480.0f + norm_eps);
  }
  __syncthreads();
  if (tid >= 1 && tid < 5) {
    const int m = tid - 1;
    const float z = statistic * inverse_rms * scale[0] + base[m];
    gates[m] = 1.0f / (1.0f + expf(-z)) + hc_eps;
  }
  __syncthreads();
  float out[8] = {};
  #pragma unroll
  for (int m = 0; m < 4; ++m) {
    float values[8];
    load_head_bf16x8(x + static_cast<int64_t>(token) * 20480 + m * 5120 + h, values);
    const float gate = gates[m];
    #pragma unroll
    for (int j = 0; j < 8; ++j) out[j] += gate * values[j];
  }
  uint4 packed;
  auto* pairs = reinterpret_cast<__maca_bfloat162*>(&packed);
  #pragma unroll
  for (int p = 0; p < 4; ++p) {
    pairs[p] = __float22bfloat162_rn(make_float2(out[2 * p], out[2 * p + 1]));
  }
  *reinterpret_cast<uint4*>(y + static_cast<int64_t>(token) * 5120 + h) = packed;
#endif
}
}  // namespace mhc_head_h5120
