/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <ATen/cuda/CUDAContext.h>
#include "utils.h"

namespace {

template<typename T>
static __device__ __forceinline__ T float_to_dstT(float v) { return static_cast<T>(v); }
template<>
__device__ __forceinline__ maca_bfloat16 float_to_dstT(float v) { return __float2bfloat16(v); }
template<>
__device__ __forceinline__ half float_to_dstT(float v) { return __float2half(v); }

template<int N>
__device__ __forceinline__ void copy(const void* src, void* dst) {
  if constexpr (N == 16)      *(float4*)dst = *(const float4*)src;
  else if constexpr (N == 8)  *(float2*)dst = *(const float2*)src;
  else if constexpr (N == 4)  *(float*)dst  = *(const float*)src;
  else {
    #pragma unroll
    for (int i = 0; i < N; i++) {
      ((int8_t*)dst)[i] = ((const int8_t*)src)[i];
    }
  }
}

template<int MAX_M, uint32_t VEC, uint32_t NREG, uint32_t BS, typename T>
__device__ __forceinline__ void load_x(const T* p, uint32_t tid, int M, uint32_t d,
                                       T (&r)[NREG][MAX_M][VEC]) {
  #pragma unroll
  for (uint32_t k = 0; k < NREG; k++) {
    const uint32_t i = tid + k * BS;
    #pragma unroll
    for (int m = 0; m < MAX_M; m++) {
      if (i < d && m < M) {
        copy<sizeof(T) * VEC>(p + (size_t)m * d + i, r[k][m]);
      } else {
        #pragma unroll
        for (uint32_t j = 0; j < VEC; j++) {
          r[k][m][j] = float_to_dstT<T>(0.0f);
        }
      }
    }
  }
}

template<int MAX_M, uint32_t VEC, uint32_t NREG, typename T, int NT>
__global__ void __launch_bounds__(NT)
FusedHcHeadPersistent(const T* __restrict__ x, const float* __restrict__ hc_fn,
                      const float* __restrict__ hc_scale, const float* __restrict__ hc_base,
                      T* __restrict__ y, const uint32_t num_tokens, const int M,
                      const uint32_t d, const uint32_t stride_x, float norm_eps, float hc_eps) {
  static_assert(VEC % 4 == 0, "VEC must be a multiple of 4");
  constexpr int R = MAX_M + 1;                      // [0,MAX_M) 内积，[MAX_M] sumsq
  constexpr int NG = NT >> 4;                       // 16-lane 组数
  constexpr uint32_t BS = NT * VEC;
  __shared__ float sm_part[R * NG];
  __shared__ float sm_tot[R];

  const uint32_t K = (uint32_t)M * d;
  const uint32_t tid = threadIdx.x * VEC;
  const int lane = threadIdx.x & 15, group = threadIdx.x >> 4;
  const float scale = hc_scale[0];
  float base[MAX_M];
  #pragma unroll
  for (int m = 0; m < MAX_M; m++) {
    base[m] = (m < M) ? hc_base[m] : 0.0f;
  }

  float w[NREG][MAX_M][MAX_M][VEC];
  #pragma unroll
  for (uint32_t k = 0; k < NREG; k++) {
    const uint32_t i = tid + k * BS;
    #pragma unroll
    for (int mp = 0; mp < MAX_M; mp++) {
      #pragma unroll
      for (int m = 0; m < MAX_M; m++) {
        if (i < d && mp < M && m < M) {
          const float* src = hc_fn + (size_t)mp * K + (size_t)m * d + i;
          #pragma unroll
          for (uint32_t q = 0; q < VEC / 4; q++) {
            copy<16>(src + 4 * q, w[k][mp][m] + 4 * q);
          }
        } else {
          #pragma unroll
          for (uint32_t j = 0; j < VEC; j++) {
            w[k][mp][m][j] = 0.0f;
          }
        }
      }
    }
  }

  T xc[NREG][MAX_M][VEC], xn[NREG][MAX_M][VEC];
  uint32_t tok = blockIdx.x;
  if (tok < num_tokens) {
    load_x<MAX_M, VEC, NREG, BS>(x + (size_t)tok * stride_x, tid, M, d, xc);
  }

  for (; tok < num_tokens; tok += gridDim.x) {
    if (tok + gridDim.x < num_tokens) {
      load_x<MAX_M, VEC, NREG, BS>(x + (size_t)(tok + gridDim.x) * stride_x, tid, M, d, xn);
    }

    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
      acc[r] = 0.0f;
    }
    #pragma unroll
    for (uint32_t k = 0; k < NREG; k++) {
      #pragma unroll
      for (int m = 0; m < MAX_M; m++) {
        float xf[VEC];
        #pragma unroll
        for (uint32_t j = 0; j < VEC; j++) {
          xf[j] = static_cast<float>(xc[k][m][j]);
          acc[MAX_M] += xf[j] * xf[j];
        }
        #pragma unroll
        for (int mp = 0; mp < MAX_M; mp++) {
          #pragma unroll
          for (uint32_t j = 0; j < VEC; j++) {
            acc[mp] += w[k][mp][m][j] * xf[j];
          }
        }
      }
    }

    #pragma unroll
    for (int r = 0; r < R; r++) {
      float s = acc[r];
      #pragma unroll
      for (int o = 8; o > 0; o >>= 1) s += __shfl_down_sync_16(0xffffffffffffffff, s, o);
      if (lane == 0) sm_part[r * NG + group] = s;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < R * 16; idx += NT) {
      const int r = idx >> 4, l = idx & 15;
      float s = 0.0f;
      #pragma unroll
      for (int g = l; g < NG; g += 16) {
        s += sm_part[r * NG + g];
      }
      #pragma unroll
      for (int o = 8; o > 0; o >>= 1) {
        s += __shfl_down_sync_16(0xffffffffffffffff, s, o);
      }
      if (l == 0) {
        sm_tot[r] = s;
      }
    }
    __syncthreads();

    float pre[MAX_M];
    const float rs = rsqrtf(sm_tot[MAX_M] / (float)K + norm_eps);
    #pragma unroll
    for (int m = 0; m < MAX_M; m++) {
      const float z = sm_tot[m] * rs * scale + base[m];
      pre[m] = (m < M) ? __builtin_mxc_rcpf(1.0f + __expf(-z)) + hc_eps : 0.0f;
    }

    T* ptr_y = y + (size_t)tok * d;
    #pragma unroll
    for (uint32_t k = 0; k < NREG; k++) {
      const uint32_t i = tid + k * BS;
      if (i < d) {
        float out[VEC];
        #pragma unroll
        for (uint32_t j = 0; j < VEC; j++) {
          out[j] = 0.0f;
        }
        #pragma unroll
        for (int m = 0; m < MAX_M; m++) {
          #pragma unroll
          for (uint32_t j = 0; j < VEC; j++) {
            out[j] += pre[m] * static_cast<float>(xc[k][m][j]);
          }
        }
        T reg_dst[VEC];
        #pragma unroll
        for (uint32_t j = 0; j < VEC; j++) {
          reg_dst[j] = float_to_dstT<T>(out[j]);
        }
        copy<sizeof(T) * VEC>(reg_dst, ptr_y + i);
      }
    }

    #pragma unroll
    for (uint32_t k = 0; k < NREG; k++) {
      #pragma unroll
      for (int m = 0; m < MAX_M; m++) {
        #pragma unroll
        for (uint32_t j = 0; j < VEC; j++) {
          xc[k][m][j] = xn[k][m][j];
        }
      }
    }
  }
}

template<typename T>
void launch_fused_hc_head(const torch::Tensor& output, const torch::Tensor& x,
                          const float* fn, const float* sc, const float* bs,
                          uint32_t num_tokens, int hc_mult, uint32_t hidden, uint32_t stride_x,
                          float ne, float he, uint32_t num_ap, cudaStream_t stream) {
  constexpr int NT = 1024;
  constexpr uint32_t PV = 4;
  const T* xp = static_cast<const T*>(x.data_ptr());
  T* yp = static_cast<T*>(output.data_ptr());

  TORCH_CHECK(((uintptr_t)xp % 16 == 0) && ((uintptr_t)yp % 16 == 0) && ((uintptr_t)fn % 16 == 0),
              "fused_hc_head: x / output / hc_fn must be 16B aligned");
  TORCH_CHECK(hidden % PV == 0 && stride_x % PV == 0,
              "fused_hc_head: hidden and x.stride(0) must be multiples of ", PV);

  const uint32_t grid = std::min<uint32_t>(num_tokens, num_ap);

#define LAUNCH(MAXM, NREG)                                                         \
  {                                                                                \
    FusedHcHeadPersistent<MAXM, PV, NREG, T, NT><<<grid, NT, 0, stream>>>(         \
        xp, fn, sc, bs, yp, num_tokens, hc_mult, hidden, stride_x, ne, he);        \
    return;                                                                        \
  }

  // 约束：MAX_M^2 * PV * NREG <= 64
  if (hc_mult <= 2 && hidden <= NT * PV * 1) LAUNCH(2, 1)
  if (hc_mult <= 2 && hidden <= NT * PV * 2) LAUNCH(2, 2)
  if (hc_mult <= 2 && hidden <= NT * PV * 4) LAUNCH(2, 4)
  if (hc_mult <= 4 && hidden <= NT * PV * 1) LAUNCH(4, 1)

  TORCH_CHECK(false, "fused_hc_head: unsupported shape hc_mult=", hc_mult, ", hidden=", hidden);
}

}  // namespace

void sgl_fused_hc_head(torch::Tensor output, torch::Tensor x, torch::Tensor hc_fn,
                       torch::Tensor hc_scale, torch::Tensor hc_base,
                       double norm_eps, double hc_eps) {
  CHECK_INPUT(output);
  CHECK_INPUT(hc_fn);
  CHECK_INPUT(hc_scale);
  CHECK_INPUT(hc_base);
  TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
  auto device = x.device();
  CHECK_EQ(output.device(), device);
  CHECK_EQ(hc_fn.device(), device);
  CHECK_EQ(hc_scale.device(), device);
  CHECK_EQ(hc_base.device(), device);
  CHECK_DIM(3, x);        // (num_tokens, hc_mult, hidden)
  CHECK_DIM(2, output);   // (num_tokens, hidden)
  CHECK_DIM(2, hc_fn);    // (hc_mult, hc_mult * hidden)
  TORCH_CHECK(hc_fn.scalar_type() == at::ScalarType::Float &&
              hc_scale.scalar_type() == at::ScalarType::Float &&
              hc_base.scalar_type() == at::ScalarType::Float,
              "hc_fn / hc_scale / hc_base must be float32");
  TORCH_CHECK(output.scalar_type() == x.scalar_type(), "output dtype must match x");

  const uint32_t num_tokens = x.size(0);
  const int hc_mult = x.size(1);
  const uint32_t hidden = x.size(2);
  TORCH_CHECK(x.stride(2) == 1 && x.stride(1) == (int64_t)hidden,
              "x must be contiguous within each token");
  TORCH_CHECK(hc_mult >= 1 && hc_mult <= 4, "hc_mult must be in [1, 4], got ", hc_mult);
  CHECK_EQ(output.size(0), num_tokens);
  CHECK_EQ(output.size(1), hidden);
  CHECK_EQ(hc_fn.size(0), hc_mult);
  CHECK_EQ(hc_fn.size(1), (int64_t)hc_mult * hidden);
  CHECK_EQ(hc_base.numel(), hc_mult);
  CHECK_EQ(hc_scale.numel(), 1);
  if (num_tokens == 0) return;

  const uint32_t num_ap = at::cuda::getDeviceProperties(device.index())->multiProcessorCount;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float ne = static_cast<float>(norm_eps);
  const float he = static_cast<float>(hc_eps);
  const float* fn = hc_fn.data_ptr<float>();
  const float* sc = hc_scale.data_ptr<float>();
  const float* bs = hc_base.data_ptr<float>();
  const uint32_t stride_x = x.stride(0);

  switch (x.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_fused_hc_head<maca_bfloat16>(output, x, fn, sc, bs, num_tokens, hc_mult, hidden,
                                          stride_x, ne, he, num_ap, stream);
      break;
    case at::ScalarType::Half:
      launch_fused_hc_head<half>(output, x, fn, sc, bs, num_tokens, hc_mult, hidden,
                                 stride_x, ne, he, num_ap, stream);
      break;
    case at::ScalarType::Float:
      launch_fused_hc_head<float>(output, x, fn, sc, bs, num_tokens, hc_mult, hidden,
                                  stride_x, ne, he, num_ap, stream);
      break;
    default:
      TORCH_CHECK(false, "fused_hc_head: unsupported dtype ", x.scalar_type());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}