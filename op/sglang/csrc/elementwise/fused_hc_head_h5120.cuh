// H5120 head: frozen H0, shape-selected H1, and H3 ownership candidate from vLLM-metax 35abce80ea5a7f67c712d808efd59acefae059e0.
// Frozen artifact b8169075388ce57569f1bb82054a96beb357804d5cbb2d5a9c96cc7ff13a3817.
// Original device.cu SHA256:
// 78df00f099a12b49eeff4e22a1168c4fb392b112b1a24fc65b45cdb265679c6e
// H0 preserved the exported kernel verbatim and passed correctness/timing parity.
// H1 changes only the kernel symbol and the five reductions: one shared exchange
// handles all statistics, preserving each value's 64/32/16/8/4/2/1 add tree.
// Launch: one CTA/token, 128 threads, 16896 dynamic shared bytes. Fixed contract:
// contiguous BF16 [N,4,5120], FP32 fn [4,20480], norm_eps=hc_eps=1e-6.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace mhc_head_h5120 {
using bfloat16_t = maca_bfloat16;
using uchar = unsigned char;

// Keep optimized entry points in every image for host registration. Only
// xcore1000/xcore1008 compile their implementation; other device images contain
// trap stubs. The host architecture guard routes those devices to H0 instead.

// The five independent reductions share one pair of CTA barriers. Their scratch
// region occupies 2560 bytes, then the existing gate-stage barrier permits the
// output pipeline to reuse it. Peak dynamic shared memory remains 16896 bytes.
__device__ __forceinline__ void reduce_head_statistics(
    float& sumsq, float (&mixes)[4], float* scratch) {
  const int tid = threadIdx.x;
  __syncthreads();
  scratch[tid] = sumsq;
  #pragma unroll
  for (int m = 0; m < 4; ++m) {
    scratch[(m + 1) * 128 + tid] = mixes[m];
  }
  __syncthreads();
  sumsq += scratch[tid ^ 64];
  #pragma unroll
  for (int m = 0; m < 4; ++m) {
    mixes[m] += scratch[(m + 1) * 128 + (tid ^ 64)];
  }
  #pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    sumsq += __shfl_xor_sync(uint64_t(-1), sumsq, offset);
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
      mixes[m] += __shfl_xor_sync(uint64_t(-1), mixes[m], offset);
    }
  }
}

extern "C" __global__ void hc_head_h5120_batched_reduce_kernel(const float* __restrict__ fn, const float* __restrict__ hc_base, const float* __restrict__ hc_scale, bfloat16_t* __restrict__ out, const bfloat16_t* __restrict__ residual, int num_tokens);
extern "C" __global__ void __launch_bounds__(128, 1) hc_head_h5120_batched_reduce_kernel(const float* __restrict__ fn, const float* __restrict__ hc_base, const float* __restrict__ hc_scale, bfloat16_t* __restrict__ out, const bfloat16_t* __restrict__ residual, int num_tokens) {
#if defined(__MACA_ARCH__) && \
    (__MACA_ARCH__ != 1000) && (__MACA_ARCH__ != 1008)
  __trap();
#else
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float sqrsum_r[1];
  float mixes_r[4];
  bfloat16_t residual_local_cast[8];
  float x_local[8];
  float fn_local[8];
  float rsqrt_val[1];
  float xl[32];
  float ol[8];
  bfloat16_t xs_local_cast_1[8];
  bfloat16_t out_local_cast_2[8];
  bfloat16_t xs_local_cast_3[8];
  bfloat16_t out_local_cast_4[8];
  bfloat16_t xs_local_cast_5[8];
  bfloat16_t out_local_cast_6[8];
  sqrsum_r[0] = 0x0p+0f/*0.000000e+00*/;
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(mixes_r + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  for (int m_c = 0; m_c < 4; ++m_c) {
    for (int i_h = 0; i_h < 5; ++i_h) {
      *(uint4*)(residual_local_cast + 0) = *(uint4*)(residual + ((((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)m_c) * (int64_t)5120)) + (((int64_t)i_h) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
      for (int i = 0; i < 2; ++i) {
        float4 __1;
        uint2 v_ = *(uint2*)(residual_local_cast + (i * 4));
        ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v_))[0]);
        ((float2*)(&__1))[1] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v_))[1]);
        *(float4*)(x_local + (i * 4)) = __1;
      }
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        sqrsum_r[0] = (sqrsum_r[0] + (x_local[i_1] * x_local[i_1]));
      }
      #pragma unroll
      for (int m_m = 0; m_m < 4; ++m_m) {
        #pragma unroll
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          *(float4*)(fn_local + (i_2 * 4)) = *(float4*)(fn + (((((m_m * 20480) + (m_c * 5120)) + (i_h * 1024)) + (((int)threadIdx.x) * 8)) + (i_2 * 4)));
        }
        #pragma unroll
        for (int i_3 = 0; i_3 < 8; ++i_3) {
          mixes_r[m_m] = (mixes_r[m_m] + (x_local[i_3] * fn_local[i_3]));
        }
      }
    }
  }
  reduce_head_statistics(sqrsum_r[0], mixes_r, reinterpret_cast<float*>(buf_dyn_shmem));
  rsqrt_val[0] = rsqrtf(((sqrsum_r[0] / 0x1.4p+14f/*2.048000e+04*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  __syncthreads();
  if ((((int)threadIdx.x) >> 2) == 0) {
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) & 3)] = ((0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - (((mixes_r[(((int)threadIdx.x) & 3)] * rsqrt_val[0]) * hc_scale[0]) + hc_base[(((int)threadIdx.x) & 3)]))))) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((i_4 * 1024) + (((int)threadIdx.x) * 8)) + 256)) = *(uint4*)(residual + (((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)i_4) * (int64_t)5120)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((i_5 * 1024) + (((int)threadIdx.x) * 8)) + 4352)) = *(uint4*)(residual + ((((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)i_5) * (int64_t)5120)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)1024));
  }
  __syncthreads();
  for (int i0_h = 0; i0_h < 3; ++i0_h) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      *(uint4*)(xs_local_cast_1 + 0) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((i0_h & 1) * 4096) + (i_6 * 1024)) + (((int)threadIdx.x) * 8)) + 256));
      for (int vec = 0; vec < 2; ++vec) {
        float4 __2;
        uint2 v__1 = *(uint2*)(xs_local_cast_1 + (vec * 4));
        ((float2*)(&__2))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__1))[0]);
        ((float2*)(&__2))[1] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__1))[1]);
        *(float4*)(xl + ((i_6 * 8) + (vec * 4))) = __2;
      }
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((i0_h & 1) * 4096) + (i_7 * 1024)) + (((int)threadIdx.x) * 8)) + 256)) = *(uint4*)(residual + (((((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)i_7) * (int64_t)5120)) + (((int64_t)i0_h) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)2048));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(ol + (i_8 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    for (int i_hc = 0; i_hc < 4; ++i_hc) {
      float pre = ((float*)buf_dyn_shmem)[i_hc];
      #pragma unroll
      for (int i_9 = 0; i_9 < 8; ++i_9) {
        ol[i_9] = (ol[i_9] + (pre * xl[((i_hc * 8) + i_9)]));
      }
    }
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      uint2 __3;
      float4 v__2 = *(float4*)(ol + (i_10 * 4));
      (reinterpret_cast<__maca_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
      (reinterpret_cast<__maca_bfloat162*>(&__3))[1] = __float22bfloat162_rn(((float2*)(&v__2))[1]);
      *(uint2*)(out_local_cast_2 + (i_10 * 4)) = __3;
    }
    *(uint4*)(out + (((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)i0_h) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(out_local_cast_2 + 0);
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 4; ++i_11) {
    *(uint4*)(xs_local_cast_3 + 0) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((i_11 * 1024) + (((int)threadIdx.x) * 8)) + 4352));
    for (int vec_1 = 0; vec_1 < 2; ++vec_1) {
      float4 __4;
      uint2 v__3 = *(uint2*)(xs_local_cast_3 + (vec_1 * 4));
      ((float2*)(&__4))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__3))[0]);
      ((float2*)(&__4))[1] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__3))[1]);
      *(float4*)(xl + ((i_11 * 8) + (vec_1 * 4))) = __4;
    }
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 2; ++i_12) {
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(ol + (i_12 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
  }
  for (int i_hc_1 = 0; i_hc_1 < 4; ++i_hc_1) {
    float pre_1 = ((float*)buf_dyn_shmem)[i_hc_1];
    #pragma unroll
    for (int i_13 = 0; i_13 < 8; ++i_13) {
      ol[i_13] = (ol[i_13] + (pre_1 * xl[((i_hc_1 * 8) + i_13)]));
    }
  }
  for (int i_14 = 0; i_14 < 2; ++i_14) {
    uint2 __5;
    float4 v__4 = *(float4*)(ol + (i_14 * 4));
    (reinterpret_cast<__maca_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
    (reinterpret_cast<__maca_bfloat162*>(&__5))[1] = __float22bfloat162_rn(((float2*)(&v__4))[1]);
    *(uint2*)(out_local_cast_4 + (i_14 * 4)) = __5;
  }
  *(uint4*)(out + (((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)3072)) = *(uint4*)(out_local_cast_4 + 0);
  #pragma unroll
  for (int i_15 = 0; i_15 < 4; ++i_15) {
    *(uint4*)(xs_local_cast_5 + 0) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((i_15 * 1024) + (((int)threadIdx.x) * 8)) + 256));
    for (int vec_2 = 0; vec_2 < 2; ++vec_2) {
      float4 __6;
      uint2 v__5 = *(uint2*)(xs_local_cast_5 + (vec_2 * 4));
      ((float2*)(&__6))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__5))[0]);
      ((float2*)(&__6))[1] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__5))[1]);
      *(float4*)(xl + ((i_15 * 8) + (vec_2 * 4))) = __6;
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 2; ++i_16) {
    float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(ol + (i_16 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
  }
  for (int i_hc_2 = 0; i_hc_2 < 4; ++i_hc_2) {
    float pre_2 = ((float*)buf_dyn_shmem)[i_hc_2];
    #pragma unroll
    for (int i_17 = 0; i_17 < 8; ++i_17) {
      ol[i_17] = (ol[i_17] + (pre_2 * xl[((i_hc_2 * 8) + i_17)]));
    }
  }
  for (int i_18 = 0; i_18 < 2; ++i_18) {
    uint2 __7;
    float4 v__6 = *(float4*)(ol + (i_18 * 4));
    (reinterpret_cast<__maca_bfloat162*>(&__7))[0] = __float22bfloat162_rn(((float2*)(&v__6))[0]);
    (reinterpret_cast<__maca_bfloat162*>(&__7))[1] = __float22bfloat162_rn(((float2*)(&v__6))[1]);
    *(uint2*)(out_local_cast_6 + (i_18 * 4)) = __7;
  }
  *(uint4*)(out + (((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)4096)) = *(uint4*)(out_local_cast_6 + 0);
#endif
}

// Original H0 stays available on every architecture and for unqualified N.

// Minimal dependencies of the generated kernel; preserve TileLang's reduction
// tree, shared-memory exchange and barriers without a TileLang build dependency.
// Derived from tl_templates/maca/common.h and reduce.h:
// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd.
// All rights reserved.
namespace tl {
template <typename T>
__forceinline__ __device__ T shfl_xor_sync(uint64_t mask, T val, int lane_mask) {
  return __shfl_xor_sync(mask, val, lane_mask);
}

struct SumOp {
  template <typename T>
  __forceinline__ __device__ T operator()(T const& x, T const& y) {
    return x + y;
  }
};

template <class Reducer, int threads, int scale, int thread_offset = 0,
          int all_threads = threads>
struct AllReduce {
  static_assert(threads == 1024 || threads == 512 || threads == 256 ||
                threads == 128 || threads == 64 || threads == 32 ||
                threads == 16 || threads == 8 || threads == 4 || threads == 2);
  static_assert(threads % scale == 0);
  template <typename T> static __device__ T run(T x, T* red_buf = nullptr) {
    constexpr int offset = threads / 2;
    constexpr int warpSize = 64;
    if constexpr (offset >= warpSize) {
      __syncthreads();
      red_buf[threadIdx.x - thread_offset] = x;
      __syncthreads();
      x = Reducer()(x, red_buf[(threadIdx.x - thread_offset) ^ offset]);
    } else {
      x = Reducer()(x, tl::shfl_xor_sync(uint64_t(-1), x, offset));
    }
    if constexpr (offset == scale) {
      return x;
    } else {
      return AllReduce<Reducer, offset, scale, thread_offset, all_threads>::run(
          x, red_buf);
    }
  }
};
}  // namespace tl

extern "C" __global__ void hc_head_fuse_tilelang_kernel(const float* __restrict__ fn, const float* __restrict__ hc_base, const float* __restrict__ hc_scale, bfloat16_t* __restrict__ out, const bfloat16_t* __restrict__ residual, int num_tokens);
extern "C" __global__ void __launch_bounds__(128, 1) hc_head_fuse_tilelang_kernel(const float* __restrict__ fn, const float* __restrict__ hc_base, const float* __restrict__ hc_scale, bfloat16_t* __restrict__ out, const bfloat16_t* __restrict__ residual, int num_tokens) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float sqrsum_r[1];
  float mixes_r[4];
  bfloat16_t residual_local_cast[8];
  float x_local[8];
  float fn_local[8];
  float rsqrt_val[1];
  float xl[32];
  float ol[8];
  bfloat16_t xs_local_cast_1[8];
  bfloat16_t out_local_cast_2[8];
  bfloat16_t xs_local_cast_3[8];
  bfloat16_t out_local_cast_4[8];
  bfloat16_t xs_local_cast_5[8];
  bfloat16_t out_local_cast_6[8];
  sqrsum_r[0] = 0x0p+0f/*0.000000e+00*/;
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(mixes_r + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  for (int m_c = 0; m_c < 4; ++m_c) {
    for (int i_h = 0; i_h < 5; ++i_h) {
      *(uint4*)(residual_local_cast + 0) = *(uint4*)(residual + ((((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)m_c) * (int64_t)5120)) + (((int64_t)i_h) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
      for (int i = 0; i < 2; ++i) {
        float4 __1;
        uint2 v_ = *(uint2*)(residual_local_cast + (i * 4));
        ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v_))[0]);
        ((float2*)(&__1))[1] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v_))[1]);
        *(float4*)(x_local + (i * 4)) = __1;
      }
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        sqrsum_r[0] = (sqrsum_r[0] + (x_local[i_1] * x_local[i_1]));
      }
      #pragma unroll
      for (int m_m = 0; m_m < 4; ++m_m) {
        #pragma unroll
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          *(float4*)(fn_local + (i_2 * 4)) = *(float4*)(fn + (((((m_m * 20480) + (m_c * 5120)) + (i_h * 1024)) + (((int)threadIdx.x) * 8)) + (i_2 * 4)));
        }
        #pragma unroll
        for (int i_3 = 0; i_3 < 8; ++i_3) {
          mixes_r[m_m] = (mixes_r[m_m] + (x_local[i_3] * fn_local[i_3]));
        }
      }
    }
  }
  sqrsum_r[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(sqrsum_r[0], (&(((float*)buf_dyn_shmem)[0])));
  __syncthreads();
  for (int __finred_0 = 0; __finred_0 < 4; ++__finred_0) {
    mixes_r[__finred_0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(mixes_r[__finred_0], (&(((float*)buf_dyn_shmem)[0])));
  }
  rsqrt_val[0] = rsqrtf(((sqrsum_r[0] / 0x1.4p+14f/*2.048000e+04*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  __syncthreads();
  if ((((int)threadIdx.x) >> 2) == 0) {
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) & 3)] = ((0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - (((mixes_r[(((int)threadIdx.x) & 3)] * rsqrt_val[0]) * hc_scale[0]) + hc_base[(((int)threadIdx.x) & 3)]))))) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((i_4 * 1024) + (((int)threadIdx.x) * 8)) + 256)) = *(uint4*)(residual + (((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)i_4) * (int64_t)5120)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((i_5 * 1024) + (((int)threadIdx.x) * 8)) + 4352)) = *(uint4*)(residual + ((((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)i_5) * (int64_t)5120)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)1024));
  }
  __syncthreads();
  for (int i0_h = 0; i0_h < 3; ++i0_h) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      *(uint4*)(xs_local_cast_1 + 0) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((i0_h & 1) * 4096) + (i_6 * 1024)) + (((int)threadIdx.x) * 8)) + 256));
      for (int vec = 0; vec < 2; ++vec) {
        float4 __2;
        uint2 v__1 = *(uint2*)(xs_local_cast_1 + (vec * 4));
        ((float2*)(&__2))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__1))[0]);
        ((float2*)(&__2))[1] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__1))[1]);
        *(float4*)(xl + ((i_6 * 8) + (vec * 4))) = __2;
      }
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((i0_h & 1) * 4096) + (i_7 * 1024)) + (((int)threadIdx.x) * 8)) + 256)) = *(uint4*)(residual + (((((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)i_7) * (int64_t)5120)) + (((int64_t)i0_h) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)2048));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(ol + (i_8 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    for (int i_hc = 0; i_hc < 4; ++i_hc) {
      float pre = ((float*)buf_dyn_shmem)[i_hc];
      #pragma unroll
      for (int i_9 = 0; i_9 < 8; ++i_9) {
        ol[i_9] = (ol[i_9] + (pre * xl[((i_hc * 8) + i_9)]));
      }
    }
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      uint2 __3;
      float4 v__2 = *(float4*)(ol + (i_10 * 4));
      (reinterpret_cast<__maca_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
      (reinterpret_cast<__maca_bfloat162*>(&__3))[1] = __float22bfloat162_rn(((float2*)(&v__2))[1]);
      *(uint2*)(out_local_cast_2 + (i_10 * 4)) = __3;
    }
    *(uint4*)(out + (((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)i0_h) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(out_local_cast_2 + 0);
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 4; ++i_11) {
    *(uint4*)(xs_local_cast_3 + 0) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((i_11 * 1024) + (((int)threadIdx.x) * 8)) + 4352));
    for (int vec_1 = 0; vec_1 < 2; ++vec_1) {
      float4 __4;
      uint2 v__3 = *(uint2*)(xs_local_cast_3 + (vec_1 * 4));
      ((float2*)(&__4))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__3))[0]);
      ((float2*)(&__4))[1] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__3))[1]);
      *(float4*)(xl + ((i_11 * 8) + (vec_1 * 4))) = __4;
    }
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 2; ++i_12) {
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(ol + (i_12 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
  }
  for (int i_hc_1 = 0; i_hc_1 < 4; ++i_hc_1) {
    float pre_1 = ((float*)buf_dyn_shmem)[i_hc_1];
    #pragma unroll
    for (int i_13 = 0; i_13 < 8; ++i_13) {
      ol[i_13] = (ol[i_13] + (pre_1 * xl[((i_hc_1 * 8) + i_13)]));
    }
  }
  for (int i_14 = 0; i_14 < 2; ++i_14) {
    uint2 __5;
    float4 v__4 = *(float4*)(ol + (i_14 * 4));
    (reinterpret_cast<__maca_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
    (reinterpret_cast<__maca_bfloat162*>(&__5))[1] = __float22bfloat162_rn(((float2*)(&v__4))[1]);
    *(uint2*)(out_local_cast_4 + (i_14 * 4)) = __5;
  }
  *(uint4*)(out + (((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)3072)) = *(uint4*)(out_local_cast_4 + 0);
  #pragma unroll
  for (int i_15 = 0; i_15 < 4; ++i_15) {
    *(uint4*)(xs_local_cast_5 + 0) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((i_15 * 1024) + (((int)threadIdx.x) * 8)) + 256));
    for (int vec_2 = 0; vec_2 < 2; ++vec_2) {
      float4 __6;
      uint2 v__5 = *(uint2*)(xs_local_cast_5 + (vec_2 * 4));
      ((float2*)(&__6))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__5))[0]);
      ((float2*)(&__6))[1] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__5))[1]);
      *(float4*)(xl + ((i_15 * 8) + (vec_2 * 4))) = __6;
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 2; ++i_16) {
    float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(ol + (i_16 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
  }
  for (int i_hc_2 = 0; i_hc_2 < 4; ++i_hc_2) {
    float pre_2 = ((float*)buf_dyn_shmem)[i_hc_2];
    #pragma unroll
    for (int i_17 = 0; i_17 < 8; ++i_17) {
      ol[i_17] = (ol[i_17] + (pre_2 * xl[((i_hc_2 * 8) + i_17)]));
    }
  }
  for (int i_18 = 0; i_18 < 2; ++i_18) {
    uint2 __7;
    float4 v__6 = *(float4*)(ol + (i_18 * 4));
    (reinterpret_cast<__maca_bfloat162*>(&__7))[0] = __float22bfloat162_rn(((float2*)(&v__6))[0]);
    (reinterpret_cast<__maca_bfloat162*>(&__7))[1] = __float22bfloat162_rn(((float2*)(&v__6))[1]);
    *(uint2*)(out_local_cast_6 + (i_18 * 4)) = __7;
  }
  *(uint4*)(out + (((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)4096)) = *(uint4*)(out_local_cast_6 + 0);
}

// H3 ownership-only screen. The reduction policy matches the retained
// per-N baseline: H1 batches on its qualified N, otherwise H0 separate sums.
__device__ __forceinline__ void reduce_head_statistics_256(
    float& sumsq, float (&mixes)[4], float* scratch) {
  const int tid = threadIdx.x;
  #pragma unroll
  for (int offset = 128; offset >= 64; offset >>= 1) {
    __syncthreads();
    scratch[tid] = sumsq;
    #pragma unroll
    for (int m = 0; m < 4; ++m) scratch[(m + 1) * 256 + tid] = mixes[m];
    __syncthreads();
    sumsq += scratch[tid ^ offset];
    #pragma unroll
    for (int m = 0; m < 4; ++m) mixes[m] += scratch[(m + 1) * 256 + (tid ^ offset)];
  }
  #pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    sumsq += __shfl_xor_sync(uint64_t(-1), sumsq, offset);
    #pragma unroll
    for (int m = 0; m < 4; ++m) mixes[m] += __shfl_xor_sync(uint64_t(-1), mixes[m], offset);
  }
}

template <bool BatchedReduction>
__global__ void __launch_bounds__(256, 1) hc_head_h5120_threads256_kernel(const float* __restrict__ fn, const float* __restrict__ hc_base, const float* __restrict__ hc_scale, bfloat16_t* __restrict__ out, const bfloat16_t* __restrict__ residual, int num_tokens) {
#if defined(__MACA_ARCH__) && \
    (__MACA_ARCH__ != 1000) && (__MACA_ARCH__ != 1008)
  __trap();
#else
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float sqrsum_r[1];
  float mixes_r[4];
  bfloat16_t residual_local_cast[4];
  float x_local[4];
  float fn_local[4];
  float rsqrt_val[1];
  float xl[16];
  float ol[4];
  bfloat16_t xs_local_cast_1[4];
  bfloat16_t out_local_cast_2[4];
  bfloat16_t xs_local_cast_3[4];
  bfloat16_t out_local_cast_4[4];
  bfloat16_t xs_local_cast_5[4];
  bfloat16_t out_local_cast_6[4];
  sqrsum_r[0] = 0x0p+0f/*0.000000e+00*/;
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(mixes_r + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  for (int m_c = 0; m_c < 4; ++m_c) {
    for (int i_h = 0; i_h < 5; ++i_h) {
      *(uint2*)(residual_local_cast + 0) = *(uint2*)(residual + ((((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)m_c) * (int64_t)5120)) + (((int64_t)i_h) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
      for (int i = 0; i < 1; ++i) {
        float4 __1;
        uint2 v_ = *(uint2*)(residual_local_cast + (i * 4));
        ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v_))[0]);
        ((float2*)(&__1))[1] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v_))[1]);
        *(float4*)(x_local + (i * 4)) = __1;
      }
      #pragma unroll
      for (int i_1 = 0; i_1 < 4; ++i_1) {
        sqrsum_r[0] = (sqrsum_r[0] + (x_local[i_1] * x_local[i_1]));
      }
      #pragma unroll
      for (int m_m = 0; m_m < 4; ++m_m) {
        #pragma unroll
        for (int i_2 = 0; i_2 < 1; ++i_2) {
          *(float4*)(fn_local + (i_2 * 4)) = *(float4*)(fn + (((((m_m * 20480) + (m_c * 5120)) + (i_h * 1024)) + (((int)threadIdx.x) * 4)) + (i_2 * 4)));
        }
        #pragma unroll
        for (int i_3 = 0; i_3 < 4; ++i_3) {
          mixes_r[m_m] = (mixes_r[m_m] + (x_local[i_3] * fn_local[i_3]));
        }
      }
    }
  }
  if constexpr (BatchedReduction) {
    reduce_head_statistics_256(sqrsum_r[0], mixes_r, reinterpret_cast<float*>(buf_dyn_shmem));
  } else {
    sqrsum_r[0] = tl::AllReduce<tl::SumOp, 256, 1, 0>::run(sqrsum_r[0], (&(((float*)buf_dyn_shmem)[0])));
    __syncthreads();
    for (int __finred_0 = 0; __finred_0 < 4; ++__finred_0) {
      mixes_r[__finred_0] = tl::AllReduce<tl::SumOp, 256, 1, 0>::run(mixes_r[__finred_0], (&(((float*)buf_dyn_shmem)[0])));
    }
  }
  rsqrt_val[0] = rsqrtf(((sqrsum_r[0] / 0x1.4p+14f/*2.048000e+04*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  __syncthreads();
  if ((((int)threadIdx.x) >> 2) == 0) {
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) & 3)] = ((0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - (((mixes_r[(((int)threadIdx.x) & 3)] * rsqrt_val[0]) * hc_scale[0]) + hc_base[(((int)threadIdx.x) & 3)]))))) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i_4 * 1024) + (((int)threadIdx.x) * 4)) + 256)) = *(uint2*)(residual + (((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)i_4) * (int64_t)5120)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i_5 * 1024) + (((int)threadIdx.x) * 4)) + 4352)) = *(uint2*)(residual + ((((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)i_5) * (int64_t)5120)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (int64_t)1024));
  }
  __syncthreads();
  for (int i0_h = 0; i0_h < 3; ++i0_h) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      *(uint2*)(xs_local_cast_1 + 0) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((i0_h & 1) * 4096) + (i_6 * 1024)) + (((int)threadIdx.x) * 4)) + 256));
      for (int vec = 0; vec < 1; ++vec) {
        float4 __2;
        uint2 v__1 = *(uint2*)(xs_local_cast_1 + (vec * 4));
        ((float2*)(&__2))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__1))[0]);
        ((float2*)(&__2))[1] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__1))[1]);
        *(float4*)(xl + ((i_6 * 4) + (vec * 4))) = __2;
      }
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((i0_h & 1) * 4096) + (i_7 * 1024)) + (((int)threadIdx.x) * 4)) + 256)) = *(uint2*)(residual + (((((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)i_7) * (int64_t)5120)) + (((int64_t)i0_h) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (int64_t)2048));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 1; ++i_8) {
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(ol + (i_8 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    for (int i_hc = 0; i_hc < 4; ++i_hc) {
      float pre = ((float*)buf_dyn_shmem)[i_hc];
      #pragma unroll
      for (int i_9 = 0; i_9 < 4; ++i_9) {
        ol[i_9] = (ol[i_9] + (pre * xl[((i_hc * 4) + i_9)]));
      }
    }
    for (int i_10 = 0; i_10 < 1; ++i_10) {
      uint2 __3;
      float4 v__2 = *(float4*)(ol + (i_10 * 4));
      (reinterpret_cast<__maca_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
      (reinterpret_cast<__maca_bfloat162*>(&__3))[1] = __float22bfloat162_rn(((float2*)(&v__2))[1]);
      *(uint2*)(out_local_cast_2 + (i_10 * 4)) = __3;
    }
    *(uint2*)(out + (((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)i0_h) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4))) = *(uint2*)(out_local_cast_2 + 0);
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 4; ++i_11) {
    *(uint2*)(xs_local_cast_3 + 0) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i_11 * 1024) + (((int)threadIdx.x) * 4)) + 4352));
    for (int vec_1 = 0; vec_1 < 1; ++vec_1) {
      float4 __4;
      uint2 v__3 = *(uint2*)(xs_local_cast_3 + (vec_1 * 4));
      ((float2*)(&__4))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__3))[0]);
      ((float2*)(&__4))[1] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__3))[1]);
      *(float4*)(xl + ((i_11 * 4) + (vec_1 * 4))) = __4;
    }
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 1; ++i_12) {
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(ol + (i_12 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
  }
  for (int i_hc_1 = 0; i_hc_1 < 4; ++i_hc_1) {
    float pre_1 = ((float*)buf_dyn_shmem)[i_hc_1];
    #pragma unroll
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      ol[i_13] = (ol[i_13] + (pre_1 * xl[((i_hc_1 * 4) + i_13)]));
    }
  }
  for (int i_14 = 0; i_14 < 1; ++i_14) {
    uint2 __5;
    float4 v__4 = *(float4*)(ol + (i_14 * 4));
    (reinterpret_cast<__maca_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
    (reinterpret_cast<__maca_bfloat162*>(&__5))[1] = __float22bfloat162_rn(((float2*)(&v__4))[1]);
    *(uint2*)(out_local_cast_4 + (i_14 * 4)) = __5;
  }
  *(uint2*)(out + (((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (int64_t)3072)) = *(uint2*)(out_local_cast_4 + 0);
  #pragma unroll
  for (int i_15 = 0; i_15 < 4; ++i_15) {
    *(uint2*)(xs_local_cast_5 + 0) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i_15 * 1024) + (((int)threadIdx.x) * 4)) + 256));
    for (int vec_2 = 0; vec_2 < 1; ++vec_2) {
      float4 __6;
      uint2 v__5 = *(uint2*)(xs_local_cast_5 + (vec_2 * 4));
      ((float2*)(&__6))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__5))[0]);
      ((float2*)(&__6))[1] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__5))[1]);
      *(float4*)(xl + ((i_15 * 4) + (vec_2 * 4))) = __6;
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 1; ++i_16) {
    float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(ol + (i_16 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
  }
  for (int i_hc_2 = 0; i_hc_2 < 4; ++i_hc_2) {
    float pre_2 = ((float*)buf_dyn_shmem)[i_hc_2];
    #pragma unroll
    for (int i_17 = 0; i_17 < 4; ++i_17) {
      ol[i_17] = (ol[i_17] + (pre_2 * xl[((i_hc_2 * 4) + i_17)]));
    }
  }
  for (int i_18 = 0; i_18 < 1; ++i_18) {
    uint2 __7;
    float4 v__6 = *(float4*)(ol + (i_18 * 4));
    (reinterpret_cast<__maca_bfloat162*>(&__7))[0] = __float22bfloat162_rn(((float2*)(&v__6))[0]);
    (reinterpret_cast<__maca_bfloat162*>(&__7))[1] = __float22bfloat162_rn(((float2*)(&v__6))[1]);
    *(uint2*)(out_local_cast_6 + (i_18 * 4)) = __7;
  }
  *(uint2*)(out + (((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (int64_t)4096)) = *(uint2*)(out_local_cast_6 + 0);
#endif
}

}  // namespace mhc_head_h5120
