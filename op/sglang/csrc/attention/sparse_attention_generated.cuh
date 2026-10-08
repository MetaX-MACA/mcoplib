#pragma once

#include <cuda_runtime.h>

#include <common/maca_bfloat16.h>

#include <cstdint>

using bfloat16_t = maca_bfloat16;
using bfloat16x4_vec =
    __attribute__((__vector_size__(4 * sizeof(short)))) short;
using float32x4 =
    __attribute__((__vector_size__(4 * sizeof(float)))) float;

using uchar = unsigned char;

#define MACART_INF_F __int_as_float(0x7f800000)

__device__ __forceinline__ unsigned __pack_maca_bfloat162(
    const bfloat16_t x,
    const bfloat16_t y) {
  const unsigned v0 = *reinterpret_cast<const unsigned short*>(&x);
  const unsigned v1 = *reinterpret_cast<const unsigned short*>(&y);
  return (v1 << 16) | v0;
}

namespace tl {

struct SumOp {
  template <typename T>
  __device__ __forceinline__ T operator()(const T& x, const T& y) const {
    return x + y;
  }
};

struct MaxOp {
  template <typename T>
  __device__ __forceinline__ T operator()(const T& x, const T& y) const {
    return x > y ? x : y;
  }
};

template <class Reducer, int Threads, int Scale, int ThreadOffset = 0>
struct AllReduce {
  static_assert(Threads == 512 || Threads == 256 || Threads == 128 ||
                Threads == 64 || Threads == 32 || Threads == 16 ||
                Threads == 8 || Threads == 4 || Threads == 2);
  static_assert(Threads % Scale == 0);

  template <typename T>
  static __device__ __forceinline__ T run(T value, T* buffer = nullptr) {
    constexpr int kOffset = Threads / 2;
    constexpr int kWarpSize = 64;
    if constexpr (kOffset >= kWarpSize) {
      __syncthreads();
      buffer[threadIdx.x - ThreadOffset] = value;
      __syncthreads();
      value = Reducer()(value, buffer[(threadIdx.x - ThreadOffset) ^ kOffset]);
    } else {
      value = Reducer()(
          value,
          __shfl_xor_sync(uint64_t(-1), value, kOffset));
    }
    if constexpr (kOffset == Scale) {
      return value;
    } else {
      return AllReduce<Reducer, kOffset, Scale, ThreadOffset>::run(
          value, buffer);
    }
  }
};

}  // namespace tl

// ---- Wide decode partial kernel ----
__global__ void __launch_bounds__(256, 1) sparse_decode_partial_generated_kernel(const int* __restrict__ Indices, const bfloat16_t* __restrict__ KV, float* __restrict__ Partial_Lse, bfloat16_t* __restrict__ Partial_O, const bfloat16_t* __restrict__ Q, int seq_len, int seq_len_kv, float sm_scale_log2) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[64];
  float m_i[1];
  bfloat16_t Q_buf[128];
  signed char mask[4];
  float acc_s[4];
  float sumexp_i[1];
  float m_i_clear[1];
  bfloat16_t S_shared_local_cast[4];
  float m_i_clear_1[1];
  bfloat16_t S_shared_local_cast_1[4];
  bfloat16_t Partial_O_local_cast_2[4];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  if (((int)threadIdx.x) < 32) {
    ((float*)buf_dyn_shmem)[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
  }
  m_i[0] = -0x1p+30f/*-1.073742e+09*/;
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    *(uint2*)(Q_buf + (i_1 * 4)) = *(uint2*)(Q + (((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + (((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)6) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + (((int64_t)i_1) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    int idx = Indices[(((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2112) + (((int64_t)((int)blockIdx.y)) * (int64_t)192)) + (((int64_t)i_2) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)6))];
    bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    int condval_1;
    if ((0 <= idx)) {
      condval_1 = idx;
    } else {
      condval_1 = 0;
    }
    int condval_2;
    if ((0 <= idx)) {
      condval_2 = idx;
    } else {
      condval_2 = 0;
    }
    uint4 condval;
    if (((0 <= condval_1) && (condval_2 < seq_len_kv))) {
      int64_t condval_3;
      if (((int64_t)0 <= ((int64_t)idx))) {
        condval_3 = ((int64_t)idx);
      } else {
        condval_3 = (int64_t)0;
      }
      int64_t condval_4;
      if (((int64_t)0 <= ((int64_t)idx))) {
        condval_4 = ((int64_t)idx);
      } else {
        condval_4 = (int64_t)0;
      }
      condval = *(uint4*)(KV + ((condval_4 * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_maca_bfloat162(broadcast_var_1, broadcast_var_1), __pack_maca_bfloat162(broadcast_var_1, broadcast_var_1), __pack_maca_bfloat162(broadcast_var_1, broadcast_var_1), __pack_maca_bfloat162(broadcast_var_1, broadcast_var_1));
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 63) >> 3) * 2048) + (i_2 * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_2 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 64)) = condval;
  }
  for (int k_i = 0; k_i < 5; ++k_i) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      int idx_1 = Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2112) + (((int64_t)((int)blockIdx.y)) * (int64_t)192)) + (((int64_t)k_i) * (int64_t)32)) + (((int64_t)i_3) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)6)) + (int64_t)32)];
      bfloat16_t broadcast_var_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      int condval_6;
      if ((0 <= idx_1)) {
        condval_6 = idx_1;
      } else {
        condval_6 = 0;
      }
      int condval_7;
      if ((0 <= idx_1)) {
        condval_7 = idx_1;
      } else {
        condval_7 = 0;
      }
      uint4 condval_5;
      if (((0 <= condval_6) && (condval_7 < seq_len_kv))) {
        int64_t condval_8;
        if (((int64_t)0 <= ((int64_t)idx_1))) {
          condval_8 = ((int64_t)idx_1);
        } else {
          condval_8 = (int64_t)0;
        }
        int64_t condval_9;
        if (((int64_t)0 <= ((int64_t)idx_1))) {
          condval_9 = ((int64_t)idx_1);
        } else {
          condval_9 = (int64_t)0;
        }
        condval_5 = *(uint4*)(KV + ((condval_9 * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) * (int64_t)8)));
      } else {
        condval_5 = make_uint4(__pack_maca_bfloat162(broadcast_var_2, broadcast_var_2), __pack_maca_bfloat162(broadcast_var_2, broadcast_var_2), __pack_maca_bfloat162(broadcast_var_2, broadcast_var_2), __pack_maca_bfloat162(broadcast_var_2, broadcast_var_2));
      }
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((k_i + 1) & 1) * 16384) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + (i_3 * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 64)) = condval_5;
    }
    int broadcast_var_3 = 0;
    int __1;
    ushort4 __2;
      int4 v_ = make_int4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
      int4 v__1 = *(int4*)(Indices + ((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2112) + (((int64_t)((int)blockIdx.y)) * (int64_t)192)) + (((int64_t)k_i) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4)));
      __2.x = (v_.x<=v__1.x);
      __2.y = (v_.y<=v__1.y);
      __2.z = (v_.z<=v__1.z);
      __2.w = (v_.w<=v__1.w);
    __1=((signed char)(__2.x) << 0);
    __1=__1 & ~(0x000000ff << 8) |((signed char)(__2.y) << 8);
    __1=__1 & ~(0x000000ff << 16) |((signed char)(__2.z) << 16);
    __1=__1 & ~(0x000000ff << 24) |((signed char)(__2.w) << 24);
    *(int*)(mask + 0) = __1;
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      float condval_10;
      if (((bool)mask[i_4])) {
        condval_10 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_10 = -MACART_INF_F;
      }
      acc_s[i_4] = condval_10;
    }
    bfloat16_t B_local[4];
    __syncthreads();
    for (int ki = 0; ki < 32; ++ki) {
      *(uint2*)(B_local + 0) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((k_i & 1) * 16384) + ((ki >> 2) * 2048)) + ((((int)threadIdx.x) >> 7) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 64));
      {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local) + 0),
                    *(((bfloat16x4_vec*)Q_buf) + ki),
                    *(((float32x4*)acc_s) + 0));
    };
    }
    if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
      ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 17984)] = m_i[0];
    }
    m_i_clear[0] = -MACART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      m_i_clear[0] = max(m_i_clear[0], acc_s[rv]);
    }
    __syncthreads();
    m_i_clear[0] = tl::AllReduce<tl::MaxOp, 256, 128, 0>::run(m_i_clear[0], (&(((float*)buf_dyn_shmem)[17696])));
    m_i_clear[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(m_i_clear[0], (&(((float*)buf_dyn_shmem)[17184])));
    m_i[0] = max(m_i[0], m_i_clear[0]);
    __syncthreads();
    if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
      ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 17952)] = exp2f(((((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 17984)] - m_i[0]) * sm_scale_log2));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      acc_s[i_5] = exp2f(((acc_s[i_5] * sm_scale_log2) - (m_i[0] * sm_scale_log2)));
    }
    sumexp_i[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
      sumexp_i[0] = (sumexp_i[0] + acc_s[rv_1]);
    }
    __syncthreads();
    sumexp_i[0] = tl::AllReduce<tl::SumOp, 256, 128, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[16928])));
    sumexp_i[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[17440])));
    __syncthreads();
    if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
      ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] = ((((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] * ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 17952)]) + sumexp_i[0]);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      for (int vec = 0; vec < 8; ++vec) {
        float4 __3;
          float4 v__2 = *(float4*)(acc_o + ((i_6 * 32) + (vec * 4)));
          float4 v__3 = make_float4(((float*)buf_dyn_shmem)[(((i_6 * 16) + (((int)threadIdx.x) & 15)) + 17952)], ((float*)buf_dyn_shmem)[(((i_6 * 16) + (((int)threadIdx.x) & 15)) + 17952)], ((float*)buf_dyn_shmem)[(((i_6 * 16) + (((int)threadIdx.x) & 15)) + 17952)], ((float*)buf_dyn_shmem)[(((i_6 * 16) + (((int)threadIdx.x) & 15)) + 17952)]);
          __3.x = (v__2.x*v__3.x);
          __3.y = (v__2.y*v__3.y);
          __3.z = (v__2.z*v__3.z);
          __3.w = (v__2.w*v__3.w);
        *(float4*)(acc_o + ((i_6 * 32) + (vec * 4))) = __3;
      }
    }
    uint2 __4;
    float4 v__4 = *(float4*)(acc_s + 0);
    (reinterpret_cast<__maca_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
    (reinterpret_cast<__maca_bfloat162*>(&__4))[1] = __float22bfloat162_rn(((float2*)(&v__4))[1]);
    *(uint2*)(S_shared_local_cast + 0) = __4;
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 127) >> 6) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 32832)) = *(uint2*)(S_shared_local_cast + 0);
    bfloat16_t A_local[8];
    bfloat16_t B_local_1[32];
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        *(uint2*)(A_local + (i_7 * 4)) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_7 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 32832));
      }
      for (int j = 0; j < 8; ++j) {
        for (int local_id = 0; local_id < 4; ++local_id) {
          B_local_1[((j * 4) + local_id)] = ((bfloat16_t*)buf_dyn_shmem)[((((((((((((k_i & 1) * 16384) + ((((int)threadIdx.x) >> 6) * 4096)) + ((j >> 2) * 2048)) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 63) >> 4) * 256)) + (local_id * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((j & 3) >> 1)) & 1) * 32)) + ((((local_id >> 1) + (j & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 64)];
        }
      }
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          {
      *(((float32x4*)acc_o) + ((i_8 * 8) + j_1)) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local_1) + j_1),
                    *(((bfloat16x4_vec*)A_local) + i_8),
                    *(((float32x4*)acc_o) + ((i_8 * 8) + j_1)));
    };
        }
      }
    }
  }
  int broadcast_var_4 = 0;
  int __5;
  ushort4 __6;
    int4 v__5 = make_int4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    int4 v__6 = *(int4*)(Indices + ((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2112) + (((int64_t)((int)blockIdx.y)) * (int64_t)192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4)) + (int64_t)160));
    __6.x = (v__5.x<=v__6.x);
    __6.y = (v__5.y<=v__6.y);
    __6.z = (v__5.z<=v__6.z);
    __6.w = (v__5.w<=v__6.w);
  __5=((signed char)(__6.x) << 0);
  __5=__5 & ~(0x000000ff << 8) |((signed char)(__6.y) << 8);
  __5=__5 & ~(0x000000ff << 16) |((signed char)(__6.z) << 16);
  __5=__5 & ~(0x000000ff << 24) |((signed char)(__6.w) << 24);
  *(int*)(mask + 0) = __5;
  #pragma unroll
  for (int i_9 = 0; i_9 < 4; ++i_9) {
    float condval_11;
    if (((bool)mask[i_9])) {
      condval_11 = 0x0p+0f/*0.000000e+00*/;
    } else {
      condval_11 = -MACART_INF_F;
    }
    acc_s[i_9] = condval_11;
  }
  bfloat16_t B_local_2[4];
  for (int ki_2 = 0; ki_2 < 32; ++ki_2) {
    *(uint2*)(B_local_2 + 0) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((ki_2 >> 2) * 2048) + ((((int)threadIdx.x) >> 7) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 16448));
    {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local_2) + 0),
                    *(((bfloat16x4_vec*)Q_buf) + ki_2),
                    *(((float32x4*)acc_s) + 0));
    };
  }
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
    ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 17984)] = m_i[0];
  }
  m_i_clear_1[0] = -MACART_INF_F;
  #pragma unroll
  for (int rv_2 = 0; rv_2 < 4; ++rv_2) {
    m_i_clear_1[0] = max(m_i_clear_1[0], acc_s[rv_2]);
  }
  __syncthreads();
  m_i_clear_1[0] = tl::AllReduce<tl::MaxOp, 256, 128, 0>::run(m_i_clear_1[0], (&(((float*)buf_dyn_shmem)[16928])));
  __syncthreads();
  m_i_clear_1[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(m_i_clear_1[0], (&(((float*)buf_dyn_shmem)[16928])));
  m_i[0] = max(m_i[0], m_i_clear_1[0]);
  __syncthreads();
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
    ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 17952)] = exp2f(((((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 17984)] - m_i[0]) * sm_scale_log2));
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 4; ++i_10) {
    acc_s[i_10] = exp2f(((acc_s[i_10] * sm_scale_log2) - (m_i[0] * sm_scale_log2)));
  }
  sumexp_i[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_3 = 0; rv_3 < 4; ++rv_3) {
    sumexp_i[0] = (sumexp_i[0] + acc_s[rv_3]);
  }
  __syncthreads();
  sumexp_i[0] = tl::AllReduce<tl::SumOp, 256, 128, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[16928])));
  __syncthreads();
  sumexp_i[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[16928])));
  __syncthreads();
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
    ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] = ((((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] * ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 17952)]) + sumexp_i[0]);
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 2; ++i_11) {
    for (int vec_1 = 0; vec_1 < 8; ++vec_1) {
      float4 __7;
        float4 v__7 = *(float4*)(acc_o + ((i_11 * 32) + (vec_1 * 4)));
        float4 v__8 = make_float4(((float*)buf_dyn_shmem)[(((i_11 * 16) + (((int)threadIdx.x) & 15)) + 17952)], ((float*)buf_dyn_shmem)[(((i_11 * 16) + (((int)threadIdx.x) & 15)) + 17952)], ((float*)buf_dyn_shmem)[(((i_11 * 16) + (((int)threadIdx.x) & 15)) + 17952)], ((float*)buf_dyn_shmem)[(((i_11 * 16) + (((int)threadIdx.x) & 15)) + 17952)]);
        __7.x = (v__7.x*v__8.x);
        __7.y = (v__7.y*v__8.y);
        __7.z = (v__7.z*v__8.z);
        __7.w = (v__7.w*v__8.w);
      *(float4*)(acc_o + ((i_11 * 32) + (vec_1 * 4))) = __7;
    }
  }
  uint2 __8;
  float4 v__9 = *(float4*)(acc_s + 0);
  (reinterpret_cast<__maca_bfloat162*>(&__8))[0] = __float22bfloat162_rn(((float2*)(&v__9))[0]);
  (reinterpret_cast<__maca_bfloat162*>(&__8))[1] = __float22bfloat162_rn(((float2*)(&v__9))[1]);
  *(uint2*)(S_shared_local_cast_1 + 0) = __8;
  *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 127) >> 6) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 32832)) = *(uint2*)(S_shared_local_cast_1 + 0);
  bfloat16_t A_local_1[8];
  bfloat16_t B_local_3[32];
  __syncthreads();
  for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      *(uint2*)(A_local_1 + (i_12 * 4)) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_12 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_3) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 32832));
    }
    for (int j_2 = 0; j_2 < 8; ++j_2) {
      for (int local_id_1 = 0; local_id_1 < 4; ++local_id_1) {
        B_local_3[((j_2 * 4) + local_id_1)] = ((bfloat16_t*)buf_dyn_shmem)[(((((((((((((int)threadIdx.x) >> 6) * 4096) + ((j_2 >> 2) * 2048)) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 63) >> 4) * 256)) + (local_id_1 * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((j_2 & 3) >> 1)) & 1) * 32)) + ((((local_id_1 >> 1) + (j_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id_1 & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 16448)];
      }
    }
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        {
      *(((float32x4*)acc_o) + ((i_13 * 8) + j_3)) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local_3) + j_3),
                    *(((bfloat16x4_vec*)A_local_1) + i_13),
                    *(((float32x4*)acc_o) + ((i_13 * 8) + j_3)));
    };
      }
    }
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 2; ++i_14) {
    for (int vec_2 = 0; vec_2 < 8; ++vec_2) {
      float4 __9;
        float4 v__10 = *(float4*)(acc_o + ((i_14 * 32) + (vec_2 * 4)));
        float condval_12;
        if ((((float*)buf_dyn_shmem)[((i_14 * 16) + (((int)threadIdx.x) & 15))] == 0x0p+0f/*0.000000e+00*/)) {
          condval_12 = 0x1p+0f/*1.000000e+00*/;
        } else {
          condval_12 = ((float*)buf_dyn_shmem)[((i_14 * 16) + (((int)threadIdx.x) & 15))];
        }
        float4 v__11 = make_float4(condval_12, condval_12, condval_12, condval_12);
        __9.x = (v__10.x/v__11.x);
        __9.y = (v__10.y/v__11.y);
        __9.z = (v__10.z/v__11.z);
        __9.w = (v__10.w/v__11.w);
      *(float4*)(acc_o + ((i_14 * 32) + (vec_2 * 4))) = __9;
    }
  }
  __syncthreads();
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
    float condval_13;
    if ((((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] == 0x0p+0f/*0.000000e+00*/)) {
      condval_13 = -0x1p+30f/*-1.073742e+09*/;
    } else {
      condval_13 = (log2f(((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))]) + (m_i[0] * sm_scale_log2));
    }
    ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] = condval_13;
  }
  #pragma unroll
  for (int i_15 = 0; i_15 < 16; ++i_15) {
    uint2 __10;
    float4 v__12 = *(float4*)(acc_o + (i_15 * 4));
    (reinterpret_cast<__maca_bfloat162*>(&__10))[0] = __float22bfloat162_rn(((float2*)(&v__12))[0]);
    (reinterpret_cast<__maca_bfloat162*>(&__10))[1] = __float22bfloat162_rn(((float2*)(&v__12))[1]);
    *(uint2*)(Partial_O_local_cast_2 + 0) = __10;
    *(uint2*)(Partial_O + (((((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)360448) + (((int64_t)((int)blockIdx.y)) * (int64_t)32768)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)16384)) + ((((int64_t)i_15) >> (int64_t)3) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)128)) + ((((int64_t)i_15) & (int64_t)7) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4))) = *(uint2*)(Partial_O_local_cast_2 + 0);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    Partial_Lse[(((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)704) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)32)) + ((int64_t)((int)threadIdx.x)))] = ((float*)buf_dyn_shmem)[((int)threadIdx.x)];
  }
}


// ---- Stage-1 decode partial kernel ----
__global__ void __launch_bounds__(256, 1) sparse_decode_partial_stage1_generated_kernel(const int* __restrict__ Indices, const bfloat16_t* __restrict__ KV, float* __restrict__ Partial_Lse, bfloat16_t* __restrict__ Partial_O, const bfloat16_t* __restrict__ Q, int seq_len, int seq_len_kv, float sm_scale_log2) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[64];
  float m_i[1];
  bfloat16_t Q_buf[128];
  signed char mask[4];
  float acc_s[4];
  float sumexp_i[1];
  float m_i_clear[1];
  bfloat16_t S_shared_local_cast[4];
  float m_i_clear_1[1];
  bfloat16_t S_shared_local_cast_1[4];
  bfloat16_t Partial_O_local_cast_2[4];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  if (((int)threadIdx.x) < 32) {
    ((float*)buf_dyn_shmem)[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
  }
  m_i[0] = -0x1p+30f/*-1.073742e+09*/;
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    *(uint2*)(Q_buf + (i_1 * 4)) = *(uint2*)(Q + (((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + (((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)6) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + (((int64_t)i_1) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    int idx = Indices[(((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2112) + (((int64_t)((int)blockIdx.y)) * (int64_t)192)) + (((int64_t)i_2) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)6))];
    bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    int condval_1;
    if ((0 <= idx)) {
      condval_1 = idx;
    } else {
      condval_1 = 0;
    }
    int condval_2;
    if ((0 <= idx)) {
      condval_2 = idx;
    } else {
      condval_2 = 0;
    }
    uint4 condval;
    if (((0 <= condval_1) && (condval_2 < seq_len_kv))) {
      int64_t condval_3;
      if (((int64_t)0 <= ((int64_t)idx))) {
        condval_3 = ((int64_t)idx);
      } else {
        condval_3 = (int64_t)0;
      }
      int64_t condval_4;
      if (((int64_t)0 <= ((int64_t)idx))) {
        condval_4 = ((int64_t)idx);
      } else {
        condval_4 = (int64_t)0;
      }
      condval = *(uint4*)(KV + ((condval_4 * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_maca_bfloat162(broadcast_var_1, broadcast_var_1), __pack_maca_bfloat162(broadcast_var_1, broadcast_var_1), __pack_maca_bfloat162(broadcast_var_1, broadcast_var_1), __pack_maca_bfloat162(broadcast_var_1, broadcast_var_1));
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 63) >> 3) * 2048) + (i_2 * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_2 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 64)) = condval;
  }
  for (int k_i = 0; k_i < 5; ++k_i) {
    int broadcast_var_2 = 0;
    int __1;
    ushort4 __2;
      int4 v_ = make_int4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
      int4 v__1 = *(int4*)(Indices + ((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2112) + (((int64_t)((int)blockIdx.y)) * (int64_t)192)) + (((int64_t)k_i) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4)));
      __2.x = (v_.x<=v__1.x);
      __2.y = (v_.y<=v__1.y);
      __2.z = (v_.z<=v__1.z);
      __2.w = (v_.w<=v__1.w);
    __1=((signed char)(__2.x) << 0);
    __1=__1 & ~(0x000000ff << 8) |((signed char)(__2.y) << 8);
    __1=__1 & ~(0x000000ff << 16) |((signed char)(__2.z) << 16);
    __1=__1 & ~(0x000000ff << 24) |((signed char)(__2.w) << 24);
    *(int*)(mask + 0) = __1;
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      float condval_5;
      if (((bool)mask[i_3])) {
        condval_5 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_5 = -MACART_INF_F;
      }
      acc_s[i_3] = condval_5;
    }
    bfloat16_t B_local[4];
    __syncthreads();
    for (int ki = 0; ki < 32; ++ki) {
      *(uint2*)(B_local + 0) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((ki >> 2) * 2048) + ((((int)threadIdx.x) >> 7) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 64));
      {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local) + 0),
                    *(((bfloat16x4_vec*)Q_buf) + ki),
                    *(((float32x4*)acc_s) + 0));
    };
    }
    if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
      ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9792)] = m_i[0];
    }
    m_i_clear[0] = -MACART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      m_i_clear[0] = max(m_i_clear[0], acc_s[rv]);
    }
    __syncthreads();
    m_i_clear[0] = tl::AllReduce<tl::MaxOp, 256, 128, 0>::run(m_i_clear[0], (&(((float*)buf_dyn_shmem)[9504])));
    m_i_clear[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(m_i_clear[0], (&(((float*)buf_dyn_shmem)[9248])));
    m_i[0] = max(m_i[0], m_i_clear[0]);
    __syncthreads();
    if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
      ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9760)] = exp2f(((((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9792)] - m_i[0]) * sm_scale_log2));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      acc_s[i_4] = exp2f(((acc_s[i_4] * sm_scale_log2) - (m_i[0] * sm_scale_log2)));
    }
    sumexp_i[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
      sumexp_i[0] = (sumexp_i[0] + acc_s[rv_1]);
    }
    __syncthreads();
    sumexp_i[0] = tl::AllReduce<tl::SumOp, 256, 128, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[8992])));
    sumexp_i[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[8736])));
    __syncthreads();
    if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
      ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] = ((((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] * ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9760)]) + sumexp_i[0]);
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      for (int vec = 0; vec < 8; ++vec) {
        float4 __3;
          float4 v__2 = *(float4*)(acc_o + ((i_5 * 32) + (vec * 4)));
          float4 v__3 = make_float4(((float*)buf_dyn_shmem)[(((i_5 * 16) + (((int)threadIdx.x) & 15)) + 9760)], ((float*)buf_dyn_shmem)[(((i_5 * 16) + (((int)threadIdx.x) & 15)) + 9760)], ((float*)buf_dyn_shmem)[(((i_5 * 16) + (((int)threadIdx.x) & 15)) + 9760)], ((float*)buf_dyn_shmem)[(((i_5 * 16) + (((int)threadIdx.x) & 15)) + 9760)]);
          __3.x = (v__2.x*v__3.x);
          __3.y = (v__2.y*v__3.y);
          __3.z = (v__2.z*v__3.z);
          __3.w = (v__2.w*v__3.w);
        *(float4*)(acc_o + ((i_5 * 32) + (vec * 4))) = __3;
      }
    }
    uint2 __4;
    float4 v__4 = *(float4*)(acc_s + 0);
    (reinterpret_cast<__maca_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
    (reinterpret_cast<__maca_bfloat162*>(&__4))[1] = __float22bfloat162_rn(((float2*)(&v__4))[1]);
    *(uint2*)(S_shared_local_cast + 0) = __4;
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 127) >> 6) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 16448)) = *(uint2*)(S_shared_local_cast + 0);
    bfloat16_t A_local[8];
    bfloat16_t B_local_1[32];
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        *(uint2*)(A_local + (i_6 * 4)) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_6 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 16448));
      }
      for (int j = 0; j < 8; ++j) {
        for (int local_id = 0; local_id < 4; ++local_id) {
          B_local_1[((j * 4) + local_id)] = ((bfloat16_t*)buf_dyn_shmem)[(((((((((((((int)threadIdx.x) >> 6) * 4096) + ((j >> 2) * 2048)) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 63) >> 4) * 256)) + (local_id * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((j & 3) >> 1)) & 1) * 32)) + ((((local_id >> 1) + (j & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 64)];
        }
      }
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          {
      *(((float32x4*)acc_o) + ((i_7 * 8) + j_1)) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local_1) + j_1),
                    *(((bfloat16x4_vec*)A_local) + i_7),
                    *(((float32x4*)acc_o) + ((i_7 * 8) + j_1)));
    };
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; ++i_8) {
      int idx_1 = Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2112) + (((int64_t)((int)blockIdx.y)) * (int64_t)192)) + (((int64_t)k_i) * (int64_t)32)) + (((int64_t)i_8) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)6)) + (int64_t)32)];
      bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      int condval_7;
      if ((0 <= idx_1)) {
        condval_7 = idx_1;
      } else {
        condval_7 = 0;
      }
      int condval_8;
      if ((0 <= idx_1)) {
        condval_8 = idx_1;
      } else {
        condval_8 = 0;
      }
      uint4 condval_6;
      if (((0 <= condval_7) && (condval_8 < seq_len_kv))) {
        int64_t condval_9;
        if (((int64_t)0 <= ((int64_t)idx_1))) {
          condval_9 = ((int64_t)idx_1);
        } else {
          condval_9 = (int64_t)0;
        }
        int64_t condval_10;
        if (((int64_t)0 <= ((int64_t)idx_1))) {
          condval_10 = ((int64_t)idx_1);
        } else {
          condval_10 = (int64_t)0;
        }
        condval_6 = *(uint4*)(KV + ((condval_10 * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) * (int64_t)8)));
      } else {
        condval_6 = make_uint4(__pack_maca_bfloat162(broadcast_var_3, broadcast_var_3), __pack_maca_bfloat162(broadcast_var_3, broadcast_var_3), __pack_maca_bfloat162(broadcast_var_3, broadcast_var_3), __pack_maca_bfloat162(broadcast_var_3, broadcast_var_3));
      }
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 63) >> 3) * 2048) + (i_8 * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_8 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 64)) = condval_6;
    }
  }
  int broadcast_var_4 = 0;
  int __5;
  ushort4 __6;
    int4 v__5 = make_int4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    int4 v__6 = *(int4*)(Indices + ((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2112) + (((int64_t)((int)blockIdx.y)) * (int64_t)192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4)) + (int64_t)160));
    __6.x = (v__5.x<=v__6.x);
    __6.y = (v__5.y<=v__6.y);
    __6.z = (v__5.z<=v__6.z);
    __6.w = (v__5.w<=v__6.w);
  __5=((signed char)(__6.x) << 0);
  __5=__5 & ~(0x000000ff << 8) |((signed char)(__6.y) << 8);
  __5=__5 & ~(0x000000ff << 16) |((signed char)(__6.z) << 16);
  __5=__5 & ~(0x000000ff << 24) |((signed char)(__6.w) << 24);
  *(int*)(mask + 0) = __5;
  #pragma unroll
  for (int i_9 = 0; i_9 < 4; ++i_9) {
    float condval_11;
    if (((bool)mask[i_9])) {
      condval_11 = 0x0p+0f/*0.000000e+00*/;
    } else {
      condval_11 = -MACART_INF_F;
    }
    acc_s[i_9] = condval_11;
  }
  bfloat16_t B_local_2[4];
  __syncthreads();
  for (int ki_2 = 0; ki_2 < 32; ++ki_2) {
    *(uint2*)(B_local_2 + 0) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((ki_2 >> 2) * 2048) + ((((int)threadIdx.x) >> 7) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 64));
    {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local_2) + 0),
                    *(((bfloat16x4_vec*)Q_buf) + ki_2),
                    *(((float32x4*)acc_s) + 0));
    };
  }
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
    ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9792)] = m_i[0];
  }
  m_i_clear_1[0] = -MACART_INF_F;
  #pragma unroll
  for (int rv_2 = 0; rv_2 < 4; ++rv_2) {
    m_i_clear_1[0] = max(m_i_clear_1[0], acc_s[rv_2]);
  }
  __syncthreads();
  m_i_clear_1[0] = tl::AllReduce<tl::MaxOp, 256, 128, 0>::run(m_i_clear_1[0], (&(((float*)buf_dyn_shmem)[8736])));
  __syncthreads();
  m_i_clear_1[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(m_i_clear_1[0], (&(((float*)buf_dyn_shmem)[8736])));
  m_i[0] = max(m_i[0], m_i_clear_1[0]);
  __syncthreads();
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
    ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9760)] = exp2f(((((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9792)] - m_i[0]) * sm_scale_log2));
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 4; ++i_10) {
    acc_s[i_10] = exp2f(((acc_s[i_10] * sm_scale_log2) - (m_i[0] * sm_scale_log2)));
  }
  sumexp_i[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_3 = 0; rv_3 < 4; ++rv_3) {
    sumexp_i[0] = (sumexp_i[0] + acc_s[rv_3]);
  }
  __syncthreads();
  sumexp_i[0] = tl::AllReduce<tl::SumOp, 256, 128, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[8736])));
  __syncthreads();
  sumexp_i[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[8736])));
  __syncthreads();
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
    ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] = ((((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] * ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9760)]) + sumexp_i[0]);
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 2; ++i_11) {
    for (int vec_1 = 0; vec_1 < 8; ++vec_1) {
      float4 __7;
        float4 v__7 = *(float4*)(acc_o + ((i_11 * 32) + (vec_1 * 4)));
        float4 v__8 = make_float4(((float*)buf_dyn_shmem)[(((i_11 * 16) + (((int)threadIdx.x) & 15)) + 9760)], ((float*)buf_dyn_shmem)[(((i_11 * 16) + (((int)threadIdx.x) & 15)) + 9760)], ((float*)buf_dyn_shmem)[(((i_11 * 16) + (((int)threadIdx.x) & 15)) + 9760)], ((float*)buf_dyn_shmem)[(((i_11 * 16) + (((int)threadIdx.x) & 15)) + 9760)]);
        __7.x = (v__7.x*v__8.x);
        __7.y = (v__7.y*v__8.y);
        __7.z = (v__7.z*v__8.z);
        __7.w = (v__7.w*v__8.w);
      *(float4*)(acc_o + ((i_11 * 32) + (vec_1 * 4))) = __7;
    }
  }
  uint2 __8;
  float4 v__9 = *(float4*)(acc_s + 0);
  (reinterpret_cast<__maca_bfloat162*>(&__8))[0] = __float22bfloat162_rn(((float2*)(&v__9))[0]);
  (reinterpret_cast<__maca_bfloat162*>(&__8))[1] = __float22bfloat162_rn(((float2*)(&v__9))[1]);
  *(uint2*)(S_shared_local_cast_1 + 0) = __8;
  *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 127) >> 6) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 16448)) = *(uint2*)(S_shared_local_cast_1 + 0);
  bfloat16_t A_local_1[8];
  bfloat16_t B_local_3[32];
  __syncthreads();
  for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      *(uint2*)(A_local_1 + (i_12 * 4)) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_12 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_3) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 16448));
    }
    for (int j_2 = 0; j_2 < 8; ++j_2) {
      for (int local_id_1 = 0; local_id_1 < 4; ++local_id_1) {
        B_local_3[((j_2 * 4) + local_id_1)] = ((bfloat16_t*)buf_dyn_shmem)[(((((((((((((int)threadIdx.x) >> 6) * 4096) + ((j_2 >> 2) * 2048)) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 63) >> 4) * 256)) + (local_id_1 * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((j_2 & 3) >> 1)) & 1) * 32)) + ((((local_id_1 >> 1) + (j_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id_1 & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 64)];
      }
    }
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        {
      *(((float32x4*)acc_o) + ((i_13 * 8) + j_3)) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local_3) + j_3),
                    *(((bfloat16x4_vec*)A_local_1) + i_13),
                    *(((float32x4*)acc_o) + ((i_13 * 8) + j_3)));
    };
      }
    }
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 2; ++i_14) {
    for (int vec_2 = 0; vec_2 < 8; ++vec_2) {
      float4 __9;
        float4 v__10 = *(float4*)(acc_o + ((i_14 * 32) + (vec_2 * 4)));
        float condval_12;
        if ((((float*)buf_dyn_shmem)[((i_14 * 16) + (((int)threadIdx.x) & 15))] == 0x0p+0f/*0.000000e+00*/)) {
          condval_12 = 0x1p+0f/*1.000000e+00*/;
        } else {
          condval_12 = ((float*)buf_dyn_shmem)[((i_14 * 16) + (((int)threadIdx.x) & 15))];
        }
        float4 v__11 = make_float4(condval_12, condval_12, condval_12, condval_12);
        __9.x = (v__10.x/v__11.x);
        __9.y = (v__10.y/v__11.y);
        __9.z = (v__10.z/v__11.z);
        __9.w = (v__10.w/v__11.w);
      *(float4*)(acc_o + ((i_14 * 32) + (vec_2 * 4))) = __9;
    }
  }
  __syncthreads();
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
    float condval_13;
    if ((((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] == 0x0p+0f/*0.000000e+00*/)) {
      condval_13 = -0x1p+30f/*-1.073742e+09*/;
    } else {
      condval_13 = (log2f(((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))]) + (m_i[0] * sm_scale_log2));
    }
    ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] = condval_13;
  }
  #pragma unroll
  for (int i_15 = 0; i_15 < 16; ++i_15) {
    uint2 __10;
    float4 v__12 = *(float4*)(acc_o + (i_15 * 4));
    (reinterpret_cast<__maca_bfloat162*>(&__10))[0] = __float22bfloat162_rn(((float2*)(&v__12))[0]);
    (reinterpret_cast<__maca_bfloat162*>(&__10))[1] = __float22bfloat162_rn(((float2*)(&v__12))[1]);
    *(uint2*)(Partial_O_local_cast_2 + 0) = __10;
    *(uint2*)(Partial_O + (((((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)360448) + (((int64_t)((int)blockIdx.y)) * (int64_t)32768)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)16384)) + ((((int64_t)i_15) >> (int64_t)3) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)128)) + ((((int64_t)i_15) & (int64_t)7) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4))) = *(uint2*)(Partial_O_local_cast_2 + 0);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    Partial_Lse[(((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)704) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)32)) + ((int64_t)((int)threadIdx.x)))] = ((float*)buf_dyn_shmem)[((int)threadIdx.x)];
  }
}


// ---- Decode combine kernel ----
__global__ void __launch_bounds__(256, 1) sparse_decode_combine_generated_kernel(bfloat16_t* __restrict__ Output, const float* __restrict__ Partial_Lse, const bfloat16_t* __restrict__ Partial_O, int seq_len) {
  extern __shared__ __align__(1024) float shared_lse[];
  float lse_max[4];
  float lse_sum[4];
  float scale[44];
  float acc_o[32];
  bfloat16_t Partial_O_local_cast[8];
  bfloat16_t Output_local_cast_1[8];
  if (((int)threadIdx.x) < 16) {
    for (int k = 0; k < 11; ++k) {
      shared_lse[((k * 16) + ((int)threadIdx.x))] = Partial_Lse[(((((((int64_t)((int)blockIdx.x)) >> (int64_t)2) * (int64_t)704) + (((int64_t)k) * (int64_t)64)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)16)) + ((int64_t)((int)threadIdx.x)))];
    }
  }
  float broadcast_var = -0x1p+30f/*-1.073742e+09*/;
  *(float4*)(lse_max + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  __syncthreads();
  for (int k_1 = 0; k_1 < 11; ++k_1) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      lse_max[i] = max(lse_max[i], shared_lse[(((k_1 * 16) + (i * 4)) + (((int)threadIdx.x) >> 6))]);
    }
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(lse_sum + 0) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  for (int k_2 = 0; k_2 < 11; ++k_2) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      lse_sum[i_1] = (lse_sum[i_1] + exp2f((shared_lse[(((k_2 * 16) + (i_1 * 4)) + (((int)threadIdx.x) >> 6))] - lse_max[i_1])));
    }
  }
  for (int k_3 = 0; k_3 < 11; ++k_3) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      scale[((i_2 * 11) + k_3)] = exp2f(((shared_lse[(((k_3 * 16) + (i_2 * 4)) + (((int)threadIdx.x) >> 6))] - lse_max[i_2]) - log2f(lse_sum[i_2])));
    }
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i_3 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
  }
  for (int k_4 = 0; k_4 < 11; ++k_4) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      *(uint4*)(Partial_O_local_cast + 0) = *(uint4*)(Partial_O + ((((((((int64_t)((int)blockIdx.x)) >> (int64_t)2) * (int64_t)360448) + (((int64_t)k_4) * (int64_t)32768)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)3) * (int64_t)8192)) + (((int64_t)i_4) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
      for (int vec = 0; vec < 2; ++vec) {
        float4 __1;
          float4 v_ = *(float4*)(acc_o + ((i_4 * 8) + (vec * 4)));
          float4 __2;
            float4 v__1 = make_float4(scale[((i_4 * 11) + k_4)], scale[((i_4 * 11) + k_4)], scale[((i_4 * 11) + k_4)], scale[((i_4 * 11) + k_4)]);
            float4 __3;
            uint2 v__2 = *(uint2*)(Partial_O_local_cast + (vec * 4));
            ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__2))[0]);
            ((float2*)(&__3))[1] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__2))[1]);
            __2.x = (v__1.x*__3.x);
            __2.y = (v__1.y*__3.y);
            __2.z = (v__1.z*__3.z);
            __2.w = (v__1.w*__3.w);
          __1.x = (v_.x+__2.x);
          __1.y = (v_.y+__2.y);
          __1.z = (v_.z+__2.z);
          __1.w = (v_.w+__2.w);
        *(float4*)(acc_o + ((i_4 * 8) + (vec * 4))) = __1;
      }
    }
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    for (int vec_1 = 0; vec_1 < 2; ++vec_1) {
      uint2 __4;
      float4 v__3 = *(float4*)(acc_o + ((i_5 * 8) + (vec_1 * 4)));
      (reinterpret_cast<__maca_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__3))[0]);
      (reinterpret_cast<__maca_bfloat162*>(&__4))[1] = __float22bfloat162_rn(((float2*)(&v__3))[1]);
      *(uint2*)(Output_local_cast_1 + (vec_1 * 4)) = __4;
    }
    *(uint4*)(Output + (((((int64_t)((int)blockIdx.x)) * (int64_t)8192) + (((int64_t)i_5) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(Output_local_cast_1 + 0);
  }
}


// ---- Wide prefill kernel ----
__global__ void __launch_bounds__(512, 1) sparse_prefill_generated_kernel(const int* __restrict__ Indices, const bfloat16_t* __restrict__ KV, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ Q, int seq_len, int seq_len_kv, float sm_scale_log2) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[64];
  float m_i[1];
  bfloat16_t Q_buf[128];
  __shared__ signed char mask[64];
  float acc_s[4];
  float sumexp_i[1];
  float m_i_clear[1];
  bfloat16_t S_shared_local_cast[4];
  float m_i_clear_1[1];
  bfloat16_t S_shared_local_cast_1[4];
  float m_i_clear_2[1];
  bfloat16_t S_shared_local_cast_2[4];
  bfloat16_t Output_local_cast_3[4];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  if (((int)threadIdx.x) < 64) {
    ((float*)buf_dyn_shmem)[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
  }
  m_i[0] = -0x1p+30f/*-1.073742e+09*/;
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    *(uint2*)(Q_buf + (i_1 * 4)) = *(uint2*)(Q + (((((((int64_t)((int)blockIdx.x)) * (int64_t)32768) + (((((int64_t)((int)threadIdx.x)) & (int64_t)255) >> (int64_t)6) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + (((int64_t)i_1) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4)));
  }
  if (((int)threadIdx.x) < 32) {
    int idx = Indices[((((int64_t)((int)blockIdx.x)) * (int64_t)2112) + ((int64_t)((int)threadIdx.x)))];
    ((int*)buf_dyn_shmem)[(((int)threadIdx.x) + 64)] = idx;
    mask[((int)threadIdx.x)] = ((signed char)(0 <= idx));
  }
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    int idx_1 = ((int*)buf_dyn_shmem)[(((i_2 * 8) + (((int)threadIdx.x) >> 6)) + 64)];
    bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    int condval_1;
    if ((0 <= idx_1)) {
      condval_1 = idx_1;
    } else {
      condval_1 = 0;
    }
    int condval_2;
    if ((0 <= idx_1)) {
      condval_2 = idx_1;
    } else {
      condval_2 = 0;
    }
    uint4 condval;
    if (((0 <= condval_1) && (condval_2 < seq_len_kv))) {
      int64_t condval_3;
      if (((int64_t)0 <= ((int64_t)idx_1))) {
        condval_3 = ((int64_t)idx_1);
      } else {
        condval_3 = (int64_t)0;
      }
      int64_t condval_4;
      if (((int64_t)0 <= ((int64_t)idx_1))) {
        condval_4 = ((int64_t)idx_1);
      } else {
        condval_4 = (int64_t)0;
      }
      condval = *(uint4*)(KV + ((condval_4 * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_maca_bfloat162(broadcast_var_1, broadcast_var_1), __pack_maca_bfloat162(broadcast_var_1, broadcast_var_1), __pack_maca_bfloat162(broadcast_var_1, broadcast_var_1), __pack_maca_bfloat162(broadcast_var_1, broadcast_var_1));
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 63) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 6) * 64)) + ((((((int)threadIdx.x) >> 8) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 255) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 192)) = condval;
  }
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    int idx_2 = Indices[(((((int64_t)((int)blockIdx.x)) * (int64_t)2112) + ((int64_t)((int)threadIdx.x))) + (int64_t)32)];
    ((int*)buf_dyn_shmem)[(((int)threadIdx.x) + 64)] = idx_2;
    mask[(((int)threadIdx.x) + 32)] = ((signed char)(0 <= idx_2));
  }
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    int idx_3 = ((int*)buf_dyn_shmem)[(((i_3 * 8) + (((int)threadIdx.x) >> 6)) + 64)];
    bfloat16_t broadcast_var_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    int condval_6;
    if ((0 <= idx_3)) {
      condval_6 = idx_3;
    } else {
      condval_6 = 0;
    }
    int condval_7;
    if ((0 <= idx_3)) {
      condval_7 = idx_3;
    } else {
      condval_7 = 0;
    }
    uint4 condval_5;
    if (((0 <= condval_6) && (condval_7 < seq_len_kv))) {
      int64_t condval_8;
      if (((int64_t)0 <= ((int64_t)idx_3))) {
        condval_8 = ((int64_t)idx_3);
      } else {
        condval_8 = (int64_t)0;
      }
      int64_t condval_9;
      if (((int64_t)0 <= ((int64_t)idx_3))) {
        condval_9 = ((int64_t)idx_3);
      } else {
        condval_9 = (int64_t)0;
      }
      condval_5 = *(uint4*)(KV + ((condval_9 * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) * (int64_t)8)));
    } else {
      condval_5 = make_uint4(__pack_maca_bfloat162(broadcast_var_2, broadcast_var_2), __pack_maca_bfloat162(broadcast_var_2, broadcast_var_2), __pack_maca_bfloat162(broadcast_var_2, broadcast_var_2), __pack_maca_bfloat162(broadcast_var_2, broadcast_var_2));
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 63) >> 3) * 2048) + (i_3 * 512)) + ((((int)threadIdx.x) >> 6) * 64)) + ((((((int)threadIdx.x) >> 8) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 255) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16576)) = condval_5;
  }
  __syncthreads();
  for (int k_i = 0; k_i < 64; ++k_i) {
    for (int i_s = 0; i_s < 4; ++i_s) {
      float condval_10;
      if (((bool)mask[(((((k_i & 1) * 32) + ((((int)threadIdx.x) >> 8) * 16)) + (((((int)threadIdx.x) & 63) >> 4) * 4)) + i_s)])) {
        condval_10 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_10 = -MACART_INF_F;
      }
      acc_s[i_s] = condval_10;
    }
    __syncthreads();
    if (((int)threadIdx.x) < 32) {
      int idx_4 = Indices[((((((int64_t)((int)blockIdx.x)) * (int64_t)2112) + (((int64_t)k_i) * (int64_t)32)) + ((int64_t)((int)threadIdx.x))) + (int64_t)64)];
      ((int*)buf_dyn_shmem)[(((int)threadIdx.x) + 64)] = idx_4;
      mask[(((k_i & 1) * 32) + ((int)threadIdx.x))] = ((signed char)(0 <= idx_4));
    }
    bfloat16_t B_local[4];
    for (int ki = 0; ki < 32; ++ki) {
      *(uint2*)(B_local + 0) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((k_i & 1) * 16384) + ((ki >> 2) * 2048)) + ((((int)threadIdx.x) >> 8) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 192));
      {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local) + 0),
                    *(((bfloat16x4_vec*)Q_buf) + ki),
                    *(((float32x4*)acc_s) + 0));
    };
    }
    if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 8)) == 0) {
      ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 19616)] = m_i[0];
    }
    m_i_clear[0] = -MACART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      m_i_clear[0] = max(m_i_clear[0], acc_s[rv]);
    }
    __syncthreads();
    m_i_clear[0] = tl::AllReduce<tl::MaxOp, 512, 256, 0>::run(m_i_clear[0], (&(((float*)buf_dyn_shmem)[17504])));
    m_i_clear[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(m_i_clear[0], (&(((float*)buf_dyn_shmem)[19040])));
    m_i[0] = max(m_i[0], m_i_clear[0]);
    __syncthreads();
    if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 8)) == 0) {
      ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 19552)] = exp2f(((((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 19616)] - m_i[0]) * sm_scale_log2));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      acc_s[i_4] = exp2f(((acc_s[i_4] * sm_scale_log2) - (m_i[0] * sm_scale_log2)));
    }
    sumexp_i[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
      sumexp_i[0] = (sumexp_i[0] + acc_s[rv_1]);
    }
    __syncthreads();
    sumexp_i[0] = tl::AllReduce<tl::SumOp, 512, 256, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[18528])));
    sumexp_i[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[18016])));
    __syncthreads();
    if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 8)) == 0) {
      ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15))] = ((((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15))] * ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 19552)]) + sumexp_i[0]);
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      for (int vec = 0; vec < 4; ++vec) {
        float4 __1;
          float4 v_ = *(float4*)(acc_o + ((i_5 * 16) + (vec * 4)));
          float4 v__1 = make_float4(((float*)buf_dyn_shmem)[(((i_5 * 16) + (((int)threadIdx.x) & 15)) + 19552)], ((float*)buf_dyn_shmem)[(((i_5 * 16) + (((int)threadIdx.x) & 15)) + 19552)], ((float*)buf_dyn_shmem)[(((i_5 * 16) + (((int)threadIdx.x) & 15)) + 19552)], ((float*)buf_dyn_shmem)[(((i_5 * 16) + (((int)threadIdx.x) & 15)) + 19552)]);
          __1.x = (v_.x*v__1.x);
          __1.y = (v_.y*v__1.y);
          __1.z = (v_.z*v__1.z);
          __1.w = (v_.w*v__1.w);
        *(float4*)(acc_o + ((i_5 * 16) + (vec * 4))) = __1;
      }
    }
    uint2 __2;
    float4 v__2 = *(float4*)(acc_s + 0);
    (reinterpret_cast<__maca_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
    (reinterpret_cast<__maca_bfloat162*>(&__2))[1] = __float22bfloat162_rn(((float2*)(&v__2))[1]);
    *(uint2*)(S_shared_local_cast + 0) = __2;
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 255) >> 6) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 8) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 32960)) = *(uint2*)(S_shared_local_cast + 0);
    bfloat16_t A_local[16];
    bfloat16_t B_local_1[16];
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      for (int i_6 = 0; i_6 < 4; ++i_6) {
        *(uint2*)(A_local + (i_6 * 4)) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_6 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 32960));
      }
      for (int j = 0; j < 4; ++j) {
        for (int local_id = 0; local_id < 4; ++local_id) {
          B_local_1[((j * 4) + local_id)] = ((bfloat16_t*)buf_dyn_shmem)[(((((((((((k_i & 1) * 16384) + ((((int)threadIdx.x) >> 6) * 2048)) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 63) >> 4) * 256)) + (local_id * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (j >> 1)) & 1) * 32)) + ((((local_id >> 1) + (j & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 192)];
        }
      }
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          {
      *(((float32x4*)acc_o) + ((i_7 * 4) + j_1)) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local_1) + j_1),
                    *(((bfloat16x4_vec*)A_local) + i_7),
                    *(((float32x4*)acc_o) + ((i_7 * 4) + j_1)));
    };
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      int idx_5 = ((int*)buf_dyn_shmem)[(((i_8 * 8) + (((int)threadIdx.x) >> 6)) + 64)];
      bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      int condval_12;
      if ((0 <= idx_5)) {
        condval_12 = idx_5;
      } else {
        condval_12 = 0;
      }
      int condval_13;
      if ((0 <= idx_5)) {
        condval_13 = idx_5;
      } else {
        condval_13 = 0;
      }
      uint4 condval_11;
      if (((0 <= condval_12) && (condval_13 < seq_len_kv))) {
        int64_t condval_14;
        if (((int64_t)0 <= ((int64_t)idx_5))) {
          condval_14 = ((int64_t)idx_5);
        } else {
          condval_14 = (int64_t)0;
        }
        int64_t condval_15;
        if (((int64_t)0 <= ((int64_t)idx_5))) {
          condval_15 = ((int64_t)idx_5);
        } else {
          condval_15 = (int64_t)0;
        }
        condval_11 = *(uint4*)(KV + ((condval_15 * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) * (int64_t)8)));
      } else {
        condval_11 = make_uint4(__pack_maca_bfloat162(broadcast_var_3, broadcast_var_3), __pack_maca_bfloat162(broadcast_var_3, broadcast_var_3), __pack_maca_bfloat162(broadcast_var_3, broadcast_var_3), __pack_maca_bfloat162(broadcast_var_3, broadcast_var_3));
      }
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((k_i & 1) * 16384) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + (i_8 * 512)) + ((((int)threadIdx.x) >> 6) * 64)) + ((((((int)threadIdx.x) >> 8) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 255) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 192)) = condval_11;
    }
  }
  __syncthreads();
  for (int i_s_1 = 0; i_s_1 < 4; ++i_s_1) {
    float condval_16;
    if (((bool)mask[((((((int)threadIdx.x) >> 8) * 16) + (((((int)threadIdx.x) & 63) >> 4) * 4)) + i_s_1)])) {
      condval_16 = 0x0p+0f/*0.000000e+00*/;
    } else {
      condval_16 = -MACART_INF_F;
    }
    acc_s[i_s_1] = condval_16;
  }
  bfloat16_t B_local_2[4];
  for (int ki_2 = 0; ki_2 < 32; ++ki_2) {
    *(uint2*)(B_local_2 + 0) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((ki_2 >> 2) * 2048) + ((((int)threadIdx.x) >> 8) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 192));
    {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local_2) + 0),
                    *(((bfloat16x4_vec*)Q_buf) + ki_2),
                    *(((float32x4*)acc_s) + 0));
    };
  }
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 8)) == 0) {
    ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 19616)] = m_i[0];
  }
  m_i_clear_1[0] = -MACART_INF_F;
  #pragma unroll
  for (int rv_2 = 0; rv_2 < 4; ++rv_2) {
    m_i_clear_1[0] = max(m_i_clear_1[0], acc_s[rv_2]);
  }
  __syncthreads();
  m_i_clear_1[0] = tl::AllReduce<tl::MaxOp, 512, 256, 0>::run(m_i_clear_1[0], (&(((float*)buf_dyn_shmem)[17504])));
  __syncthreads();
  m_i_clear_1[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(m_i_clear_1[0], (&(((float*)buf_dyn_shmem)[17504])));
  m_i[0] = max(m_i[0], m_i_clear_1[0]);
  __syncthreads();
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 8)) == 0) {
    ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 19552)] = exp2f(((((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 19616)] - m_i[0]) * sm_scale_log2));
  }
  #pragma unroll
  for (int i_9 = 0; i_9 < 4; ++i_9) {
    acc_s[i_9] = exp2f(((acc_s[i_9] * sm_scale_log2) - (m_i[0] * sm_scale_log2)));
  }
  sumexp_i[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_3 = 0; rv_3 < 4; ++rv_3) {
    sumexp_i[0] = (sumexp_i[0] + acc_s[rv_3]);
  }
  __syncthreads();
  sumexp_i[0] = tl::AllReduce<tl::SumOp, 512, 256, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[17504])));
  __syncthreads();
  sumexp_i[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[17504])));
  __syncthreads();
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 8)) == 0) {
    ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15))] = ((((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15))] * ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 19552)]) + sumexp_i[0]);
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 4; ++i_10) {
    for (int vec_1 = 0; vec_1 < 4; ++vec_1) {
      float4 __3;
        float4 v__3 = *(float4*)(acc_o + ((i_10 * 16) + (vec_1 * 4)));
        float4 v__4 = make_float4(((float*)buf_dyn_shmem)[(((i_10 * 16) + (((int)threadIdx.x) & 15)) + 19552)], ((float*)buf_dyn_shmem)[(((i_10 * 16) + (((int)threadIdx.x) & 15)) + 19552)], ((float*)buf_dyn_shmem)[(((i_10 * 16) + (((int)threadIdx.x) & 15)) + 19552)], ((float*)buf_dyn_shmem)[(((i_10 * 16) + (((int)threadIdx.x) & 15)) + 19552)]);
        __3.x = (v__3.x*v__4.x);
        __3.y = (v__3.y*v__4.y);
        __3.z = (v__3.z*v__4.z);
        __3.w = (v__3.w*v__4.w);
      *(float4*)(acc_o + ((i_10 * 16) + (vec_1 * 4))) = __3;
    }
  }
  uint2 __4;
  float4 v__5 = *(float4*)(acc_s + 0);
  (reinterpret_cast<__maca_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__5))[0]);
  (reinterpret_cast<__maca_bfloat162*>(&__4))[1] = __float22bfloat162_rn(((float2*)(&v__5))[1]);
  *(uint2*)(S_shared_local_cast_1 + 0) = __4;
  *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 255) >> 6) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 8) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 32960)) = *(uint2*)(S_shared_local_cast_1 + 0);
  bfloat16_t A_local_1[16];
  bfloat16_t B_local_3[16];
  __syncthreads();
  for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      *(uint2*)(A_local_1 + (i_11 * 4)) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_11 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_3) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 32960));
    }
    for (int j_2 = 0; j_2 < 4; ++j_2) {
      for (int local_id_1 = 0; local_id_1 < 4; ++local_id_1) {
        B_local_3[((j_2 * 4) + local_id_1)] = ((bfloat16_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) >> 6) * 2048) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 63) >> 4) * 256)) + (local_id_1 * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (j_2 >> 1)) & 1) * 32)) + ((((local_id_1 >> 1) + (j_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id_1 & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 192)];
      }
    }
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      for (int j_3 = 0; j_3 < 4; ++j_3) {
        {
      *(((float32x4*)acc_o) + ((i_12 * 4) + j_3)) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local_3) + j_3),
                    *(((bfloat16x4_vec*)A_local_1) + i_12),
                    *(((float32x4*)acc_o) + ((i_12 * 4) + j_3)));
    };
      }
    }
  }
  for (int i_s_2 = 0; i_s_2 < 4; ++i_s_2) {
    float condval_17;
    if (((bool)mask[(((((((int)threadIdx.x) >> 8) * 16) + (((((int)threadIdx.x) & 63) >> 4) * 4)) + i_s_2) + 32)])) {
      condval_17 = 0x0p+0f/*0.000000e+00*/;
    } else {
      condval_17 = -MACART_INF_F;
    }
    acc_s[i_s_2] = condval_17;
  }
  bfloat16_t B_local_4[4];
  for (int ki_4 = 0; ki_4 < 32; ++ki_4) {
    *(uint2*)(B_local_4 + 0) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((ki_4 >> 2) * 2048) + ((((int)threadIdx.x) >> 8) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 16576));
    {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local_4) + 0),
                    *(((bfloat16x4_vec*)Q_buf) + ki_4),
                    *(((float32x4*)acc_s) + 0));
    };
  }
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 8)) == 0) {
    ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 19616)] = m_i[0];
  }
  m_i_clear_2[0] = -MACART_INF_F;
  #pragma unroll
  for (int rv_4 = 0; rv_4 < 4; ++rv_4) {
    m_i_clear_2[0] = max(m_i_clear_2[0], acc_s[rv_4]);
  }
  __syncthreads();
  m_i_clear_2[0] = tl::AllReduce<tl::MaxOp, 512, 256, 0>::run(m_i_clear_2[0], (&(((float*)buf_dyn_shmem)[17504])));
  __syncthreads();
  m_i_clear_2[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(m_i_clear_2[0], (&(((float*)buf_dyn_shmem)[17504])));
  m_i[0] = max(m_i[0], m_i_clear_2[0]);
  __syncthreads();
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 8)) == 0) {
    ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 19552)] = exp2f(((((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 19616)] - m_i[0]) * sm_scale_log2));
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 4; ++i_13) {
    acc_s[i_13] = exp2f(((acc_s[i_13] * sm_scale_log2) - (m_i[0] * sm_scale_log2)));
  }
  sumexp_i[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_5 = 0; rv_5 < 4; ++rv_5) {
    sumexp_i[0] = (sumexp_i[0] + acc_s[rv_5]);
  }
  __syncthreads();
  sumexp_i[0] = tl::AllReduce<tl::SumOp, 512, 256, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[17504])));
  __syncthreads();
  sumexp_i[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[17504])));
  __syncthreads();
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 8)) == 0) {
    ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15))] = ((((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15))] * ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 255) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 19552)]) + sumexp_i[0]);
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 4; ++i_14) {
    for (int vec_2 = 0; vec_2 < 4; ++vec_2) {
      float4 __5;
        float4 v__6 = *(float4*)(acc_o + ((i_14 * 16) + (vec_2 * 4)));
        float4 v__7 = make_float4(((float*)buf_dyn_shmem)[(((i_14 * 16) + (((int)threadIdx.x) & 15)) + 19552)], ((float*)buf_dyn_shmem)[(((i_14 * 16) + (((int)threadIdx.x) & 15)) + 19552)], ((float*)buf_dyn_shmem)[(((i_14 * 16) + (((int)threadIdx.x) & 15)) + 19552)], ((float*)buf_dyn_shmem)[(((i_14 * 16) + (((int)threadIdx.x) & 15)) + 19552)]);
        __5.x = (v__6.x*v__7.x);
        __5.y = (v__6.y*v__7.y);
        __5.z = (v__6.z*v__7.z);
        __5.w = (v__6.w*v__7.w);
      *(float4*)(acc_o + ((i_14 * 16) + (vec_2 * 4))) = __5;
    }
  }
  uint2 __6;
  float4 v__8 = *(float4*)(acc_s + 0);
  (reinterpret_cast<__maca_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&v__8))[0]);
  (reinterpret_cast<__maca_bfloat162*>(&__6))[1] = __float22bfloat162_rn(((float2*)(&v__8))[1]);
  *(uint2*)(S_shared_local_cast_2 + 0) = __6;
  *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 255) >> 6) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 8) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 32960)) = *(uint2*)(S_shared_local_cast_2 + 0);
  bfloat16_t A_local_2[16];
  bfloat16_t B_local_5[16];
  __syncthreads();
  for (int ki_5 = 0; ki_5 < 2; ++ki_5) {
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      *(uint2*)(A_local_2 + (i_15 * 4)) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_15 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_5) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 32960));
    }
    for (int j_4 = 0; j_4 < 4; ++j_4) {
      for (int local_id_2 = 0; local_id_2 < 4; ++local_id_2) {
        B_local_5[((j_4 * 4) + local_id_2)] = ((bfloat16_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) >> 6) * 2048) + (ki_5 * 1024)) + (((((int)threadIdx.x) & 63) >> 4) * 256)) + (local_id_2 * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (j_4 >> 1)) & 1) * 32)) + ((((local_id_2 >> 1) + (j_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id_2 & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 16576)];
      }
    }
    for (int i_16 = 0; i_16 < 4; ++i_16) {
      for (int j_5 = 0; j_5 < 4; ++j_5) {
        {
      *(((float32x4*)acc_o) + ((i_16 * 4) + j_5)) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local_5) + j_5),
                    *(((bfloat16x4_vec*)A_local_2) + i_16),
                    *(((float32x4*)acc_o) + ((i_16 * 4) + j_5)));
    };
      }
    }
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 4; ++i_17) {
    for (int vec_3 = 0; vec_3 < 4; ++vec_3) {
      float4 __7;
        float4 v__9 = *(float4*)(acc_o + ((i_17 * 16) + (vec_3 * 4)));
        float condval_18;
        if ((((float*)buf_dyn_shmem)[((i_17 * 16) + (((int)threadIdx.x) & 15))] == 0x0p+0f/*0.000000e+00*/)) {
          condval_18 = 0x1p+0f/*1.000000e+00*/;
        } else {
          condval_18 = ((float*)buf_dyn_shmem)[((i_17 * 16) + (((int)threadIdx.x) & 15))];
        }
        float4 v__10 = make_float4(condval_18, condval_18, condval_18, condval_18);
        __7.x = (v__9.x/v__10.x);
        __7.y = (v__9.y/v__10.y);
        __7.z = (v__9.z/v__10.z);
        __7.w = (v__9.w/v__10.w);
      *(float4*)(acc_o + ((i_17 * 16) + (vec_3 * 4))) = __7;
    }
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 16; ++i_18) {
    uint2 __8;
    float4 v__11 = *(float4*)(acc_o + (i_18 * 4));
    (reinterpret_cast<__maca_bfloat162*>(&__8))[0] = __float22bfloat162_rn(((float2*)(&v__11))[0]);
    (reinterpret_cast<__maca_bfloat162*>(&__8))[1] = __float22bfloat162_rn(((float2*)(&v__11))[1]);
    *(uint2*)(Output_local_cast_3 + 0) = __8;
    *(uint2*)(Output + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)32768) + ((((int64_t)i_18) >> (int64_t)2) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + ((((int64_t)i_18) & (int64_t)3) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4))) = *(uint2*)(Output_local_cast_3 + 0);
  }
}


// ---- Stage-1 prefill kernel ----
__global__ void __launch_bounds__(256, 1) sparse_prefill_stage1_generated_kernel(const int* __restrict__ Indices, const bfloat16_t* __restrict__ KV, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ Q, int seq_len, int seq_len_kv, float sm_scale_log2) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[64];
  float m_i[1];
  bfloat16_t Q_buf[128];
  __shared__ signed char mask[32];
  float acc_s[4];
  float sumexp_i[1];
  float m_i_clear[1];
  bfloat16_t S_shared_local_cast[4];
  float m_i_clear_1[1];
  bfloat16_t S_shared_local_cast_1[4];
  bfloat16_t Output_local_cast_2[4];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  if (((int)threadIdx.x) < 32) {
    ((float*)buf_dyn_shmem)[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
  }
  m_i[0] = -0x1p+30f/*-1.073742e+09*/;
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    *(uint2*)(Q_buf + (i_1 * 4)) = *(uint2*)(Q + (((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + (((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)6) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + (((int64_t)i_1) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4)));
  }
  if (((int)threadIdx.x) < 32) {
    int idx = Indices[(((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2112) + ((int64_t)((int)threadIdx.x)))];
    ((int*)buf_dyn_shmem)[(((int)threadIdx.x) + 32)] = idx;
    mask[((int)threadIdx.x)] = ((signed char)(0 <= idx));
  }
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    int idx_1 = ((int*)buf_dyn_shmem)[(((i_2 * 4) + (((int)threadIdx.x) >> 6)) + 32)];
    bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    int condval_1;
    if ((0 <= idx_1)) {
      condval_1 = idx_1;
    } else {
      condval_1 = 0;
    }
    int condval_2;
    if ((0 <= idx_1)) {
      condval_2 = idx_1;
    } else {
      condval_2 = 0;
    }
    uint4 condval;
    if (((0 <= condval_1) && (condval_2 < seq_len_kv))) {
      int64_t condval_3;
      if (((int64_t)0 <= ((int64_t)idx_1))) {
        condval_3 = ((int64_t)idx_1);
      } else {
        condval_3 = (int64_t)0;
      }
      int64_t condval_4;
      if (((int64_t)0 <= ((int64_t)idx_1))) {
        condval_4 = ((int64_t)idx_1);
      } else {
        condval_4 = (int64_t)0;
      }
      condval = *(uint4*)(KV + ((condval_4 * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_maca_bfloat162(broadcast_var_1, broadcast_var_1), __pack_maca_bfloat162(broadcast_var_1, broadcast_var_1), __pack_maca_bfloat162(broadcast_var_1, broadcast_var_1), __pack_maca_bfloat162(broadcast_var_1, broadcast_var_1));
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 63) >> 3) * 2048) + (i_2 * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_2 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval;
  }
  for (int k_i = 0; k_i < 65; ++k_i) {
    __syncthreads();
    for (int i_s = 0; i_s < 4; ++i_s) {
      float condval_5;
      if (((bool)mask[((((((int)threadIdx.x) >> 7) * 16) + (((((int)threadIdx.x) & 63) >> 4) * 4)) + i_s)])) {
        condval_5 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_5 = -MACART_INF_F;
      }
      acc_s[i_s] = condval_5;
    }
    __syncthreads();
    if (((int)threadIdx.x) < 32) {
      int idx_2 = Indices[(((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2112) + (((int64_t)k_i) * (int64_t)32)) + ((int64_t)((int)threadIdx.x))) + (int64_t)32)];
      ((int*)buf_dyn_shmem)[(((int)threadIdx.x) + 32)] = idx_2;
      mask[((int)threadIdx.x)] = ((signed char)(0 <= idx_2));
    }
    bfloat16_t B_local[4];
    for (int ki = 0; ki < 32; ++ki) {
      *(uint2*)(B_local + 0) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((ki >> 2) * 2048) + ((((int)threadIdx.x) >> 7) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 128));
      {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local) + 0),
                    *(((bfloat16x4_vec*)Q_buf) + ki),
                    *(((float32x4*)acc_s) + 0));
    };
    }
    if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
      ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9824)] = m_i[0];
    }
    m_i_clear[0] = -MACART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      m_i_clear[0] = max(m_i_clear[0], acc_s[rv]);
    }
    __syncthreads();
    m_i_clear[0] = tl::AllReduce<tl::MaxOp, 256, 128, 0>::run(m_i_clear[0], (&(((float*)buf_dyn_shmem)[9280])));
    m_i_clear[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(m_i_clear[0], (&(((float*)buf_dyn_shmem)[9536])));
    m_i[0] = max(m_i[0], m_i_clear[0]);
    __syncthreads();
    if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
      ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9792)] = exp2f(((((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9824)] - m_i[0]) * sm_scale_log2));
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      acc_s[i_3] = exp2f(((acc_s[i_3] * sm_scale_log2) - (m_i[0] * sm_scale_log2)));
    }
    sumexp_i[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
      sumexp_i[0] = (sumexp_i[0] + acc_s[rv_1]);
    }
    __syncthreads();
    sumexp_i[0] = tl::AllReduce<tl::SumOp, 256, 128, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[8768])));
    sumexp_i[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[9024])));
    __syncthreads();
    if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
      ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] = ((((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] * ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9792)]) + sumexp_i[0]);
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      for (int vec = 0; vec < 8; ++vec) {
        float4 __1;
          float4 v_ = *(float4*)(acc_o + ((i_4 * 32) + (vec * 4)));
          float4 v__1 = make_float4(((float*)buf_dyn_shmem)[(((i_4 * 16) + (((int)threadIdx.x) & 15)) + 9792)], ((float*)buf_dyn_shmem)[(((i_4 * 16) + (((int)threadIdx.x) & 15)) + 9792)], ((float*)buf_dyn_shmem)[(((i_4 * 16) + (((int)threadIdx.x) & 15)) + 9792)], ((float*)buf_dyn_shmem)[(((i_4 * 16) + (((int)threadIdx.x) & 15)) + 9792)]);
          __1.x = (v_.x*v__1.x);
          __1.y = (v_.y*v__1.y);
          __1.z = (v_.z*v__1.z);
          __1.w = (v_.w*v__1.w);
        *(float4*)(acc_o + ((i_4 * 32) + (vec * 4))) = __1;
      }
    }
    uint2 __2;
    float4 v__2 = *(float4*)(acc_s + 0);
    (reinterpret_cast<__maca_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
    (reinterpret_cast<__maca_bfloat162*>(&__2))[1] = __float22bfloat162_rn(((float2*)(&v__2))[1]);
    *(uint2*)(S_shared_local_cast + 0) = __2;
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 127) >> 6) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 16512)) = *(uint2*)(S_shared_local_cast + 0);
    bfloat16_t A_local[8];
    bfloat16_t B_local_1[32];
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        *(uint2*)(A_local + (i_5 * 4)) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_5 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 16512));
      }
      for (int j = 0; j < 8; ++j) {
        for (int local_id = 0; local_id < 4; ++local_id) {
          B_local_1[((j * 4) + local_id)] = ((bfloat16_t*)buf_dyn_shmem)[(((((((((((((int)threadIdx.x) >> 6) * 4096) + ((j >> 2) * 2048)) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 63) >> 4) * 256)) + (local_id * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((j & 3) >> 1)) & 1) * 32)) + ((((local_id >> 1) + (j & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 128)];
        }
      }
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          {
      *(((float32x4*)acc_o) + ((i_6 * 8) + j_1)) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local_1) + j_1),
                    *(((bfloat16x4_vec*)A_local) + i_6),
                    *(((float32x4*)acc_o) + ((i_6 * 8) + j_1)));
    };
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_7 = 0; i_7 < 8; ++i_7) {
      int idx_3 = ((int*)buf_dyn_shmem)[(((i_7 * 4) + (((int)threadIdx.x) >> 6)) + 32)];
      bfloat16_t broadcast_var_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      int condval_7;
      if ((0 <= idx_3)) {
        condval_7 = idx_3;
      } else {
        condval_7 = 0;
      }
      int condval_8;
      if ((0 <= idx_3)) {
        condval_8 = idx_3;
      } else {
        condval_8 = 0;
      }
      uint4 condval_6;
      if (((0 <= condval_7) && (condval_8 < seq_len_kv))) {
        int64_t condval_9;
        if (((int64_t)0 <= ((int64_t)idx_3))) {
          condval_9 = ((int64_t)idx_3);
        } else {
          condval_9 = (int64_t)0;
        }
        int64_t condval_10;
        if (((int64_t)0 <= ((int64_t)idx_3))) {
          condval_10 = ((int64_t)idx_3);
        } else {
          condval_10 = (int64_t)0;
        }
        condval_6 = *(uint4*)(KV + ((condval_10 * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) * (int64_t)8)));
      } else {
        condval_6 = make_uint4(__pack_maca_bfloat162(broadcast_var_2, broadcast_var_2), __pack_maca_bfloat162(broadcast_var_2, broadcast_var_2), __pack_maca_bfloat162(broadcast_var_2, broadcast_var_2), __pack_maca_bfloat162(broadcast_var_2, broadcast_var_2));
      }
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 63) >> 3) * 2048) + (i_7 * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_7 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 128)) = condval_6;
    }
  }
  __syncthreads();
  for (int i_s_1 = 0; i_s_1 < 4; ++i_s_1) {
    float condval_11;
    if (((bool)mask[((((((int)threadIdx.x) >> 7) * 16) + (((((int)threadIdx.x) & 63) >> 4) * 4)) + i_s_1)])) {
      condval_11 = 0x0p+0f/*0.000000e+00*/;
    } else {
      condval_11 = -MACART_INF_F;
    }
    acc_s[i_s_1] = condval_11;
  }
  bfloat16_t B_local_2[4];
  for (int ki_2 = 0; ki_2 < 32; ++ki_2) {
    *(uint2*)(B_local_2 + 0) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((ki_2 >> 2) * 2048) + ((((int)threadIdx.x) >> 7) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 128));
    {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local_2) + 0),
                    *(((bfloat16x4_vec*)Q_buf) + ki_2),
                    *(((float32x4*)acc_s) + 0));
    };
  }
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
    ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9824)] = m_i[0];
  }
  m_i_clear_1[0] = -MACART_INF_F;
  #pragma unroll
  for (int rv_2 = 0; rv_2 < 4; ++rv_2) {
    m_i_clear_1[0] = max(m_i_clear_1[0], acc_s[rv_2]);
  }
  __syncthreads();
  m_i_clear_1[0] = tl::AllReduce<tl::MaxOp, 256, 128, 0>::run(m_i_clear_1[0], (&(((float*)buf_dyn_shmem)[8768])));
  __syncthreads();
  m_i_clear_1[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(m_i_clear_1[0], (&(((float*)buf_dyn_shmem)[8768])));
  m_i[0] = max(m_i[0], m_i_clear_1[0]);
  __syncthreads();
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
    ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9792)] = exp2f(((((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9824)] - m_i[0]) * sm_scale_log2));
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 4; ++i_8) {
    acc_s[i_8] = exp2f(((acc_s[i_8] * sm_scale_log2) - (m_i[0] * sm_scale_log2)));
  }
  sumexp_i[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_3 = 0; rv_3 < 4; ++rv_3) {
    sumexp_i[0] = (sumexp_i[0] + acc_s[rv_3]);
  }
  __syncthreads();
  sumexp_i[0] = tl::AllReduce<tl::SumOp, 256, 128, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[8768])));
  __syncthreads();
  sumexp_i[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(sumexp_i[0], (&(((float*)buf_dyn_shmem)[8768])));
  __syncthreads();
  if (((((((int)threadIdx.x) & 63) >> 4) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
    ((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] = ((((float*)buf_dyn_shmem)[((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15))] * ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 127) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 9792)]) + sumexp_i[0]);
  }
  #pragma unroll
  for (int i_9 = 0; i_9 < 2; ++i_9) {
    for (int vec_1 = 0; vec_1 < 8; ++vec_1) {
      float4 __3;
        float4 v__3 = *(float4*)(acc_o + ((i_9 * 32) + (vec_1 * 4)));
        float4 v__4 = make_float4(((float*)buf_dyn_shmem)[(((i_9 * 16) + (((int)threadIdx.x) & 15)) + 9792)], ((float*)buf_dyn_shmem)[(((i_9 * 16) + (((int)threadIdx.x) & 15)) + 9792)], ((float*)buf_dyn_shmem)[(((i_9 * 16) + (((int)threadIdx.x) & 15)) + 9792)], ((float*)buf_dyn_shmem)[(((i_9 * 16) + (((int)threadIdx.x) & 15)) + 9792)]);
        __3.x = (v__3.x*v__4.x);
        __3.y = (v__3.y*v__4.y);
        __3.z = (v__3.z*v__4.z);
        __3.w = (v__3.w*v__4.w);
      *(float4*)(acc_o + ((i_9 * 32) + (vec_1 * 4))) = __3;
    }
  }
  uint2 __4;
  float4 v__5 = *(float4*)(acc_s + 0);
  (reinterpret_cast<__maca_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__5))[0]);
  (reinterpret_cast<__maca_bfloat162*>(&__4))[1] = __float22bfloat162_rn(((float2*)(&v__5))[1]);
  *(uint2*)(S_shared_local_cast_1 + 0) = __4;
  *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 127) >> 6) * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 16512)) = *(uint2*)(S_shared_local_cast_1 + 0);
  bfloat16_t A_local_1[8];
  bfloat16_t B_local_3[32];
  __syncthreads();
  for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      *(uint2*)(A_local_1 + (i_10 * 4)) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((((((i_10 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_3) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 16512));
    }
    for (int j_2 = 0; j_2 < 8; ++j_2) {
      for (int local_id_1 = 0; local_id_1 < 4; ++local_id_1) {
        B_local_3[((j_2 * 4) + local_id_1)] = ((bfloat16_t*)buf_dyn_shmem)[(((((((((((((int)threadIdx.x) >> 6) * 4096) + ((j_2 >> 2) * 2048)) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 63) >> 4) * 256)) + (local_id_1 * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((j_2 & 3) >> 1)) & 1) * 32)) + ((((local_id_1 >> 1) + (j_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id_1 & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7)) + 128)];
      }
    }
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        {
      *(((float32x4*)acc_o) + ((i_11 * 8) + j_3)) = __builtin_mxc_mma_16x16x16bf16(*(((bfloat16x4_vec*)B_local_3) + j_3),
                    *(((bfloat16x4_vec*)A_local_1) + i_11),
                    *(((float32x4*)acc_o) + ((i_11 * 8) + j_3)));
    };
      }
    }
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 2; ++i_12) {
    for (int vec_2 = 0; vec_2 < 8; ++vec_2) {
      float4 __5;
        float4 v__6 = *(float4*)(acc_o + ((i_12 * 32) + (vec_2 * 4)));
        float condval_12;
        if ((((float*)buf_dyn_shmem)[((i_12 * 16) + (((int)threadIdx.x) & 15))] == 0x0p+0f/*0.000000e+00*/)) {
          condval_12 = 0x1p+0f/*1.000000e+00*/;
        } else {
          condval_12 = ((float*)buf_dyn_shmem)[((i_12 * 16) + (((int)threadIdx.x) & 15))];
        }
        float4 v__7 = make_float4(condval_12, condval_12, condval_12, condval_12);
        __5.x = (v__6.x/v__7.x);
        __5.y = (v__6.y/v__7.y);
        __5.z = (v__6.z/v__7.z);
        __5.w = (v__6.w/v__7.w);
      *(float4*)(acc_o + ((i_12 * 32) + (vec_2 * 4))) = __5;
    }
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 16; ++i_13) {
    uint2 __6;
    float4 v__8 = *(float4*)(acc_o + (i_13 * 4));
    (reinterpret_cast<__maca_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&v__8))[0]);
    (reinterpret_cast<__maca_bfloat162*>(&__6))[1] = __float22bfloat162_rn(((float2*)(&v__8))[1]);
    *(uint2*)(Output_local_cast_2 + 0) = __6;
    *(uint2*)(Output + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + ((((int64_t)i_13) >> (int64_t)3) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)128)) + ((((int64_t)i_13) & (int64_t)7) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4))) = *(uint2*)(Output_local_cast_2 + 0);
  }
}

