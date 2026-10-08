#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

// Production specializations: hidden=7168, mhc_mult=4, n_splits=16/64.

using bfloat16_t = __maca_bfloat16;

__device__ __forceinline__ float mhc_row_max4(float value) {
  value = fmaxf(value, __shfl_xor_sync(uint64_t(-1), value, 2));
  return fmaxf(value, __shfl_xor_sync(uint64_t(-1), value, 1));
}

__device__ __forceinline__ float mhc_row_sum4(float value) {
  value += __shfl_xor_sync(uint64_t(-1), value, 2);
  return value + __shfl_xor_sync(uint64_t(-1), value, 1);
}

__device__ __forceinline__ float mhc_col_sum4(float value) {
  value += __shfl_xor_sync(uint64_t(-1), value, 8);
  return value + __shfl_xor_sync(uint64_t(-1), value, 4);
}

template <bool FastDivide>
__device__ __forceinline__ float mhc_sinkhorn_div(float numerator,
                                                  float denominator) {
  if constexpr (FastDivide) {
    return __fdividef(numerator, denominator);
  }
  return numerator / denominator;
}

template <int Split, int HiddenBlock = 1024, int ResidualWarps = 1,
          bool FastSinkhornDivide = false, int StaticTokens = 0>
__global__ void __launch_bounds__(64 * (1 + ResidualWarps), 1) mhc_pre_big_fuse_kernel_kernel(float* __restrict__ comb_mix, const float* __restrict__ gemm_out_mul, const float* __restrict__ gemm_out_sqrsum, bfloat16_t* __restrict__ layer_input, const float* __restrict__ mhc_base, const float* __restrict__ mhc_scale, float* __restrict__ post_mix, const bfloat16_t* __restrict__ residual, int num_tokens) {
  static_assert(Split == 16 || Split == 64, "unsupported mHC pre split");
  static_assert(HiddenBlock == 1024, "unsupported hidden block");
  static_assert(ResidualWarps == 1 || ResidualWarps == 2,
                "unsupported residual warp count");
  static_assert(StaticTokens == 0 || StaticTokens == 6 || StaticTokens == 12 ||
                    StaticTokens == 18,
                "unsupported static token specialization");
  constexpr int kControlTokens = StaticTokens == 0 ? 0 : StaticTokens;
  const int control_tokens = kControlTokens == 0 ? num_tokens : kControlTokens;
  constexpr int kMixLanes = Split == 64 ? 4 : 2;
  constexpr int kPreOffset = 8 + 24 * kMixLanes;
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float rms[1];
  float rms_1[1];
  float mix_partial[1];
  float mix_value[1];
  float cm[1];
  float row_max[1];
  float row_sum[1];
  float col_sum[1];
  if (((int)threadIdx.x) < 8) {
    rms[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int i_part = 0; i_part < Split / 8; ++i_part) {
      rms[0] = (rms[0] + gemm_out_sqrsum[((((((int64_t)i_part) * (int64_t)8) + ((int64_t)((int)threadIdx.x))) * ((int64_t)control_tokens)) + ((int64_t)((int)blockIdx.x)))]);
    }
    ((float*)buf_dyn_shmem)[((int)threadIdx.x)] = rms[0];
  }
  if (((int)threadIdx.x) < 24 * kMixLanes) {
    mix_partial[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int i_part_1 = 0; i_part_1 < Split / kMixLanes; ++i_part_1) {
      mix_partial[0] = (mix_partial[0] + gemm_out_mul[(((((int64_t)((int)blockIdx.x)) * (int64_t)24) + ((((((int64_t)i_part_1) * (int64_t)kMixLanes) + (((int64_t)((int)threadIdx.x)) / (int64_t)24)) * ((int64_t)control_tokens)) * (int64_t)24)) + (((int64_t)((int)threadIdx.x)) % (int64_t)24))]);
    }
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 8)] = mix_partial[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    rms_1[0] = 0x0p+0f/*0.000000e+00*/;
    for (int i_lane = 0; i_lane < 8; ++i_lane) {
      rms_1[0] = (rms_1[0] + ((float*)buf_dyn_shmem)[i_lane]);
    }
    ((float*)buf_dyn_shmem)[0] = rsqrtf(((rms_1[0] / 0x1.cp+14f/*2.867200e+04*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  }
  if (((int)threadIdx.x) < 24) {
    mix_value[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int i_lane_1 = 0; i_lane_1 < kMixLanes; ++i_lane_1) {
      mix_value[0] = (mix_value[0] + ((float*)buf_dyn_shmem)[(((i_lane_1 * 24) + ((int)threadIdx.x)) + 8)]);
    }
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 8)] = mix_value[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    if (((int)threadIdx.x) < 4) {
      post_mix[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + ((int64_t)((int)threadIdx.x)))] = ((0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - (((((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 12)] * ((float*)buf_dyn_shmem)[0]) * mhc_scale[1]) + mhc_base[(((int)threadIdx.x) + 4)]))))) * 0x1p+1f/*2.000000e+00*/);
    }
    cm[0] = (((((float*)buf_dyn_shmem)[((((int)threadIdx.x) & 15) + 16)] * ((float*)buf_dyn_shmem)[0]) * mhc_scale[2]) + mhc_base[((((int)threadIdx.x) & 15) + 8)]);
    row_max[0] = -INFINITY;
    row_max[0] = max(row_max[0], cm[0]);
    row_max[0] = mhc_row_max4(row_max[0]);
    cm[0] = expf((cm[0] - row_max[0]));
    row_sum[0] = 0x0p+0f/*0.000000e+00*/;
    row_sum[0] = (row_sum[0] + cm[0]);
    row_sum[0] = mhc_row_sum4(row_sum[0]);
    cm[0] = (mhc_sinkhorn_div<FastSinkhornDivide>(cm[0], row_sum[0]) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
    col_sum[0] = 0x0p+0f/*0.000000e+00*/;
    col_sum[0] = (col_sum[0] + cm[0]);
    col_sum[0] = mhc_col_sum4(col_sum[0]);
    cm[0] = mhc_sinkhorn_div<FastSinkhornDivide>(cm[0], col_sum[0] + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
    #pragma unroll 19
    for (int __1 = 0; __1 < 19; ++__1) {
      row_sum[0] = 0x0p+0f/*0.000000e+00*/;
      row_sum[0] = (row_sum[0] + cm[0]);
      row_sum[0] = mhc_row_sum4(row_sum[0]);
      cm[0] = mhc_sinkhorn_div<FastSinkhornDivide>(cm[0], row_sum[0] + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
      col_sum[0] = 0x0p+0f/*0.000000e+00*/;
      col_sum[0] = (col_sum[0] + cm[0]);
      col_sum[0] = mhc_col_sum4(col_sum[0]);
      cm[0] = mhc_sinkhorn_div<FastSinkhornDivide>(cm[0], col_sum[0] + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
    }
    if ((((int)threadIdx.x) >> 4) == 0) {
      comb_mix[((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) & (int64_t)15))] = cm[0];
    }
  } else {
    const int residual_lane = ((int)threadIdx.x) - 64;
    float pre0 = 0.0f;
    float pre1 = 0.0f;
    float pre2 = 0.0f;
    float pre3 = 0.0f;
    if constexpr (ResidualWarps == 1) {
      if (((int)threadIdx.x) < 68) {
        ((float*)buf_dyn_shmem)[kPreOffset + residual_lane] = ((0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - (((((float*)buf_dyn_shmem)[residual_lane + 8] * ((float*)buf_dyn_shmem)[0]) * mhc_scale[0]) + mhc_base[residual_lane]))))) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
      }
    } else {
      const int residual_warp_lane = residual_lane & 63;
      float lane_pre = 0.0f;
      if (residual_warp_lane < 4) {
        lane_pre =
            (1.0f /
             (1.0f + expf(-((((float*)buf_dyn_shmem)[residual_warp_lane + 8] *
                              ((float*)buf_dyn_shmem)[0]) *
                                 mhc_scale[0] +
                             mhc_base[residual_warp_lane])))) +
            1.0e-6f;
      }
      pre0 = __shfl_sync(uint64_t(-1), lane_pre, 0);
      pre1 = __shfl_sync(uint64_t(-1), lane_pre, 1);
      pre2 = __shfl_sync(uint64_t(-1), lane_pre, 2);
      pre3 = __shfl_sync(uint64_t(-1), lane_pre, 3);
    }
    for (int i0_h = 0; i0_h < 7168 / HiddenBlock; ++i0_h) {
      float4 acc0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float4 acc1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float4 acc2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float4 acc3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      #pragma unroll
      for (int i_mhc = 0; i_mhc < 4; ++i_mhc) {
        float pre;
        if constexpr (ResidualWarps == 1) {
          pre = ((float*)buf_dyn_shmem)[kPreOffset + i_mhc];
        } else {
          pre = i_mhc == 0 ? pre0 :
                i_mhc == 1 ? pre1 :
                i_mhc == 2 ? pre2 : pre3;
        }
        const int64_t offset =
            (((int64_t)((int)blockIdx.x)) * (int64_t)28672) +
            (((int64_t)i_mhc) * (int64_t)7168) +
            (((int64_t)i0_h) * (int64_t)HiddenBlock) +
            (((int64_t)residual_lane) * (int64_t)8);
        uint4 packed0 = *(uint4*)(residual + offset);
        const __maca_bfloat162* values0 =
            reinterpret_cast<const __maca_bfloat162*>(&packed0);
        float2 v0 = __bfloat1622float2(values0[0]);
        float2 v1 = __bfloat1622float2(values0[1]);
        float2 v2 = __bfloat1622float2(values0[2]);
        float2 v3 = __bfloat1622float2(values0[3]);
        acc0.x = acc0.x + pre * v0.x;
        acc0.y = acc0.y + pre * v0.y;
        acc0.z = acc0.z + pre * v1.x;
        acc0.w = acc0.w + pre * v1.y;
        acc1.x = acc1.x + pre * v2.x;
        acc1.y = acc1.y + pre * v2.y;
        acc1.z = acc1.z + pre * v3.x;
        acc1.w = acc1.w + pre * v3.y;
        if constexpr (ResidualWarps == 1) {
          uint4 packed1 = *(uint4*)(residual + offset + HiddenBlock / 2);
          const __maca_bfloat162* values1 =
              reinterpret_cast<const __maca_bfloat162*>(&packed1);
          float2 v4 = __bfloat1622float2(values1[0]);
          float2 v5 = __bfloat1622float2(values1[1]);
          float2 v6 = __bfloat1622float2(values1[2]);
          float2 v7 = __bfloat1622float2(values1[3]);
          acc2.x = acc2.x + pre * v4.x;
          acc2.y = acc2.y + pre * v4.y;
          acc2.z = acc2.z + pre * v5.x;
          acc2.w = acc2.w + pre * v5.y;
          acc3.x = acc3.x + pre * v6.x;
          acc3.y = acc3.y + pre * v6.y;
          acc3.z = acc3.z + pre * v7.x;
          acc3.w = acc3.w + pre * v7.y;
        }
      }
      uint4 packed_output0;
      __maca_bfloat162* output0 =
          reinterpret_cast<__maca_bfloat162*>(&packed_output0);
      output0[0] = __float22bfloat162_rn(make_float2(acc0.x, acc0.y));
      output0[1] = __float22bfloat162_rn(make_float2(acc0.z, acc0.w));
      output0[2] = __float22bfloat162_rn(make_float2(acc1.x, acc1.y));
      output0[3] = __float22bfloat162_rn(make_float2(acc1.z, acc1.w));
      uint4 packed_output1;
      if constexpr (ResidualWarps == 1) {
        __maca_bfloat162* output1 =
            reinterpret_cast<__maca_bfloat162*>(&packed_output1);
        output1[0] = __float22bfloat162_rn(make_float2(acc2.x, acc2.y));
        output1[1] = __float22bfloat162_rn(make_float2(acc2.z, acc2.w));
        output1[2] = __float22bfloat162_rn(make_float2(acc3.x, acc3.y));
        output1[3] = __float22bfloat162_rn(make_float2(acc3.z, acc3.w));
      }
      const int64_t output_offset =
          (((int64_t)((int)blockIdx.x)) * (int64_t)7168) +
          (((int64_t)i0_h) * (int64_t)HiddenBlock) +
          (((int64_t)residual_lane) * (int64_t)8);
      *(uint4*)(layer_input + output_offset) = packed_output0;
      if constexpr (ResidualWarps == 1) {
        *(uint4*)(layer_input + output_offset + HiddenBlock / 2) =
            packed_output1;
      }
    }
  }
}
