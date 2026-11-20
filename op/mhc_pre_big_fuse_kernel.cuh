#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

// Production specializations: mhc_mult=4, n_splits=16/64, hidden=4096|7168.

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

template <bool FastDivide, int Hidden = 7168>
__device__ __forceinline__ float mhc_sinkhorn_div(float numerator,
                                                  float denominator) {
  if constexpr (FastDivide) {
    return __fdividef(numerator, denominator);
  }
  if constexpr (Hidden == 4096) {
    // hidden=4096 runs only 4 hidden-block iterations per CTA, so the Sinkhorn
    // chain is a much larger share of the block; the hardware reciprocal is a
    // uniform win there (measured +9.8..12.6% at N=24..64, +1% at N>=334, no
    // regressing shape).  hidden=7168 loses 2.5-3.6% at N>=4096 with the same
    // substitution, so it keeps the precise divide.
    return numerator * __builtin_mxc_rcpf(denominator);
  }
  return numerator / denominator;
}

template <int Split, int HiddenBlock = 1024, int ResidualWarps = 1,
          bool FastSinkhornDivide = false, int StaticTokens = 0,
          int Hidden = 7168>
__global__ void __launch_bounds__(64 * (1 + ResidualWarps), 1) mhc_pre_big_fuse_kernel_kernel(float* __restrict__ comb_mix, const float* __restrict__ gemm_out_mul, const float* __restrict__ gemm_out_sqrsum, bfloat16_t* __restrict__ layer_input, const float* __restrict__ mhc_base, const float* __restrict__ mhc_scale, float* __restrict__ post_mix, const bfloat16_t* __restrict__ residual, int num_tokens) {
  static_assert(Split == 16 || Split == 64, "unsupported mHC pre split");
  static_assert(HiddenBlock == 1024, "unsupported hidden block");
  static_assert(ResidualWarps == 1 || ResidualWarps == 2,
                "unsupported residual warp count");
  static_assert(StaticTokens == 0 || StaticTokens == 6 || StaticTokens == 12 ||
                    StaticTokens == 18,
                "unsupported static token specialization");
  static_assert(Hidden == 4096 || Hidden == 7168,
                "supported hidden sizes are 4096 and 7168");
  // One iteration must cover exactly HiddenBlock elements with the residual
  // lane mapping below: ResidualWarps==1 uses 64 lanes x 8 elements x 2 chunks,
  // ResidualWarps>=2 uses 64*ResidualWarps lanes x 8 elements.
  static_assert(HiddenBlock == 64 * 8 * (ResidualWarps == 1 ? 2 : ResidualWarps),
                "hidden block must match the residual lane coverage");
  static_assert(Hidden % HiddenBlock == 0,
                "hidden must be a multiple of the hidden block");
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
    ((float*)buf_dyn_shmem)[0] = rsqrtf(((rms_1[0] / (float)(4 * Hidden)) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
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
    cm[0] = (mhc_sinkhorn_div<FastSinkhornDivide, Hidden>(cm[0], row_sum[0]) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
    col_sum[0] = 0x0p+0f/*0.000000e+00*/;
    col_sum[0] = (col_sum[0] + cm[0]);
    col_sum[0] = mhc_col_sum4(col_sum[0]);
    cm[0] = mhc_sinkhorn_div<FastSinkhornDivide, Hidden>(cm[0], col_sum[0] + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
    #pragma unroll 19
    for (int __1 = 0; __1 < 19; ++__1) {
      row_sum[0] = 0x0p+0f/*0.000000e+00*/;
      row_sum[0] = (row_sum[0] + cm[0]);
      row_sum[0] = mhc_row_sum4(row_sum[0]);
      cm[0] = mhc_sinkhorn_div<FastSinkhornDivide, Hidden>(cm[0], row_sum[0] + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
      col_sum[0] = 0x0p+0f/*0.000000e+00*/;
      col_sum[0] = (col_sum[0] + cm[0]);
      col_sum[0] = mhc_col_sum4(col_sum[0]);
      cm[0] = mhc_sinkhorn_div<FastSinkhornDivide, Hidden>(cm[0], col_sum[0] + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
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
    for (int i0_h = 0; i0_h < Hidden / HiddenBlock; ++i0_h) {
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
            (((int64_t)((int)blockIdx.x)) * (int64_t)(4 * Hidden)) +
            (((int64_t)i_mhc) * (int64_t)Hidden) +
            (((int64_t)i0_h) * (int64_t)HiddenBlock) +
            (((int64_t)residual_lane) * (int64_t)8);
        // residual is read exactly once: stream it through the cache
        // instead of allocating L2 lines that are never reused.
        uint4 packed0 =
            __ldcs(reinterpret_cast<const uint4*>(residual + offset));
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
          uint4 packed1 = __ldcs(reinterpret_cast<const uint4*>(
              residual + offset + HiddenBlock / 2));
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
          (((int64_t)((int)blockIdx.x)) * (int64_t)Hidden) +
          (((int64_t)i0_h) * (int64_t)HiddenBlock) +
          (((int64_t)residual_lane) * (int64_t)8);
      // layer_input is write-only for this operator: bypass L2 residency.
      __stcs(reinterpret_cast<uint4*>(layer_input + output_offset),
             packed_output0);
      if constexpr (ResidualWarps == 1) {
        __stcs(
            reinterpret_cast<uint4*>(layer_input + output_offset +
                                                       HiddenBlock / 2),
            packed_output1);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Extended big-fuse specialization: hidden=5120, hc_mult=4, n_splits in
// {1,4,16}.  Adds the delayed mHC pre contract (carried pre-mix in/out),
// fused RMSNorm and an optional plain-stream-mean aux output on top of the
// legacy control/collapse structure above.
//
// Numerics mirror the vLLM-metax reference `mhc_pre_big_fuse_with_norm_tilelang`
// (== MHC_PRE_NORM_KERNEL): when SavePreMix is set the collapsed value is
// rounded to BF16 *before* the RMSNorm square sum; otherwise the fp32
// accumulate feeds the square sum.  The final layer input is
// bf16(float(bf16(collapsed)) * inv_rms * float(norm_weight)).
// ---------------------------------------------------------------------------
template <int HiddenBlock, int ResidualWarps, int Hidden, bool FuseNorm,
          bool SavePreMix, int CachedTailBlocks = 0, bool WriteAux = true,
          bool DecoupleControl = false, int PreMixMode = -1,
          bool PrefetchCarriedChunk = false,
          bool ParallelNormEpilogue = false, bool BroadcastResidual = false,
          int QuantMode = 0>
__global__ void __launch_bounds__(64 * (1 + ResidualWarps), 1)
mhc_pre_big_fuse_ext_kernel(
    float* __restrict__ comb_mix, const float* __restrict__ gemm_out_mul,
    const float* __restrict__ gemm_out_sqrsum,
    bfloat16_t* __restrict__ layer_input, const float* __restrict__ mhc_base,
    const float* __restrict__ mhc_scale, float* __restrict__ post_mix,
    const bfloat16_t* __restrict__ residual, float* __restrict__ pre_mix_out,
    const float* __restrict__ pre_mix_in,
    const bfloat16_t* __restrict__ norm_weight,
    bfloat16_t* __restrict__ aux_out, int num_tokens, int n_splits,
    int gemm_last_dim, float rms_numel, float rms_eps, float hc_pre_eps,
    float hc_sinkhorn_eps, float hc_post_mult_value, float norm_eps,
    bfloat16_t* __restrict__ residual_out,
    int8_t* __restrict__ quant_output,
    float* __restrict__ quant_scales,
    bfloat16_t* __restrict__ output_rms,
    int8_t* __restrict__ output_quant_int8) {
  static_assert(QuantMode >= 0 && QuantMode <= 2,
                "quant mode must be BF16, INT8, or packed INT8");
  static_assert(QuantMode == 0 || (FuseNorm && CachedTailBlocks == 0 &&
                                   !ParallelNormEpilogue && !BroadcastResidual),
                "quantized output requires the ordinary fused norm path");
  static_assert(QuantMode == 0 || Hidden == 5120,
                "quantized output is specialized for hidden size 5120");
  static_assert(ResidualWarps == 1,
                "extended big-fuse currently supports one residual warp");
  static_assert(HiddenBlock == 64 * 8 * (ResidualWarps == 1 ? 2 : ResidualWarps),
                "hidden block must match the residual lane coverage");
  static_assert(Hidden % HiddenBlock == 0,
                "hidden must be a multiple of the hidden block");
  static_assert(CachedTailBlocks >= 0 && CachedTailBlocks <= 3,
                "tail caching supports zero to three hidden blocks");
  static_assert(CachedTailBlocks * HiddenBlock <= Hidden,
                "cached blocks must fit within hidden");
  static_assert(CachedTailBlocks == 0 || (FuseNorm && SavePreMix),
                "tail caching requires the delayed fused norm path");
  static_assert(WriteAux || BroadcastResidual ||
                    (FuseNorm && SavePreMix && CachedTailBlocks >= 2),
                "static no-aux requires the cached delayed norm path");
  static_assert(!DecoupleControl ||
                    (FuseNorm && SavePreMix && CachedTailBlocks >= 2 && !WriteAux),
                "control decoupling requires the static no-aux delayed norm path");
  static_assert(CachedTailBlocks != 3 ||
                    (Hidden == 5120 && DecoupleControl && PreMixMode == -1),
                "third cached block requires the dynamic delayed stream0 dispatch");
  static_assert(PreMixMode == -1 ||
                    ((PreMixMode == 0 || PreMixMode == 1) && DecoupleControl),
                "static input mode requires the decoupled delayed norm path");
  static_assert(!PrefetchCarriedChunk ||
                    (Hidden == 5120 && FuseNorm && SavePreMix &&
                     CachedTailBlocks == 2 && !WriteAux && DecoupleControl &&
                     PreMixMode == 1),
                "chunk prefetch requires the cached carried norm path");
  static_assert(!ParallelNormEpilogue ||
                    (Hidden == 5120 && FuseNorm && SavePreMix && !WriteAux &&
                     DecoupleControl &&
                     ((CachedTailBlocks == 2 && PreMixMode == 1 &&
                       PrefetchCarriedChunk) ||
                      (CachedTailBlocks == 3 && PreMixMode == -1 &&
                       !PrefetchCarriedChunk))),
                "parallel norm epilogue requires the carried N2048 cache2 or "
                "the stream0 N2048 cache3 scheduling");
  static_assert(!BroadcastResidual ||
                    (Hidden == 5120 && FuseNorm && !SavePreMix &&
                     CachedTailBlocks == 0 && !DecoupleControl &&
                     PreMixMode == -1 && !PrefetchCarriedChunk &&
                     !ParallelNormEpilogue),
                "broadcast residual requires the plain current-pre fused norm path");
  constexpr int kLanes = 64 * ResidualWarps;
  constexpr int kChunks = HiddenBlock / (kLanes * 8);
  static_assert(kLanes * 8 * kChunks == HiddenBlock,
                "hidden block must tile the residual lanes");
  // Compact only the C500 device images (xcore1000/xcore1008). Other
  // architectures retain the original layout until separately measured.
  // Keep this offset in sync with the host's device-dependent shared size.
#if defined(__MACA_ARCH__) && \
    ((__MACA_ARCH__ == 1000) || (__MACA_ARCH__ == 1008))
  constexpr int kCollapsedOffset = 128;
  constexpr int kCachedTailBlocks = CachedTailBlocks;
  constexpr bool kWriteAux = WriteAux;
  constexpr bool kDecoupleControl = DecoupleControl;
  constexpr int kPreMixMode = PreMixMode;
  constexpr bool kPrefetchCarriedChunk = PrefetchCarriedChunk;
  constexpr bool kParallelNormEpilogue = ParallelNormEpilogue;
  constexpr bool kBroadcastResidual = BroadcastResidual;
#else
  constexpr int kCollapsedOffset = 1024;
  constexpr int kCachedTailBlocks = 0;
  constexpr bool kWriteAux = BroadcastResidual ? false : true;
  constexpr bool kDecoupleControl = false;
  constexpr int kPreMixMode = -1;
  constexpr bool kPrefetchCarriedChunk = false;
  constexpr bool kParallelNormEpilogue = false;
  constexpr bool kBroadcastResidual = BroadcastResidual;
#endif
  constexpr bool kCacheNormTail = kCachedTailBlocks > 0;
  static_assert((8 + 24) * sizeof(float) <= kCollapsedOffset,
                "collapsed storage must not overlap the control values");
  static_assert(kCollapsedOffset % alignof(uint4) == 0,
                "collapsed vector accesses must remain aligned");
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float* sh_rms = reinterpret_cast<float*>(buf_dyn_shmem);
  float* sh_mix = sh_rms + 8;
  bfloat16_t* sh_collapsed =
      reinterpret_cast<bfloat16_t*>(buf_dyn_shmem + kCollapsedOffset);
  float* sh_norm_inv = reinterpret_cast<float*>(
      sh_collapsed + Hidden - kCachedTailBlocks * HiddenBlock);

  const int tid = static_cast<int>(threadIdx.x);
  const int token = static_cast<int>(blockIdx.x);

  // Stage 1: reduce the split-K GEMM partials (runtime n_splits) and the rms
  // scale for the mix. Fallback keeps the original CTA communication.
  if constexpr (!kDecoupleControl) {
    if (tid < 8) {
      float partial = 0.0f;
      for (int i_split = tid; i_split < n_splits; i_split += 8) {
        partial += gemm_out_sqrsum[static_cast<int64_t>(i_split) * num_tokens +
                                   token];
      }
      sh_rms[tid] = partial;
    }
    if (tid < 24) {
      float mix_partial = 0.0f;
      const float* mix_base =
          gemm_out_mul + static_cast<int64_t>(token) * gemm_last_dim + tid;
      const int64_t split_stride =
          static_cast<int64_t>(num_tokens) * gemm_last_dim;
      for (int i_split = 0; i_split < n_splits; ++i_split) {
        mix_partial += mix_base[static_cast<int64_t>(i_split) * split_stride];
      }
      sh_mix[tid] = mix_partial;
    }
    __syncthreads();
    if (tid == 0) {
      float total = 0.0f;
#pragma unroll
      for (int i_lane = 0; i_lane < 8; ++i_lane) {
        total += sh_rms[i_lane];
      }
      sh_rms[0] = rsqrtf(total / rms_numel + rms_eps);
    }
    __syncthreads();
  }
  // The delayed residual warp has no current-control dependency.
  const float shared_rms = kDecoupleControl ? 0.0f : sh_rms[0];
  uint4 parallel_cached_tail0{};
  uint4 parallel_cached_tail1{};
  uint4 parallel_cached_penultimate0{};
  uint4 parallel_cached_penultimate1{};
  // The stream0 cache3 dual-wave epilogue is emitted outside the residual-warp
  // scope, so its third cached block needs a payload that outlives that scope.
  // The serial cache3 epilogue keeps its own cached_third_last pair below.
  uint4 parallel_cached_third0{};
  uint4 parallel_cached_third1{};

  if (tid < 64) {
    if constexpr (kDecoupleControl) {
      // All 64 control lanes participate; header and collapsed storage are disjoint.
      if (tid < 8) {
        float partial = 0.0f;
        for (int i_split = tid; i_split < n_splits; i_split += 8) {
          partial += gemm_out_sqrsum[static_cast<int64_t>(i_split) * num_tokens +
                                     token];
        }
        sh_rms[tid] = partial;
      }
      if (tid < 24) {
        float mix_partial = 0.0f;
        const float* mix_base =
            gemm_out_mul + static_cast<int64_t>(token) * gemm_last_dim + tid;
        const int64_t split_stride =
            static_cast<int64_t>(num_tokens) * gemm_last_dim;
        for (int i_split = 0; i_split < n_splits; ++i_split) {
          mix_partial += mix_base[static_cast<int64_t>(i_split) * split_stride];
        }
        sh_mix[tid] = mix_partial;
      }
      __syncwarp(uint64_t(-1));
      if (tid == 0) {
        float total = 0.0f;
#pragma unroll
        for (int i_lane = 0; i_lane < 8; ++i_lane) {
          total += sh_rms[i_lane];
        }
        sh_rms[0] = rsqrtf(total / rms_numel + rms_eps);
      }
      __syncwarp(uint64_t(-1));
    }
    const float rms = kDecoupleControl ? sh_rms[0] : shared_rms;
    // Control warp: post mix, delayed next pre-mix and the Sinkhorn comb mix.
    if (tid < 4) {
      post_mix[static_cast<int64_t>(token) * 4 + tid] =
          (1.0f / (1.0f + expf(-(((sh_mix[tid + 4] * rms) * mhc_scale[1]) +
                                 mhc_base[tid + 4])))) *
          hc_post_mult_value;
      if constexpr (SavePreMix) {
        pre_mix_out[static_cast<int64_t>(token) * 4 + tid] =
            (1.0f / (1.0f + expf(-(((sh_mix[tid] * rms) * mhc_scale[0]) +
                                   mhc_base[tid])))) +
            hc_pre_eps;
      }
    }
    float cm = ((sh_mix[(tid & 15) + 8] * rms) * mhc_scale[2]) +
               mhc_base[(tid & 15) + 8];
    float row_max = -INFINITY;
    row_max = fmaxf(row_max, cm);
    row_max = mhc_row_max4(row_max);
    cm = expf(cm - row_max);
    float row_sum = mhc_row_sum4(cm);
    cm = mhc_sinkhorn_div<false, Hidden>(cm, row_sum) + hc_sinkhorn_eps;
    float col_sum = mhc_col_sum4(cm);
    cm = mhc_sinkhorn_div<false, Hidden>(cm, col_sum + hc_sinkhorn_eps);
#pragma unroll 19
    for (int i_step = 0; i_step < 19; ++i_step) {
      row_sum = mhc_row_sum4(cm);
      cm = mhc_sinkhorn_div<false, Hidden>(cm, row_sum + hc_sinkhorn_eps);
      col_sum = mhc_col_sum4(cm);
      cm = mhc_sinkhorn_div<false, Hidden>(cm, col_sum + hc_sinkhorn_eps);
    }
    if ((tid >> 4) == 0) {
      comb_mix[static_cast<int64_t>(token) * 16 + (tid & 15)] = cm;
    }
  } else {
    const float rms = shared_rms;
    const int lane = tid - 64;
    // -1 keeps the original runtime mode; C500 delayed-norm dispatch alone
    // specializes 0 (stream zero) or 1 (carried four-stream weights).
    const bool use_pre_mix_in =
        kPreMixMode < 0 ? pre_mix_in != nullptr : kPreMixMode == 1;
    const bool one_hot = SavePreMix && !use_pre_mix_in;
    float active_pre[4];
#pragma unroll
    for (int i_mhc = 0; i_mhc < 4; ++i_mhc) {
      if (use_pre_mix_in) {
        active_pre[i_mhc] =
            pre_mix_in[static_cast<int64_t>(token) * 4 + i_mhc];
      } else if constexpr (SavePreMix) {
        active_pre[i_mhc] = i_mhc == 0 ? 1.0f : 0.0f;
      } else {
        active_pre[i_mhc] =
            (1.0f / (1.0f + expf(-(((sh_mix[i_mhc] * rms) * mhc_scale[0]) +
                                   mhc_base[i_mhc])))) +
            hc_pre_eps;
      }
    }
    // Path A is a one-hot collapse onto a broadcast residual; when no aux
    // output is requested only stream zero has to be read.
    const int num_streams =
        (one_hot && (!kWriteAux || aux_out == nullptr)) ? 1 : 4;

    // Cache each tail block as two uint4 values per residual lane.
    // Named payloads avoid runtime-indexed arrays becoming private memory.
    uint4 cached_tail0{};
    uint4 cached_tail1{};
    uint4 cached_penultimate0{};
    uint4 cached_penultimate1{};
    uint4 cached_third_last0{};
    uint4 cached_third_last1{};
    float sumsq = 0.0f;
    // QuantMode uses max(abs(BF16(collapsed) * weight)) before applying the
    // positive per-token inv_rms.  This is mathematically equivalent to
    // reducing max(abs(norm)) but lets the final epilogue normalize and
    // quantize in one pass instead of round-tripping normalized BF16 through
    // shared memory.
    float weighted_amax = 0.0f;
#pragma unroll
    for (int i0_h = 0; i0_h < Hidden / HiddenBlock; ++i0_h) {
#pragma unroll
      for (int i_chunk = 0; i_chunk < kChunks; ++i_chunk) {
        const int hidden0 =
            i0_h * HiddenBlock + i_chunk * (kLanes * 8) + lane * 8;
        const int64_t residual_offset =
            static_cast<int64_t>(token) *
                (kBroadcastResidual ? Hidden : 4 * Hidden) +
            hidden0;
        float acc[8];
        float stream_sum[8];
#pragma unroll
        for (int i_elem = 0; i_elem < 8; ++i_elem) {
          acc[i_elem] = 0.0f;
          if constexpr (kWriteAux) {
            stream_sum[i_elem] = 0.0f;
          }
        }
        if constexpr (kBroadcastResidual) {
          const uint4 packed = __ldcs(
              reinterpret_cast<const uint4*>(residual + residual_offset));
          const __maca_bfloat162* values =
              reinterpret_cast<const __maca_bfloat162*>(&packed);
#pragma unroll
          for (int i_mhc = 0; i_mhc < 4; ++i_mhc) {
            __stcs(reinterpret_cast<uint4*>(
                       residual_out + static_cast<int64_t>(token) *
                                          (4 * Hidden) +
                       static_cast<int64_t>(i_mhc) * Hidden + hidden0),
                   packed);
#pragma unroll
            for (int i_pair = 0; i_pair < 4; ++i_pair) {
              const float2 v = __bfloat1622float2(values[i_pair]);
              acc[2 * i_pair] += active_pre[i_mhc] * v.x;
              acc[2 * i_pair + 1] += active_pre[i_mhc] * v.y;
              if constexpr (kWriteAux) {
                stream_sum[2 * i_pair] += v.x;
                stream_sum[2 * i_pair + 1] += v.y;
              }
            }
          }
        } else if constexpr (kPrefetchCarriedChunk) {
          // L0 loads four named vectors, then computes each BF16 pair with
          // the original stream0->3 accumulation order. No indexed cache.
          const uint4 packed0 = __ldcs(reinterpret_cast<const uint4*>(
              residual + residual_offset));
          const uint4 packed1 = __ldcs(reinterpret_cast<const uint4*>(
              residual + residual_offset + static_cast<int64_t>(Hidden)));
          const uint4 packed2 = __ldcs(reinterpret_cast<const uint4*>(
              residual + residual_offset + static_cast<int64_t>(2 * Hidden)));
          const uint4 packed3 = __ldcs(reinterpret_cast<const uint4*>(
              residual + residual_offset + static_cast<int64_t>(3 * Hidden)));
          const __maca_bfloat162* values0 =
              reinterpret_cast<const __maca_bfloat162*>(&packed0);
          const __maca_bfloat162* values1 =
              reinterpret_cast<const __maca_bfloat162*>(&packed1);
          const __maca_bfloat162* values2 =
              reinterpret_cast<const __maca_bfloat162*>(&packed2);
          const __maca_bfloat162* values3 =
              reinterpret_cast<const __maca_bfloat162*>(&packed3);
#pragma unroll
          for (int i_pair = 0; i_pair < 4; ++i_pair) {
            const float2 v0 = __bfloat1622float2(values0[i_pair]);
            const float2 v1 = __bfloat1622float2(values1[i_pair]);
            const float2 v2 = __bfloat1622float2(values2[i_pair]);
            const float2 v3 = __bfloat1622float2(values3[i_pair]);
            acc[2 * i_pair] += active_pre[0] * v0.x;
            acc[2 * i_pair + 1] += active_pre[0] * v0.y;
            acc[2 * i_pair] += active_pre[1] * v1.x;
            acc[2 * i_pair + 1] += active_pre[1] * v1.y;
            acc[2 * i_pair] += active_pre[2] * v2.x;
            acc[2 * i_pair + 1] += active_pre[2] * v2.y;
            acc[2 * i_pair] += active_pre[3] * v3.x;
            acc[2 * i_pair + 1] += active_pre[3] * v3.y;
          }
        } else {
#pragma unroll
          for (int i_mhc = 0; i_mhc < 4; ++i_mhc) {
            if (i_mhc < num_streams) {
              const uint4 packed =
                  __ldcs(reinterpret_cast<const uint4*>(
                      residual + residual_offset +
                      static_cast<int64_t>(i_mhc) * Hidden));
              const __maca_bfloat162* values =
                  reinterpret_cast<const __maca_bfloat162*>(&packed);
#pragma unroll
              for (int i_pair = 0; i_pair < 4; ++i_pair) {
                const float2 v = __bfloat1622float2(values[i_pair]);
                acc[2 * i_pair] += active_pre[i_mhc] * v.x;
                acc[2 * i_pair + 1] += active_pre[i_mhc] * v.y;
                if constexpr (kWriteAux) {
                  stream_sum[2 * i_pair] += v.x;
                  stream_sum[2 * i_pair + 1] += v.y;
                }
              }
            }
          }
        }
        uint4 packed_out;
        __maca_bfloat162* values_out =
            reinterpret_cast<__maca_bfloat162*>(&packed_out);
#pragma unroll
        for (int i_pair = 0; i_pair < 4; ++i_pair) {
          values_out[i_pair] = __float22bfloat162_rn(
              make_float2(acc[2 * i_pair], acc[2 * i_pair + 1]));
        }
        if constexpr (FuseNorm) {
          if constexpr (QuantMode != 0) {
            const uint4 packed_weight =
                *reinterpret_cast<const uint4*>(norm_weight + hidden0);
            const __maca_bfloat162* weight_values =
                reinterpret_cast<const __maca_bfloat162*>(&packed_weight);
#pragma unroll
            for (int i_pair = 0; i_pair < 4; ++i_pair) {
              const float2 rounded = __bfloat1622float2(values_out[i_pair]);
              const float2 weight =
                  __bfloat1622float2(weight_values[i_pair]);
              weighted_amax =
                  fmaxf(weighted_amax, fabsf(rounded.x * weight.x));
              weighted_amax =
                  fmaxf(weighted_amax, fabsf(rounded.y * weight.y));
            }
          }
          if constexpr (kCacheNormTail) {
            if (i0_h == Hidden / HiddenBlock - 1) {
              if (i_chunk == 0) {
                if constexpr (kParallelNormEpilogue) {
                  parallel_cached_tail0 = packed_out;
                } else {
                  cached_tail0 = packed_out;
                }
              } else {
                if constexpr (kParallelNormEpilogue) {
                  parallel_cached_tail1 = packed_out;
                } else {
                  cached_tail1 = packed_out;
                }
              }
            } else if (kCachedTailBlocks >= 2 &&
                       i0_h == Hidden / HiddenBlock - 2) {
              if (i_chunk == 0) {
                if constexpr (kParallelNormEpilogue) {
                  parallel_cached_penultimate0 = packed_out;
                } else {
                  cached_penultimate0 = packed_out;
                }
              } else {
                if constexpr (kParallelNormEpilogue) {
                  parallel_cached_penultimate1 = packed_out;
                } else {
                  cached_penultimate1 = packed_out;
                }
              }
            } else if (kCachedTailBlocks == 3 &&
                       i0_h == Hidden / HiddenBlock - 3) {
              if (i_chunk == 0) {
                if constexpr (kParallelNormEpilogue) {
                  parallel_cached_third0 = packed_out;
                } else {
                  cached_third_last0 = packed_out;
                }
              } else {
                if constexpr (kParallelNormEpilogue) {
                  parallel_cached_third1 = packed_out;
                } else {
                  cached_third_last1 = packed_out;
                }
              }
            } else {
              *reinterpret_cast<uint4*>(sh_collapsed + hidden0) = packed_out;
            }
          } else {
            *reinterpret_cast<uint4*>(sh_collapsed + hidden0) = packed_out;
          }
#pragma unroll
          for (int i_pair = 0; i_pair < 4; ++i_pair) {
            if constexpr (SavePreMix) {
              const float2 rounded = __bfloat1622float2(values_out[i_pair]);
              sumsq += rounded.x * rounded.x + rounded.y * rounded.y;
            } else {
              sumsq += acc[2 * i_pair] * acc[2 * i_pair] +
                       acc[2 * i_pair + 1] * acc[2 * i_pair + 1];
            }
          }
        } else {
          __stcs(reinterpret_cast<uint4*>(layer_input +
                                          static_cast<int64_t>(token) *
                                              Hidden +
                                          hidden0),
                 packed_out);
        }
        if constexpr (kWriteAux) {
          if (aux_out != nullptr) {
            uint4 packed_aux;
            __maca_bfloat162* values_aux =
                reinterpret_cast<__maca_bfloat162*>(&packed_aux);
#pragma unroll
            for (int i_pair = 0; i_pair < 4; ++i_pair) {
              values_aux[i_pair] = __float22bfloat162_rn(
                  make_float2(stream_sum[2 * i_pair] * 0.25f,
                              stream_sum[2 * i_pair + 1] * 0.25f));
            }
            __stcs(reinterpret_cast<uint4*>(aux_out +
                                            static_cast<int64_t>(token) *
                                                Hidden +
                                            hidden0),
                   packed_aux);
          }
        }
      }
    }

    if constexpr (FuseNorm) {
#pragma unroll
      for (int i_offset = 32; i_offset > 0; i_offset >>= 1) {
        sumsq += __shfl_xor_sync(uint64_t(-1), sumsq, i_offset);
        if constexpr (QuantMode != 0) {
          weighted_amax = fmaxf(
              weighted_amax,
              __shfl_xor_sync(uint64_t(-1), weighted_amax, i_offset));
        }
      }
      const float inv_rms =
          rsqrtf(sumsq / static_cast<float>(Hidden) + norm_eps);
      if constexpr (kParallelNormEpilogue) {
        if (lane == 0) {
          *sh_norm_inv = inv_rms;
        }
      } else if constexpr (QuantMode != 0) {
        // The nonzero floor handles an all-zero token without dividing by 0.
        const float scale =
            fmaxf(weighted_amax * inv_rms / 127.0f, 1.1920929e-7f);
        const float inv_scale = 1.0f / scale;
        constexpr int kPackedRowBytes = ((Hidden + 4 + 511) / 512) * 512;
        const int64_t row_bytes =
            QuantMode == 2 ? kPackedRowBytes : Hidden;
        int8_t* row_output = quant_output + static_cast<int64_t>(token) * row_bytes;
        if (lane == 0) {
          quant_scales[token] = scale;
          if constexpr (QuantMode == 2) {
            *reinterpret_cast<float*>(row_output + Hidden) = scale;
          }
        }
        for (int hidden0 = lane * 8; hidden0 < Hidden; hidden0 += kLanes * 8) {
          const uint4 packed_collapsed =
              *reinterpret_cast<const uint4*>(sh_collapsed + hidden0);
          const uint4 packed_weight =
              *reinterpret_cast<const uint4*>(norm_weight + hidden0);
          const __maca_bfloat162* collapsed_values =
              reinterpret_cast<const __maca_bfloat162*>(&packed_collapsed);
          const __maca_bfloat162* weight_values =
              reinterpret_cast<const __maca_bfloat162*>(&packed_weight);
          uint64_t packed_int8 = 0;
          uint4 packed_norm_out;
          __maca_bfloat162* norm_values =
              reinterpret_cast<__maca_bfloat162*>(&packed_norm_out);
#pragma unroll
          for (int i_pair = 0; i_pair < 4; ++i_pair) {
            const float2 collapsed =
                __bfloat1622float2(collapsed_values[i_pair]);
            const float2 weight =
                __bfloat1622float2(weight_values[i_pair]);
            const __maca_bfloat162 norm_pair = __float22bfloat162_rn(
                make_float2(collapsed.x * inv_rms * weight.x,
                            collapsed.y * inv_rms * weight.y));
            if constexpr (QuantMode != 0) {
              norm_values[i_pair] = norm_pair;
            }
            const float2 value = __bfloat1622float2(norm_pair);
            int q0 = __float2int_rn(value.x * inv_scale);
            int q1 = __float2int_rn(value.y * inv_scale);
            q0 = max(-127, min(127, q0));
            q1 = max(-127, min(127, q1));
            packed_int8 |= static_cast<uint64_t>(static_cast<uint8_t>(q0))
                           << (16 * i_pair);
            packed_int8 |= static_cast<uint64_t>(static_cast<uint8_t>(q1))
                           << (16 * i_pair + 8);
          }
          *reinterpret_cast<uint64_t*>(row_output + hidden0) = packed_int8;
          if constexpr (QuantMode != 0) {
            __stcs(reinterpret_cast<uint4*>(
                       output_rms + static_cast<int64_t>(token) * Hidden +
                       hidden0),
                   packed_norm_out);
          }
          if constexpr (QuantMode == 2) {
            *reinterpret_cast<uint64_t*>(
                output_quant_int8 + static_cast<int64_t>(token) * Hidden +
                hidden0) = packed_int8;
          }
        }
        if constexpr (QuantMode == 2) {
          // The embedded scale ends at byte Hidden+4.  Clear the 12-byte
          // prefix up to the next uint4 boundary, then use 31 aligned vector
          // stores for the remaining 496 padding bytes.
          if (lane < 12) {
            row_output[Hidden + 4 + lane] = 0;
          }
          if (lane < 31) {
            *reinterpret_cast<uint4*>(row_output + Hidden + 16 + lane * 16) =
                uint4{};
          }
        }
      } else {
#pragma unroll
        for (int i0_h = 0; i0_h < Hidden / HiddenBlock; ++i0_h) {
#pragma unroll
          for (int i_chunk = 0; i_chunk < kChunks; ++i_chunk) {
            const int hidden0 =
                i0_h * HiddenBlock + i_chunk * (kLanes * 8) + lane * 8;
            uint4 packed_collapsed;
            if constexpr (kCacheNormTail) {
              if (i0_h == Hidden / HiddenBlock - 1) {
                packed_collapsed = i_chunk == 0 ? cached_tail0 : cached_tail1;
              } else if (kCachedTailBlocks >= 2 &&
                         i0_h == Hidden / HiddenBlock - 2) {
                packed_collapsed =
                    i_chunk == 0 ? cached_penultimate0 : cached_penultimate1;
              } else if (kCachedTailBlocks == 3 &&
                         i0_h == Hidden / HiddenBlock - 3) {
                packed_collapsed =
                    i_chunk == 0 ? cached_third_last0 : cached_third_last1;
              } else {
                packed_collapsed =
                    *reinterpret_cast<const uint4*>(sh_collapsed + hidden0);
              }
            } else {
              packed_collapsed =
                  *reinterpret_cast<const uint4*>(sh_collapsed + hidden0);
            }
            const uint4 packed_weight =
                *reinterpret_cast<const uint4*>(norm_weight + hidden0);
            const __maca_bfloat162* collapsed_values =
                reinterpret_cast<const __maca_bfloat162*>(&packed_collapsed);
            const __maca_bfloat162* weight_values =
                reinterpret_cast<const __maca_bfloat162*>(&packed_weight);
            uint4 packed_norm_out;
            __maca_bfloat162* norm_values =
                reinterpret_cast<__maca_bfloat162*>(&packed_norm_out);
#pragma unroll
            for (int i_pair = 0; i_pair < 4; ++i_pair) {
              const float2 collapsed =
                  __bfloat1622float2(collapsed_values[i_pair]);
              const float2 weight = __bfloat1622float2(weight_values[i_pair]);
              norm_values[i_pair] = __float22bfloat162_rn(
                  make_float2(collapsed.x * inv_rms * weight.x,
                              collapsed.y * inv_rms * weight.y));
            }
            __stcs(reinterpret_cast<uint4*>(layer_input +
                                            static_cast<int64_t>(token) *
                                                Hidden +
                                            hidden0),
                   packed_norm_out);
          }
        }
      }
    }
  }

  if constexpr (kParallelNormEpilogue && kCachedTailBlocks == 3) {
    // Stream0 N2048 instance: both waves rendezvous after control/Sinkhorn and
    // residual/RMS work. The control wave emits shared blocks 0..1 while the
    // residual wave emits the three register-cached blocks 2..4. Block 2 keeps
    // its own parallel pair so the registers the serial cache3 instance
    // reserves stay untouched there; blocks 3..4 reuse the parallel cached
    // pairs, so per-element math and the emitted addresses are unchanged.
    __syncthreads();
    const float inv_rms = *sh_norm_inv;
    const int epilogue_lane = tid & 63;
    if (tid < 64) {
#pragma unroll
      for (int i0_h = 0; i0_h < Hidden / HiddenBlock - kCachedTailBlocks;
           ++i0_h) {
#pragma unroll
        for (int i_chunk = 0; i_chunk < kChunks; ++i_chunk) {
          const int hidden0 = i0_h * HiddenBlock +
                              i_chunk * (kLanes * 8) + epilogue_lane * 8;
          const uint4 packed_collapsed =
              *reinterpret_cast<const uint4*>(sh_collapsed + hidden0);
          const uint4 packed_weight =
              *reinterpret_cast<const uint4*>(norm_weight + hidden0);
          const __maca_bfloat162* collapsed_values =
              reinterpret_cast<const __maca_bfloat162*>(&packed_collapsed);
          const __maca_bfloat162* weight_values =
              reinterpret_cast<const __maca_bfloat162*>(&packed_weight);
          uint4 packed_norm_out;
          __maca_bfloat162* norm_values =
              reinterpret_cast<__maca_bfloat162*>(&packed_norm_out);
#pragma unroll
          for (int i_pair = 0; i_pair < 4; ++i_pair) {
            const float2 collapsed =
                __bfloat1622float2(collapsed_values[i_pair]);
            const float2 weight = __bfloat1622float2(weight_values[i_pair]);
            norm_values[i_pair] = __float22bfloat162_rn(
                make_float2(collapsed.x * inv_rms * weight.x,
                            collapsed.y * inv_rms * weight.y));
          }
          __stcs(reinterpret_cast<uint4*>(layer_input +
                                          static_cast<int64_t>(token) * Hidden +
                                          hidden0),
                 packed_norm_out);
        }
      }
    } else {
#pragma unroll
      for (int i0_h = Hidden / HiddenBlock - kCachedTailBlocks;
           i0_h < Hidden / HiddenBlock; ++i0_h) {
#pragma unroll
        for (int i_chunk = 0; i_chunk < kChunks; ++i_chunk) {
          const int hidden0 = i0_h * HiddenBlock +
                              i_chunk * (kLanes * 8) + epilogue_lane * 8;
          const uint4 packed_collapsed =
              i0_h == Hidden / HiddenBlock - 1
                  ? (i_chunk == 0 ? parallel_cached_tail0
                                  : parallel_cached_tail1)
                  : (i0_h == Hidden / HiddenBlock - 2
                         ? (i_chunk == 0 ? parallel_cached_penultimate0
                                         : parallel_cached_penultimate1)
                         : (i_chunk == 0 ? parallel_cached_third0
                                         : parallel_cached_third1));
          const uint4 packed_weight =
              *reinterpret_cast<const uint4*>(norm_weight + hidden0);
          const __maca_bfloat162* collapsed_values =
              reinterpret_cast<const __maca_bfloat162*>(&packed_collapsed);
          const __maca_bfloat162* weight_values =
              reinterpret_cast<const __maca_bfloat162*>(&packed_weight);
          uint4 packed_norm_out;
          __maca_bfloat162* norm_values =
              reinterpret_cast<__maca_bfloat162*>(&packed_norm_out);
#pragma unroll
          for (int i_pair = 0; i_pair < 4; ++i_pair) {
            const float2 collapsed =
                __bfloat1622float2(collapsed_values[i_pair]);
            const float2 weight = __bfloat1622float2(weight_values[i_pair]);
            norm_values[i_pair] = __float22bfloat162_rn(
                make_float2(collapsed.x * inv_rms * weight.x,
                            collapsed.y * inv_rms * weight.y));
          }
          __stcs(reinterpret_cast<uint4*>(layer_input +
                                          static_cast<int64_t>(token) * Hidden +
                                          hidden0),
                 packed_norm_out);
        }
      }
    }
  } else if constexpr (kParallelNormEpilogue) {
    // Both waves rendezvous after control/Sinkhorn and residual/RMS work.
    // The control wave emits shared blocks 0..2 while the residual wave emits
    // its two cached blocks 3..4; per-element math is unchanged.
    __syncthreads();
    const float inv_rms = *sh_norm_inv;
    const int epilogue_lane = tid & 63;
    if (tid < 64) {
#pragma unroll
      for (int i0_h = 0; i0_h < 3; ++i0_h) {
#pragma unroll
        for (int i_chunk = 0; i_chunk < kChunks; ++i_chunk) {
          const int hidden0 = i0_h * HiddenBlock +
                              i_chunk * (kLanes * 8) + epilogue_lane * 8;
          const uint4 packed_collapsed =
              *reinterpret_cast<const uint4*>(sh_collapsed + hidden0);
          const uint4 packed_weight =
              *reinterpret_cast<const uint4*>(norm_weight + hidden0);
          const __maca_bfloat162* collapsed_values =
              reinterpret_cast<const __maca_bfloat162*>(&packed_collapsed);
          const __maca_bfloat162* weight_values =
              reinterpret_cast<const __maca_bfloat162*>(&packed_weight);
          uint4 packed_norm_out;
          __maca_bfloat162* norm_values =
              reinterpret_cast<__maca_bfloat162*>(&packed_norm_out);
#pragma unroll
          for (int i_pair = 0; i_pair < 4; ++i_pair) {
            const float2 collapsed =
                __bfloat1622float2(collapsed_values[i_pair]);
            const float2 weight = __bfloat1622float2(weight_values[i_pair]);
            norm_values[i_pair] = __float22bfloat162_rn(
                make_float2(collapsed.x * inv_rms * weight.x,
                            collapsed.y * inv_rms * weight.y));
          }
          __stcs(reinterpret_cast<uint4*>(layer_input +
                                          static_cast<int64_t>(token) * Hidden +
                                          hidden0),
                 packed_norm_out);
        }
      }
    } else {
#pragma unroll
      for (int i0_h = 3; i0_h < 5; ++i0_h) {
#pragma unroll
        for (int i_chunk = 0; i_chunk < kChunks; ++i_chunk) {
          const int hidden0 = i0_h * HiddenBlock +
                              i_chunk * (kLanes * 8) + epilogue_lane * 8;
          const uint4 packed_collapsed =
              i0_h == 4
                  ? (i_chunk == 0 ? parallel_cached_tail0
                                  : parallel_cached_tail1)
                  : (i_chunk == 0 ? parallel_cached_penultimate0
                                  : parallel_cached_penultimate1);
          const uint4 packed_weight =
              *reinterpret_cast<const uint4*>(norm_weight + hidden0);
          const __maca_bfloat162* collapsed_values =
              reinterpret_cast<const __maca_bfloat162*>(&packed_collapsed);
          const __maca_bfloat162* weight_values =
              reinterpret_cast<const __maca_bfloat162*>(&packed_weight);
          uint4 packed_norm_out;
          __maca_bfloat162* norm_values =
              reinterpret_cast<__maca_bfloat162*>(&packed_norm_out);
#pragma unroll
          for (int i_pair = 0; i_pair < 4; ++i_pair) {
            const float2 collapsed =
                __bfloat1622float2(collapsed_values[i_pair]);
            const float2 weight = __bfloat1622float2(weight_values[i_pair]);
            norm_values[i_pair] = __float22bfloat162_rn(
                make_float2(collapsed.x * inv_rms * weight.x,
                            collapsed.y * inv_rms * weight.y));
          }
          __stcs(reinterpret_cast<uint4*>(layer_input +
                                          static_cast<int64_t>(token) * Hidden +
                                          hidden0),
                 packed_norm_out);
        }
      }
    }
  }
}
