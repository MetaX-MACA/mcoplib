#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include "../kernel/utils.h"

// ============================================================================
// scale_dynamic_quant : per-token bf16 -> int8 dynamic (symmetric) quant with a
// per-channel smooth scale.  For each token t and channel c:
//     v_c        = hidden[t,c] * smooth_scale[c]
//     absmax     = max_c |v_c|
//     scale[t]   = absmax / 127
//     out[t,c]   = round(v_c * 127 / absmax)   (saturated to [-128,127])
//
// This kernel is strictly MEMORY BOUND: DRAM traffic ~= tokens*hidden*(2+1) B
// (read bf16 + write int8; smooth_scale is shared across tokens -> L2 resident).
//
// C600-U key params (warp=64, up to 32 warps/AP, 128 KB SMEM, 255 reg/thread,
// mpc=28 APs; single-die HBM copy ceiling ~1.26 TB/s measured):
//   * vectorize global access to 16 B/thread (float4 = 8 bf16)  -> full xactions
//   * SINGLE global read of hidden: keep the smooth-scaled values register
//     resident across the max-reduce and the quant-write (no 2nd DRAM read).
//   * REDUCTION: the block-wide __syncthreads tree is the #1 serial bottleneck
//     on this HW.  We use the vendor pattern instead: a 16-lane SUBGROUP shuffle
//     (the native fast path -- __shfl_down_sync_16, offsets 8/4/2/1), one value
//     per 64-wide warp, then a SINGLE __syncthreads to merge the few warp maxes
//     via shared memory.  Measured +8% over the tree at (T=32768,H=8192).
// ============================================================================

#ifndef FULL_MASK64
#define FULL_MASK64 0xffffffffffffffffULL
#endif

__host__ __device__ __forceinline__
int ceil_div_i(int x, int y) { return (x + y - 1) / y; }

// Saturating round scaled float -> int8.  Matches the reference clamp [-128,127].
static __device__ __forceinline__ int8_t quant_i8(float x) {
    int q = __float2int_rn(x);
    q = min(127, max(-128, q));
    return static_cast<int8_t>(q);
}

// ---------------------------------------------------------------------------
// Output-dtype traits.  The kernel is written once over the quant target T2;
// only the symmetric quant ceiling QMAX (and its reciprocal) differ:
//   int8         : QMAX = 127   (finite range [-127,127], the ref clamps -128 too)
//   float8_e4m3fn: QMAX = 448   (e4m3fn finite max)
// scale[token] = absmax / QMAX ,  inv_scale = QMAX / absmax .  Mathematically
// identical dynamic symmetric quant for both types -- ONLY the ceiling changes.
// ---------------------------------------------------------------------------
template <typename out_t> struct QuantTraits;
template <> struct QuantTraits<int8_t> {
    static constexpr float QMAX     = 127.0f;
    static constexpr float INV_QMAX = 0.00787401574803f;   // 1/127
};
template <> struct QuantTraits<c10::Float8_e4m3fn> {
    static constexpr float QMAX     = 448.0f;
    static constexpr float INV_QMAX = 0.00223214285714f;   // 1/448
};

// Vectorized quant-store of VEC pre-scaled fp32 values -> VEC output bytes,
// written as ONE aligned vector store.  int8 uses the scalar HW round+clamp;
// fp8_e4m3 uses the 4-wide HW pack __builtin_mxc_cvt_pk4_f32tof8 (arch 1600 =
// C600-U), 1 instruction per 4 elements.  VEC is a multiple of 4 (VPT=8/16).
// Function templates cannot be partially specialized, so int8 vs fp8 selection
// is done with `if constexpr` inside one primary template.
template <typename out_t, int VEC>
static __device__ __forceinline__ void quant_store_vec(out_t* dst, const float* v) {
  if constexpr (std::is_same_v<out_t, int8_t>) {
    AlignedArrayI4<int8_t, VEC> ov;
#pragma unroll
    for (int j = 0; j < VEC; ++j) ov.data[j] = quant_i8(v[j]);
    *reinterpret_cast<AlignedArrayI4<int8_t, VEC>*>(dst) = ov;
  } else {
#if defined(__MACA_ARCH__) && (__MACA_ARCH__ >= 1600) && (__MACA_ARCH__ < 1700)
    using v4f32 = float __attribute__((ext_vector_type(4)));
    AlignedArrayI4<uint32_t, VEC / 4> packed;
#pragma unroll
    for (int g = 0; g < VEC / 4; ++g) {
      v4f32 a = {v[g * 4 + 0], v[g * 4 + 1], v[g * 4 + 2], v[g * 4 + 3]};
      packed.data[g] = __builtin_mxc_cvt_pk4_f32tof8(a);   // 4x f32 -> 4x e4m3
    }
    *reinterpret_cast<AlignedArrayI4<uint32_t, VEC / 4>*>(dst) = packed;
#else
    AlignedArrayI4<out_t, VEC> ov;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      float x = fmaxf(-448.0f, fminf(v[j], 448.0f));
      ov.data[j] = static_cast<out_t>(x);
    }
    *reinterpret_cast<AlignedArrayI4<out_t, VEC>*>(dst) = ov;
#endif
  }
}

// Max within a 64-wide warp using the native 16-lane subgroup shuffle fast path
// (offsets 8..1), then merge the four 16-lane groups by broadcast reads.  Result
// is valid in EVERY lane of the warp.  No shared memory, no __syncthreads.
static __device__ __forceinline__ float warpReduceMax64(float v) {
    #pragma unroll
    for (int i = 8; i >= 1; i >>= 1)
        v = fmaxf(v, __shfl_down_sync_16(FULL_MASK64, v, i));
    // group maxes are now held at lanes 0,16,32,48; broadcast+merge.
    float g0 = __shfl_sync(FULL_MASK64, v,  0, 64);
    float g1 = __shfl_sync(FULL_MASK64, v, 16, 64);
    float g2 = __shfl_sync(FULL_MASK64, v, 32, 64);
    float g3 = __shfl_sync(FULL_MASK64, v, 48, 64);
    return fmaxf(fmaxf(g0, g1), fmaxf(g2, g3));
}

// Block-wide max via warp-shuffle + a single cross-warp merge.  One __syncthreads
// total (vs log2(NUM_THREADS) for the tree).  Result broadcast to all threads.
template<int NUM_THREADS>
static __device__ __forceinline__ float blockReduceMax(float val, float* sm) {
    constexpr int WARPS = NUM_THREADS / 64;
    const int tid = threadIdx.x;
    float w = warpReduceMax64(val);
    if ((tid & 63) == 0) sm[tid >> 6] = w;
    __syncthreads();
    float m = sm[0];
    #pragma unroll
    for (int i = 1; i < WARPS; ++i) m = fmaxf(m, sm[i]);
    return m;
}

// Register-resident single-pass kernel.
//   NUM_THREADS : block size (one block == one token row)
//   NUM_REG     : register tiles per thread (covers NUM_THREADS*NUM_REG*VPT cols)
//   VPT         : bf16 elements per vector access (8 -> float4 load / int64 store)
template<typename T1, typename T2, int NUM_THREADS, int NUM_REG, int VPT>
__global__ __launch_bounds__(NUM_THREADS)
void scale_dynamic_quant_reg(const T1* __restrict__ hidden_states,
                             const float* __restrict__ smooth_scales,
                             const int hidden_size,
                             T2* __restrict__ output,
                             float* __restrict__ scales) {
    using VecIn  = AlignedArrayI4<T1, VPT>;      // VPT*2 B  (float4 for VPT=8)

    const int token = blockIdx.x;
    const int tid   = threadIdx.x;
    const int stride = NUM_THREADS * VPT;

    const T1* in  = hidden_states + static_cast<size_t>(token) * hidden_size;
    T2*       out = output        + static_cast<size_t>(token) * hidden_size;

    // Pass 1: vectorized load hidden, apply per-channel smooth, keep the scaled
    // fp32 values in registers, track the per-thread absmax.
    float vreg[NUM_REG][VPT];
    float local_max = 0.0f;

    #pragma unroll
    for (int k = 0; k < NUM_REG; ++k) {
        const int base = tid * VPT + k * stride;
        if (base < hidden_size) {
            VecIn hv = *reinterpret_cast<const VecIn*>(in + base);
            // smooth_scales is fp32 and shared across tokens (L2 resident);
            // read VPT floats via two float4 loads.
            #pragma unroll
            for (int q = 0; q < VPT; q += 4) {
                float4 s4 = *reinterpret_cast<const float4*>(smooth_scales + base + q);
                float sm[4] = {s4.x, s4.y, s4.z, s4.w};
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float v = __bfloat162float(hv.data[q + j]) * sm[j];
                    vreg[k][q + j] = v;
                    local_max = fmaxf(local_max, fabsf(v));
                }
            }
        }
    }

    __shared__ float sred[NUM_THREADS / 64];
    float absmax = blockReduceMax<NUM_THREADS>(local_max, sred);

    if (tid == 0) {
        scales[token] = absmax * QuantTraits<T2>::INV_QMAX;  // absmax / QMAX
    }

    // inv_scale = QMAX/absmax; guard the all-zero token (-> emit zeros cleanly).
    const float inv_scale = absmax > 0.0f ? QuantTraits<T2>::QMAX * __builtin_mxc_rcpf(absmax) : 0.0f;

    // Pass 2: quantize straight from registers (NO second global read of hidden).
    #pragma unroll
    for (int k = 0; k < NUM_REG; ++k) {
        const int base = tid * VPT + k * stride;
        if (base < hidden_size) {
            float sv[VPT];
            #pragma unroll
            for (int j = 0; j < VPT; ++j) sv[j] = vreg[k][j] * inv_scale;
            quant_store_vec<T2, VPT>(out + base, sv);
        }
    }
}

// ============================================================================
// WARP-PER-TOKEN kernel.  Each 64-lane warp owns EXACTLY ONE token row.  The
// per-row abs-max reduce is a pure in-warp shuffle (warpReduceMax64) -- there is
// NO shared memory and NO __syncthreads at all.  This is the strongest possible
// structure for this op when a single warp can span the row: every warp is fully
// independent, so nothing ever serializes on a block barrier (the code's own
// cost breakdown identified that barrier as the #1 residual cost at large H).
//
// Applicable when H is a multiple of 64*VPT (=1024 for VPT=16): the warp's 64
// lanes each own NR contiguous-strided VPT-wide tiles, covering H = 64*VPT*NR.
//   H=1024 -> NR=1 (one tile/lane)      H=4096 -> NR=4 (four tiles/lane)
// WPB warps share a block only for launch/occupancy; they never communicate.
// ============================================================================
template<typename T1, typename T2, int WPB, int VPT, int NR>
__global__ __launch_bounds__(WPB * 64)
void scale_dynamic_quant_warprow(const T1* __restrict__ hidden_states,
                                 const float* __restrict__ smooth_scales,
                                 const int hidden_size,
                                 const int token_num,
                                 T2* __restrict__ output,
                                 float* __restrict__ scales) {
    using VecIn = AlignedArrayI4<T1, VPT>;
    const int lane  = threadIdx.x & 63;
    const int token = blockIdx.x * WPB + (threadIdx.x >> 6);
    if (token >= token_num) return;

    constexpr int warp_stride = 64 * VPT;   // cols advanced per tile inside the warp
    const T1* in  = hidden_states + static_cast<size_t>(token) * hidden_size;
    T2*       out = output        + static_cast<size_t>(token) * hidden_size;

    float vreg[NR][VPT];
    float local_max = 0.0f;
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
        const int base = lane * VPT + r * warp_stride;
        if (base < hidden_size) {
            VecIn hv = *reinterpret_cast<const VecIn*>(in + base);
            #pragma unroll
            for (int q = 0; q < VPT; q += 4) {
                float4 s4 = *reinterpret_cast<const float4*>(smooth_scales + base + q);
                float sm[4] = {s4.x, s4.y, s4.z, s4.w};
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float v = __bfloat162float(hv.data[q + j]) * sm[j];
                    vreg[r][q + j] = v;
                    local_max = fmaxf(local_max, fabsf(v));
                }
            }
        }
    }

    float absmax = warpReduceMax64(local_max);
    if (lane == 0) scales[token] = absmax * QuantTraits<T2>::INV_QMAX;
    const float inv_scale = absmax > 0.0f ? QuantTraits<T2>::QMAX * __builtin_mxc_rcpf(absmax) : 0.0f;

    #pragma unroll
    for (int r = 0; r < NR; ++r) {
        const int base = lane * VPT + r * warp_stride;
        if (base < hidden_size) {
            float sv[VPT];
            #pragma unroll
            for (int j = 0; j < VPT; ++j) sv[j] = vreg[r][j] * inv_scale;
            quant_store_vec<T2, VPT>(out + base, sv);
        }
    }
}

// ============================================================================
// Multi-token register-resident kernel.  Each block owns a fixed set of columns
// (single tile, NUM_REG==1) and processes M consecutive token rows.  The key
// win: the per-channel smooth_scales for this block's columns are loaded ONCE
// into registers and REUSED across all M tokens.  Profiling the single-token
// kernel showed the smooth read (H fp32 = 2x the hidden bf16 footprint, re-read
// per token from L2) was the single largest cost after the raw copy -- larger
// than the reduction barrier at H>=7168.  Amortizing it by M cuts that traffic
// M-fold.  smooth stays fp32 in registers -> NO precision change vs the caller.
//   NUM_THREADS : block size (== ceil(H/VPT) rounded to a warp multiple)
//   VPT         : bf16 elements per vector access (16 -> two float4 loads)
//   M           : token rows handled per block (grid = ceil(token_num / M))
// Measured (T=32768) vs the single-token VPT=16 path:
//   H=4096  M=2 : 700 -> 734    H=7168 M=4 : 651 -> 717    H=8192 M=4 : 705 -> 801
// ============================================================================
template<typename T1, typename T2, int NUM_THREADS, int VPT, int NR, int M, bool PIPE = false>
__global__ __launch_bounds__(NUM_THREADS)
void scale_dynamic_quant_mtok(const T1* __restrict__ hidden_states,
                              const float* __restrict__ smooth_scales,
                              const int hidden_size,
                              const int token_num,
                              T2* __restrict__ output,
                              float* __restrict__ scales) {
    using VecIn  = AlignedArrayI4<T1, VPT>;

    const int tid    = threadIdx.x;
    const int stride = NUM_THREADS * VPT;   // columns advanced per register tile

    // Load this thread's NR*VPT smooth values ONCE (fp32, exact), reuse across M.
    // NR>1 lets one block cover a wide row with FEWER warps -> a cheaper cross-
    // warp reduce (the dominant cost at large H, per the cost breakdown).
    float smv[NR][VPT];
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
        const int base = tid * VPT + r * stride;
        if (base < hidden_size) {
            #pragma unroll
            for (int q = 0; q < VPT; q += 4) {
                float4 s4 = *reinterpret_cast<const float4*>(smooth_scales + base + q);
                smv[r][q] = s4.x; smv[r][q + 1] = s4.y; smv[r][q + 2] = s4.z; smv[r][q + 3] = s4.w;
            }
        }
    }

    // Double-buffered cross-warp scratch: iteration m writes/reads sred[m&1].
    // The single barrier inside the reduce of iteration m+1 guarantees every
    // thread has finished reading sred[m&1] before iteration m+2 (same parity)
    // overwrites it -- so the trailing __syncthreads() per token is REMOVED.
    // At large H the barrier is the dominant residual cost (bare-copy 1243 vs
    // full 800 GB/s); halving the per-token barrier count lifts H>=6144 ~4-7%.
    constexpr int WARPS = NUM_THREADS / 64;
    __shared__ float sred[2][WARPS];
    const int t0 = blockIdx.x * M;

    // Two M-loop bodies selected at compile time by PIPE.
    // Software pipelining hides the cross-warp reduce barrier (H=8192: no-reduce
    // 995 vs full 789 GB/s) by issuing token m+1's HBM loads BEFORE token m's
    // barrier, so the next row's read latency overlaps the barrier stall.  But
    // the extra pref/pnext register buffers cut occupancy, so it only wins where
    // the barrier truly dominates (wide rows, NT=256/NR=2 -> H>=7168).  Narrower
    // rows keep the leaner non-pipelined body (PIPE=false).
    if constexpr (PIPE) {
        VecIn pref[NR];
        #pragma unroll
        for (int r = 0; r < NR; ++r) {
            const int base = tid * VPT + r * stride;
            if (base < hidden_size)
                pref[r] = *reinterpret_cast<const VecIn*>(
                    hidden_states + static_cast<size_t>(t0) * hidden_size + base);
        }

        #pragma unroll
        for (int m = 0; m < M; ++m) {
            const int token = t0 + m;
            if (token >= token_num) break;
            T2* out = output + static_cast<size_t>(token) * hidden_size;

            float vreg[NR][VPT];
            float local_max = 0.0f;
            #pragma unroll
            for (int r = 0; r < NR; ++r) {
                const int base = tid * VPT + r * stride;
                if (base < hidden_size) {
                    #pragma unroll
                    for (int j = 0; j < VPT; ++j) {
                        float v = __bfloat162float(pref[r].data[j]) * smv[r][j];
                        vreg[r][j] = v;
                        local_max = fmaxf(local_max, fabsf(v));
                    }
                }
            }

            // Prefetch token m+1's rows NOW (loads issue before the barrier stall).
            const int ntok = token + 1;
            VecIn pnext[NR];
            if (m + 1 < M && ntok < token_num) {
                #pragma unroll
                for (int r = 0; r < NR; ++r) {
                    const int base = tid * VPT + r * stride;
                    if (base < hidden_size)
                        pnext[r] = *reinterpret_cast<const VecIn*>(
                            hidden_states + static_cast<size_t>(ntok) * hidden_size + base);
                }
            }

            float w = warpReduceMax64(local_max);
            if ((tid & 63) == 0) sred[m & 1][tid >> 6] = w;
            __syncthreads();
            float absmax = sred[m & 1][0];
            #pragma unroll
            for (int i = 1; i < WARPS; ++i) absmax = fmaxf(absmax, sred[m & 1][i]);

            if (tid == 0) scales[token] = absmax * QuantTraits<T2>::INV_QMAX;
            const float inv_scale = absmax > 0.0f ? QuantTraits<T2>::QMAX * __builtin_mxc_rcpf(absmax) : 0.0f;

            #pragma unroll
            for (int r = 0; r < NR; ++r) {
                const int base = tid * VPT + r * stride;
                if (base < hidden_size) {
                    float sv[VPT];
                    #pragma unroll
                    for (int j = 0; j < VPT; ++j) sv[j] = vreg[r][j] * inv_scale;
                    quant_store_vec<T2, VPT>(out + base, sv);
                }
            }

            #pragma unroll
            for (int r = 0; r < NR; ++r) pref[r] = pnext[r];
        }
    } else {
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            const int token = t0 + m;
            if (token >= token_num) break;
            const T1* in  = hidden_states + static_cast<size_t>(token) * hidden_size;
            T2*       out = output        + static_cast<size_t>(token) * hidden_size;

            float vreg[NR][VPT];
            float local_max = 0.0f;
            #pragma unroll
            for (int r = 0; r < NR; ++r) {
                const int base = tid * VPT + r * stride;
                if (base < hidden_size) {
                    VecIn hv = *reinterpret_cast<const VecIn*>(in + base);
                    #pragma unroll
                    for (int j = 0; j < VPT; ++j) {
                        float v = __bfloat162float(hv.data[j]) * smv[r][j];
                        vreg[r][j] = v;
                        local_max = fmaxf(local_max, fabsf(v));
                    }
                }
            }

            float w = warpReduceMax64(local_max);
            if ((tid & 63) == 0) sred[m & 1][tid >> 6] = w;
            __syncthreads();
            float absmax = sred[m & 1][0];
            #pragma unroll
            for (int i = 1; i < WARPS; ++i) absmax = fmaxf(absmax, sred[m & 1][i]);

            if (tid == 0) scales[token] = absmax * QuantTraits<T2>::INV_QMAX;
            const float inv_scale = absmax > 0.0f ? QuantTraits<T2>::QMAX * __builtin_mxc_rcpf(absmax) : 0.0f;

            #pragma unroll
            for (int r = 0; r < NR; ++r) {
                const int base = tid * VPT + r * stride;
                if (base < hidden_size) {
                    float sv[VPT];
                    #pragma unroll
                    for (int j = 0; j < VPT; ++j) sv[j] = vreg[r][j] * inv_scale;
                    quant_store_vec<T2, VPT>(out + base, sv);
                }
            }
        }
    }
}

template<typename T1, typename T2>
__global__ void scale_dynamic_quant_generic(const T1* __restrict__ hidden_states,
                                            const float* __restrict__ smooth_scales,
                                            const int hidden_size,
                                            T2* __restrict__ output,
                                            float* __restrict__ scales) {
    extern __shared__ float sbuf[];  // hidden_size floats (scaled values)
    const int token = blockIdx.x;
    const int tid   = threadIdx.x;
    const int bdim  = blockDim.x;

    const T1* in  = hidden_states + static_cast<size_t>(token) * hidden_size;
    T2*       out = output        + static_cast<size_t>(token) * hidden_size;

    float local_max = 0.0f;
    for (int i = tid; i < hidden_size; i += bdim) {
        float v = __bfloat162float(in[i]) * smooth_scales[i];
        sbuf[i] = v;
        local_max = fmaxf(local_max, fabsf(v));
    }

    __shared__ float sred[1024];
    sred[tid] = local_max;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sred[tid] = fmaxf(sred[tid], sred[tid + stride]);
        __syncthreads();
    }
    const float absmax = sred[0];
    if (tid == 0) scales[token] = absmax * QuantTraits<T2>::INV_QMAX;
    const float inv_scale = absmax > 0.0f ? QuantTraits<T2>::QMAX * __builtin_mxc_rcpf(absmax) : 0.0f;

    for (int i = tid; i < hidden_size; i += bdim) {
        float x = sbuf[i] * inv_scale;
        if constexpr (std::is_same_v<T2, int8_t>) {
            out[i] = quant_i8(x);
        } else {
            out[i] = static_cast<T2>(fmaxf(-448.0f, fminf(x, 448.0f)));
        }
    }
}

template<typename T1, typename T2>
void launch_scale_dynamic_quant(const T1* hidden_states, const float* smooth_scales,
                                size_t hidden_size, size_t token_num,
                                T2* output, float* scales,
                                const cudaStream_t stream) {
    const int H = static_cast<int>(hidden_size);
    dim3 grid(token_num);

    // Diagnostic ablation router (default 0 = normal adaptive dispatch, unchanged).
    // All paths compute the SAME correct result; this only SELECTS which tier runs
    // so each optimization's contribution can be measured live on one binary:
    //   SDQ_FORCE=1 -> generic (scalar load + shared-mem stage + tree reduce)
    //   SDQ_FORCE=2 -> PATH B  (VPT=8 register-resident + 16-lane shuffle reduce)
    // (PATH A multi-token tiling is already selectable via SDQ_M.)
    static const int sdq_force = [](){ const char* e = getenv("SDQ_FORCE"); return e ? atoi(e) : 0; }();

    // -----------------------------------------------------------------------
    // FAST PATH W: WARP-PER-TOKEN (VPT=16), zero __syncthreads.  Applies when a
    // single 64-lane warp can span the row: H % 1024 == 0 and NR=H/1024 small.
    // Each warp is fully independent -> no block barrier ever serializes it,
    // removing the cross-warp reduce that dominated PATH A at large H.
    // WPB (warps per block) is a pure occupancy knob (no inter-warp comm); env
    // SDQ_WPB overrides for tuning, SDQ_WARPROW=0 disables the path.
    // -----------------------------------------------------------------------
    {
        static const int warprow_on = [](){ const char* e = getenv("SDQ_WARPROW"); return e ? atoi(e) : 0; }();
        static const int wpb_env    = [](){ const char* e = getenv("SDQ_WPB");     return e ? atoi(e) : 0; }();
        constexpr int VPTW = 16;
        if (warprow_on && sdq_force == 0 && (H % (64 * VPTW)) == 0) {
            const int NRv = H / (64 * VPTW);           // tiles per lane (1 for 1024, 4 for 4096)
            if (NRv >= 1 && NRv <= 8) {
                const int WPB = wpb_env > 0 ? wpb_env : 4;
                const int grid_w = static_cast<int>((token_num + WPB - 1) / WPB);
                #define LAUNCH_WARPROW(WPBV, NRV)                                        \
                    scale_dynamic_quant_warprow<T1, T2, WPBV, VPTW, NRV>                 \
                        <<<grid_w, WPBV * 64, 0, stream>>>(hidden_states, smooth_scales, \
                            H, static_cast<int>(token_num), output, scales)
                #define DISPATCH_WPB(NRV)                                                \
                    do {                                                                 \
                        switch (WPB) {                                                   \
                            case 1:  LAUNCH_WARPROW(1,  NRV); return;                     \
                            case 2:  LAUNCH_WARPROW(2,  NRV); return;                     \
                            case 4:  LAUNCH_WARPROW(4,  NRV); return;                     \
                            case 8:  LAUNCH_WARPROW(8,  NRV); return;                     \
                            case 16: LAUNCH_WARPROW(16, NRV); return;                     \
                            default: LAUNCH_WARPROW(4,  NRV); return;                     \
                        }                                                                \
                    } while (0)
                switch (NRv) {
                    case 1: DISPATCH_WPB(1);
                    case 2: DISPATCH_WPB(2);
                    case 3: DISPATCH_WPB(3);
                    case 4: DISPATCH_WPB(4);
                    case 5: DISPATCH_WPB(5);
                    case 6: DISPATCH_WPB(6);
                    case 7: DISPATCH_WPB(7);
                    case 8: DISPATCH_WPB(8);
                    default: break;
                }
                #undef DISPATCH_WPB
                #undef LAUNCH_WARPROW
            }
        }
    }

    // -----------------------------------------------------------------------
    // FAST PATH A: 32 B/thread (VPT=16) MULTI-TOKEN, register-resident smooth.
    // Applies when H%16==0 and cols16 = H/16 <= 512 (H <= 8192) -- covers every
    // mainstream LLM hidden size.  Each block loads its smooth slice once and
    // reuses it across M token rows (M=1/2/4), cutting the (dominant at small H)
    // smooth L2 traffic M-fold.
    //
    // Two structural knobs, tuned from the on-device cost breakdown:
    //  * NT matched to columns (round_up to a warp) -- forcing a coarse 512 left
    //    up to half the threads idle for H=5120/6144/7168.
    //  * NR (column-tiles per thread): at LARGE H the kernel is BARRIER-bound
    //    (bare-copy=1243, no-reduce=1049, full=800 @H=8192 -> the cross-warp
    //    reduce costs ~25%).  Using NR=2 lets one block span the row with HALF
    //    the warps (8->4), making that reduce cheaper.  Small H is already cheap
    //    and loses from the reduced occupancy, so it stays NR=1.
    //   Measured (T=32768):  H=4096 779(NR1)  H=5120 732(NR1)
    //                        H=6144 732->759(NR2)  H=7168 705->771(NR2)
    //                        H=8192 733->849(NR2, +16%)
    // The kernel guards every partial tile via `base < hidden_size`.
    // -----------------------------------------------------------------------
    constexpr int VPT16 = 16;
    if (sdq_force == 0 && (H % VPT16) == 0 && (H / VPT16) <= 512) {
        const int c16 = H / VPT16;                    // vectorized cols @ 16/thread
        // Adaptive token-tiling.  Larger M reuses the register-resident smooth
        // slice across MORE token rows, cutting the fp32 smooth L2 re-read (the
        // dominant secondary traffic: H*4 B = 2x the bf16 hidden footprint).
        // The sweet spot is DTYPE-DEPENDENT (measured, T=65536):
        //   int8: M=4 best (H=1024 867 / H=4096 873 GB/s; M=2 is 850/844)
        //   fp8 : M=2 best (H=1024 1098 / H=4096 980;  M=4 collapses to 899/970)
        // fp8's 4-wide HW pack makes the store cheap, so its bottleneck shifts to
        // grid occupancy -- M=4 halves the grid and starves the 28 APs.  int8's
        // scalar store is heavier, so it benefits more from the extra smooth
        // amortization of M=4.  M>=8 starves the grid for BOTH -> never used.
        // Only M in {1,2,4} are emitted (each has a matching template); env SDQ_M
        // overrides for tuning but is clamped to those valid values.
        constexpr bool is_i8 = std::is_same_v<T2, int8_t>;
        static const int m_env = [](){ const char* e = getenv("SDQ_M"); return e ? atoi(e) : 0; }();
        int M = 1;
        if (is_i8) {
            // int8: heavy scalar store -> smooth amortization pays.  But M=4 only
            // wins when the row is WIDE enough to fill a multi-warp block.  At
            // H=1024 the block is a SINGLE warp (NT=64), so M=4 both under-fills
            // the grid AND serializes 4 tokens on one warp -> measured M=2 beats
            // M=4 (987 vs 965 GB/s @T=65536).  Gate M=4 on c16>=128 (H>=2048),
            // where the block spans >=2 warps; narrow rows stay at M=2.
            if      (token_num >= 8192 && c16 >= 128) M = 4;
            else if (token_num >= 1024) M = 2;
        } else {
            // fp8: 4-wide HW-pack store is cheap -> grid occupancy dominates;
            // M=4 halves the grid and starves the APs, so cap at M=2.
            if      (token_num >= 1024) M = 2;
        }
        if (m_env == 1 || m_env == 2 || m_env == 4) M = m_env;
        const int grid_m = static_cast<int>((token_num + M - 1) / M);

        // NR=2 (fewer warps -> cheaper barrier) pays off once H is barrier-bound;
        // empirically that is H>=6144 (c16>=384).  Below that, NR=1 matched-NT.
        const bool use_nr2 = (c16 >= 384);
        const int tiles = use_nr2 ? 2 : 1;
        const int per   = (c16 + tiles - 1) / tiles;  // cols per tile
        const int NT16  = ((per + 63) / 64) * 64;     // block, warp-rounded

        #define LAUNCH_MTOK(NT, NRV, MM, PIPE)                                    \
            scale_dynamic_quant_mtok<T1, T2, NT, VPT16, NRV, MM, PIPE>             \
                <<<grid_m, NT, 0, stream>>>(hidden_states, smooth_scales, H,       \
                                            static_cast<int>(token_num),           \
                                            output, scales)
        #define DISPATCH_M(NT, NRV, PIPE)                                         \
            do {                                                                   \
                if      (M == 4) { LAUNCH_MTOK(NT, NRV, 4, PIPE); return; }         \
                else if (M == 2) { LAUNCH_MTOK(NT, NRV, 2, PIPE); return; }         \
                else             { LAUNCH_MTOK(NT, NRV, 1, PIPE); return; }         \
            } while (0)
        if (use_nr2) {
            // c16 in [384,512] -> per in [192,256] -> NT16 in {192,256}.
            // Software pipelining (PIPE=true) hides the barrier only for the wide
            // NT=256 rows (H>=7168); NT=192 (H=6144) prefers the leaner body.
            switch (NT16) {
                case 192: DISPATCH_M(192, 2, false);
                case 256: DISPATCH_M(256, 2, true);
                default:  DISPATCH_M(256, 2, true);   // per<=256 always covered by 256
            }
        } else {
            // c16 in [1,383] -> NT16 in {64,128,192,256,320,384}
            switch (NT16) {
                case  64: DISPATCH_M(64, 1, false);
                case 128: DISPATCH_M(128, 1, false);
                case 192: DISPATCH_M(192, 1, false);
                case 256: DISPATCH_M(256, 1, false);
                case 320: DISPATCH_M(320, 1, false);
                case 384: DISPATCH_M(384, 1, false);
                default:  DISPATCH_M(384, 1, false);   // c16<=383 always covered by 384
            }
        }
        #undef DISPATCH_M
        #undef LAUNCH_MTOK
    }

    // -----------------------------------------------------------------------
    // FAST PATH B: 16 B/thread (VPT=8).  Register-resident single global read;
    // pick the (threads, tiles) pair that covers cols = H/8 while keeping the
    // block a power-of-two multiple of the 64-wide warp for occupancy.
    // -----------------------------------------------------------------------
    constexpr int VPT = 8;  // float4 load (8 bf16) / int64 store (8 int8)
    if (sdq_force != 1 && (H % VPT) == 0) {
        const int cols = H / VPT;
        #define LAUNCH_REG(NT, NR)                                                  \
            scale_dynamic_quant_reg<T1, T2, NT, NR, VPT>                            \
                <<<grid, NT, 0, stream>>>(hidden_states, smooth_scales, H,          \
                                          output, scales)
        if      (cols <= 256)  { LAUNCH_REG(256, 1);  return; }  // H <= 2048
        else if (cols <= 512)  { LAUNCH_REG(512, 1);  return; }  // H <= 4096
        else if (cols <= 1024) { LAUNCH_REG(512, 2);  return; }  // H <= 8192
        else if (cols <= 2048) { LAUNCH_REG(1024, 2); return; }  // H <= 16384
        else if (cols <= 3072) { LAUNCH_REG(1024, 3); return; }  // H <= 24576
        else if (cols <= 4096) { LAUNCH_REG(1024, 4); return; }  // H <= 32768
        #undef LAUNCH_REG
        // else fall through to generic
    }

    // Generic fallback: shared-memory staged single pass.
    int block = H < 1024 ? H : 1024;
    block = ((block + 63) / 64) * 64;
    if (block == 0) block = 64;
    if (block > 1024) block = 1024;
    const size_t smem = sizeof(float) * hidden_size;
    scale_dynamic_quant_generic<T1, T2>
        <<<grid, block, smem, stream>>>(hidden_states, smooth_scales, H, output, scales);
}

std::tuple<at::Tensor, at::Tensor> scale_dynamic_quant(
    const at::Tensor& hidden_states,
    const at::Tensor& smooth_scales,
    at::ScalarType dst_dtype = at::ScalarType::Char
) {
    CHECK_DEVICE(hidden_states);
    CHECK_DEVICE(smooth_scales);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const size_t token_num   = hidden_states.numel() / hidden_states.size(-1);
    const size_t hidden_size = hidden_states.size(-1);
    at::Tensor output = at::empty_like(hidden_states, hidden_states.options().dtype(dst_dtype));
    at::Tensor scales = at::empty({hidden_states.numel() / hidden_states.size(-1)}, smooth_scales.options());
    if (token_num != hidden_states.numel() / hidden_size) {
        TORCH_CHECK(false, "token count doesn't match smooth scale count");
    }
    if (hidden_states.dtype() != at::ScalarType::BFloat16) {
        TORCH_CHECK(false, "Only support bfloat16 input");
    }
    bfloat16* in_ptr = reinterpret_cast<bfloat16*>(hidden_states.data_ptr<at::BFloat16>());
    const float* sm_ptr = smooth_scales.data_ptr<float>();
    float* sc_ptr = scales.data_ptr<float>();
    if (dst_dtype == at::ScalarType::Char) {
        launch_scale_dynamic_quant(in_ptr, sm_ptr, hidden_size, token_num,
                                   output.data_ptr<int8_t>(), sc_ptr, stream);
    } else if (dst_dtype == at::ScalarType::Float8_e4m3fn) {
        launch_scale_dynamic_quant(in_ptr, sm_ptr, hidden_size, token_num,
                                   reinterpret_cast<c10::Float8_e4m3fn*>(
                                       output.data_ptr<at::Float8_e4m3fn>()),
                                   sc_ptr, stream);
    } else {
        TORCH_CHECK(false, "Only support int8 (Char) or float8_e4m3fn output");
    }
    return std::make_tuple(output, scales);
}
