#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include <maca_fp8.h>
#include "../kernel/utils.h"
#include "../kernel/all_reduce_kernel.cuh"
#include "../include/moe_scatter_dynamic_quant.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

static constexpr int kWave = 64;
using wave_mask_t = uint64_t;
static constexpr wave_mask_t kFullWaveMask = 0xffffffffffffffffULL;

__device__ __forceinline__ int32_t WarpInclusiveScan64(int32_t val) {
    int lane = threadIdx.x & (kWave - 1);
    #pragma unroll
    for (int off = 1; off < kWave; off <<= 1) {
        int32_t t = __shfl_up_sync(kFullWaveMask, val, off, kWave);
        if (lane >= off) val += t;
    }
    return val;
}
__device__ __forceinline__ int32_t WarpInclusiveScan8(int32_t val) {
    int lane = threadIdx.x & (kWave - 1);
    int32_t t;
    t = __shfl_up_sync(kFullWaveMask, val, 1, kWave); if (lane >= 1 && lane < 8) val += t;
    t = __shfl_up_sync(kFullWaveMask, val, 2, kWave); if (lane >= 2 && lane < 8) val += t;
    t = __shfl_up_sync(kFullWaveMask, val, 4, kWave); if (lane >= 4 && lane < 8) val += t;
    return val;
}

constexpr int kFastScatterTopK = 8;
constexpr int kFastScatterHidden = 4096;
constexpr int kFastScatterTokenBits = 20;
constexpr int kFastScatterTokenMask = (1 << kFastScatterTokenBits) - 1;
constexpr int kFastScatterSlotShift = kFastScatterTokenBits;
constexpr int kFastScatterExpertShift = kFastScatterSlotShift + 3;

template <typename scalar_t, int vec_size>
struct __align__(vec_size * sizeof(scalar_t)) ScatterAlignedVec {
    scalar_t data[vec_size];
};

__device__ __forceinline__ int8_t float_to_int8_rn_bounded(float const x) {
    int32_t d = __float2int_rn(x);
    d = max(-128, min(127, d));
    return static_cast<int8_t>(d);
}

__device__ __forceinline__ __maca_fp8_e4m3 float_to_fp8_rn_bounded(float const x) {
    float const c = fminf(fmaxf(x, -448.0f), 448.0f);
    return static_cast<__maca_fp8_e4m3>(c);
}

template <typename out_t> struct QuantTraits;
template <> struct QuantTraits<int8_t> {
    static constexpr float kMax    = 127.0f;
    static constexpr float kInvMax = 1.0f / 127.0f;   // 0.007874015748031496f
    static __device__ __forceinline__ int8_t convert(float v) {
        return float_to_int8_rn_bounded(v);
    }
    // Quant + pack 16 scaled floats into 16 contiguous int8 bytes.
    static __device__ __forceinline__ void pack16(const float* v, int8_t* dst) {
        #pragma unroll
        for (int k = 0; k < 16; ++k) dst[k] = float_to_int8_rn_bounded(v[k]);
    }
};
template <> struct QuantTraits<__maca_fp8_e4m3> {
    static constexpr float kMax    = 448.0f;
    static constexpr float kInvMax = 1.0f / 448.0f;
    static __device__ __forceinline__ __maca_fp8_e4m3 convert(float v) {
        return float_to_fp8_rn_bounded(v);
    }
    // Hardware pack: 16 scaled floats -> 16 e4m3 bytes via 4x the C600
    // f16->f8 vector convert (RNE + saturate to +/-448, no software clamp).
    // Same path torch's .to(float8_e4m3fn) lowers to on this arch; verified
    // cos_sim-accurate in the vendor sglang/reshape_and_cache kernels.
    static __device__ __forceinline__ void pack16(const float* v,
                                                  __maca_fp8_e4m3* dst) {
        typedef __NATIVE_VECTOR__(4, _Float16) v4f16;
        uint32_t* d = reinterpret_cast<uint32_t*>(dst);
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            v4f16 t;
            t[0] = _Float16(v[g * 4 + 0]);
            t[1] = _Float16(v[g * 4 + 1]);
            t[2] = _Float16(v[g * 4 + 2]);
            t[3] = _Float16(v[g * 4 + 3]);
            d[g] = __builtin_mxc_cvt_pk4_f16tof8(t);
        }
    }
};

template <int NUM_EXPERTS>
__global__ void moe_scatter_count_fast_16x8(
        const int* __restrict__ selected_experts,
        int* __restrict__ experts_token_count,
        int* __restrict__ experts_token_start,
        const int num_tokens) {
    constexpr int kBlock = 256;
    constexpr int kWavesPerBlock = kBlock / kWave;
    __shared__ int wave_counts[kWavesPerBlock][NUM_EXPERTS];
    __shared__ int counts[NUM_EXPERTS];

    for (int i = threadIdx.x; i < kWavesPerBlock * NUM_EXPERTS; i += blockDim.x) {
        reinterpret_cast<int*>(wave_counts)[i] = 0;
    }
    __syncthreads();

    const int wave_id = threadIdx.x >> 6;
    const int total_routes = num_tokens * kFastScatterTopK;
    for (int r = threadIdx.x; r < total_routes; r += blockDim.x) {
        int e = selected_experts[r];
        if (static_cast<unsigned>(e) < NUM_EXPERTS) {
            atomicAdd(&wave_counts[wave_id][e], 1);
        }
    }
    __syncthreads();

    if (threadIdx.x < NUM_EXPERTS) {
        int c = 0;
        #pragma unroll
        for (int w = 0; w < kWavesPerBlock; ++w) c += wave_counts[w][threadIdx.x];
        counts[threadIdx.x] = c;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int start = 0;
        #pragma unroll
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            int c = counts[e];
            experts_token_count[e] = c;
            experts_token_start[e] = start;
            start += c;
        }
        experts_token_start[NUM_EXPERTS] = start;
    }
}

template <int BLOCK_SIZE, bool BUILD_ROUTE_MAP>
__global__ void moe_scatter_build_offsets_fast_16x8(
        const int* __restrict__ selected_experts,
        int* __restrict__ scatter_tokens_offset,
        int* __restrict__ packed_route_workspace,
        int* __restrict__ route_output_map,
        const int* __restrict__ experts_token_start,
        const int num_tokens) {
    using BlockScan   = cub::BlockScan<int, BLOCK_SIZE, cub::BLOCK_SCAN_WARP_SCANS>;
    using SelectedVec = ScatterAlignedVec<int, 4>;
    __shared__ typename BlockScan::TempStorage scan_storage;

    int const expert = blockIdx.x;
    int write_base = experts_token_start[expert];

    for (int token_base = 0; token_base < num_tokens; token_base += BLOCK_SIZE) {
        int const token = token_base + threadIdx.x;
        int slot = -1;
        if (token < num_tokens) {
            auto const* selected_vec =
                reinterpret_cast<const SelectedVec*>(selected_experts);
            SelectedVec const first  = selected_vec[token * 2];
            SelectedVec const second = selected_vec[token * 2 + 1];
            slot = first.data[0] == expert ? 0 :
                   first.data[1] == expert ? 1 :
                   first.data[2] == expert ? 2 :
                   first.data[3] == expert ? 3 :
                   second.data[0] == expert ? 4 :
                   second.data[1] == expert ? 5 :
                   second.data[2] == expert ? 6 :
                   second.data[3] == expert ? 7 : -1;
        }
        int const selected = slot >= 0;
        int local_offset = 0, tile_count = 0;
        BlockScan(scan_storage).ExclusiveSum(selected, local_offset, tile_count);

        if (selected) {
            int const packed = token
                             | (slot << kFastScatterSlotShift)
                             | (expert << kFastScatterExpertShift);
            int const output_idx = write_base + local_offset;
            scatter_tokens_offset[output_idx]   = token;
            packed_route_workspace[output_idx]  = packed;
            if (BUILD_ROUTE_MAP) {
                route_output_map[token * kFastScatterTopK + slot] = output_idx;
            }
        }
        write_base += tile_count;
        __syncthreads();
    }
}
// Wave-per-row output kernel: each wave (64 threads) produces one output row,
// so the per-row abs-max reduction is a pure warp __shfl_xor with NO
// __syncthreads and NO shared memory -- removing the block-wide barrier that
// serialized the block-per-row design. WAVES_PER_BLOCK independent waves are
// packed per block purely for occupancy; they never interact. hidden=4096 ->
// 64 elems/thread (128 B coalesced bf16 read + 64 B int8 write per thread).
template <typename scalar_t, typename out_t, int ITEMS_PER_THREAD, int WAVES_PER_BLOCK, int NUM_EXPERTS>
__global__ __launch_bounds__(kWave * WAVES_PER_BLOCK)
void moe_scatter_dynamic_quant_output_wave_4096(
        const scalar_t* __restrict__ hidden_states,
        const float* __restrict__ moe_weights,
        const float* __restrict__ smooth_scale,
        out_t* __restrict__ scatter_tokens,
        float* __restrict__ scatter_per_token_scale,
        const int* __restrict__ packed_route_workspace,
        const int* __restrict__ experts_token_count,
        const int* __restrict__ experts_token_start,
        const int output_capacity,
        const int blocks_per_expert) {
    static_assert(kWave * ITEMS_PER_THREAD == kFastScatterHidden, "");
    static_assert(ITEMS_PER_THREAD % 16 == 0, "");

    using InputVec  = ScatterAlignedVec<scalar_t, 8>;
    using SmoothVec = ScatterAlignedVec<float, 4>;
    using OutputVec = ScatterAlignedVec<out_t, 16>;
    using QT = QuantTraits<out_t>;
    // Ownership is tiled in 16-element chunks so load AND store cover the
    // SAME element positions: thread `lane` owns, for chunk c, the 16 elems
    // [c*(64*16) + lane*16 .. +16). Coalesced: 64 threads * 16 = 1024
    // contiguous elems per chunk. Each chunk = two 8-wide bf16 input vecs.
    constexpr int CHUNKS = ITEMS_PER_THREAD / 16;   // 128-bit int8 stores
    constexpr int TILE   = kWave * 16;              // elems per chunk, whole wave

    const int wave_in_block = threadIdx.x >> 6;     // 0..WAVES_PER_BLOCK-1
    const int global_worker  = blockIdx.x * WAVES_PER_BLOCK + wave_in_block;
    const int expert         = global_worker / blocks_per_expert;
    const int expert_worker  = global_worker - expert * blocks_per_expert;
    if (expert >= NUM_EXPERTS) return;

    const int expert_start = experts_token_start[expert];
    const int counted_end  = expert_start + experts_token_count[expert];
    const int expert_end   = counted_end < output_capacity
                             ? counted_end : output_capacity;
    const int lane = threadIdx.x & (kWave - 1);   // 0..63 within this wave

    // Cache this thread's smooth channels once; reuse across all its rows.
    float smooth_values[ITEMS_PER_THREAD];
    const float* smooth_base = smooth_scale
        + static_cast<int64_t>(expert) * kFastScatterHidden;
    #pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const float* sp = smooth_base + c * TILE + lane * 16;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            SmoothVec const sv =
                *reinterpret_cast<SmoothVec const*>(sp + q * 4);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                smooth_values[c * 16 + q * 4 + j] = sv.data[j];
        }
    }

    const int stride = blocks_per_expert;   // one row per block per iteration
    for (int out = expert_start + expert_worker; out < expert_end;
         out += stride) {
        int const packed = packed_route_workspace[out];
        int const token  = packed & kFastScatterTokenMask;

        const scalar_t* in_ptr = hidden_states
            + static_cast<int64_t>(token) * kFastScatterHidden;

        InputVec ivs[CHUNKS][2];   // two 8-wide bf16 vecs per 16-elem chunk
        float m = 0.f;
        #pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                ivs[c][h] = *reinterpret_cast<InputVec const*>(
                    in_ptr + c * TILE + lane * 16 + h * 8);
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    int const idx = c * 16 + h * 8 + pair * 2;
                    float2 const p = __bfloat1622float2(
                        *reinterpret_cast<const __maca_bfloat162*>(
                            ivs[c][h].data + pair * 2));
                    m = fmaxf(m, fmaxf(fabsf(p.x * smooth_values[idx]),
                                       fabsf(p.y * smooth_values[idx + 1])));
                }
            }
        }

        // Row abs-max: warp-only reduction, no barrier.
        #pragma unroll
        for (int off = kWave >> 1; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(kFullWaveMask, m, off, kWave));

        int const slot = (packed >> kFastScatterSlotShift) & 0x7;
        float const weight = moe_weights[token * kFastScatterTopK + slot];
        float const mm  = m > 0.f ? QT::kMax * __builtin_mxc_rcpf(m) : 0.f;
        float const mul = weight > 0.f ? mm : (weight < 0.f ? -mm : 0.f);
        if (lane == 0)
            scatter_per_token_scale[out] =
                m * fabsf(weight) * QT::kInvMax;

        out_t* out_ptr = scatter_tokens + static_cast<int64_t>(out) * kFastScatterHidden;
        #pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
            // Build the 16 scaled floats for this chunk, then quant+pack in one
            // shot. For fp8 pack16 uses the hardware f16->f8 vector convert
            // (4 elems/instr); for int8 it is the per-element rn+clamp.
            float sc[16];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    int const k   = h * 8 + pair * 2;
                    int const idx = c * 16 + k;
                    float2 const p = __bfloat1622float2(
                        *reinterpret_cast<const __maca_bfloat162*>(
                            ivs[c][h].data + pair * 2));
                    sc[k]     = p.x * smooth_values[idx]   * mul;
                    sc[k + 1] = p.y * smooth_values[idx+1] * mul;
                }
            }
            OutputVec ov;
            QT::pack16(sc, ov.data);
            *reinterpret_cast<OutputVec*>(
                out_ptr + c * TILE + lane * 16) = ov;
        }
    }
}

// ---------------------------------------------------------------------------
// Block-per-row output kernel (high-occupancy variant).
//
// The wave-per-row kernel above owns a whole 4096-wide row with ONE 64-lane
// wave => 64 elems/thread, forcing smooth_values[64] + ivs[64] ~= 96 data
// registers/thread and capping occupancy near 33%.  The .59 MODE diagnostic
// showed the raw 2R1W memory pattern sustains ~980 GB/s at that occupancy, so
// the ~50-60% plateau is ALU-latency starved: too few resident waves to hide
// the per-element bf16->f32 decode + abs-max + quant-convert.
//
// This kernel copies moe_gather's proven recipe instead: MANY threads per row,
// FEW elements per thread (16 or 32), so the per-thread data footprint drops to
// ~32-64 registers and 2-4x more waves stay resident.  Each element is decoded
// bf16->f32 and smoothed EXACTLY ONCE (cached in `scaled`); the second pass
// only multiplies by the row scale and quant-converts.  The per-row abs-max is
// a warp __shfl_xor reduction followed by a single-barrier cross-wave combine.
//
// THREADS_PER_ROW must be a multiple of kWave (64) and divide 4096.
//   TPR=256 -> 16 elems/thread, 4 waves/row (one 128-bit store/thread)
//   TPR=128 -> 32 elems/thread, 2 waves/row (two 128-bit stores/thread)
// ---------------------------------------------------------------------------
template <typename scalar_t, typename out_t, int THREADS_PER_ROW, int NUM_EXPERTS>
__global__ __launch_bounds__(THREADS_PER_ROW)
void moe_scatter_dynamic_quant_output_block_4096(
        const scalar_t* __restrict__ hidden_states,
        const float* __restrict__ moe_weights,
        const float* __restrict__ smooth_scale,
        out_t* __restrict__ scatter_tokens,
        float* __restrict__ scatter_per_token_scale,
        const int* __restrict__ packed_route_workspace,
        const int* __restrict__ experts_token_count,
        const int* __restrict__ experts_token_start,
        const int output_capacity,
        const int blocks_per_expert) {
    static_assert(THREADS_PER_ROW % kWave == 0, "TPR multiple of wave");
    static_assert(kFastScatterHidden % THREADS_PER_ROW == 0, "TPR divides H");

    constexpr int EPT    = kFastScatterHidden / THREADS_PER_ROW;  // 16 or 32
    constexpr int CHUNKS = EPT / 16;                              // 128-bit tiles
    constexpr int TILE   = THREADS_PER_ROW * 16;                 // elems/chunk
    constexpr int WAVES  = THREADS_PER_ROW / kWave;              // 2 or 4
    static_assert(EPT % 16 == 0, "EPT multiple of 16");

    using InputVec  = ScatterAlignedVec<scalar_t, 8>;
    using SmoothVec = ScatterAlignedVec<float, 4>;
    using OutputVec = ScatterAlignedVec<out_t, 16>;
    using QT = QuantTraits<out_t>;

    __shared__ float s_wave_max[WAVES];

    const int tid  = threadIdx.x;
    const int wave = tid >> 6;

    const int global_worker = blockIdx.x;
    const int expert        = global_worker / blocks_per_expert;
    const int expert_worker = global_worker - expert * blocks_per_expert;
    if (expert >= NUM_EXPERTS) return;

    const int expert_start = experts_token_start[expert];
    const int counted_end  = expert_start + experts_token_count[expert];
    const int expert_end   = counted_end < output_capacity
                             ? counted_end : output_capacity;

    // Cache this thread's smooth channels once; reuse across all its rows.
    float smooth_values[EPT];
    const float* smooth_base = smooth_scale
        + static_cast<int64_t>(expert) * kFastScatterHidden;
    #pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const float* sp = smooth_base + c * TILE + tid * 16;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            SmoothVec const sv = *reinterpret_cast<SmoothVec const*>(sp + q * 4);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                smooth_values[c * 16 + q * 4 + j] = sv.data[j];
        }
    }

    for (int out = expert_start + expert_worker; out < expert_end;
         out += blocks_per_expert) {
        int const packed = packed_route_workspace[out];
        int const token  = packed & kFastScatterTokenMask;

        const scalar_t* in_ptr = hidden_states
            + static_cast<int64_t>(token) * kFastScatterHidden;

        // Pass 1: decode+smooth ONCE, cache scaled f32, track thread-local max.
        float scaled[EPT];
        float m = 0.f;
        #pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                InputVec const iv = *reinterpret_cast<InputVec const*>(
                    in_ptr + c * TILE + tid * 16 + h * 8);
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    int const idx = c * 16 + h * 8 + pair * 2;
                    float2 const p = __bfloat1622float2(
                        *reinterpret_cast<const __maca_bfloat162*>(
                            iv.data + pair * 2));
                    float const a = p.x * smooth_values[idx];
                    float const b = p.y * smooth_values[idx + 1];
                    scaled[idx]     = a;
                    scaled[idx + 1] = b;
                    m = fmaxf(m, fmaxf(fabsf(a), fabsf(b)));
                }
            }
        }

        // Warp-local abs-max (no barrier), then single-barrier cross-wave max.
        #pragma unroll
        for (int off = kWave >> 1; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(kFullWaveMask, m, off, kWave));
        if (WAVES > 1) {
            if ((tid & (kWave - 1)) == 0) s_wave_max[wave] = m;
            __syncthreads();
            #pragma unroll
            for (int w = 0; w < WAVES; ++w) m = fmaxf(m, s_wave_max[w]);
        }

        int const slot = (packed >> kFastScatterSlotShift) & 0x7;
        float const weight = moe_weights[token * kFastScatterTopK + slot];
        float const mm  = m > 0.f ? QT::kMax * __builtin_mxc_rcpf(m) : 0.f;
        float const mul = weight > 0.f ? mm : (weight < 0.f ? -mm : 0.f);
        if (tid == 0)
            scatter_per_token_scale[out] = m * fabsf(weight) * QT::kInvMax;

        // Pass 2: apply row scale + quant-convert the cached f32, 128-bit stores.
        out_t* out_ptr = scatter_tokens
            + static_cast<int64_t>(out) * kFastScatterHidden;
        #pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
            OutputVec ov;
            #pragma unroll
            for (int k = 0; k < 16; ++k)
                ov.data[k] = QT::convert(scaled[c * 16 + k] * mul);
            *reinterpret_cast<OutputVec*>(out_ptr + c * TILE + tid * 16) = ov;
        }
        if (WAVES > 1) __syncthreads();  // protect s_wave_max reuse next row
    }
}

// ---------------------------------------------------------------------------
// Wave-per-row kernel with per-expert smooth staged in SHARED memory.
//
// Identical row algorithm to moe_scatter_dynamic_quant_output_wave_4096 (each
// 64-lane wave owns one 4096-wide row, abs-max via __shfl_xor, NO per-row
// barrier), but the 4096 smooth channels are loaded ONCE into shared memory at
// block start instead of into a 64-entry-per-thread register array.  This frees
// ~64 registers/thread (the smooth_values[64] array was the dominant occupancy
// limiter), letting more waves stay resident to hide the decode/quant ALU
// latency -- the .59 MODE diagnostic showed the raw 2R1W pattern sustains
// ~980 GB/s, so occupancy, not the memory pattern, is the ~50-60% plateau.
//
// One block owns ONE expert (grid.x = experts * blocks_per_expert laid out so
// that block b -> expert b / blocks_per_expert), so all WPB waves in the block
// share the same expert's smooth row.  Shared cost: 4096 * 4 B = 16 KB/block.
// ---------------------------------------------------------------------------
template <typename scalar_t, typename out_t, int WAVES_PER_BLOCK, int NUM_EXPERTS>
__global__ __launch_bounds__(kWave * WAVES_PER_BLOCK)
void moe_scatter_dynamic_quant_output_wave_smsh_4096(
        const scalar_t* __restrict__ hidden_states,
        const float* __restrict__ moe_weights,
        const float* __restrict__ smooth_scale,
        out_t* __restrict__ scatter_tokens,
        float* __restrict__ scatter_per_token_scale,
        const int* __restrict__ packed_route_workspace,
        const int* __restrict__ experts_token_count,
        const int* __restrict__ experts_token_start,
        const int output_capacity,
        const int blocks_per_expert) {
    constexpr int ITEMS_PER_THREAD = kFastScatterHidden / kWave;  // 64
    constexpr int CHUNKS = ITEMS_PER_THREAD / 16;
    constexpr int TILE   = kWave * 16;

    using InputVec  = ScatterAlignedVec<scalar_t, 8>;
    using OutputVec = ScatterAlignedVec<out_t, 16>;
    using Float4Vec = ScatterAlignedVec<float, 4>;
    using QT = QuantTraits<out_t>;

    __shared__ float s_smooth[kFastScatterHidden];  // 16 KB, one expert

    // Block-per-expert layout: blocks_per_expert BLOCKS serve each expert, so
    // every wave in this block shares the same expert row => one shared copy.
    const int expert         = blockIdx.x / blocks_per_expert;
    const int block_in_expert= blockIdx.x - expert * blocks_per_expert;
    const int wave_in_block  = threadIdx.x >> 6;
    const int lane           = threadIdx.x & (kWave - 1);
    if (expert >= NUM_EXPERTS) return;

    const float* smooth_base = smooth_scale
        + static_cast<int64_t>(expert) * kFastScatterHidden;
    for (int i = threadIdx.x * 4; i < kFastScatterHidden;
         i += blockDim.x * 4) {
        *reinterpret_cast<Float4Vec*>(s_smooth + i) =
            *reinterpret_cast<const Float4Vec*>(smooth_base + i);
    }
    __syncthreads();

    // Global worker id within this expert, and the stride over its rows.
    const int expert_worker = block_in_expert * WAVES_PER_BLOCK + wave_in_block;
    const int worker_stride = blocks_per_expert * WAVES_PER_BLOCK;
    const int expert_start = experts_token_start[expert];
    const int counted_end  = expert_start + experts_token_count[expert];
    const int expert_end   = counted_end < output_capacity
                             ? counted_end : output_capacity;

    for (int out = expert_start + expert_worker; out < expert_end;
         out += worker_stride) {
        int const packed = packed_route_workspace[out];
        int const token  = packed & kFastScatterTokenMask;

        const scalar_t* in_ptr = hidden_states
            + static_cast<int64_t>(token) * kFastScatterHidden;

        InputVec ivs[CHUNKS][2];
        float m = 0.f;
        #pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                ivs[c][h] = *reinterpret_cast<InputVec const*>(
                    in_ptr + c * TILE + lane * 16 + h * 8);
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    int const idx = c * 16 + h * 8 + pair * 2;
                    int const sidx = c * TILE + lane * 16 + h * 8 + pair * 2;
                    float2 const p = __bfloat1622float2(
                        *reinterpret_cast<const __maca_bfloat162*>(
                            ivs[c][h].data + pair * 2));
                    m = fmaxf(m, fmaxf(fabsf(p.x * s_smooth[sidx]),
                                       fabsf(p.y * s_smooth[sidx + 1])));
                }
            }
        }

        #pragma unroll
        for (int off = kWave >> 1; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(kFullWaveMask, m, off, kWave));

        int const slot = (packed >> kFastScatterSlotShift) & 0x7;
        float const weight = moe_weights[token * kFastScatterTopK + slot];
        float const mm  = m > 0.f ? QT::kMax * __builtin_mxc_rcpf(m) : 0.f;
        float const mul = weight > 0.f ? mm : (weight < 0.f ? -mm : 0.f);
        if (lane == 0)
            scatter_per_token_scale[out] = m * fabsf(weight) * QT::kInvMax;

        out_t* out_ptr = scatter_tokens
            + static_cast<int64_t>(out) * kFastScatterHidden;
        #pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
            OutputVec ov;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    int const k    = h * 8 + pair * 2;
                    int const sidx = c * TILE + lane * 16 + k;
                    float2 const p = __bfloat1622float2(
                        *reinterpret_cast<const __maca_bfloat162*>(
                            ivs[c][h].data + pair * 2));
                    ov.data[k]     = QT::convert(p.x * s_smooth[sidx] * mul);
                    ov.data[k + 1] = QT::convert(p.y * s_smooth[sidx + 1] * mul);
                }
            }
            *reinterpret_cast<OutputVec*>(
                out_ptr + c * TILE + lane * 16) = ov;
        }
    }
}

template<int NUM_EXPERTS>
__global__ void moe_align_token_offset(
        const int* selected_experts,
        int* scatter_tokens_offset,
        int* experts_token_count,
        int* experts_token_start,
        int topk, int num_tokens, int num_experts_per_rank,
        const int shared_tokens_per_sp,
        const int num_shared_experts_per_rank) {
    static constexpr int kBlock = 512;
    static constexpr int kWavesPerBlock = kBlock / kWave;

    __shared__ int shared_counts[kWavesPerBlock][NUM_EXPERTS];
    __shared__ int32_t wave_sum[kWavesPerBlock];
    __shared__ int32_t cumsum[NUM_EXPERTS + 1];

    const int wave_id = threadIdx.x >> 6;
    const int lane    = threadIdx.x & 63;

    if (threadIdx.x < num_shared_experts_per_rank) {
        experts_token_count[threadIdx.x] = num_tokens;
        experts_token_start[threadIdx.x] = threadIdx.x * shared_tokens_per_sp;
    }
    for (int idx = threadIdx.x;
         idx < shared_tokens_per_sp * num_shared_experts_per_rank;
         idx += blockDim.x) {
        scatter_tokens_offset[idx] = idx;
    }

    for (int e = threadIdx.x;
         e < kWavesPerBlock * NUM_EXPERTS;
         e += blockDim.x) {
        reinterpret_cast<int*>(shared_counts)[e] = 0;
    }
    if (threadIdx.x == 0) cumsum[0] = 0;
    __syncthreads();

    for (int idx = threadIdx.x; idx < topk * num_tokens; idx += blockDim.x) {
        int e = selected_experts[idx];
        if (static_cast<unsigned>(e) < static_cast<unsigned>(num_experts_per_rank)) {
            atomicAdd(&shared_counts[wave_id][e], 1);
        }
    }
    __syncthreads();

    int val = 0;
    if (threadIdx.x < NUM_EXPERTS) {
        int s = 0;
        #pragma unroll
        for (int w = 0; w < kWavesPerBlock; ++w) s += shared_counts[w][threadIdx.x];
        val = s;
        if (threadIdx.x < num_experts_per_rank) {
            experts_token_count[threadIdx.x + num_shared_experts_per_rank] = val;
        }
    }
    __syncthreads();

    if (threadIdx.x < NUM_EXPERTS) {
        val = WarpInclusiveScan64(val);
        if (lane == kWave - 1) wave_sum[wave_id] = val;
    }
    __syncthreads();

    if (threadIdx.x < kWavesPerBlock) {
        int s = wave_sum[threadIdx.x];
        wave_sum[threadIdx.x] = WarpInclusiveScan8(s);
    }
    __syncthreads();

    if (threadIdx.x < NUM_EXPERTS && wave_id > 0) {
        val += wave_sum[wave_id - 1];
    }
    if (threadIdx.x < NUM_EXPERTS) {
        cumsum[threadIdx.x + 1] = val;
    }
    __syncthreads();

    if (threadIdx.x < num_experts_per_rank) {
        experts_token_start[threadIdx.x + num_shared_experts_per_rank] =
            cumsum[threadIdx.x];
    }
}

template<int BLOCK_SIZE>
__global__ void stable_scatter_offset(
        int* scatter_tokens_offset,
        const int* experts_token_count,
        const int* experts_token_start,
        const int* selected_experts,
        const int num_tokens, const int topk,
        const int shared_tokens_per_sp,
        const int num_shared_experts_per_rank) {
    using BlockScan = cub::BlockScan<int, BLOCK_SIZE, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename BlockScan::TempStorage scan_storage;

    const int expert_id  = blockIdx.x;
    const int shared_off = shared_tokens_per_sp * num_shared_experts_per_rank;
    const int write_base = experts_token_start[expert_id + num_shared_experts_per_rank] + shared_off;
    const int expected   = experts_token_count[expert_id + num_shared_experts_per_rank];
    if (expected == 0) return;

    int written = 0;
    for (int tok_base = 0; tok_base < num_tokens; tok_base += BLOCK_SIZE) {
        int tok = tok_base + threadIdx.x;
        int selected = 0;
        if (tok < num_tokens) {
            #pragma unroll 8
            for (int k = 0; k < topk; ++k) {
                if (selected_experts[tok * topk + k] == expert_id) {
                    selected = 1;
                    break;
                }
            }
        }
        int local_off = 0, tile_cnt = 0;
        BlockScan(scan_storage).ExclusiveSum(selected, local_off, tile_cnt);

        if (selected) {
            scatter_tokens_offset[write_base + written + local_off] = tok;
        }
        written += tile_cnt;
        __syncthreads();
        if (written >= expected) break;
    }
}
template<typename scalar_t, typename out_t>
__global__ void moe_scatter_dynamic_quant_kernel(
        const scalar_t* hidden_states,
        const int* selected_experts,
        const float* moe_weights,
        const float* smooth_scale,
        out_t* scatter_tokens,
        float* scatter_per_token_scale,
        const int* scatter_tokens_offset,
        const int* experts_token_start,
        const int topk, const int hidden_size,
        const int num_experts_per_rank,
        const int num_shared_total_tokens,
        const int num_shared_experts_per_rank,
        const int shared_tokens_per_sp,
        const int num_tokens) {
    const int output_idx = blockIdx.x;
    const int total_output = gridDim.x;
    if (output_idx >= total_output) return;
    const int input_token_idx = scatter_tokens_offset[output_idx];
    if (input_token_idx < 0 || input_token_idx >= num_tokens) return;

    int   expert_idx;
    bool  is_shared;
    int   k_idx = 0;
    float weight_val = 1.0f;
    using QT = QuantTraits<out_t>;

    if (output_idx < num_shared_total_tokens) {
        is_shared  = true;
        expert_idx = output_idx / shared_tokens_per_sp;
        if (expert_idx >= num_shared_experts_per_rank) return;
    } else {
        const int routed_idx = output_idx - num_shared_total_tokens;
        int lo = 0, hi = num_experts_per_rank;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (experts_token_start[mid + num_shared_experts_per_rank] <= routed_idx)
                lo = mid + 1;
            else
                hi = mid;
        }
        expert_idx = lo - 1;
        if (expert_idx < 0 || expert_idx >= num_experts_per_rank) return;
        is_shared = false;
    }

    __shared__ int   s_k_idx;
    __shared__ float s_weight;
    __shared__ float s_scale;

    if (threadIdx.x == 0) {
        if (is_shared) {
            s_k_idx  = -1;
            s_weight = 1.0f;
        } else {
            int k = -1;
            #pragma unroll 8
            for (int i = 0; i < topk; ++i) {
                if (selected_experts[input_token_idx * topk + i] == expert_idx) {
                    k = i;
                    break;
                }
            }
            s_k_idx  = k;
            s_weight = (k >= 0)
                     ? moe_weights[input_token_idx * topk + k]
                     : 0.0f;
        }
    }
    __syncthreads();

    if (!is_shared && s_k_idx < 0) {
        auto out_ptr = scatter_tokens + (int64_t)output_idx * hidden_size;
        out_t const zero = QuantTraits<out_t>::convert(0.f);
        for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
            out_ptr[idx] = zero;
        }
        if (threadIdx.x == 0) scatter_per_token_scale[output_idx] = 0.f;
        return;
    }

    const float weight = s_weight;

    auto input_ptr  = hidden_states + (int64_t)input_token_idx * hidden_size;
    auto output_ptr = scatter_tokens + (int64_t)output_idx     * hidden_size;
    auto smooth_ptr = smooth_scale   + (int64_t)expert_idx     * hidden_size;

    constexpr int MAX_PER_THREAD = 32;
    float cached[MAX_PER_THREAD];
    int   n_local = 0;
    float max_val = 0.0f;

    for (int idx = threadIdx.x, s = 0; idx < hidden_size;
         idx += blockDim.x, ++s) {
        float v = __bfloat162float(input_ptr[idx]) * weight * smooth_ptr[idx];
        cached[s] = v;
        max_val = fmaxf(max_val, fabsf(v));
        n_local = s + 1;
    }
    max_val = BlockReduceMax<float>(max_val);

    if (threadIdx.x == 0) {
        scatter_per_token_scale[output_idx] = max_val * QT::kInvMax;
        s_scale = max_val > 0.f ? QT::kMax * __builtin_mxc_rcpf(max_val) : 0.f;
    }
    __syncthreads();

    for (int i = 0, idx = threadIdx.x; i < n_local; ++i, idx += blockDim.x) {
        output_ptr[idx] = QT::convert(cached[i] * s_scale);
    }
}

constexpr int kFusedSmallMaxTokens = 128;
constexpr int kFusedSmallMaxRoutes = kFusedSmallMaxTokens * kFastScatterTopK; // 1024

template <typename scalar_t, typename out_t, int WAVES_PER_BLOCK, int NUM_EXPERTS>
__global__ __launch_bounds__(kWave * WAVES_PER_BLOCK)
void moe_scatter_fused_small_16x8_4096(
        const scalar_t* __restrict__ hidden_states,
        const int* __restrict__ selected_experts,
        const float* __restrict__ moe_weights,
        const float* __restrict__ smooth_scale,
        out_t* __restrict__ scatter_tokens,
        float* __restrict__ scatter_per_token_scale,
        int* __restrict__ scatter_tokens_offset,
        int* __restrict__ experts_token_count,
        int* __restrict__ experts_token_start,
        const int num_tokens,
        const int output_capacity) {
    constexpr int kThreads   = kWave * WAVES_PER_BLOCK;
    constexpr int ITEMS      = kFastScatterHidden / kWave;   // 64
    constexpr int CHUNKS     = ITEMS / 16;                   // 4
    constexpr int TILE       = kWave * 16;                   // 1024 elems/chunk

    using InputVec  = ScatterAlignedVec<scalar_t, 8>;
    using SmoothVec = ScatterAlignedVec<float, 4>;
    using OutputVec = ScatterAlignedVec<out_t, 16>;
    using QT = QuantTraits<out_t>;

    __shared__ unsigned char s_exp[kFusedSmallMaxRoutes];
    __shared__ int   s_packed[kFusedSmallMaxRoutes];
    __shared__ int   s_count[NUM_EXPERTS];
    __shared__ int   s_start[NUM_EXPERTS + 1];
    __shared__ int   s_run[NUM_EXPERTS];
    __shared__ int   s_wcnt[NUM_EXPERTS];
    __shared__ int   s_total;

    const int tid   = threadIdx.x;
    const int lane  = tid & (kWave - 1);
    const int wave  = tid >> 6;
    const int total_routes = num_tokens * kFastScatterTopK;

    if (tid < NUM_EXPERTS) { s_count[tid] = 0; s_run[tid] = 0; }
    __syncthreads();

    for (int r = tid; r < total_routes; r += kThreads) {
        int e = selected_experts[r];
        unsigned ue = static_cast<unsigned>(e);
        s_exp[r] = (ue < NUM_EXPERTS)
                 ? static_cast<unsigned char>(e)
                 : static_cast<unsigned char>(0xFF);
        if (ue < NUM_EXPERTS) atomicAdd(&s_count[e], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int acc = 0;
        #pragma unroll
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            s_start[e] = acc;
            acc += s_count[e];
        }
        s_start[NUM_EXPERTS] = acc;
        s_total = acc < output_capacity ? acc : output_capacity;
    }
    __syncthreads();

    if (wave == 0) {
        for (int base = 0; base < total_routes; base += kWave) {
            int const r = base + lane;
            int const e = (r < total_routes) ? static_cast<int>(s_exp[r]) : 0xFF;

            wave_mask_t mymask = 0;
            #pragma unroll
            for (int eq = 0; eq < NUM_EXPERTS; ++eq) {
                wave_mask_t m = __ballot_sync(kFullWaveMask, e == eq);
                if (e == eq) mymask = m;
                if (lane == eq) s_wcnt[eq] = __popcll(m);
            }

            if (e != 0xFF) {
                wave_mask_t const below = (lane == 0)
                    ? wave_mask_t(0)
                    : ((wave_mask_t(1) << lane) - 1);
                int const rank = __popcll(mymask & below);
                int const out  = s_start[e] + s_run[e] + rank;
                if (out < output_capacity) {
                    int const token = r / kFastScatterTopK;
                    int const slot  = r - token * kFastScatterTopK;
                    s_packed[out] = token
                                  | (slot << kFastScatterSlotShift)
                                  | (e    << kFastScatterExpertShift);
                }
            }
            if (lane < NUM_EXPERTS) s_run[lane] += s_wcnt[lane];
        }
    }
    __syncthreads();

    if (blockIdx.x == 0) {
        if (tid < NUM_EXPERTS) {
            experts_token_count[tid] = s_count[tid];
            experts_token_start[tid] = s_start[tid];
        }
        if (tid == 0) experts_token_start[NUM_EXPERTS] =
                          s_start[NUM_EXPERTS];
        for (int o = tid; o < s_total; o += kThreads) {
            scatter_tokens_offset[o] = s_packed[o] & kFastScatterTokenMask;
        }
    }

    int const total_rows = s_total;
    int const wid    = blockIdx.x * WAVES_PER_BLOCK + wave;
    int const stride = gridDim.x * WAVES_PER_BLOCK;

    for (int out = wid; out < total_rows; out += stride) {
        int const packed = s_packed[out];
        int const token  = packed & kFastScatterTokenMask;
        int const slot   = (packed >> kFastScatterSlotShift) & 0x7;
        int const expert = (packed >> kFastScatterExpertShift) & 0x1F;

        const scalar_t* in_ptr = hidden_states
            + static_cast<int64_t>(token) * kFastScatterHidden;
        const float* sm_ptr = smooth_scale
            + static_cast<int64_t>(expert) * kFastScatterHidden;

        float cached[ITEMS];
        float m = 0.f;
        #pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
            const int off = c * TILE + lane * 16;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                InputVec const iv =
                    *reinterpret_cast<InputVec const*>(in_ptr + off + h * 8);
                #pragma unroll
                for (int q = 0; q < 2; ++q) {
                    SmoothVec const s0 = *reinterpret_cast<SmoothVec const*>(
                        sm_ptr + off + h * 8 + q * 4);
                    #pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        int const k = h * 8 + q * 4 + p * 2;
                        float2 const v = __bfloat1622float2(
                            *reinterpret_cast<const __maca_bfloat162*>(
                                iv.data + q * 4 + p * 2));
                        float const a = v.x * s0.data[p * 2];
                        float const b = v.y * s0.data[p * 2 + 1];
                        cached[c * 16 + k]     = a;
                        cached[c * 16 + k + 1] = b;
                        m = fmaxf(m, fmaxf(fabsf(a), fabsf(b)));
                    }
                }
            }
        }

        #pragma unroll
        for (int off = kWave >> 1; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(kFullWaveMask, m, off, kWave));

        float const weight = moe_weights[token * kFastScatterTopK + slot];
        float const mm  = m > 0.f ? QT::kMax * __builtin_mxc_rcpf(m) : 0.f;
        float const mul = weight > 0.f ? mm : (weight < 0.f ? -mm : 0.f);
        if (lane == 0)
            scatter_per_token_scale[out] = m * fabsf(weight) * QT::kInvMax;

        out_t* out_ptr = scatter_tokens
            + static_cast<int64_t>(out) * kFastScatterHidden;
        #pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
            OutputVec ov;
            #pragma unroll
            for (int k = 0; k < 16; ++k)
                ov.data[k] = QT::convert(cached[c * 16 + k] * mul);
            *reinterpret_cast<OutputVec*>(out_ptr + c * TILE + lane * 16) = ov;
        }
    }
}

template<typename scalar_t, typename out_t>
void launch_moe_scatter_dynamic_quant_kernel(
        const scalar_t* hidden_status, const int* selected_experts,
        const float* moe_weights, const float* smooth_scale,
        out_t* scatter_tokens, float* scatter_per_token_scale,
        int* scatter_tokens_offset, int* experts_token_count,
        int* experts_token_start,
        int* packed_route_workspace,
        const int num_shared_experts_per_rank, const int hidden_size,
        const int num_tokens, const int output_capacity,
        const int num_experts_per_rank, const int topk,
        const int shared_tokens_per_sp, const cudaStream_t& stream) {
        if (hidden_size == kFastScatterHidden &&
        topk == kFastScatterTopK &&
        (num_experts_per_rank == 8 || num_experts_per_rank == 16) &&
        num_shared_experts_per_rank == 0 &&
        num_tokens <= kFastScatterTokenMask) {
        if (num_tokens > 0 && num_tokens < kFusedSmallMaxTokens) {
            constexpr int WPB = 4;
            int const max_rows = num_tokens * kFastScatterTopK;
            int const mpc =
                at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
            int grid = (max_rows + WPB - 1) / WPB;
            if (grid > mpc) grid = mpc;
            if (num_experts_per_rank == 8) {
                moe_scatter_fused_small_16x8_4096<scalar_t, out_t, WPB, 8>
                    <<<grid, kWave * WPB, 0, stream>>>(
                        hidden_status, selected_experts, moe_weights, smooth_scale,
                        scatter_tokens, scatter_per_token_scale,
                        scatter_tokens_offset, experts_token_count,
                        experts_token_start, num_tokens, output_capacity);
            } else if (num_experts_per_rank == 16) {
                moe_scatter_fused_small_16x8_4096<scalar_t, out_t, WPB, 16>
                    <<<grid, kWave * WPB, 0, stream>>>(
                        hidden_status, selected_experts, moe_weights, smooth_scale,
                        scatter_tokens, scatter_per_token_scale,
                        scatter_tokens_offset, experts_token_count,
                        experts_token_start, num_tokens, output_capacity);
            }            
            return;
        }

        constexpr int route_block = 256;
        if (num_experts_per_rank == 8) {
            moe_scatter_count_fast_16x8<8><<<1, route_block, 0, stream>>>(
                selected_experts, experts_token_count,
                experts_token_start, num_tokens);

            moe_scatter_build_offsets_fast_16x8<route_block, false>
                <<<8, route_block, 0, stream>>>(
                    selected_experts,
                    scatter_tokens_offset,
                    packed_route_workspace,
                    nullptr,
                    experts_token_start,
                    num_tokens);
        } else if (num_experts_per_rank == 16) {
            moe_scatter_count_fast_16x8<16><<<1, route_block, 0, stream>>>(
                selected_experts, experts_token_count,
                experts_token_start, num_tokens);

            moe_scatter_build_offsets_fast_16x8<route_block, false>
                <<<16, route_block, 0, stream>>>(
                    selected_experts,
                    scatter_tokens_offset,
                    packed_route_workspace,
                    nullptr,
                    experts_token_start,
                    num_tokens);
        }

        const int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

        // Output kernel selection. MSDQ_TPR env picks threads-per-row:
        //   0  (default) -> wave-per-row (64 threads, 64 elems/thread)  [legacy]
        //   128 / 256    -> block-per-row high-occupancy variant
        // A/B knob for the ReAct occupancy experiment; default keeps legacy.
        static const int tpr = [](){
            const char* e = getenv("MSDQ_TPR");
            return e ? atoi(e) : 0;
        }();

        if (tpr == 128 || tpr == 256) {
            // One block == one worker row-group. Size grid to ~output_blocks_per_mpc
            // resident blocks per AP, distributed evenly across the 16 experts.
            constexpr int output_blocks_per_mpc = 8;
            int blocks_per_expert =
                (mpc * output_blocks_per_mpc + num_experts_per_rank - 1)
                / num_experts_per_rank;
            int total_workers = blocks_per_expert * num_experts_per_rank;
            if (tpr == 256) {
                if (num_experts_per_rank == 8) {
                    moe_scatter_dynamic_quant_output_block_4096<scalar_t, out_t, 256, 8>
                        <<<total_workers, 256, 0, stream>>>(
                            hidden_status, moe_weights, smooth_scale,
                            scatter_tokens, scatter_per_token_scale,
                            packed_route_workspace,
                            experts_token_count, experts_token_start,
                            output_capacity, blocks_per_expert);
                } else if (num_experts_per_rank == 16) {
                    moe_scatter_dynamic_quant_output_block_4096<scalar_t, out_t, 256, 16>
                        <<<total_workers, 256, 0, stream>>>(
                            hidden_status, moe_weights, smooth_scale,
                            scatter_tokens, scatter_per_token_scale,
                            packed_route_workspace,
                            experts_token_count, experts_token_start,
                            output_capacity, blocks_per_expert);
                }
            }
            else {
                if (num_experts_per_rank == 8) {
                    moe_scatter_dynamic_quant_output_block_4096<scalar_t, out_t, 128, 8>
                        <<<total_workers, 128, 0, stream>>>(
                            hidden_status, moe_weights, smooth_scale,
                            scatter_tokens, scatter_per_token_scale,
                            packed_route_workspace,
                            experts_token_count, experts_token_start,
                            output_capacity, blocks_per_expert);
                } else if (num_experts_per_rank == 16) {
                    moe_scatter_dynamic_quant_output_block_4096<scalar_t, out_t, 128, 16>
                        <<<total_workers, 128, 0, stream>>>(
                            hidden_status, moe_weights, smooth_scale,
                            scatter_tokens, scatter_per_token_scale,
                            packed_route_workspace,
                            experts_token_count, experts_token_start,
                            output_capacity, blocks_per_expert);
                }
            }
            return;
        }

        // Smooth-in-shared wave kernel (MSDQ_TPR = 641/642/644 -> WPB 1/2/4).
        // blocks_per_expert BLOCKS per expert, WPB waves each; the 4096 smooth
        // channels live in shared instead of a 64-reg array to lift occupancy.
        if (tpr == 641 || tpr == 642 || tpr == 644) {
            constexpr int output_blocks_per_mpc = 32;
            const int wpb = tpr == 644 ? 4 : (tpr == 642 ? 2 : 1);
            int blocks_per_expert =
                (mpc * output_blocks_per_mpc / wpb + num_experts_per_rank - 1)
                / num_experts_per_rank;
            if (blocks_per_expert < 1) blocks_per_expert = 1;
            int total_blocks = blocks_per_expert * num_experts_per_rank;
            #define LAUNCH_SMSH(WPB, NUM_EXPERTS) \
                moe_scatter_dynamic_quant_output_wave_smsh_4096<scalar_t, out_t, WPB, NUM_EXPERTS> \
                    <<<total_blocks, kWave * WPB, 0, stream>>>( \
                        hidden_status, moe_weights, smooth_scale, \
                        scatter_tokens, scatter_per_token_scale, \
                        packed_route_workspace, \
                        experts_token_count, experts_token_start, \
                        output_capacity, blocks_per_expert)
            if (num_experts_per_rank == 8) {                        
                if (tpr == 644)      LAUNCH_SMSH(4, 8);
                else if (tpr == 642) LAUNCH_SMSH(2, 8);
                else                 LAUNCH_SMSH(1, 8);
            } else if (num_experts_per_rank == 16) {
                if (tpr == 644)      LAUNCH_SMSH(4, 16);
                else if (tpr == 642) LAUNCH_SMSH(2, 16);
                else                 LAUNCH_SMSH(1, 16);
            }
            #undef LAUNCH_SMSH
            return;
        }

        // Wave-per-row output kernel: one 64-thread wave produces one output row,
        // so the per-row abs-max is a pure warp __shfl_xor reduction with no
        // __syncthreads. Grid is sized to give each AP many resident waves.
        // Adaptive grid sizing. Each expert is served by `blocks_per_expert`
        // worker-waves that stride over its output rows. Large T => many rows
        // per expert => it pays to launch more resident waves per AP to hide
        // HBM latency; small/mid T keeps the proven baseline (output_blocks_
        // per_mpc = 32 -> 56 blocks/expert on 28 APs), so no small-shape
        // regression. Measured: ramping up to ~192 blocks-per-mpc lifts the
        // large-T tail +6-10% (T>=16384) and gains flatten past that, so we
        // cap there. rows_per_expert ~ total_routes / 16.
        // MSDQ_OBPM (>0) forces a fixed output_blocks_per_mpc for tuning.
        static const int obpm_override = [](){
            const char* e = getenv("MSDQ_OBPM");
            return e ? atoi(e) : 0;
        }();
        int blocks_per_expert;
        if (obpm_override > 0) {
            blocks_per_expert =
                (mpc * obpm_override + num_experts_per_rank - 1) / num_experts_per_rank;
        } else {
            const int total_routes    = num_tokens * kFastScatterTopK;
            const int rows_per_expert =
                (total_routes + num_experts_per_rank - 1) / num_experts_per_rank;
            const int base_bpe = (mpc * 32  + num_experts_per_rank - 1) / num_experts_per_rank;
            const int max_bpe  = (mpc * 192 + num_experts_per_rank - 1) / num_experts_per_rank;
            blocks_per_expert = rows_per_expert / 12;
            if (blocks_per_expert < base_bpe) blocks_per_expert = base_bpe;
            if (blocks_per_expert > max_bpe)  blocks_per_expert = max_bpe;
        }
        if (blocks_per_expert < 1) blocks_per_expert = 1;
        int total_workers = blocks_per_expert * num_experts_per_rank;

        constexpr int wave_items = kFastScatterHidden / kWave;  // 64
        if (num_experts_per_rank == 8) {
            moe_scatter_dynamic_quant_output_wave_4096<scalar_t, out_t, wave_items, 1, 8>
                <<<total_workers, kWave, 0, stream>>>(
                    hidden_status, moe_weights, smooth_scale,
                    scatter_tokens, scatter_per_token_scale,
                    packed_route_workspace,
                    experts_token_count, experts_token_start,
                    output_capacity, blocks_per_expert);
        } else if (num_experts_per_rank == 16) {
            moe_scatter_dynamic_quant_output_wave_4096<scalar_t, out_t, wave_items, 1, 16>
                <<<total_workers, kWave, 0, stream>>>(
                    hidden_status, moe_weights, smooth_scale,
                    scatter_tokens, scatter_per_token_scale,
                    packed_route_workspace,
                    experts_token_count, experts_token_start,
                    output_capacity, blocks_per_expert);
        }
        return;
    }
    constexpr int max_experts = 256;
    const int total_num = num_tokens * topk;
    const int num_shared_total_tokens = num_shared_experts_per_rank * shared_tokens_per_sp;

    if (num_experts_per_rank == 8) {
        moe_align_token_offset<8><<<1, 512, 0, stream>>>(
            selected_experts, scatter_tokens_offset, experts_token_count,
            experts_token_start, topk, num_tokens, num_experts_per_rank,
            shared_tokens_per_sp, num_shared_experts_per_rank);
    } else if (num_experts_per_rank == 16) {
        moe_align_token_offset<16><<<1, 512, 0, stream>>>(
            selected_experts, scatter_tokens_offset, experts_token_count,
            experts_token_start, topk, num_tokens, num_experts_per_rank,
            shared_tokens_per_sp, num_shared_experts_per_rank);
    }

    if (total_num < 128)       stable_scatter_offset<64>  <<<num_experts_per_rank,  64,0,stream>>>(scatter_tokens_offset, experts_token_count, experts_token_start, selected_experts, num_tokens, topk, shared_tokens_per_sp, num_shared_experts_per_rank);
    else if (total_num < 512)  stable_scatter_offset<128> <<<num_experts_per_rank, 128,0,stream>>>(scatter_tokens_offset, experts_token_count, experts_token_start, selected_experts, num_tokens, topk, shared_tokens_per_sp, num_shared_experts_per_rank);
    else if (total_num < 4096) stable_scatter_offset<256> <<<num_experts_per_rank, 256,0,stream>>>(scatter_tokens_offset, experts_token_count, experts_token_start, selected_experts, num_tokens, topk, shared_tokens_per_sp, num_shared_experts_per_rank);
    else                       stable_scatter_offset<512> <<<num_experts_per_rank, 512,0,stream>>>(scatter_tokens_offset, experts_token_count, experts_token_start, selected_experts, num_tokens, topk, shared_tokens_per_sp, num_shared_experts_per_rank);
    
    const int quant_grid = output_capacity;
    moe_scatter_dynamic_quant_kernel<scalar_t, out_t><<<quant_grid, 512, 0, stream>>>(
         hidden_status, selected_experts, moe_weights, smooth_scale,
         scatter_tokens, scatter_per_token_scale, scatter_tokens_offset,
         experts_token_start, topk, hidden_size, num_experts_per_rank,
         num_shared_total_tokens,
         num_shared_experts_per_rank,
         shared_tokens_per_sp,
         num_tokens);
}

void moe_scatter_dynamic_quant(
        at::Tensor hidden_status, at::Tensor selected_experts,
        at::Tensor moe_weights, at::Tensor smooth_scale,
        at::Tensor scatter_tokens, at::Tensor scatter_per_token_scale,
        at::Tensor scatter_tokens_offset,
        at::Tensor experts_token_count, at::Tensor experts_token_start,
        const int experts_per_rank,
        const int shared_experts_per_rank,
        const int shared_tokens_per_sp) {
      DEBUG_TRACE_PARAMS(hidden_status, selected_experts, moe_weights, smooth_scale, scatter_tokens, scatter_per_token_scale, scatter_tokens_offset, experts_token_count, experts_token_start, experts_per_rank, shared_experts_per_rank, shared_tokens_per_sp);
  DEBUG_DUMP_PARAMS(hidden_status, selected_experts, moe_weights, smooth_scale, scatter_tokens, scatter_per_token_scale, scatter_tokens_offset, experts_token_count, experts_token_start, experts_per_rank, shared_experts_per_rank, shared_tokens_per_sp);

            CHECK_DEVICE(hidden_status);
    CHECK_DEVICE(selected_experts);
    CHECK_DEVICE(smooth_scale);
    CHECK_CONTIGUOUS(hidden_status);
    CHECK_CONTIGUOUS(selected_experts);
    CHECK_CONTIGUOUS(smooth_scale);

    const int hidden_size = hidden_status.size(-1);
    const int num_tokens  = hidden_status.numel() / hidden_size;
    const int topk        = selected_experts.size(-1);

    const int scatter_capacity = scatter_tokens.numel() / hidden_size;
    const int scale_capacity   = scatter_per_token_scale.numel();
    const int offset_capacity  = scatter_tokens_offset.numel();
    const int output_capacity  =
        std::min({scatter_capacity, scale_capacity, offset_capacity});

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    TORCH_CHECK(hidden_status.dtype() == at::ScalarType::BFloat16,
                "Only bfloat16 input is supported");
    at::Tensor packed_ws = at::empty(
        {offset_capacity},
        at::TensorOptions().dtype(at::kInt).device(hidden_status.device()));

#define LAUNCH_SCATTER(OUT_T, OUT_PTR)                                        \
    launch_moe_scatter_dynamic_quant_kernel<bfloat16, OUT_T>(                 \
        reinterpret_cast<bfloat16*>(hidden_status.data_ptr<at::BFloat16>()),  \
        selected_experts.data_ptr<int>(), moe_weights.data_ptr<float>(),      \
        smooth_scale.data_ptr<float>(), (OUT_PTR),                            \
        scatter_per_token_scale.data_ptr<float>(),                            \
        scatter_tokens_offset.data_ptr<int>(),                                \
        experts_token_count.data_ptr<int>(),                                  \
        experts_token_start.data_ptr<int>(), packed_ws.data_ptr<int>(),       \
        shared_experts_per_rank, hidden_size, num_tokens, output_capacity,    \
        experts_per_rank, topk, shared_tokens_per_sp, stream)

    if (scatter_tokens.dtype() == at::ScalarType::Char) {
        LAUNCH_SCATTER(int8_t, scatter_tokens.data_ptr<int8_t>());
    } else if (scatter_tokens.dtype() == at::ScalarType::Float8_e4m3fn) {
        LAUNCH_SCATTER(__maca_fp8_e4m3,
            reinterpret_cast<__maca_fp8_e4m3*>(scatter_tokens.data_ptr()));
    } else {
        TORCH_CHECK(false, "scatter_tokens must be int8 or float8_e4m3fn");
    }
}
