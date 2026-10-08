// Single-pass register-based 4096-bin histogram TopK for MetaX C500/C280.
// Adapted from mcoplib/op/vllm/topk_histogram_4096.cuh, rewritten for
// warp_size = 64 (MACA) instead of 32 (NVIDIA).
//
// Single HBM pass: each thread loads its share of the row into registers,
// builds a 4096-bin (12-bit FP16-key) coarse histogram in shared memory,
// scans the histogram to find the threshold bin, then scatters from registers
// — no second HBM read. Tie-breaking via 4-round FP32 radix on the threshold
// bin's candidates (also register/staged, no HBM re-read).
//
// Capacity: kThreadsPerBlock * kVecsPerThread * 4 = 1024 * 4 * 4 = 16384
// elements max. Caller must fall back to the 2-pass radix path for longer rows.

#ifndef TOPK_V1_HISTOGRAM_4096_CUH_
#define TOPK_V1_HISTOGRAM_4096_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace topk_v1_hist4096 {

// C500 / C280 has warp_size = 64.
constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t kWarpSize = 64;
constexpr uint32_t kNumWarps = kBlockSize / kWarpSize;  // 16
constexpr uint32_t RADIX = 256;
// 4 float4/thread = 16 floats/thread. kMaxLen = 4*4*1024 = 16384.
// Larger kVecsPerThread (8, 16) causes RF spills on C280 (64 regs/thread
// at 1024 threads = 1 block/SM; going higher spills to local mem, killing
// perf in tests: bs=4096 seq=16384 dropped from 410 to 125 GB/s at
// kVecsPerThread=16). Keep at 4; longer rows use the 2-pass radix path.
constexpr uint32_t kVecsPerThread = 4;
constexpr uint32_t kMaxLen = kVecsPerThread * 4 * kBlockSize;  // 16384

struct alignas(16) MatchBin {
  uint32_t bin, above_count, equal_count;
};
struct alignas(8) Tie {
  uint32_t idx;
  float score;
};

__device__ __forceinline__ uint32_t convert_to_uint32_v2(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// 12-bit bin from FP16 sign-magnitude key (ascending: high values -> high bins).
// FP16's 5-bit exponent + 10-bit mantissa gives finer 12-bit binning
// (5 exp + 7 mantissa) than FP32 high-12 (8 exp + 4 mantissa) — fewer
// candidates land in the threshold bin, so tie-breaking is cheaper.
template <uint32_t kBits>
__device__ __forceinline__ uint32_t extract_coarse_bin_N(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return key >> (16 - kBits);
}

// Warp inclusive prefix sum for warp_size = 64 (6 rounds of shuffle-up).
__device__ __forceinline__ uint32_t warp_inclusive_sum64(uint32_t lane,
                                                         uint32_t v) {
#pragma unroll
  for (uint32_t o = 1; o < kWarpSize; o *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFFFFFFFFFFULL, v, o);
    if (lane >= o) v += n;
  }
  return v;
}

// Warp-wide reduce (butterfly xor) for warp_size = 64.
__device__ __forceinline__ uint32_t warp_reduce_sum_full64(uint32_t v) {
#pragma unroll
  for (uint32_t mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    v += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, v, mask);
  }
  return v;
}

// 4-round FP32 radix tie-breaking. ties[] is in shared memory, num_ties
// elements. Each thread handles one tie (so num_ties <= kBlockSize).
template <uint32_t TopK>
__device__ void tie_handle(const Tie* ties, uint32_t num_ties,
                           uint32_t num_above, int32_t* output, void* _smem) {
  struct TS {
    alignas(128) uint32_t counter;
    alignas(128) MatchBin match;
    uint32_t histogram[RADIX];
    uint32_t warp_sum[kNumWarps];
  };
  auto* s = static_cast<TS*>(_smem);
  const auto tx = threadIdx.x;
  const auto li = tx % kWarpSize, wi = tx / kWarpSize;

  const bool has = tx < num_ties;
  const auto tie = has ? ties[tx] : Tie{0, 0.0f};
  const uint32_t key = convert_to_uint32_v2(tie.score);

  bool active = has;
  uint32_t remain = TopK - num_above;
  uint32_t wpos = TopK;
  if (tx == 0) s->counter = 0;
  __syncthreads();

#pragma unroll
  for (int r = 0; r < 4; r++) {
    const uint32_t sh = 24 - r * 8;
    const uint32_t bin = (key >> sh) & 0xFF;

    if (tx < RADIX) s->histogram[tx] = 0;
    __syncthreads();
    if (active) atomicAdd(&s->histogram[bin], 1);
    __syncthreads();

    // 256-bin prefix scan: 4 warps cover 256 bins (64 bins/warp).
    // Each thread in the first 4 warps handles one bin.
    uint32_t hv = 0, wi2 = 0;
    if (tx < RADIX) {
      hv = s->histogram[tx];
      wi2 = warp_inclusive_sum64(li, hv);
      if (li == kWarpSize - 1) s->warp_sum[wi] = wi2;
    }
    __syncthreads();

    if (tx < RADIX) {
      // wi = tx / 64, li = tx % 64. Cross-warp prefix.
      const auto tmp = (li < (RADIX / kWarpSize)) ? s->warp_sum[li] : 0;
      // sum of warps 0..wi-1
      const auto inter = warp_reduce_sum_full64(li < wi ? tmp : 0);
      const auto tot = warp_reduce_sum_full64(tmp);
      const auto above = tot - (inter + wi2);
      if (above < remain && above + hv >= remain) {
        s->match = {tx, above, remain - above};
      }
    }
    __syncthreads();

    const auto [thr, na, _] = s->match;
    if (active) {
      if (bin > thr) {
        wpos = num_above + atomicAdd(&s->counter, 1);
        active = false;
      } else if (bin < thr) {
        active = false;
      } else if (r == 3) {
        wpos = TopK - atomicAdd(&s->match.equal_count, -1u);
      }
    }
    remain -= na;
    if (!remain) break;
  }
  if (wpos < TopK) output[wpos] = static_cast<int32_t>(tie.idx);
}

// Main single-pass 4096-bin TopK. HIST_BITS = 12 (4096 bins).
template <uint32_t TopK, uint32_t HIST_BITS = 12>
__device__ void histogram_4096_topk(const float* __restrict__ scores,
                                    int32_t* __restrict__ output,
                                    uint32_t length, void* _smem) {
  constexpr uint32_t HIST_BINS = 1 << HIST_BITS;
  constexpr uint32_t ITEMS_PER_THREAD = HIST_BINS / kBlockSize;  // 4
  static_assert(HIST_BINS >= kBlockSize, "HIST_BITS must give >= kBlockSize bins");
  static_assert(kMaxLen >= HIST_BINS, "kMaxLen too small");

  struct Smem {
    alignas(128) uint32_t counter_gt;
    alignas(128) uint32_t counter_eq;
    MatchBin match;
    uint32_t warp_sum[kNumWarps];
    uint32_t scan_buf[kBlockSize];  // 1024-entry prefix-scan scratch
    union {
      uint32_t histogram[HIST_BINS];
      Tie tie_buffer[TopK > 1024 ? TopK : 1024];
    };
  };
  auto* smem = static_cast<Smem*>(_smem);
  const auto tx = threadIdx.x;
  const auto lane_id = tx % kWarpSize;
  const auto warp_id = tx / kWarpSize;

  // Phase 1: Load all data into RF + build histogram.
  float4 vecs[kVecsPerThread];
  // Zero the histogram (4 bins per thread, 1 uint4 write).
  if constexpr (ITEMS_PER_THREAD >= 4) {
#pragma unroll
    for (uint32_t i = 0; i < ITEMS_PER_THREAD / 4; i++) {
      reinterpret_cast<uint4*>(smem->histogram)[tx * (ITEMS_PER_THREAD / 4) + i] =
          make_uint4(0, 0, 0, 0);
    }
  } else {
    if (tx < HIST_BINS) smem->histogram[tx] = 0;
  }
  if (tx == 0) {
    smem->counter_gt = 0;
    smem->counter_eq = 0;
    // Sentinel: HIST_BINS means "no match found yet". The scan below will
    // overwrite this with the real threshold bin. If no thread matches
    // (shouldn't happen for length >= TopK), the scatter phase will treat
    // thr_bin = HIST_BINS as "above everything" -> 0 elements selected,
    // which is at least deterministic rather than garbage.
    smem->match = {HIST_BINS, 0, 0};
  }

  // Load up to kVecsPerThread float4 per thread. If length < kMaxLen, the
  // predicated branch leaves the trailing elements as 0.0f (which maps to a
  // valid low bin — harmless since we only output indices < length).
  const bool row_aligned = (reinterpret_cast<uintptr_t>(scores) & 0xFu) == 0;
  const float kNegInf = __uint_as_float(0xFF800000u);
#pragma unroll
  for (uint32_t v = 0; v < kVecsPerThread; v++) {
    const uint32_t base = (tx + v * kBlockSize) * 4;
    if (base < length) {
      if (row_aligned && base + 3 < length) {
        vecs[v] = *reinterpret_cast<const float4*>(scores + base);
      } else {
        // Per-element predicated load.
        vecs[v].x = (base + 0 < length) ? scores[base + 0] : kNegInf;
        vecs[v].y = (base + 1 < length) ? scores[base + 1] : kNegInf;
        vecs[v].z = (base + 2 < length) ? scores[base + 2] : kNegInf;
        vecs[v].w = (base + 3 < length) ? scores[base + 3] : kNegInf;
      }
    } else {
      vecs[v] = {kNegInf, kNegInf, kNegInf, kNegInf};
    }
  }
  __syncthreads();

  // Build histogram from RF.
  bool done = false;
#pragma unroll
  for (uint32_t v = 0; v < kVecsPerThread && !done; v++) {
    const float* elems = reinterpret_cast<const float*>(&vecs[v]);
#pragma unroll
    for (uint32_t e = 0; e < 4 && !done; e++) {
      const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
      if (idx >= length) {
        done = true;
      } else {
        atomicAdd(&smem->histogram[extract_coarse_bin_N<HIST_BITS>(elems[e])], 1);
      }
    }
  }
  __syncthreads();

  // Phase 2: 4096-bin exclusive prefix scan to find threshold bin.
  // 4 bins per thread (ITEMS_PER_THREAD=4). 1024 threads * 4 = 4096 bins.
  //
  // Approach: smem-based Hillis-Steele scan, robust to any warp size.
  //   1. Each thread reads its 4 bins into orig[4]; writes local_sum to a
  //      1024-entry scan buffer.
  //   2. 1024-element exclusive prefix scan over local_sums (10 rounds of
  //      strided add in smem).
  //   3. Each thread's exclusive prefix at bin i =
  //        scan_buf[tx] + sum(orig[0..i-1]).
  // Avoids __shfl entirely — works on any warp width (MACA warp=64, NVIDIA 32).
  uint32_t orig[ITEMS_PER_THREAD];
  uint32_t local_sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    orig[i] = smem->histogram[tx * ITEMS_PER_THREAD + i];
    local_sum += orig[i];
  }
  uint32_t* scan_buf = smem->scan_buf;  // separate scratch, doesn't touch histogram/tie_buffer
  scan_buf[tx] = local_sum;
  __syncthreads();

  // Hillis-Steele exclusive prefix scan over 1024 elements.
  // Each round: scan_buf[tx] += scan_buf[tx - stride] (if tx >= stride).
#pragma unroll
  for (uint32_t stride = 1; stride < kBlockSize; stride <<= 1) {
    uint32_t v = 0;
    if (tx >= stride) v = scan_buf[tx - stride];
    __syncthreads();
    if (tx >= stride) scan_buf[tx] += v;
    // For inclusive->exclusive: shift later. Actually we want inclusive here
    // and subtract local_sum at the end to get exclusive.
    __syncthreads();
  }
  // scan_buf[tx] is now inclusive prefix sum of local_sums.
  uint32_t exclusive_thread_prefix = scan_buf[tx] - local_sum;

  uint32_t prefix = exclusive_thread_prefix;
#pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    const auto cum_here = prefix;  // exclusive prefix before bin i
    const auto above = length - cum_here - orig[i];
    if (above < TopK && above + orig[i] >= TopK) {
      smem->match = {.bin = tx * ITEMS_PER_THREAD + i,
                     .above_count = above,
                     .equal_count = orig[i]};
    }
    prefix += orig[i];
  }
  __syncthreads();

  // Phase 3: Scatter from registers.
  const auto [thr_bin, num_above, num_equal] = smem->match;
  const bool need_tie = (num_equal + num_above > TopK);

  done = false;
#pragma unroll
  for (uint32_t v = 0; v < kVecsPerThread && !done; v++) {
    const float* elems = reinterpret_cast<const float*>(&vecs[v]);
#pragma unroll
    for (uint32_t e = 0; e < 4 && !done; e++) {
      const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
      if (idx >= length) {
        done = true;
      } else {
        const uint32_t bin = extract_coarse_bin_N<HIST_BITS>(elems[e]);
        if (bin > thr_bin) {
          // Guard against counter_gt overflow (defensive: should never exceed
          // num_above < TopK if the threshold is correct, but a scan bug could
          // otherwise write past s_topk_indices[TopK] and corrupt smem).
          const auto pos = ::atomicAdd(&smem->counter_gt, 1);
          if (pos < TopK) output[pos] = static_cast<int32_t>(idx);
        } else if (bin == thr_bin) {
          const auto pos = ::atomicAdd(&smem->counter_eq, 1);
          if (!need_tie) {
            if (pos + num_above < TopK) {
              output[pos + num_above] = static_cast<int32_t>(idx);
            }
          } else {
            if (pos < TopK) {
              smem->tie_buffer[pos] = {idx, elems[e]};
            }
          }
        }
      }
    }
  }

  if (!need_tie) return;
  __syncthreads();

  const uint32_t num_ties = (num_equal < TopK) ? num_equal : TopK;
  const uint32_t topk_remain = TopK - num_above;

  // Tie-breaking. NOTE: the warp-ballot path ranks target = tie_buffer[warp_id],
  // so it requires kNumWarps >= num_ties. With warp=64 and 1024 threads we have
  // only 16 warps — so the ballot path only works for num_ties <= 16. For
  // 17..1024 ties, fall back to the 4-round radix tie_handle (handles up to
  // kBlockSize=1024 ties, one per thread).
  if (num_ties <= kNumWarps) {
    if (lane_id >= num_ties || warp_id >= num_ties) return;
    const uint64_t mask = (num_ties >= 64)
                              ? 0xFFFFFFFFFFFFFFFFULL
                              : ((1ULL << num_ties) - 1ULL);
    const auto tie = smem->tie_buffer[lane_id];
    const auto target = smem->tie_buffer[warp_id];
    const bool pred = (tie.score > target.score) ||
                      (tie.score == target.score && tie.idx < target.idx);
    const auto rank = static_cast<uint32_t>(__popcll(__ballot_sync(mask, pred)));
    if (lane_id == 0 && rank < topk_remain) {
      output[num_above + rank] = static_cast<int32_t>(target.idx);
    }
  } else {
    tie_handle<TopK>(smem->tie_buffer, num_ties, num_above, output, smem);
  }
}

}  // namespace topk_v1_hist4096

#endif  // TOPK_V1_HISTOGRAM_4096_CUH_
