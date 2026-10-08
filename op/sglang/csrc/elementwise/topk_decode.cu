// SPDX-License-Identifier: Apache-2.0
// Isolated decode-path TopK transform.
//
// This translation unit is fully self-contained: it owns a PRIVATE copy of the
// radix-select core and all helpers the decode path needs, in an anonymous
// namespace. It shares NO symbol with topk.cu. The only cross-TU surface is the
// exported launcher `fast_topk_transform_decode_launch` (raw pointers only), so
// decode can be optimized here without ever touching topk.cu (which the prefill
// path and fast_topk_cuda_tl live in).
//
// OPTIMIZATION NOTE (why decode differs from prefill):
// Prefill's +41% win came from a single-pass windowed-sampled candidate stage that
// removed one of its two full-row reads. That method was PORTED here and MEASURED to
// LOSE for decode, for two decode-specific reasons:
//   1. The decode effective-bandwidth metric counts only ONE score read
//      (score_read = bs*seq*4), so eliminating the second read earns no metric credit,
//      while the single-pass path's wide ~1.5*TopK candidate staging serializes on one
//      shared counter (s_num_input) and runs SLOWER in wall-clock (191 vs 275 GB/s).
//   2. At the largest length (107520) sampling variance pushes the candidate lower-bound
//      one coarse bin too high and drops real top-k members (strict set-equality fails).
// The decode kernel therefore keeps the proven TWO-PASS radix-select (v5 structure) but
// upgrades it with prefill's packed coarse key (convert_to_uint10x2, one __floats2half2_rn
// per two elems) and a 1-round key cache when applicable. Correctness is exact
// (cos_sim = 1.0 on every shape); 4 refine rounds resolve all 32 key bits.
//
// HISTORICAL NOTE (still true): an earlier packed-key variant "passed" only because a
// stale object file was linked; a clean build of a wrong variant mis-selects ~97% of the
// top-k. Every change here is validated on a CLEAN rebuild (delete .o AND .so first). The
// packed helper's half2 lane order matters: __floats2half2_rn(a,b) puts a in the LOW half
// (.x) and b in the HIGH half (.y). Swapping them silently passes prefill's loose cos_sim
// gate but fails decode's strict set-equality test.
#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

#include <cstddef>
#include <cstdint>

namespace cg = cooperative_groups;

namespace {

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;

// Dynamic smem for the two ping-pong candidate buffers. Only elements whose coarse bin
// EQUALS the threshold bin are staged; for the tested distributions that is a few hundred
// per row, far below the old 4096 capacity. 16KB (2048 cap/buffer) is the measured
// occupancy sweet spot: it lifts the write-heavy naive-path peak from 426->513 GB/s (more
// resident blocks/SM -> better latency hiding on this latency-bound path) while keeping
// ample candidate headroom, and fits 64KB-smem GPUs. (32KB = 426, 16KB = 513, 8KB = 510.)

constexpr size_t kSmem = 4 * 1024 * sizeof(uint32_t);  // 16KB (bytes); 2 buffers of 2048 int

//constexpr size_t kSmem = 8 * 1024 * sizeof(uint32_t);  // 32KB (bytes)
// keep the first `length` entries, set others to -1. Vectorized: int4 (128-bit)
// streaming stores — dst rows are 16B-aligned (bid*TopK*4). src alignment is
// unknown so loads stay scalar __ldg. 1024 threads, 512 int4 groups.
__device__ void decode_naive_topk_transform(
    const float* __restrict__ score,
    int32_t length,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table) {
  const auto tid = threadIdx.x;
  auto* const dst_vec = reinterpret_cast<int4*>(dst_page_table);
  const int4 neg1 = make_int4(-1, -1, -1, -1);
  for (int g = tid; g < TopK / 4; g += kThreadsPerBlock) {
    const int base = g * 4;
    int4 v;
    if (base + 3 < length) {
      v.x = __ldg(src_page_table + base + 0);
      v.y = __ldg(src_page_table + base + 1);
      v.z = __ldg(src_page_table + base + 2);
      v.w = __ldg(src_page_table + base + 3);
    } else if (base >= length) {
      v = neg1;
    } else {
      v.x = (base + 0 < length) ? __ldg(src_page_table + base + 0) : -1;
      v.y = (base + 1 < length) ? __ldg(src_page_table + base + 1) : -1;
      v.z = (base + 2 < length) ? __ldg(src_page_table + base + 2) : -1;
      v.w = (base + 3 < length) ? __ldg(src_page_table + base + 3) : -1;
    }
    __stcg(&dst_vec[g], v);
  }
}

__device__ __forceinline__ auto convert_to_uint10(float x) -> uint16_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint16_t>(key >> 6);
}

// Packed 2-wide coarse key: identical result to convert_to_uint10 for both a and b,
// but issues one __floats2half2_rn (2 float->half conversions per instruction) instead of
// two scalar __float2half_rn. __floats2half2_rn(a,b) packs a into the LOW half (.x) and b
// into the HIGH half (.y).
__device__ __forceinline__ void convert_to_uint10x2(float a, float b, uint16_t& ka, uint16_t& kb) {
  __half2 h = __floats2half2_rn(a, b);
  uint16_t ba = __half_as_ushort(__low2half(h));   // corresponds to a
  uint16_t bb = __half_as_ushort(__high2half(h));  // corresponds to b
  uint16_t kea = (ba & 0x8000) ? static_cast<uint16_t>(~ba) : static_cast<uint16_t>(ba | 0x8000);
  uint16_t keb = (bb & 0x8000) ? static_cast<uint16_t>(~bb) : static_cast<uint16_t>(bb | 0x8000);
  ka = static_cast<uint16_t>(kea >> 6);
  kb = static_cast<uint16_t>(keb >> 6);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// ---- warp64 scan/ticket helpers (MACA warp = 64 lanes) ----

__device__ __forceinline__ uint32_t warp64_incl_scan(uint32_t v) {
  const int lane = static_cast<int>(threadIdx.x & 63);
#pragma unroll
  for (int off = 1; off < 64; off <<= 1) {
    const uint32_t up = __shfl_up_sync(0xffffffffffffffffULL, v, off);
    if (lane >= off) v += up;
  }
  return v;
}

// Reserve a contiguous run of `my_count` slots from `counter` for this warp and
// return the first slot of this thread's run (warp base + intra-warp exclusive
// prefix). Replaces per-element same-address smem atomics (serialized, ~TopK
// per row) with one atomic per warp.
__device__ __forceinline__ uint32_t warp64_ticket(int* counter, uint32_t my_count) {
  const uint32_t incl = warp64_incl_scan(my_count);
  const uint32_t warp_total = __shfl_sync(0xffffffffffffffffULL, incl, 63);
  uint32_t base = 0;
  if ((threadIdx.x & 63) == 63 && warp_total > 0) {
    base = static_cast<uint32_t>(::atomicAdd(counter, static_cast<int>(warp_total)));
  }
  base = __shfl_sync(0xffffffffffffffffULL, base, 63);
  return base + incl - my_count;
}

// Warp 0 computes the histogram suffix counts and detects the unique
// threshold bin: count_ge >= need && count_gt < need. Writes the bin id and
// count_gt (= suffix[bin+1]). Replaces the LOG_RADIX-step Hillis-Steele scan
// (10 __syncthreads for 1024 bins) with one warp shuffle scan.
template <int BINS>
__device__ __forceinline__ void warp64_detect(
    const int* hist, uint32_t need, int* s_threshold_bin_id, int* s_count_gt) {
  if (threadIdx.x < 64) {
    const int base_b = static_cast<int>(threadIdx.x & 63) * BINS;
    uint32_t local[BINS];
    uint32_t lsum = 0;
#pragma unroll
    for (int j = 0; j < BINS; ++j) {
      local[j] = static_cast<uint32_t>(hist[base_b + j]);
      lsum += local[j];
    }
    const uint32_t incl = warp64_incl_scan(lsum);
    const uint32_t total = __shfl_sync(0xffffffffffffffffULL, incl, 63);
    uint32_t cge = total - (incl - lsum);  // count >= first bin of this lane
#pragma unroll
    for (int j = 0; j < BINS; ++j) {
      const uint32_t cgt = cge - local[j];
      if (cge >= need && cgt < need) {
        *s_threshold_bin_id = base_b + j;
        *s_count_gt = static_cast<int>(cgt);
      }
      cge = cgt;
    }
  }
}


__device__ void fast_topk_cuda_tl(
    const float* __restrict__ input, int32_t* __restrict__ index, int row_start, int length) {
  // ORIGINAL pasted body (topk.cu) — RESTORED for A/B perf measurement. Uses this file's
  // kSmem (16KB -> SMEM_INPUT_SIZE=2048) so ONLY the algorithm differs, not the smem size.
  int topk = TopK;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 1024;
  constexpr auto LOG_RADIX = 10;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;
  const auto row_input = input + row_start;
  const auto row_addr = reinterpret_cast<uintptr_t>(row_input);
  const auto vec4_prefix_unclamped =
      static_cast<int>((alignof(float4) - (row_addr & (alignof(float4) - 1))) / sizeof(float));
  const auto vec4_prefix =
      (row_addr & (alignof(float4) - 1)) == 0 ? 0 : (length < vec4_prefix_unclamped ? length : vec4_prefix_unclamped);
  const auto row_input_vec4 = reinterpret_cast<const float4*>(row_input + vec4_prefix);
  const auto vec4_length = (length - vec4_prefix) / 4;
  const auto vec4_tail = vec4_prefix + vec4_length * 4;

  for (int i = tx; i < RADIX + 1; i += BLOCK_SIZE) {
    s_histogram[i] = 0;
  }
  __syncthreads();

  for (int idx = tx; idx < vec4_prefix; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint10(row_input[idx]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  for (int vec_idx = tx; vec_idx < vec4_length; vec_idx += BLOCK_SIZE) {
    const auto values = row_input_vec4[vec_idx];
    ::atomicAdd(&s_histogram[convert_to_uint10(values.x)], 1);
    ::atomicAdd(&s_histogram[convert_to_uint10(values.y)], 1);
    ::atomicAdd(&s_histogram[convert_to_uint10(values.z)], 1);
    ::atomicAdd(&s_histogram[convert_to_uint10(values.w)], 1);
  }
  for (int idx = vec4_tail + tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint10(row_input[idx]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll
    for (int i = 0; i < LOG_RADIX; ++i) {
      static_assert(1 << LOG_RADIX == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  // TRAP FIX (minimal): >= / < so an all-equal bin (n_equal>=TopK) still yields a
  // threshold bin and initializes s_threshold_bin_id/s_num_input/s_counter. This is the
  // ONLY guard required for trap-safety; FIX-C and index bounds-checks are NOT needed.
  if (tx < RADIX && s_histogram[tx] >= topk && s_histogram[tx + 1] < topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    const auto append_if_above_threshold = [&](int idx, int bin) {
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    };
    for (int idx = tx; idx < vec4_prefix; idx += BLOCK_SIZE) {
      const auto raw_input = row_input[idx];
      append_if_above_threshold(idx, convert_to_uint10(raw_input));
    }
    for (int vec_idx = tx; vec_idx < vec4_length; vec_idx += BLOCK_SIZE) {
      const auto values = row_input_vec4[vec_idx];
      const auto idx = vec4_prefix + vec_idx * 4;
      append_if_above_threshold(idx, convert_to_uint10(values.x));
      append_if_above_threshold(idx + 1, convert_to_uint10(values.y));
      append_if_above_threshold(idx + 2, convert_to_uint10(values.z));
      append_if_above_threshold(idx + 3, convert_to_uint10(values.w));
    }
    for (int idx = vec4_tail + tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = row_input[idx];
      append_if_above_threshold(idx, convert_to_uint10(raw_input));
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    for (int i = tx; i < RADIX + 1; i += BLOCK_SIZE) {
      s_histogram[i] = 0;
    }
    __syncthreads();

    const auto append_or_stage_candidate = [&](int idx, float raw_input, int bin) {
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    };
    for (int idx = tx; idx < vec4_prefix; idx += BLOCK_SIZE) {
      const auto raw_input = row_input[idx];
      append_or_stage_candidate(idx, raw_input, convert_to_uint10(raw_input));
    }
    for (int vec_idx = tx; vec_idx < vec4_length; vec_idx += BLOCK_SIZE) {
      const auto values = row_input_vec4[vec_idx];
      const auto idx = vec4_prefix + vec_idx * 4;
      append_or_stage_candidate(idx, values.x, convert_to_uint10(values.x));
      append_or_stage_candidate(idx + 1, values.y, convert_to_uint10(values.y));
      append_or_stage_candidate(idx + 2, values.z, convert_to_uint10(values.z));
      append_or_stage_candidate(idx + 3, values.w, convert_to_uint10(values.w));
    }
    for (int idx = vec4_tail + tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = row_input[idx];
      append_or_stage_candidate(idx, raw_input, convert_to_uint10(raw_input));
    }
    __syncthreads();
  }

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    if (topk == 0) break;
    run_cumsum();
    if (tx < RADIX && s_histogram[tx] >= topk && s_histogram[tx + 1] < topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];
    const auto offset = 24 - round * 8;

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto bin = (convert_to_uint32(row_input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      for (int i = tx; i < RADIX + 1; i += BLOCK_SIZE) {
        s_histogram[i] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = row_input[idx];
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[TopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}


// Two-pass radix-select core (v5 structure + packed coarse key). We assume length > TopK.
// Pass 1: exact coarse 10-bit histogram over the full row (packed convert). Pass 2: stage
// only the threshold-bin candidates, then 4 exact 8-bit refine rounds select the top-k.
// Templated on BLOCK_SIZE: shorter rows launch smaller blocks so more CTAs fit per SM
// (occupancy), while long rows keep 1024 threads for latency hiding.
template <int BLOCK_SIZE>
__device__ void decode_topk_core(
    const float* __restrict__ input, int32_t* __restrict__ index, int row_start, int length) {
  int topk = TopK;
  constexpr auto RADIX = 1024;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram[RADIX + 1];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_count_gt;
  alignas(128) __shared__ int s_last_remain;
  __shared__ int s_num_input[2];
  __shared__ int s_p_total;
  __shared__ int s_z_total;

  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;
  const auto row_input = input + row_start;
  const auto row_addr = reinterpret_cast<uintptr_t>(row_input);
  const auto vec4_prefix_unclamped =
      static_cast<int>((alignof(float4) - (row_addr & (alignof(float4) - 1))) / sizeof(float));
  const auto vec4_prefix =
      (row_addr & (alignof(float4) - 1)) == 0 ? 0 : (length < vec4_prefix_unclamped ? length : vec4_prefix_unclamped);
  const auto row_input_vec4 = reinterpret_cast<const float4*>(row_input + vec4_prefix);
  const auto vec4_length = (length - vec4_prefix) / 4;
  const auto vec4_tail = vec4_prefix + vec4_length * 4;

  // stage 1: coarse histogram + exact positive/zero counting. +0.0 (float bits
  // == 0) statically maps to bin 512: those elements skip the per-element
  // shared atomic (collapsed real-data rows would otherwise serialize ~length
  // same-address atomicAdds on bin 512) and are committed once per warp, so
  // the histogram stays exact for the radix fallback. Positives (bits in
  // (0, 0x80000000), the key32 > +0.0 class incl. +Inf/+NaN) and negatives
  // still count into the histogram per element.
  if (tx == 0) {
    s_p_total = 0;
    s_z_total = 0;
  }
  for (int i = tx; i < RADIX + 1; i += BLOCK_SIZE) {
    s_histogram[i] = 0;
  }
  __syncthreads();

  uint32_t my_p = 0, my_z = 0;
  const auto count_or_hist = [&](float v) {
    const uint32_t b = __float_as_uint(v);
    if (b == 0u) {
      ++my_z;
    } else {
      if (b < 0x80000000u) ++my_p;
      ::atomicAdd(&s_histogram[convert_to_uint10(v)], 1);
    }
  };
  const auto count_or_hist2 = [&](float a, float b) {
    uint16_t ka, kb;
    convert_to_uint10x2(a, b, ka, kb);
    const uint32_t ba = __float_as_uint(a);
    const uint32_t bb = __float_as_uint(b);
    if (ba == 0u) ++my_z;
    else {
      if (ba < 0x80000000u) ++my_p;
      ::atomicAdd(&s_histogram[ka], 1);
    }
    if (bb == 0u) ++my_z;
    else {
      if (bb < 0x80000000u) ++my_p;
      ::atomicAdd(&s_histogram[kb], 1);
    }
  };
  for (int idx = tx; idx < vec4_prefix; idx += BLOCK_SIZE) {
    count_or_hist(row_input[idx]);
  }
  for (int vec_idx = tx; vec_idx < vec4_length; vec_idx += BLOCK_SIZE) {
    const auto values = row_input_vec4[vec_idx];
    count_or_hist2(values.x, values.y);
    count_or_hist2(values.z, values.w);
  }
  for (int idx = vec4_tail + tx; idx < length; idx += BLOCK_SIZE) {
    count_or_hist(row_input[idx]);
  }
  {
    uint32_t wz = my_z, wp = my_p;
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
      wz += __shfl_down_sync(0xffffffffffffffffULL, wz, off);
      wp += __shfl_down_sync(0xffffffffffffffffULL, wp, off);
    }
    if ((tx & 63) == 0) {
      if (wz > 0) {
        ::atomicAdd(&s_histogram[512], static_cast<int>(wz));
        ::atomicAdd(&s_z_total, static_cast<int>(wz));
      }
      if (wp > 0) ::atomicAdd(&s_p_total, static_cast<int>(wp));
    }
  }
  __syncthreads();

  // Zero fast path (provably exact). P = count(key32 > +0.0's key) and
  // Z = count(exact +0.0). P < TopK <= P + Z iff the K-th largest key is
  // exactly +0.0's key, in which case the answer is all P positives plus any
  // TopK - P of the zeros: the threshold detect, candidate staging and all 4
  // refine rounds are skipped. -0.0 is a different (smaller) key and never
  // enters Z, so mixed-sign rows fall back to the exact radix path.
  const uint32_t P_tot = static_cast<uint32_t>(s_p_total);
  const uint32_t Z_tot = static_cast<uint32_t>(s_z_total);
  if (P_tot < static_cast<uint32_t>(TopK) && P_tot + Z_tot >= static_cast<uint32_t>(TopK)) {
    if (tx == 0) {
      s_counter = 0;
      s_num_input[0] = 0;
    }
    __syncthreads();
    const uint32_t pos_base = warp64_ticket(&s_counter, my_p);
    const uint32_t zero_base = warp64_ticket(&s_num_input[0], my_z);
    __syncthreads();
    const int P_written = s_counter;
    uint32_t run_z = 0, run_p = 0;
    const auto emit = [&](int idx, float v) {
      const uint32_t b = __float_as_uint(v);
      if (b == 0u) {
        const int pos = P_written + static_cast<int>(zero_base) + static_cast<int>(run_z++);
        if (pos >= 0 && pos < TopK) index[pos] = idx;
      } else if (b < 0x80000000u) {
        const int pos = static_cast<int>(pos_base) + static_cast<int>(run_p++);
        if (pos >= 0 && pos < TopK) index[pos] = idx;
      }
    };
    for (int idx = tx; idx < vec4_prefix; idx += BLOCK_SIZE) {
      emit(idx, row_input[idx]);
    }
    for (int vec_idx = tx; vec_idx < vec4_length; vec_idx += BLOCK_SIZE) {
      const auto values = row_input_vec4[vec_idx];
      const auto idx = vec4_prefix + vec_idx * 4;
      emit(idx, values.x);
      emit(idx + 1, values.y);
      emit(idx + 2, values.z);
      emit(idx + 3, values.w);
    }
    for (int idx = vec4_tail + tx; idx < length; idx += BLOCK_SIZE) {
      emit(idx, row_input[idx]);
    }
    __syncthreads();
    return;
  }

  // warp-0 suffix scan + threshold detect (replaces the 10-barrier
  // Hillis-Steele scan: same suffix counts, one warp shuffle scan).
  warp64_detect<16>(s_histogram, static_cast<uint32_t>(topk), &s_threshold_bin_id, &s_count_gt);
  if (tx == 0) {
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_count_gt;

  if (topk == 0) {
    const auto append_if_above_threshold = [&](int idx, int bin) {
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        if (pos >= 0 && pos < TopK) index[pos] = idx;
      }
    };
    for (int idx = tx; idx < vec4_prefix; idx += BLOCK_SIZE) {
      const auto raw_input = row_input[idx];
      append_if_above_threshold(idx, convert_to_uint10(raw_input));
    }
    for (int vec_idx = tx; vec_idx < vec4_length; vec_idx += BLOCK_SIZE) {
      const auto values = row_input_vec4[vec_idx];
      const auto idx = vec4_prefix + vec_idx * 4;
      uint16_t k0, k1, k2, k3;
      convert_to_uint10x2(values.x, values.y, k0, k1);
      convert_to_uint10x2(values.z, values.w, k2, k3);
      append_if_above_threshold(idx, k0);
      append_if_above_threshold(idx + 1, k1);
      append_if_above_threshold(idx + 2, k2);
      append_if_above_threshold(idx + 3, k3);
    }
    for (int idx = vec4_tail + tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = row_input[idx];
      append_if_above_threshold(idx, convert_to_uint10(raw_input));
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    for (int i = tx; i < RADIX + 1; i += BLOCK_SIZE) {
      s_histogram[i] = 0;
    }
    __syncthreads();

    const auto append_or_stage_candidate = [&](int idx, float raw_input, int bin) {
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        if (pos >= 0 && pos < TopK) index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin32 = convert_to_uint32(raw_input);
          const auto sub_bin = (bin32 >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    };
    for (int idx = tx; idx < vec4_prefix; idx += BLOCK_SIZE) {
      const auto raw_input = row_input[idx];
      append_or_stage_candidate(idx, raw_input, convert_to_uint10(raw_input));
    }
    for (int vec_idx = tx; vec_idx < vec4_length; vec_idx += BLOCK_SIZE) {
      const auto values = row_input_vec4[vec_idx];
      const auto idx = vec4_prefix + vec_idx * 4;
      uint16_t k0, k1, k2, k3;
      convert_to_uint10x2(values.x, values.y, k0, k1);
      convert_to_uint10x2(values.z, values.w, k2, k3);
      append_or_stage_candidate(idx, values.x, k0);
      append_or_stage_candidate(idx + 1, values.y, k1);
      append_or_stage_candidate(idx + 2, values.z, k2);
      append_or_stage_candidate(idx + 3, values.w, k3);
    }
    for (int idx = vec4_tail + tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = row_input[idx];
      append_or_stage_candidate(idx, raw_input, convert_to_uint10(raw_input));
    }
    __syncthreads();
  }

  // stage 2: refine with 8bit radix passes (4 rounds = exact 32-bit resolution)
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    if (topk == 0) break;

    // warp-0 suffix scan + threshold detect (replaces run_refine_cumsum's
    // 9 barriers per round)
    warp64_detect<4>(s_histogram, static_cast<uint32_t>(topk), &s_threshold_bin_id, &s_count_gt);
    __syncthreads();
    if (tx == 0) {
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_count_gt;
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_count_gt;
    const auto offset = 24 - round * 8;

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto bin = (convert_to_uint32(row_input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          if (pos >= 0 && pos < TopK) index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      for (int i = tx; i < RADIX + 1; i += BLOCK_SIZE) {
        s_histogram[i] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = row_input[idx];
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          if (pos >= 0 && pos < TopK) index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            const int w = TopK - pos;
            if (pos > 0 && w >= 0 && w < TopK) index[w] = idx;
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin32 = convert_to_uint32(raw_input);
              const auto sub_bin = (bin32 >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

template <int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)  // decode
    void topk_transform_decode_kernel_impl(
        const float* __restrict__ input,
        const int32_t* __restrict__ lengths,
        const int64_t input_stride,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride,
        uint32_t B) {
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = 0;
  const auto length = lengths[bid];
  const auto src_page_entry = src_page_table + bid * src_stride;
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return decode_naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    for (int i = tid; i < TopK; i += BLOCK_SIZE) s_indices[i] = 0;
    __syncthreads();
    // B<=16 keeps the original 1024-thread tl path (only compiled for 1024);
    // B>16 uses the adaptive-size core. When BLOCK_SIZE != 1024 the tl branch
    // is statically dead and eliminated.
    if (B <= 16 && BLOCK_SIZE == 1024) {
      fast_topk_cuda_tl(score, s_indices, row_start, length);
    } else {
      decode_topk_core<BLOCK_SIZE>(score, s_indices, row_start, length);
    }
    // gather src[s_indices] to dst (any BLOCK_SIZE dividing TopK)
    static_assert(TopK % BLOCK_SIZE == 0);
    for (int i = tid; i < TopK; i += BLOCK_SIZE) {
      const auto pos = s_indices[i];
      dst_page_entry[i] = (pos >= 0 && pos < length) ? src_page_entry[pos] : -1;
    }
  }
}

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
#ifdef USE_ROCM
    return ::cudaFuncSetAttribute(
        reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
    return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
  }();
  TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

// =========================================================================
// Split-K: multi-CTA cooperative radix select for small B (B*2 <= SMs) with
// long rows (length > TopK), ported from the vLLM persistent_topk multi-CTA
// radix. Each row is handled by a group of G CTAs; every CTA stages its chunk
// in shared memory as ordered uint32 keys; 4 rounds of 8-bit radix over
// global histograms + group arrival barriers converge to the exact 32-bit
// pivot; output collection reserves global ranges per CTA. Workspace comes
// from the caller (at::zeros per launch, stream-ordered -> no cross-stream
// UAF) and is zero-initialized.
// =========================================================================
constexpr int kSplitKRadix = 256;
//   C600U (>=128KB): keep the historical 16384 (64KB dyn, ample headroom).
//   C500  (<=64KB) : cap dynamic smem at 64KB*0.8 = 52428B -> 13056 elems
//                    (13056*4 = 52224B; + 1044B static = 53268B < 65536B).
constexpr uint32_t kSplitKMaxChunk = 16384;      // C600U ceiling (64KB dyn smem)
constexpr uint32_t kSplitKMaxChunkC500 = 13056;  // C500 ceiling (52224B dyn smem)

struct SplitKRowState {
  uint32_t histogram[3][kSplitKRadix];
  int arrival_counter;
  int output_counter;
  int eq_counter;
  int p_counter;
  int z_counter;
};

__device__ __forceinline__ void sk_red_release(int* ptr, int val) {
  __threadfence();
  atomicAdd(ptr, val);
}

__device__ __forceinline__ void sk_wait_ge(int* ptr, int target_val, int thread_idx) {
  if (thread_idx == 0) {
    volatile int* v = reinterpret_cast<volatile int*>(ptr);
    int spins = 0;
#pragma unroll 1
    while (*v < target_val) {
      // Watchdog atomic read: bounds staleness in case volatile loads are
      // served from a private cache on this architecture.
      if ((++spins & 1023) == 0 && atomicAdd(ptr, 0) >= target_val) break;
    }
    __threadfence();
  }
  __syncthreads();
}

__global__ __launch_bounds__(kThreadsPerBlock)
    void topk_transform_splitk_kernel(
        const float* __restrict__ input,
        const int32_t* __restrict__ lengths,
        int64_t input_stride,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        int64_t src_stride,
        uint32_t B,
        uint32_t G,            // CTAs per row group
        uint32_t chunk_size,   // host-computed elements per CTA
        SplitKRowState* __restrict__ state,
        int32_t* __restrict__ row_output) {
  const uint32_t group = blockIdx.x / G;
  const uint32_t cta_in_group = blockIdx.x % G;
  const uint32_t tx = threadIdx.x;
  if (group >= B) return;

  const int64_t len64 = lengths[group];
  const uint32_t length = len64 > 0 ? static_cast<uint32_t>(len64) : 0;

  const int32_t* src_page_entry = src_page_table + static_cast<int64_t>(group) * src_stride;
  int32_t* dst_page_entry = dst_page_table + static_cast<int64_t>(group) * TopK;

  if (length <= TopK) {
    // Short row: the group leader alone performs the naive transform; the
    // other CTAs of the group exit without touching any barrier state.
    if (cta_in_group == 0) {
      decode_naive_topk_transform(input + static_cast<int64_t>(group) * input_stride, length,
                                  dst_page_entry, src_page_entry);
    }
    return;
  }

  SplitKRowState* st = state + group;
  int32_t* out = row_output + static_cast<int64_t>(group) * TopK;
  const float* row_input = input + static_cast<int64_t>(group) * input_stride;
  const uint32_t my_chunk_start = cta_in_group * chunk_size;
  const uint32_t my_chunk_end =
      (my_chunk_start + chunk_size < length) ? (my_chunk_start + chunk_size) : length;
  const uint32_t actual_chunk_size =
      (my_chunk_start < length) ? (my_chunk_end - my_chunk_start) : 0;

  extern __shared__ uint32_t shared_ordered[];
  __shared__ uint32_t s_scan_hist[kSplitKRadix];  // per-round local histogram
  __shared__ uint32_t s_scratch[3];               // stage-3 cursor/base/eq-count
  __shared__ uint32_t s_scalars[2];               // 0: prefix, 1: remaining_k

  // -- Stage 1: load the chunk as ordered uint32 keys (vectorized float4) --
  // Also counts exact positives (key > +0.0's key 0x80000000, i.e. every
  // value strictly above +0.0 incl. +Inf/+NaN) and exact +0.0s for the zero
  // fast-path probe.
  uint32_t my_p = 0, my_z = 0;
  const auto load_key = [&](float v) {
    const uint32_t k = convert_to_uint32(v);
    if (k > 0x80000000u) ++my_p;
    else if (k == 0x80000000u) ++my_z;
    return k;
  };
  if ((my_chunk_start & 3u) == 0u) {
    const uint32_t n4 = actual_chunk_size >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(row_input + my_chunk_start);
    for (uint32_t v = tx; v < n4; v += kThreadsPerBlock) {
      const float4 f = in4[v];
      shared_ordered[v * 4 + 0] = load_key(f.x);
      shared_ordered[v * 4 + 1] = load_key(f.y);
      shared_ordered[v * 4 + 2] = load_key(f.z);
      shared_ordered[v * 4 + 3] = load_key(f.w);
    }
    for (uint32_t i = (n4 << 2) + tx; i < actual_chunk_size; i += kThreadsPerBlock) {
      shared_ordered[i] = load_key(row_input[my_chunk_start + i]);
    }
  } else {
    for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
      shared_ordered[i] = load_key(row_input[my_chunk_start + i]);
    }
  }
  __syncthreads();

  if (tx == 0) {
    s_scalars[0] = 0;     // prefix
    s_scalars[1] = TopK;  // remaining_k
  }
  __syncthreads();

  // The workspace is zeroed by a prior op on the same stream and kernel
  // launch boundaries fence memory, so all counters start at 0 with no
  // initial barrier needed.
  int arrival_target = 0;

  // -- Probe: exact global positive/zero counts decide the zero fast path --
  {
    uint32_t wz = my_z, wp = my_p;
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
      wz += __shfl_down_sync(0xffffffffffffffffULL, wz, off);
      wp += __shfl_down_sync(0xffffffffffffffffULL, wp, off);
    }
    if ((tx & 63) == 0) {
      if (wz > 0) sk_red_release(&st->z_counter, static_cast<int>(wz));
      if (wp > 0) sk_red_release(&st->p_counter, static_cast<int>(wp));
    }
  }
  __syncthreads();  // p/z releases are fenced before the arrival release below
  if (tx == 0) {
    sk_red_release(&st->arrival_counter, 1);
  }
  arrival_target += static_cast<int>(G);
  sk_wait_ge(&st->arrival_counter, arrival_target, static_cast<int>(tx));
  if (tx == 0) {
    s_scratch[0] = static_cast<uint32_t>(atomicAdd(&st->p_counter, 0));
    s_scratch[1] = static_cast<uint32_t>(atomicAdd(&st->z_counter, 0));
  }
  __syncthreads();
  const uint32_t P_g = s_scratch[0];
  const uint32_t Z_g = s_scratch[1];
  // Zero fast path (provably exact): P < TopK <= P + Z iff the K-th largest
  // key is exactly +0.0's key: output = all positives + any TopK - P zeros.
  // Skips all 4 radix rounds (collapsed real data keeps ~every staged key
  // alive through every round with all-same-byte histograms).
  const bool zero_fast =
      P_g < static_cast<uint32_t>(TopK) && P_g + Z_g >= static_cast<uint32_t>(TopK);

  // -- Stage 2: 4 rounds of 8-bit radix (exact 32-bit resolution) --
  uint32_t ordered_pivot;
  uint32_t total_gt;
  if (zero_fast) {
    ordered_pivot = 0x80000000u;
    total_gt = P_g;
  } else {
  for (uint32_t round = 0; round < 4; ++round) {
    const uint32_t shift = 24 - round * 8;
    const uint32_t prefix = s_scalars[0];
    const uint32_t remaining_k = s_scalars[1];
    uint32_t* current_hist = st->histogram[round % 3];
    uint32_t* next_hist = st->histogram[(round + 1) % 3];

    for (uint32_t i = tx; i < kSplitKRadix; i += kThreadsPerBlock) {
      s_scan_hist[i] = 0;
    }
    __syncthreads();

    const uint32_t mask = (round == 0) ? 0u : (~0u << (32 - round * 8));
    for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
      const uint32_t ordered = shared_ordered[i];
      if ((ordered & mask) == prefix) {
        atomicAdd(&s_scan_hist[(ordered >> shift) & 0xFF], 1);
      }
    }
    __syncthreads();

    for (uint32_t i = tx; i < kSplitKRadix; i += kThreadsPerBlock) {
      if (s_scan_hist[i] > 0) {
        atomicAdd(&current_hist[i], s_scan_hist[i]);
      }
    }
    if (cta_in_group == 0) {
      for (uint32_t i = tx; i < kSplitKRadix; i += kThreadsPerBlock) {
        next_hist[i] = 0;
      }
    }
    if (tx == 0) {
      sk_red_release(&st->arrival_counter, 1);
    }
    arrival_target += static_cast<int>(G);
    sk_wait_ge(&st->arrival_counter, arrival_target, static_cast<int>(tx));

    // Warp 0 scans the 256-bin global histogram in place (4 bins per lane,
    // 64-lane MACA warp, shuffle scan) and commits the new prefix /
    // remaining_k. Exactly one lane matches the threshold bin.
    if (tx < 64) {
      const uint32_t base_b = tx * 4;
      uint32_t local[4];
      uint32_t lsum = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        local[j] = current_hist[base_b + j];
        lsum += local[j];
      }
      uint32_t incl = lsum;
#pragma unroll
      for (int off = 1; off < 64; off <<= 1) {
        const uint32_t up = __shfl_up_sync(0xffffffffffffffffULL, incl, off);
        if (tx >= off) incl += up;
      }
      const uint32_t total = __shfl_sync(0xffffffffffffffffULL, incl, 63);
      uint32_t cge = total - (incl - lsum);  // count >= first bin of this lane
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t cgt = cge - local[j];
        if (cge >= remaining_k && cgt < remaining_k) {
          s_scalars[0] = prefix | ((base_b + j) << shift);
          s_scalars[1] = remaining_k - cgt;
        }
        cge = cgt;
      }
    }
    __syncthreads();
  }
  ordered_pivot = s_scalars[0];
  // Global count of elements strictly greater than the pivot is TopK minus
  // the final k_in_bin, so the >pivot region is exactly out[0, total_gt) and
  // ==pivot fills [total_gt, TopK) — no barrier needed between the two.
  total_gt = TopK - s_scalars[1];
  }

  // -- Stage 3: collect top-k indices around the exact pivot --
  if (tx == 0) s_scratch[0] = 0;
  __syncthreads();

  uint32_t my_gt_count = 0;
  uint32_t my_eq_count = 0;
  for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
    if (shared_ordered[i] > ordered_pivot) my_gt_count++;
    else if (shared_ordered[i] == ordered_pivot) ++my_eq_count;
  }
  // MACA warp is 64 lanes: reduce within warp, commit per warp leader
  for (int offset = 32; offset > 0; offset /= 2) {
    my_gt_count += __shfl_down_sync(0xffffffffffffffffULL, my_gt_count, offset);
  }
  if (tx % 64 == 0 && my_gt_count > 0) {
    atomicAdd(&s_scratch[0], my_gt_count);
  }
  __syncthreads();
  const uint32_t local_gt_count = s_scratch[0];

  if (tx == 0) {
    s_scratch[1] = 0;
    if (local_gt_count > 0) {
      s_scratch[2] = static_cast<uint32_t>(atomicAdd(&st->output_counter, static_cast<int>(local_gt_count)));
    }
  }
  __syncthreads();
  for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
    if (shared_ordered[i] > ordered_pivot) {
      const uint32_t local_pos = atomicAdd(&s_scratch[1], 1);
      const int pos = static_cast<int>(s_scratch[2]) + static_cast<int>(local_pos);
      if (pos >= 0 && pos < TopK) out[pos] = static_cast<int32_t>(my_chunk_start + i);
    }
  }

  // fill the remaining slots from == pivot elements (bounded, tie-tolerant).
  // Warp-ticketed slot reservation: one global atomic per warp instead of one
  // per candidate (collapsed rows push ~every element through this path and
  // the per-element atomicAdd on the single eq_counter serializes).
  const uint32_t eq_base = warp64_ticket(&st->eq_counter, my_eq_count);
  uint32_t run_eq = 0;
  for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
    if (shared_ordered[i] == ordered_pivot) {
      const uint32_t pos = total_gt + eq_base + run_eq++;
      if (pos < TopK) out[pos] = static_cast<int32_t>(my_chunk_start + i);
    }
  }

  // final barrier: all output slots written before the page-table gather
  __syncthreads();
  if (tx == 0) {
    sk_red_release(&st->arrival_counter, 1);
  }
  arrival_target += static_cast<int>(G);
  sk_wait_ge(&st->arrival_counter, arrival_target, static_cast<int>(tx));

  for (int i = static_cast<int>(tx); i < TopK; i += kThreadsPerBlock) {
    const int idx = out[i];
    dst_page_entry[i] = (idx >= 0 && idx < static_cast<int>(length)) ? src_page_entry[idx] : -1;
  }
}

}  // namespace

// Exported launcher. Cross-TU surface is raw pointers + scalars only (no shared
// struct / ODR coupling with topk.cu). topk.cu forward-declares this and forwards
// its is_decode branch here. max_len is the score row width (host-known tensor
// shape, NOT a per-row length) used only to pick the launch block size.
// Block-size policy (occupancy-driven, C600U: 2048 threads + 128KB smem per SM):
//   max_len <= 2048        -> 1024 (naive copy path dominates)
//   2048 < max_len <= 4096 -> 512  (16KB dyn smem, 4 resident CTAs/SM)
//   4096 < max_len < 8192  -> pow2ceil(max_len/8) clamped to [512, 1024]
//   max_len >= 8192        -> 1024
// B <= 16 always takes the 1024-thread tl path (unchanged baseline behavior).
namespace {
inline int pow2_ceil_host(uint32_t x) {
  uint32_t b = 1;
  while (b < x) b <<= 1;
  return static_cast<int>(b);
}

inline int pick_decode_block_size(int64_t max_len, uint32_t B) {
  if (max_len <= 2048) return 1024;  // naive copy path: unchanged (measured best)
  if (max_len >= 8192) return 1024;  // long rows: per-row latency hiding wins
  const int b = max_len <= 4096 ? 512 : pow2_ceil_host(static_cast<uint32_t>(max_len) >> 3);
  const int blk = b < 512 ? 512 : (b > 1024 ? 1024 : b);
  // Smaller blocks only pay off when they still fill the machine (>= 1 full
  // wave). Below that, halving threads per row just stretches each row.
  // Measured on C600U (32 SM x 2048 threads): B=64 loses 0.73x, B=128 gains 1.41x.
  static const int kSMs = [] {
    int dev = 0, sms = 32;
    if (::cudaGetDevice(&dev) == cudaSuccess)
      ::cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return sms > 0 ? sms : 32;
  }();
  if (static_cast<uint64_t>(B) * static_cast<uint64_t>(blk) <
      static_cast<uint64_t>(kSMs) * 2048ull) {
    return 1024;
  }
  return blk;
}
}  // namespace

void fast_topk_transform_decode_launch(
    const float* input,
    const int32_t* lengths,
    int64_t input_stride,
    int32_t* dst_page_table,
    const int32_t* src_page_table,
    int64_t src_stride,
    uint32_t B,
    cudaStream_t stream) {
  // input_stride == score row width == max seq_len bound for this launch
  const int block_size = pick_decode_block_size(input_stride, B);
  const auto block = dim3{static_cast<uint32_t>(block_size)};
  const auto grid = dim3{B};
  if (block_size == 512) {
    setup_kernel_smem_once<topk_transform_decode_kernel_impl<512>, kSmem>();
    topk_transform_decode_kernel_impl<512><<<grid, block, kSmem, stream>>>(
        input, lengths, input_stride, dst_page_table, src_page_table, src_stride, B);
  } else {
    setup_kernel_smem_once<topk_transform_decode_kernel_impl<1024>, kSmem>();
    topk_transform_decode_kernel_impl<1024><<<grid, block, kSmem, stream>>>(
        input, lengths, input_stride, dst_page_table, src_page_table, src_stride, B);
  }
}

// ---- Split-K exported surface (called from topk.cu's is_decode branch) ----
// Workspace layout: [B * sizeof(SplitKRowState) (256-aligned)] then [B * TopK int32].
size_t fast_topk_splitk_workspace_bytes(uint32_t B) {
  const size_t state_bytes = ((sizeof(SplitKRowState) + 255) / 256) * 256;
  return state_bytes * B + static_cast<size_t>(B) * TopK * sizeof(int32_t);
}

void fast_topk_transform_decode_splitk_launch(
    const float* input,
    const int32_t* lengths,
    int64_t input_stride,
    int32_t* dst_page_table,
    const int32_t* src_page_table,
    int64_t src_stride,
    uint32_t B,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
  TORCH_CHECK(workspace != nullptr && workspace_bytes >= fast_topk_splitk_workspace_bytes(B),
              "split-K workspace too small");
  static const int kSMs = [] {
    int dev = 0, sms = 32;
    if (::cudaGetDevice(&dev) == cudaSuccess)
      ::cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return sms > 0 ? sms : 32;
  }();


  //   >=128KB (C600U): keep historical 16384 (64KB dyn, ample headroom).
  //   <=64KB  (C500) : cap dynamic smem at 64KB*0.8 = 52428B -> 13056 elems
  //                    (13056*4 = 52224B; + 1044B static = 53268B < 65536B).
  //   otherwise      : general 80%-of-optin budget, aligned down to 128 elems.
  static const auto kSmemBudget = [] {
    int dev = 0, optin = 64 * 1024;
    if (::cudaGetDevice(&dev) == cudaSuccess)
      ::cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (optin <= 0) optin = 64 * 1024;
    uint32_t chunk_cap;
    if (optin >= 128 * 1024) {
      chunk_cap = kSplitKMaxChunk;            // C600U: 16384
    } else if (optin <= 64 * 1024) {
      chunk_cap = kSplitKMaxChunkC500;        // C500: 13056
    } else {
      // General device: 80% of opt-in as dynamic smem, in uint32 elems, /128.
      uint32_t budget = static_cast<uint32_t>((static_cast<uint64_t>(optin) * 8) / 10);
      chunk_cap = (budget / sizeof(uint32_t)) & ~static_cast<uint32_t>(127);
      if (chunk_cap == 0) chunk_cap = kSplitKMaxChunkC500;
    }
    struct Budget { uint32_t chunk_cap; size_t dyn_optin; };
    return Budget{chunk_cap, static_cast<size_t>(chunk_cap) * sizeof(uint32_t)};
  }();
  const uint32_t chunk_cap = kSmemBudget.chunk_cap;
  const size_t dyn_smem_optin = kSmemBudget.dyn_optin;

  // Opt-in dynamic smem limit is a RUNTIME value (device-dependent), so we call
  // cudaFuncSetAttribute directly instead of the compile-time
  // setup_kernel_smem_once<> template. Set once per process.
  static const cudaError_t smem_attr = [] {
#ifdef USE_ROCM
    return ::cudaFuncSetAttribute(
        reinterpret_cast<const void*>(topk_transform_splitk_kernel),
        ::cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSmemBudget.dyn_optin));
#else
    return ::cudaFuncSetAttribute(
        topk_transform_splitk_kernel,
        ::cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSmemBudget.dyn_optin));
#endif
  }();
  TORCH_CHECK(smem_attr == cudaSuccess,
              "split-K smem opt-in failed: ", ::cudaGetErrorString(smem_attr));
  (void)dyn_smem_optin;

  // chunk cap one 1024-thread block already needs ~53KB, so only 1 block fits per
  // SM (2 on C600U's 128KB). Query the real per-SM block count at THIS smem so the
  // ceiling adapts to the device instead of assuming 1 block/SM.
  static const int kMaxResidentBlocks = [] {
    int blocks_per_sm = 1;
    // a smaller launch only ever fits MORE blocks, so this is a safe lower bound.
    const int probe_smem = static_cast<int>(kSmemBudget.dyn_optin);
    if (::cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, topk_transform_splitk_kernel,
            static_cast<int>(kThreadsPerBlock), probe_smem) != cudaSuccess ||
        blocks_per_sm < 1) {
      blocks_per_sm = 1;
    }
    return blocks_per_sm;
  }() * kSMs;

  // G: CTAs per row. At least 8 (measured best for B=4 and B=8 on C600U:
  // shorter chunks cut per-round latency more than a second wave costs),
  // capped by (a) the smem chunk budget and (b) the co-residency ceiling
  // B*G <= kMaxResidentBlocks. Cap (b) is what prevents the deadlock the C500
  // chunk-cap reduction would otherwise introduce for long rows at small B.
  const uint64_t g_resident_cap = static_cast<uint64_t>(kMaxResidentBlocks) / B;
  uint64_t g = kSMs / B;
  if (g < 8) g = 8;
  const uint64_t need = (static_cast<uint64_t>(input_stride) + g - 1) / g;
  if (need > chunk_cap) {
    g = (static_cast<uint64_t>(input_stride) + chunk_cap - 1) / chunk_cap;
  }
 

  if (g > g_resident_cap) g = g_resident_cap;
  if (g < 1) g = 1;
  const uint32_t G = static_cast<uint32_t>(g);
  const uint32_t chunk = static_cast<uint32_t>(
      (static_cast<uint64_t>(input_stride) + G - 1) / G);
  if (chunk > chunk_cap) {
    fast_topk_transform_decode_launch(input, lengths, input_stride, dst_page_table,
                                      src_page_table, src_stride, B, stream);
    return;
  }

  const size_t state_bytes = ((sizeof(SplitKRowState) + 255) / 256) * 256;
  auto* state = reinterpret_cast<SplitKRowState*>(workspace);
  auto* row_output = reinterpret_cast<int32_t*>(
      reinterpret_cast<uint8_t*>(workspace) + state_bytes * B);

  const auto grid = dim3{static_cast<uint32_t>(B) * G};
  const auto block = dim3{kThreadsPerBlock};
  const size_t smem = static_cast<size_t>(chunk) * sizeof(uint32_t);
  topk_transform_splitk_kernel<<<grid, block, smem, stream>>>(
      input, lengths, input_stride, dst_page_table, src_page_table, src_stride, B, G, chunk,
      state, row_output);
}
