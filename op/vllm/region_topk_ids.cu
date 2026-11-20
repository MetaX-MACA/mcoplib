#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>

namespace {
constexpr int kShortRowThreads = 256;
constexpr int kFillAllThreads = 256;
constexpr int kLargeRowsThreads = 256;
constexpr int kMidRowsThreads = 512;
constexpr int kFewRowsThreads = 1024;
constexpr int kDecodeTopK = 512;
constexpr int kRadixKeyCacheMaxRegions = 3072;
constexpr int kCoarseFilterMinRegions = 4096;
constexpr int kCandidateCacheMinRegions = 18000;
constexpr int kCoarseBuckets = 1024;
constexpr int kCoarseBucketBits = 10;
// Keep the coarse cutoff bucket on chip for the remaining radix bytes. 2048
// entries overflowed on normal 93K-98K rows and forced the full-row fallback;
// the CuTeDSL selector's 4096-entry budget removes that cliff on C600U. Keep
// a smaller budget for 256-thread blocks to reduce their shared-memory cost,
// while retaining enough headroom for the cutoff bucket of wide normal rows.
constexpr int kSmallBlockCoarseCandidateCapacity = 3072;
constexpr int kLargeBlockCoarseCandidateCapacity = 4096;
constexpr int kDecodeFastPathMinRegions = 2048;
constexpr int kOrderedCompactionMaxRegions = 12288;
constexpr int kRadixBuckets = 256;
constexpr int kRadixPasses = 4;

enum class KernelMode {
    kGeneral,
    kCachedRadixKeys,
    kCoarseFilter,
    kCoarseFilterAndCachedCandidates,
};

inline KernelMode coarse_filter_mode(bool cache_candidates) {
    return cache_candidates
               ? KernelMode::kCoarseFilterAndCachedCandidates
               : KernelMode::kCoarseFilter;
}

#ifdef USE_MACA
constexpr int kWaveSize = 64;
constexpr uint64_t kFullWaveMask = 0xffffffffffffffffULL;
#else
constexpr int kWaveSize = 32;
constexpr uint32_t kFullWaveMask = 0xffffffffu;
#endif
static_assert(kShortRowThreads % kWaveSize == 0 &&
                  kMidRowsThreads % kWaveSize == 0 &&
                  kFewRowsThreads % kWaveSize == 0 &&
                  kLargeRowsThreads % kWaveSize == 0,
              "all block sizes must contain complete waves");

// Inclusive prefix for a boolean flag. A 64-bit ballot handles a complete C600U
// wave without shared memory. Shared memory contains one 4-byte count per wave;
// at most 16 distinct words are touched, so they map to distinct 32-bit banks.
template <int BLOCK_THREADS>
__device__ __forceinline__ int block_flag_prefix(bool flag, int* wave_counts) {
    constexpr int kWaves = BLOCK_THREADS / kWaveSize;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWaveSize - 1);
    const int wave = tid / kWaveSize;

    const uint64_t ballot = static_cast<uint64_t>(__ballot_sync(kFullWaveMask, flag));
    if (lane == 0) wave_counts[wave] = __popcll(ballot);
    __syncthreads();

    const uint64_t lower_mask = lane == 0 ? 0 : ((uint64_t{1} << lane) - 1);
    int prefix = __popcll(ballot & lower_mask) + static_cast<int>(flag);
#pragma unroll
    for (int preceding_wave = 0; preceding_wave < kWaves; ++preceding_wave) {
        if (preceding_wave < wave) prefix += wave_counts[preceding_wave];
    }
    return prefix;
}

// Exclusive prefix for an arbitrary non-negative count.  Each wave scans in
// registers; only one total per wave crosses shared memory.  On C600U this
// replaces a 1024-element shared-memory Blelloch scan and its ~20 barriers
// with two block barriers.
template <int BLOCK_THREADS>
__device__ __forceinline__ int block_count_exclusive_prefix(
        int count, int* wave_offsets, int* block_total) {
    constexpr int kWaves = BLOCK_THREADS / kWaveSize;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWaveSize - 1);
    const int wave = tid / kWaveSize;

    int inclusive = count;
#pragma unroll
    for (int offset = 1; offset < kWaveSize; offset <<= 1) {
        const int preceding = __shfl_up_sync(kFullWaveMask, inclusive, offset);
        if (lane >= offset) inclusive += preceding;
    }
    if (lane == kWaveSize - 1) wave_offsets[wave] = inclusive;
    __syncthreads();

    if (wave == 0) {
        const int wave_count = lane < kWaves ? wave_offsets[lane] : 0;
        int wave_inclusive = wave_count;
#pragma unroll
        for (int offset = 1; offset < kWaveSize; offset <<= 1) {
            const int preceding =
                __shfl_up_sync(kFullWaveMask, wave_inclusive, offset);
            if (lane >= offset) wave_inclusive += preceding;
        }
        if (lane < kWaves) wave_offsets[lane] = wave_inclusive - wave_count;
        if (lane == kWaves - 1) *block_total = wave_inclusive;
    }
    __syncthreads();
    return wave_offsets[wave] + inclusive - count;
}

// This is deliberately selected by the numeric comparison rather than the sign
// bit.  Consequently -0.0f and +0.0f have the same key, exactly as in Triton.
__device__ __forceinline__ uint32_t ordered_float_key(float score) {
    const uint32_t bits = __float_as_uint(score);
    return score < 0.0f ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ int aligned_float4_head(const float* values,
                                                    int count) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(values);
    const int alignment_values =
        static_cast<int>(((16u - (address & 15u)) & 15u) / sizeof(float));
    return count < alignment_values ? count : alignment_values;
}

template <int BLOCK_THREADS>
__global__ void region_topk_ids_fill_all_kernel(
        const int32_t* __restrict__ lengths,
        int32_t* __restrict__ out,
        int32_t seq_regions,
        int64_t stride_out_row,
        int32_t topk) {
    const int tid = static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(blockIdx.x);
    int visible = lengths[row];
    visible = visible < 0 ? 0 : (visible > seq_regions ? seq_regions : visible);
    int32_t* row_out = out + static_cast<int64_t>(row) * stride_out_row;
    for (int slot = tid; slot < topk; slot += BLOCK_THREADS) {
        row_out[slot] = slot < visible ? slot : -1;
    }
}

template <int BLOCK_THREADS, bool CACHE_CANDIDATES, bool CACHE_RADIX_KEYS,
          bool COARSE_FILTER>
__global__ void region_topk_ids_kernel(const float* __restrict__ logits,
                                        const int32_t* __restrict__ lengths,
                                        int32_t* __restrict__ out,
                                        int32_t seq_regions,
                                        int64_t stride_logits_row,
                                        int64_t stride_out_row,
                                        int32_t topk) {
    const int tid = static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(blockIdx.x);
    const float* row_logits =
        logits + static_cast<int64_t>(row) * stride_logits_row;
    int32_t* row_out = out + static_cast<int64_t>(row) * stride_out_row;

    constexpr int kWaves = BLOCK_THREADS / kWaveSize;
    constexpr bool kHasCandidateFastPath =
        BLOCK_THREADS > kShortRowThreads;
    constexpr int kCandidateIdCapacity =
        kHasCandidateFastPath ? kDecodeTopK : kWaves;
    constexpr int kCoarseCandidateCapacity =
        BLOCK_THREADS >= kMidRowsThreads
            ? kLargeBlockCoarseCandidateCapacity
            : kSmallBlockCoarseCandidateCapacity;
    static_assert(!CACHE_CANDIDATES || COARSE_FILTER,
                  "candidate caching requires coarse filtering");
    static_assert(!(CACHE_RADIX_KEYS && COARSE_FILTER),
                  "radix-key caching and coarse filtering are exclusive");
    static_assert(!CACHE_CANDIDATES || kHasCandidateFastPath,
                  "candidate caching requires the candidate fast path");

    __shared__ uint32_t histogram[kRadixBuckets];
    __shared__ int selected_wave_counts[kWaves];
    __shared__ int tied_wave_counts[kWaves];
    __shared__ int candidate_ids[kCandidateIdCapacity];
    __shared__ int candidate_total;
    __shared__ int candidate_overflow;
    __shared__ uint32_t threshold;
    __shared__ int remaining;
    __shared__ int threshold_ties;
    __shared__ int emit_cursor;
    __shared__ int ties_seen;
    __shared__ uint32_t coarse_histogram[COARSE_FILTER ? kCoarseBuckets : 1];
    __shared__ uint32_t
        coarse_candidate_keys[COARSE_FILTER ? kCoarseCandidateCapacity : 1];
    __shared__ int coarse_candidate_count;
    __shared__ uint32_t coarse_threshold_bucket;

    // Clamp the row's valid region count to the allocated width.
    int visible = lengths[row];
    visible = visible < 0 ? 0 : (visible > seq_regions ? seq_regions : visible);

    // Emit all visible ids and trailing padding in one pass.
    if (topk >= visible) {
        for (int slot = tid; slot < topk; slot += BLOCK_THREADS) {
            row_out[slot] = slot < visible ? slot : -1;
        }
        return;
    }

    // When only one visible id must be discarded, four radix passes are much
    // more expensive than a direct block reduction. Stable descending score
    // order keeps the smaller id on a cutoff tie, so the discarded minimum
    // is the largest id among equal minimum keys.
    if (visible == topk + 1) {
        const int lane = tid & (kWaveSize - 1);
        const int wave = tid / kWaveSize;
        uint32_t bottom_key = UINT32_MAX;
        int bottom_id = -1;
        for (int region = tid; region < visible; region += BLOCK_THREADS) {
            const uint32_t key = ordered_float_key(row_logits[region]);
            if (key < bottom_key || (key == bottom_key && region > bottom_id)) {
                bottom_key = key;
                bottom_id = region;
            }
        }
#pragma unroll
        for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
            const uint32_t other_key =
                __shfl_down_sync(kFullWaveMask, bottom_key, offset);
            const int other_id =
                __shfl_down_sync(kFullWaveMask, bottom_id, offset);
            if (lane + offset < kWaveSize &&
                (other_key < bottom_key ||
                 (other_key == bottom_key && other_id > bottom_id))) {
                bottom_key = other_key;
                bottom_id = other_id;
            }
        }
        if (lane == 0) {
            histogram[wave] = bottom_key;
            candidate_ids[wave] = bottom_id;
        }
        __syncthreads();

        if (wave == 0) {
            bottom_key = lane < kWaves ? histogram[lane] : UINT32_MAX;
            bottom_id = lane < kWaves ? candidate_ids[lane] : -1;
#pragma unroll
            for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
                const uint32_t other_key =
                    __shfl_down_sync(kFullWaveMask, bottom_key, offset);
                const int other_id =
                    __shfl_down_sync(kFullWaveMask, bottom_id, offset);
                if (lane + offset < kWaveSize &&
                    (other_key < bottom_key ||
                     (other_key == bottom_key && other_id > bottom_id))) {
                    bottom_key = other_key;
                    bottom_id = other_id;
                }
            }
            if (lane == 0) emit_cursor = bottom_id;
        }
        __syncthreads();

        const int excluded_id = emit_cursor;
        for (int region = tid; region < visible; region += BLOCK_THREADS) {
            if (region != excluded_id) {
                row_out[region - static_cast<int>(region > excluded_id)] =
                    region;
            }
        }
        return;
    }

    if (tid == 0) {
        threshold = 0;
        remaining = topk;
        if constexpr (kHasCandidateFastPath) threshold_ties = 0;
    }
    __syncthreads();

    // Wide rows first use the high 10 bits of the exact ordered-float key to
    // isolate the bucket containing the top-k boundary. Only that bucket is
    // refined by the exact radix passes below. If it does not initially fit
    // on chip, retain its prefix and rank, refine from HBM until a later radix
    // bucket fits, then cache that smaller bucket for the remaining passes.
    bool use_coarse_candidates = false;
    int radix_visible = visible;
    if constexpr (COARSE_FILTER) {
        static_assert(kCoarseBuckets % BLOCK_THREADS == 0,
                      "coarse histogram must split evenly across threads");
        constexpr int kBinsPerThread = kCoarseBuckets / BLOCK_THREADS;
        for (int bucket = tid; bucket < kCoarseBuckets;
             bucket += BLOCK_THREADS) {
            coarse_histogram[bucket] = 0;
        }
        __syncthreads();

        const int scalar_head = aligned_float4_head(row_logits, visible);
        for (int region = tid; region < scalar_head; region += BLOCK_THREADS) {
            const uint32_t key = ordered_float_key(row_logits[region]);
            atomicAdd(&coarse_histogram[key >> (32 - kCoarseBucketBits)], 1u);
        }
        const int vector_count = (visible - scalar_head) / 4;
        const float4* vector_logits =
            reinterpret_cast<const float4*>(row_logits + scalar_head);
        for (int vector_id = tid; vector_id < vector_count;
             vector_id += BLOCK_THREADS) {
            const float4 scores = vector_logits[vector_id];
            const uint32_t key0 = ordered_float_key(scores.x);
            const uint32_t key1 = ordered_float_key(scores.y);
            const uint32_t key2 = ordered_float_key(scores.z);
            const uint32_t key3 = ordered_float_key(scores.w);
            atomicAdd(&coarse_histogram[key0 >> (32 - kCoarseBucketBits)], 1u);
            atomicAdd(&coarse_histogram[key1 >> (32 - kCoarseBucketBits)], 1u);
            atomicAdd(&coarse_histogram[key2 >> (32 - kCoarseBucketBits)], 1u);
            atomicAdd(&coarse_histogram[key3 >> (32 - kCoarseBucketBits)], 1u);
        }
        const int scalar_tail = scalar_head + vector_count * 4;
        for (int region = scalar_tail + tid; region < visible;
             region += BLOCK_THREADS) {
            const uint32_t key = ordered_float_key(row_logits[region]);
            atomicAdd(&coarse_histogram[key >> (32 - kCoarseBucketBits)], 1u);
        }
        __syncthreads();

        int local_count = 0;
#pragma unroll
        for (int item = 0; item < kBinsPerThread; ++item) {
            const int bucket =
                kCoarseBuckets - 1 - (tid * kBinsPerThread + item);
            local_count += static_cast<int>(coarse_histogram[bucket]);
        }
        const int target_rank = topk;
        const int count_above_chunk =
            block_count_exclusive_prefix<BLOCK_THREADS>(
                local_count, selected_wave_counts, &candidate_total);
        if (count_above_chunk < target_rank &&
            count_above_chunk + local_count >= target_rank) {
            int count_above_bucket = count_above_chunk;
#pragma unroll
            for (int item = 0; item < kBinsPerThread; ++item) {
                const int bucket =
                    kCoarseBuckets - 1 - (tid * kBinsPerThread + item);
                const int bucket_count =
                    static_cast<int>(coarse_histogram[bucket]);
                if (count_above_bucket < target_rank &&
                    count_above_bucket + bucket_count >= target_rank) {
                    coarse_threshold_bucket = static_cast<uint32_t>(bucket);
                    remaining = target_rank - count_above_bucket;
                    coarse_candidate_count = bucket_count;
                }
                count_above_bucket += bucket_count;
            }
        }
        __syncthreads();

        const bool coarse_bucket_fits = coarse_candidate_count <= kCoarseCandidateCapacity;
        __syncthreads();
        if (tid == 0) coarse_candidate_count = 0;
        __syncthreads();
        if (coarse_bucket_fits) {
            for (int region = tid; region < scalar_head;
                 region += BLOCK_THREADS) {
                const uint32_t key = ordered_float_key(row_logits[region]);
                if ((key >> (32 - kCoarseBucketBits)) ==
                    coarse_threshold_bucket) {
                    const int position = atomicAdd(&coarse_candidate_count, 1);
                    coarse_candidate_keys[position] = key;
                }
            }
            for (int vector_id = tid; vector_id < vector_count;
                 vector_id += BLOCK_THREADS) {
                const float4 scores = vector_logits[vector_id];
                const uint32_t keys[4] = {
                    ordered_float_key(scores.x), ordered_float_key(scores.y),
                    ordered_float_key(scores.z), ordered_float_key(scores.w)};
#pragma unroll
                for (int item = 0; item < 4; ++item) {
                    const uint32_t key = keys[item];
                    if ((key >> (32 - kCoarseBucketBits)) ==
                        coarse_threshold_bucket) {
                        const int position =
                            atomicAdd(&coarse_candidate_count, 1);
                        coarse_candidate_keys[position] = key;
                    }
                }
            }
            for (int region = scalar_tail + tid; region < visible;
                 region += BLOCK_THREADS) {
                const uint32_t key = ordered_float_key(row_logits[region]);
                if ((key >> (32 - kCoarseBucketBits)) ==
                    coarse_threshold_bucket) {
                    const int position = atomicAdd(&coarse_candidate_count, 1);
                    coarse_candidate_keys[position] = key;
                }
            }
        }
        __syncthreads();
        if (coarse_bucket_fits) {
            use_coarse_candidates = true;
            radix_visible = coarse_candidate_count;
            if (tid == 0) {
                threshold = coarse_threshold_bucket <<
                            (32 - kCoarseBucketBits);
            }
        } else {
            if (tid == 0) {
                threshold = coarse_threshold_bucket <<
                            (32 - kCoarseBucketBits);
            }
        }
        __syncthreads();
    }

    // The decode shapes around 2K have only two or three values per thread.
    // Keep their transformed radix keys in registers so the four radix passes
    // do not reload the same logits from global memory.
    constexpr int kCachedKeysPerThread =
        CACHE_RADIX_KEYS
            ? (kRadixKeyCacheMaxRegions + BLOCK_THREADS - 1) / BLOCK_THREADS
            : 1;
    uint32_t cached_radix_keys[kCachedKeysPerThread];
    if constexpr (CACHE_RADIX_KEYS) {
#pragma unroll
        for (int item = 0; item < kCachedKeysPerThread; ++item) {
            const int region = tid + item * BLOCK_THREADS;
            cached_radix_keys[item] =
                region < visible ? ordered_float_key(row_logits[region]) : 0;
        }
    }

    // Exact radix selection, equivalent to Triton's _radix_topk_threshold.
#pragma unroll
    for (int pass = 0; pass < kRadixPasses; ++pass) {
        // The coarse bucket already fixes bits 31..22, including the entire
        // first radix byte. Refine only bytes 2..4 in shared memory.
        if constexpr (COARSE_FILTER) {
            if (pass == kRadixPasses - 1) continue;
        }
        for (int bucket = tid; bucket < kRadixBuckets; bucket += BLOCK_THREADS) {
            histogram[bucket] = 0;
        }
        __syncthreads();

        // Build the histogram for this byte.
        const int effective_pass = COARSE_FILTER ? pass + 1 : pass;
        const int shift = 24 - effective_pass * 8;
        const int high = shift + 8;
        if constexpr (COARSE_FILTER) {
            if (use_coarse_candidates) {
                for (int item = tid; item < radix_visible;
                     item += BLOCK_THREADS) {
                    const uint32_t key = coarse_candidate_keys[item];
                    const bool matches =
                        pass == 0 || ((key >> high) == (threshold >> high));
                    if (matches) {
                        const int bucket =
                            static_cast<int>((key >> shift) & 0xffu);
                        atomicAdd(&histogram[bucket], 1u);
                    }
                }
            } else {
                for (int region = tid; region < visible;
                     region += BLOCK_THREADS) {
                    const uint32_t key = ordered_float_key(row_logits[region]);
                    const bool matches =
                        pass == 0
                            ? ((key >> (32 - kCoarseBucketBits)) ==
                               coarse_threshold_bucket)
                            : ((key >> high) == (threshold >> high));
                    if (matches) {
                        const int bucket =
                            static_cast<int>((key >> shift) & 0xffu);
                        atomicAdd(&histogram[bucket], 1u);
                    }
                }
            }
        } else if constexpr (CACHE_RADIX_KEYS) {
#pragma unroll
            for (int item = 0; item < kCachedKeysPerThread; ++item) {
                const int region = tid + item * BLOCK_THREADS;
                if (region < visible) {
                    const uint32_t key = cached_radix_keys[item];
                    const bool matches =
                        pass == 0 || ((key >> high) == (threshold >> high));
                    if (matches) {
                        const int bucket =
                            static_cast<int>((key >> shift) & 0xffu);
                        atomicAdd(&histogram[bucket], 1u);
                    }
                }
            }
        } else {
            for (int region = tid; region < visible; region += BLOCK_THREADS) {
                const uint32_t key = ordered_float_key(row_logits[region]);
                const bool matches =
                    pass == 0 || ((key >> high) == (threshold >> high));
                if (matches) {
                    const int bucket = static_cast<int>((key >> shift) & 0xffu);
                    atomicAdd(&histogram[bucket], 1u);
                }
            }
        }
        __syncthreads();

        // Locate the cutoff bucket with a block-wide descending prefix.
        // Scan buckets in descending key order. Exactly one thread owns the
        // interval containing `remaining`, removing the serial 256-bin loop.
        const int descending_bucket = kRadixBuckets - 1 - tid;
        const int bucket_count =
            tid < kRadixBuckets
                ? static_cast<int>(histogram[descending_bucket])
                : 0;
        const int target_rank = remaining;
        // Every wave must snapshot the old rank before the winning thread
        // publishes the next pass's rank into shared memory.
        __syncthreads();
        const int count_above = block_count_exclusive_prefix<BLOCK_THREADS>(
            bucket_count, selected_wave_counts, &candidate_total);
        if (tid < kRadixBuckets && count_above < target_rank &&
            count_above + bucket_count >= target_rank) {
            remaining = target_rank - count_above;
            threshold |= static_cast<uint32_t>(descending_bucket) << shift;
            if constexpr (COARSE_FILTER) {
                if (!use_coarse_candidates) {
                    coarse_candidate_count = bucket_count;
                }
            }
            if constexpr (kHasCandidateFastPath) {
                if (effective_pass == kRadixPasses - 1) {
                    threshold_ties = bucket_count;
                }
            }
        }
        __syncthreads();

        // A coarse bucket may be too large for shared memory while its cutoff
        // child after one or two radix bytes is small. Delay materialization
        // until that exact child fits, then perform all later passes on chip.
        if constexpr (COARSE_FILTER) {
            if (!use_coarse_candidates &&
                effective_pass < kRadixPasses - 1 &&
                coarse_candidate_count <= kCoarseCandidateCapacity) {
                if (tid == 0) coarse_candidate_count = 0;
                __syncthreads();
                for (int region = tid; region < visible;
                     region += BLOCK_THREADS) {
                    const uint32_t key = ordered_float_key(row_logits[region]);
                    if ((key >> shift) == (threshold >> shift)) {
                        const int position =
                            atomicAdd(&coarse_candidate_count, 1);
                        coarse_candidate_keys[position] = key;
                    }
                }
                __syncthreads();
                use_coarse_candidates = true;
                radix_visible = coarse_candidate_count;
            }
        }

    }

    // Production decode uses topk=512 and does not require score ordering in
    // the output.  With a unique threshold key, exactly topk values satisfy
    // key >= threshold. Collect those values with one block scan instead of
    // running two block-wide Boolean scans for every input tile, then sort the
    // fixed 512 ids to preserve ascending logical-id output order. If several
    // values tie at the cutoff, candidate_total exceeds topk and the stable
    // fallback below preserves the tie rule. Small rows stay on the original
    // path because its single read is cheaper than this path.
    if (kHasCandidateFastPath && topk == kDecodeTopK &&
        visible >= kDecodeFastPathMinRegions &&
        threshold_ties == remaining) {
        // For medium-width rows, compact one logical tile at a time. Thread
        // order matches region-id order, so this directly produces ascending
        // output and avoids the fixed 512-element sorting network.
        if (visible <= kOrderedCompactionMaxRegions) {
            if (tid == 0) emit_cursor = 0;
            __syncthreads();
            for (int base = 0; base < visible; base += BLOCK_THREADS) {
                const int output_base = emit_cursor;
                const int region = base + tid;
                const bool selected =
                    region < visible &&
                    ordered_float_key(row_logits[region]) >= threshold;
                const int selected_prefix = block_flag_prefix<BLOCK_THREADS>(
                    selected, selected_wave_counts);
                if (selected) {
                    row_out[output_base + selected_prefix - 1] = region;
                }
                if (tid == 0) {
                    int tile_selected = 0;
#pragma unroll
                    for (int wave = 0; wave < kWaves; ++wave) {
                        tile_selected += selected_wave_counts[wave];
                    }
                    emit_cursor = output_base + tile_selected;
                }
                __syncthreads();
            }
            return;
        }

        constexpr int kCandidateCacheCapacity =
            CACHE_CANDIDATES ? 8 : 1;
        int cached_candidate_ids[kCandidateCacheCapacity];
        int local_candidates = 0;
        if constexpr (CACHE_CANDIDATES) {
            bool local_overflow = false;
            const int scalar_head = aligned_float4_head(row_logits, visible);
            for (int region = tid; region < scalar_head;
                 region += BLOCK_THREADS) {
                const uint32_t key = ordered_float_key(row_logits[region]);
                if (key >= threshold) {
                    if (local_candidates < kCandidateCacheCapacity) {
                        cached_candidate_ids[local_candidates] = region;
                    } else {
                        local_overflow = true;
                    }
                    ++local_candidates;
                }
            }
            const int vector_count = (visible - scalar_head) / 4;
            const float4* vector_logits =
                reinterpret_cast<const float4*>(row_logits + scalar_head);
            for (int vector_id = tid; vector_id < vector_count;
                 vector_id += BLOCK_THREADS) {
                const float4 scores = vector_logits[vector_id];
                const uint32_t keys[4] = {
                    ordered_float_key(scores.x), ordered_float_key(scores.y),
                    ordered_float_key(scores.z), ordered_float_key(scores.w)};
                const int first_region = scalar_head + vector_id * 4;
#pragma unroll
                for (int item = 0; item < 4; ++item) {
                    if (keys[item] >= threshold) {
                        if (local_candidates < kCandidateCacheCapacity) {
                            cached_candidate_ids[local_candidates] =
                                first_region + item;
                        } else {
                            local_overflow = true;
                        }
                        ++local_candidates;
                    }
                }
            }
            const int scalar_tail = scalar_head + vector_count * 4;
            for (int region = scalar_tail + tid; region < visible;
                 region += BLOCK_THREADS) {
                const uint32_t key = ordered_float_key(row_logits[region]);
                if (key >= threshold) {
                    if (local_candidates < kCandidateCacheCapacity) {
                        cached_candidate_ids[local_candidates] = region;
                    } else {
                        local_overflow = true;
                    }
                    ++local_candidates;
                }
            }
            if (tid == 0) candidate_overflow = 0;
            __syncthreads();
            if (local_overflow) atomicExch(&candidate_overflow, 1);
        } else {
            for (int region = tid; region < visible; region += BLOCK_THREADS) {
                const uint32_t key = ordered_float_key(row_logits[region]);
                local_candidates += static_cast<int>(key >= threshold);
            }
        }
        const int candidate_offset = block_count_exclusive_prefix<BLOCK_THREADS>(
            local_candidates, selected_wave_counts, &candidate_total);

        bool use_candidate_fast_path = candidate_total == topk;
        if constexpr (CACHE_CANDIDATES) {
            // The cached count uses float4/vector ownership. Re-scanning with
            // scalar thread ownership would invalidate candidate_offset and
            // could overlap or overrun candidate_ids. Fall back to the stable
            // full-row emission path whenever any thread exceeds its cache.
            use_candidate_fast_path =
                use_candidate_fast_path && candidate_overflow == 0;
        }
        if (use_candidate_fast_path) {
            if constexpr (CACHE_CANDIDATES) {
                for (int index = 0; index < local_candidates; ++index) {
                    candidate_ids[candidate_offset + index] =
                        cached_candidate_ids[index];
                }
            } else {
                int position = candidate_offset;
                for (int region = tid; region < visible;
                     region += BLOCK_THREADS) {
                    const uint32_t key = ordered_float_key(row_logits[region]);
                    if (key >= threshold) {
                        candidate_ids[position++] = region;
                    }
                }
            }
            __syncthreads();

            // Fixed-size in-block bitonic sort restores the public contract:
            // valid region ids are emitted in ascending logical-id order.
#pragma unroll
            for (int size = 2; size <= kDecodeTopK; size <<= 1) {
#pragma unroll
                for (int stride = size >> 1; stride > 0; stride >>= 1) {
                    for (int index = tid; index < kDecodeTopK;
                         index += BLOCK_THREADS) {
                        const int partner = index ^ stride;
                        if (partner > index) {
                            const int lhs = candidate_ids[index];
                            const int rhs = candidate_ids[partner];
                            const bool ascending = (index & size) == 0;
                            if ((ascending && lhs > rhs) ||
                                (!ascending && lhs < rhs)) {
                                candidate_ids[index] = rhs;
                                candidate_ids[partner] = lhs;
                            }
                        }
                    }
                    __syncthreads();
                }
            }
            for (int slot = tid; slot < kDecodeTopK; slot += BLOCK_THREADS) {
                row_out[slot] = candidate_ids[slot];
            }
            return;
        }
    }

    if (tid == 0) {
        emit_cursor = 0;
        ties_seen = 0;
    }
    __syncthreads();

    // Stable fallback for a partially selected threshold tie.
    for (int base = 0; base < visible; base += BLOCK_THREADS) {
        const int region = base + tid;
        const bool live = region < visible;
        uint32_t key = 0;
        if (live) {
            key = ordered_float_key(row_logits[region]);
        }
        const bool tied = live && key == threshold;
        const bool greater = live && key > threshold;

        const int tied_prefix =
            block_flag_prefix<BLOCK_THREADS>(tied, tied_wave_counts);

        const bool selected = greater || (tied && ties_seen + tied_prefix <= remaining);
        const int selected_prefix = block_flag_prefix<BLOCK_THREADS>(
            selected, selected_wave_counts);

        if (selected) {
            const int position = emit_cursor + selected_prefix - 1;
            if (position < topk) {
                row_out[position] = region;
            }
        }
        __syncthreads();
        if (tid == 0) {
            int tile_selected = 0;
            int tile_tied = 0;
#pragma unroll
            for (int wave = 0; wave < kWaves; ++wave) {
                tile_selected += selected_wave_counts[wave];
                tile_tied += tied_wave_counts[wave];
            }
            emit_cursor += tile_selected;
            ties_seen += tile_tied;
        }
        __syncthreads();
    }
}

template <int BLOCK_THREADS>
void launch_region_topk_ids_kernel(const float* logits,
                                   const int32_t* lengths,
                                   int32_t* out,
                                   int32_t rows,
                                   int32_t seq_regions,
                                   int64_t stride_logits_row,
                                   int64_t stride_out_row,
                                   int32_t topk,
                                   KernelMode mode,
                                   cudaStream_t stream) {
    const unsigned int blocks = static_cast<unsigned int>(rows);
    switch (mode) {
        case KernelMode::kCoarseFilterAndCachedCandidates:
            if constexpr (BLOCK_THREADS > kShortRowThreads) {
                region_topk_ids_kernel<BLOCK_THREADS, true, false, true>
                    <<<blocks, BLOCK_THREADS, 0, stream>>>(
                        logits, lengths, out, seq_regions, stride_logits_row,
                        stride_out_row, topk);
                break;
            }
            [[fallthrough]];
        case KernelMode::kCoarseFilter:
            region_topk_ids_kernel<BLOCK_THREADS, false, false, true>
                <<<blocks, BLOCK_THREADS, 0, stream>>>(
                    logits, lengths, out, seq_regions, stride_logits_row,
                    stride_out_row, topk);
            break;
        case KernelMode::kCachedRadixKeys:
            region_topk_ids_kernel<BLOCK_THREADS, false, true, false>
                <<<blocks, BLOCK_THREADS, 0, stream>>>(
                    logits, lengths, out, seq_regions, stride_logits_row,
                    stride_out_row, topk);
            break;
        case KernelMode::kGeneral:
            region_topk_ids_kernel<BLOCK_THREADS, false, false, false>
                <<<blocks, BLOCK_THREADS, 0, stream>>>(
                    logits, lengths, out, seq_regions, stride_logits_row,
                    stride_out_row, topk);
            break;
    }
}

inline bool invalid_region_topk_shape(int32_t rows,
                                      int32_t seq_regions,
                                      int64_t stride_logits_row,
                                      int64_t stride_out_row,
                                      int32_t topk) {
    return rows < 0 || seq_regions < 0 || topk < 0 ||
           stride_logits_row < seq_regions || stride_out_row < topk;
}

// Public C++ interface. Row strides are measured in elements. The call is
// asynchronous with respect to stream and all pointers refer to device memory.
cudaError_t launch_region_topk_ids(const float* logits,
                            const int32_t* lengths,
                            int32_t* out,
                            int32_t rows,
                            int32_t seq_regions,
                            int64_t stride_logits_row,
                            int64_t stride_out_row,
                            int32_t topk,
                            cudaStream_t stream) {
    if (invalid_region_topk_shape(rows, seq_regions, stride_logits_row,
                                  stride_out_row, topk)) {
        return cudaErrorInvalidValue;
    }
    if (rows == 0 || topk == 0) {
        return cudaSuccess;
    }
    if (lengths == nullptr || out == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (topk >= seq_regions) {
        region_topk_ids_fill_all_kernel<kFillAllThreads>
            <<<static_cast<unsigned int>(rows), kFillAllThreads, 0, stream>>>(
                lengths, out, seq_regions, stride_out_row, topk);
        return cudaGetLastError();
    }
    if (logits == nullptr) {
        return cudaErrorInvalidValue;
    }
    // Candidate caching is useful only for long decode rows, where coarse
    // filtering has already reduced the radix input set.
    const bool cache_candidates =
        topk == kDecodeTopK && seq_regions >= kCandidateCacheMinRegions;
    const bool cache_radix_keys =
        kRadixKeyCacheMaxRegions > 0 &&
        seq_regions <= kRadixKeyCacheMaxRegions;
    const bool coarse_filter =
        topk == kDecodeTopK && seq_regions >= kCoarseFilterMinRegions;
    if (topk == kDecodeTopK && seq_regions <= 1024 &&
        seq_regions > kShortRowThreads) {
        launch_region_topk_ids_kernel<kMidRowsThreads>(
            logits, lengths, out, rows, seq_regions, stride_logits_row,
            stride_out_row, topk, KernelMode::kGeneral, stream);
    } else if (rows <= 56 && seq_regions > kShortRowThreads) {
        const KernelMode mode =
            coarse_filter
                ? coarse_filter_mode(cache_candidates)
                : (rows <= 32 && cache_radix_keys
                       ? KernelMode::kCachedRadixKeys
                       : KernelMode::kGeneral);
        launch_region_topk_ids_kernel<kFewRowsThreads>(
            logits, lengths, out, rows, seq_regions, stride_logits_row,
            stride_out_row, topk, mode, stream);
    } else if (rows <= 64 && seq_regions > kShortRowThreads) {
        const KernelMode mode =
            coarse_filter
                ? coarse_filter_mode(cache_candidates)
                : KernelMode::kGeneral;
        launch_region_topk_ids_kernel<kMidRowsThreads>(
            logits, lengths, out, rows, seq_regions, stride_logits_row,
            stride_out_row, topk, mode, stream);
    } else {
        const KernelMode mode = coarse_filter
                                    ? KernelMode::kCoarseFilter
                                    : KernelMode::kGeneral;
        launch_region_topk_ids_kernel<kLargeRowsThreads>(
            logits, lengths, out, rows, seq_regions, stride_logits_row,
            stride_out_row, topk, mode, stream);
    }
    return cudaGetLastError();
}

}  // namespace

torch::Tensor region_topk_ids(const torch::Tensor& logits,
                              const torch::Tensor& lengths, int64_t topk) {
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  TORCH_CHECK(lengths.is_cuda(), "lengths must be a CUDA tensor");
  TORCH_CHECK(logits.device() == lengths.device(),
              "logits and lengths must be on the same device");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must have dtype float32");
  TORCH_CHECK(lengths.scalar_type() == torch::kInt32,
              "lengths must have dtype int32");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be 1D");
  TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");
  TORCH_CHECK(logits.stride(1) == 1, "logits stride(1) must be 1");
  TORCH_CHECK(logits.stride(0) >= logits.size(1),
              "logits row stride must cover seq_regions");
  TORCH_CHECK(lengths.numel() == logits.size(0),
              "lengths size must match logits rows");
  TORCH_CHECK(topk >= 0, "topk must be non-negative");
  TORCH_CHECK(logits.size(0) <= std::numeric_limits<int32_t>::max() &&
                  logits.size(1) <= std::numeric_limits<int32_t>::max() &&
                  topk <= std::numeric_limits<int32_t>::max(),
              "rows, seq_regions, and topk must fit in int32");

  at::cuda::OptionalCUDAGuard const device_guard(logits.device());
  auto out = torch::empty({logits.size(0), topk},
                          logits.options().dtype(torch::kInt32));
  const cudaError_t status = launch_region_topk_ids(
      logits.data_ptr<float>(), lengths.data_ptr<int32_t>(),
      out.data_ptr<int32_t>(),
      static_cast<int32_t>(logits.size(0)),
      static_cast<int32_t>(logits.size(1)), logits.stride(0), out.stride(0),
      static_cast<int32_t>(topk), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cudaSuccess,
              "region_topk_ids launch failed: ", cudaGetErrorString(status));
  return out;
}
