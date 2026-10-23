#include "cuda_compat.h"
#include "dispatch_utils.h"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstring>
#include <cfloat>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>

namespace vllm {

template <typename scalar_t>
__global__ void apply_repetition_penalties_kernel(
    scalar_t* __restrict__ logits,         // [num_seqs, vocab_size]
    const bool* __restrict__ prompt_mask,  // [num_seqs, vocab_size]
    const bool* __restrict__ output_mask,  // [num_seqs, vocab_size]
    const scalar_t* __restrict__ repetition_penalties,  // [num_seqs]
    const int num_seqs, const int vocab_size, const int tile_size) {
  // Each block handles one sequence and a tile of vocab
  const int seq_idx = blockIdx.x;
  if (seq_idx >= num_seqs) return;

  const int tile_start = blockIdx.y * tile_size;
  const int tile_end = min(tile_start + tile_size, vocab_size);

  // Load repetition penalty for this sequence
  const scalar_t penalty = repetition_penalties[seq_idx];

  // Each thread processes multiple vocab items within the tile
  for (int vocab_idx = tile_start + threadIdx.x; vocab_idx < tile_end;
       vocab_idx += blockDim.x) {
    const int64_t idx = static_cast<int64_t>(seq_idx) * vocab_size + vocab_idx;
    const bool is_repeated = prompt_mask[idx] || output_mask[idx];
    if (is_repeated) {
      scalar_t logit = logits[idx];
      if (logit > 0) {
        logits[idx] = logit / penalty;
      } else {
        logits[idx] = logit * penalty;
      }
    }
  }
}

}  // namespace vllm
// =============================================================================
// Legacy topKPerRowPrefill histogram-sort fallback (rows > kTopkSbCap).
// Ported verbatim from sampler_prefill.cu. Uses half-11-11-10 bit iterations
// and cub::BlockRadixSort / BlockScan. Names verified collision-free with the
// vllm::topk_impl single-pass kernels.
// =============================================================================
namespace vllm {
__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
}

template <int step>
static inline __device__ uint32_t extractBinIdx(float x) {
  if constexpr (step == 0) {
    __half hx = __float2half(x);
    uint16_t bits = __half_as_ushort(hx);
    bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
    return bits >> 5;
  } else {
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;

    if constexpr (step == 1) {
      return bits >> 21;
    } else if constexpr (step == 2) {
      return (bits >> 10) & 0x7ff;
    } else if constexpr (step == 3) {
      return bits & 0x3ff;
    }
  }
}

template <int shift>
static inline __device__ bool isPartialMatch(float x, uint32_t pattern) {
  if constexpr (shift == 0) {
    return true;
  }
  uint32_t bits = __float_as_uint(x);
  bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
  return (bits ^ pattern) >> shift == 0;
}

/**
 * Map a Func over the input data, using vectorized load instructions if
 * possible.
 *
 * @tparam T element type
 * @tparam IdxT indexing type
 * @tparam Func void (T x, IdxT idx)
 *
 * @param thread_rank rank of the calling thread among all participating threads
 * @param num_threads number of the threads that participate in processing
 * @param in the input data
 * @param len the number of elements to read
 * @param f the lambda taking two arguments (T x, IdxT idx)
 */
template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads,
                                   const T* in, idxT len, Func f) {
  // Use dynamic WARP_SIZE from cuda_compat.h to support both
  // Wave64 (MI300X/gfx942) and Wave32 (Strix Halo/gfx1151) architectures
  constexpr int kWarpSize = WARP_SIZE;
  using WideT = float4;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (idxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
    // TODO: it's UB
    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_cnt =
        (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) /
               sizeof(T))
            : 0;
    if (skip_cnt > len) {
      skip_cnt = len;
    }
    const WideT* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    const idxT len_cast = (len - skip_cnt) / items_per_scalar;

    for (idxT i = thread_rank; i < len_cast; i += num_threads) {
      wide.scalar = in_cast[i];
      const idxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    static_assert(kWarpSize >= items_per_scalar);
    // and because items_per_scalar > skip_cnt, kWarpSize > skip_cnt
    // no need to use loop
    if (thread_rank < skip_cnt) {
      f(in[thread_rank], thread_rank);
    }
    // because len_cast = (len - skip_cnt) / items_per_scalar,
    // len_cast * items_per_scalar + items_per_scalar > len - skip_cnt;
    // and so
    // len - (skip_cnt + len_cast * items_per_scalar) < items_per_scalar <=
    // kWarpSize no need to use loop
    const idxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
    if (remain_i < len) {
      f(in[remain_i], remain_i);
    }
  }
}

template <int step, int kNumThreadsPerBlock, int kNumBins, int kNumFinalItems,
          bool multipleBlocksPerRow, bool mergeBlocks, typename SmemFinalType,
          typename SmemOutputType>
__device__ bool processHistogramStep(
    const int* indices, const float* logits, int rowEnd, uint32_t& logitPattern,
    int& thresholdBinIdx, SmemOutputType& smemOutput, int* smemThresholdBinIdx,
    int* smemFinalDstIdx, int* smemFinalBinSize, int* smemFoundTopKValues,
    SmemFinalType& smemFinal, int stride1, int rowStart, int topK) {
  // Clear the histogram.
#pragma unroll
  for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
    smemFinal.histo.data[idx] = 0;
  }

  // Make sure the histogram is ready.
  __syncthreads();

  // Update pattern
  constexpr auto patternShift = step < 2 ? 0 : step == 2 ? 21 : 10;
  if constexpr (step == 2) {
    logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff)
                   << patternShift;
  } else if constexpr (step == 3) {
    logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff)
                    << patternShift;
  }

  auto distributeToBins = [&](float logit, int /* idx */ = 0) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      atomicAdd(&smemFinal.histo.data[binIdx], 1);
    }
  };

  // Distribute the elements to the histogram bins.
  if (stride1 == 1) {
    vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart,
                       rowEnd - rowStart, distributeToBins);
  } else {
    for (int idx = rowStart + threadIdx.x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      distributeToBins(logit, idx);
    }
  }
  // Make sure the histogram is ready.
  __syncthreads();

  // Reads the value of the starting position in the smemOutput array
  int lastValue = smemFoundTopKValues[0];

  for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++) {
    // Read the values from SMEM.
    int idx = threadIdx.x + kNumThreadsPerBlock * round;
    int binCount{0};
    binCount = smemFinal.histo.data[idx];

    // Make sure each thread has read its value.
    __syncthreads();

    // Compute the prefix sum.
    int prefixSum{0}, totalSum{0};
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
    Scan(smemFinal.histo.scan).ExclusiveSum(binCount, prefixSum, totalSum);

    // Update the histogram with the prefix sums.
    prefixSum += lastValue;
    totalSum += lastValue;
    smemFinal.histo.data[idx] = prefixSum;

    // Make sure the data is in shared memory.
    __syncthreads();

    // Find the last valid bin.
    bool foundThreshold = false;
    if (prefixSum < topK) {
      int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1
                              ? totalSum
                              : smemFinal.histo.data[idx + 1];

      if (nextPrefixSum >= topK) {
        smemThresholdBinIdx[0] = idx;
        smemFinalBinSize[0] = nextPrefixSum - prefixSum;
        foundThreshold = true;
      }
    }

    // Early exit: if any thread found the threshold, we can skip remaining
    // rounds
    if (__syncthreads_or(foundThreshold)) {
      break;
    }

    lastValue = totalSum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The threshold bin.
  thresholdBinIdx = smemThresholdBinIdx[0];

  auto processBins = [&](float logit, int idx) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      // Only write elements with binIdx < thresholdBinIdx when:
      // 1. This is step 0 and the threshold bin is small enough (no step 1)
      // 2. This is step >= 1 (where pattern matching filters correctly)
      // This prevents duplicates when step 0 and step 1 both run.
      bool shouldWriteDirectly =
          (step == 0 && smemFinalBinSize[0] <= kNumFinalItems) || (step >= 1);
      if (binIdx < thresholdBinIdx && shouldWriteDirectly) {
        // The element is part of the top-k selection
        int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);

        if constexpr (mergeBlocks) {
          smemOutput[dstIdx] = indices[idx];
        } else if constexpr (multipleBlocksPerRow) {
          smemOutput[dstIdx] = idx + rowStart;
          reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
        } else {
          smemOutput[dstIdx] = idx;
        }
      }
      if constexpr (step < 3) {
        // Only fill the final items for sorting if the threshold bin fits
        if (binIdx == thresholdBinIdx &&
            smemFinalBinSize[0] <= kNumFinalItems) {
          int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
          smemFinal.items.logits[dstIdx] = logit;
          if constexpr (mergeBlocks) {
            smemFinal.items.indices[dstIdx] = indices[idx];
          } else if constexpr (multipleBlocksPerRow) {
            smemFinal.items.indices[dstIdx] = idx + rowStart;
          } else {
            smemFinal.items.indices[dstIdx] = idx;
          }
        }
      } else {
        if (binIdx == thresholdBinIdx) {
          // The elements in the threshold bin share the same 32 bits at step 3
          int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
          if (dstIdx < topK) {
            if constexpr (mergeBlocks) {
              smemOutput[dstIdx] = indices[idx];
            } else if constexpr (multipleBlocksPerRow) {
              smemOutput[dstIdx] = idx + rowStart;
              reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
            } else {
              smemOutput[dstIdx] = idx;
            }
          }
        }
      }
    }
  };

  if (stride1 == 1) {
    vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart,
                       rowEnd - rowStart, processBins);
  } else {
    for (int idx = rowStart + threadIdx.x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      processBins(logit, idx);
    }
  }

  // Make sure the elements are in shared memory.
  __syncthreads();

  // Check if we should continue to next step
  return smemFinalBinSize[0] > kNumFinalItems;
}

// Follows half - 11 - 11 - 10 bit iterations
template <int kNumThreadsPerBlock, int kNumBins, bool useRadixSort,
          bool multipleBlocksPerRow = false, bool mergeBlocks = false>
static __device__ void topKPerRowJob(const int* indices, const float* logits,
                                     int rowStart, int rowEnd, int* outIndices,
                                     float* outLogits, int stride1, int topK) {
  // The number of slots for the final pass.
  static constexpr int kNumFinalItems = 2048;
  // The number of elements per thread for the final sort.
  static constexpr int kNumFinalItemsPerThread =
      kNumFinalItems / kNumThreadsPerBlock;
  // The class to sort the elements during the final pass.
  using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock,
                                        kNumFinalItemsPerThread, int>;
  using FinalSortTempStorage =
      std::conditional_t<useRadixSort, typename FinalSort::TempStorage, int>;
  // The class to compute the inclusive prefix-sum over the histogram.
  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

  // The structure to store the final items (for the final pass).
  struct FinalItems {
    // Shared memory to store the indices for the final pass.
    int indices[kNumFinalItems];
    // Shared memory to store the logits for the final pass.
    float logits[kNumFinalItems];
  };

  struct Histogram {
    typename Scan::TempStorage scan;
    int data[kNumBins];
  };

  // Shared memory to compute the block sort.
  __shared__ union {
    FinalItems items;
    FinalSortTempStorage finalSort;
    Histogram histo;
  } smemFinal;

  // Shared memory to store the selected indices.
  // If we are processing using multiple blocks, we need to store the logits and
  // indices.
  extern __shared__ int32_t smemOutput[];

  // Shared memory to store the threshold bin.
  __shared__ int smemThresholdBinIdx[1];
  // Shared memory counter to register the candidates for the final phase.
  __shared__ int smemFinalDstIdx[1];
  // Shared memory to determine if the threshold bin fits in the final items.
  __shared__ int smemFinalBinSize[1];
  // Shared memory to keep track of the top-k values found so far by the
  // previous iterations
  __shared__ int smemFoundTopKValues[1];

  // The length of the row.
  int rowLen = rowEnd - rowStart;

  // Shortcut if the length of the row is smaller than Top-K. Indices are not
  // sorted by their corresponding logit.
  if (rowLen <= topK) {
    for (int rowIt = threadIdx.x; rowIt < rowLen;
         rowIt += kNumThreadsPerBlock) {
      if constexpr (multipleBlocksPerRow) {
        outIndices[rowIt] = rowIt + rowStart;
        outLogits[rowIt] = logits[rowIt + rowStart];
      } else {
        outIndices[rowIt] = rowIt;
      }
    }
    for (int rowIt = rowLen + threadIdx.x; rowIt < topK;
         rowIt += kNumThreadsPerBlock) {
      outIndices[rowIt] = -1;
      if constexpr (multipleBlocksPerRow) {
        outLogits[rowIt] = -FLT_MAX;
      }
    }

    return;
  }
  // Initialize values
  if (threadIdx.x == 0) {
    smemFinalDstIdx[0] = 0;
    smemFoundTopKValues[0] = 0;
  }
  __syncthreads();
  int thresholdBinIdx = -1;
  uint32_t logitPattern = 0;

  // Step 0: Process first 11 bits of half representation
  bool continueToNextStep =
      processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                           multipleBlocksPerRow, mergeBlocks>(
          indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
          smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
          smemFoundTopKValues, smemFinal, stride1, rowStart, topK);

  if (continueToNextStep) {
    // Step 1: Process next 11 bits
    continueToNextStep =
        processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                             multipleBlocksPerRow, mergeBlocks>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
            smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (continueToNextStep) {
    // Step 2: Process next 11 bits
    continueToNextStep =
        processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                             multipleBlocksPerRow, mergeBlocks>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
            smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (continueToNextStep) {
    // Step 3: Process last 10 bits
    processHistogramStep<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                         multipleBlocksPerRow, mergeBlocks>(
        indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
        smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
        smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (!continueToNextStep) {
    // The histogram did not proceed to the final 10 bits, therefore we need to
    // sort the final items The logits of the elements to be sorted in the final
    // pass.
    if constexpr (useRadixSort) {
      // Sorting with radix sort
      float finalLogits[kNumFinalItemsPerThread];
      // The indices of the elements to be sorted in the final pass.
      int finalIndices[kNumFinalItemsPerThread];

#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        finalLogits[ii] = -FLT_MAX;
      }

      // Read the elements from SMEM.
#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
        if (srcIdx < smemFinalDstIdx[0]) {
          finalLogits[ii] = smemFinal.items.logits[srcIdx];
          finalIndices[ii] = smemFinal.items.indices[srcIdx];
        }
      }
      // Make sure the shared memory has been read.
      __syncthreads();

      // Sort the elements.
      FinalSort(smemFinal.finalSort)
          .SortDescendingBlockedToStriped(finalLogits, finalIndices);

      // Copy the data back to the shared memory storage.
      int baseIdx = smemFoundTopKValues[0];

#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
        int dstIdx = baseIdx + srcIdx;

        if (dstIdx < topK) {
          smemOutput[dstIdx] = finalIndices[ii];
          if constexpr (multipleBlocksPerRow) {
            reinterpret_cast<float*>(smemOutput + topK)[dstIdx] =
                finalLogits[ii];
          }
        }
      }
    } else {
      // Sorting with insertion sort
      auto baseIdx = smemFoundTopKValues[0];
      for (int i = threadIdx.x; i < smemFinalDstIdx[0];
           i += kNumThreadsPerBlock) {
        int outIndex = 0;
        auto logit = smemFinal.items.logits[i];
        for (int j = 0; j < smemFinalDstIdx[0]; j++) {
          auto otherLogit = smemFinal.items.logits[j];
          if (logit < otherLogit || (logit == otherLogit && i < j)) {
            outIndex++;
          }
        }
        // Store if outIndex is in bounds
        if (outIndex + baseIdx < topK) {
          smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
          if constexpr (multipleBlocksPerRow) {
            reinterpret_cast<float*>(smemOutput + topK)[outIndex + baseIdx] =
                smemFinal.items.logits[i];
          }
        }
      }
    }
    __syncthreads();
  }

  // Store to global memory.
  for (int i = threadIdx.x; i < topK; i += kNumThreadsPerBlock) {
    if constexpr (multipleBlocksPerRow) {
      outIndices[i] = smemOutput[i];
      outLogits[i] = reinterpret_cast<float*>(smemOutput + topK)[i];
    } else {
      if (stride1 == 1) {
        // stride1 == 1 will use vectorized_process, which indexes already skip
        // the rowStart.
        outIndices[i] = smemOutput[i];
      } else {
        outIndices[i] = smemOutput[i] - rowStart;
      }
    }
  }
}

template <int kNumThreadsPerBlock, bool useRadixSort>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowPrefill(
    const float* logits, const int* rowStarts, const int* rowEnds,
    int* outIndices, int stride0, int stride1, const int topK,
    const int offsetIndex) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 2048;

  // The row computed by this block.
  int rowIdx = blockIdx.x + offsetIndex;

  // The range of logits within the row.
  int rowStart = rowStarts[rowIdx];
  int rowEnd = rowEnds[rowIdx];

  // Local pointers to this block
  outIndices += static_cast<int64_t>(rowIdx) * topK;
  logits += static_cast<int64_t>(rowIdx) * stride0;

  topKPerRowJob<kNumThreadsPerBlock, kNumBins, useRadixSort>(
      nullptr, logits, rowStart, rowEnd, outIndices, nullptr, stride1, topK);
}
}  // namespace vllm

// =============================================================================
// top_k_per_row: single-pass register-cached exact top-K (prefill fast path).
// Ported verbatim from sampler_prefill.cu. Used by the prefill launcher for
// rows <= kTopkSbCap (32768). The decode launcher is untouched and lives in
// vllm::topk_impl.
// =============================================================================
namespace vllm {
// top_k_per_row: single-pass register-cached exact top-K
// -----------------------------------------------------------------------------
// Reads each row from DRAM exactly ONCE into registers, converts to a 32-bit
// ordered key (convert_to_uint32_v2 semantics), then resolves the exact 32-bit
// pivot with 4 x 8-bit masked radix-select rounds. Histograms are built
// warp-privately in shared memory (no cross-warp atomic contention) and reduced
// to a single 256-bin array before warp-0 suffix-detect. The final collect
// re-scans registers: key > pivot is emitted directly, key == pivot fills the
// remaining slots (all equal values -> any of them are valid). No second DRAM
// pass, no candidate staging that can truncate ties.
//
// Handles rows up to kTopkSbCap = kBlockSize*kVecsPerThread*4. Rows beyond the
// capacity are dispatched by the host to the existing multi-block paths.
// Output indices are row-local within [rowStart, rowEnd), matching
// topKPerRowJob single-block semantics.
// =============================================================================
constexpr int kTopkSbBlockSize = 1024;
constexpr int kTopkSbVecsPerThread = 8;  // 32 float elements per thread
constexpr int kTopkSbCap =
    kTopkSbBlockSize * kTopkSbVecsPerThread * 4;  // 32768
constexpr int kTopkSbRadixBins = 256;             // 8 bits per fine round
constexpr int kTopkSbCoarseBins = 1024;           // 10-bit coarse histogram
constexpr int kTopkSbCoarseShift = 22;            // key >> 22 = top 10 bits
constexpr int kTopkSbStageCap = 2048;             // max staged candidates
constexpr uint64_t kTopkSbFullMask = 0xffffffffffffffffULL;

__device__ __forceinline__ uint32_t topk_sb_key(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}


// Inclusive prefix sum over the 64 lanes of a MACA warp (hardware shuffle
// subgroup is 16 lanes, so use the full-warp log-sweep).
__device__ __forceinline__ uint32_t topk_sb_warp_scan(uint32_t v) {
  const int lane = static_cast<int>(threadIdx.x & 63);
#pragma unroll
  for (int off = 1; off < 64; off <<= 1) {
    const uint32_t up = __shfl_up_sync(kTopkSbFullMask, v, off);
    if (lane >= off) v += up;
  }
  return v;
}

// Warp 0 computes the histogram suffix counts and locates the unique
// threshold bin: count_ge >= need && count_gt < need. Replaces a full-block
// Hillis-Steele scan with one (or two) warp shuffle scan(s).
template <int kBins>
__device__ __forceinline__ void topk_sb_detect(const uint32_t* hist,
                                               uint32_t need, int* s_bin,
                                               int* s_cgt) {
  if (threadIdx.x < 64) {
    const int base_b = static_cast<int>(threadIdx.x & 63) * kBins;
    uint32_t local[kBins], lsum = 0;
#pragma unroll
    for (int j = 0; j < kBins; ++j) {
      local[j] = hist[base_b + j];
      lsum += local[j];
    }
    const uint32_t incl = topk_sb_warp_scan(lsum);
    const uint32_t total = __shfl_sync(kTopkSbFullMask, incl, 63);
    uint32_t cge = total - (incl - lsum);  // count in bins >= this lane's 1st
#pragma unroll
    for (int j = 0; j < kBins; ++j) {
      const uint32_t cgt = cge - local[j];
      if (cge >= need && cgt < need) {
        *s_bin = base_b + j;
        *s_cgt = static_cast<int>(cgt);
      }
      cge = cgt;
    }
  }
}

// Reserve a contiguous run of `my_count` slots from a shared counter for this
// warp and return the first slot of this thread's run. Replaces per-element
// same-address smem atomics with one atomic per warp.
__device__ __forceinline__ uint32_t topk_sb_warp_ticket(int* counter,
                                                        uint32_t my_count) {
  const uint32_t incl = topk_sb_warp_scan(my_count);
  const uint32_t warp_total = __shfl_sync(kTopkSbFullMask, incl, 63);
  uint32_t base = 0;
  if ((threadIdx.x & 63) == 63 && warp_total > 0)
    base = static_cast<uint32_t>(::atomicAdd(counter, static_cast<int>(warp_total)));
  base = __shfl_sync(kTopkSbFullMask, base, 63);
  return base + incl - my_count;
}

template <int kBlockSize, int kVecsPerThread>
__global__ __launch_bounds__(kBlockSize) void topKPerRowSingleBlockKernel(
    const float* __restrict__ logits, const int* __restrict__ seqLens,
    const int* __restrict__ rowStarts, const int* __restrict__ rowEnds,
    int* __restrict__ outIndices, int numRows, int stride0, int stride1,
    int topK, int isDecode, int next_n, int seqLensIs2D) {
  constexpr int kPerThread = kVecsPerThread * 4;
  constexpr int kNumWarps = kBlockSize / 64;
  const int tx = threadIdx.x;
  const int lane = tx & 63;
  const int warp = tx >> 6;

  // Staging capacity scales down with block size so small-row blocks use far
  // less shared memory and achieve higher occupancy. Overflowing candidates
  // fall back to the exact register radix path, so a smaller cap never breaks
  // correctness.
  constexpr int kStageCap =
      (kBlockSize >= 1024) ? kTopkSbStageCap : (kBlockSize >= 512) ? 1024 : 512;
  __shared__ uint32_t s_hist_coarse[kTopkSbCoarseBins];  // 10-bit coarse hist
  __shared__ uint32_t s_hist_fine[kTopkSbRadixBins];     // 8-bit fine hist
  __shared__ uint32_t s_stage_key[2][kStageCap];         // staged candidate keys
  __shared__ int s_stage_idx[2][kStageCap];              // staged candidate idx
  __shared__ int s_threshold_bin;
  __shared__ int s_count_gt;
  __shared__ int s_out_counter;
  __shared__ int s_stage_counter;
  __shared__ int s_tie_counter;
  __shared__ int s_last_remain;

  const int rowIdx = blockIdx.x;
  if (rowIdx >= numRows) return;

  int rowStart, rowEnd;
  if (isDecode) {
    const int batch = rowIdx / next_n;
    const int ni = rowIdx % next_n;
    const int seq_len = seqLensIs2D ? seqLens[rowIdx] : seqLens[batch];
    rowStart = 0;
    rowEnd = seqLensIs2D ? max(0, seq_len)
                         : max(0, seq_len - next_n + ni + 1);
  } else {
    rowStart = rowStarts[rowIdx];
    rowEnd = rowEnds[rowIdx];
  }
  const int rowLen = rowEnd - rowStart;

  if (rowLen <= topK) {  // shortcut: emit everything, pad -1
    for (int i = tx; i < rowLen; i += kBlockSize)
      outIndices[(int64_t)rowIdx * topK + i] = i;
    for (int i = rowLen + tx; i < topK; i += kBlockSize)
      outIndices[(int64_t)rowIdx * topK + i] = -1;
    return;
  }
  if (rowLen > kTopkSbCap) return;  // host guarantees <= cap; defensive

  const float* rowPtr = logits + (int64_t)rowIdx * stride0 + rowStart;
  const bool canVec = (stride1 == 1) &&
                      ((reinterpret_cast<uintptr_t>(rowPtr) & 0xFu) == 0);

  // Phase 0: single DRAM read; convert to ordered uint32 keys in registers.
  // The coarse histogram is folded into the same loop: each vec read feeds 4
  // smem atomics immediately, so DRAM latency overlaps histogram compute.
  uint32_t key[kPerThread];
  if (canVec) {
    const float kNegInf = __int_as_float(0xFF800000);
    if (tx == 0) { s_out_counter = 0; s_stage_counter = 0; s_tie_counter = 0; }
    for (int i = tx; i < kTopkSbCoarseBins; i += kBlockSize)
      s_hist_coarse[i] = 0;
    __syncthreads();
#pragma unroll
    for (int v = 0; v < kVecsPerThread; ++v) {
      const int base = (tx + v * kBlockSize) * 4;
      float4 f;
      if (base + 3 < rowLen) {
        f = *reinterpret_cast<const float4*>(rowPtr + base);
      } else {
        const float* p = rowPtr + base;
        f.x = (base + 0 < rowLen) ? p[0] : kNegInf;
        f.y = (base + 1 < rowLen) ? p[1] : kNegInf;
        f.z = (base + 2 < rowLen) ? p[2] : kNegInf;
        f.w = (base + 3 < rowLen) ? p[3] : kNegInf;
      }
      const uint32_t k0 = topk_sb_key(f.x), k1 = topk_sb_key(f.y);
      const uint32_t k2 = topk_sb_key(f.z), k3 = topk_sb_key(f.w);
      key[v * 4 + 0] = k0; key[v * 4 + 1] = k1;
      key[v * 4 + 2] = k2; key[v * 4 + 3] = k3;
      atomicAdd(&s_hist_coarse[k0 >> kTopkSbCoarseShift], 1u);
      atomicAdd(&s_hist_coarse[k1 >> kTopkSbCoarseShift], 1u);
      atomicAdd(&s_hist_coarse[k2 >> kTopkSbCoarseShift], 1u);
      atomicAdd(&s_hist_coarse[k3 >> kTopkSbCoarseShift], 1u);
    }
  } else {
    const float kNegInf = __int_as_float(0xFF800000);
    if (tx == 0) { s_out_counter = 0; s_stage_counter = 0; s_tie_counter = 0; }
    for (int i = tx; i < kTopkSbCoarseBins; i += kBlockSize)
      s_hist_coarse[i] = 0;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const int idx = tx + i * kBlockSize;
      const uint32_t k = topk_sb_key((idx < rowLen) ? rowPtr[idx * stride1] : kNegInf);
      key[i] = k;
      atomicAdd(&s_hist_coarse[k >> kTopkSbCoarseShift], 1u);
    }
  }
  __syncthreads();

  // ============================ Phase 1/2 ============================
  // One 10-bit coarse histogram of the ordered keys locates the threshold bin.
  // All elements with coarse bin strictly above it are provably part of the
  // top-K and are emitted directly; the (topK - count_gt) largest elements
  // inside the threshold bin are selected by staging the ==bin candidates and
  // refining them with 8-bit masked-radix passes. Extreme ties (candidates
  // overflowing the staging cap) fall back to an exact full-width masked radix
  // over the register keys.
  const int64_t rowOut = static_cast<int64_t>(rowIdx) * topK;

  topk_sb_detect<kTopkSbCoarseBins / 64>(s_hist_coarse,
                                         static_cast<uint32_t>(topK),
                                         &s_threshold_bin, &s_count_gt);
  __syncthreads();
  const uint32_t tb = static_cast<uint32_t>(s_threshold_bin);
  const int n_tb = static_cast<int>(s_hist_coarse[tb]);
  int need = topK - s_count_gt;

  if (n_tb > kStageCap) {
    // ---- fallback: exact 4-round masked radix over the register keys ----
    uint32_t prefix = 0;
    int need0 = topK;
    int count_gt_total = 0;
    int shift = 24;
    for (int r = 0; r < 4; ++r) {
      for (int i = tx; i < kTopkSbRadixBins; i += kBlockSize)
        s_hist_fine[i] = 0;
      __syncthreads();
      // Round 0 shifts by 32 (shift+8 on a uint32) which is UB in C++; the
      // intent is "keep everything" (prefix_high == 0). Guard it explicitly.
      const uint32_t prefix_high = (shift == 24) ? 0u : (prefix >> (shift + 8));
      for (int i = 0; i < kPerThread; ++i) {
        if (shift == 24 || (key[i] >> (shift + 8)) == prefix_high)
          atomicAdd(&s_hist_fine[(key[i] >> shift) & 0xFF], 1u);
      }
      __syncthreads();
      topk_sb_detect<4>(s_hist_fine, static_cast<uint32_t>(need0),
                        &s_threshold_bin, &s_count_gt);
      __syncthreads();
      const uint32_t b = static_cast<uint32_t>(s_threshold_bin);
      const int cgt = s_count_gt;
      count_gt_total += cgt;
      need0 -= cgt;
      prefix |= (b << shift);
      if (need0 == 0) {
        prefix |= (1u << shift) - 1u;
        break;
      }
      shift -= 8;
    }
    // collect from registers.
    const uint32_t pivot = prefix;
    if (tx == 0) {
      s_out_counter = 0;
      s_tie_counter = 0;
    }
    __syncthreads();
    if (canVec) {
#pragma unroll
      for (int v = 0; v < kVecsPerThread; ++v) {
        const int base = (tx + v * kBlockSize) * 4;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const uint32_t k = key[v * 4 + e];
          if (k > pivot) {
            const int pos = atomicAdd(&s_out_counter, 1);
            outIndices[rowOut + pos] = base + e;
          } else if (k == pivot) {
            const int pos = atomicAdd(&s_tie_counter, 1);
            if (pos < need0) outIndices[rowOut + count_gt_total + pos] = base + e;
          }
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < kPerThread; ++i) {
        const uint32_t k = key[i];
        if (k > pivot) {
          const int pos = atomicAdd(&s_out_counter, 1);
          outIndices[rowOut + pos] = tx + i * kBlockSize;
        } else if (k == pivot) {
          const int pos = atomicAdd(&s_tie_counter, 1);
          if (pos < need0) outIndices[rowOut + count_gt_total + pos] = tx + i * kBlockSize;
        }
      }
    }
    return;
  }

  // ---- fast path: emit bin>tb, stage bin==tb candidates, refine ----
  const auto idx_of = [&](int i) -> int {
    if (canVec) return (tx + (i >> 2) * kBlockSize) * 4 + (i & 3);
    return tx + i * kBlockSize;
  };

  // Reserve contiguous output runs with one shared atomic per warp.
  int my_gt = 0, my_stage = 0;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    const uint32_t cb = key[i] >> kTopkSbCoarseShift;
    if (cb > tb) ++my_gt;
    else if (cb == tb) ++my_stage;
  }
  // Conditional per-thread ticket: only threads that actually hold >tb / ==tb
  // keys touch a counter (probe: 440 GB/s vs 396 for warp-aggregated tickets,
  // whose 7-shuffle scan runs on all 1024 threads even when my_gt==my_stage==0).
  uint32_t gt_base = 0u, st_base = 0u;
  if (my_gt) gt_base = static_cast<uint32_t>(atomicAdd(&s_out_counter, my_gt));
  if (my_stage)
    st_base = static_cast<uint32_t>(atomicAdd(&s_stage_counter, my_stage));

  for (int i = tx; i < kTopkSbRadixBins; i += kBlockSize)
    s_hist_fine[i] = 0;
  __syncthreads();
  int gi = 0, si = 0;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    const uint32_t cb = key[i] >> kTopkSbCoarseShift;
    if (cb > tb) {
      outIndices[rowOut + gt_base + (gi++)] = idx_of(i);
    } else if (cb == tb) {
      const uint32_t sp = st_base + (si++);
      s_stage_key[0][sp] = key[i];
      s_stage_idx[0][sp] = idx_of(i);
      atomicAdd(&s_hist_fine[(key[i] >> 16) & 0xFF], 1u);
    }
  }
  __syncthreads();

  // Refine the staged candidates with 3 x 8-bit masked-radix passes
  // (bits [23:16], [15:8], [7:0]); the coarse top-10 bits are already fixed.
  // Each pass detects the fine threshold bin, emits everything above it, and
  // re-stages the ==bin candidates for the next (lower 8-bit) pass. On the
  // last pass the ==bin ties are filled with a decrementing counter.
  int src = 0, dst = 1;
  int n_src = n_tb;
#pragma unroll 3
  for (int r = 0; r < 3; ++r) {
    if (need <= 0) break;
    topk_sb_detect<4>(s_hist_fine, static_cast<uint32_t>(need),
                      &s_threshold_bin, &s_count_gt);
    __syncthreads();
    const uint32_t sub_tb = static_cast<uint32_t>(s_threshold_bin);
    const int sub_cgt = s_count_gt;
    need -= sub_cgt;
    const int shift_off = 16 - r * 8;
    if (tx == 0) {
      s_stage_counter = 0;
      s_last_remain = need;
    }
    // Rebuild s_hist_fine from scratch for this round's re-stage so the next
    // round's detect reads a clean histogram.
    for (int i = tx; i < kTopkSbRadixBins; i += kBlockSize)
      s_hist_fine[i] = 0;
    __syncthreads();

    // Count out / re-stage / tie elements, reserve runs with warp tickets.
    int my_out_r = 0, my_st_r = 0, my_tie_r = 0;
    for (int i = tx; i < n_src; i += kBlockSize) {
      const uint32_t k = s_stage_key[src][i];
      const uint32_t b = (k >> shift_off) & 0xFF;
      if (b > sub_tb) ++my_out_r;
      else if (b == sub_tb) {
        if (r == 2) ++my_tie_r;
        else ++my_st_r;
      }
    }
    uint32_t out_base = 0u, next_base = 0u;
    if (my_out_r) out_base = static_cast<uint32_t>(atomicAdd(&s_out_counter, my_out_r));
    if (my_st_r) next_base = static_cast<uint32_t>(atomicAdd(&s_stage_counter, my_st_r));
    int oi = 0, ni = 0;
    for (int i = tx; i < n_src; i += kBlockSize) {
      const uint32_t k = s_stage_key[src][i];
      const uint32_t b = (k >> shift_off) & 0xFF;
      if (b > sub_tb) {
        outIndices[rowOut + out_base + (oi++)] = s_stage_idx[src][i];
      } else if (b == sub_tb) {
        if (r == 2) {
          const int pos = atomicAdd(&s_last_remain, -1);  // remaining ties
          const int w = topK - pos;
          if (pos > 0 && w >= 0 && w < topK)
            outIndices[rowOut + w] = s_stage_idx[src][i];
        } else {
          const uint32_t np = next_base + (ni++);
          s_stage_key[dst][np] = k;
          s_stage_idx[dst][np] = s_stage_idx[src][i];
          atomicAdd(&s_hist_fine[(k >> (shift_off - 8)) & 0xFF], 1u);
        }
      }
    }
    __syncthreads();
    n_src = s_stage_counter;
    src ^= 1;
    dst ^= 1;
  }
}

// =============================================================================
// Two-kernel split fast path for topK >= 8 (rows up to kTopkSbCap).
// Ported verbatim from sampler_prefill.cu. See the per-kernel comments.
// =============================================================================
constexpr int kTopkSplitCap = 2048;    // max staged ==tb candidates per row
constexpr int kTopkSplitK2Max = 256;   // K2 smem direct rank handles nTb<=256
constexpr int kTopkSplitRowsPerBlock = 16;  // warps per K2 block (block=1024)
// Dispatch gate: split beats single-block only for small topK on wide rows
// with enough rows (38-shape C500 bench). topK>32 / cols<16384 regress ~2x
// (==tb staged set exceeds K2 cap -> K3 re-read); tiny grids make the extra
// launches dominate.
constexpr int kTopkSplitTopKMin = 8;
constexpr int kTopkSplitTopKMax = 32;
constexpr int kTopkSplitColsMin = 16384;
constexpr int kTopkSplitRowsMin = 32;

template <int kBlockSize, int kVecsPerThread>
__global__ __launch_bounds__(kBlockSize) void topkEmitKernel(
    const float* __restrict__ logits, const int* __restrict__ seqLens,
    const int* __restrict__ rowStarts, const int* __restrict__ rowEnds,
    int* __restrict__ outIndices, uint32_t* __restrict__ stageKey,
    int* __restrict__ stageIdx, int* __restrict__ rowG, int* __restrict__ rowN,
    int numRows, int stride0, int stride1, int topK, int isDecode, int next_n,
    int seqLensIs2D) {
  constexpr int kPerThread = kVecsPerThread * 4;
  __shared__ uint32_t s_hist_coarse[kTopkSbCoarseBins];
  __shared__ int s_threshold_bin, s_count_gt, s_out_counter, s_stage_counter;
  const int tx = threadIdx.x;
  const int rowIdx = blockIdx.x;
  if (rowIdx >= numRows) return;

  int rowStart, rowEnd;
  if (isDecode) {
    const int batch = rowIdx / next_n;
    const int ni = rowIdx % next_n;
    const int seq_len = seqLensIs2D ? seqLens[rowIdx] : seqLens[batch];
    rowStart = 0;
    rowEnd =
        seqLensIs2D ? max(0, seq_len) : max(0, seq_len - next_n + ni + 1);
  } else {
    rowStart = rowStarts[rowIdx];
    rowEnd = rowEnds[rowIdx];
  }
  const int rowLen = rowEnd - rowStart;

  const int64_t rowOut = static_cast<int64_t>(rowIdx) * topK;
  if (rowLen <= topK) {  // shortcut: emit everything, pad -1
    for (int i = tx; i < rowLen; i += kBlockSize) outIndices[rowOut + i] = i;
    for (int i = rowLen + tx; i < topK; i += kBlockSize)
      outIndices[rowOut + i] = -1;
    if (tx == 0) { rowG[rowIdx] = topK; rowN[rowIdx] = 0; }
    return;
  }
  if (rowLen > kTopkSbCap) {  // defensive (host guarantees <= cap)
    if (tx == 0) { rowG[rowIdx] = topK; rowN[rowIdx] = 0; }
    return;
  }

  const float* rowPtr = logits + (int64_t)rowIdx * stride0 + rowStart;
  const bool canVec = (stride1 == 1) &&
                      ((reinterpret_cast<uintptr_t>(rowPtr) & 0xFu) == 0);
  const float kNegInf = __int_as_float(0xFF800000);
  if (tx == 0) { s_out_counter = 0; s_stage_counter = 0; }
  for (int i = tx; i < kTopkSbCoarseBins; i += kBlockSize)
    s_hist_coarse[i] = 0;
  __syncthreads();
  uint32_t key[kPerThread];
  if (canVec) {
#pragma unroll
    for (int v = 0; v < kVecsPerThread; ++v) {
      const int base = (tx + v * kBlockSize) * 4;
      float4 f;
      if (base + 3 < rowLen) {
        f = *reinterpret_cast<const float4*>(rowPtr + base);
      } else {
        const float* p = rowPtr + base;
        f.x = (base + 0 < rowLen) ? p[0] : kNegInf;
        f.y = (base + 1 < rowLen) ? p[1] : kNegInf;
        f.z = (base + 2 < rowLen) ? p[2] : kNegInf;
        f.w = (base + 3 < rowLen) ? p[3] : kNegInf;
      }
      const uint32_t k0 = topk_sb_key(f.x), k1 = topk_sb_key(f.y);
      const uint32_t k2 = topk_sb_key(f.z), k3 = topk_sb_key(f.w);
      key[v * 4 + 0] = k0; key[v * 4 + 1] = k1;
      key[v * 4 + 2] = k2; key[v * 4 + 3] = k3;
      atomicAdd(&s_hist_coarse[k0 >> kTopkSbCoarseShift], 1u);
      atomicAdd(&s_hist_coarse[k1 >> kTopkSbCoarseShift], 1u);
      atomicAdd(&s_hist_coarse[k2 >> kTopkSbCoarseShift], 1u);
      atomicAdd(&s_hist_coarse[k3 >> kTopkSbCoarseShift], 1u);
    }
  } else {
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const int idx = tx + i * kBlockSize;
      const uint32_t k = topk_sb_key(
          (idx < rowLen) ? rowPtr[idx * stride1] : kNegInf);
      key[i] = k;
      atomicAdd(&s_hist_coarse[k >> kTopkSbCoarseShift], 1u);
    }
  }
  __syncthreads();

  topk_sb_detect<kTopkSbCoarseBins / 64>(s_hist_coarse,
                                         static_cast<uint32_t>(topK),
                                         &s_threshold_bin, &s_count_gt);
  __syncthreads();
  const uint32_t tb = static_cast<uint32_t>(s_threshold_bin);
  const int cgt = s_count_gt;

  const auto idx_of = [&](int i) -> int {
    if (canVec) return (tx + (i >> 2) * kBlockSize) * 4 + (i & 3);
    return tx + i * kBlockSize;
  };

  // Pass 1: emit bin>tb straight to outIndices. A dedicated emit loop then a
  // dedicated stage loop keeps key[] liveness low; measured 16% cheaper than
  // the interleaved single-pass staging.
  int my_gt = 0;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i)
    if ((key[i] >> kTopkSbCoarseShift) > tb) ++my_gt;
  uint32_t gt_base = 0u;
  if (my_gt) gt_base = static_cast<uint32_t>(atomicAdd(&s_out_counter, my_gt));
  int gi = 0;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i)
    if ((key[i] >> kTopkSbCoarseShift) > tb)
      outIndices[rowOut + gt_base + (gi++)] = idx_of(i);

  // Pass 2: stage bin==tb candidates (key, idx) to global scratch.
  int my_eq = 0;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i)
    if ((key[i] >> kTopkSbCoarseShift) == tb) ++my_eq;
  uint32_t eq_base = 0u;
  if (my_eq) eq_base = static_cast<uint32_t>(atomicAdd(&s_stage_counter, my_eq));
  const int64_t rb = static_cast<int64_t>(rowIdx) * kTopkSplitCap;
  int ei = 0;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i)
    if ((key[i] >> kTopkSbCoarseShift) == tb) {
      const int pos = static_cast<int>(eq_base + (ei++));
      if (pos < kTopkSplitCap) {  // never OOB even on overflow rows (K3 redoes)
        stageKey[rb + pos] = key[i];
        stageIdx[rb + pos] = idx_of(i);
      }
    }
  __syncthreads();
  if (tx == 0) {
    rowG[rowIdx] = cgt;
    rowN[rowIdx] = (s_stage_counter > kTopkSplitCap) ? -1 : s_stage_counter;
  }
}

// One warp per row: exact total-order rank of the staged ==tb candidates from
// smem. For each owned element compute rank = #(keys > it) + #(== it, earlier
// slot). Ranks are unique in [0,nTb) so the top `need` can be written without
// conflicts. Keys in the same coarse bin are ranked by their full 32-bit key,
// so the chosen subset is exactly the largest-`need` values; == keys are equal
// floats, so any subset is value-correct.
__global__ void topkRefineKernel(const uint32_t* __restrict__ stageKey,
                                 const int* __restrict__ stageIdx,
                                 const int* __restrict__ rowG,
                                 const int* __restrict__ rowN,
                                 int* __restrict__ outIndices, int numRows,
                                 int topK) {
  __shared__ uint32_t s_key[kTopkSplitRowsPerBlock][kTopkSplitK2Max];
  __shared__ int s_idx[kTopkSplitRowsPerBlock][kTopkSplitK2Max];
  const int lane = threadIdx.x & 63;
  const int warp = threadIdx.x >> 6;
  const int rowIdx = blockIdx.x * kTopkSplitRowsPerBlock + warp;
  if (rowIdx >= numRows) return;
  const int nTb = rowN[rowIdx];
  if (nTb <= 0 || nTb > kTopkSplitK2Max) return;  // overflow rows -> K3
  const int need = topK - rowG[rowIdx];
  if (need <= 0) return;
  const int64_t rb = static_cast<int64_t>(rowIdx) * kTopkSplitCap;
  for (int i = lane; i < nTb; i += 64) {
    s_key[warp][i] = stageKey[rb + i];
    s_idx[warp][i] = stageIdx[rb + i];
  }
  __syncwarp();  // warp-uniform branch; cross-lane smem visibility
  const int baseSlot = static_cast<int>(
      static_cast<int64_t>(rowIdx) * topK + rowG[rowIdx]);
  const int m = (nTb + 63) >> 6;
  for (int j = 0; j < m; ++j) {
    const int pos = lane + j * 64;
    if (pos >= nTb) break;
    const uint32_t kp = s_key[warp][pos];
    const int idxv = s_idx[warp][pos];
    uint32_t cnt = 0;
    for (int q = 0; q < nTb; ++q) {
      const uint32_t kt = s_key[warp][q];
      cnt += (kt > kp) ? 1u : ((kt == kp && q < pos) ? 1u : 0u);
    }
    if (static_cast<int>(cnt) < need) outIndices[baseSlot + cnt] = idxv;
  }
}

// Rare overflow fallback: re-read the row and run the exact 4-round masked
// radix over the register keys, then collect (deployed fallback logic).
template <int kBlockSize, int kVecsPerThread>
__global__ __launch_bounds__(kBlockSize) void topkOverflowKernel(
    const float* __restrict__ logits, const int* __restrict__ seqLens,
    const int* __restrict__ rowStarts, const int* __restrict__ rowEnds,
    int* __restrict__ outIndices, const int* __restrict__ rowN, int numRows,
    int stride0, int stride1, int topK, int isDecode, int next_n,
    int seqLensIs2D) {
  constexpr int kPerThread = kVecsPerThread * 4;
  __shared__ uint32_t s_hist_coarse[kTopkSbCoarseBins];
  __shared__ uint32_t s_hist_fine[kTopkSbRadixBins];
  __shared__ int s_threshold_bin, s_count_gt, s_out_counter, s_tie_counter;
  const int tx = threadIdx.x;
  const int rowIdx = blockIdx.x;
  if (rowIdx >= numRows) return;
  if (rowN[rowIdx] >= 0 && rowN[rowIdx] <= kTopkSplitK2Max) return;
  int rowStart, rowEnd;
  if (isDecode) {
    const int batch = rowIdx / next_n;
    const int ni = rowIdx % next_n;
    const int seq_len = seqLensIs2D ? seqLens[rowIdx] : seqLens[batch];
    rowStart = 0;
    rowEnd =
        seqLensIs2D ? max(0, seq_len) : max(0, seq_len - next_n + ni + 1);
  } else {
    rowStart = rowStarts[rowIdx];
    rowEnd = rowEnds[rowIdx];
  }
  const int rowLen = rowEnd - rowStart;
  if (rowLen > kTopkSbCap) return;  // defensive
  const int64_t rowOut = static_cast<int64_t>(rowIdx) * topK;
  const float* rowPtr = logits + (int64_t)rowIdx * stride0 + rowStart;
  const bool canVec = (stride1 == 1) &&
                      ((reinterpret_cast<uintptr_t>(rowPtr) & 0xFu) == 0);
  const float kNegInf = __int_as_float(0xFF800000);
  if (tx == 0) s_out_counter = 0;
  for (int i = tx; i < kTopkSbCoarseBins; i += kBlockSize)
    s_hist_coarse[i] = 0;
  __syncthreads();
  uint32_t key[kPerThread];
  if (canVec) {
#pragma unroll
    for (int v = 0; v < kVecsPerThread; ++v) {
      const int base = (tx + v * kBlockSize) * 4;
      float4 f;
      if (base + 3 < rowLen) {
        f = *reinterpret_cast<const float4*>(rowPtr + base);
      } else {
        const float* p = rowPtr + base;
        f.x = (base + 0 < rowLen) ? p[0] : kNegInf;
        f.y = (base + 1 < rowLen) ? p[1] : kNegInf;
        f.z = (base + 2 < rowLen) ? p[2] : kNegInf;
        f.w = (base + 3 < rowLen) ? p[3] : kNegInf;
      }
      const uint32_t k0 = topk_sb_key(f.x), k1 = topk_sb_key(f.y);
      const uint32_t k2 = topk_sb_key(f.z), k3 = topk_sb_key(f.w);
      key[v * 4 + 0] = k0; key[v * 4 + 1] = k1;
      key[v * 4 + 2] = k2; key[v * 4 + 3] = k3;
    }
  } else {
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const int idx = tx + i * kBlockSize;
      key[i] = topk_sb_key((idx < rowLen) ? rowPtr[idx * stride1] : kNegInf);
    }
  }
  __syncthreads();
  // Exact 4-round masked radix to find the 32-bit pivot, then collect.
  uint32_t prefix = 0;
  int need0 = topK;
  int count_gt_total = 0;
  int shift = 24;
  for (int r = 0; r < 4; ++r) {
    for (int i = tx; i < kTopkSbRadixBins; i += kBlockSize)
      s_hist_fine[i] = 0;
    __syncthreads();
    const uint32_t prefix_high = (shift == 24) ? 0u : (prefix >> (shift + 8));
    for (int i = 0; i < kPerThread; ++i) {
      if (shift == 24 || (key[i] >> (shift + 8)) == prefix_high)
        atomicAdd(&s_hist_fine[(key[i] >> shift) & 0xFF], 1u);
    }
    __syncthreads();
    topk_sb_detect<4>(s_hist_fine, static_cast<uint32_t>(need0),
                      &s_threshold_bin, &s_count_gt);
    __syncthreads();
    const uint32_t b = static_cast<uint32_t>(s_threshold_bin);
    const int cgt = s_count_gt;
    count_gt_total += cgt;
    need0 -= cgt;
    prefix |= (b << shift);
    if (need0 == 0) {
      prefix |= (1u << shift) - 1u;
      break;
    }
    shift -= 8;
  }
  const uint32_t pivot = prefix;
  if (tx == 0) { s_out_counter = 0; s_tie_counter = 0; }
  __syncthreads();
  if (canVec) {
#pragma unroll
    for (int v = 0; v < kVecsPerThread; ++v) {
      const int base = (tx + v * kBlockSize) * 4;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const uint32_t k = key[v * 4 + e];
        if (k > pivot) {
          const int pos = atomicAdd(&s_out_counter, 1);
          outIndices[rowOut + pos] = base + e;
        } else if (k == pivot) {
          const int pos = atomicAdd(&s_tie_counter, 1);
          if (pos < need0)
            outIndices[rowOut + count_gt_total + pos] = base + e;
        }
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const uint32_t k = key[i];
      if (k > pivot) {
        const int pos = atomicAdd(&s_out_counter, 1);
        outIndices[rowOut + pos] = tx + i * kBlockSize;
      } else if (k == pivot) {
        const int pos = atomicAdd(&s_tie_counter, 1);
        if (pos < need0)
          outIndices[rowOut + count_gt_total + pos] = tx + i * kBlockSize;
      }
    }
  }
}

// Launch the three-kernel split on the current stream (decode: isDecode=1;
// prefill: isDecode=0). All three launches share one stream so a surrounding
// CUDA graph captures them atomically.
template <int kBlockSize, int kVecsPerThread>
static void launchTopkSplit(const float* logits, const int* seqLens,
                            const int* rowStarts, const int* rowEnds,
                            int* outIndices, uint32_t* stageKey, int* stageIdx,
                            int* rowG, int* rowN, int numRows, int stride0,
                            int stride1, int topK, int isDecode, int next_n,
                            int seqLensIs2D) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::topkEmitKernel<kBlockSize, kVecsPerThread>
      <<<numRows, kBlockSize, 0, stream>>>(
          logits, seqLens, rowStarts, rowEnds, outIndices, stageKey, stageIdx,
          rowG, rowN, numRows, stride0, stride1, topK, isDecode, next_n,
          seqLensIs2D);
  vllm::topkRefineKernel<<<(numRows + 15) / 16, 1024, 0, stream>>>(
      stageKey, stageIdx, rowG, rowN, outIndices, numRows, topK);
  vllm::topkOverflowKernel<kBlockSize, kVecsPerThread>
      <<<numRows, kBlockSize, 0, stream>>>(
          logits, seqLens, rowStarts, rowEnds, outIndices, rowN, numRows,
          stride0, stride1, topK, isDecode, next_n, seqLensIs2D);
}

}  // namespace vllm



namespace vllm {
namespace topk_impl {

constexpr int kRadixBins = 256;      // 8-bit radix
// Round-0 width for the two-pass kernel: the top ELEVEN bits (sign +
// exponent + 2 mantissa bits). A byte-0 bucket spans two binades, so at
// 8-bit granularity a randn row's pivot bucket holds ~2.3% of its length
// (5977 keys at L=262144 for the [2,8) crossing) and only the 49152B pool
// — one block per SM — can collect that in two passes. 11-bit bins
// quarter each binade instead: the same row's pivot bucket is ~0.56% of
// L (~1460 keys at 262144, worst measured ~1814 candidates including
// cnt_gt at L=262144 K=1024), so the 20480B pool co-schedules two blocks
// per SM at every row length, and the L=32768 K=1024 / L=16384 K=512
// overflow fallbacks (54.7% / 49.3% of block cycles, probed) vanish.
constexpr int kRadixBinsC = 2048;    // 11-bit round-0 radix (two-pass kernel)
constexpr int kRounds = 4;           // round 0 + 3 refinements = exact key
constexpr int kSmallBlock = 512;     // Path A block for short rows
constexpr int kBigBlock = 1024;      // Path A block for long rows / Path B
constexpr uint32_t kWarpLanes = 64;  // MACA wave64
constexpr uint64_t kFullWarpMask = 0xffffffffffffffffULL;
constexpr uint32_t kChunkCapLarge = 16384;  // C600U (>=128KB opt-in smem)
constexpr uint32_t kChunkCapSmall = 13056;  // C500 (52224B dyn + ~1KB static)
// Candidate buffers (key, index pairs) for the pivot byte's members, which
// rounds 1-3 select from after the compact pass. Path A's buffer must leave
// room for the fully staged row: 52224B dyn + ~11.4KB static <= 64KB.
constexpr uint32_t kCandCapA = 1280;   // Path A / split solo: 10KB

constexpr uint32_t kCandCapS1 = 2560;  // B2 stage 1: 20KB, no dynamic smem

constexpr uint32_t kCandCapC = 6144;      // 49152B dyn + ~9KB static <= 64KB
constexpr uint32_t kCandCapCSmall = 2560; // 20480B dyn: 2 blocks/SM on C500
// At or above this row length, when rows outnumber SMs, dispatch prefers
// Path C over Path A: staged selection costs ~27.5K cycles/row at L=4096
// (probed) versus ~19K for Path C's two light global passes, and beyond
// ~5.2K columns staging confines Path A to one block per SM. Shorter rows
// keep Path A — its single read and 2-3 block co-residency win there.
constexpr uint32_t kStagedHighRowsMinLen = 4096;

constexpr int kProbeRows = 1024;
constexpr int kProbeSlots = 16;
__device__ unsigned long long g_probe[kProbeRows][kProbeSlots];
__device__ int g_probe_on = 0;
__shared__ unsigned long long g_probe_ts;
__shared__ unsigned long long g_probe_acc[kProbeSlots];

__device__ __forceinline__ bool probe_enabled() { return g_probe_on != 0; }

__device__ __forceinline__ void probe_begin(int kind) {
  if (g_probe_on != 0 && threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < kProbeSlots; ++i) g_probe_acc[i] = 0ull;
    const unsigned long long t = clock64();
    g_probe_acc[0] = static_cast<unsigned long long>(kind);
    g_probe_acc[4] = t;
    g_probe_ts = t;
  }
}

__device__ __forceinline__ void probe_mark(int slot) {
  if (g_probe_on != 0 && threadIdx.x == 0) {
    const unsigned long long t = clock64();
    g_probe_acc[slot] += t - g_probe_ts;
    g_probe_ts = t;
  }
}

__device__ __forceinline__ void probe_count(int slot) {
  if (g_probe_on != 0 && threadIdx.x == 0) ++g_probe_acc[slot];
}

__device__ __forceinline__ void probe_end() {
  // Grids beyond kProbeRows (the barrier-free split's stage 1 launches
  // rows x G blocks) must not write past the record buffer; extra blocks
  // simply go unrecorded.
  if (g_probe_on != 0 && threadIdx.x == 0 &&
      blockIdx.x < static_cast<unsigned>(kProbeRows)) {
    g_probe_acc[2] = clock64() - g_probe_acc[4];
    unsigned long long* prow = g_probe[blockIdx.x];
#pragma unroll
    for (int i = 0; i < kProbeSlots; ++i) prow[i] = g_probe_acc[i];
  }
}

// Order-preserving float -> uint32 map: larger value => larger key.
// (-0.0 maps below +0.0; bit-equal values stay bit-equal.)
__device__ __forceinline__ uint32_t ordered_key(float x) {
  const uint32_t b = __float_as_uint(x);
  return (b & 0x80000000u) ? ~b : (b ^ 0x80000000u);
}

template <bool kCrossCTA>
__device__ __forceinline__ void scan_pick_threshold(uint32_t* hist,
                                                    volatile uint32_t* scan,
                                                    uint32_t shift) {
  const uint32_t lane = threadIdx.x;  // caller guarantees lane < 64
  const uint32_t remaining = scan[1];
  const uint32_t b0 = lane * 4;
  uint32_t l0, l1, l2, l3;
  if constexpr (kCrossCTA) {
    l0 = atomicAdd(&hist[b0 + 0], 0u);
    l1 = atomicAdd(&hist[b0 + 1], 0u);
    l2 = atomicAdd(&hist[b0 + 2], 0u);
    l3 = atomicAdd(&hist[b0 + 3], 0u);
  } else {
    l0 = hist[b0 + 0];
    l1 = hist[b0 + 1];
    l2 = hist[b0 + 2];
    l3 = hist[b0 + 3];
  }
  const uint32_t lsum = l0 + l1 + l2 + l3;
  uint32_t incl = lsum;
#pragma unroll
  for (int off = 1; off < 64; off <<= 1) {
    const uint32_t up = __shfl_up_sync(kFullWarpMask, incl, off);
    if (lane >= (uint32_t)off) incl += up;
  }
  const uint32_t total = __shfl_sync(kFullWarpMask, incl, 63);
  const uint32_t s0 = total - (incl - lsum);  // count(bins >= b0)
  const uint32_t s1 = s0 - l0;                // count(bins >= b0+1)
  const uint32_t s2 = s1 - l1;                // count(bins >= b0+2)
  const uint32_t s3 = s2 - l2;                // count(bins >= b0+3)
  if (s3 >= remaining && s3 - l3 < remaining) {
    scan[0] |= (b0 + 3) << shift;
    scan[1] = remaining - (s3 - l3);
  } else if (s2 >= remaining && s3 < remaining) {
    scan[0] |= (b0 + 2) << shift;
    scan[1] = remaining - s3;
  } else if (s1 >= remaining && s2 < remaining) {
    scan[0] |= (b0 + 1) << shift;
    scan[1] = remaining - s2;
  } else if (s0 >= remaining && s1 < remaining) {
    scan[0] |= (b0 + 0) << shift;
    scan[1] = remaining - s1;
  }
}

__device__ __forceinline__ void scan_pick_threshold_2048(
    uint32_t* hist, volatile uint32_t* scan, uint32_t shift) {
  const uint32_t lane = threadIdx.x;  // caller guarantees lane < 64
  const uint32_t remaining = scan[1];
  const uint32_t b0 = lane * 32;
  uint32_t lsum = 0;
#pragma unroll
  for (int j = 0; j < 32; ++j) lsum += hist[b0 + j];
  uint32_t incl = lsum;
#pragma unroll
  for (int off = 1; off < 64; off <<= 1) {
    const uint32_t up = __shfl_up_sync(kFullWarpMask, incl, off);
    if (lane >= (uint32_t)off) incl += up;
  }
  const uint32_t total = __shfl_sync(kFullWarpMask, incl, 63);
  const uint32_t s_ge = total - (incl - lsum);  // count(bins >= b0)
  const uint32_t s_above =
      (lane == 63) ? 0u : __shfl_sync(kFullWarpMask, s_ge, lane + 1);
  if (s_ge >= remaining && s_above < remaining) {
    // count(bins >= b0+j) = s_above + suffix(j); walk from the top.
    uint32_t run = 0;  // suffix(j): hist[b0+j] .. hist[b0+31]
#pragma unroll
    for (int j = 31; j >= 0; --j) {
      const uint32_t hj = hist[b0 + j];
      run += hj;
      const uint32_t cge = s_above + run;  // count(bins >= b0+j)
      const uint32_t cge_next = cge - hj;  // count(bins >= b0+j+1)
      if (cge >= remaining && cge_next < remaining) {
        scan[0] |= (b0 + (uint32_t)j) << shift;
        scan[1] = remaining - cge_next;
      }
    }
  }
}

__device__ __forceinline__ void emit_or_collect(
    uint32_t k, int i, uint32_t mask, uint32_t prefix, uint32_t cnt_gt,
    uint32_t cand_cap, uint32_t* __restrict__ cnt, int* __restrict__ out_row,
    uint32_t* __restrict__ cand_key, int* __restrict__ cand_idx) {
  const uint32_t km = k & mask;
  if (km > prefix) {
    const uint32_t pos = atomicAdd(&cnt[0], 1u);
    if (pos < cnt_gt) out_row[pos] = i;
  } else if (km == prefix) {
    const uint32_t pos = atomicAdd(&cnt[1], 1u);
    if (pos < cand_cap) {
      cand_key[pos] = k;
      cand_idx[pos] = i;
    }
  }
}

// Overflow-fallback round element handling: histogram survivors of the
// current prefix (multi-pass variant for degenerate distributions).
__device__ __forceinline__ void hist_round_el(uint32_t k, uint32_t prefix,
                                              uint32_t mask, uint32_t shift,
                                              uint32_t* __restrict__ hist) {
  if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 0xFF], 1u);
}

// Overflow-fallback final element handling.
__device__ __forceinline__ void emit_final_el(uint32_t k, int i,
                                              uint32_t pivot,
                                              uint32_t total_gt, int top_k,
                                              uint32_t* __restrict__ cnt,
                                              int* __restrict__ out_row) {
  if (k > pivot) {
    const uint32_t pos = atomicAdd(&cnt[0], 1u);
    if (pos < total_gt) out_row[pos] = i;
  } else if (k == pivot) {
    const uint32_t pos = total_gt + atomicAdd(&cnt[1], 1u);
    if (pos < (uint32_t)top_k) out_row[pos] = i;
  }
}

template <int BS>
__device__ __forceinline__ void tail_select(
    uint32_t tx, const uint32_t* __restrict__ cand_key,
    const int* __restrict__ cand_idx, uint32_t n_cand, uint32_t cnt_hi,
    int top_k, int* __restrict__ out_row, uint32_t* __restrict__ hist,
    uint32_t* __restrict__ scan, uint32_t* __restrict__ cnt, uint32_t p0) {
 
  if (tx == 0) {
    cnt[2] = 0;  // survivor cursor: key > pivot
    cnt[3] = 0;  // survivor cursor: key == pivot
  }
  __syncthreads();
  for (uint32_t p = p0; p < 32; p += 8) {
    const uint32_t shift = (p >= 24) ? 0u : (24 - p);
    const uint32_t mask = ~0u << (32 - p);
    const uint32_t prefix = scan[0];
    for (uint32_t i = tx; i < (uint32_t)kRadixBins; i += BS) hist[i] = 0;
    __syncthreads();
    for (uint32_t i = tx; i < n_cand; i += BS) {
      const uint32_t k = cand_key[i];
      if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 0xFF], 1u);
    }
    __threadfence_block();  // flush deferred histogram reductions
    __syncthreads();
    if (tx < kWarpLanes) scan_pick_threshold<false>(hist, scan, shift);
    __syncthreads();
  }
  const uint32_t pivot = scan[0];
  const uint32_t remaining = scan[1];
  const uint32_t total_gt = (uint32_t)top_k - remaining;
  for (uint32_t i = tx; i < n_cand; i += BS) {
    const uint32_t k = cand_key[i];
    if (k > pivot) {
      const uint32_t pos = cnt_hi + atomicAdd(&cnt[2], 1u);
      if (pos < total_gt) out_row[pos] = cand_idx[i];
    } else if (k == pivot) {
      const uint32_t pos = total_gt + atomicAdd(&cnt[3], 1u);
      if (pos < (uint32_t)top_k) out_row[pos] = cand_idx[i];
    }
  }
}

template <int BS>
__device__ void select_staged(const uint32_t* __restrict__ smem_keys,
                              uint32_t row_len, int top_k, int* out_row,
                              uint32_t* __restrict__ hist,
                              uint32_t* __restrict__ scan,
                              uint32_t* __restrict__ cnt,
                              uint32_t* __restrict__ cand_key,
                              int* __restrict__ cand_idx) {
  const uint32_t tx = threadIdx.x;
  // Forensic capture (probe mode only): the caller's decision inputs.
  // Entry state is stable — every writer before this point is separated by
  // a block barrier.
  if (g_probe_on != 0 && tx == 0) {
    const volatile uint32_t* vscan = scan;
    const volatile uint32_t* vhist = hist;
    const uint32_t b = vscan[0] >> 24;
    g_probe_acc[15] =
        (static_cast<unsigned long long>(vscan[0]) << 32) | vscan[1];
    g_probe_acc[5] = (static_cast<unsigned long long>(vhist[b]) << 32) | b;
  }

  for (uint32_t r = 1; r < (uint32_t)kRounds; ++r) {
    const uint32_t shift = 24 - r * 8;
    const uint32_t mask = ~0u << (32 - r * 8);  // survivors match this prefix
    const uint32_t prefix = scan[0];
    for (uint32_t i = tx; i < (uint32_t)kRadixBins; i += BS) hist[i] = 0;
    __syncthreads();
    for (uint32_t i = tx; i < row_len; i += BS) {
      const uint32_t k = smem_keys[i];
      if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 0xFF], 1u);
    }
    __threadfence_block();  // flush deferred histogram reductions
    __syncthreads();
    probe_mark(r == 1 ? 11 : 12);  // staged histogram, rounds 1-3
    if (tx < kWarpLanes) scan_pick_threshold<false>(hist, scan, shift);
    // Forensic capture (probe mode only), read while every other warp sits
    // at the barrier below: the fresh (pivot, remaining) pair plus this
    // pass's own recount of the pivot bin.
    if (g_probe_on != 0 && tx == 0 && (r == 1 || r == kRounds - 1)) {
      const volatile uint32_t* vscan = scan;
      const volatile uint32_t* vhist = hist;
      const uint32_t b = (vscan[0] >> shift) & 0xFF;
      const unsigned long long pair =
          (static_cast<unsigned long long>(vscan[0]) << 32) | vscan[1];
      const unsigned long long hb =
          (static_cast<unsigned long long>(vhist[b]) << 32) | b;
      if (r == 1) {
        g_probe_acc[6] = pair;
        g_probe_acc[9] = hb;
      } else {
        g_probe_acc[10] = pair;
      }
    }
    __syncthreads();
    probe_mark(7);  // threshold scan
    // Survivor count at this granularity, block-uniform via thread 0:
    // strictly-greater keys plus the pivot bin. Plain read — the producing
    // loop's fence + the two barriers above committed the bin.
    if (tx == 0) cnt[2] = hist[(scan[0] >> shift) & 0xFF];
    __syncthreads();
    const uint32_t cnt_gt = (uint32_t)top_k - scan[1];
    if (r < (uint32_t)kRounds - 1 && cnt_gt + cnt[2] <= kCandCapA) {

      const uint32_t mask_c = ~0u << (24 - r * 8);
      const uint32_t prefix_c = scan[0];
      if (tx == 0) {
        cnt[0] = 0;  // emit cursor: strictly greater
        cnt[1] = 0;  // candidate cursor
      }
      __syncthreads();
      for (uint32_t i = tx; i < row_len; i += BS)
        emit_or_collect(smem_keys[i], (int)i, mask_c, prefix_c, cnt_gt,
                        kCandCapA, cnt, out_row, cand_key, cand_idx);
      __syncthreads();
      if (g_probe_on != 0 && tx == 0) {
        const volatile uint32_t* vcnt = cnt;
        g_probe_acc[14] =
            (static_cast<unsigned long long>(vcnt[0]) << 32) | vcnt[1];
      }
      tail_select<BS>(tx, cand_key, cand_idx, cnt[1], cnt_gt, top_k, out_row,
                      hist, scan, cnt, 8 * (r + 1));
      if (probe_enabled()) __syncthreads();
      probe_mark(8);  // collect + tail emit
      return;
    }
  }

  // Survivor set still overflows the buffer after round 3: the pivot is now
  // the exact 32-bit key, so one final staged pass emits everything.
  const uint32_t pivot = scan[0];
  const uint32_t remaining = scan[1];
  const uint32_t total_gt = (uint32_t)top_k - remaining;
  for (uint32_t i = tx; i < row_len; i += BS) {
    const uint32_t k = smem_keys[i];
    if (k > pivot) {
      const uint32_t pos = atomicAdd(&cnt[0], 1u);
      if (pos < total_gt) out_row[pos] = (int)i;
    } else if (k == pivot) {
      const uint32_t pos = total_gt + atomicAdd(&cnt[1], 1u);
      if (pos < (uint32_t)top_k) out_row[pos] = (int)i;
    }
  }
  if (probe_enabled()) __syncthreads();
  // Forensic capture (probe mode only): emit cursors after the probe-only
  // barrier above, i.e. how many entries this block actually wrote.
  if (g_probe_on != 0 && tx == 0) {
    const volatile uint32_t* vcnt = cnt;
    g_probe_acc[14] =
        (static_cast<unsigned long long>(vcnt[0]) << 32) | vcnt[1];
  }
  probe_mark(8);  // emit
}

template <int BS>
__device__ void one_shot_body(const float* __restrict__ logits, int row_idx,
                              int row_start, int row_end, int stride0,
                              int stride1, int top_k, int* out_indices,
                              uint32_t* __restrict__ smem_keys,
                              uint32_t* __restrict__ hist,
                              uint32_t* __restrict__ scan,
                              uint32_t* __restrict__ cnt) {
  __shared__ uint32_t s_cand_key[kCandCapA];
  __shared__ int s_cand_idx[kCandCapA];
  const uint32_t tx = threadIdx.x;
  int* out_row = out_indices + static_cast<int64_t>(row_idx) * top_k;
  const int row_len = row_end - row_start;

  if (row_len <= top_k) {
    for (int i = (int)tx; i < row_len; i += BS) out_row[i] = i;
    for (int i = row_len + (int)tx; i < top_k; i += BS) out_row[i] = -1;
    if (probe_enabled()) __syncthreads();
    probe_mark(10);  // short row
    probe_count(14);
    return;
  }

  const float* seg = logits + static_cast<int64_t>(row_idx) * stride0 +
                     static_cast<int64_t>(row_start) * stride1;
  // Fused pass 1: stage the row as ordered keys while histogramming the top
  // byte in the same sweep.
  for (uint32_t i = tx; i < (uint32_t)kRadixBins; i += BS) hist[i] = 0;
  if (tx == 0) {
    scan[0] = 0;
    scan[1] = (uint32_t)top_k;
    cnt[0] = 0;  // cursor: keys above the pivot byte
    cnt[1] = 0;  // cursor: pivot-byte candidates
  }
  __syncthreads();
  if (stride1 == 1 &&
      (reinterpret_cast<size_t>(seg) & 15u) == 0u) {
    const uint32_t n4 = (uint32_t)row_len >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(seg);
    // Two independent float4 loads in flight: one block per SM is
    // latency-bound while staging, so a second outstanding quad hides the
    // first quad's key stores + histogram atomics.
    uint32_t v = tx;
    for (; v + BS < n4; v += 2 * BS) {
      const float4 f0 = in4[v];
      const float4 f1 = in4[v + BS];
      const uint32_t b = v * 4;
      const uint32_t e = b + BS * 4;
      const uint32_t k0 = ordered_key(f0.x);
      const uint32_t k1 = ordered_key(f0.y);
      const uint32_t k2 = ordered_key(f0.z);
      const uint32_t k3 = ordered_key(f0.w);
      const uint32_t k4 = ordered_key(f1.x);
      const uint32_t k5 = ordered_key(f1.y);
      const uint32_t k6 = ordered_key(f1.z);
      const uint32_t k7 = ordered_key(f1.w);
      smem_keys[b + 0] = k0;
      smem_keys[b + 1] = k1;
      smem_keys[b + 2] = k2;
      smem_keys[b + 3] = k3;
      smem_keys[e + 0] = k4;
      smem_keys[e + 1] = k5;
      smem_keys[e + 2] = k6;
      smem_keys[e + 3] = k7;
      atomicAdd(&hist[k0 >> 24], 1u);
      atomicAdd(&hist[k1 >> 24], 1u);
      atomicAdd(&hist[k2 >> 24], 1u);
      atomicAdd(&hist[k3 >> 24], 1u);
      atomicAdd(&hist[k4 >> 24], 1u);
      atomicAdd(&hist[k5 >> 24], 1u);
      atomicAdd(&hist[k6 >> 24], 1u);
      atomicAdd(&hist[k7 >> 24], 1u);
    }
    for (; v < n4; v += BS) {
      const float4 f = in4[v];
      const uint32_t k0 = ordered_key(f.x);
      const uint32_t k1 = ordered_key(f.y);
      const uint32_t k2 = ordered_key(f.z);
      const uint32_t k3 = ordered_key(f.w);
      smem_keys[v * 4 + 0] = k0;
      smem_keys[v * 4 + 1] = k1;
      smem_keys[v * 4 + 2] = k2;
      smem_keys[v * 4 + 3] = k3;
      atomicAdd(&hist[k0 >> 24], 1u);
      atomicAdd(&hist[k1 >> 24], 1u);
      atomicAdd(&hist[k2 >> 24], 1u);
      atomicAdd(&hist[k3 >> 24], 1u);
    }
    for (uint32_t i = (n4 << 2) + tx; i < (uint32_t)row_len; i += BS) {
      const uint32_t k = ordered_key(seg[i]);
      smem_keys[i] = k;
      atomicAdd(&hist[k >> 24], 1u);
    }
  } else {
    for (uint32_t i = tx; i < (uint32_t)row_len; i += BS) {
      const uint32_t k = ordered_key(seg[static_cast<int64_t>(i) * stride1]);
      smem_keys[i] = k;
      atomicAdd(&hist[k >> 24], 1u);
    }
  }
  // The histogram reductions may be deferred; flush them before the barrier
  // so the threshold scan observes the complete histogram.
  __threadfence_block();
  __syncthreads();
  probe_mark(3);  // fused staging + round-0 histogram
  if (tx < kWarpLanes) scan_pick_threshold<false>(hist, scan, 24);
  __syncthreads();
  probe_mark(7);  // round-0 threshold scan

  if (tx == 0) cnt[2] = atomicAdd(&hist[scan[0] >> 24], 0u);
  __syncthreads();
  const uint32_t pivot8 = scan[0] >> 24;
  const uint32_t n_cand = cnt[2];
  if (n_cand > kCandCapA) {
    select_staged<BS>(smem_keys, (uint32_t)row_len, top_k, out_row, hist, scan,
                      cnt, s_cand_key, s_cand_idx);
    probe_mark(12);  // overflow fallback
    return;
  }

  // Compact pass: everything above the pivot byte is guaranteed top-K; the
  // pivot byte's members become the tail candidates.
  const uint32_t cnt_hi = (uint32_t)top_k - scan[1];
  for (uint32_t i = tx; i < (uint32_t)row_len; i += BS) {
    emit_or_collect(smem_keys[i], (int)i, 0xFF000000u, pivot8 << 24, cnt_hi,
                    kCandCapA, cnt, out_row, s_cand_key, s_cand_idx);
  }
  __syncthreads();
  probe_mark(11);  // compact pass (guaranteed-high emit + candidate collect)

  // Tail: exact pivot among the candidates + survivor emit, all threads.
  tail_select<BS>(tx, s_cand_key, s_cand_idx, n_cand, cnt_hi, top_k, out_row,
                  hist, scan, cnt, 8u);
  if (probe_enabled()) __syncthreads();
  probe_mark(8);  // tail selection
}

}  // namespace topk_impl
}  // namespace vllm

// ===========================================================================
// Path A kernels: one block per row, whole row staged in shared memory.
// ===========================================================================
namespace vllm {
namespace topk_impl {

constexpr uint32_t kWideUnrollQuads = 20;

template <int BS>
static __global__ __launch_bounds__(BS) void topk_row_one_shot_decode(
    const float* __restrict__ logits, const int* __restrict__ seq_lens,
    int* __restrict__ out_indices, int stride0, int stride1, int top_k,
    int next_n, int seq_lens_is2d, int num_columns) {
  extern __shared__ uint32_t smem_keys[];
  __shared__ uint32_t s_hist[kRadixBins];
  __shared__ uint32_t s_scan[2];
  __shared__ uint32_t s_cnt[4];
  const int row_idx = blockIdx.x;
  probe_begin(1);
  const int seq_len = seq_lens_is2d ? seq_lens[row_idx]
                                    : seq_lens[row_idx / next_n];
  int row_end = seq_lens_is2d
                    ? max(0, seq_len)
                    : max(0, seq_len - next_n + row_idx % next_n + 1);
  if (row_end > num_columns) row_end = num_columns;
  one_shot_body<BS>(logits, row_idx, 0, row_end, stride0, stride1, top_k,
                    out_indices, smem_keys, s_hist, s_scan, s_cnt);
  probe_end();
}

template <int BS>
static __global__ __launch_bounds__(BS) void topk_row_one_shot_prefill(
    const float* __restrict__ logits, const int* __restrict__ row_starts,
    const int* __restrict__ row_ends, int* __restrict__ out_indices,
    int stride0, int stride1, int top_k, int num_columns) {
  extern __shared__ uint32_t smem_keys[];
  __shared__ uint32_t s_hist[kRadixBins];
  __shared__ uint32_t s_scan[2];
  __shared__ uint32_t s_cnt[4];
  const int row_idx = blockIdx.x;
  probe_begin(1);
  int row_start = row_starts[row_idx];
  int row_end = row_ends[row_idx];
  if (row_start < 0) row_start = 0;
  if (row_start > num_columns) row_start = num_columns;
  if (row_end < row_start) row_end = row_start;
  if (row_end > num_columns) row_end = num_columns;
  one_shot_body<BS>(logits, row_idx, row_start, row_end, stride0, stride1,
                    top_k, out_indices, smem_keys, s_hist, s_scan, s_cnt);
  probe_end();
}

// ===========================================================================
// Path B: persistent split-K over chunks.
// ===========================================================================

struct RowState {
  uint32_t hist[3][kRadixBins];  // rotating per-round global histograms
  int arrival_counter;           // barrier: accumulated arrivals
  int output_counter;            // global ticket base for key > pivot
  int eq_counter;                // global ticket base for key == pivot
};

__device__ __forceinline__ void sk_red_release(int* ptr, int val) {
  __threadfence();
  atomicAdd(ptr, val);
}

__device__ __forceinline__ void sk_wait_ge(int* ptr, int target, int tx) {
  if (tx == 0) {
    volatile int* v = reinterpret_cast<volatile int*>(ptr);
    int spins = 0;
#pragma unroll 1
    while (*v < target) {
      // Watchdog atomic read: bounds staleness in case volatile loads are
      // served from a private cache on this architecture.
      if ((++spins & 1023) == 0 && atomicAdd(ptr, 0) >= target) break;
    }
    __threadfence();
  }
  __syncthreads();
}

// Persistent kernel: grid <= resident capacity; each CTA sweeps flat chunk
// indices c = blockIdx.x, blockIdx.x + gridDim.x, ... Rows shorter than a
// chunk are handled solo by chunk 0 (no cross-CTA state). Longer rows run
// the full split-K barrier protocol; per-row state comes from the zeroed
// workspace.
template <int BS, bool kIsPrefill>
static __global__ __launch_bounds__(BS) void topk_split_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ starts_or_seq,  // prefill: row starts; decode: seq lens
    const int* __restrict__ ends_or_null,   // prefill: row ends
    int* __restrict__ out_indices, int stride0, int stride1, int top_k,
    int next_n, int seq_lens_is2d, int num_columns, int num_rows, uint32_t G,
    uint32_t chunk, RowState* __restrict__ states) {
  extern __shared__ uint32_t smem_keys[];
  __shared__ uint32_t s_hist[kRadixBins];
  __shared__ uint32_t s_scan[2];
  __shared__ uint32_t s_cnt[4];
  __shared__ uint32_t s_scratch[5];
  const int tx = threadIdx.x;
  probe_begin(2);
  const uint32_t total =
      static_cast<uint32_t>(static_cast<int64_t>(num_rows) * G);

  for (uint32_t c = blockIdx.x; c < total; c += gridDim.x) {
    // Barrier at the top protects shared memory from the previous iteration's
    // tail (emit loops read smem_keys without a trailing barrier).
    __syncthreads();
    const int row = static_cast<int>(c / G);
    const uint32_t kin = c % G;
    int row_start = 0;
    int row_end = 0;
    if (kIsPrefill) {
      row_start = starts_or_seq[row];
      row_end = ends_or_null[row];
      if (row_start < 0) row_start = 0;
      if (row_start > num_columns) row_start = num_columns;
      if (row_end < row_start) row_end = row_start;
      if (row_end > num_columns) row_end = num_columns;
    } else {
      const int seq_len =
          seq_lens_is2d ? starts_or_seq[row] : starts_or_seq[row / next_n];
      row_end = seq_lens_is2d
                    ? max(0, seq_len)
                    : max(0, seq_len - next_n + row % next_n + 1);
      if (row_end > num_columns) row_end = num_columns;
    }
    const int row_len = row_end - row_start;
    int* out_row = out_indices + static_cast<int64_t>(row) * top_k;

    if (row_len <= top_k) {
      // Short row: chunk 0 writes [0..len) + (-1) padding, others move on.
      if (kin == 0) {
        for (int i = tx; i < row_len; i += BS) out_row[i] = i;
        for (int i = row_len + tx; i < top_k; i += BS) out_row[i] = -1;
        if (probe_enabled()) __syncthreads();
        probe_mark(10);  // short row
        probe_count(14);
      }
      continue;
    }
    if (static_cast<uint32_t>(row_len) <= chunk) {
      // Whole row fits one chunk: chunk 0 handles it solo (Path A logic,
      // smem_keys is chunk-sized >= row_len).
      if (kin == 0) {
        one_shot_body<BS>(logits, row, row_start, row_end, stride0, stride1,
                          top_k, out_indices, smem_keys, s_hist, s_scan,
                          s_cnt);
        if (probe_enabled()) __syncthreads();
        probe_mark(9);  // solo (whole row in one chunk)
        probe_count(13);
      }
      continue;
    }

    // ---- split-K path ----
    RowState* st = states + row;
    const uint32_t len = static_cast<uint32_t>(row_len);
    const uint32_t my_start = kin * chunk;
    const uint32_t my_end = min(my_start + chunk, len);
    const uint32_t actual = my_start < len ? (my_end - my_start) : 0;
    const float* seg = logits + static_cast<int64_t>(row) * stride0 +
                       static_cast<int64_t>(row_start + my_start) * stride1;

    // Stage the chunk as ordered keys (vectorized when possible).
    if (stride1 == 1 && actual > 0 &&
        (reinterpret_cast<size_t>(seg) & 15u) == 0u) {
      const uint32_t n4 = actual >> 2;
      const float4* in4 = reinterpret_cast<const float4*>(seg);
      // Two independent float4 loads in flight (same rationale as Path A).
      uint32_t v = tx;
      for (; v + BS < n4; v += 2 * BS) {
        const float4 f0 = in4[v];
        const float4 f1 = in4[v + BS];
        const uint32_t b = v * 4;
        const uint32_t e = b + BS * 4;
        smem_keys[b + 0] = ordered_key(f0.x);
        smem_keys[b + 1] = ordered_key(f0.y);
        smem_keys[b + 2] = ordered_key(f0.z);
        smem_keys[b + 3] = ordered_key(f0.w);
        smem_keys[e + 0] = ordered_key(f1.x);
        smem_keys[e + 1] = ordered_key(f1.y);
        smem_keys[e + 2] = ordered_key(f1.z);
        smem_keys[e + 3] = ordered_key(f1.w);
      }
      for (; v < n4; v += BS) {
        const float4 f = in4[v];
        smem_keys[v * 4 + 0] = ordered_key(f.x);
        smem_keys[v * 4 + 1] = ordered_key(f.y);
        smem_keys[v * 4 + 2] = ordered_key(f.z);
        smem_keys[v * 4 + 3] = ordered_key(f.w);
      }
      for (uint32_t i = (n4 << 2) + tx; i < actual; i += BS)
        smem_keys[i] = ordered_key(seg[i]);
    } else {
      for (uint32_t i = tx; i < actual; i += BS)
        smem_keys[i] = ordered_key(seg[static_cast<int64_t>(i) * stride1]);
    }
    if (tx == 0) {
      s_scan[0] = 0;
      s_scan[1] = (uint32_t)top_k;
    }
    __syncthreads();
    probe_mark(3);  // staging

    // 4 rounds: local histogram over survivors -> merge into the row's
    // global histogram -> barrier -> warp-0 scan picks the threshold bin.
    int arrival = 0;
    for (uint32_t r = 0; r < (uint32_t)kRounds; ++r) {
      const uint32_t shift = 24 - r * 8;
      const uint32_t mask = (r == 0) ? 0u : (~0u << (32 - r * 8));
      const uint32_t prefix = s_scan[0];
      const uint32_t remaining = s_scan[1];
      for (uint32_t i = tx; i < (uint32_t)kRadixBins; i += BS) s_hist[i] = 0;
      __syncthreads();
      for (uint32_t i = tx; i < actual; i += BS) {
        const uint32_t k = smem_keys[i];
        if ((k & mask) == prefix) atomicAdd(&s_hist[(k >> shift) & 0xFF], 1u);
      }
      __threadfence_block();  // flush deferred histogram reductions
      __syncthreads();
      probe_mark(r == 0 ? 11 : 12);  // local histogram (round 0 / rounds 1-3)
      uint32_t* cur = st->hist[r % 3];
      uint32_t* nxt = st->hist[(r + 1) % 3];
      // Plain read of the local bin is committed-visible: the producing
      // loop's __threadfence_block() above flushed its deferred reductions.
      // The merge TARGET stays a global atomic (other CTAs add to it too).
      for (uint32_t i = tx; i < (uint32_t)kRadixBins; i += BS) {
        const uint32_t v = s_hist[i];
        if (v > 0) atomicAdd(&cur[i], v);
      }
      if (kin == 0) {
        // Leader pre-zeroes next round's buffer. Safe: that buffer was last
        // touched in round r-2, whose barrier everyone already passed; and
        // the zeroing is fenced before this round's release below.
        for (uint32_t i = tx; i < (uint32_t)kRadixBins; i += BS) nxt[i] = 0;
      }
      if (tx == 0) {
        probe_mark(5);  // global histogram merge
        sk_red_release(&st->arrival_counter, 1);
      }
      arrival += (int)G;
      sk_wait_ge(&st->arrival_counter, arrival, tx);
      probe_mark(6);  // barrier wait
      if (tx < kWarpLanes) scan_pick_threshold<true>(cur, s_scan, shift);
      __syncthreads();
      probe_mark(7);  // threshold scan
    }

    // Emit: key > pivot reserves slots [0, total_gt) via one global atomic
    // per CTA; key == pivot fills [total_gt, top_k) via the eq ticket counter.
    const uint32_t pivot = s_scan[0];
    const uint32_t remaining = s_scan[1];
    const uint32_t total_gt = (uint32_t)top_k - remaining;
    uint32_t my_gt = 0, my_eq = 0;
    for (uint32_t i = tx; i < actual; i += BS) {
      const uint32_t k = smem_keys[i];
      if (k > pivot) ++my_gt;
      else if (k == pivot) ++my_eq;
    }
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
      my_gt += __shfl_down_sync(kFullWarpMask, my_gt, off);
      my_eq += __shfl_down_sync(kFullWarpMask, my_eq, off);
    }
    if (tx == 0) {
      s_scratch[0] = 0;
      s_scratch[1] = 0;
    }
    __syncthreads();
    if ((tx & 63) == 0) {
      if (my_gt > 0) atomicAdd(&s_scratch[0], my_gt);
      if (my_eq > 0) atomicAdd(&s_scratch[1], my_eq);
    }
    __syncthreads();
    if (tx == 0) {
      const uint32_t lg = s_scratch[0];
      const uint32_t le = s_scratch[1];
      s_scratch[1] = 0;  // gt cursor
      s_scratch[3] = 0;  // eq cursor
      s_scratch[2] =
          (lg > 0) ? (uint32_t)atomicAdd(&st->output_counter, (int)lg) : 0u;
      s_scratch[4] =
          (le > 0) ? (uint32_t)atomicAdd(&st->eq_counter, (int)le) : 0u;
    }
    __syncthreads();
    const uint32_t gt_base = s_scratch[2];
    const uint32_t eq_base = s_scratch[4];
    for (uint32_t i = tx; i < actual; i += BS) {
      const uint32_t k = smem_keys[i];
      if (k > pivot) {
        const uint32_t pos = gt_base + atomicAdd(&s_scratch[1], 1u);
        if (pos < (uint32_t)top_k)
          out_row[pos] = static_cast<int>(my_start + i);
      } else if (k == pivot) {
        const uint32_t pos = total_gt + eq_base + atomicAdd(&s_scratch[3], 1u);
        if (pos < (uint32_t)top_k)
          out_row[pos] = static_cast<int>(my_start + i);
      }
    }
    if (probe_enabled()) __syncthreads();
    probe_mark(8);  // emit (count + tickets + writes)
    probe_count(1);
  }
  probe_end();
}

template <int BS>
static __global__ __launch_bounds__(BS) void topk_chunk_local_kernel(
    const float* __restrict__ logits, const int* __restrict__ seq_lens,
    int* __restrict__ ws_idx, uint32_t* __restrict__ ws_key, int stride0,
    int stride1, int top_k, int next_n, int seq_lens_is2d, int num_columns,
    uint32_t G, uint32_t chunk) {
  __shared__ uint32_t s_hist[kRadixBins];
  __shared__ uint32_t s_scan[2];
  __shared__ uint32_t s_cnt[4];
  __shared__ uint32_t s_cand_key[kCandCapS1];
  __shared__ int s_cand_idx[kCandCapS1];
  constexpr uint32_t kR0Shift = 24u;           // NARROW: top byte
  constexpr uint32_t kMaskP2 = ~0u << kR0Shift;  // resolved-top-bits mask
  const uint32_t tx = threadIdx.x;
  probe_begin(5);
  const int row = static_cast<int>(blockIdx.x / G);
  const uint32_t kin = blockIdx.x - static_cast<uint32_t>(row) * G;
  const int seq_len =
      seq_lens_is2d ? seq_lens[row] : seq_lens[row / next_n];
  int row_end = seq_lens_is2d ? max(0, seq_len)
                              : max(0, seq_len - next_n + row % next_n + 1);
  if (row_end > num_columns) row_end = num_columns;
  int* ws_i = ws_idx + (static_cast<int64_t>(row) * G + kin) * top_k;
  uint32_t* ws_k = ws_key + (static_cast<int64_t>(row) * G + kin) * top_k;
  const uint32_t my_start = kin * chunk;
  if (my_start >= static_cast<uint32_t>(row_end)) {
    // Chunk entirely past this row's end: all-sentinel candidates.
    for (int i = (int)tx; i < top_k; i += BS) {
      ws_i[i] = -1;
      ws_k[i] = 0u;
    }
    if (probe_enabled()) __syncthreads();
    probe_mark(10);  // empty chunk (sentinel pad)
    probe_count(14);
    probe_end();
    return;
  }
  const int my_end =
      static_cast<int>(min(my_start + chunk, static_cast<uint32_t>(row_end)));
  const int row_len = my_end - static_cast<int>(my_start);
  const float* seg = logits + static_cast<int64_t>(row) * stride0 +
                     static_cast<int64_t>(my_start) * stride1;
  if (row_len <= top_k) {
    // Short chunk: every position survives in order; pad with sentinels.
    for (int i = (int)tx; i < row_len; i += BS) {
      ws_i[i] = static_cast<int>(my_start) + i;
      ws_k[i] = ordered_key(seg[static_cast<int64_t>(i) * stride1]);
    }
    for (int i = row_len + (int)tx; i < top_k; i += BS) {
      ws_i[i] = -1;
      ws_k[i] = 0u;
    }
    if (probe_enabled()) __syncthreads();
    probe_mark(10);  // short chunk
    probe_count(14);
    probe_end();
    return;
  }
  const bool vec =
      stride1 == 1 && (reinterpret_cast<size_t>(seg) & 15u) == 0u;
  const uint32_t len = static_cast<uint32_t>(row_len);
  int* out_row = ws_i;  // chunk-local top-K, emitted as GLOBAL columns

  // Pass 1: byte-0 histogram straight from global memory — the two-pass
  // kernel's streaming pattern (staging is what throttled the old body).
  for (uint32_t i = tx; i < (uint32_t)kRadixBins; i += BS) s_hist[i] = 0;
  if (tx == 0) {
    s_scan[0] = 0;
    s_scan[1] = (uint32_t)top_k;
    s_cnt[0] = 0;
    s_cnt[1] = 0;
  }
  __syncthreads();
  if (vec) {
    const uint32_t n4 = len >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(seg);
    uint32_t v = tx;
    for (; n4 >= kWideUnrollQuads * BS && v + 3 * BS < n4; v += 4 * BS) {
      const float4 f0 = in4[v];
      const float4 f1 = in4[v + BS];
      const float4 f2 = in4[v + 2 * BS];
      const float4 f3 = in4[v + 3 * BS];
      atomicAdd(&s_hist[ordered_key(f0.x) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f0.y) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f0.z) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f0.w) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.x) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.y) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.z) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.w) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f2.x) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f2.y) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f2.z) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f2.w) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f3.x) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f3.y) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f3.z) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f3.w) >> kR0Shift], 1u);
    }
    for (; v + BS < n4; v += 2 * BS) {
      const float4 f0 = in4[v];
      const float4 f1 = in4[v + BS];
      atomicAdd(&s_hist[ordered_key(f0.x) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f0.y) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f0.z) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f0.w) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.x) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.y) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.z) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.w) >> kR0Shift], 1u);
    }
    for (; v < n4; v += BS) {
      const float4 f = in4[v];
      atomicAdd(&s_hist[ordered_key(f.x) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f.y) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f.z) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f.w) >> kR0Shift], 1u);
    }
    for (uint32_t i = (n4 << 2) + tx; i < len; i += BS)
      atomicAdd(&s_hist[ordered_key(seg[i]) >> kR0Shift], 1u);
  } else {
    for (uint32_t i = tx; i < len; i += BS)
      atomicAdd(
          &s_hist[ordered_key(seg[static_cast<int64_t>(i) * stride1]) >> kR0Shift],
          1u);
  }
  __threadfence_block();  // flush deferred histogram reductions
  __syncthreads();
  probe_mark(3);  // pass 1: byte-0 histogram
  if (tx < kWarpLanes) scan_pick_threshold<false>(s_hist, s_scan, 24);
  __syncthreads();
  probe_mark(7);  // round-0 threshold scan
  const uint32_t pivot_r0 = s_scan[0] >> kR0Shift;
  const uint32_t cnt_hi = (uint32_t)top_k - s_scan[1];
  if (tx == 0) s_cnt[2] = atomicAdd(&s_hist[pivot_r0], 0u);
  __syncthreads();
  const uint32_t n_cand = s_cnt[2];

  if (n_cand > kCandCapS1) {
    // Overflow: the byte-0 pivot bucket is heavier than the candidate
    // buffer. Same refinement ladder as the two-pass NARROW kernel — full
    // re-read rounds, breaking out into one collect pass as soon as the
    // surviving window fits. Only truly degenerate chunks (huge tied-bit
    // plateaus) still run all rounds plus the final emit pass.
    bool broke_out = false;
    for (uint32_t r = 1; r < (uint32_t)kRounds; ++r) {
      const uint32_t p = (32 - kR0Shift) + 8 * (r - 1);
      const uint32_t shift = (p >= 24) ? 0u : (24 - p);
      const uint32_t mask = ~0u << (32 - p);
      const uint32_t prefix = s_scan[0];
      for (uint32_t i = tx; i < (uint32_t)kRadixBins; i += BS) s_hist[i] = 0;
      __syncthreads();
      if (vec) {
        const uint32_t n4 = len >> 2;
        const float4* in4 = reinterpret_cast<const float4*>(seg);
        uint32_t v = tx;
        for (; v + BS < n4; v += 2 * BS) {
          const float4 f0 = in4[v];
          const float4 f1 = in4[v + BS];
          hist_round_el(ordered_key(f0.x), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f0.y), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f0.z), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f0.w), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f1.x), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f1.y), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f1.z), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f1.w), prefix, mask, shift, s_hist);
        }
        for (; v < n4; v += BS) {
          const float4 f = in4[v];
          hist_round_el(ordered_key(f.x), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f.y), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f.z), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f.w), prefix, mask, shift, s_hist);
        }
        for (uint32_t i = (n4 << 2) + tx; i < len; i += BS)
          hist_round_el(ordered_key(seg[i]), prefix, mask, shift, s_hist);
      } else {
        for (uint32_t i = tx; i < len; i += BS)
          hist_round_el(
              ordered_key(seg[static_cast<int64_t>(i) * stride1]), prefix,
              mask, shift, s_hist);
      }
      __threadfence_block();  // flush deferred histogram reductions
      __syncthreads();
      if (tx < kWarpLanes) scan_pick_threshold<false>(s_hist, s_scan, shift);
      __syncthreads();
      // Survivor count at this granularity, block-uniform via thread 0.
      if (tx == 0) s_cnt[2] = s_hist[(s_scan[0] >> shift) & 0xFF];
      __syncthreads();
      const uint32_t cnt_gt = (uint32_t)top_k - s_scan[1];
      if (r < (uint32_t)kRounds - 1 && cnt_gt + s_cnt[2] <= kCandCapS1) {
        // Break-out: survivors fit. One collect pass emits the
        // strictly-greater keys and captures the pivot-prefix matches;
        // (p+8)-bit prefix mask — NOT mask>>8, which would drop the
        // already-fixed top bits.
        const uint32_t mask_c = ~0u << (24 - p);
        const uint32_t prefix_c = s_scan[0];
        if (tx == 0) {
          s_cnt[0] = 0;  // emit cursor: strictly greater
          s_cnt[1] = 0;  // candidate cursor
        }
        __syncthreads();
        if (vec) {
          const uint32_t n4 = len >> 2;
          const float4* in4 = reinterpret_cast<const float4*>(seg);
          uint32_t v = tx;
          for (; v + BS < n4; v += 2 * BS) {
            const float4 f0 = in4[v];
            const float4 f1 = in4[v + BS];
            const uint32_t i0 = v * 4;
            const uint32_t i1 = i0 + BS * 4;
            emit_or_collect(ordered_key(f0.x),
                            (int)(my_start + i0 + 0), mask_c, prefix_c, cnt_gt,
                            kCandCapS1, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f0.y),
                            (int)(my_start + i0 + 1), mask_c, prefix_c, cnt_gt,
                            kCandCapS1, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f0.z),
                            (int)(my_start + i0 + 2), mask_c, prefix_c, cnt_gt,
                            kCandCapS1, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f0.w),
                            (int)(my_start + i0 + 3), mask_c, prefix_c, cnt_gt,
                            kCandCapS1, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f1.x),
                            (int)(my_start + i1 + 0), mask_c, prefix_c, cnt_gt,
                            kCandCapS1, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f1.y),
                            (int)(my_start + i1 + 1), mask_c, prefix_c, cnt_gt,
                            kCandCapS1, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f1.z),
                            (int)(my_start + i1 + 2), mask_c, prefix_c, cnt_gt,
                            kCandCapS1, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f1.w),
                            (int)(my_start + i1 + 3), mask_c, prefix_c, cnt_gt,
                            kCandCapS1, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
          }
          for (; v < n4; v += BS) {
            const float4 f = in4[v];
            const uint32_t i0 = v * 4;
            emit_or_collect(ordered_key(f.x),
                            (int)(my_start + i0 + 0), mask_c, prefix_c, cnt_gt,
                            kCandCapS1, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f.y),
                            (int)(my_start + i0 + 1), mask_c, prefix_c, cnt_gt,
                            kCandCapS1, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f.z),
                            (int)(my_start + i0 + 2), mask_c, prefix_c, cnt_gt,
                            kCandCapS1, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f.w),
                            (int)(my_start + i0 + 3), mask_c, prefix_c, cnt_gt,
                            kCandCapS1, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
          }
          for (uint32_t i = (n4 << 2) + tx; i < len; i += BS)
            emit_or_collect(ordered_key(seg[i]), (int)(my_start + i), mask_c,
                            prefix_c, cnt_gt, kCandCapS1, s_cnt, out_row,
                            s_cand_key, s_cand_idx);
        } else {
          for (uint32_t i = tx; i < len; i += BS)
            emit_or_collect(
                ordered_key(seg[static_cast<int64_t>(i) * stride1]),
                (int)(my_start + i), mask_c, prefix_c, cnt_gt, kCandCapS1,
                s_cnt, out_row, s_cand_key, s_cand_idx);
        }
        __syncthreads();
        probe_mark(12);  // overflow rounds + collect pass
        tail_select<BS>(tx, s_cand_key, s_cand_idx, s_cnt[1], cnt_gt, top_k,
                        out_row, s_hist, s_scan, s_cnt,
                        (32 - kR0Shift) + 8 * r);
        broke_out = true;
        break;
      }
    }
    if (!broke_out) {
      const uint32_t pivot = s_scan[0];
      const uint32_t total_gt = (uint32_t)top_k - s_scan[1];
      if (tx == 0) {
        s_cnt[0] = 0;
        s_cnt[1] = 0;
      }
      __syncthreads();
      if (vec) {
        const uint32_t n4 = len >> 2;
        const float4* in4 = reinterpret_cast<const float4*>(seg);
        uint32_t v = tx;
        for (; v + BS < n4; v += 2 * BS) {
          const float4 f0 = in4[v];
          const float4 f1 = in4[v + BS];
          const uint32_t i0 = v * 4;
          const uint32_t i1 = i0 + BS * 4;
          emit_final_el(ordered_key(f0.x), (int)(my_start + i0 + 0), pivot,
                        total_gt, top_k, s_cnt, out_row);
          emit_final_el(ordered_key(f0.y), (int)(my_start + i0 + 1), pivot,
                        total_gt, top_k, s_cnt, out_row);
          emit_final_el(ordered_key(f0.z), (int)(my_start + i0 + 2), pivot,
                        total_gt, top_k, s_cnt, out_row);
          emit_final_el(ordered_key(f0.w), (int)(my_start + i0 + 3), pivot,
                        total_gt, top_k, s_cnt, out_row);
          emit_final_el(ordered_key(f1.x), (int)(my_start + i1 + 0), pivot,
                        total_gt, top_k, s_cnt, out_row);
          emit_final_el(ordered_key(f1.y), (int)(my_start + i1 + 1), pivot,
                        total_gt, top_k, s_cnt, out_row);
          emit_final_el(ordered_key(f1.z), (int)(my_start + i1 + 2), pivot,
                        total_gt, top_k, s_cnt, out_row);
          emit_final_el(ordered_key(f1.w), (int)(my_start + i1 + 3), pivot,
                        total_gt, top_k, s_cnt, out_row);
        }
        for (; v < n4; v += BS) {
          const float4 f = in4[v];
          const uint32_t i0 = v * 4;
          emit_final_el(ordered_key(f.x), (int)(my_start + i0 + 0), pivot,
                        total_gt, top_k, s_cnt, out_row);
          emit_final_el(ordered_key(f.y), (int)(my_start + i0 + 1), pivot,
                        total_gt, top_k, s_cnt, out_row);
          emit_final_el(ordered_key(f.z), (int)(my_start + i0 + 2), pivot,
                        total_gt, top_k, s_cnt, out_row);
          emit_final_el(ordered_key(f.w), (int)(my_start + i0 + 3), pivot,
                        total_gt, top_k, s_cnt, out_row);
        }
        for (uint32_t i = (n4 << 2) + tx; i < len; i += BS)
          emit_final_el(ordered_key(seg[i]), (int)(my_start + i), pivot,
                        total_gt, top_k, s_cnt, out_row);
      } else {
        for (uint32_t i = tx; i < len; i += BS)
          emit_final_el(
              ordered_key(seg[static_cast<int64_t>(i) * stride1]),
              (int)(my_start + i), pivot, total_gt, top_k, s_cnt, out_row);
      }
      probe_mark(12);  // overflow fallback (rounds 1-3 + emit)
    }
  } else {
    // Pass 2: emit guaranteed-high keys, collect the pivot bucket.
    const uint32_t prefix_p2 = pivot_r0 << kR0Shift;
    if (vec) {
      const uint32_t n4 = len >> 2;
      const float4* in4 = reinterpret_cast<const float4*>(seg);
      uint32_t v = tx;
      for (; v + BS < n4; v += 2 * BS) {
        const float4 f0 = in4[v];
        const float4 f1 = in4[v + BS];
        const uint32_t i0 = v * 4;
        const uint32_t i1 = i0 + BS * 4;
        emit_or_collect(ordered_key(f0.x), (int)(my_start + i0 + 0), kMaskP2,
                        prefix_p2, cnt_hi, kCandCapS1, s_cnt, out_row,
                        s_cand_key, s_cand_idx);
        emit_or_collect(ordered_key(f0.y), (int)(my_start + i0 + 1), kMaskP2,
                        prefix_p2, cnt_hi, kCandCapS1, s_cnt, out_row,
                        s_cand_key, s_cand_idx);
        emit_or_collect(ordered_key(f0.z), (int)(my_start + i0 + 2), kMaskP2,
                        prefix_p2, cnt_hi, kCandCapS1, s_cnt, out_row,
                        s_cand_key, s_cand_idx);
        emit_or_collect(ordered_key(f0.w), (int)(my_start + i0 + 3), kMaskP2,
                        prefix_p2, cnt_hi, kCandCapS1, s_cnt, out_row,
                        s_cand_key, s_cand_idx);
        emit_or_collect(ordered_key(f1.x), (int)(my_start + i1 + 0), kMaskP2,
                        prefix_p2, cnt_hi, kCandCapS1, s_cnt, out_row,
                        s_cand_key, s_cand_idx);
        emit_or_collect(ordered_key(f1.y), (int)(my_start + i1 + 1), kMaskP2,
                        prefix_p2, cnt_hi, kCandCapS1, s_cnt, out_row,
                        s_cand_key, s_cand_idx);
        emit_or_collect(ordered_key(f1.z), (int)(my_start + i1 + 2), kMaskP2,
                        prefix_p2, cnt_hi, kCandCapS1, s_cnt, out_row,
                        s_cand_key, s_cand_idx);
        emit_or_collect(ordered_key(f1.w), (int)(my_start + i1 + 3), kMaskP2,
                        prefix_p2, cnt_hi, kCandCapS1, s_cnt, out_row,
                        s_cand_key, s_cand_idx);
      }
      for (; v < n4; v += BS) {
        const float4 f = in4[v];
        const uint32_t i0 = v * 4;
        emit_or_collect(ordered_key(f.x), (int)(my_start + i0 + 0), kMaskP2,
                        prefix_p2, cnt_hi, kCandCapS1, s_cnt, out_row,
                        s_cand_key, s_cand_idx);
        emit_or_collect(ordered_key(f.y), (int)(my_start + i0 + 1), kMaskP2,
                        prefix_p2, cnt_hi, kCandCapS1, s_cnt, out_row,
                        s_cand_key, s_cand_idx);
        emit_or_collect(ordered_key(f.z), (int)(my_start + i0 + 2), kMaskP2,
                        prefix_p2, cnt_hi, kCandCapS1, s_cnt, out_row,
                        s_cand_key, s_cand_idx);
        emit_or_collect(ordered_key(f.w), (int)(my_start + i0 + 3), kMaskP2,
                        prefix_p2, cnt_hi, kCandCapS1, s_cnt, out_row,
                        s_cand_key, s_cand_idx);
      }
      for (uint32_t i = (n4 << 2) + tx; i < len; i += BS)
        emit_or_collect(ordered_key(seg[i]), (int)(my_start + i), kMaskP2,
                        prefix_p2, cnt_hi, kCandCapS1, s_cnt, out_row,
                        s_cand_key, s_cand_idx);
    } else {
      for (uint32_t i = tx; i < len; i += BS)
        emit_or_collect(
            ordered_key(seg[static_cast<int64_t>(i) * stride1]),
            (int)(my_start + i), kMaskP2, prefix_p2, cnt_hi, kCandCapS1,
            s_cnt, out_row, s_cand_key, s_cand_idx);
    }
    __syncthreads();
    probe_mark(11);  // pass 2: guaranteed-high emit + candidate collect
    tail_select<BS>(tx, s_cand_key, s_cand_idx, n_cand, cnt_hi, top_k, out_row,
                    s_hist, s_scan, s_cnt, 32 - kR0Shift);
  }
  __syncthreads();  // the ladders' out_row writes span the block; commit
                    // them before the key gather re-reads them
  // Key gather: <= top_k survivors, one 4B read each (L2-resident — the
  // same floats were just streamed). Sentinel slots (index -1) get key 0.
  {
    const float* lrow = logits + static_cast<int64_t>(row) * stride0;
    for (int i = (int)tx; i < top_k; i += BS) {
      const int v = ws_i[i];
      ws_k[i] = (v >= 0)
                    ? ordered_key(lrow[static_cast<int64_t>(v) * stride1])
                    : 0u;
    }
  }
  if (probe_enabled()) __syncthreads();
  probe_mark(8);  // key gather (stage 1 tail)
  probe_end();
}

template <int BS>
static __global__ __launch_bounds__(BS) void topk_merge_cand_kernel(
    const uint32_t* __restrict__ ws_key, const int* __restrict__ ws_idx,
    int* __restrict__ out_indices, int cand_len, int top_k,
    uint32_t cand_cap) {
  constexpr uint32_t kR0Shift = 21u;
  constexpr uint32_t kMaskP2 = ~0u << kR0Shift;  // resolved-top-bits mask
  __shared__ uint32_t s_hist[kRadixBinsC];
  __shared__ uint32_t s_scan[2];
  __shared__ uint32_t s_cnt[4];
  extern __shared__ uint32_t s_cand_pool[];
  uint32_t* s_cand_key = s_cand_pool;
  int* s_cand_idx = reinterpret_cast<int*>(s_cand_pool + cand_cap);
  const uint32_t tx = threadIdx.x;
  probe_begin(6);
  const uint32_t row = blockIdx.x;
  const uint32_t* __restrict__ keys =
      ws_key + static_cast<int64_t>(row) * cand_len;
  const int* __restrict__ idxs =
      ws_idx + static_cast<int64_t>(row) * cand_len;
  int* out_row = out_indices + static_cast<int64_t>(row) * top_k;
  const uint32_t len = static_cast<uint32_t>(cand_len);
  const bool vec =
      ((reinterpret_cast<size_t>(keys) | reinterpret_cast<size_t>(idxs)) &
       15u) == 0u &&
      (len & 3u) == 0u;

  // Pass 1: 11-bit histogram of the candidate keys.
  for (uint32_t i = tx; i < (uint32_t)kRadixBinsC; i += BS) s_hist[i] = 0;
  if (tx == 0) {
    s_scan[0] = 0;
    s_scan[1] = (uint32_t)top_k;
    s_cnt[0] = 0;
    s_cnt[1] = 0;
  }
  __syncthreads();
  if (vec) {
    const uint32_t n4 = len >> 2;
    const uint4* k4 = reinterpret_cast<const uint4*>(keys);
    uint32_t v = tx;
    for (; v + BS < n4; v += 2 * BS) {
      const uint4 a = k4[v];
      const uint4 b = k4[v + BS];
      atomicAdd(&s_hist[a.x >> kR0Shift], 1u);
      atomicAdd(&s_hist[a.y >> kR0Shift], 1u);
      atomicAdd(&s_hist[a.z >> kR0Shift], 1u);
      atomicAdd(&s_hist[a.w >> kR0Shift], 1u);
      atomicAdd(&s_hist[b.x >> kR0Shift], 1u);
      atomicAdd(&s_hist[b.y >> kR0Shift], 1u);
      atomicAdd(&s_hist[b.z >> kR0Shift], 1u);
      atomicAdd(&s_hist[b.w >> kR0Shift], 1u);
    }
    for (; v < n4; v += BS) {
      const uint4 a = k4[v];
      atomicAdd(&s_hist[a.x >> kR0Shift], 1u);
      atomicAdd(&s_hist[a.y >> kR0Shift], 1u);
      atomicAdd(&s_hist[a.z >> kR0Shift], 1u);
      atomicAdd(&s_hist[a.w >> kR0Shift], 1u);
    }
  } else {
    uint32_t v = tx;
    for (; v + BS < len; v += 2 * BS) {
      const uint32_t k0 = keys[v];
      const uint32_t k1 = keys[v + BS];
      atomicAdd(&s_hist[k0 >> kR0Shift], 1u);
      atomicAdd(&s_hist[k1 >> kR0Shift], 1u);
    }
    for (; v < len; v += BS)
      atomicAdd(&s_hist[keys[v] >> kR0Shift], 1u);
  }
  __threadfence_block();  // flush deferred histogram reductions
  __syncthreads();
  probe_mark(3);  // pass 1: candidate histogram
  if (tx < kWarpLanes) scan_pick_threshold_2048(s_hist, s_scan, 21);
  __syncthreads();
  probe_mark(7);  // round-0 threshold scan
  const uint32_t pivot_r0 = s_scan[0] >> kR0Shift;
  const uint32_t cnt_hi = (uint32_t)top_k - s_scan[1];
  // Same block-uniform overflow decision as the two-pass kernel: the
  // pivot-bin count must be read atomically by thread 0 and published
  // through a barrier, or a mid-flight reduction could split the block.
  if (tx == 0) s_cnt[2] = atomicAdd(&s_hist[pivot_r0], 0u);
  __syncthreads();
  const uint32_t n_cand = s_cnt[2];

  if (n_cand > cand_cap) {
    // Overflow: same refinement ladder as the two-pass kernel — full
    // re-read rounds, breaking out into one collect pass as soon as the
    // surviving window (strictly-greater keys + pivot bin) fits the
    // buffer. All-equal boundary candidates walk every round and finish
    // in the final emit; sentinel keys (0) resolve below every real key.
    for (uint32_t r = 1; r < (uint32_t)kRounds; ++r) {
      const uint32_t p = (32 - kR0Shift) + 8 * (r - 1);
      const uint32_t shift = (p >= 24) ? 0u : (24 - p);
      const uint32_t mask = ~0u << (32 - p);
      const uint32_t prefix = s_scan[0];
      for (uint32_t i = tx; i < (uint32_t)kRadixBins; i += BS) s_hist[i] = 0;
      __syncthreads();
      if (vec) {
        const uint32_t n4 = len >> 2;
        const uint4* k4 = reinterpret_cast<const uint4*>(keys);
        uint32_t v = tx;
        for (; v + BS < n4; v += 2 * BS) {
          const uint4 a = k4[v];
          const uint4 b = k4[v + BS];
          hist_round_el(a.x, prefix, mask, shift, s_hist);
          hist_round_el(a.y, prefix, mask, shift, s_hist);
          hist_round_el(a.z, prefix, mask, shift, s_hist);
          hist_round_el(a.w, prefix, mask, shift, s_hist);
          hist_round_el(b.x, prefix, mask, shift, s_hist);
          hist_round_el(b.y, prefix, mask, shift, s_hist);
          hist_round_el(b.z, prefix, mask, shift, s_hist);
          hist_round_el(b.w, prefix, mask, shift, s_hist);
        }
        for (; v < n4; v += BS) {
          const uint4 a = k4[v];
          hist_round_el(a.x, prefix, mask, shift, s_hist);
          hist_round_el(a.y, prefix, mask, shift, s_hist);
          hist_round_el(a.z, prefix, mask, shift, s_hist);
          hist_round_el(a.w, prefix, mask, shift, s_hist);
        }
      } else {
        uint32_t v = tx;
        for (; v + BS < len; v += 2 * BS) {
          hist_round_el(keys[v], prefix, mask, shift, s_hist);
          hist_round_el(keys[v + BS], prefix, mask, shift, s_hist);
        }
        for (; v < len; v += BS)
          hist_round_el(keys[v], prefix, mask, shift, s_hist);
      }
      __threadfence_block();  // flush deferred histogram reductions
      __syncthreads();
      if (tx < kWarpLanes) scan_pick_threshold<false>(s_hist, s_scan, shift);
      __syncthreads();
      // Survivor count at this granularity, block-uniform via thread 0.
      if (tx == 0) s_cnt[2] = s_hist[(s_scan[0] >> shift) & 0xFF];
      __syncthreads();
      const uint32_t cnt_gt = (uint32_t)top_k - s_scan[1];
      if (r < (uint32_t)kRounds - 1 && cnt_gt + s_cnt[2] <= cand_cap) {
        // Break-out: survivors fit. One collect pass emits the
        // strictly-greater keys and captures the pivot-prefix matches;
        // (p+8)-bit prefix mask — NOT mask>>8, which would drop the
        // already-fixed top bits.
        const uint32_t mask_c = ~0u << (24 - p);
        const uint32_t prefix_c = s_scan[0];
        if (tx == 0) {
          s_cnt[0] = 0;  // emit cursor: strictly greater
          s_cnt[1] = 0;  // candidate cursor
        }
        __syncthreads();
        if (vec) {
          const uint32_t n4 = len >> 2;
          const uint4* k4 = reinterpret_cast<const uint4*>(keys);
          const int4* i4 = reinterpret_cast<const int4*>(idxs);
          uint32_t v = tx;
          for (; v + BS < n4; v += 2 * BS) {
            const uint4 a = k4[v];
            const int4 ia = i4[v];
            const uint4 b = k4[v + BS];
            const int4 ib = i4[v + BS];
            emit_or_collect(a.x, ia.x, mask_c, prefix_c, cnt_gt, cand_cap,
                            s_cnt, out_row, s_cand_key, s_cand_idx);
            emit_or_collect(a.y, ia.y, mask_c, prefix_c, cnt_gt, cand_cap,
                            s_cnt, out_row, s_cand_key, s_cand_idx);
            emit_or_collect(a.z, ia.z, mask_c, prefix_c, cnt_gt, cand_cap,
                            s_cnt, out_row, s_cand_key, s_cand_idx);
            emit_or_collect(a.w, ia.w, mask_c, prefix_c, cnt_gt, cand_cap,
                            s_cnt, out_row, s_cand_key, s_cand_idx);
            emit_or_collect(b.x, ib.x, mask_c, prefix_c, cnt_gt, cand_cap,
                            s_cnt, out_row, s_cand_key, s_cand_idx);
            emit_or_collect(b.y, ib.y, mask_c, prefix_c, cnt_gt, cand_cap,
                            s_cnt, out_row, s_cand_key, s_cand_idx);
            emit_or_collect(b.z, ib.z, mask_c, prefix_c, cnt_gt, cand_cap,
                            s_cnt, out_row, s_cand_key, s_cand_idx);
            emit_or_collect(b.w, ib.w, mask_c, prefix_c, cnt_gt, cand_cap,
                            s_cnt, out_row, s_cand_key, s_cand_idx);
          }
          for (; v < n4; v += BS) {
            const uint4 a = k4[v];
            const int4 ia = i4[v];
            emit_or_collect(a.x, ia.x, mask_c, prefix_c, cnt_gt, cand_cap,
                            s_cnt, out_row, s_cand_key, s_cand_idx);
            emit_or_collect(a.y, ia.y, mask_c, prefix_c, cnt_gt, cand_cap,
                            s_cnt, out_row, s_cand_key, s_cand_idx);
            emit_or_collect(a.z, ia.z, mask_c, prefix_c, cnt_gt, cand_cap,
                            s_cnt, out_row, s_cand_key, s_cand_idx);
            emit_or_collect(a.w, ia.w, mask_c, prefix_c, cnt_gt, cand_cap,
                            s_cnt, out_row, s_cand_key, s_cand_idx);
          }
        } else {
          uint32_t v = tx;
          for (; v + BS < len; v += 2 * BS) {
            emit_or_collect(keys[v], idxs[v], mask_c, prefix_c, cnt_gt,
                            cand_cap, s_cnt, out_row, s_cand_key, s_cand_idx);
            emit_or_collect(keys[v + BS], idxs[v + BS], mask_c, prefix_c,
                            cnt_gt, cand_cap, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
          }
          for (; v < len; v += BS)
            emit_or_collect(keys[v], idxs[v], mask_c, prefix_c, cnt_gt,
                            cand_cap, s_cnt, out_row, s_cand_key, s_cand_idx);
        }
        __syncthreads();
        probe_mark(12);  // overflow rounds + collect pass
        tail_select<BS>(tx, s_cand_key, s_cand_idx, s_cnt[1], cnt_gt, top_k,
                        out_row, s_hist, s_scan, s_cnt,
                        (32 - kR0Shift) + 8 * r);
        if (probe_enabled()) __syncthreads();
        probe_mark(8);  // tail selection
        probe_end();
        return;
      }
    }
    const uint32_t pivot = s_scan[0];
    const uint32_t total_gt = (uint32_t)top_k - s_scan[1];
    if (tx == 0) {
      s_cnt[0] = 0;
      s_cnt[1] = 0;
    }
    __syncthreads();
    if (vec) {
      const uint32_t n4 = len >> 2;
      const uint4* k4 = reinterpret_cast<const uint4*>(keys);
      const int4* i4 = reinterpret_cast<const int4*>(idxs);
      uint32_t v = tx;
      for (; v + BS < n4; v += 2 * BS) {
        const uint4 a = k4[v];
        const int4 ia = i4[v];
        const uint4 b = k4[v + BS];
        const int4 ib = i4[v + BS];
        emit_final_el(a.x, ia.x, pivot, total_gt, top_k, s_cnt, out_row);
        emit_final_el(a.y, ia.y, pivot, total_gt, top_k, s_cnt, out_row);
        emit_final_el(a.z, ia.z, pivot, total_gt, top_k, s_cnt, out_row);
        emit_final_el(a.w, ia.w, pivot, total_gt, top_k, s_cnt, out_row);
        emit_final_el(b.x, ib.x, pivot, total_gt, top_k, s_cnt, out_row);
        emit_final_el(b.y, ib.y, pivot, total_gt, top_k, s_cnt, out_row);
        emit_final_el(b.z, ib.z, pivot, total_gt, top_k, s_cnt, out_row);
        emit_final_el(b.w, ib.w, pivot, total_gt, top_k, s_cnt, out_row);
      }
      for (; v < n4; v += BS) {
        const uint4 a = k4[v];
        const int4 ia = i4[v];
        emit_final_el(a.x, ia.x, pivot, total_gt, top_k, s_cnt, out_row);
        emit_final_el(a.y, ia.y, pivot, total_gt, top_k, s_cnt, out_row);
        emit_final_el(a.z, ia.z, pivot, total_gt, top_k, s_cnt, out_row);
        emit_final_el(a.w, ia.w, pivot, total_gt, top_k, s_cnt, out_row);
      }
    } else {
      uint32_t v = tx;
      for (; v + BS < len; v += 2 * BS) {
        emit_final_el(keys[v], idxs[v], pivot, total_gt, top_k, s_cnt,
                      out_row);
        emit_final_el(keys[v + BS], idxs[v + BS], pivot, total_gt, top_k,
                      s_cnt, out_row);
      }
      for (; v < len; v += BS)
        emit_final_el(keys[v], idxs[v], pivot, total_gt, top_k, s_cnt,
                      out_row);
    }
    probe_mark(12);  // overflow fallback (rounds 1-3 + emit)
    probe_end();
    return;
  }

  // Pass 2: emit guaranteed-high keys, collect the pivot bucket.
  {
    const uint32_t prefix_p2 = pivot_r0 << kR0Shift;
    if (vec) {
      const uint32_t n4 = len >> 2;
      const uint4* k4 = reinterpret_cast<const uint4*>(keys);
      const int4* i4 = reinterpret_cast<const int4*>(idxs);
      uint32_t v = tx;
      for (; v + BS < n4; v += 2 * BS) {
        const uint4 a = k4[v];
        const int4 ia = i4[v];
        const uint4 b = k4[v + BS];
        const int4 ib = i4[v + BS];
        emit_or_collect(a.x, ia.x, kMaskP2, prefix_p2, cnt_hi, cand_cap,
                        s_cnt, out_row, s_cand_key, s_cand_idx);
        emit_or_collect(a.y, ia.y, kMaskP2, prefix_p2, cnt_hi, cand_cap,
                        s_cnt, out_row, s_cand_key, s_cand_idx);
        emit_or_collect(a.z, ia.z, kMaskP2, prefix_p2, cnt_hi, cand_cap,
                        s_cnt, out_row, s_cand_key, s_cand_idx);
        emit_or_collect(a.w, ia.w, kMaskP2, prefix_p2, cnt_hi, cand_cap,
                        s_cnt, out_row, s_cand_key, s_cand_idx);
        emit_or_collect(b.x, ib.x, kMaskP2, prefix_p2, cnt_hi, cand_cap,
                        s_cnt, out_row, s_cand_key, s_cand_idx);
        emit_or_collect(b.y, ib.y, kMaskP2, prefix_p2, cnt_hi, cand_cap,
                        s_cnt, out_row, s_cand_key, s_cand_idx);
        emit_or_collect(b.z, ib.z, kMaskP2, prefix_p2, cnt_hi, cand_cap,
                        s_cnt, out_row, s_cand_key, s_cand_idx);
        emit_or_collect(b.w, ib.w, kMaskP2, prefix_p2, cnt_hi, cand_cap,
                        s_cnt, out_row, s_cand_key, s_cand_idx);
      }
      for (; v < n4; v += BS) {
        const uint4 a = k4[v];
        const int4 ia = i4[v];
        emit_or_collect(a.x, ia.x, kMaskP2, prefix_p2, cnt_hi, cand_cap,
                        s_cnt, out_row, s_cand_key, s_cand_idx);
        emit_or_collect(a.y, ia.y, kMaskP2, prefix_p2, cnt_hi, cand_cap,
                        s_cnt, out_row, s_cand_key, s_cand_idx);
        emit_or_collect(a.z, ia.z, kMaskP2, prefix_p2, cnt_hi, cand_cap,
                        s_cnt, out_row, s_cand_key, s_cand_idx);
        emit_or_collect(a.w, ia.w, kMaskP2, prefix_p2, cnt_hi, cand_cap,
                        s_cnt, out_row, s_cand_key, s_cand_idx);
      }
    } else {
      uint32_t v = tx;
      for (; v + BS < len; v += 2 * BS) {
        emit_or_collect(keys[v], idxs[v], kMaskP2, prefix_p2, cnt_hi, cand_cap,
                        s_cnt, out_row, s_cand_key, s_cand_idx);
        emit_or_collect(keys[v + BS], idxs[v + BS], kMaskP2, prefix_p2, cnt_hi,
                        cand_cap, s_cnt, out_row, s_cand_key, s_cand_idx);
      }
      for (; v < len; v += BS)
        emit_or_collect(keys[v], idxs[v], kMaskP2, prefix_p2, cnt_hi, cand_cap,
                        s_cnt, out_row, s_cand_key, s_cand_idx);
    }
  }
  __syncthreads();
  probe_mark(11);  // pass 2: guaranteed-high emit + candidate collect

  tail_select<BS>(tx, s_cand_key, s_cand_idx, n_cand, cnt_hi, top_k, out_row,
                  s_hist, s_scan, s_cnt, 32 - kR0Shift);
  if (probe_enabled()) __syncthreads();
  probe_mark(8);  // tail selection
  probe_end();
}

// ===========================================================================
// Path C: two-pass global radix for long rows with many rows (or a <= 2
// chunk split, where split-K barrier overhead dominates). One block per row,
// no staging, no barriers, no workspace: pass 1 histograms the top bits
// straight from global memory — the top ELEVEN bits when WIDE (long rows,
// where 2048 bins cut hot-bin atomic contention by ~8x and thin the pivot
// bucket enough to co-schedule two blocks per SM), the top byte otherwise;
// pass 2 re-reads the row, emits every key above the pivot prefix directly
// (guaranteed top-K) and collects the pivot bucket's candidates; a
// block-wide tail resolves the exact pivot. Overflow rows (pivot bucket
// heavier than the candidate buffer) refine with full re-read rounds but
// break out into a collect pass as soon as the refined survivor window fits
// the buffer. The launcher picks WIDE from the row length alone (measured:
// the 2048-bin threshold scan's ~4.7K-cycle fixed cost wins from ~49K
// columns up, loses below — there the byte ladder is faster).
// ===========================================================================
template <int BS, bool kIsPrefill, bool WIDE>
static __global__ __launch_bounds__(BS) void topk_2pass_row_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ starts_or_seq,  // prefill: row starts; decode: seq lens
    const int* __restrict__ ends_or_null,   // prefill: row ends
    int* __restrict__ out_indices, int stride0, int stride1, int top_k,
    int next_n, int seq_lens_is2d, int num_columns, uint32_t cand_cap) {
  // Round-0 radix width. The ladder formulas below are written against
  // kR0Shift so both widths share one code path: overflow round r resolves
  // p = (32 - kR0Shift) + 8*(r-1) bits, the compact-pass mask is
  // ~0u << kR0Shift, and tail_select starts at 32 - kR0Shift.
  constexpr uint32_t kR0Shift = WIDE ? 21u : 24u;
  constexpr int kR0Bins = WIDE ? kRadixBinsC : kRadixBins;
  constexpr uint32_t kMaskP2 = ~0u << kR0Shift;  // resolved-top-bits mask
  __shared__ uint32_t s_hist[kRadixBinsC];
  __shared__ uint32_t s_scan[2];
  __shared__ uint32_t s_cnt[4];
  // Candidate buffer as DYNAMIC shared memory (opt-in via the launcher):
  // the pool holds cand_cap (key, index) pairs; the launcher sizes it for
  // occupancy — 6144 pairs (49152B) when one block per SM is fine, or 2560
  // pairs (20480B) so two blocks co-reside on C500's 64KB when there are
  // enough rows to fill them (per-CTA streaming at mid row lengths is
  // latency-bound, so a second block nearly doubles per-SM throughput).
  extern __shared__ uint32_t s_cand_pool[];
  uint32_t* s_cand_key = s_cand_pool;
  int* s_cand_idx = reinterpret_cast<int*>(s_cand_pool + cand_cap);
  const int tx = threadIdx.x;
  const int row = blockIdx.x;
  probe_begin(4);
  int row_start = 0;
  int row_end = 0;
  if (kIsPrefill) {
    row_start = starts_or_seq[row];
    row_end = ends_or_null[row];
    if (row_start < 0) row_start = 0;
    if (row_start > num_columns) row_start = num_columns;
    if (row_end < row_start) row_end = row_start;
    if (row_end > num_columns) row_end = num_columns;
  } else {
    const int seq_len = seq_lens_is2d ? starts_or_seq[row]
                                      : starts_or_seq[row / next_n];
    row_end = seq_lens_is2d
                  ? max(0, seq_len)
                  : max(0, seq_len - next_n + row % next_n + 1);
    if (row_end > num_columns) row_end = num_columns;
  }
  int* out_row = out_indices + static_cast<int64_t>(row) * top_k;
  const int row_len = row_end - row_start;
  if (row_len <= top_k) {
    for (int i = tx; i < row_len; i += BS) out_row[i] = i;
    for (int i = row_len + tx; i < top_k; i += BS) out_row[i] = -1;
    if (probe_enabled()) __syncthreads();
    probe_mark(10);  // short row
    probe_count(14);
    probe_end();
    return;
  }
  const float* seg = logits + static_cast<int64_t>(row) * stride0 +
                     static_cast<int64_t>(row_start) * stride1;
  const bool vec =
      stride1 == 1 && (reinterpret_cast<size_t>(seg) & 15u) == 0u;
  const uint32_t len = static_cast<uint32_t>(row_len);

  // Pass 1: round-0 histogram directly from global memory (11-bit when WIDE,
  // else the top byte).
  for (uint32_t i = tx; i < (uint32_t)kR0Bins; i += BS) s_hist[i] = 0;
  if (tx == 0) {
    s_scan[0] = 0;
    s_scan[1] = (uint32_t)top_k;
    s_cnt[0] = 0;
    s_cnt[1] = 0;
  }
  __syncthreads();
  if (vec) {
    const uint32_t n4 = len >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(seg);
    // Four independent float4 loads in flight: with one block per SM the
    // load-issue-to-first-use chain caps streaming well below the SM's
    // LSU/fabric capability (probed: unroll-1 ~9.1GB/s, unroll-2 ~11.3GB/s
    // per CTA); extra outstanding quads keep the memory pipe busy while
    // earlier quads' histogram atomics retire. Long rows only — see
    // kWideUnrollQuads.
    uint32_t v = tx;
    for (; n4 >= kWideUnrollQuads * BS && v + 3 * BS < n4; v += 4 * BS) {
      const float4 f0 = in4[v];
      const float4 f1 = in4[v + BS];
      const float4 f2 = in4[v + 2 * BS];
      const float4 f3 = in4[v + 3 * BS];
      atomicAdd(&s_hist[ordered_key(f0.x) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f0.y) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f0.z) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f0.w) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.x) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.y) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.z) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.w) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f2.x) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f2.y) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f2.z) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f2.w) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f3.x) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f3.y) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f3.z) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f3.w) >> kR0Shift], 1u);
    }
    for (; v + BS < n4; v += 2 * BS) {
      const float4 f0 = in4[v];
      const float4 f1 = in4[v + BS];
      atomicAdd(&s_hist[ordered_key(f0.x) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f0.y) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f0.z) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f0.w) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.x) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.y) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.z) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f1.w) >> kR0Shift], 1u);
    }
    for (; v < n4; v += BS) {
      const float4 f = in4[v];
      atomicAdd(&s_hist[ordered_key(f.x) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f.y) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f.z) >> kR0Shift], 1u);
      atomicAdd(&s_hist[ordered_key(f.w) >> kR0Shift], 1u);
    }
    for (uint32_t i = (n4 << 2) + tx; i < len; i += BS)
      atomicAdd(&s_hist[ordered_key(seg[i]) >> kR0Shift], 1u);
  } else {
    for (uint32_t i = tx; i < len; i += BS)
      atomicAdd(
          &s_hist[ordered_key(seg[static_cast<int64_t>(i) * stride1]) >> kR0Shift],
          1u);
  }
  __threadfence_block();  // flush deferred histogram reductions
  __syncthreads();
  probe_mark(3);  // pass 1: global round-0 histogram
  if (tx < kWarpLanes) {
    if (WIDE)
      scan_pick_threshold_2048(s_hist, s_scan, 21);
    else
      scan_pick_threshold<false>(s_hist, s_scan, 24);
  }
  __syncthreads();
  probe_mark(7);  // round-0 threshold scan
  const uint32_t pivot_r0 = s_scan[0] >> kR0Shift;
  const uint32_t cnt_hi = (uint32_t)top_k - s_scan[1];
  // Decide from the pass-1 histogram: s_hist[pivot_r0] equals exactly the
  // number of pivot-bucket candidates pass 2 would collect. Overflow rows
  // skip pass 2 entirely and go straight to the full re-read rounds.
  // The count is read atomically by thread 0 and published through a
  // barrier: shared-memory reductions may still be in flight for a plain
  // post-barrier read, and a non-uniform decision would split the block
  // across the two branches. s_cnt[2] is scratch: both branches re-init or
  // never read it.
  if (tx == 0) s_cnt[2] = atomicAdd(&s_hist[pivot_r0], 0u);
  __syncthreads();
  const uint32_t n_cand = s_cnt[2];

  if (n_cand > cand_cap) {
    // Overflow: the round-0 pivot bucket is heavier than the candidate
    // buffer. Refine with full global re-read rounds, but break out as soon
    // as the surviving window (strictly-greater keys + pivot bin) fits the
    // buffer — one collect pass then narrows to the refined prefix and the
    // tail finishes in shared memory. Only truly degenerate distributions
    // (huge tied-bit plateaus) still run all rounds plus the final emit
    // pass. Round r resolves 8 bits past round 0: p = (32 - kR0Shift) +
    // 8*(r-1) bits are already fixed, so this round histograms
    // shift = 24 - p and the survivors match mask = ~0u << (32 - p).
    for (uint32_t r = 1; r < (uint32_t)kRounds; ++r) {
      const uint32_t p = (32 - kR0Shift) + 8 * (r - 1);
      const uint32_t shift = (p >= 24) ? 0u : (24 - p);
      const uint32_t mask = ~0u << (32 - p);
      const uint32_t prefix = s_scan[0];
      for (uint32_t i = tx; i < (uint32_t)kRadixBins; i += BS) s_hist[i] = 0;
      __syncthreads();
      if (vec) {
        const uint32_t n4 = len >> 2;
        const float4* in4 = reinterpret_cast<const float4*>(seg);
        uint32_t v = tx;
        for (; v + BS < n4; v += 2 * BS) {
          const float4 f0 = in4[v];
          const float4 f1 = in4[v + BS];
          hist_round_el(ordered_key(f0.x), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f0.y), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f0.z), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f0.w), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f1.x), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f1.y), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f1.z), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f1.w), prefix, mask, shift, s_hist);
        }
        for (; v < n4; v += BS) {
          const float4 f = in4[v];
          hist_round_el(ordered_key(f.x), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f.y), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f.z), prefix, mask, shift, s_hist);
          hist_round_el(ordered_key(f.w), prefix, mask, shift, s_hist);
        }
        for (uint32_t i = (n4 << 2) + tx; i < len; i += BS)
          hist_round_el(ordered_key(seg[i]), prefix, mask, shift, s_hist);
      } else {
        for (uint32_t i = tx; i < len; i += BS)
          hist_round_el(
              ordered_key(seg[static_cast<int64_t>(i) * stride1]), prefix,
              mask, shift, s_hist);
      }
      __threadfence_block();  // flush deferred histogram reductions
      __syncthreads();
      if (tx < kWarpLanes) scan_pick_threshold<false>(s_hist, s_scan, shift);
      __syncthreads();
      // Survivor count at this granularity (strictly-greater + pivot bin),
      // block-uniform via thread 0; plain read — the producing loop's fence
      // plus the barriers above committed the bin.
      if (tx == 0) s_cnt[2] = s_hist[(s_scan[0] >> shift) & 0xFF];
      __syncthreads();
      const uint32_t cnt_gt = (uint32_t)top_k - s_scan[1];
      if (r < (uint32_t)kRounds - 1 && cnt_gt + s_cnt[2] <= cand_cap) {
        // Break-out: the survivors fit the candidate buffer. One collect
        // pass emits the strictly-greater keys (exactly cnt_gt of them) and
        // captures the pivot-prefix matches; the tail refines just those.
        // (p+8)-bit prefix mask: bits above 32-p-8. NOT mask>>8, which
        // drops the already-fixed top bits.
        const uint32_t mask_c = ~0u << (24 - p);
        const uint32_t prefix_c = s_scan[0];
        if (tx == 0) {
          s_cnt[0] = 0;  // emit cursor: strictly greater
          s_cnt[1] = 0;  // candidate cursor
        }
        __syncthreads();
        if (vec) {
          const uint32_t n4 = len >> 2;
          const float4* in4 = reinterpret_cast<const float4*>(seg);
          uint32_t v = tx;
          for (; v + BS < n4; v += 2 * BS) {
            const float4 f0 = in4[v];
            const float4 f1 = in4[v + BS];
            const uint32_t i0 = v * 4;
            const uint32_t i1 = i0 + BS * 4;
            emit_or_collect(ordered_key(f0.x), (int)(i0 + 0), mask_c, prefix_c,
                            cnt_gt, cand_cap, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f0.y), (int)(i0 + 1), mask_c, prefix_c,
                            cnt_gt, cand_cap, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f0.z), (int)(i0 + 2), mask_c, prefix_c,
                            cnt_gt, cand_cap, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f0.w), (int)(i0 + 3), mask_c, prefix_c,
                            cnt_gt, cand_cap, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f1.x), (int)(i1 + 0), mask_c, prefix_c,
                            cnt_gt, cand_cap, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f1.y), (int)(i1 + 1), mask_c, prefix_c,
                            cnt_gt, cand_cap, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f1.z), (int)(i1 + 2), mask_c, prefix_c,
                            cnt_gt, cand_cap, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f1.w), (int)(i1 + 3), mask_c, prefix_c,
                            cnt_gt, cand_cap, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
          }
          for (; v < n4; v += BS) {
            const float4 f = in4[v];
            const uint32_t i0 = v * 4;
            emit_or_collect(ordered_key(f.x), (int)(i0 + 0), mask_c, prefix_c,
                            cnt_gt, cand_cap, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f.y), (int)(i0 + 1), mask_c, prefix_c,
                            cnt_gt, cand_cap, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f.z), (int)(i0 + 2), mask_c, prefix_c,
                            cnt_gt, cand_cap, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
            emit_or_collect(ordered_key(f.w), (int)(i0 + 3), mask_c, prefix_c,
                            cnt_gt, cand_cap, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
          }
          for (uint32_t i = (n4 << 2) + tx; i < len; i += BS)
            emit_or_collect(ordered_key(seg[i]), (int)i, mask_c, prefix_c,
                            cnt_gt, cand_cap, s_cnt, out_row, s_cand_key,
                            s_cand_idx);
        } else {
          for (uint32_t i = tx; i < len; i += BS)
            emit_or_collect(
                ordered_key(seg[static_cast<int64_t>(i) * stride1]), (int)i,
                mask_c, prefix_c, cnt_gt, cand_cap, s_cnt, out_row,
                s_cand_key, s_cand_idx);
        }
        __syncthreads();
        probe_mark(12);  // overflow rounds + collect pass
        tail_select<BS>(tx, s_cand_key, s_cand_idx, s_cnt[1], cnt_gt, top_k,
                        out_row, s_hist, s_scan, s_cnt, (32 - kR0Shift) + 8 * r);
        if (probe_enabled()) __syncthreads();
        probe_mark(8);  // tail selection
        probe_end();
        return;
      }
    }
    const uint32_t pivot = s_scan[0];
    const uint32_t total_gt = (uint32_t)top_k - s_scan[1];
    if (tx == 0) {
      s_cnt[0] = 0;
      s_cnt[1] = 0;
    }
    __syncthreads();
    if (vec) {
      const uint32_t n4 = len >> 2;
      const float4* in4 = reinterpret_cast<const float4*>(seg);
      uint32_t v = tx;
      for (; v + BS < n4; v += 2 * BS) {
        const float4 f0 = in4[v];
        const float4 f1 = in4[v + BS];
        const uint32_t i0 = v * 4;
        const uint32_t i1 = i0 + BS * 4;
        emit_final_el(ordered_key(f0.x), (int)(i0 + 0), pivot, total_gt, top_k,
                      s_cnt, out_row);
        emit_final_el(ordered_key(f0.y), (int)(i0 + 1), pivot, total_gt, top_k,
                      s_cnt, out_row);
        emit_final_el(ordered_key(f0.z), (int)(i0 + 2), pivot, total_gt, top_k,
                      s_cnt, out_row);
        emit_final_el(ordered_key(f0.w), (int)(i0 + 3), pivot, total_gt, top_k,
                      s_cnt, out_row);
        emit_final_el(ordered_key(f1.x), (int)(i1 + 0), pivot, total_gt, top_k,
                      s_cnt, out_row);
        emit_final_el(ordered_key(f1.y), (int)(i1 + 1), pivot, total_gt, top_k,
                      s_cnt, out_row);
        emit_final_el(ordered_key(f1.z), (int)(i1 + 2), pivot, total_gt, top_k,
                      s_cnt, out_row);
        emit_final_el(ordered_key(f1.w), (int)(i1 + 3), pivot, total_gt, top_k,
                      s_cnt, out_row);
      }
      for (; v < n4; v += BS) {
        const float4 f = in4[v];
        const uint32_t i0 = v * 4;
        emit_final_el(ordered_key(f.x), (int)(i0 + 0), pivot, total_gt, top_k,
                      s_cnt, out_row);
        emit_final_el(ordered_key(f.y), (int)(i0 + 1), pivot, total_gt, top_k,
                      s_cnt, out_row);
        emit_final_el(ordered_key(f.z), (int)(i0 + 2), pivot, total_gt, top_k,
                      s_cnt, out_row);
        emit_final_el(ordered_key(f.w), (int)(i0 + 3), pivot, total_gt, top_k,
                      s_cnt, out_row);
      }
      for (uint32_t i = (n4 << 2) + tx; i < len; i += BS)
        emit_final_el(ordered_key(seg[i]), (int)i, pivot, total_gt, top_k,
                      s_cnt, out_row);
    } else {
      for (uint32_t i = tx; i < len; i += BS)
        emit_final_el(
            ordered_key(seg[static_cast<int64_t>(i) * stride1]), (int)i, pivot,
            total_gt, top_k, s_cnt, out_row);
    }
    probe_mark(12);  // overflow fallback (rounds 1-3 + emit)
    probe_end();
    return;
  }

  // Pass 2: re-read the row; emit guaranteed-high keys, collect candidates.
  // Four independent float4 loads in flight, same rationale as pass 1 (and
  // the same long-row gate — kWideUnrollQuads).
  if (vec) {
    const uint32_t n4 = len >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(seg);
    uint32_t v = tx;
    for (; n4 >= kWideUnrollQuads * BS && v + 3 * BS < n4; v += 4 * BS) {
      const float4 f0 = in4[v];
      const float4 f1 = in4[v + BS];
      const float4 f2 = in4[v + 2 * BS];
      const float4 f3 = in4[v + 3 * BS];
      const uint32_t i0 = v * 4;
      const uint32_t i1 = i0 + BS * 4;
      const uint32_t i2 = i0 + 2 * BS * 4;
      const uint32_t i3 = i0 + 3 * BS * 4;
      emit_or_collect(ordered_key(f0.x), (int)(i0 + 0), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f0.y), (int)(i0 + 1), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f0.z), (int)(i0 + 2), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f0.w), (int)(i0 + 3), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f1.x), (int)(i1 + 0), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f1.y), (int)(i1 + 1), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f1.z), (int)(i1 + 2), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f1.w), (int)(i1 + 3), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f2.x), (int)(i2 + 0), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f2.y), (int)(i2 + 1), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f2.z), (int)(i2 + 2), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f2.w), (int)(i2 + 3), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f3.x), (int)(i3 + 0), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f3.y), (int)(i3 + 1), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f3.z), (int)(i3 + 2), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f3.w), (int)(i3 + 3), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
    }
    for (; v + BS < n4; v += 2 * BS) {
      const float4 f0 = in4[v];
      const float4 f1 = in4[v + BS];
      const uint32_t i0 = v * 4;
      const uint32_t i1 = i0 + BS * 4;
      emit_or_collect(ordered_key(f0.x), (int)(i0 + 0), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f0.y), (int)(i0 + 1), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f0.z), (int)(i0 + 2), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f0.w), (int)(i0 + 3), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f1.x), (int)(i1 + 0), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f1.y), (int)(i1 + 1), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f1.z), (int)(i1 + 2), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f1.w), (int)(i1 + 3), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
    }
    for (; v < n4; v += BS) {
      const float4 f = in4[v];
      const uint32_t i0 = v * 4;
      emit_or_collect(ordered_key(f.x), (int)(i0 + 0), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f.y), (int)(i0 + 1), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f.z), (int)(i0 + 2), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
      emit_or_collect(ordered_key(f.w), (int)(i0 + 3), kMaskP2,
                      pivot_r0 << kR0Shift, cnt_hi, cand_cap, s_cnt, out_row,
                      s_cand_key, s_cand_idx);
    }
    for (uint32_t i = (n4 << 2) + tx; i < len; i += BS)
      emit_or_collect(ordered_key(seg[i]), (int)i, kMaskP2, pivot_r0 << kR0Shift,
                      cnt_hi, cand_cap, s_cnt, out_row, s_cand_key,
                      s_cand_idx);
  } else {
    for (uint32_t i = tx; i < len; i += BS)
      emit_or_collect(ordered_key(seg[static_cast<int64_t>(i) * stride1]),
                      (int)i, kMaskP2, pivot_r0 << kR0Shift, cnt_hi, cand_cap,
                      s_cnt, out_row, s_cand_key, s_cand_idx);
  }
  __syncthreads();
  probe_mark(11);  // pass 2: guaranteed-high emit + candidate collect

  tail_select<BS>(tx, s_cand_key, s_cand_idx, n_cand, cnt_hi, top_k, out_row,
                  s_hist, s_scan, s_cnt, 32 - kR0Shift);
  if (probe_enabled()) __syncthreads();
  probe_mark(8);  // tail selection
  probe_end();
}

}  // namespace topk_impl
}  // namespace vllm

// ===========================================================================
// Host dispatch. Shape-only decisions (CUDA-graph safe): the row length
// bound numColumns (logits.size(1)) and the row count select Path A / B / C;
// no host reads of device data.
// ===========================================================================
namespace {

using vllm::topk_impl::RowState;

constexpr int64_t kTwoPassMinRows = 20;

constexpr uint32_t kTwoPassMaxSplit = 2;

constexpr int64_t kTwoPassSmallPoolMaxLen = 32768;

constexpr int64_t kTwoPassWideMinLen = 49152;

constexpr int64_t kSplitNoBarrierMinJobs = 160;

// Per-process chunk cap from the device's opt-in smem limit.
//   >=128KB (C600U): 16384 floats (64KB dynamic, ample headroom).
//   <=64KB  (C500) : 13056 floats (52224B dynamic + ~1KB static < 65536B).
//   otherwise      : 80% of the opt-in budget, aligned down to 128 floats.
inline uint32_t query_chunk_cap() {
  static const uint32_t cap = [] {
    int dev = 0, optin = 64 * 1024;
    if (::cudaGetDevice(&dev) == cudaSuccess)
      ::cudaDeviceGetAttribute(&optin,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (optin <= 0) optin = 64 * 1024;
    if (optin >= 128 * 1024) return vllm::topk_impl::kChunkCapLarge;
    if (optin <= 64 * 1024) return vllm::topk_impl::kChunkCapSmall;
    const uint32_t budget =
        static_cast<uint32_t>((static_cast<uint64_t>(optin) * 8) / 10);
    uint32_t c = (budget / sizeof(uint32_t)) & ~static_cast<uint32_t>(127);
    if (c == 0) c = vllm::topk_impl::kChunkCapSmall;
    return c;
  }();
  return cap;
}

// Templated on the kernel POINTER VALUE so every kernel gets its own one-time
// initialization (kernels sharing a signature must not share the static).
template <auto kKernel>
inline void optin_smem_once(int max_dyn_smem) {
  static const cudaError_t err = [max_dyn_smem] {
#ifdef USE_ROCM
    return ::cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kKernel),
        ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn_smem);
#else
    return ::cudaFuncSetAttribute(
        kKernel, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn_smem);
#endif
  }();
  TORCH_CHECK(err == cudaSuccess, "top-k smem opt-in failed: ",
              ::cudaGetErrorString(err));
}

// Resident block capacity at the max chunk smem. A smaller launch only ever
// fits more blocks, so this is a safe (conservative) co-residency bound.
template <auto kKernel>
inline int resident_blocks_once(int threads, int dyn_smem) {
  static const int r = [threads, dyn_smem] {
    int dev = 0, sms = 0;
    if (::cudaGetDevice(&dev) != cudaSuccess) return 1;
    if (::cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
            cudaSuccess ||
        sms <= 0)
      sms = 32;
    int per_sm = 1;
    if (::cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, kKernel, threads, dyn_smem) != cudaSuccess ||
        per_sm < 1)
      per_sm = 1;
    return per_sm * sms;
  }();
  return r;
}

// SM count of the current device (cached once).
inline int sm_count_once() {
  static const int sms = [] {
    int dev = 0, sms = 0;
    if (::cudaGetDevice(&dev) != cudaSuccess) return 104;
    if (::cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
            cudaSuccess ||
        sms <= 0)
      sms = 104;
    return sms;
  }();
  return sms;
}

// Occupancy at one FIXED dynamic smem size, cached per (kernel, size).
// Register/CTA limits can cap co-residency below what smem alone allows,
// so callers verify the block count instead of assuming it.
template <auto kKernel, int kDynSmem>
inline int resident_blocks_fixed(int threads) {
  static const int r = [threads] {
    int dev = 0, sms = 0;
    if (::cudaGetDevice(&dev) != cudaSuccess) return 1;
    if (::cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
            cudaSuccess ||
        sms <= 0)
      sms = 32;
    int per_sm = 1;
    if (::cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, kKernel, threads, kDynSmem) != cudaSuccess ||
        per_sm < 1)
      per_sm = 1;
    return per_sm * sms;
  }();
  return r;
}

struct SplitPlan {
  uint32_t G;      // chunks per row
  uint32_t chunk;  // elements per chunk (<= cap)
  uint32_t grid;   // persistent grid size (<= resident capacity)
};

inline SplitPlan plan_split(int64_t num_columns, int64_t num_rows,
                            uint32_t cap, int resident) {
  // Only as many chunks per row as the smem chunk cap forces: a handful of
  // CTAs already saturates a row's HBM share, while every extra participant
  // serializes on the row's arrival counter (same-address atomics), so
  // over-splitting only adds barrier latency.
  uint32_t g = static_cast<uint32_t>((num_columns + cap - 1) / cap);
  if (g > (uint32_t)resident) g = (uint32_t)resident;
  if (g < 1) g = 1;
  const uint32_t chunk = static_cast<uint32_t>((num_columns + g - 1) / g);
  const uint64_t total = static_cast<uint64_t>(num_rows) * g;
  const uint32_t grid = static_cast<uint32_t>(
      std::min<uint64_t>(total, static_cast<uint64_t>(resident)));
  return SplitPlan{g, chunk, grid};
}

inline size_t split_state_bytes(int64_t num_rows) {
  const size_t state_bytes =
      ((sizeof(RowState) + 255) / 256) * 256;  // 256B aligned per row
  return state_bytes * static_cast<size_t>(num_rows);
}

}  // namespace

void apply_repetition_penalties_(
    torch::Tensor& logits,             // [num_seqs, vocab_size], in-place
    const torch::Tensor& prompt_mask,  // [num_seqs, vocab_size]
    const torch::Tensor& output_mask,  // [num_seqs, vocab_size]
    const torch::Tensor& repetition_penalties) {  // [num_seqs]
  TORCH_CHECK(logits.is_contiguous());
  TORCH_CHECK(prompt_mask.is_contiguous());
  TORCH_CHECK(output_mask.is_contiguous());
  TORCH_CHECK(repetition_penalties.is_contiguous());

  int vocab_size = logits.size(-1);
  int num_seqs = logits.size(0);

  if (num_seqs == 0) return;

  // Get number of SMs on the current device
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                         logits.get_device());

  // Compute tile_num and tile_size
  int tile_num =
      std::min(vocab_size, std::max(1, (sms + num_seqs - 1) / num_seqs));
  int tile_size = (vocab_size + tile_num - 1) / tile_num;

  // Each block handles one sequence and a tile of vocab
  dim3 grid(num_seqs, tile_num);
  dim3 block(std::min(tile_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      logits.scalar_type(), "apply_repetition_penalties_kernel", [&] {
        vllm::apply_repetition_penalties_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                logits.data_ptr<scalar_t>(), prompt_mask.data_ptr<bool>(),
                output_mask.data_ptr<bool>(),
                repetition_penalties.data_ptr<scalar_t>(), num_seqs, vocab_size,
                tile_size);
      });
}

void top_k_per_row_decode(const torch::Tensor& logits, int64_t next_n,
                          const torch::Tensor& seqLens, torch::Tensor& indices,
                          int64_t numRows, int64_t stride0, int64_t stride1,
                          int64_t topK) {
  const at::cuda::CUDAGuard device_guard(logits.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t numColumns = logits.size(1);
  if (numRows <= 0 || topK <= 0 || numColumns <= 0) return;
  const int nn = std::max(1, static_cast<int>(next_n));
  const int seqLensIs2D = seqLens.dim() == 2 ? 1 : 0;
  const uint32_t cap = query_chunk_cap();
  const int sms = sm_count_once();


  // Path A stages the whole row in shared memory. From ~4K columns up, once
  // rows outnumber SMs, Path C's two light global passes win instead:
  // staged selection costs ~27.5K cycles/row at L=4096 (probed) versus ~19K
  // for Path C, and beyond ~5.2K columns staging confines Path A to one
  // block per SM.
  if (numColumns <= static_cast<int64_t>(cap) &&
      (numColumns <
           static_cast<int64_t>(vllm::topk_impl::kStagedHighRowsMinLen) ||
       numRows <= sms)) {
    // Path A: one block per row, single global pass.
    const size_t smem = static_cast<size_t>(numColumns) * sizeof(float);
    if (numColumns > 4096) {
      optin_smem_once<vllm::topk_impl::topk_row_one_shot_decode<1024>>(
          static_cast<int>(cap * sizeof(float)));
      vllm::topk_impl::topk_row_one_shot_decode<1024>
          <<<numRows, 1024, smem, stream>>>(
              logits.data_ptr<float>(), seqLens.data_ptr<int>(),
              indices.data_ptr<int>(), static_cast<int>(stride0),
              static_cast<int>(stride1), static_cast<int>(topK), nn,
              seqLensIs2D, static_cast<int>(numColumns));
    } else {
      optin_smem_once<vllm::topk_impl::topk_row_one_shot_decode<512>>(
          static_cast<int>(cap * sizeof(float)));
      vllm::topk_impl::topk_row_one_shot_decode<512>
          <<<numRows, 512, smem, stream>>>(
              logits.data_ptr<float>(), seqLens.data_ptr<int>(),
              indices.data_ptr<int>(), static_cast<int>(stride0),
              static_cast<int>(stride1), static_cast<int>(topK), nn,
              seqLensIs2D, static_cast<int>(numColumns));
    }
    return;
  }

  // Long rows (or occupancy-confined mid rows): Path C (two-pass global
  // radix, one block per row, no barriers, no workspace) for enough rows,
  // a <= 2-chunk split, or rows Path A cannot co-schedule; Path B
  // (persistent split-K) for small batches of very long rows.
  const uint32_t g_min = static_cast<uint32_t>(
      (numColumns + static_cast<int64_t>(cap) - 1) / cap);
  if (numColumns <= static_cast<int64_t>(cap) ||
      g_min <= kTwoPassMaxSplit || numRows >= kTwoPassMinRows) {
    // Round-0 width from the row length alone (see kTwoPassWideMinLen):
    // the 11-bit histogram (2048 bins) for long rows, the byte histogram
    // below, matching the two template instantiations.
    const bool wide = numColumns >= kTwoPassWideMinLen;
    // Candidate buffer (key, index pairs) as dynamic shared memory. With
    // many rows, shrink the pool so two blocks co-reside per SM — streaming
    // is latency-bound there and a second block nearly doubles per-SM
    // throughput. Beyond kTwoPassSmallPoolMaxLen only the wide variant's
    // thinned pivot bucket (worst measured ~1814 candidates at L=262144
    // K=1024, vs the 2560-entry pool) keeps two passes viable; top_k must
    // leave the overflow rounds' break-out room for cnt_gt; the hardware
    // check guards register-capped co-scheduling. Fewer rows keep the large
    // pool: one block per SM anyway, its headroom avoids any third read.
    constexpr int kThreadsC = 1024;
    constexpr size_t kSmemLarge =
        static_cast<size_t>(vllm::topk_impl::kCandCapC) * 2 * sizeof(uint32_t);
    constexpr size_t kSmemSmall = static_cast<size_t>(
        vllm::topk_impl::kCandCapCSmall) * 2 * sizeof(uint32_t);
    const int resident_small =
        wide ? resident_blocks_fixed<
                   vllm::topk_impl::topk_2pass_row_kernel<1024, false, true>,
                   static_cast<int>(kSmemSmall)>(kThreadsC)
             : resident_blocks_fixed<
                   vllm::topk_impl::topk_2pass_row_kernel<1024, false, false>,
                   static_cast<int>(kSmemSmall)>(kThreadsC);
    uint32_t cand_cap = vllm::topk_impl::kCandCapC;
    if (numRows > sms &&
        topK * 2 <= static_cast<int64_t>(vllm::topk_impl::kCandCapCSmall) &&
        (wide || numColumns <= kTwoPassSmallPoolMaxLen) &&
        resident_small >= 2 * sms) {
      cand_cap = vllm::topk_impl::kCandCapCSmall;
    }
    const size_t smem_c = static_cast<size_t>(cand_cap) * 2 * sizeof(uint32_t);
    // Opt in at the large size once: the attribute is per-kernel, and a
    // small-pool launch is still legal below the registered maximum.
    if (wide) {
      optin_smem_once<vllm::topk_impl::topk_2pass_row_kernel<1024, false, true>>(
          static_cast<int>(kSmemLarge));
      vllm::topk_impl::topk_2pass_row_kernel<1024, false, true>
          <<<static_cast<unsigned>(numRows), 1024, smem_c, stream>>>(
              logits.data_ptr<float>(), seqLens.data_ptr<int>(), nullptr,
              indices.data_ptr<int>(), static_cast<int>(stride0),
              static_cast<int>(stride1), static_cast<int>(topK), nn,
              seqLensIs2D, static_cast<int>(numColumns), cand_cap);
    } else {
      optin_smem_once<
          vllm::topk_impl::topk_2pass_row_kernel<1024, false, false>>(
          static_cast<int>(kSmemLarge));
      vllm::topk_impl::topk_2pass_row_kernel<1024, false, false>
          <<<static_cast<unsigned>(numRows), 1024, smem_c, stream>>>(
              logits.data_ptr<float>(), seqLens.data_ptr<int>(), nullptr,
              indices.data_ptr<int>(), static_cast<int>(stride0),
              static_cast<int>(stride1), static_cast<int>(topK), nn,
              seqLensIs2D, static_cast<int>(numColumns), cand_cap);
    }

    return;
  }

  // Path B: persistent split-K, or its barrier-free two-kernel variant
  // once the split runs multi-wave.
  constexpr int kThreads = 1024;
  if (static_cast<int64_t>(numRows) * g_min >= kSplitNoBarrierMinJobs) {

    uint32_t G = static_cast<uint32_t>(
        (2 * static_cast<int64_t>(sms) + numRows - 1) / numRows);
    if (G < 1u) G = 1u;
    if (G > 64u) G = 64u;
    uint32_t chunk = static_cast<uint32_t>(
        (numColumns + static_cast<int64_t>(G) - 1) /
        static_cast<int64_t>(G));
    chunk = (chunk + 3u) & ~3u;  // 16B-align each chunk's float4 window
    const int64_t cand = static_cast<int64_t>(numRows) * G * topK;
    auto ws = at::empty({2 * cand}, logits.options().dtype(torch::kInt32));
    int* ws_idx = ws.data_ptr<int>();
    uint32_t* ws_key = reinterpret_cast<uint32_t*>(ws_idx + cand);
    const int n_jobs = static_cast<int>(numRows) * static_cast<int>(G);
    vllm::topk_impl::topk_chunk_local_kernel<1024>
        <<<static_cast<unsigned>(n_jobs), kThreads, 0, stream>>>(
            logits.data_ptr<float>(), seqLens.data_ptr<int>(), ws_idx,
            ws_key, static_cast<int>(stride0), static_cast<int>(stride1),
            static_cast<int>(topK), nn, seqLensIs2D,
            static_cast<int>(numColumns), G, chunk);
    constexpr size_t kSmemM =
        static_cast<size_t>(vllm::topk_impl::kCandCapC) * 2 * sizeof(uint32_t);
    optin_smem_once<vllm::topk_impl::topk_merge_cand_kernel<1024>>(
        static_cast<int>(kSmemM));
    vllm::topk_impl::topk_merge_cand_kernel<1024>
        <<<static_cast<unsigned>(numRows), kThreads, kSmemM, stream>>>(
            ws_key, ws_idx, indices.data_ptr<int>(),
            static_cast<int>(static_cast<int64_t>(G) * topK),
            static_cast<int>(topK), vllm::topk_impl::kCandCapC);
    return;
  }
  const int resident = resident_blocks_once<vllm::topk_impl::topk_split_kernel<1024, false>>(
      kThreads, static_cast<int>(cap * sizeof(float)));
  // FIX: Check capacity and fallback to barrier-free split-K if exceeded
  const int64_t split_capacity = static_cast<int64_t>(cap) * resident;
  if (numColumns > split_capacity) {
    // Barrier-free split-K (Path B2) - no capacity limit
    const int sms = sm_count_once();
    uint32_t G = static_cast<uint32_t>(
        (2 * static_cast<int64_t>(sms) + numRows - 1) / numRows);
    if (G < 1u) G = 1u;
    if (G > 64u) G = 64u;
    uint32_t chunk = static_cast<uint32_t>(
        (numColumns + static_cast<int64_t>(G) - 1) /
        static_cast<int64_t>(G));
    chunk = (chunk + 3u) & ~3u;
    const int64_t cand = static_cast<int64_t>(numRows) * G * topK;
    auto ws = at::empty({2 * cand}, logits.options().dtype(torch::kInt32));
    int* ws_idx = ws.data_ptr<int>();
    uint32_t* ws_key = reinterpret_cast<uint32_t*>(ws_idx + cand);
    const int n_jobs = static_cast<int>(numRows) * static_cast<int>(G);
    vllm::topk_impl::topk_chunk_local_kernel<1024>
        <<<static_cast<unsigned>(n_jobs), kThreads, 0, stream>>>(
            logits.data_ptr<float>(), seqLens.data_ptr<int>(), ws_idx,
            ws_key, static_cast<int>(stride0), static_cast<int>(stride1),
            static_cast<int>(topK), nn, seqLensIs2D,
            static_cast<int>(numColumns), G, chunk);
    constexpr size_t kSmemM =
        static_cast<size_t>(vllm::topk_impl::kCandCapC) * 2 * sizeof(uint32_t);
    optin_smem_once<vllm::topk_impl::topk_merge_cand_kernel<1024>>(
        static_cast<int>(kSmemM));
    vllm::topk_impl::topk_merge_cand_kernel<1024>
        <<<static_cast<unsigned>(numRows), kThreads, kSmemM, stream>>>(
            ws_key, ws_idx, indices.data_ptr<int>(),
            static_cast<int>(static_cast<int64_t>(G) * topK),
            static_cast<int>(topK), vllm::topk_impl::kCandCapC);
    return;
  }
  const SplitPlan plan = plan_split(numColumns, numRows, cap, resident);
  auto ws = at::zeros({(int64_t)split_state_bytes(numRows)},
                      logits.options().dtype(torch::kByte));
  optin_smem_once<vllm::topk_impl::topk_split_kernel<1024, false>>(
      static_cast<int>(cap * sizeof(float)));
  vllm::topk_impl::topk_split_kernel<1024, false>
      <<<plan.grid, kThreads, static_cast<size_t>(plan.chunk) * sizeof(float),
         stream>>>(
          logits.data_ptr<float>(), seqLens.data_ptr<int>(), nullptr,
          indices.data_ptr<int>(), static_cast<int>(stride0),
          static_cast<int>(stride1), static_cast<int>(topK), nn, seqLensIs2D,
          static_cast<int>(numColumns), static_cast<int>(numRows), plan.G,
          plan.chunk, reinterpret_cast<RowState*>(ws.data_ptr()));
}


void top_k_per_row_prefill(const torch::Tensor& logits,
                           const torch::Tensor& rowStarts,
                           const torch::Tensor& rowEnds, torch::Tensor& indices,
                           int64_t numRows, int64_t stride0, int64_t stride1,
                           int64_t topK) {
  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kNumThreadsPerBlock = 512;
  const at::cuda::CUDAGuard device_guard(logits.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Fast single-pass path: every row fits in one block (rowLen <= 32768).
  // Tier the block size by columns so small rows use smaller blocks (higher
  // occupancy, less per-block fixed cost); a smaller stage cap follows.
  if (logits.size(1) <= vllm::kTopkSbCap) {
  
    if (topK >= vllm::kTopkSplitTopKMin &&
        topK <= vllm::kTopkSplitTopKMax && logits.size(1) >= vllm::kTopkSplitColsMin &&
        numRows >= vllm::kTopkSplitRowsMin) {
      // Two-kernel split (same path as decode): K1 emit bin>tb straight out,
      // K2 refine ==tb with one warp per row, K3 rare overflow fallback.
      // Prefill passes row boundaries via rowStarts/rowEnds (isDecode=0).
      const auto stageKey = torch::empty(
          {numRows, vllm::kTopkSplitCap},
          torch::dtype(torch::kUInt32).device(logits.device()));
      const auto stageIdx = torch::empty(
          {numRows, vllm::kTopkSplitCap},
          torch::dtype(torch::kInt32).device(logits.device()));
      const auto rowG = torch::empty(
          {numRows}, torch::dtype(torch::kInt32).device(logits.device()));
      const auto rowN = torch::empty(
          {numRows}, torch::dtype(torch::kInt32).device(logits.device()));
      const int kNumRows = static_cast<int>(numRows);
      const int kStride0 = static_cast<int>(stride0);
      const int kStride1 = static_cast<int>(stride1);
      const int kTopK = static_cast<int>(topK);
      if (logits.size(1) <= 4096) {
        vllm::launchTopkSplit<256, 4>(
            logits.data_ptr<float>(), nullptr, rowStarts.data_ptr<int>(),
            rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
            stageKey.data_ptr<uint32_t>(), stageIdx.data_ptr<int>(),
            rowG.data_ptr<int>(), rowN.data_ptr<int>(), kNumRows, kStride0,
            kStride1, kTopK, 0, 0, 0);
      } else if (logits.size(1) <= 8192) {
        vllm::launchTopkSplit<256, 8>(
            logits.data_ptr<float>(), nullptr, rowStarts.data_ptr<int>(),
            rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
            stageKey.data_ptr<uint32_t>(), stageIdx.data_ptr<int>(),
            rowG.data_ptr<int>(), rowN.data_ptr<int>(), kNumRows, kStride0,
            kStride1, kTopK, 0, 0, 0);
      } else if (logits.size(1) <= 16384) {
        vllm::launchTopkSplit<512, 8>(
            logits.data_ptr<float>(), nullptr, rowStarts.data_ptr<int>(),
            rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
            stageKey.data_ptr<uint32_t>(), stageIdx.data_ptr<int>(),
            rowG.data_ptr<int>(), rowN.data_ptr<int>(), kNumRows, kStride0,
            kStride1, kTopK, 0, 0, 0);
      } else {
        vllm::launchTopkSplit<vllm::kTopkSbBlockSize, vllm::kTopkSbVecsPerThread>(
            logits.data_ptr<float>(), nullptr, rowStarts.data_ptr<int>(),
            rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
            stageKey.data_ptr<uint32_t>(), stageIdx.data_ptr<int>(),
            rowG.data_ptr<int>(), rowN.data_ptr<int>(), kNumRows, kStride0,
            kStride1, kTopK, 0, 0, 0);
      }
      return;
    }
    // topK 3..7: single-block kernel.
    if (logits.size(1) <= 4096) {
      vllm::topKPerRowSingleBlockKernel<256, 4>
          <<<numRows, 256, 0, stream>>>(
              logits.data_ptr<float>(), nullptr, rowStarts.data_ptr<int>(),
              rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
              static_cast<int>(numRows), static_cast<int>(stride0),
              static_cast<int>(stride1), static_cast<int>(topK), 0, 0, 0);
    } else if (logits.size(1) <= 8192) {
      vllm::topKPerRowSingleBlockKernel<256, 8>
          <<<numRows, 256, 0, stream>>>(
              logits.data_ptr<float>(), nullptr, rowStarts.data_ptr<int>(),
              rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
              static_cast<int>(numRows), static_cast<int>(stride0),
              static_cast<int>(stride1), static_cast<int>(topK), 0, 0, 0);
    } else if (logits.size(1) <= 16384) {
      vllm::topKPerRowSingleBlockKernel<512, 8>
          <<<numRows, 512, 0, stream>>>(
              logits.data_ptr<float>(), nullptr, rowStarts.data_ptr<int>(),
              rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
              static_cast<int>(numRows), static_cast<int>(stride0),
              static_cast<int>(stride1), static_cast<int>(topK), 0, 0, 0);
    } else {
      vllm::topKPerRowSingleBlockKernel<vllm::kTopkSbBlockSize,
                                        vllm::kTopkSbVecsPerThread>
          <<<numRows, vllm::kTopkSbBlockSize, 0, stream>>>(
              logits.data_ptr<float>(), nullptr, rowStarts.data_ptr<int>(),
              rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
              static_cast<int>(numRows), static_cast<int>(stride0),
              static_cast<int>(stride1), static_cast<int>(topK), 0, 0, 0);
    }
    return;
  }

  int numInsertionBlocks =
      std::min(static_cast<int>(numRows), kSortingAlgorithmThreshold);
  vllm::topKPerRowPrefill<kNumThreadsPerBlock, false>
      <<<numInsertionBlocks, kNumThreadsPerBlock, topK * sizeof(int32_t),
         stream>>>(logits.data_ptr<float>(), rowStarts.data_ptr<int>(),
                   rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
                   static_cast<int>(stride0), static_cast<int>(stride1),
                   static_cast<int>(topK), 0);

  if (numRows > kSortingAlgorithmThreshold) {
    int numRadixBlocks = numRows - kSortingAlgorithmThreshold;
    vllm::topKPerRowPrefill<kNumThreadsPerBlock, true>
        <<<numRadixBlocks, kNumThreadsPerBlock, topK * sizeof(int32_t),
           stream>>>(logits.data_ptr<float>(), rowStarts.data_ptr<int>(),
                     rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
                     static_cast<int>(stride0), static_cast<int>(stride1),
                     static_cast<int>(topK), kSortingAlgorithmThreshold);
  }
}
