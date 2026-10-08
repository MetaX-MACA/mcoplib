// SPDX-License-Identifier: Apache-2.0
//
// AOT implementation of SGLang's kpool_topk_transform operator for mcoplib.
//
// Each block selects the highest-scoring pool groups from one row, expands
// them to token indices, and applies an optional offset or page-table mapping.
// The normal path uses staged radix selection; oversized radix buckets fall
// back to an exact full-row rescan. Long prefill rows cache their coarse radix
// byte in shared memory to avoid a second full FP32 score scan.

#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace {

constexpr int kDefaultThreads = 1024;
constexpr int kMediumThreads = 512;
constexpr int kLongStagingSize = 4096;
constexpr std::size_t kLongDynamicSmem =
    2 * kLongStagingSize * sizeof(int32_t);  // 32 KiB
// Short rows cannot produce more candidates than their width, so a dedicated
// instance avoids reserving the long-row staging allocation.
constexpr int kShortStagingSize = 1024;
constexpr std::size_t kShortDynamicSmem =
    2 * kShortStagingSize * sizeof(int32_t);  // 8 KiB
constexpr int64_t kShortStagingMinBatch = 32;
constexpr int64_t kCoarseBinCacheMaxScoreWidth =
    kLongStagingSize * static_cast<int64_t>(sizeof(int32_t));
constexpr int64_t kCoarseBinCacheMinBatch = 32;
constexpr int kRadix = 256;
constexpr int64_t kMediumMaxScoreWidth = 2048;
// The GLM-5 63-Ki workloads have score widths 16126 and 16383.
constexpr int64_t kVec8MinScoreWidth = 16126;
constexpr int32_t kVec8Stride = 2;  // Two float4 loads per thread and iteration.

struct KpoolTopKParams {
  const float* __restrict__ score;
  const int32_t* __restrict__ lengths;
  const int32_t* __restrict__ row_starts;
  int32_t* __restrict__ output;
  const int32_t* __restrict__ page_table;
  const int32_t* __restrict__ page_table_row_index;
  const int32_t* __restrict__ topk_indices_offset;
  const int32_t* __restrict__ seq_lens;
  int64_t score_stride;
  int64_t output_stride;
  int64_t page_table_stride;
  int32_t pool_size;
  int32_t token_topk;
  int32_t out_cols;
};

// Convert a score to an order-preserving coarse radix byte.
__device__ __forceinline__ uint8_t ordered_fp16_high_byte(float value) {
  const __half half_value = __float2half_rn(value);
  const uint16_t bits = __half_as_ushort(half_value);
  const uint16_t key = (bits & 0x8000u)
      ? static_cast<uint16_t>(~bits)
      : static_cast<uint16_t>(bits | 0x8000u);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ void ordered_fp16_high_bytes(
    float first, float second, uint8_t& first_key, uint8_t& second_key) {
  const __half2 half_values = __floats2half2_rn(first, second);
  const uint16_t first_bits = __half_as_ushort(__low2half(half_values));
  const uint16_t second_bits = __half_as_ushort(__high2half(half_values));
  const uint16_t first_ordered =
      (first_bits & 0x8000u) ? static_cast<uint16_t>(~first_bits)
                             : static_cast<uint16_t>(first_bits | 0x8000u);
  const uint16_t second_ordered =
      (second_bits & 0x8000u) ? static_cast<uint16_t>(~second_bits)
                              : static_cast<uint16_t>(second_bits | 0x8000u);
  first_key = static_cast<uint8_t>(first_ordered >> 8);
  second_key = static_cast<uint8_t>(second_ordered >> 8);
}

__device__ __forceinline__ uint32_t ordered_fp32_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

struct Vec8Layout {
  const float4* values;
  int32_t prefix;
  int32_t vec4_length;
  int32_t tail;
  int32_t vec8_length;
};

__device__ __forceinline__ Vec8Layout make_vec8_layout(
    const float* row_input, int32_t length) {
  const uintptr_t row_address = reinterpret_cast<uintptr_t>(row_input);
  const int32_t prefix_unclamped = static_cast<int32_t>(
      (alignof(float4) - (row_address & (alignof(float4) - 1))) / sizeof(float));
  const int32_t prefix =
      (row_address & (alignof(float4) - 1)) == 0
      ? 0
      : (length < prefix_unclamped ? length : prefix_unclamped);
  const int32_t vec4_length = (length - prefix) / 4;
  return {
      reinterpret_cast<const float4*>(row_input + prefix),
      prefix,
      vec4_length,
      prefix + vec4_length * 4,
      vec4_length / kVec8Stride,
  };
}

__device__ __forceinline__ int32_t transform_token(
    int32_t raw_token,
    const int32_t* __restrict__ page_table_row,
    const int32_t* __restrict__ topk_indices_offset,
    int32_t offset) {
  if (page_table_row != nullptr) {
    return page_table_row[raw_token];
  }
  if (topk_indices_offset != nullptr) {
    return raw_token + offset;
  }
  return raw_token;
}

template <int K, int Threads, bool UseVec8, int StagingSize, bool CacheCoarseBins>
__device__ void radix_topk_exact_with_staged_fast_path(
    const float* __restrict__ input,
    int32_t* __restrict__ selected_indices,
    int32_t row_start,
    int32_t length) {
  static_assert(Threads >= kRadix + 1, "radix histogram requires at least 257 threads");
  static_assert(!UseVec8 || Threads == kDefaultThreads, "vec8 is tuned for 1024 threads");
  static_assert(StagingSize >= K, "staging buffer must hold at least K candidates");
  static_assert(!CacheCoarseBins || UseVec8, "coarse-bin caching requires vec8");

  // Two staged buffers handle normal score distributions. In the coarse-cache
  // specialization, the first buffer is
  // placed in the upper half while the lower half temporarily holds one radix
  // byte per score; the lower half becomes the second buffer after coarse
  // classification. If a threshold bucket exceeds this kernel instance's
  // StagingSize, an exact full-row rescan prevents candidate truncation.
  alignas(128) __shared__ uint32_t histogram_buffer[2][kRadix + 128];
  alignas(128) __shared__ int32_t output_counter;
  alignas(128) __shared__ int32_t threshold_bin_id;
  alignas(128) __shared__ int32_t num_input[2];
  alignas(128) __shared__ int32_t last_remain;

  // State used only by the exact overflow fallback.
  alignas(128) __shared__ uint32_t prefix_mask;
  alignas(128) __shared__ uint32_t prefix_value;
  alignas(128) __shared__ int32_t candidate_count;
  alignas(128) __shared__ int32_t next_remain;
  alignas(128) __shared__ int32_t tie_remaining;

  extern __shared__ int32_t staged_indices[];

  const int32_t tid = threadIdx.x;
  const float* row_input = input + row_start;
  auto& histogram = histogram_buffer[0];
  auto* coarse_bin_cache = reinterpret_cast<uint8_t*>(staged_indices);
  const auto staged_buffer = [&](int32_t buffer_index) {
    if constexpr (CacheCoarseBins) {
      // Buffer 0 occupies the upper half while the lower half caches one byte
      // per score. After coarse classification, the cache is dead and becomes
      // refinement buffer 1.
      return staged_indices + (buffer_index == 0 ? StagingSize : 0);
    }
    return staged_indices + buffer_index * StagingSize;
  };
  int32_t remain = K;

  const auto run_suffix_sum = [&] {
#pragma unroll 8
    for (int round = 0; round < 8; ++round) {
      const int distance = 1 << round;
      const int source = round & 1;
      if (tid < kRadix) {
        uint32_t value = histogram_buffer[source][tid];
        if (tid + distance < kRadix) {
          value += histogram_buffer[source][tid + distance];
        }
        histogram_buffer[source ^ 1][tid] = value;
      }
      __syncthreads();
    }
  };

  // Stage 1: coarse 8-bit histogram over the valid score region.
  if (tid < kRadix + 1) {
    histogram[tid] = 0;
  }
  __syncthreads();

  if constexpr (UseVec8) {
    const Vec8Layout layout = make_vec8_layout(row_input, length);
    for (int32_t index = tid; index < layout.prefix; index += Threads) {
      const uint8_t coarse_bin = ordered_fp16_high_byte(row_input[index]);
      if constexpr (CacheCoarseBins) {
        coarse_bin_cache[index] = coarse_bin;
      }
      ::atomicAdd(&histogram[coarse_bin], 1u);
    }
    for (int32_t vec_index = tid; vec_index < layout.vec8_length;
         vec_index += Threads) {
      const int32_t base = vec_index * kVec8Stride;
      const float4 first = layout.values[base];
      const float4 second = layout.values[base + 1];
      uint8_t key0, key1, key2, key3, key4, key5, key6, key7;
      ordered_fp16_high_bytes(first.x, first.y, key0, key1);
      ordered_fp16_high_bytes(first.z, first.w, key2, key3);
      ordered_fp16_high_bytes(second.x, second.y, key4, key5);
      ordered_fp16_high_bytes(second.z, second.w, key6, key7);
      if constexpr (CacheCoarseBins) {
        const int32_t index = layout.prefix + base * 4;
        coarse_bin_cache[index] = key0;
        coarse_bin_cache[index + 1] = key1;
        coarse_bin_cache[index + 2] = key2;
        coarse_bin_cache[index + 3] = key3;
        coarse_bin_cache[index + 4] = key4;
        coarse_bin_cache[index + 5] = key5;
        coarse_bin_cache[index + 6] = key6;
        coarse_bin_cache[index + 7] = key7;
      }
      ::atomicAdd(&histogram[key0], 1u);
      ::atomicAdd(&histogram[key1], 1u);
      ::atomicAdd(&histogram[key2], 1u);
      ::atomicAdd(&histogram[key3], 1u);
      ::atomicAdd(&histogram[key4], 1u);
      ::atomicAdd(&histogram[key5], 1u);
      ::atomicAdd(&histogram[key6], 1u);
      ::atomicAdd(&histogram[key7], 1u);
    }
    for (int32_t vec_index = layout.vec8_length * kVec8Stride + tid;
         vec_index < layout.vec4_length;
         vec_index += Threads) {
      const float4 values = layout.values[vec_index];
      uint8_t key0, key1, key2, key3;
      ordered_fp16_high_bytes(values.x, values.y, key0, key1);
      ordered_fp16_high_bytes(values.z, values.w, key2, key3);
      if constexpr (CacheCoarseBins) {
        const int32_t index = layout.prefix + vec_index * 4;
        coarse_bin_cache[index] = key0;
        coarse_bin_cache[index + 1] = key1;
        coarse_bin_cache[index + 2] = key2;
        coarse_bin_cache[index + 3] = key3;
      }
      ::atomicAdd(&histogram[key0], 1u);
      ::atomicAdd(&histogram[key1], 1u);
      ::atomicAdd(&histogram[key2], 1u);
      ::atomicAdd(&histogram[key3], 1u);
    }
    for (int32_t index = layout.tail + tid; index < length; index += Threads) {
      const uint8_t coarse_bin = ordered_fp16_high_byte(row_input[index]);
      if constexpr (CacheCoarseBins) {
        coarse_bin_cache[index] = coarse_bin;
      }
      ::atomicAdd(&histogram[coarse_bin], 1u);
    }
  } else {
    for (int32_t index = tid; index < length; index += Threads) {
      ::atomicAdd(&histogram[ordered_fp16_high_byte(row_input[index])], 1u);
    }
  }
  __syncthreads();

  run_suffix_sum();
  if (tid < kRadix && histogram[tid] > static_cast<uint32_t>(remain) &&
      histogram[tid + 1] <= static_cast<uint32_t>(remain)) {
    threshold_bin_id = tid;
    num_input[0] = 0;
    output_counter = 0;
  }
  __syncthreads();

  const int32_t coarse_threshold = threshold_bin_id;
  remain -= static_cast<int32_t>(histogram[coarse_threshold + 1]);

  if (remain == 0) {
    const auto append_if_above_threshold = [&](int32_t index, uint8_t coarse_bin) {
      if (coarse_bin > coarse_threshold) {
        const int32_t position = ::atomicAdd(&output_counter, 1);
        selected_indices[position] = index;
      }
    };

    if constexpr (CacheCoarseBins) {
      for (int32_t index = tid; index < length; index += Threads) {
        append_if_above_threshold(index, coarse_bin_cache[index]);
      }
    } else if constexpr (UseVec8) {
      const Vec8Layout layout = make_vec8_layout(row_input, length);
      for (int32_t index = tid; index < layout.prefix; index += Threads) {
        append_if_above_threshold(index, ordered_fp16_high_byte(row_input[index]));
      }
      for (int32_t vec_index = tid; vec_index < layout.vec8_length;
           vec_index += Threads) {
        const int32_t base = vec_index * kVec8Stride;
        const float4 first = layout.values[base];
        const float4 second = layout.values[base + 1];
        uint8_t key0, key1, key2, key3, key4, key5, key6, key7;
        ordered_fp16_high_bytes(first.x, first.y, key0, key1);
        ordered_fp16_high_bytes(first.z, first.w, key2, key3);
        ordered_fp16_high_bytes(second.x, second.y, key4, key5);
        ordered_fp16_high_bytes(second.z, second.w, key6, key7);
        const int32_t index = layout.prefix + base * 4;
        append_if_above_threshold(index, key0);
        append_if_above_threshold(index + 1, key1);
        append_if_above_threshold(index + 2, key2);
        append_if_above_threshold(index + 3, key3);
        append_if_above_threshold(index + 4, key4);
        append_if_above_threshold(index + 5, key5);
        append_if_above_threshold(index + 6, key6);
        append_if_above_threshold(index + 7, key7);
      }
      for (int32_t vec_index = layout.vec8_length * kVec8Stride + tid;
           vec_index < layout.vec4_length;
           vec_index += Threads) {
        const float4 values = layout.values[vec_index];
        uint8_t key0, key1, key2, key3;
        ordered_fp16_high_bytes(values.x, values.y, key0, key1);
        ordered_fp16_high_bytes(values.z, values.w, key2, key3);
        const int32_t index = layout.prefix + vec_index * 4;
        append_if_above_threshold(index, key0);
        append_if_above_threshold(index + 1, key1);
        append_if_above_threshold(index + 2, key2);
        append_if_above_threshold(index + 3, key3);
      }
      for (int32_t index = layout.tail + tid; index < length; index += Threads) {
        append_if_above_threshold(index, ordered_fp16_high_byte(row_input[index]));
      }
    } else {
      for (int32_t index = tid; index < length; index += Threads) {
        append_if_above_threshold(index, ordered_fp16_high_byte(row_input[index]));
      }
    }
    __syncthreads();
    return;
  }

  // All threads must consume the coarse suffix sum before it is cleared.
  __syncthreads();

  // Collect the threshold bucket and build the first FP32-byte histogram.
  if (tid < kRadix + 1) {
    histogram[tid] = 0;
  }
  __syncthreads();

  const auto append_or_stage_candidate = [&](int32_t index, float value, uint8_t coarse_bin) {
    if (coarse_bin > coarse_threshold) {
      const int32_t position = ::atomicAdd(&output_counter, 1);
      selected_indices[position] = index;
    } else if (coarse_bin == coarse_threshold) {
      const int32_t position = ::atomicAdd(&num_input[0], 1);
      if (C10_LIKELY(position < StagingSize)) {
        staged_buffer(0)[position] = index;
        const int32_t sub_bin = static_cast<int32_t>((ordered_fp32_key(value) >> 24) & 0xFFu);
        ::atomicAdd(&histogram[sub_bin], 1u);
      }
    }
  };

  if constexpr (CacheCoarseBins) {
    for (int32_t index = tid; index < length; index += Threads) {
      const uint8_t coarse_bin = coarse_bin_cache[index];
      if (coarse_bin > coarse_threshold) {
        const int32_t position = ::atomicAdd(&output_counter, 1);
        selected_indices[position] = index;
      } else if (coarse_bin == coarse_threshold) {
        const float value = row_input[index];
        const int32_t position = ::atomicAdd(&num_input[0], 1);
        if (C10_LIKELY(position < StagingSize)) {
          staged_buffer(0)[position] = index;
          const int32_t sub_bin =
              static_cast<int32_t>((ordered_fp32_key(value) >> 24) & 0xFFu);
          ::atomicAdd(&histogram[sub_bin], 1u);
        }
      }
    }
  } else if constexpr (UseVec8) {
    const Vec8Layout layout = make_vec8_layout(row_input, length);
    for (int32_t index = tid; index < layout.prefix; index += Threads) {
      const float value = row_input[index];
      append_or_stage_candidate(index, value, ordered_fp16_high_byte(value));
    }
    for (int32_t vec_index = tid; vec_index < layout.vec8_length;
         vec_index += Threads) {
      const int32_t base = vec_index * kVec8Stride;
      const float4 first = layout.values[base];
      const float4 second = layout.values[base + 1];
      uint8_t key0, key1, key2, key3, key4, key5, key6, key7;
      ordered_fp16_high_bytes(first.x, first.y, key0, key1);
      ordered_fp16_high_bytes(first.z, first.w, key2, key3);
      ordered_fp16_high_bytes(second.x, second.y, key4, key5);
      ordered_fp16_high_bytes(second.z, second.w, key6, key7);
      const int32_t index = layout.prefix + base * 4;
      append_or_stage_candidate(index, first.x, key0);
      append_or_stage_candidate(index + 1, first.y, key1);
      append_or_stage_candidate(index + 2, first.z, key2);
      append_or_stage_candidate(index + 3, first.w, key3);
      append_or_stage_candidate(index + 4, second.x, key4);
      append_or_stage_candidate(index + 5, second.y, key5);
      append_or_stage_candidate(index + 6, second.z, key6);
      append_or_stage_candidate(index + 7, second.w, key7);
    }
    for (int32_t vec_index = layout.vec8_length * kVec8Stride + tid;
         vec_index < layout.vec4_length;
         vec_index += Threads) {
      const float4 values = layout.values[vec_index];
      uint8_t key0, key1, key2, key3;
      ordered_fp16_high_bytes(values.x, values.y, key0, key1);
      ordered_fp16_high_bytes(values.z, values.w, key2, key3);
      const int32_t index = layout.prefix + vec_index * 4;
      append_or_stage_candidate(index, values.x, key0);
      append_or_stage_candidate(index + 1, values.y, key1);
      append_or_stage_candidate(index + 2, values.z, key2);
      append_or_stage_candidate(index + 3, values.w, key3);
    }
    for (int32_t index = layout.tail + tid; index < length; index += Threads) {
      const float value = row_input[index];
      append_or_stage_candidate(index, value, ordered_fp16_high_byte(value));
    }
  } else {
    for (int32_t index = tid; index < length; index += Threads) {
      const float value = row_input[index];
      append_or_stage_candidate(index, value, ordered_fp16_high_byte(value));
    }
  }
  __syncthreads();

  // Exact fallback for adversarial/dense score distributions.  Every refine
  // round rescans the original row while narrowing an FP32 prefix.  This path
  // is slower but cannot drop candidates when the staged buffers overflow.
  if (num_input[0] > StagingSize) {
    if (tid == 0) {
      prefix_mask = 0;
      prefix_value = 0;
    }
    __syncthreads();

#pragma unroll 4
    for (int round = 0; round < 4; ++round) {
      if (tid < kRadix + 1) {
        histogram[tid] = 0;
      }
      if (tid == 0) {
        candidate_count = 0;
      }
      __syncthreads();

      const int shift = 24 - round * 8;
      for (int32_t index = tid; index < length; index += Threads) {
        const float value = input[row_start + index];
        if (ordered_fp16_high_byte(value) != coarse_threshold) {
          continue;
        }
        const uint32_t key = ordered_fp32_key(value);
        if (round > 0 && (key & prefix_mask) != prefix_value) {
          continue;
        }
        ::atomicAdd(&candidate_count, 1);
        ::atomicAdd(&histogram[(key >> shift) & 0xFFu], 1u);
      }
      __syncthreads();

      // The whole current prefix is required; no further radix refinement is
      // necessary and every matching index can be emitted directly.
      if (candidate_count == remain) {
        for (int32_t index = tid; index < length; index += Threads) {
          const float value = input[row_start + index];
          if (ordered_fp16_high_byte(value) != coarse_threshold) {
            continue;
          }
          const uint32_t key = ordered_fp32_key(value);
          if (round == 0 || (key & prefix_mask) == prefix_value) {
            const int32_t position = ::atomicAdd(&output_counter, 1);
            selected_indices[position] = index;
          }
        }
        __syncthreads();
        return;
      }

      run_suffix_sum();
      if (tid < kRadix && histogram[tid] > static_cast<uint32_t>(remain) &&
          histogram[tid + 1] <= static_cast<uint32_t>(remain)) {
        threshold_bin_id = tid;
        next_remain = remain - static_cast<int32_t>(histogram[tid + 1]);
        tie_remaining = next_remain;
      }
      __syncthreads();

      const int32_t refine_threshold = threshold_bin_id;
      for (int32_t index = tid; index < length; index += Threads) {
        const float value = input[row_start + index];
        if (ordered_fp16_high_byte(value) != coarse_threshold) {
          continue;
        }
        const uint32_t key = ordered_fp32_key(value);
        if (round > 0 && (key & prefix_mask) != prefix_value) {
          continue;
        }
        const int32_t bin = static_cast<int32_t>((key >> shift) & 0xFFu);
        if (bin > refine_threshold) {
          const int32_t position = ::atomicAdd(&output_counter, 1);
          selected_indices[position] = index;
        } else if (round == 3 && bin == refine_threshold) {
          const int32_t slot = ::atomicAdd(&tie_remaining, -1);
          if (slot > 0) {
            const int32_t position = ::atomicAdd(&output_counter, 1);
            selected_indices[position] = index;
          }
        }
      }
      __syncthreads();

      remain = next_remain;
      if (remain == 0 || round == 3) {
        return;
      }

      if (tid == 0) {
        prefix_mask |= 0xFFu << shift;
        prefix_value |= static_cast<uint32_t>(refine_threshold) << shift;
      }
      __syncthreads();
    }
    return;
  }

  // Normal staged path: refine only the threshold candidates in shared
  // memory, matching the low-overhead behavior of the supplied JIT kernel.
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    const int buffer_index = round & 1;
    const int32_t current_count = num_input[buffer_index];

    if (current_count == remain) {
      const int32_t output_base = output_counter;
      for (int32_t index = tid; index < current_count; index += Threads) {
        selected_indices[output_base + index] = staged_buffer(buffer_index)[index];
      }
      __syncthreads();
      return;
    }

    run_suffix_sum();
    if (tid < kRadix && histogram[tid] > static_cast<uint32_t>(remain) &&
        histogram[tid + 1] <= static_cast<uint32_t>(remain)) {
      threshold_bin_id = tid;
      num_input[buffer_index ^ 1] = 0;
      last_remain = remain - static_cast<int32_t>(histogram[tid + 1]);
    }
    __syncthreads();

    const int32_t refine_threshold = threshold_bin_id;
    remain -= static_cast<int32_t>(histogram[refine_threshold + 1]);
    const int shift = 24 - round * 8;

    if (remain == 0) {
      for (int32_t index = tid; index < current_count; index += Threads) {
        const int32_t score_index = staged_buffer(buffer_index)[index];
        const int32_t bin =
            static_cast<int32_t>((ordered_fp32_key(input[row_start + score_index]) >> shift) & 0xFFu);
        if (bin > refine_threshold) {
          const int32_t position = ::atomicAdd(&output_counter, 1);
          selected_indices[position] = score_index;
        }
      }
      __syncthreads();
      return;
    }

    // All threads must consume the suffix sum before it is cleared.
    __syncthreads();

    if (tid < kRadix + 1) {
      histogram[tid] = 0;
    }
    __syncthreads();
    for (int32_t index = tid; index < current_count; index += Threads) {
      const int32_t score_index = staged_buffer(buffer_index)[index];
      const float value = input[row_start + score_index];
      const uint32_t key = ordered_fp32_key(value);
      const int32_t bin = static_cast<int32_t>((key >> shift) & 0xFFu);
      if (bin > refine_threshold) {
        const int32_t position = ::atomicAdd(&output_counter, 1);
        selected_indices[position] = score_index;
      } else if (bin == refine_threshold) {
        if (round == 3) {
          const int32_t slot = ::atomicAdd(&last_remain, -1);
          if (slot > 0) {
            const int32_t position = ::atomicAdd(&output_counter, 1);
            selected_indices[position] = score_index;
          }
        } else {
          const int32_t position = ::atomicAdd(&num_input[buffer_index ^ 1], 1);
          if (C10_LIKELY(position < StagingSize)) {
            staged_buffer(buffer_index ^ 1)[position] = score_index;
            const int32_t sub_bin = static_cast<int32_t>((key >> (shift - 8)) & 0xFFu);
            ::atomicAdd(&histogram[sub_bin], 1u);
          }
        }
      }
    }
    __syncthreads();
  }
}

template <int K, int Threads, bool UseVec8, int StagingSize, bool CacheCoarseBins>
__global__ __launch_bounds__(Threads) void kpool_topk_transform_kernel(
    KpoolTopKParams params) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int32_t tid = threadIdx.x;
  const int32_t length = params.lengths[row];
  const int32_t row_start = params.row_starts == nullptr ? 0 : params.row_starts[row];
  const float* score_row = params.score + row * params.score_stride;
  int32_t* output_row = params.output + row * params.output_stride;

  const int64_t page_table_row =
      params.page_table_row_index == nullptr ? row : static_cast<int64_t>(params.page_table_row_index[row]);
  const int32_t* page_table_entry =
      params.page_table == nullptr ? nullptr : params.page_table + page_table_row * params.page_table_stride;
  const int32_t offset = params.topk_indices_offset == nullptr ? 0 : params.topk_indices_offset[row];
  const bool append_tail = params.seq_lens != nullptr;
  const int32_t full_pool_token_len = length * params.pool_size;
  const int32_t history_len =
      full_pool_token_len < params.token_topk ? full_pool_token_len : params.token_topk;
  const int32_t tail_count = append_tail ? params.seq_lens[row] % params.pool_size : 0;

  // If every valid group is required, selection is unnecessary.
  if (length <= K) {
    // With no Top-K selection the history is the identity sequence.  Its tail
    // continues that same sequence, so both regions collapse into one bounds
    // check and no per-element division/modulo is required.
    const int32_t valid_columns = history_len + tail_count;
    if (page_table_entry == nullptr) {
      for (int32_t column = tid; column < params.out_cols; column += Threads) {
        output_row[column] = column < valid_columns ? column + offset : -1;
      }
    } else {
      for (int32_t column = tid; column < params.out_cols; column += Threads) {
        output_row[column] =
            column < valid_columns ? page_table_entry[column] : -1;
      }
    }
    return;
  }

  // Selecting K groups from K + 1 candidates only requires excluding the
  // single minimum-score group. Avoid the full multi-pass radix selection.
  if (length == K + 1) {
    alignas(8) __shared__ unsigned long long minimum_key_index;

    if (tid == 0) {
      minimum_key_index = ~0ULL;
    }
    __syncthreads();

    for (int32_t index = tid; index < length; index += Threads) {
      const uint32_t key = ordered_fp32_key(score_row[row_start + index]);
      const unsigned long long candidate =
          (static_cast<unsigned long long>(key) << 32) | static_cast<uint32_t>(index);
      ::atomicMin(&minimum_key_index, candidate);
    }
    __syncthreads();

    const int32_t excluded_group =
        static_cast<int32_t>(minimum_key_index & 0xFFFFFFFFULL);
    for (int32_t column = tid; column < params.out_cols; column += Threads) {
      if (column < history_len) {
        const int32_t group_rank = column / params.pool_size;
        const int32_t slot = column % params.pool_size;
        const int32_t group = group_rank + (group_rank >= excluded_group);
        const int32_t raw_token = group * params.pool_size + slot;
        output_row[column] = transform_token(
            raw_token, page_table_entry, params.topk_indices_offset, offset);
      } else if (append_tail && column < history_len + tail_count) {
        const int32_t raw_token = full_pool_token_len + column - history_len;
        output_row[column] = transform_token(
            raw_token, page_table_entry, params.topk_indices_offset, offset);
      } else {
        output_row[column] = -1;
      }
    }
    return;
  }

  __shared__ int32_t selected_groups[K];
  radix_topk_exact_with_staged_fast_path<
      K, Threads, UseVec8, StagingSize, CacheCoarseBins>(
      score_row, selected_groups, row_start, length);

  // GLM-5 fixes pool_size at four.  Specialize the output expansion so the
  // compiler can replace integer division/modulo with shifts and masks, and
  // hoist the uniform page-table/offset choice out of the element loop.
  if (params.pool_size == 4) {
    if (page_table_entry == nullptr) {
      for (int32_t column = tid; column < params.out_cols; column += Threads) {
        if (column < history_len) {
          const int32_t group_rank = column >> 2;
          const int32_t slot = column & 3;
          const int32_t raw_token = (selected_groups[group_rank] << 2) + slot;
          output_row[column] = raw_token + offset;
        } else if (append_tail && column < history_len + tail_count) {
          output_row[column] =
              full_pool_token_len + column - history_len + offset;
        } else {
          output_row[column] = -1;
        }
      }
    } else {
      for (int32_t column = tid; column < params.out_cols; column += Threads) {
        if (column < history_len) {
          const int32_t group_rank = column >> 2;
          const int32_t slot = column & 3;
          const int32_t raw_token = (selected_groups[group_rank] << 2) + slot;
          output_row[column] = page_table_entry[raw_token];
        } else if (append_tail && column < history_len + tail_count) {
          const int32_t raw_token =
              full_pool_token_len + column - history_len;
          output_row[column] = page_table_entry[raw_token];
        } else {
          output_row[column] = -1;
        }
      }
    }
    return;
  }

  for (int32_t column = tid; column < params.out_cols; column += Threads) {
    if (column < history_len) {
      const int32_t group_rank = column / params.pool_size;
      const int32_t slot = column % params.pool_size;
      const int32_t raw_token = selected_groups[group_rank] * params.pool_size + slot;
      output_row[column] = transform_token(
          raw_token, page_table_entry, params.topk_indices_offset, offset);
    } else if (append_tail && column < history_len + tail_count) {
      const int32_t raw_token = full_pool_token_len + column - history_len;
      output_row[column] = transform_token(
          raw_token, page_table_entry, params.topk_indices_offset, offset);
    } else {
      output_row[column] = -1;
    }
  }
}

template <auto* Kernel, std::size_t MaxDynamicSmem>
void set_kernel_smem_once() {
  [[maybe_unused]] static const cudaError_t result = [] {
    return ::cudaFuncSetAttribute(
        Kernel, ::cudaFuncAttributeMaxDynamicSharedMemorySize, MaxDynamicSmem);
  }();
  TORCH_CHECK(
      result == cudaSuccess,
      "kpool_topk_transform cudaFuncSetAttribute failed: ",
      ::cudaGetErrorString(result));
}

template <
    int K,
    int Threads,
    bool UseVec8,
    int StagingSize,
    bool CacheCoarseBins,
    std::size_t DynamicSmem>
void launch_kpool_topk_transform_impl(
    const KpoolTopKParams& params, int64_t batch_size, cudaStream_t stream) {
  set_kernel_smem_once<
      kpool_topk_transform_kernel<
          K, Threads, UseVec8, StagingSize, CacheCoarseBins>,
      DynamicSmem>();
  const dim3 grid{static_cast<uint32_t>(batch_size)};
  const dim3 block{Threads};
  kpool_topk_transform_kernel<
      K, Threads, UseVec8, StagingSize, CacheCoarseBins>
      <<<grid, block, DynamicSmem, stream>>>(params);
}

template <int K>
void launch_kpool_topk_transform(
    const KpoolTopKParams& params,
    int64_t batch_size,
    bool use_short_staging,
    bool use_medium_threads,
    bool use_vec8,
    bool use_coarse_bin_cache,
    cudaStream_t stream) {
  if (use_vec8) {
    if (use_coarse_bin_cache) {
      launch_kpool_topk_transform_impl<
          K, kDefaultThreads, true, kLongStagingSize, true, kLongDynamicSmem>(
          params, batch_size, stream);
    } else {
      launch_kpool_topk_transform_impl<
          K, kDefaultThreads, true, kLongStagingSize, false, kLongDynamicSmem>(
          params, batch_size, stream);
    }
  } else if (use_short_staging) {
    launch_kpool_topk_transform_impl<
        K, kMediumThreads, false, kShortStagingSize, false, kShortDynamicSmem>(
        params, batch_size, stream);
  } else if (use_medium_threads) {
    launch_kpool_topk_transform_impl<
        K, kMediumThreads, false, kLongStagingSize, false, kLongDynamicSmem>(
        params, batch_size, stream);
  } else {
    launch_kpool_topk_transform_impl<
        K, kDefaultThreads, false, kLongStagingSize, false, kLongDynamicSmem>(
        params, batch_size, stream);
  }
}

void check_cuda_tensor(const at::Tensor& tensor, const char* name, const at::Device& device) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == device, name, " must be on the same device as score");
}

void check_int32_vector(
    const at::Tensor& tensor, const char* name, int64_t batch_size, const at::Device& device) {
  check_cuda_tensor(tensor, name, device);
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Int, name, " must have dtype int32");
  TORCH_CHECK(tensor.dim() == 1 && tensor.is_contiguous(), name, " must be a contiguous 1D tensor");
  TORCH_CHECK(tensor.size(0) == batch_size, name, " must have batch_size elements");
}

const int32_t* optional_int32_data(const std::optional<at::Tensor>& tensor) {
  return tensor.has_value() ? tensor->data_ptr<int32_t>() : nullptr;
}

}  // namespace

void kpool_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& output,
    int64_t pool_size,
    std::optional<at::Tensor> page_table_opt,
    std::optional<at::Tensor> topk_indices_offset_opt,
    std::optional<at::Tensor> row_starts_opt,
    std::optional<at::Tensor> seq_lens_opt,
    std::optional<at::Tensor> page_table_row_index_opt) {
  TORCH_CHECK(score.is_cuda(), "score must be a CUDA tensor");
  const at::Device device = score.device();
  TORCH_CHECK(score.scalar_type() == at::ScalarType::Float, "score must have dtype float32");
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1, "score must be 2D with stride(1) == 1");

  const int64_t batch_size = score.size(0);
  check_int32_vector(lengths, "lengths", batch_size, device);

  check_cuda_tensor(output, "output", device);
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Int, "output must have dtype int32");
  TORCH_CHECK(output.dim() == 2 && output.is_contiguous(), "output must be a contiguous 2D tensor");
  TORCH_CHECK(output.size(0) == batch_size, "output batch dimension must match score");

  TORCH_CHECK(pool_size > 1, "pool_size must be greater than 1");
  TORCH_CHECK(pool_size <= std::numeric_limits<int32_t>::max(), "pool_size exceeds int32 range");
  TORCH_CHECK(
      !(page_table_opt.has_value() && topk_indices_offset_opt.has_value()),
      "page_table and topk_indices_offset are mutually exclusive");
  TORCH_CHECK(
      !page_table_row_index_opt.has_value() || page_table_opt.has_value(),
      "page_table_row_index requires page_table");

  const int64_t tail_cols = seq_lens_opt.has_value() ? pool_size - 1 : 0;
  TORCH_CHECK(output.size(1) > tail_cols, "output has too few columns");
  const int64_t token_topk = output.size(1) - tail_cols;
  TORCH_CHECK(token_topk % pool_size == 0, "token_topk must be divisible by pool_size");
  const int64_t group_topk = token_topk / pool_size;
  TORCH_CHECK(
      group_topk == 128 || group_topk == 160 || group_topk == 192 ||
          group_topk == 224 || group_topk == 256 || group_topk == 512,
      "unsupported group_topk: ",
      group_topk);
  TORCH_CHECK(
      output.size(1) <= std::numeric_limits<int32_t>::max(),
      "output column count exceeds int32 range");

  if (topk_indices_offset_opt.has_value()) {
    check_int32_vector(*topk_indices_offset_opt, "topk_indices_offset", batch_size, device);
  }
  if (row_starts_opt.has_value()) {
    check_int32_vector(*row_starts_opt, "row_starts", batch_size, device);
  }
  if (seq_lens_opt.has_value()) {
    check_int32_vector(*seq_lens_opt, "seq_lens", batch_size, device);
  }
  if (page_table_row_index_opt.has_value()) {
    check_int32_vector(*page_table_row_index_opt, "page_table_row_index", batch_size, device);
  }

  const int32_t* page_table_ptr = nullptr;
  int64_t page_table_stride = 0;
  if (page_table_opt.has_value()) {
    const at::Tensor& page_table = *page_table_opt;
    check_cuda_tensor(page_table, "page_table", device);
    TORCH_CHECK(page_table.scalar_type() == at::ScalarType::Int, "page_table must have dtype int32");
    TORCH_CHECK(
        page_table.dim() == 2 && page_table.stride(1) == 1,
        "page_table must be 2D with stride(1) == 1");
    if (!page_table_row_index_opt.has_value()) {
      TORCH_CHECK(page_table.size(0) == batch_size, "page_table batch dimension must match score");
    } else {
      TORCH_CHECK(page_table.size(0) > 0, "page_table must contain at least one row");
    }
    page_table_ptr = page_table.data_ptr<int32_t>();
    page_table_stride = page_table.stride(0);
  }

  if (batch_size == 0) {
    return;
  }

  const KpoolTopKParams params{
      .score = score.data_ptr<float>(),
      .lengths = lengths.data_ptr<int32_t>(),
      .row_starts = optional_int32_data(row_starts_opt),
      .output = output.data_ptr<int32_t>(),
      .page_table = page_table_ptr,
      .page_table_row_index = optional_int32_data(page_table_row_index_opt),
      .topk_indices_offset = optional_int32_data(topk_indices_offset_opt),
      .seq_lens = optional_int32_data(seq_lens_opt),
      .score_stride = score.stride(0),
      .output_stride = output.stride(0),
      .page_table_stride = page_table_stride,
      .pool_size = static_cast<int32_t>(pool_size),
      .token_topk = static_cast<int32_t>(token_topk),
      .out_cols = static_cast<int32_t>(output.size(1)),
  };

  // Use the smaller staging instance only when enough row blocks are present
  // to benefit from its higher occupancy.
  const bool use_short_staging =
      score.size(1) <= kShortStagingSize && batch_size >= kShortStagingMinBatch;
  const bool use_medium_threads =
      score.size(1) > group_topk + 1 && score.size(1) <= kMediumMaxScoreWidth;
  const bool use_vec8 = score.size(1) >= kVec8MinScoreWidth;
  // Cache one coarse byte per score when it fits in half of the long staging
  // allocation and the batch is large enough to amortize the extra writes.
  const bool use_coarse_bin_cache =
      use_vec8 && score.size(1) <= kCoarseBinCacheMaxScoreWidth &&
      batch_size >= kCoarseBinCacheMinBatch;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  switch (group_topk) {
    case 128:
      launch_kpool_topk_transform<128>(
          params,
          batch_size,
          use_short_staging,
          use_medium_threads,
          use_vec8,
          use_coarse_bin_cache,
          stream);
      break;
    case 160:
      launch_kpool_topk_transform<160>(
          params,
          batch_size,
          use_short_staging,
          use_medium_threads,
          use_vec8,
          use_coarse_bin_cache,
          stream);
      break;
    case 192:
      launch_kpool_topk_transform<192>(
          params,
          batch_size,
          use_short_staging,
          use_medium_threads,
          use_vec8,
          use_coarse_bin_cache,
          stream);
      break;
    case 224:
      launch_kpool_topk_transform<224>(
          params,
          batch_size,
          use_short_staging,
          use_medium_threads,
          use_vec8,
          use_coarse_bin_cache,
          stream);
      break;
    case 256:
      launch_kpool_topk_transform<256>(
          params,
          batch_size,
          use_short_staging,
          use_medium_threads,
          use_vec8,
          use_coarse_bin_cache,
          stream);
      break;
    case 512:
      launch_kpool_topk_transform<512>(
          params,
          batch_size,
          use_short_staging,
          use_medium_threads,
          use_vec8,
          use_coarse_bin_cache,
          stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported group_topk: ", group_topk);
  }

  const cudaError_t result = ::cudaGetLastError();
  TORCH_CHECK(
      result == cudaSuccess,
      "kpool_topk_transform kernel launch failed: ",
      ::cudaGetErrorString(result));
}
