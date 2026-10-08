/*
 * Adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
 * Copyright (c) 2024, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <type_traits>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../cuda_compat.h"
#include "../cub_helpers.h"

#ifndef USE_ROCM
    #include <cuda_bf16.h>
    #include <cuda_fp16.h>
#else
    #include <hip/hip_bf16.h>
    #include <hip/hip_fp16.h>
    typedef __hip_bfloat16 __nv_bfloat16;
    typedef __hip_bfloat162 __nv_bfloat162;
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {
namespace moe {

// C600U schedules 16 lanes at a time (4 sub-groups make one 64-lane warp), so
// width-16 shuffles map to the native scheduling unit. Guarded here so this TU
// is self-contained even when the decode path is instantiated for E!=128.
#ifndef __shfl_down_sync_16
  #define __shfl_down_sync_16(mask, val, offset) __shfl_down_sync(mask, val, offset, 16)
#endif

/// Aligned array type
template <
    typename T,
    /// Number of elements in the array
    int N,
    /// Alignment requirement in bytes
    int Alignment = sizeof(T) * N
>
struct alignas(Alignment) AlignedArray {
    T data[N];
};

__device__ __forceinline__ float get_weight(const int64_t& v) {
    const float* idx_and_weight = (const float*)&v;
    return idx_and_weight[0];
}

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(value);
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(value);
    }
}

template<uint32_t MASK=0xffffffff>
__device__ __forceinline__ void warpSortDescendingNew(float (&idx_and_weight)[2], int tid) {

    int64_t val = *(int64_t*)idx_and_weight;
    for (int width = 2; width < WARP_SIZE; width <<=1 ) {
        for (int step = width >> 1; step > 0; step >>=1) {
            const bool direction = ((tid & width) == 0);
            int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
            int other_tid = tid ^ step;

            float current_weight_bits = get_weight(val);
            float other_weight_bits = get_weight(other_temp_val);
            int current_index = val >> 32;
            int other_index = other_temp_val >> 32;

            bool weight_gt = other_weight_bits > current_weight_bits;
            bool weight_eq = other_weight_bits == current_weight_bits;
            bool index_lt = other_index < current_index;

            bool other_is_big = weight_gt | (weight_eq & index_lt);
            bool swap = (tid < other_tid) ^ (other_is_big) ^ (direction);

            val = swap ? other_temp_val : val;
        }
    }
    for (int step = 16; step > 0; step >>= 1) {
        int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
        int other_tid = tid ^ step;

        float current_weight_bits = get_weight(val);
        float other_weight_bits = get_weight(other_temp_val);
        int current_index = val >> 32;
        int other_index = other_temp_val >> 32;

        bool weight_gt = other_weight_bits > current_weight_bits;
        bool weight_eq = other_weight_bits == current_weight_bits;
        bool index_lt = other_index < current_index;

        bool other_is_big = weight_gt | (weight_eq & index_lt);
        bool swap = (tid < other_tid) ^ (!other_is_big);
        val = swap ? other_temp_val : val;
    }
    *(int64_t*)idx_and_weight = val;
}

template<uint64_t MASK=0xffffffffffffffff>
__device__ __forceinline__ void warpSortDescendingUpdate(float (&idx_and_weight)[2], int tid) {

    int64_t val = *(int64_t*)idx_and_weight;
    for (int width = 2; width < 64; width <<=1 ) {
        for (int step = width >> 1; step > 0; step >>=1) {
            const bool direction = ((tid & width) == 0);
            int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
            int other_tid = tid ^ step;

            float current_weight_bits = get_weight(val);
            float other_weight_bits = get_weight(other_temp_val);
            int current_index = val >> 32;
            int other_index = other_temp_val >> 32;

            bool weight_gt = other_weight_bits > current_weight_bits;
            bool weight_eq = other_weight_bits == current_weight_bits;
            bool index_lt = other_index < current_index;

            bool other_is_big = weight_gt | (weight_eq & index_lt);
            bool swap = (tid < other_tid) ^ (other_is_big) ^ (direction);

            val = swap ? other_temp_val : val;
        }
    }
    for (int step = 32; step > 0; step >>= 1) {
        int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
        int other_tid = tid ^ step;

        float current_weight_bits = get_weight(val);
        float other_weight_bits = get_weight(other_temp_val);
        int current_index = val >> 32;
        int other_index = other_temp_val >> 32;

        bool weight_gt = other_weight_bits > current_weight_bits;
        bool weight_eq = other_weight_bits == current_weight_bits;
        bool index_lt = other_index < current_index;

        bool other_is_big = weight_gt | (weight_eq & index_lt);
        bool swap = (tid < other_tid) ^ (!other_is_big);
        val = swap ? other_temp_val : val;
    }
    *(int64_t*)idx_and_weight = val;
}

// Scoring function enums
enum ScoringFunc {
  SCORING_SOFTMAX = 0, // apply softmax
  SCORING_SIGMOID = 1  // apply sigmoid
};

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing the output
// in the softmax kernel when we extend this module to support expert-choice routing.
template <int TPB, typename InputType>
__launch_bounds__(TPB) __global__
    void moeSoftmax(const InputType* input, const bool* finished, float* output, const int num_cols)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizing_factor;
    __shared__ float float_max;

    const int thread_row_offset = blockIdx.x * num_cols;

    float threadData(-FLT_MAX);

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[blockIdx.x])
    {
        return;
    }

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        threadData = max(val, threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, CubMaxOp());
    if (threadIdx.x == 0)
    {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        threadData += expf(val - float_max);
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, CubAddOp());

    if (threadIdx.x == 0)
    {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        float softmax_val = expf(val - float_max) * normalizing_factor;
        // Clamp NaN/Inf to 0 to prevent duplicate expert IDs downstream.
        if (isnan(softmax_val) || isinf(softmax_val)) softmax_val = 0.f;
        output[idx] = softmax_val;
    }
}

template <int TPB, typename InputType>
__launch_bounds__(TPB) __global__
    void moeSigmoid(const InputType* input, const bool* finished, float* output, const int num_cols)
{
    const int thread_row_offset = blockIdx.x * num_cols;

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[blockIdx.x])
    {
        return;
    }

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        float sigmoid_val = 1.0f / (1.0f + __expf(-val));
        // Clamp NaN/Inf to 0 to prevent duplicate expert IDs downstream.
        if (isnan(sigmoid_val) || isinf(sigmoid_val)) sigmoid_val = 0.f;
        output[idx] = sigmoid_val;
    }
}

template <int TPB, typename IndType>
__launch_bounds__(TPB) __global__ void moeTopK(
    const float* inputs_after_softmax,
    const bool* finished,
    float* output,
    IndType* indices,
    int* source_rows,
    const int num_experts,
    const int k,
    const int start_expert,
    const int end_expert,
    const bool renormalize,
    const float* bias,
    const double routed_scaling_factor,
    const bool* is_padding)
{

    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const int num_rows = gridDim.x;
    const int block_row = blockIdx.x;

    const bool row_is_active = finished ? !finished[block_row] : true;
    const int thread_read_offset = blockIdx.x * num_experts;
    float selected_sum = 0.f;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        thread_kvp.key = 0;
        thread_kvp.value = -1.f; // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB)
        {
            const int idx = thread_read_offset + expert;
            inp_kvp.key = expert;

            // Apply correction bias if provided
            if (bias != nullptr) {
              inp_kvp.value = inputs_after_softmax[idx] + bias[expert];
            } else {
              inp_kvp.value = inputs_after_softmax[idx];
            }

            for (int prior_k = 0; prior_k < k_idx; ++prior_k)
            {
                const int prior_winning_expert = indices[k * block_row + prior_k];

                if (prior_winning_expert == expert)
                {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0)
        {
            // Ignore experts the node isn't responsible for with expert parallelism
            const int expert = result_kvp.key;
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;
            const bool is_pad_row = is_padding != nullptr && is_padding[block_row];

            const int idx = k * block_row + k_idx;
            // Return the unbiased scores for output weights
            output[idx] = inputs_after_softmax[thread_read_offset + expert];
            indices[idx] = is_pad_row ? static_cast<IndType>(-1)
                                       : (should_process_row ? (expert - start_expert) : num_experts);
            assert(is_pad_row || indices[idx] >= 0);
            source_rows[idx] = k_idx * num_rows + block_row;
            if (renormalize) {
                selected_sum += inputs_after_softmax[thread_read_offset + expert];
            }
        }
        __syncthreads();
    }

    // Apply renormalization and routed scaling factor to final weights.
    if (threadIdx.x == 0) {
        float scale = static_cast<float>(routed_scaling_factor);
        if (renormalize) {
            const float denom = selected_sum > 0.f ? selected_sum : 1.f;
            scale /= denom;
        }
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int idx = k * block_row + k_idx;
            output[idx] = output[idx] * scale;
        }
    }
}

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the MoE layers
  are a small power of 2. This allows us to cleanly share the rows among the threads in
  a single warp and eliminate communication between warps (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is optimized for when the number of experts is a small power of 2.
     Additionally it also supports when number of experts is multiple of 64 which is still
     faster than the computing softmax and topK separately (only tested on CUDA yet).
  2) This implementation assumes k is small, but will work for any k.
*/

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, int WARP_SIZE_PARAM, typename IndType,
          typename InputType = float, ScoringFunc SF>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE_PARAM) __global__
    void topkGating(const InputType* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int k, const int start_expert, const int end_expert, const bool renormalize,
        const float* bias, const double routed_scaling_factor, const bool* is_padding)
{
    static_assert(std::is_same_v<InputType, float> || std::is_same_v<InputType, __nv_bfloat16> ||
                      std::is_same_v<InputType, __half>,
                  "InputType must be float, __nv_bfloat16, or __half");

    // We begin by enforcing compile time assertions and setting up compile time constants.
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    if constexpr (std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>) {
        static_assert(ELTS_PER_LDG == 1 || ELTS_PER_LDG % 2 == 0,
            "ELTS_PER_LDG must be 1 or even for 16-bit conversion");
    }

    // Restrictions based on previous section.
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE_PARAM % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE_PARAM, "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and a block contains WARPS_PER_CTA warps.
    // This, each block processes a chunk of rows. We start by computing the start row for each block.
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the base row per warp.
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows)
    {
        return;
    }
    const bool row_is_active =
        (finished == nullptr || !finished[thread_row]) &&
        (is_padding == nullptr || !is_padding[thread_row]);

    // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
    // row it will read.
    const InputType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the first column to start loads.
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const InputType* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    // Finally, we pull in the data from global mem
    float row_chunk[VPT];

    // NOTE(zhuhaoran): dispatch different input types loading, BF16/FP16 convert to float
    if constexpr (std::is_same_v<InputType, float>) {
        using VecType = AlignedArray<float, ELTS_PER_LDG>;
        VecType* row_chunk_vec_ptr = reinterpret_cast<VecType*>(&row_chunk);
        const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
        for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
            row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        }
    } else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
        if constexpr (ELTS_PER_LDG >= 2) {
            using VecType = AlignedArray<__nv_bfloat16, ELTS_PER_LDG>;
            float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
            const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
                int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
                for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
                    row_chunk_f2[base_idx_f2 + jj] = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(vec.data + jj * 2)
                    );
                }
            }
        } else { // ELTS_PER_LDG == 1
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                const __nv_bfloat16* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
                row_chunk[ii] = __bfloat162float(*scalar_ptr);
            }
        }
    } else if constexpr (std::is_same_v<InputType, __half>) {
        if constexpr (ELTS_PER_LDG >= 2) {
            using VecType = AlignedArray<__half, ELTS_PER_LDG>;
            float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
            const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
                int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
                for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
                    row_chunk_f2[base_idx_f2 + jj] = __half22float2(
                        *reinterpret_cast<const __half2*>(vec.data + jj * 2)
                    );
                }
            }
        } else { // ELTS_PER_LDG == 1
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                const __half* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
                row_chunk[ii] = __half2float(*scalar_ptr);
            }
        }
    }

    if constexpr (SF == SCORING_SOFTMAX) {
      // First, we perform a max reduce within the thread.
      float thread_max = row_chunk[0];
#pragma unroll
      for (int ii = 1; ii < VPT; ++ii) {
        thread_max = max(thread_max, row_chunk[ii]);
      }

// Now, we find the max within the thread group and distribute among the threads. We use a butterfly reduce.
#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
      {
        thread_max = max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
      }

      // From this point, thread max in all the threads have the max within the row.
      // Now, we subtract the max from each element in the thread and take the exp. We also compute the thread local sum.
      float row_sum = 0;
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii)
      {
        row_chunk[ii] = __builtin_expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
      }

// Now, we perform the sum reduce within each thread group. Similar to the max reduce, we use a bufferfly pattern.
#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
      {
        row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
      }

      // From this point, all threads have the max and the sum for their rows in the thread_max and thread_sum variables
      // respectively. Finally, we can scale the rows for the softmax. Technically, for top-k gating we don't need to
      // compute the entire softmax row. We can likely look at the maxes and only compute for the top-k values in the row.
      // However, this kernel will likely not be a bottle neck and it seems better to closer match torch and find the
      // argmax after computing the softmax.
      const float reciprocal_row_sum = __builtin_mxc_rcpf(row_sum);

#pragma unroll
      for (int ii = 0; ii < VPT; ++ii)
      {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
      }
    } else if constexpr (SF == SCORING_SIGMOID) {
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii)
      {
        row_chunk[ii] = 1.0f / (1.0f + __expf(-row_chunk[ii]));
      }
    }

    // Fix: clamp NaN/Inf values to 0 to prevent duplicate expert IDs.
    // NaN gating (from degenerate hidden states in CUDA graph padding) causes
    // softmax to produce all-NaN, which makes the argmax loop always pick
    // expert 0 for every top-k slot, producing duplicate expert IDs that
    // crash FlashInfer's three-step MoE sort.
    // With 0s, the argmax uses index tie-breaking to pick [0,1,2,...,k-1].
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      if (isnan(row_chunk[ii]) || isinf(row_chunk[ii])) {
        row_chunk[ii] = 0.f;
      }
    }

    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    // If bias is not null, use biased value for selection
    float row_chunk_for_choice[VPT];
    // Apply correction bias
    if (bias != nullptr) {
#pragma unroll
      for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
#pragma unroll
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          const int expert = first_elt_read_by_thread + ldg * COLS_PER_GROUP_LDG + ii;
          float bias_val = expert < NUM_EXPERTS ? bias[expert] : 0.0f;
          row_chunk_for_choice[ldg * ELTS_PER_LDG + ii] = row_chunk[ldg * ELTS_PER_LDG + ii] + bias_val;
        }
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii) {
        row_chunk_for_choice[ii] = row_chunk[ii];
      }
    }

    // Now, row_chunk contains the softmax / sigmoid of the row chunk. Now, I want to find the topk elements in each row, along
    // with the max index.
    int start_col = first_elt_read_by_thread;

    float selected_sum = 0.f;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        // First, each thread does the local argmax
        float max_val_for_choice = row_chunk_for_choice[0];
        float max_val = row_chunk[0];
        int expert = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                float val_for_choice = row_chunk_for_choice[ldg * ELTS_PER_LDG + ii];
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];

                // No check on the experts here since columns with the smallest index are processed first and only
                // updated if > (not >=)
                if (val_for_choice > max_val_for_choice)
                {
                    max_val_for_choice = val_for_choice;
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads reach consensus about the max.
// This will be useful for K > 1 so that the threads can agree on "who" had the max value. That thread can
// then blank out their max with -inf and the warp can run more iterations...
        if (bias == nullptr) {
          // Fast path (no correction bias). Pack the argmax ordering key into a
          // single 64-bit word so the butterfly reduce shuffles one value per
          // step instead of two. This k-loop argmax reduce is the compute-bound
          // critical path (k * log2(THREADS_PER_ROW) dependent steps).
          //   high 32 bits: order-preserving unsigned image of max_val_for_choice
          //     (choice value carries the -10000 sentinel for already-selected
          //      experts and can be negative, so a plain float-bit cast is not
          //      monotonic; the xor transform makes it a total order over floats).
          //   low  32 bits: ~expert, so on a value tie the SMALLER expert id
          //      produces the LARGER key ("lower index wins" tie-break).
          uint32_t cb = __float_as_uint(max_val_for_choice);
          cb ^= (cb >> 31) ? 0xffffffffu : 0x80000000u;
          uint64_t key = (static_cast<uint64_t>(cb) << 32) |
                         static_cast<uint32_t>(~static_cast<uint32_t>(expert));
#pragma unroll
          for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
          {
            uint64_t other = VLLM_SHFL_XOR_SYNC_WIDTH(key, mask, THREADS_PER_ROW);
            if (other > key) key = other;
          }
          expert = static_cast<int>(~static_cast<uint32_t>(key & 0xffffffffu));
          uint32_t ob = static_cast<uint32_t>(key >> 32);
          ob ^= (ob >> 31) ? 0x80000000u : 0xffffffffu;   // inverse transform
          max_val_for_choice = __uint_as_float(ob);
          // For the winning expert the choice value was never blanked, so with
          // no bias it equals the (unbiased) weight exactly -> no extra shuffle.
          max_val = max_val_for_choice;
        } else {
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            float other_max_for_choice = VLLM_SHFL_XOR_SYNC_WIDTH(max_val_for_choice, mask, THREADS_PER_ROW);
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);

            // We want lower indices to "win" in every thread so we break ties this way
            if (other_max_for_choice > max_val_for_choice || (other_max_for_choice == max_val_for_choice && other_expert < expert))
            {
                max_val_for_choice = other_max_for_choice;
                max_val = other_max;
                expert = other_expert;
            }
        }
        }

        // Write the max for this k iteration to global memory.
        if (thread_group_idx == 0)
        {
            // Add a guard to ignore experts not included by this node
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;
            const bool is_pad_row = is_padding != nullptr && is_padding[thread_row];

            // The lead thread from each sub-group will write out the final results to global memory. (This will be a
            // single) thread per row of the input/output matrices.
            const int idx = k * thread_row + k_idx;
            output[idx] = should_process_row ? max_val : 0.0f;
            indices[idx] = should_process_row ? static_cast<IndType>(expert - start_expert) : static_cast<IndType>(NUM_EXPERTS);
            source_rows[idx] = k_idx * num_rows + thread_row;
            if (renormalize && should_process_row) {
                selected_sum += max_val;
            }
        }

        // Finally, we clear the value in the thread with the current max if there is another iteration to run.
        if (k_idx + 1 < k)
        {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            // Only the thread in the group which produced the max will reset the "winning" value to -inf.
            if (thread_group_idx == thread_to_clear_in_group)
            {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                // Safe to set to any negative value since row_chunk values must be between 0 and 1.
                row_chunk_for_choice[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
            }
        }
    }

    // Apply renormalization and routed scaling factor to final weights.
    if (thread_group_idx == 0) {
      float scale = static_cast<float>(routed_scaling_factor);
      if (renormalize) {
        const float denom = selected_sum > 0.f ? selected_sum : 1.0f;
        scale /= denom;
      }
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        const int idx = k * thread_row + k_idx;
        output[idx] *= scale;
      }
    }
}

template <typename index_t, typename InputType, int VPT = 4, int NUM_EXPERTS = 128, int WARPS_PER_CTA = 8, int BYTES_PER_LDG = 4, int ROWS_PER_CTA = 2, int WARPS_PER_ROW = 4>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topkGatingSoftmaxDecode(const InputType* input, const bool* finished, float* output, const int num_rows, index_t* indices,
        int* source_rows, const int k, const int start_expert, const int end_expert, const bool renormalize, 
        const float* bias, const int topk = 8, const double routed_scaling_factor = 1.0, const bool* is_padding = nullptr)
{
    static int THREADS_PER_ROW = WARPS_PER_ROW * WARP_SIZE;
    const int thread_row = blockIdx.x * ROWS_PER_CTA + threadIdx.x / THREADS_PER_ROW;
    if (thread_row >= num_rows)
    {
        return;
    }
    const int warp_id = (threadIdx.x / WARP_SIZE);
    const int tid_in_row = threadIdx.x % THREADS_PER_ROW;
    const int row_in_block = threadIdx.x / THREADS_PER_ROW;
    float idx_and_weight[2];
    float idx_and_weight_2[2];
    const bool row_is_active = (finished ? !finished[thread_row] : true) && (is_padding ? !is_padding[thread_row] : true);
    __shared__ float shared_experts[64][2];
    __shared__ float max_val[128];
    __shared__ float sum_val[128];
    __shared__ float max_val_final[ROWS_PER_CTA];
    __shared__ float rep[ROWS_PER_CTA];
    __shared__ float norm[128];

    // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
    // row it will read.

    float row_val = static_cast<float>(input[thread_row * NUM_EXPERTS + tid_in_row]);
    float thread_max = row_val;
    float sum_data = __builtin_expf(row_val);

    thread_max = fmaxf(thread_max, __shfl_down_sync_16(0xffffffffffffffff, thread_max, 1));
    thread_max = fmaxf(thread_max, __shfl_down_sync_16(0xffffffffffffffff, thread_max, 2));
    thread_max = fmaxf(thread_max, __shfl_down_sync_16(0xffffffffffffffff, thread_max, 4));
    thread_max = fmaxf(thread_max, __shfl_down_sync_16(0xffffffffffffffff, thread_max, 8));

    sum_data += __shfl_down_sync_16(0xffffffffffffffff, sum_data, 1);
    sum_data += __shfl_down_sync_16(0xffffffffffffffff, sum_data, 2);
    sum_data += __shfl_down_sync_16(0xffffffffffffffff, sum_data, 4);
    sum_data += __shfl_down_sync_16(0xffffffffffffffff, sum_data, 8);

    if (threadIdx.x % 16 == 0) {
        max_val[threadIdx.x / 16] = thread_max;
        sum_val[threadIdx.x / 16] = sum_data;
    }
    __syncthreads();
    float sm_v = 0;
    float mx_v = -9999;

    if (tid_in_row < topk) {
        sm_v = sum_val[row_in_block * topk + tid_in_row];
        mx_v = max_val[row_in_block * topk + tid_in_row];

        sm_v += __shfl_down_sync_16(0x000000000000ffff, sm_v, 1);
        sm_v += __shfl_down_sync_16(0x000000000000ffff, sm_v, 2);
        sm_v += __shfl_down_sync_16(0x000000000000ffff, sm_v, 4);

        mx_v = fmaxf(mx_v, __shfl_down_sync_16(0x00000000000000ff, mx_v, 1));
        mx_v = fmaxf(mx_v, __shfl_down_sync_16(0x00000000000000ff, mx_v, 2));
        mx_v = fmaxf(mx_v, __shfl_down_sync_16(0x00000000000000ff, mx_v, 4));

        if (tid_in_row == 0) {
            max_val_final[row_in_block] = mx_v;
            rep[row_in_block] = 1.0f / (sm_v * __builtin_expf(-mx_v));
        }
    }
    __syncthreads();

    row_val = __builtin_expf(row_val - max_val_final[row_in_block]) * rep[row_in_block];
    //logits----> softmax ----> row_val
    //row_val + bias----->row_val_for_choice，[num_expert],acts at topk select
    float row_val_for_choice = row_val;
    if (bias != nullptr) {
        const int expert_id = tid_in_row;
        float bias_val = (expert_id < NUM_EXPERTS) ? bias[expert_id] : 0.0f;
        row_val_for_choice = row_val + bias_val;
    }

    int64_t expert_id_opt = static_cast<int64_t>(tid_in_row);

    idx_and_weight[0] = row_val_for_choice;
    idx_and_weight[1] = 0.0f;
    *((int64_t*)idx_and_weight) |= (expert_id_opt << 32);
    warpSortDescendingNew<0xffffffff>(idx_and_weight, threadIdx.x);
    if (tid_in_row % WARP_SIZE < topk) {
        *((int64_t*)(shared_experts) + warp_id * topk + tid_in_row % WARP_SIZE) = *(int64_t*)idx_and_weight;
    }
    __syncthreads();
    float max_val_ordered;
    if(tid_in_row < WARP_SIZE){
        *(int64_t*)idx_and_weight_2 = *((int64_t*)shared_experts + (warp_id / WARPS_PER_ROW) * WARP_SIZE + tid_in_row);
        warpSortDescendingNew<0xffffffff>(idx_and_weight_2, threadIdx.x);
        if (tid_in_row < topk) {
            int64_t res = *(int64_t*)idx_and_weight_2;
            max_val_ordered = get_weight(res);
            int expert_id_ordered = static_cast<int>(res >> 32);

            if (bias != nullptr && expert_id_ordered >= 0 && expert_id_ordered < NUM_EXPERTS) {
                max_val_ordered -= bias[expert_id_ordered];
            }
            // subtract bias，restore origin row_val(after topk)
            // if (bias != nullptr) {
            //     float bias_val = (expert_id_ordered < NUM_EXPERTS) ? bias[expert_id_ordered] : 0.0f;
            //     max_val_ordered = max_val_ordered - bias_val;
            // }

            // Add a guard to ignore experts not included by this node
            const bool node_uses_expert = expert_id_ordered >= start_expert && expert_id_ordered < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;

            // The lead thread from each sub-group will write out the final results to global memory. (This will be a
            // single) thread per row of the input/output matrices.
            const int idx = topk * thread_row + tid_in_row;
            indices[idx] = should_process_row ? (expert_id_ordered - start_expert) : NUM_EXPERTS;
            source_rows[idx] = tid_in_row * num_rows + thread_row;
            if (renormalize) {
                norm[row_in_block * topk + tid_in_row] = should_process_row ? static_cast<float>(max_val_ordered) : 0.0f;
            } else {
                output[idx] = should_process_row ? static_cast<float>(max_val_ordered) * static_cast<float>(routed_scaling_factor) : 0.0f;
            }
        }
    }
    __shared__ float norm_val[ROWS_PER_CTA];
    if (renormalize) {
        __syncthreads();
        if (tid_in_row == 0) {
            norm_val[row_in_block] = 0.0f;
            for (int i = 0; i < topk; i++) {
                norm_val[row_in_block] += norm[row_in_block * topk + i];
            }
            if (norm_val[row_in_block] <= 0.0f) {
                norm_val[row_in_block] = 1.0f;
            }
        }
        __syncthreads();
        if (tid_in_row < topk) {
            const int idx = topk * thread_row + tid_in_row;
            const float denom = norm_val[row_in_block];
            output[idx] = row_is_active ? static_cast<float>(max_val_ordered) / denom * static_cast<float>(routed_scaling_factor) : 0.0f;
        }
    }
}

// ============================================================================
// Dedicated DECODE softmax kernel for E=256, k=8 (C600U 16-lane scheduling).
// Strictly follows the "16 threads x 4 = 64-lane warp" structure the user
// specified: 1 row per block, 256 threads = 4 physical 64-lane warps, one
// thread per expert. Softmax max/sum reduce use width-16 shuffles (native
// scheduling unit) escalated across groups; top-k uses the proven 64-wide
// bitonic sorter (warpSortDescendingUpdate, correct 64-bit mask): each 64-lane
// warp reduces its 64 experts to a sorted top-8, the 4x8=32 survivors merge in
// one 64-lane warp (32 <= 64 fits). renorm handled in-block.
// ============================================================================
template <typename index_t, typename InputType>
__launch_bounds__(256) __global__
    void topkGatingSoftmaxDecode256(const InputType* input, const bool* finished, float* output,
        const int num_rows, index_t* indices, int* source_rows, const int k,
        const int start_expert, const int end_expert, const bool renormalize,
        const float* bias, const int topk = 8, const double routed_scaling_factor = 1.0,const bool* is_padding = nullptr)
{
    constexpr int NUM_EXPERTS = 256;
    constexpr int WARP = 64;              // physical warp lanes
    constexpr int WARPS_PER_ROW = 4;      // 4*64 = 256 experts
    const int thread_row = blockIdx.x;    // 1 row per block
    if (thread_row >= num_rows) return;
    const bool row_is_active = (finished == nullptr || !finished[thread_row]) && (is_padding == nullptr || !is_padding[thread_row]);

    const int tid = threadIdx.x;          // 0..255, == expert id
    const int warp_id = tid / WARP;       // 0..3
    const int lane = tid % WARP;          // 0..63

    __shared__ float s_max[16];           // per 16-lane group max
    __shared__ float s_sum[16];           // per 16-lane group exp-sum
    __shared__ float s_rowmax;
    __shared__ float s_rep;               // 1 / (sum * exp(-rowmax)) = softmax denom recip
    __shared__ int64_t s_cand[64];        // 4 warps * 8 candidates (top of 64) padded to 64
    __shared__ float s_norm[8];

    float logit = static_cast<float>(input[thread_row * NUM_EXPERTS + tid]);

    // --- per-16-lane-group max & exp-sum (width-16 shuffles, native unit) ---
    float m = logit;
    m = fmaxf(m, __shfl_down_sync_16(0xffffffffffffffffULL, m, 1));
    m = fmaxf(m, __shfl_down_sync_16(0xffffffffffffffffULL, m, 2));
    m = fmaxf(m, __shfl_down_sync_16(0xffffffffffffffffULL, m, 4));
    m = fmaxf(m, __shfl_down_sync_16(0xffffffffffffffffULL, m, 8));
    float e = __builtin_expf(logit);
    e += __shfl_down_sync_16(0xffffffffffffffffULL, e, 1);
    e += __shfl_down_sync_16(0xffffffffffffffffULL, e, 2);
    e += __shfl_down_sync_16(0xffffffffffffffffULL, e, 4);
    e += __shfl_down_sync_16(0xffffffffffffffffULL, e, 8);
    if (lane % 16 == 0) {
        s_max[tid / 16] = m;
        s_sum[tid / 16] = e;
    }
    __syncthreads();

    // --- reduce 16 group-partials to the row max, then the true exp-sum ---
    if (tid == 0) {
        float rmax = s_max[0];
        for (int i = 1; i < 16; ++i) rmax = fmaxf(rmax, s_max[i]);
        // s_sum[i] == sum_j exp(logit_j) over group i with NO max subtraction.
        // denom = sum_j exp(logit_j - rmax) = (sum_j exp(logit_j)) * exp(-rmax).
        float total = 0.f;
        for (int i = 0; i < 16; ++i) total += s_sum[i];
        s_rowmax = rmax;
        s_rep = 1.0f / (total * __builtin_expf(-rmax));
    }
    __syncthreads();

    // softmax probability for this expert
    float prob = __builtin_expf(logit - s_rowmax) * s_rep;
    float choice = prob;
    if (bias != nullptr) choice = prob + bias[tid];

    // --- top-k within each 64-lane warp via 64-wide bitonic sort ---
    float kw[2];
    kw[0] = choice;
    kw[1] = 0.f;
    *((int64_t*)kw) |= (static_cast<int64_t>(tid) << 32);
    warpSortDescendingUpdate<0xffffffffffffffffULL>(kw, lane);
    // lanes 0..7 of each warp hold that warp's top-8 (sorted desc)
    if (lane < 8) {
        s_cand[warp_id * 8 + lane] = *(int64_t*)kw;
    }
    __syncthreads();

    // --- merge 4*8 = 32 candidates in one 64-lane warp ---
    float mw[2];
    float sel_prob = 0.f;
    int sel_expert = NUM_EXPERTS;
    if (tid < WARP) {
        int64_t v;
        if (lane < 32) {
            v = s_cand[lane];
        } else {
            // pad upper 32 lanes with a sentinel that always loses
            float pad[2]; pad[0] = -3.0e38f; pad[1] = 0.f;
            *((int64_t*)pad) |= (static_cast<int64_t>(NUM_EXPERTS + lane) << 32);
            v = *(int64_t*)pad;
        }
        *(int64_t*)mw = v;
        warpSortDescendingUpdate<0xffffffffffffffffULL>(mw, lane);
        if (lane < topk) {
            int64_t res = *(int64_t*)mw;
            sel_prob = get_weight(res);
            sel_expert = (int)(res >> 32);
            if (bias != nullptr) sel_prob -= bias[sel_expert];  // restore raw prob
            const bool node_uses_expert = sel_expert >= start_expert && sel_expert < end_expert;
            const bool should = row_is_active && node_uses_expert;
            const int idx = topk * thread_row + lane;
            indices[idx] = should ? (sel_expert - start_expert) : NUM_EXPERTS;
            source_rows[idx] = lane * num_rows + thread_row;
            if (renormalize) {
                s_norm[lane] = should ? sel_prob : 0.0f;
            } else {
                output[idx] = should ? sel_prob * static_cast<float>(routed_scaling_factor) : 0.0f;
            }
        }
    }
    if (renormalize) {
        __syncthreads();
        if (tid == 0) {
            float z = 0.f;
            for (int i = 0; i < topk; ++i) z += s_norm[i];
            s_rep = z > 0.f ? 1.0f / z : 1.0f;   // reuse
        }
        __syncthreads();
        if (tid < WARP && lane < topk) {
            const int idx = topk * thread_row + lane;
            output[idx] = row_is_active ? sel_prob * s_rep * static_cast<float>(routed_scaling_factor) : 0.0f;
        }
    }
}

template <typename IndType, typename InputType = float>
__launch_bounds__(640)
__global__ void topkGatingSigmoid288Opt(
    const InputType* __restrict__ input,
    const bool* __restrict__ finished,
    float* __restrict__ output,
    const int num_rows,
    IndType* __restrict__ indices,
    const int start_expert,
    const int end_expert,
    const bool renormalize,
    const float* __restrict__ bias,
    const double routed_scaling_factor,
    const bool* __restrict__ is_padding = nullptr)
{
    constexpr int NUM_EXPERTS = 288;
    constexpr int TOPK = 8;
    constexpr int WARP_SIZE_C500 = 64;
    constexpr int WARPS_PER_ROW = 5; // 288 5 个 Warp
    constexpr int THREADS_PER_ROW = WARPS_PER_ROW * WARP_SIZE_C500; // 320
    constexpr int ROWS_PER_CTA = 2;

    const int row_in_block = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = blockIdx.x * ROWS_PER_CTA + row_in_block;

    if (thread_row >= num_rows) return;

    const int tid_in_row = threadIdx.x % THREADS_PER_ROW;
    const int warp_in_row = tid_in_row / WARP_SIZE_C500;
    const int lane_id = tid_in_row % WARP_SIZE_C500;

    float row_val_for_choice = -3.4e38f;

    if (tid_in_row < NUM_EXPERTS) {
        float val = static_cast<float>(input[thread_row * NUM_EXPERTS + tid_in_row]);
        val = 1.0f / (1.0f + __expf(-val)); // Fast Sigmoid
        if (bias != nullptr) {
            row_val_for_choice = val + bias[tid_in_row];
        } else {
            row_val_for_choice = val;
        }
    }

    bool row_is_active = finished ? !finished[thread_row] : true;
    if (is_padding != nullptr && is_padding[thread_row]) {
        row_is_active = false;
    }
    
    float idx_and_weight[2];
    idx_and_weight[0] = row_val_for_choice;
    idx_and_weight[1] = 0.0f;
    *((int64_t*)idx_and_weight) |= (static_cast<int64_t>(tid_in_row) << 32);

   
    warpSortDescendingUpdate<0xffffffffffffffff>(idx_and_weight, lane_id);

    __shared__ int64_t shared_experts[ROWS_PER_CTA][WARPS_PER_ROW * TOPK];
    __shared__ float sm_norm_vals[ROWS_PER_CTA][TOPK];
    __shared__ float sm_sum_norm[ROWS_PER_CTA];

    if (lane_id < TOPK) {
        shared_experts[row_in_block][warp_in_row * TOPK + lane_id] = *(int64_t*)idx_and_weight;
    }
    __syncthreads();

    
    if (warp_in_row == 0) {
        float final_idx_weight[2];
        final_idx_weight[0] = -3.4e38f;
        final_idx_weight[1] = 0.0f; 
        *((int64_t*)final_idx_weight) |= (static_cast<int64_t>(NUM_EXPERTS + lane_id) << 32);

        // 加载 5个Warp * 8 = 40个局部Top候选者
        if (lane_id < WARPS_PER_ROW * TOPK) { 
            *(int64_t*)final_idx_weight = shared_experts[row_in_block][lane_id];
        }

        warpSortDescendingUpdate<0xffffffffffffffff>(final_idx_weight, lane_id);

        if (lane_id < TOPK) {
            int64_t res = *(int64_t*)final_idx_weight;
            float max_val_ordered = get_weight(res);
            int expert_id_ordered = static_cast<int>(res >> 32);

            if (bias != nullptr && expert_id_ordered < NUM_EXPERTS) {
                max_val_ordered -= bias[expert_id_ordered];
            }

            const bool node_uses_expert = expert_id_ordered >= start_expert && expert_id_ordered < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;

            const int idx = TOPK * thread_row + lane_id;
            indices[idx] = should_process_row ? (expert_id_ordered - start_expert) : NUM_EXPERTS;

            if (renormalize) {
                sm_norm_vals[row_in_block][lane_id] = should_process_row ? max_val_ordered : 0.0f;
            } else {
                output[idx] = should_process_row ? max_val_ordered * static_cast<float>(routed_scaling_factor) : 0.0f;
            }
        }
    }

    if (renormalize) {
        __syncthreads();
        if (tid_in_row == 0) {
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < TOPK; i++) {
                sum += sm_norm_vals[row_in_block][i];
            }
            sm_sum_norm[row_in_block] = sum > 0.0f ? sum : 1.0f;
        }
        __syncthreads();

        if (warp_in_row == 0 && lane_id < TOPK) {
            const int idx = TOPK * thread_row + lane_id;
            float const rcp_norm = __builtin_mxc_rcpf(sm_sum_norm[row_in_block]);
            output[idx] = row_is_active ? sm_norm_vals[row_in_block][lane_id] * rcp_norm * static_cast<float>(routed_scaling_factor) : 0.0f;
        }
    }
}

template <int VPT, typename IndType,
          typename InputType = float>
__global__ void topkGatingSigmoidCommonOpt(const InputType* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        const int start_expert, const int end_expert, const int num_experts, const int rows_per_cta, const bool renormalize,
        const float* bias, const int topk = 8, const double routed_scaling_factor = 1.0, const bool* is_padding = nullptr)
{
    static_assert(std::is_same_v<InputType, float> || std::is_same_v<InputType, __nv_bfloat16> ||
                      std::is_same_v<InputType, __half>,
                  "InputType must be float, __nv_bfloat16, or __half");

    int warps_per_row = (num_experts + 64 - 1) / 64;
    int threads_per_row = warps_per_row * 64;
    int row_in_block = threadIdx.x / threads_per_row; 
    const int thread_row = blockIdx.x * rows_per_cta + row_in_block;
    if(thread_row >= num_rows) {
        return;
    }
    const int warp_id = threadIdx.x / 64;
    const int tid_in_row = threadIdx.x % threads_per_row;
    
    float row_val, row_val_for_choice = -3.4e38;
    if(tid_in_row < num_experts) {
        row_val = static_cast<float>(input[thread_row * num_experts + tid_in_row]);
        row_val = 1.0f / (1.0f + __expf(-row_val));
        row_val_for_choice = row_val;
        if(bias != nullptr) {
            const int expert_id = tid_in_row;
            float bias_val = (expert_id < num_experts) ? bias[expert_id] : 0.0f;
            row_val_for_choice = row_val + bias_val;
        }
    }

    bool row_is_active = finished ? !finished[thread_row] : true;
    if (is_padding != nullptr && is_padding[thread_row]) {
        row_is_active = false;
    }
    
    int64_t expert_id_opt = static_cast<int64_t>(tid_in_row);

    float idx_and_weight[2];
    float idx_and_weight_2[2];
    extern __shared__ int8_t shared_memory[];
    float *norm = (float*)shared_memory;
    idx_and_weight[0] = row_val_for_choice;
    idx_and_weight[1] = 0.0f;
    *((int64_t*)idx_and_weight) |= (expert_id_opt << 32);
    warpSortDescendingUpdate<0xffffffffffffffff>(idx_and_weight, threadIdx.x);
    __shared__ float shared_experts[64][2];
    int col_index = tid_in_row % 64;
    if (col_index < topk) {
        *((int64_t*)(shared_experts) + warp_id * topk + col_index) = *(int64_t*)idx_and_weight;
    }
    __syncthreads();

    float max_val_ordered;
    bool should_process_row = false;
    if(tid_in_row < 64){
        idx_and_weight_2[0] = -3.4e38;
        *((int64_t*)idx_and_weight) |= (tid_in_row << 32);
        if(tid_in_row < topk * warps_per_row) {
            *(int64_t*)idx_and_weight_2 = *((int64_t*)shared_experts + (warp_id / warps_per_row) * (topk * warps_per_row) + tid_in_row);
        }
        warpSortDescendingUpdate<0xffffffffffffffff>(idx_and_weight_2, threadIdx.x);
        if (tid_in_row < topk) {
            int64_t res = *(int64_t*)idx_and_weight_2;
            max_val_ordered = get_weight(res);
            int expert_id_ordered = (res >> 32);
            // subtract bias，restore origin row_val(after topk)
            if (bias != nullptr) {
                float bias_val = (expert_id_ordered < num_experts) ? bias[expert_id_ordered] : 0.0f;
                max_val_ordered = max_val_ordered - bias_val;
            }

            // Add a guard to ignore experts not included by this node
            const bool node_uses_expert = expert_id_ordered >= start_expert && expert_id_ordered < end_expert;
            should_process_row = row_is_active && node_uses_expert;

            // The lead thread from each sub-group will write out the final results to global memory. (This will be a
            // single) thread per row of the input/output matrices.
            const int idx = topk * thread_row + tid_in_row;
            indices[idx] = should_process_row ? (expert_id_ordered - start_expert) : num_experts;
            
        // source_rows[idx] = tid_in_row * num_rows + thread_row;
            if (renormalize) {
                norm[row_in_block * topk + tid_in_row] = should_process_row ? static_cast<float>(max_val_ordered) : 0.0f;
            } else {
                output[idx] = should_process_row ? static_cast<float>(max_val_ordered) * static_cast<float>(routed_scaling_factor) : 0.0f;
            }
        }
    }
    
    float* norm_val = norm + (blockDim.x / threads_per_row)*topk;
    if (renormalize) {
        __syncthreads();
        if (tid_in_row == 0){
            norm_val[row_in_block] = 0;
            for (int i = 0; i < topk; i++) {
                norm_val[row_in_block] += norm[row_in_block * topk + i];
            }
            if (norm_val[row_in_block] <= 0.0f) {
                norm_val[row_in_block] = 1.0f;
            }
        }
        __syncthreads();
        if (tid_in_row < topk) {
            const int idx = topk * thread_row + tid_in_row;
            const float denom = norm_val[row_in_block];
            output[idx] = should_process_row ? static_cast<float>(max_val_ordered) / denom * static_cast<float>(routed_scaling_factor) : 0.0f;
        }
    }
}

// ================= Softmax high-perf path (C500) =================
// Adapted from topkGatingSigmoidCommonOpt / topkGatingSigmoid288Opt: one thread
// per expert + 64-wide warp bitonic sort. The only change vs sigmoid is the
// scoring: softmax needs a row-wide max then a row-wide sum-of-exp reduction
// (across warps_per_row warps of a row's thread-group) before the sort. Because
// softmax is monotonic, selection order == logit order, so the identical bitonic
// sort / merge / renormalize tail is reused.
// Layout: ROWS_PER_CTA token-rows per block, each owning a thread-group of
// threads_per_row == warps_per_row*64 lanes. Packing rows raises occupancy so
// the block is large (~512 threads) instead of one tiny row-per-block. Requires
// warps_per_row*topk <= 64 so each row's merge fits a single 64-lane warp.
template <typename IndType, typename InputType = float>
__global__ void topkGatingSoftmaxCommonOpt(
    const InputType* __restrict__ input,
    const bool* __restrict__ finished,
    float* __restrict__ output,
    const int num_rows,
    IndType* __restrict__ indices,
    int* __restrict__ token_expert_indices,
    const int start_expert,
    const int end_expert,
    const int num_experts,
    const int rows_per_cta,
    const bool renormalize,
    const float* __restrict__ bias,
    const int topk,
    const bool* is_padding = nullptr)
{
    constexpr int WARP = 64;
    const int warps_per_row = (num_experts + WARP - 1) / WARP;
    const int threads_per_row = warps_per_row * WARP;

    const int row_in_block = threadIdx.x / threads_per_row;
    const int thread_row = blockIdx.x * rows_per_cta + row_in_block;
    const bool active_row = thread_row < num_rows;

    const int tid = threadIdx.x % threads_per_row;   // 0 .. threads_per_row-1
    const int warp_in_row = tid / WARP;
    const int lane = tid % WARP;
    const bool valid = active_row && (tid < num_experts);
    const bool row_is_active = active_row && (finished == nullptr || !finished[thread_row]) && (is_padding == nullptr || !is_padding[thread_row]);

    // 1) Load this thread's logit (one expert per thread).
    float logit = -3.4e38f;
    if (valid) logit = static_cast<float>(input[thread_row * num_experts + tid]);

    // Per-row shared scratch (row_in_block picks the slice).
    extern __shared__ float smem[];
    // layout: smax[rows][8] | ssum[rows][8] | merge[rows][64] (int64) | norm[rows][16]
    float* smax = smem;                                     // rows_per_cta * 8
    float* ssum = smax + rows_per_cta * 8;                  // rows_per_cta * 8
    int64_t* merge = (int64_t*)(ssum + rows_per_cta * 8);   // rows_per_cta * 64
    float* norm = (float*)(merge + rows_per_cta * 64);      // rows_per_cta * 16
    float* smaxR = smax + row_in_block * 8;
    float* ssumR = ssum + row_in_block * 8;
    int64_t* mergeR = merge + row_in_block * 64;
    float* normR = norm + row_in_block * 16;

    // 2) Row max = max over all warps_per_row warps. Butterfly within the 64-lane
    //    warp, then combine per-warp partials through shared memory.
    float m = logit;
#pragma unroll
    for (int mask = 32; mask > 0; mask >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffffffffffULL, m, mask));
    }
    if (lane == 0) smaxR[warp_in_row] = m;
    __syncthreads();
    float row_max = smaxR[0];
    for (int w = 1; w < warps_per_row; ++w) row_max = fmaxf(row_max, smaxR[w]);

    // 3) Row sum of exp(logit - row_max). Invalid lanes contribute 0.
    float e = valid ? __builtin_expf(logit - row_max) : 0.0f;
    float s = e;
#pragma unroll
    for (int mask = 32; mask > 0; mask >>= 1) {
        s += __shfl_xor_sync(0xffffffffffffffffULL, s, mask);
    }
    if (lane == 0) ssumR[warp_in_row] = s;
    __syncthreads();
    float row_sum = ssumR[0];
    for (int w = 1; w < warps_per_row; ++w) row_sum += ssumR[w];

    float prob = e * __builtin_mxc_rcpf(row_sum);   // softmax probability (0 if invalid)

    // Selection value: prob (+ correction bias if provided). Invalid -> -inf.
    float row_val_for_choice = valid ? prob : -3.4e38f;
    if (valid && bias != nullptr) row_val_for_choice = prob + bias[tid];

    // 4) Per-warp descending bitonic sort of (weight, expert_id) packed in int64.
    float idx_and_weight[2];
    idx_and_weight[0] = row_val_for_choice;
    idx_and_weight[1] = 0.0f;
    *((int64_t*)idx_and_weight) |= (static_cast<int64_t>(tid) << 32);
    warpSortDescendingUpdate<0xffffffffffffffff>(idx_and_weight, lane);

    if (lane < topk) {
        mergeR[warp_in_row * topk + lane] = *(int64_t*)idx_and_weight;
    }
    __syncthreads();

    // 5) Merge the warps_per_row*topk local winners in this row's warp 0.
    if (active_row && tid < WARP) {
        float m2[2];
        m2[0] = -3.4e38f;
        m2[1] = 0.0f;
        *((int64_t*)m2) |= (static_cast<int64_t>(tid) << 32);
        if (tid < warps_per_row * topk) {
            *(int64_t*)m2 = mergeR[tid];
        }
        warpSortDescendingUpdate<0xffffffffffffffff>(m2, lane);

        if (tid < topk) {
            int64_t res = *(int64_t*)m2;
            float w = get_weight(res);
            int eid = static_cast<int>(res >> 32);
            // Restore the unbiased softmax prob for the output weight.
            if (bias != nullptr && eid < num_experts) w = w - bias[eid];

            const bool node_uses_expert = eid >= start_expert && eid < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;
            const int idx = topk * thread_row + tid;
            indices[idx] = should_process_row ? static_cast<IndType>(eid - start_expert) : static_cast<IndType>(num_experts);
            if (token_expert_indices != nullptr) {
                token_expert_indices[idx] = tid * num_rows + thread_row;
            }
            if (renormalize) {
                normR[tid] = should_process_row ? w : 0.0f;
            } else {
                output[idx] = should_process_row ? w : 0.0f;
            }
        }
    }

    // 6) Renormalize the topk weights to sum to 1, if requested.
    if (renormalize) {
        __syncthreads();
        if (active_row && tid < topk) {
            float sum = 0.0f;
#pragma unroll 1
            for (int i = 0; i < topk; ++i) sum += normR[i];
            const float denom = sum > 0.0f ? sum : 1.0f;
            const int idx = topk * thread_row + tid;
            output[idx] = normR[tid] * __builtin_mxc_rcpf(denom);
        }
    }
}

namespace detail
{
// Constructs some constants needed to partition the work across threads at compile time.
template <int EXPERTS, int BYTES_PER_LDG, int WARP_SIZE_PARAM, typename InputType>
struct TopkConstants
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0, "");
    static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static const int ROWS_PER_WARP = WARP_SIZE_PARAM / THREADS_PER_ROW;
};
} // namespace detail

template <int EXPERTS, int WARPS_PER_TB, int WARP_SIZE_PARAM, int MAX_BYTES_PER_LDG, typename IndType, typename InputType, ScoringFunc SF>
void topkGatingLauncherHelper(const InputType* input, const bool* finished, float* output, IndType* indices,
    int* source_row, const int num_rows, const int k, const int start_expert, const int end_expert, const bool renormalize,
    const float* bias, const double routed_scaling_factor, cudaStream_t stream, const bool* is_padding)
{
    static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(InputType) * EXPERTS);
    using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM, InputType>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE_PARAM, WARPS_PER_TB);
    topkGating<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG, WARP_SIZE_PARAM, IndType, InputType, SF><<<num_blocks, block_dim, 0, stream>>>(
        input, finished, output, num_rows, indices, source_row, k, start_expert, end_expert, renormalize, bias, routed_scaling_factor, is_padding);
}

#ifndef USE_ROCM
  // Explicit warp-size variant. On C600U the physical warp is 64 lanes; passing
  // WS=64 lets one physical warp own ROWS_PER_WARP rows explicitly (block_dim.x
  // = WS). Only valid where TopkConstants' divisibility holds for that WS AND
  // THREADS_PER_ROW stays <= 32 (else width-64 butterfly shuffles need the
  // 64-bit mask, which VLLM_SHFL_XOR_SYNC_WIDTH does not use). E=256 @16B keeps
  // THREADS_PER_ROW=32 at WS=64, so it is safe; other shapes stay at WS=32.
  #define LAUNCH_TOPK_WS(NUM_EXPERTS, WARPS_PER_TB, WS, MAX_BYTES)            \
    topkGatingLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, WS, MAX_BYTES,        \
                             IndType, InputType, SF>(                         \
        gating_output, nullptr, topk_weights, topk_indices,                   \
        token_expert_indices, num_tokens, topk, 0, num_experts, renormalize,  \
        bias, routed_scaling_factor, stream, is_padding);
  #define LAUNCH_TOPK(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)                   \
    LAUNCH_TOPK_WS(NUM_EXPERTS, WARPS_PER_TB, 32, MAX_BYTES)
#else
  #define LAUNCH_TOPK(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)                    \
    if (WARP_SIZE == 64) {                                                     \
      topkGatingLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, 64, MAX_BYTES,       \
                               IndType, InputType, SF>(                        \
          gating_output, nullptr, topk_weights, topk_indices,                  \
          token_expert_indices, num_tokens, topk, 0, num_experts, renormalize, \
          bias, routed_scaling_factor, stream, is_padding);                                                       \
    } else if (WARP_SIZE == 32) {                                              \
      topkGatingLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, 32, MAX_BYTES,       \
                               IndType, InputType, SF>(                        \
          gating_output, nullptr, topk_weights, topk_indices,                  \
          token_expert_indices, num_tokens, topk, 0, num_experts, renormalize, \
          bias, routed_scaling_factor, stream, is_padding);                                                       \
    } else {                                                                   \
      assert(false &&                                                          \
             "Unsupported warp size. Only 32 and 64 are supported for ROCm");  \
    }
#endif

template <int EXPERTS, int WARPS_PER_TB, int WARP_SIZE_PARAM, typename IndType, typename InputType, ScoringFunc SF>
void topkDecodeGatingSoftmaxLauncherHelper(const InputType* input, const bool* finished, float* output, IndType* indices,
    int* source_row, const int num_rows, const int k, const int start_expert, const int end_expert, const bool renormalize,
    const float* bias, const double routed_scaling_factor, cudaStream_t stream, const bool* is_padding) {
    if (k == 8 and num_rows < 1024){
        static constexpr int MAX_BYTES_PER_LDG = 8;
        static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(float) * EXPERTS);

        using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM, InputType>;
        static constexpr int VPT = Constants::VPT;
        static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
        const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
        const int num_blocks = (num_rows + 1) / 2;
        dim3 block_dim(WARP_SIZE * 2, WARPS_PER_TB);
        topkGatingSoftmaxDecode<IndType, InputType><<<num_blocks, 256, 0, stream>>>(
            input, finished, output, num_rows, indices, source_row, k, start_expert, end_expert, renormalize, bias, k, routed_scaling_factor, is_padding);
    } else {
        static constexpr std::size_t MAX_BYTES_PER_LDG = 16;

        static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(float) * EXPERTS);
        using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM, InputType>;
        static constexpr int VPT = Constants::VPT;
        static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
        const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
        const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

        dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
        topkGating<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG, WARP_SIZE_PARAM, IndType, InputType, SF><<<num_blocks, block_dim, 0, stream>>>(
            input, finished, output, num_rows, indices, source_row, k, start_expert, end_expert, renormalize, bias, routed_scaling_factor, is_padding);
    }
}

#define LAUNCH_SOFTMAX_OPT(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES) \
    topkDecodeGatingSoftmaxLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, WARP_SIZE, IndType, InputType, SF>( \
        gating_output, nullptr, topk_weights, topk_indices, token_expert_indices, num_tokens, topk, 0, num_experts, \
        renormalize, bias, routed_scaling_factor, stream, is_padding);


template <int WARPS_PER_TB, int MAX_BYTES, typename IndType, typename InputType, ScoringFunc SF>
void launchTopk256Decode(const InputType* gating_output, float* topk_weights, IndType* topk_indices,
    int* token_expert_indices, const int num_tokens, const int topk, const bool renormalize,
    const int num_experts, const float* bias, const double routed_scaling_factor, cudaStream_t stream, const bool* is_padding) {
    if constexpr (SF == vllm::moe::SCORING_SOFTMAX) {
        if (topk == 8 && num_tokens < 1024) {
            topkGatingSoftmaxDecode256<IndType, InputType><<<num_tokens, 256, 0, stream>>>(
                gating_output, nullptr, topk_weights, num_tokens, topk_indices, token_expert_indices, topk, 0, num_experts, renormalize, bias, topk, routed_scaling_factor, is_padding);
            return;
        }
        if (topk == 8 && num_tokens >= 1024) {
            topkGatingLauncherHelper<256, 32, 8, MAX_BYTES, IndType, InputType, SF>(
                gating_output, nullptr, topk_weights, topk_indices, token_expert_indices, num_tokens, topk, 0, num_experts,
                renormalize, bias, routed_scaling_factor, stream, is_padding);
            return;
        }
    }
    topkGatingLauncherHelper<256, WARPS_PER_TB, 32, MAX_BYTES, IndType, InputType, SF>(
        gating_output, nullptr, topk_weights, topk_indices, token_expert_indices, num_tokens, topk, 0, num_experts,
        renormalize, bias, routed_scaling_factor, stream, is_padding);
}

template <typename IndType, typename InputType, ScoringFunc SF>
void topkGatingKernelLauncher(
    const InputType* gating_output,
    float* topk_weights,
    IndType* topk_indices,
    int* token_expert_indices,
    float* workspace,
    const int num_tokens,
    const int num_experts,
    const int topk,
    const bool renormalize,
    const float* bias,
    const double routed_scaling_factor,
    cudaStream_t stream,
    const bool* is_padding) {
    static constexpr int WARPS_PER_TB = 4;
    static constexpr int BYTES_PER_LDG_POWER_OF_2 = 16;
#ifndef USE_ROCM
    // for bfloat16 dtype, we need 4 bytes loading to make sure num_experts
    // elements can be loaded by a warp
    static constexpr int BYTES_PER_LDG_MULTIPLE_64 =
    (std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>) ? 4 : 8;
    // For E divisible by 32 but not 64 (160, 288): one element per load
    // (ELTS_PER_LDG==1) so E % (1*32)==0 satisfies TopkConstants' static_assert.
    static constexpr int BYTES_PER_LDG_SCALAR = sizeof(InputType);
#endif
    if constexpr(SF == vllm::moe::SCORING_SOFTMAX) {
        const int warps_per_row = (num_experts + 64 - 1) / 64;
        const bool handled_by_fast_switch =
            (num_experts == 256 || num_experts == 320 ||
             num_experts == 384 || num_experts == 448 || num_experts == 512 ||
             num_experts == 128 || num_experts == 192 || num_experts == 576 ||
             num_experts == 896 ||
             num_experts == 160 || num_experts == 288 ||
             (num_experts & (num_experts - 1)) == 0);
        if (!handled_by_fast_switch && warps_per_row * topk <= 64 && num_experts <= 512) {
            const int threads_per_row = warps_per_row * 64;
            int rows_per_cta = 512 / threads_per_row;       // aim ~512 threads/block
            if (rows_per_cta < 1) rows_per_cta = 1;
            const int block_threads = rows_per_cta * threads_per_row;
            const int num_blocks = (num_tokens + rows_per_cta - 1) / rows_per_cta;
            const size_t smem_bytes = rows_per_cta * ((8 + 8 + 16) * sizeof(float) + 64 * sizeof(int64_t));
            topkGatingSoftmaxCommonOpt<IndType, InputType><<<num_blocks, block_threads, smem_bytes, stream>>>(
                (const InputType*)gating_output, nullptr, topk_weights, num_tokens,
                topk_indices, token_expert_indices, 0, num_experts, num_experts,
                rows_per_cta, renormalize, bias, topk, is_padding);
            return;
        }
    }
    if constexpr(SF == vllm::moe::SCORING_SIGMOID) {
        if (num_experts == 288 && topk == 8) {
            constexpr int ROWS_PER_CTA = 2;
            constexpr int THREADS_PER_ROW = 5 * 64; // 320
            constexpr int BLOCK_THREADS = ROWS_PER_CTA * THREADS_PER_ROW; // 640 threads per block
            
            const int num_blocks = (num_tokens + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
            topkGatingSigmoid288Opt<IndType, InputType><<<num_blocks, BLOCK_THREADS, 0, stream>>>(
                (const InputType*)gating_output, 
                nullptr, 
                topk_weights, 
                num_tokens, 
                topk_indices,
                0,             
                num_experts,   
                renormalize, 
                bias,
                routed_scaling_factor,
                is_padding
            );
            return;
        } else if(num_experts <= 256 && topk <=16 ) {
            int warps_per_row = (num_experts + 64 - 1) / 64;
            int threads_per_row = warps_per_row * 64;
            int blocksize = 256;
            int rows_per_cta = blocksize / threads_per_row;
            int block_threads = rows_per_cta * threads_per_row;
            int num_blocks = (num_tokens + rows_per_cta - 1) / rows_per_cta;

            topkGatingSigmoidCommonOpt<1, IndType, InputType><<<num_blocks, block_threads, sizeof(float) * (rows_per_cta * topk + rows_per_cta), stream>>>(
                (const InputType*)gating_output, nullptr, topk_weights, num_tokens, topk_indices, 0, num_experts, num_experts, rows_per_cta, renormalize, bias, topk, routed_scaling_factor, is_padding);
            return; 
        }
    }
    switch (num_experts) {
        case 1:
            LAUNCH_TOPK(1, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 2:
            LAUNCH_TOPK(2, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 4:
            LAUNCH_TOPK(4, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 8:
            LAUNCH_TOPK(8, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 16:
            LAUNCH_TOPK(16, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 32:
            LAUNCH_TOPK(32, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 64:
            LAUNCH_TOPK(64, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 128:
            LAUNCH_SOFTMAX_OPT(128, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 256:
            // Decode regime (T<1024, k=8, softmax): dedicated 16x4=64-lane
            // topkGatingSoftmaxDecode256. Prefill: generic topkGating (WS=32).
            launchTopk256Decode<WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2, IndType, InputType, SF>(
                gating_output, topk_weights, topk_indices, token_expert_indices,
                num_tokens, topk, renormalize, num_experts, bias, routed_scaling_factor,stream, is_padding);
            break;
        case 512:
            LAUNCH_TOPK(512, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        // (CUDA only) support multiples of 64 when num_experts is not power of 2.
        // ROCm uses WARP_SIZE 64 so 8 bytes loading won't fit for some of num_experts,
        // alternatively we can test 4 bytes loading and enable it in future.
#ifndef USE_ROCM
        case 192:
            LAUNCH_TOPK(192, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
            break;
        case 320:
            LAUNCH_TOPK(320, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
            break;
        case 384:
            LAUNCH_TOPK(384, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
            break;
        case 448:
            LAUNCH_TOPK(448, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
            break;
        case 576:
            LAUNCH_TOPK(576, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
            break;
        case 896:
            LAUNCH_TOPK(896, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
            break;
        case 160:
            LAUNCH_TOPK(160, WARPS_PER_TB, BYTES_PER_LDG_SCALAR);
            break;
        case 288:
            LAUNCH_TOPK(288, WARPS_PER_TB, BYTES_PER_LDG_SCALAR);
            break;
#endif
        default: {
            TORCH_CHECK(workspace != nullptr,
                "workspace must be provided for num_experts that are not a power of 2 or multiple of 64.");
            static constexpr int TPB = 256;
            if constexpr (SF == SCORING_SOFTMAX) {
              moeSoftmax<TPB, InputType><<<num_tokens, TPB, 0, stream>>>(
                gating_output, nullptr, workspace, num_experts);
            } else if constexpr (SF == SCORING_SIGMOID) {
              moeSigmoid<TPB, InputType><<<num_tokens, TPB, 0, stream>>>(
                gating_output, nullptr, workspace, num_experts);
            } else {
                TORCH_CHECK(false, "Unsupported scoring func");
            }
            moeTopK<TPB><<<num_tokens, TPB, 0, stream>>>(
                workspace, nullptr, topk_weights, topk_indices, token_expert_indices,
                num_experts, topk, 0, num_experts, renormalize, bias, routed_scaling_factor, is_padding);
        }
    }
}

} // namespace moe
} // namespace vllm


template<typename ComputeType, vllm::moe::ScoringFunc SF>
void dispatch_topk_launch(
    torch::Tensor& gating_output,
    torch::Tensor& topk_weights,
    torch::Tensor& topk_indices,
    torch::Tensor& token_expert_indices,
    torch::Tensor& softmax_workspace,
    int num_tokens, int num_experts, int topk, bool renormalize,
    std::optional<torch::Tensor> bias,
    double routed_scaling_factor,
    cudaStream_t stream,
    std::optional<torch::Tensor> is_padding)
 {
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
      const torch::Tensor& bias_tensor = bias.value();
      TORCH_CHECK(bias_tensor.scalar_type() == at::ScalarType::Float, "bias tensor must be float32");
      TORCH_CHECK(bias_tensor.dim() == 1, "bias tensor must be 1D");
      TORCH_CHECK(bias_tensor.size(0) == num_experts, "bias size mismatch, expected: ", num_experts);
      TORCH_CHECK(bias_tensor.is_contiguous(), "bias tensor must be contiguous");
      bias_ptr = bias_tensor.data_ptr<float>();
    }

    const bool* is_padding_ptr = nullptr;
    if (is_padding.has_value()) {
      const torch::Tensor& is_padding_tensor = is_padding.value();
      TORCH_CHECK(is_padding_tensor.scalar_type() == torch::headeronly::ScalarType::Bool,
                      "is_padding tensor must be bool");
      TORCH_CHECK(is_padding_tensor.dim() == 1, "is_padding tensor must be 1D");
      TORCH_CHECK(is_padding_tensor.size(0) == num_tokens,
                      "is_padding size mismatch, expected: ", num_tokens);
      TORCH_CHECK(is_padding_tensor.is_contiguous(), "is_padding tensor must be contiguous");
      is_padding_ptr = is_padding_tensor.const_data_ptr<bool>();
    }

    if (topk_indices.scalar_type() == at::ScalarType::Int) {
        vllm::moe::topkGatingKernelLauncher<int, ComputeType, SF>(
            reinterpret_cast<const ComputeType*>(gating_output.data_ptr()),
            topk_weights.data_ptr<float>(),
            topk_indices.data_ptr<int>(),
            token_expert_indices.data_ptr<int>(),
            softmax_workspace.data_ptr<float>(),
            num_tokens, num_experts, topk, renormalize,
            bias_ptr, routed_scaling_factor, stream, is_padding_ptr);
    } else if (topk_indices.scalar_type() == at::ScalarType::UInt32) {
        vllm::moe::topkGatingKernelLauncher<uint32_t, ComputeType, SF>(
            reinterpret_cast<const ComputeType*>(gating_output.data_ptr()),
            topk_weights.data_ptr<float>(),
            topk_indices.data_ptr<uint32_t>(),
            token_expert_indices.data_ptr<int>(),
            softmax_workspace.data_ptr<float>(),
            num_tokens, num_experts, topk, renormalize,
            bias_ptr, routed_scaling_factor, stream, is_padding_ptr);
    } else {
        TORCH_CHECK(topk_indices.scalar_type() == at::ScalarType::Long);
        vllm::moe::topkGatingKernelLauncher<int64_t, ComputeType, SF>(
            reinterpret_cast<const ComputeType*>(gating_output.data_ptr()),
            topk_weights.data_ptr<float>(),
            topk_indices.data_ptr<int64_t>(),
            token_expert_indices.data_ptr<int>(),
            softmax_workspace.data_ptr<float>(),
            num_tokens, num_experts, topk, renormalize,
            bias_ptr, routed_scaling_factor, stream, is_padding_ptr);
    }
}

void topk_softmax(
    torch::Tensor& topk_weights,                // [num_tokens, topk]
    torch::Tensor& topk_indices,                // [num_tokens, topk]
    torch::Tensor& token_expert_indices,        // [num_tokens, topk]
    torch::Tensor& gating_output,               // [num_tokens, num_experts]
    bool renormalize,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> is_padding)
{
    const int num_experts = gating_output.size(-1);
    const auto num_tokens = gating_output.numel() / num_experts;
    const int topk = topk_weights.size(-1);

    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    const bool needs_workspace = !is_pow_2 || num_experts > 256;
    const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto workspace_options = gating_output.options().dtype(at::ScalarType::Float);
    torch::Tensor softmax_workspace = torch::empty({workspace_size}, workspace_options);

    if (gating_output.scalar_type() == at::ScalarType::Float) {
        dispatch_topk_launch<float, vllm::moe::SCORING_SOFTMAX>(gating_output, topk_weights, topk_indices,
            token_expert_indices, softmax_workspace, num_tokens, num_experts, topk, renormalize,
            bias, 1.0, stream, is_padding);
    } else if (gating_output.scalar_type() == at::ScalarType::Half) {
        dispatch_topk_launch<__half, vllm::moe::SCORING_SOFTMAX>(gating_output, topk_weights, topk_indices,
            token_expert_indices, softmax_workspace, num_tokens, num_experts, topk, renormalize,
            bias, 1.0, stream, is_padding);
    } else if (gating_output.scalar_type() == at::ScalarType::BFloat16) {
        dispatch_topk_launch<__nv_bfloat16, vllm::moe::SCORING_SOFTMAX>(gating_output, topk_weights, topk_indices,
            token_expert_indices, softmax_workspace, num_tokens, num_experts, topk, renormalize,
            bias, 1.0, stream, is_padding);
    } else {
        TORCH_CHECK(false, "Unsupported gating_output data type: ", gating_output.scalar_type());
    }
}

void topk_sigmoid(
    torch::Tensor& topk_weights,                // [num_tokens, topk]
    torch::Tensor& topk_indices,                // [num_tokens, topk]
    torch::Tensor& token_expert_indices,        // [num_tokens, topk]
    torch::Tensor& gating_output,               // [num_tokens, num_experts]
    bool renormalize,
    std::optional<torch::Tensor> bias,
    double routed_scaling_factor,
    std::optional<torch::Tensor> is_padding)
{

    // ================= 开始插入的单条打印语句 =================
    // std::cout << "[topk_sigmoid DEBUG] "
    //           << "gating_output: shape=" << gating_output.sizes() << ", type=" << gating_output.scalar_type() << " | "
    //           << "topk_weights: shape=" << topk_weights.sizes() << ", type=" << topk_weights.scalar_type() << " | "
    //           << "topk_indices: shape=" << topk_indices.sizes() << ", type=" << topk_indices.scalar_type() << " | "
    //           << "token_expert_indices: shape=" << token_expert_indices.sizes() << ", type=" << token_expert_indices.scalar_type() << " | "
    //           << "renormalize: val=" << (renormalize ? "True" : "False") << ", type=bool | "
    //           << "bias: " << (bias.has_value() ? "shape=" : "None") 
    //           << (bias.has_value() ? bias.value().sizes() : c10::IntArrayRef{})
    //           << (bias.has_value() ? ", type=" : "") 
    //           << (bias.has_value() ? bias.value().scalar_type() : at::ScalarType::Undefined) 
    //           << std::endl;
    // // ==========================================================
    const int num_experts = gating_output.size(-1);
    const auto num_tokens = gating_output.numel() / num_experts;
    const int topk = topk_weights.size(-1);

    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    const bool needs_workspace = !is_pow_2 || num_experts > 256;
    const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto workspace_options = gating_output.options().dtype(at::ScalarType::Float);
    torch::Tensor workspace = torch::empty({workspace_size}, workspace_options);

    if (gating_output.scalar_type() == at::ScalarType::Float) {
        dispatch_topk_launch<float, vllm::moe::SCORING_SIGMOID>(gating_output, topk_weights, topk_indices,
            token_expert_indices, workspace, num_tokens, num_experts, topk, renormalize,
            bias, routed_scaling_factor, stream, is_padding);
    } else if (gating_output.scalar_type() == at::ScalarType::Half) {
        dispatch_topk_launch<__half, vllm::moe::SCORING_SIGMOID>(gating_output, topk_weights, topk_indices,
            token_expert_indices, workspace, num_tokens, num_experts, topk, renormalize,
            bias, routed_scaling_factor, stream, is_padding);
    } else if (gating_output.scalar_type() == at::ScalarType::BFloat16) {
        dispatch_topk_launch<__nv_bfloat16, vllm::moe::SCORING_SIGMOID>(gating_output, topk_weights, topk_indices,
            token_expert_indices, workspace, num_tokens, num_experts, topk, renormalize,
            bias, routed_scaling_factor, stream, is_padding);
    } else {
        TORCH_CHECK(false, "Unsupported gating_output data type: ", gating_output.scalar_type());
    }
}
