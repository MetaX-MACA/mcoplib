#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace {

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;

constexpr int kSmemInputSize = 2048;
constexpr size_t kSmemHigh = 3 * kSmemInputSize * sizeof(uint32_t);  // 24KB
constexpr size_t kSmemLow = 2 * kSmemInputSize * sizeof(uint32_t);   // 16KB

inline bool device_prefers_high_smem_scheme() {
  static const bool prefers_high = [] {
    int device = 0;
    if (::cudaGetDevice(&device) != cudaSuccess) {
      return false;  // conservative: fall back to the 32KB scheme
    }
    int max_smem_per_block = 0;
    if (::cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) !=
        cudaSuccess) {
      return false;
    }
    return max_smem_per_block > 96 * 1024;  // > 96KB -> 48KB HIGH; <= 64KB -> 32KB LOW
  }();
  return prefers_high;
}

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, TopK]
  int32_t* __restrict__ lengths;           // [B]
  int64_t input_stride;
};

// when length <= TopK, we can directly write the indices
__device__ void naive_topk_cuda(const float* __restrict__ score, int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform(
    const float* __restrict__ score,
    int32_t length,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform_ragged(
    const float* __restrict__ score, int32_t length, int32_t* __restrict__ topk_indices_ragged, int32_t offset) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    topk_indices_ragged[i] = (i < length) ? static_cast<int32_t>(i) + offset : -1;
  }
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint10(float x) -> uint16_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint16_t>(key >> 6);
}


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


__device__ void fast_topk_cuda_tl_exact(
    const float* __restrict__ input, int32_t* __restrict__ index, int row_start, int length) {
  int topk = TopK;
  constexpr auto BLOCK_SIZE = kThreadsPerBlock;
  constexpr auto RADIX = 1024;
  constexpr auto LOG_RADIX = 10;
  constexpr auto SMEM_INPUT_SIZE = kSmemInputSize;

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];
  // Round 2: cache candidate raw_input values in smem so stage-2 refine
  // passes 2..4 don't re-read row_input[idx] from global. Stage 1 already
  // streamed the full input once; stage 2 round 1 fills this cache, and
  // subsequent rounds read from smem instead of re-issuing global loads.
  // 2 * 2048 * 4B = 16KB static smem (fits alongside 16KB/24KB dynamic).
  alignas(128) __shared__ float s_input_value[2][SMEM_INPUT_SIZE];

  auto& s_histogram = s_histogram_buf[0];
  // allocate for two rounds
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

  // stage 1: coarse histogram (packed convert on the vectorized bulk).
  // Round 5: process 8 floats (2x float4 = 32B) per iteration per thread.
  // C500 guide recommends >= 32B per thread per access; vec4 (16B) underfills
  // the load pipeline. Two ldg.b128 issued back-to-back expose more ILP and
  // halve loop overhead. 8 atomicAdd per iter (vs 4), but total atomic count
  // is unchanged — only the loop trip count drops.
  for (int i = tx; i < RADIX + 1; i += BLOCK_SIZE) {
    s_histogram[i] = 0;
  }
  __syncthreads();

  for (int idx = tx; idx < vec4_prefix; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint10(row_input[idx]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  constexpr int VEC8_STRIDE = 2;  // 2 float4 = 8 floats per vec_idx
  const int vec8_length = vec4_length / VEC8_STRIDE;
  const int vec8_tail_start = vec4_prefix + vec8_length * VEC8_STRIDE * 4;
  for (int vec_idx = tx; vec_idx < vec8_length; vec_idx += BLOCK_SIZE) {
    const int base = vec_idx * VEC8_STRIDE;
    const auto v0 = row_input_vec4[base];
    const auto v1 = row_input_vec4[base + 1];
    uint16_t k0, k1, k2, k3, k4, k5, k6, k7;
    convert_to_uint10x2(v0.x, v0.y, k0, k1);
    convert_to_uint10x2(v0.z, v0.w, k2, k3);
    convert_to_uint10x2(v1.x, v1.y, k4, k5);
    convert_to_uint10x2(v1.z, v1.w, k6, k7);
    ::atomicAdd(&s_histogram[k0], 1);
    ::atomicAdd(&s_histogram[k1], 1);
    ::atomicAdd(&s_histogram[k2], 1);
    ::atomicAdd(&s_histogram[k3], 1);
    ::atomicAdd(&s_histogram[k4], 1);
    ::atomicAdd(&s_histogram[k5], 1);
    ::atomicAdd(&s_histogram[k6], 1);
    ::atomicAdd(&s_histogram[k7], 1);
  }
  // Handle remaining vec4 elements (when vec4_length is odd)
  for (int vec_idx = vec8_length * VEC8_STRIDE + tx; vec_idx < vec4_length; vec_idx += BLOCK_SIZE) {
    const auto values = row_input_vec4[vec_idx];
    uint16_t k0, k1, k2, k3;
    convert_to_uint10x2(values.x, values.y, k0, k1);
    convert_to_uint10x2(values.z, values.w, k2, k3);
    ::atomicAdd(&s_histogram[k0], 1);
    ::atomicAdd(&s_histogram[k1], 1);
    ::atomicAdd(&s_histogram[k2], 1);
    ::atomicAdd(&s_histogram[k3], 1);
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

  const auto run_refine_cumsum = [&] {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (tx < 256) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < 256 - j) value += s_histogram_buf[k][tx + j];
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
    if (tx == 0) s_histogram[256] = 0;
    __syncthreads();
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
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
        if (pos >= 0 && pos < TopK) index[pos] = idx;
        //index[pos] = idx;
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
        // const auto pos = ::atomicAdd(&s_num_input[0], 1);
        // /// NOTE: (dark) fuse the histogram computation here
        // if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
        //   s_input_idx[0][pos] = idx;
        //   const auto bin32 = convert_to_uint32(raw_input);
        //   const auto sub_bin = (bin32 >> 24) & 0xFF;
        //   ::atomicAdd(&s_histogram[sub_bin], 1);
        // }
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        // FIX-C: 直方图对“所有”阈值桶候选计数，即使缓冲已满也计。缓冲满只影响
        // 能否进入下一轮精细化，不该影响阈值账目——否则阈值搜索不收敛。
        const auto bin32 = convert_to_uint32(raw_input);
        const auto sub_bin = (bin32 >> 24) & 0xFF;
        ::atomicAdd(&s_histogram[sub_bin], 1);
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          s_input_value[0][pos] = raw_input;
        }
      }
    };
    for (int idx = tx; idx < vec4_prefix; idx += BLOCK_SIZE) {
      const auto raw_input = row_input[idx];
      append_or_stage_candidate(idx, raw_input, convert_to_uint10(raw_input));
    }
    // Round 5: vec8 staging pass — same 32B-per-thread access pattern as stage 1.
    for (int vec_idx = tx; vec_idx < vec8_length; vec_idx += BLOCK_SIZE) {
      const int base = vec_idx * VEC8_STRIDE;
      const auto v0 = row_input_vec4[base];
      const auto v1 = row_input_vec4[base + 1];
      const int idx0 = vec4_prefix + (base + 0) * 4;
      const int idx1 = vec4_prefix + (base + 1) * 4;
      uint16_t k0, k1, k2, k3, k4, k5, k6, k7;
      convert_to_uint10x2(v0.x, v0.y, k0, k1);
      convert_to_uint10x2(v0.z, v0.w, k2, k3);
      convert_to_uint10x2(v1.x, v1.y, k4, k5);
      convert_to_uint10x2(v1.z, v1.w, k6, k7);
      append_or_stage_candidate(idx0 + 0, v0.x, k0);
      append_or_stage_candidate(idx0 + 1, v0.y, k1);
      append_or_stage_candidate(idx0 + 2, v0.z, k2);
      append_or_stage_candidate(idx0 + 3, v0.w, k3);
      append_or_stage_candidate(idx1 + 0, v1.x, k4);
      append_or_stage_candidate(idx1 + 1, v1.y, k5);
      append_or_stage_candidate(idx1 + 2, v1.z, k6);
      append_or_stage_candidate(idx1 + 3, v1.w, k7);
    }
    for (int vec_idx = vec8_length * VEC8_STRIDE + tx; vec_idx < vec4_length; vec_idx += BLOCK_SIZE) {
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
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_refine_cumsum();
    if (tx < 256 && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
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
        const auto raw_input = s_input_value[r_idx][i];
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          if (pos >= 0 && pos < TopK) index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      for (int i = tx; i < 257; i += BLOCK_SIZE) {
        s_histogram[i] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = s_input_value[r_idx][i];
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            // if (pos > 0) {
            //   index[TopK - pos] = idx;
            // }
            const int w = TopK - pos;
            if (pos > 0 && w >= 0 && w < TopK) index[w] = idx;
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              /// NOTE: (dark) fuse the histogram computation here
              s_input_idx[r_idx ^ 1][pos] = idx;
              s_input_value[r_idx ^ 1][pos] = raw_input;
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

// Both prefill schemes now delegate to the single exact core. The HIGH/LOW split is
// retained only so the launch smem (48KB vs 32KB) and dispatch stay unchanged.
__device__ __forceinline__ void fast_topk_cuda_tl_low(
    const float* __restrict__ input, int32_t* __restrict__ index, int row_start, int length) {
  fast_topk_cuda_tl_exact(input, index, row_start, length);
}
__device__ __forceinline__ void fast_topk_cuda_tl_high(
    const float* __restrict__ input, int32_t* __restrict__ index, int row_start, int length) {
  fast_topk_cuda_tl_exact(input, index, row_start, length);
}


__global__ __launch_bounds__(kThreadsPerBlock)  // topk
    void topk_kernel(const FastTopKParams params) {
  const auto& [input, row_starts, indices, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_cuda(score, indice, length);
  } else {
    return fast_topk_cuda_tl_low(score, indice, row_start, length);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // decode
    void topk_transform_decode_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride) {
  const auto& [input, _1, _2, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = 0;
  const auto length = lengths[bid];
  const auto src_page_entry = src_page_table + bid * src_stride;
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    for (int i = tid; i < TopK; i += kThreadsPerBlock) s_indices[i] = 0;
    fast_topk_cuda_tl_low(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = (pos_0 >= 0 && pos_0 < length) ? src_page_entry[pos_0] : -1;
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = (pos_1 >= 0 && pos_1 < length) ? src_page_entry[pos_1] : -1;
  }
}

template <bool HIGH_SMEM>
__global__ __launch_bounds__(kThreadsPerBlock)  // prefill
    void topk_transform_prefill_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride,
        const int32_t* __restrict__ cu_seqlens_q,
        const int64_t prefill_bs) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;

  __shared__ const int32_t* s_src_page_entry;
  if (C10_LIKELY(prefill_bs <= kThreadsPerBlock)) {
    if (tid < prefill_bs) {
      if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
      if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
        s_src_page_entry = src_page_table + i * src_stride;
      }
    }
  }
  __syncthreads();
  const auto src_page_entry = s_src_page_entry;

  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    // Zero-init every slot before the select: the candidate selection writes only the
    // chosen top-k positions; any slot left unwritten (defensive, in case a scheme stages
    // fewer than TopK) would carry stale shared memory and feed an out-of-bounds gather.
    // 0 is always an in-range page-table index, so an unfilled slot degrades to a benign
    // duplicate rather than an illegal address.
    for (int i = tid; i < TopK; i += kThreadsPerBlock) s_indices[i] = 0;
    __syncthreads();
    if constexpr (HIGH_SMEM) {
      fast_topk_cuda_tl_high(score, s_indices, row_start, length);
    } else {
      fast_topk_cuda_tl_low(score, s_indices, row_start, length);
    }
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = (pos_0 >= 0 && pos_0 < length) ? src_page_entry[pos_0] : -1;
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = (pos_1 >= 0 && pos_1 < length) ? src_page_entry[pos_1] : -1;
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill, ragged kv
    void topk_transform_prefill_ragged_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ topk_indices_ragged,
        const int32_t* __restrict__ topk_indices_offset) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto dst_indices_entry = topk_indices_ragged + bid * TopK;
  const auto score = input + bid * input_stride;
  const auto offset = topk_indices_offset[bid];

  if (length <= TopK) {
    return naive_topk_transform_ragged(score, length, dst_indices_entry, offset);
  } else {
    __shared__ int s_indices[TopK];
    for (int i = tid; i < TopK; i += kThreadsPerBlock) s_indices[i] = 0;
    fast_topk_cuda_tl_low(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_indices_entry[idx_0] = (pos_0 >= 0 && pos_0 < length) ? pos_0 + offset : -1;  // FIX-B
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_indices_entry[idx_1] = (pos_1 >= 0 && pos_1 < length) ? pos_1 + offset : -1;  // FIX-B
  }
}

auto get_params(
    const at::Tensor& score,
    const at::Tensor& lengths,
    std::optional<at::Tensor> row_starts_opt = std::nullopt,
    std::optional<at::Tensor> indices_opt = std::nullopt) -> FastTopKParams {
  const auto B = score.size(0);
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
  if (row_starts_opt.has_value()) {
    const auto& row_starts = row_starts_opt.value();
    TORCH_CHECK(row_starts.dim() == 1);
    TORCH_CHECK(row_starts.size(0) == B);
  }
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
  TORCH_CHECK(lengths.size(0) == B);
  int32_t* indices_data_ptr = nullptr;
  if (indices_opt.has_value()) {
    const auto& indices = indices_opt.value();
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
    TORCH_CHECK(indices.size(0) == B);
    TORCH_CHECK(indices.size(1) == TopK);
    indices_data_ptr = indices.data_ptr<int32_t>();
  }

  return FastTopKParams{
      .input = score.data_ptr<float>(),
      .row_starts = row_starts_opt.has_value() ? row_starts_opt->data_ptr<int32_t>() : nullptr,
      .indices = indices_data_ptr,
      .lengths = lengths.data_ptr<int32_t>(),
      .input_stride = score.stride(0),
  };
}

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
#ifdef USE_ROCM
    // hipify will turn cudaFuncSetAttribute -> hipFuncSetAttribute. On ROCm,
    // hipFuncSetAttribute expects `const void*` and hipcc does not accept passing
    // a function pointer directly, so cast explicitly.
    return ::cudaFuncSetAttribute(
        reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
    // CUDA: keep original behavior (no cast needed).
    return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
  }();
  TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

}  // namespace

// Decode TopK transform lives in an isolated TU (topk_decode.cu) so it can be
// optimized without touching this file (prefill / ragged / fast_topk_cuda_tl).
// Cross-TU surface is raw pointers only.
void fast_topk_transform_decode_launch(
    const float* input,
    const int32_t* lengths,
    int64_t input_stride,
    int32_t* dst_page_table,
    const int32_t* src_page_table,
    int64_t src_stride,
    uint32_t B,
    cudaStream_t stream);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void fast_topk_interface(
    const at::Tensor& score, at::Tensor& indices, const at::Tensor& lengths, std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(indices);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  CHECK_CUDA(lengths);
  const auto params = get_params(score, lengths, row_starts_opt, indices);
  const auto B = score.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  // topk_kernel uses the LOW (32KB) scheme so it runs on <= 64KB-smem GPUs too.
  setup_kernel_smem_once<topk_kernel, kSmemLow>();
  topk_kernel<<<grid, block, kSmemLow, stream>>>(params);
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(dst_page_table);
  CHECK_CUDA(src_page_table);
  CHECK_CUDA(cu_seqlens_q);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(dst_page_table.dim() == 2 && dst_page_table.is_contiguous());
  TORCH_CHECK(src_page_table.dim() == 2 && src_page_table.stride(1) == 1);
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous());
  const auto prefill_bs = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(dst_page_table.size(0) == B);
  TORCH_CHECK(dst_page_table.size(1) == TopK);
  TORCH_CHECK(src_page_table.size(0) == prefill_bs);
  TORCH_CHECK(prefill_bs <= B);  // prefill_bs should be smaller than expanded bs

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  const auto src_stride = src_page_table.stride(0);

  // dispatch to decode or prefill
  // extend and draft extend: row_starts_opt is not null, invokes the prefill kernel
  // decode: row_starts_opt is null, invokes the decode kernel
  // target verify: row_starts_opt is null, invokes the prefill kernel
  const auto is_decode = !row_starts_opt.has_value() && prefill_bs == B;
  if (is_decode) {
    // isolated decode TU owns this path (see topk_decode.cu)
    fast_topk_transform_decode_launch(
        params.input, params.lengths, params.input_stride, dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(), src_stride, static_cast<uint32_t>(B), stream);
  } else {
    // Device-adaptive: GPUs with > 96KB optin smem run the 24KB HIGH scheme; GPUs
    // with <= 64KB smem run the 16KB LOW scheme. Both templates are instantiated.
    if (device_prefers_high_smem_scheme()) {
      setup_kernel_smem_once<topk_transform_prefill_kernel<true>, kSmemHigh>();
      topk_transform_prefill_kernel<true><<<grid, block, kSmemHigh, stream>>>(
          params,
          dst_page_table.data_ptr<int32_t>(),
          src_page_table.data_ptr<int32_t>(),
          src_stride,
          cu_seqlens_q.data_ptr<int32_t>(),
          prefill_bs);
    } else {
      setup_kernel_smem_once<topk_transform_prefill_kernel<false>, kSmemLow>();
      topk_transform_prefill_kernel<false><<<grid, block, kSmemLow, stream>>>(
          params,
          dst_page_table.data_ptr<int32_t>(),
          src_page_table.data_ptr<int32_t>(),
          src_stride,
          cu_seqlens_q.data_ptr<int32_t>(),
          prefill_bs);
    }
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_ragged_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& topk_indices_ragged,
    const at::Tensor& topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(topk_indices_ragged);
  CHECK_CUDA(topk_indices_offset);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }

  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(topk_indices_ragged.dim() == 2 && topk_indices_ragged.is_contiguous());
  TORCH_CHECK(topk_indices_offset.dim() == 1);

  TORCH_CHECK(topk_indices_ragged.size(0) == B);
  TORCH_CHECK(topk_indices_ragged.size(1) == TopK);
  TORCH_CHECK(topk_indices_offset.size(0) == B);

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};

  setup_kernel_smem_once<topk_transform_prefill_ragged_kernel, kSmemLow>();
  topk_transform_prefill_ragged_kernel<<<grid, block, kSmemLow, stream>>>(
      params, topk_indices_ragged.data_ptr<int32_t>(), topk_indices_offset.data_ptr<int32_t>());

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}
