// Optimized topk_transform_kernel for MetaX C500/C280 (SM80, warp_size=64).
//
// Two algorithmic branches selected at launch time by smem budget:
//
//   A. histogram_4096 (single HBM pass, register-based): for seq_len <= 16384.
//      Each thread loads 16 floats (4 float4) into registers, builds a 4096-bin
//      coarse histogram in shared memory, scans it for the threshold bin, then
//      scatters from registers — no second HBM read. This is the bandwidth
//      winner: peak 1.875 TB/s HBM, single-pass → ~800 GB/s achievable.
//
//   B. radix_256 (two-pass): for seq_len > 16384. Coarse 8-bit FP16 histogram
//      (pass 1) + 4-round FP32 radix refine (pass 2). The threshold-bin
//      candidates are re-read from HBM in pass 2, but L2 caches the row when
//      it fits (<=8MB). For seq=66551 (256KB/row) and bs>=132 this thrashes
//      L2; bandwidth capped ~300 GB/s.
//
// Both branches fuse a page-table index transform (page_to_indices) on the
// selected indices. For seq_len <= TopK a naive transform pads with -1.
//
// Ported from the JIT implementation in op/sglang/jit_kernels/topk_v1.cuh,
// with the register-based single-pass technique adapted from
// op/vllm/topk_histogram_4096.cuh (rewritten for warp_size=64).

#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <optional>

#include "topk_v1_histogram_4096.cuh"

namespace {

constexpr int TopK = 512;
constexpr int kThreadsPerBlock = 1024;  // matches histogram_4096 + persistent_topk
constexpr int kWarpSize = 64;           // MACA warp size
constexpr int kNumWarpsBlock = kThreadsPerBlock / kWarpSize;  // 16

// Dynamic smem for the 2-pass radix candidate staging buffer.
// 32KB = fast_topk_cuda_tl budget. MACA runtime rejects 64KB dynamic.
constexpr size_t kSmem = 8 * 1024 * sizeof(uint32_t);  // 32KB (bytes)
constexpr size_t kSmemInputSize = kSmem / (2 * sizeof(int32_t));  // 4096

// Larger staging buffer for chips with >= 128KB smem (e.g. the 128K-smem
// MetaX variant). Mirrors vllm FilteredTopK's 2 x 7K = 56KB dynamic smem
// (sizeof(int) * 2 * (7 * 1024) + 2048 bytes, see op/vllm/topk.cu:38).
// 14336 = 7 * 1024 entries per buffer; 2 buffers x 14336 x 4B = 112KB ?
// No: kFilteredSmemInputSize * 2 * sizeof(int) = 14336 * 2 * 4 = 112KB.
// That exceeds even 128KB. Use 7K per buffer = 56KB total dynamic, fits 64KB
// static+dynamic on a 128KB-smem chip.
constexpr size_t kLargeSmemInputSize = 7 * 1024;          // 7K per buffer
constexpr size_t kLargeSmem = 2 * kLargeSmemInputSize * sizeof(int32_t);  // 56KB

// Threshold above which the 2-pass radix path runs (instead of histogram_4096).
// histogram_4096 max capacity = kVecsPerThread * 4 * kBlockSize = 16384 floats.
constexpr uint32_t kHist4096MaxLen = topk_v1_hist4096::kMaxLen;

struct TopKTransformParams {
  const float* __restrict__ input;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;
  int64_t input_stride;
  int64_t page_table_stride;
  uint32_t page_bits;
};

// 8-bit coarse bin from FP16 sign-magnitude key. FP16's 5-bit exponent + 10-bit
// mantissa gives finer 8-bit binning (5 exp + 3 mantissa bits) than raw FP32
// high-8 (8 exponent bits, no mantissa) — the latter lumps all values in the
// same power-of-2 range into one bin, overflowing the staging buffer at the
// threshold bin. The fp16 detour costs a few cycles but the much smaller
// threshold-bin candidate set more than pays for it.
__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ int32_t page_to_indices(
    const int32_t* __restrict__ page_table, uint32_t i, uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

__device__ void naive_transform(
    const float* __restrict__ score,
    const int32_t* __restrict__ page_table,
    int32_t* __restrict__ indices,
    int32_t* __restrict__ raw_indices,
    const uint32_t length,
    const uint32_t page_bits) {
  const auto tid = threadIdx.x;
  for (uint32_t i = tid; i < TopK; i += kThreadsPerBlock) {
    if (i < length) {
      indices[i] = page_to_indices(page_table, i, page_bits);
      if (raw_indices != nullptr) raw_indices[i] = static_cast<int32_t>(i);
    } else {
      indices[i] = -1;
      if (raw_indices != nullptr) raw_indices[i] = -1;
    }
  }
}

// 2-pass radix-256 TopK for seq_len > 16384. Same structure as the JIT
// topk_v1.cuh radix_topk, with float4 vectorized loads and launch_bounds.
//
// Template parameter kStagingSize controls the dynamic-smem staging buffer
// that holds threshold-bin candidates between pass 1 and pass 2:
//   - 64K-smem chips (C500/C280): kStagingSize = 4096 (32KB dynamic smem)
//   - 128K-smem chips (other MetaX variant): kStagingSize = 7168 (56KB)
//     mirrors vllm FilteredTopK's 2 x 7K = 56KB dynamic smem.
// A larger staging buffer reduces the chance of dropping threshold-bin
// candidates when the bin is dense (a real correctness risk for long rows
// with many equal FP16 keys), and lets pass 2 re-read fewer elements from
// HBM (better L2 behavior).
template <uint32_t kStagingSize>
__device__ void radix_topk(
    const float* __restrict__ input, int32_t* __restrict__ output, const uint32_t length) {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t BLOCK_SIZE = kThreadsPerBlock;

  alignas(128) __shared__ uint32_t _s_histogram_buf[2][RADIX + 32];
  alignas(128) __shared__ uint32_t s_counter;
  alignas(128) __shared__ uint32_t s_threshold_bin_id;
  alignas(128) __shared__ uint32_t s_num_input[2];
  alignas(128) __shared__ int32_t s_last_remain;
  // Per-warp private coarse histograms. 16 warps * 256 bins * 4B = 16KB.
  // Each warp atomicAdds into its OWN 256-bin slice (no cross-warp contention),
  // then a single cross-warp reduction sums them into s_histogram. This cuts
  // atomicAdd contention by ~kNumWarps=16x in pass 1, which is the dominant
  // cost on MACA (fewer atomic units than NVIDIA).
  alignas(128) __shared__ uint32_t s_warp_hist[kNumWarpsBlock][RADIX];

  // Flat dynamic smem; indexed as s_input_idx[buf][i] = s_input_idx_flat[buf * kStagingSize + i].
  // MACA rejects `extern __shared__ T arr[][N]` when N differs across template
  // instantiations compiled in the same TU, so we use a flat 1D array.
  extern __shared__ int32_t s_input_idx_flat[];

  const uint32_t tx = threadIdx.x;
  uint32_t remain_topk = TopK;
  auto& s_histogram = _s_histogram_buf[0];

  // vec4 alignment setup.
  const auto row_addr = reinterpret_cast<uintptr_t>(input);
  const auto vec4_prefix_unclamped =
      static_cast<uint32_t>((alignof(float4) - (row_addr & (alignof(float4) - 1))) / sizeof(float));
  const auto vec4_prefix =
      (row_addr & (alignof(float4) - 1)) == 0 ? 0u
                                              : (length < vec4_prefix_unclamped ? length : vec4_prefix_unclamped);
  const auto row_input_vec4 = reinterpret_cast<const float4*>(input + vec4_prefix);
  const auto vec4_length = (length - vec4_prefix) / 4;
  const auto vec4_tail = vec4_prefix + vec4_length * 4;

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int32_t i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = _s_histogram_buf[k][tx];
        if (tx + j < RADIX) {
          value += _s_histogram_buf[k][tx + j];
        }
        _s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  // Pass 1: 8-bit coarse histogram (vec4 vectorized).
  // Warp-private histograms: each warp atomicAdds into its own 256-bin slice
  // (s_warp_hist[warp_id]), eliminating cross-warp contention. After the row
  // is consumed, reduce the 16 warp histograms into the global s_histogram.
  // Each warp zero-initializes its slice: 256 bins / 64 lanes = 4 bins/lane.
  const uint32_t warp_id_p1 = tx / kWarpSize;
  const uint32_t lane_id_p1 = tx % kWarpSize;
#pragma unroll
  for (uint32_t i = 0; i < RADIX / kWarpSize; i++) {
    s_warp_hist[warp_id_p1][lane_id_p1 + i * kWarpSize] = 0;
  }
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  if (tx == 0) {
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();
  for (uint32_t idx = tx; idx < vec4_prefix; idx += BLOCK_SIZE) {
    ::atomicAdd(&s_warp_hist[warp_id_p1][convert_to_uint8(input[idx])], 1);
  }
  for (uint32_t vec_idx = tx; vec_idx < vec4_length; vec_idx += BLOCK_SIZE) {
    const auto values = row_input_vec4[vec_idx];
    ::atomicAdd(&s_warp_hist[warp_id_p1][convert_to_uint8(values.x)], 1);
    ::atomicAdd(&s_warp_hist[warp_id_p1][convert_to_uint8(values.y)], 1);
    ::atomicAdd(&s_warp_hist[warp_id_p1][convert_to_uint8(values.z)], 1);
    ::atomicAdd(&s_warp_hist[warp_id_p1][convert_to_uint8(values.w)], 1);
  }
  for (uint32_t idx = vec4_tail + tx; idx < length; idx += BLOCK_SIZE) {
    ::atomicAdd(&s_warp_hist[warp_id_p1][convert_to_uint8(input[idx])], 1);
  }
  __syncthreads();
  // Cross-warp reduction: 16 warps * 256 bins. 256 threads (4 warps) handle
  // 256 bins, each summing across 16 warps. Then s_histogram is the full
  // coarse histogram.
  if (tx < RADIX) {
    uint32_t sum = 0;
#pragma unroll
    for (uint32_t w = 0; w < kNumWarpsBlock; w++) {
      sum += s_warp_hist[w][tx];
    }
    s_histogram[tx] = sum;
  }
  __syncthreads();
  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
    s_threshold_bin_id = tx;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  remain_topk -= s_histogram[threshold_bin + 1];
  if (remain_topk == 0) {
    for (uint32_t idx = tx; idx < vec4_prefix; idx += BLOCK_SIZE) {
      if (convert_to_uint8(input[idx]) > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = idx;
      }
    }
    for (uint32_t vec_idx = tx; vec_idx < vec4_length; vec_idx += BLOCK_SIZE) {
      const auto values = row_input_vec4[vec_idx];
      const uint32_t base = vec4_prefix + vec_idx * 4;
      if (convert_to_uint8(values.x) > threshold_bin) { const auto pos = ::atomicAdd(&s_counter, 1); output[pos] = static_cast<int32_t>(base + 0); }
      if (convert_to_uint8(values.y) > threshold_bin) { const auto pos = ::atomicAdd(&s_counter, 1); output[pos] = static_cast<int32_t>(base + 1); }
      if (convert_to_uint8(values.z) > threshold_bin) { const auto pos = ::atomicAdd(&s_counter, 1); output[pos] = static_cast<int32_t>(base + 2); }
      if (convert_to_uint8(values.w) > threshold_bin) { const auto pos = ::atomicAdd(&s_counter, 1); output[pos] = static_cast<int32_t>(base + 3); }
    }
    for (uint32_t idx = vec4_tail + tx; idx < length; idx += BLOCK_SIZE) {
      if (convert_to_uint8(input[idx]) > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    auto step = [&](uint32_t idx, float raw_input) {
      const uint32_t bin = convert_to_uint8(raw_input);
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = static_cast<int32_t>(idx);
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (pos < kStagingSize) {
          [[likely]] s_input_idx_flat[0 * kStagingSize + pos] = static_cast<int32_t>(idx);
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    };
    for (uint32_t idx = tx; idx < vec4_prefix; idx += BLOCK_SIZE) {
      step(idx, input[idx]);
    }
    for (uint32_t vec_idx = tx; vec_idx < vec4_length; vec_idx += BLOCK_SIZE) {
      const auto values = row_input_vec4[vec_idx];
      const uint32_t base = vec4_prefix + vec_idx * 4;
      step(base + 0, values.x);
      step(base + 1, values.y);
      step(base + 2, values.z);
      step(base + 3, values.w);
    }
    for (uint32_t idx = vec4_tail + tx; idx < length; idx += BLOCK_SIZE) {
      step(idx, input[idx]);
    }
    __syncthreads();
  }

  // Pass 2: 4-round radix refine.
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    const auto r_idx = round % 2;

    const auto raw_num_input = s_num_input[r_idx];
    const auto num_input = raw_num_input < kStagingSize ? raw_num_input : kStagingSize;

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = remain_topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    remain_topk -= s_histogram[threshold_bin + 1];

    if (remain_topk == 0) {
      for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx_flat[r_idx * kStagingSize + i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) s_histogram[tx] = 0;
      __syncthreads();
      for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx_flat[r_idx * kStagingSize + i];
        const auto raw_input = input[idx];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              output[TopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (pos < kStagingSize) {
              [[likely]] s_input_idx_flat[(r_idx ^ 1) * kStagingSize + pos] = idx;
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

template <bool kUseHist4096, uint32_t kRadixStagingSize = kSmemInputSize>
__global__ __launch_bounds__(kThreadsPerBlock)
    void topk_transform_kernel(const TopKTransformParams params) {
  const auto& [input, seq_lens, page_table, page_indices, raw_indices,
               input_stride, page_table_stride, page_bits] = params;
  const uint32_t work_id = blockIdx.x;

  const uint32_t seq_len = seq_lens[work_id];
  const auto score_ptr = input + work_id * input_stride;
  const auto page_ptr = page_table + work_id * page_table_stride;
  const auto indices_ptr = page_indices + work_id * TopK;
  const auto raw_indices_ptr = raw_indices != nullptr ? raw_indices + work_id * TopK : nullptr;

  if (seq_len <= TopK) {
    naive_transform(score_ptr, page_ptr, indices_ptr, raw_indices_ptr, seq_len, page_bits);
    return;
  }

  // s_topk_indices holds the raw TopK indices (before page transform).
  // For histogram_4096: output goes directly here. For radix: same.
  __shared__ int32_t s_topk_indices[TopK];

  if constexpr (kUseHist4096) {
    // Single-pass register-based 4096-bin histogram TopK.
    // Smem layout: histogram_4096 uses its own Smem struct (16KB histogram
    // union 8KB tie_buffer + scalars). Reuse the dynamic smem region.
    extern __shared__ uint8_t s_dyn_raw[];
    void* hist_smem = static_cast<void*>(s_dyn_raw);
    topk_v1_hist4096::histogram_4096_topk<TopK, 12>(
        score_ptr, s_topk_indices, seq_len, hist_smem);
  } else {
    radix_topk<kRadixStagingSize>(score_ptr, s_topk_indices, seq_len);
  }
  __syncthreads();

  // Page transform.
  const auto tx = threadIdx.x;
  for (uint32_t i = tx; i < TopK; i += kThreadsPerBlock) {
    indices_ptr[i] = page_to_indices(page_ptr, s_topk_indices[i], page_bits);
    if (raw_indices_ptr != nullptr) {
      raw_indices_ptr[i] = s_topk_indices[i];
    }
  }
}

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
    return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
  }();
  TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void topk_transform_v1_interface(
    const at::Tensor& scores,
    const at::Tensor& seq_lens,
    at::Tensor& page_indices,
    const at::Tensor& page_table,
    const int64_t page_size,
    std::optional<at::Tensor> raw_indices_opt) {
  CHECK_CUDA(scores);
  CHECK_CUDA(seq_lens);
  CHECK_CUDA(page_table);
  CHECK_CUDA(page_indices);
  if (raw_indices_opt.has_value()) {
    CHECK_CUDA(raw_indices_opt.value());
  }

  const auto B = scores.size(0);
  TORCH_CHECK(scores.dim() == 2 && scores.stride(1) == 1, "scores must be 2D strided with stride(1)==1");
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.is_contiguous(), "seq_lens must be 1D contiguous");
  TORCH_CHECK(seq_lens.size(0) == B, "seq_lens size mismatch");
  TORCH_CHECK(page_table.dim() == 2 && page_table.stride(1) == 1, "page_table must be 2D strided with stride(1)==1");
  TORCH_CHECK(page_table.size(0) == B, "page_table batch mismatch");
  TORCH_CHECK(page_indices.dim() == 2 && page_indices.is_contiguous(), "page_indices must be 2D contiguous");
  TORCH_CHECK(page_indices.size(0) == B, "page_indices batch mismatch");
  TORCH_CHECK(page_indices.size(1) == TopK, "page_indices.size(1) must equal TopK (512)");

  int32_t* raw_indices_ptr = nullptr;
  if (raw_indices_opt.has_value()) {
    const auto& raw = raw_indices_opt.value();
    TORCH_CHECK(raw.dim() == 2 && raw.is_contiguous(), "raw_indices must be 2D contiguous");
    TORCH_CHECK(raw.size(0) == B && raw.size(1) == TopK, "raw_indices shape mismatch");
    raw_indices_ptr = raw.data_ptr<int32_t>();
  }

  TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0,
              "page_size must be a power of 2");
  uint32_t page_bits = 0;
  for (auto v = static_cast<uint32_t>(page_size); v > 1; v >>= 1) ++page_bits;

  const auto params = TopKTransformParams{
      .input = scores.data_ptr<float>(),
      .seq_lens = seq_lens.data_ptr<int32_t>(),
      .page_table = page_table.data_ptr<int32_t>(),
      .page_indices = page_indices.data_ptr<int32_t>(),
      .raw_indices = raw_indices_ptr,
      .input_stride = scores.stride(0),
      .page_table_stride = page_table.stride(0),
      .page_bits = page_bits,
  };

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};

  // Decide which kernel to launch based on max seq_len in this batch.
  // histogram_4096 supports up to kHist4096MaxLen (16384) floats per row.
  // If any row exceeds that, fall back to the 2-pass radix path for ALL rows
  // (mixing kernels in one launch is awkward; the radix path handles any len).
  // We pick per-launch, using max_seq_len across the batch.
  int32_t max_seq_len_host = 0;
  // Cheap host-side max: copy seq_lens (small, B elements) to host.
  // For very large B this is a tiny D2H; acceptable.
  auto seq_lens_cpu = seq_lens.cpu();
  const auto* seq_lens_ptr = seq_lens_cpu.data_ptr<int32_t>();
  for (int64_t i = 0; i < B; ++i) {
    if (seq_lens_ptr[i] > max_seq_len_host) max_seq_len_host = seq_lens_ptr[i];
  }

  const bool use_hist4096 = (static_cast<uint32_t>(max_seq_len_host) <= kHist4096MaxLen);

  // SMEM-size branching for the radix path (seq_len > 16384):
  //   - 64K-smem chips (C500/C280): dynamic smem capped at ~32KB by MACA
  //     runtime -> staging = 4096 entries (32KB).
  //   - 128K-smem chips (other MetaX variant): can afford 56KB dynamic smem
  //     -> staging = 7168 entries, mirroring vllm FilteredTopK
  //     (kFilteredTopKMinSmem = sizeof(int)*2*7K + 2048 in op/vllm/topk.cu:38).
  //     A larger staging buffer reduces threshold-bin candidate drops when the
  //     coarse FP16 bin is dense, and improves L2 reuse for pass 2.
  static int max_smem_per_block = 0;
  if (max_smem_per_block == 0) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&max_smem_per_block,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  }
  // Threshold: 96KB optin smem clearly distinguishes 128K-smem chips from
  // 64K-smem chips (C500/C280 optin cap is 64KB).
  constexpr int kLargeSmemThreshold = 96 * 1024;
  const bool use_large_smem = (max_smem_per_block >= kLargeSmemThreshold);

  if (use_hist4096) {
    // histogram_4096 Smem: 16KB histogram (union 8KB tie_buffer) + 4KB scan_buf
    // + scalars (counter_gt/eq, match, warp_sum[16]) under alignas(128) padding.
    // 24KB dynamic smem covers the struct with headroom.
    constexpr size_t kHist4096Smem = 24 * 1024;
    setup_kernel_smem_once<topk_transform_kernel<true>, kHist4096Smem>();
    topk_transform_kernel<true><<<grid, block, kHist4096Smem, stream>>>(params);
  } else if (use_large_smem) {
    // 128K-smem chip: 56KB dynamic staging (7168 entries per buffer).
    setup_kernel_smem_once<topk_transform_kernel<false, kLargeSmemInputSize>, kLargeSmem>();
    topk_transform_kernel<false, kLargeSmemInputSize><<<grid, block, kLargeSmem, stream>>>(params);
  } else {
    // 64K-smem chip (C500/C280): 32KB dynamic staging (4096 entries per buffer).
    setup_kernel_smem_once<topk_transform_kernel<false, kSmemInputSize>, kSmem>();
    topk_transform_kernel<false, kSmemInputSize><<<grid, block, kSmem, stream>>>(params);
  }
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk_transform_v1 kernel failed:", ::cudaGetErrorString(result));
}
