#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kShortRowThreads = 256;
constexpr int kRadixBuckets = 256;
constexpr int kRadixPasses = 4;
#ifdef USE_MACA
constexpr int kWaveSize = 64;
constexpr uint64_t kFullWaveMask = 0xffffffffffffffffULL;
#else
constexpr int kWaveSize = 32;
constexpr uint32_t kFullWaveMask = 0xffffffffu;
#endif

// Inclusive prefix for a boolean flag. Shared memory contains one count per
// wave, so the scan works for both CUDA warps and MetaX waves.
template <int BLOCK_THREADS>
__device__ __forceinline__ int block_flag_prefix(bool flag, int* wave_counts) {
  constexpr int kWaves = BLOCK_THREADS / kWaveSize;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & (kWaveSize - 1);
  const int wave = tid / kWaveSize;

  const uint64_t ballot =
      static_cast<uint64_t>(__ballot_sync(kFullWaveMask, flag));
  if (lane == 0) wave_counts[wave] = __popcll(ballot);
  __syncthreads();

  const uint64_t lower_mask =
      lane == 0 ? 0 : ((uint64_t{1} << lane) - 1);
  int prefix = __popcll(ballot & lower_mask) + static_cast<int>(flag);
#pragma unroll
  for (int preceding_wave = 0; preceding_wave < kWaves; ++preceding_wave) {
    if (preceding_wave < wave) prefix += wave_counts[preceding_wave];
  }
  return prefix;
}

// Numeric comparison deliberately makes -0.0f and +0.0f share a key, which
// matches the Triton implementation used by the original operator.
__device__ __forceinline__ uint32_t ordered_float_key(float score) {
  const uint32_t bits = __float_as_uint(score);
  return score < 0.0f ? ~bits : (bits | 0x80000000u);
}

template <int BLOCK_THREADS>
__global__ void region_topk_ids_kernel(
    const float* __restrict__ logits, const int32_t* __restrict__ lengths,
    int32_t* __restrict__ out, int32_t seq_regions,
    int64_t stride_logits_row, int64_t stride_out_row, int32_t topk) {
  const int tid = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.x);
  const float* row_logits =
      logits + static_cast<int64_t>(row) * stride_logits_row;
  int32_t* row_out = out + static_cast<int64_t>(row) * stride_out_row;

  constexpr int kWaves = BLOCK_THREADS / kWaveSize;
  __shared__ uint32_t histogram[kRadixBuckets];
  __shared__ int selected_wave_counts[kWaves];
  __shared__ int tied_wave_counts[kWaves];
  __shared__ uint32_t threshold;
  __shared__ int remaining;
  __shared__ int emit_cursor;
  __shared__ int ties_seen;

  for (int slot = tid; slot < topk; slot += BLOCK_THREADS) {
    row_out[slot] = -1;
  }

  int visible = lengths[row];
  visible = visible < 0 ? 0 : (visible > seq_regions ? seq_regions : visible);
  if (visible == 0 || topk == 0) return;

  if (topk >= visible) {
    __syncthreads();
    for (int region = tid; region < visible; region += BLOCK_THREADS) {
      row_out[region] = region;
    }
    return;
  }

  // Exact rank is faster than four radix passes for a single short tile.
  if (BLOCK_THREADS == kShortRowThreads && visible <= kShortRowThreads) {
    const uint32_t key =
        tid < visible ? ordered_float_key(row_logits[tid]) : 0;
    histogram[tid] = key;
    __syncthreads();

    bool selected = false;
    if (tid < visible) {
      int rank = 0;
      for (int other_id = 0; other_id < visible; ++other_id) {
        const uint32_t other_key = histogram[other_id];
        rank += other_key > key || (other_key == key && other_id < tid);
      }
      selected = rank < topk;
    }
    const int position =
        block_flag_prefix<BLOCK_THREADS>(selected, selected_wave_counts) - 1;
    if (selected) row_out[position] = tid;
    return;
  }

  if (tid == 0) {
    threshold = 0;
    remaining = topk < visible ? topk : visible;
  }
  __syncthreads();

#pragma unroll
  for (int pass = 0; pass < kRadixPasses; ++pass) {
    for (int bucket = tid; bucket < kRadixBuckets;
         bucket += BLOCK_THREADS) {
      histogram[bucket] = 0;
    }
    __syncthreads();

    const int shift = 24 - pass * 8;
    const int high = shift + 8;
    for (int region = tid; region < visible; region += BLOCK_THREADS) {
      const uint32_t key = ordered_float_key(row_logits[region]);
      const bool matches =
          pass == 0 || ((key >> high) == (threshold >> high));
      if (matches) {
        const int bucket = static_cast<int>((key >> shift) & 0xffu);
        atomicAdd(&histogram[bucket], 1u);
      }
    }
    __syncthreads();

    if (tid == 0) {
      int count_above = 0;
      int chosen = 0;
      for (int bucket = kRadixBuckets - 1; bucket >= 0; --bucket) {
        const int count = static_cast<int>(histogram[bucket]);
        if (count_above + count >= remaining) {
          chosen = bucket;
          remaining -= count_above;
          break;
        }
        count_above += count;
      }
      threshold |= static_cast<uint32_t>(chosen) << shift;
    }
    __syncthreads();
  }

  if (tid == 0) {
    emit_cursor = 0;
    ties_seen = 0;
  }
  __syncthreads();

  for (int base = 0; base < visible; base += BLOCK_THREADS) {
    const int region = base + tid;
    const bool live = region < visible;
    uint32_t key = 0;
    if (live) key = ordered_float_key(row_logits[region]);
    const bool tied = live && key == threshold;
    const bool greater = live && key > threshold;

    const int tied_prefix =
        block_flag_prefix<BLOCK_THREADS>(tied, tied_wave_counts);
    const bool selected =
        greater || (tied && ties_seen + tied_prefix <= remaining);
    const int selected_prefix =
        block_flag_prefix<BLOCK_THREADS>(selected, selected_wave_counts);

    if (selected) {
      const int position = emit_cursor + selected_prefix - 1;
      if (position < topk) row_out[position] = region;
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

cudaError_t launch_region_topk_ids(
    const float* logits, const int32_t* lengths, int32_t* out, int32_t rows,
    int32_t seq_regions, int64_t stride_logits_row, int64_t stride_out_row,
    int32_t topk, cudaStream_t stream) {
  if (rows == 0 || topk == 0) return cudaSuccess;

  if (rows < 32 && seq_regions > kShortRowThreads) {
    region_topk_ids_kernel<1024>
        <<<static_cast<unsigned int>(rows), 1024, 0, stream>>>(
            logits, lengths, out, seq_regions, stride_logits_row,
            stride_out_row, topk);
  } else {
    region_topk_ids_kernel<kShortRowThreads>
        <<<static_cast<unsigned int>(rows), kShortRowThreads, 0, stream>>>(
            logits, lengths, out, seq_regions, stride_logits_row,
            stride_out_row, topk);
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
      out.data_ptr<int32_t>(), static_cast<int32_t>(logits.size(0)),
      static_cast<int32_t>(logits.size(1)), logits.stride(0), out.stride(0),
      static_cast<int32_t>(topk), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cudaSuccess,
              "region_topk_ids launch failed: ", cudaGetErrorString(status));
  return out;
}
