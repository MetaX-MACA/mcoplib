// SPDX-License-Identifier: Apache-2.0
// 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.
// All Rights Reserved.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kBlock = 256;
constexpr int kThreads = 64;

__device__ __forceinline__ float neg_inf() {
  return __int_as_float(0xFF800000);
}

enum class ShapeKind : int32_t {
  kScalar,
  kAligned,
  kDenseTiles,
};

__device__ __forceinline__ void store_valid_float4(
    const float* scores_in,
    const uint8_t* summary_valid,
    float* scores_out,
    int64_t in_offset,
    int64_t out_offset) {
  const float4 values =
      *reinterpret_cast<const float4*>(scores_in + in_offset);
  const uint32_t bits =
      *reinterpret_cast<const uint32_t*>(summary_valid + in_offset);
  if (bits == 0x01010101u) {
    *reinterpret_cast<float4*>(scores_out + out_offset) = values;
    return;
  }

  float4 out;
  out.x = (bits & 0xffu) ? values.x : neg_inf();
  out.y = ((bits >> 8) & 0xffu) ? values.y : neg_inf();
  out.z = ((bits >> 16) & 0xffu) ? values.z : neg_inf();
  out.w = ((bits >> 24) & 0xffu) ? values.w : neg_inf();
  *reinterpret_cast<float4*>(scores_out + out_offset) = out;
}

template <bool HasSummaryValid, ShapeKind Shape>
__global__ void prepare_topk_scores_kernel(
    const float* __restrict__ scores_in,
    const uint8_t* __restrict__ summary_valid,
    const int32_t* __restrict__ row_starts,
    const int32_t* __restrict__ row_ends,
    float* __restrict__ scores_out,
    int32_t num_regions,
    int32_t padded_regions) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int32_t block_start =
      static_cast<int32_t>(blockIdx.y) * kBlock;
  const int32_t block_end = block_start + kBlock;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);

  if constexpr (HasSummaryValid) {
    const int32_t start = row_starts[row];
    const int32_t end = row_ends[row];
    if (block_start >= end || block_end <= start) {
      return;
    }

    const int64_t in_base =
        static_cast<int64_t>(row) * num_regions;
    const int64_t out_base =
        static_cast<int64_t>(row) * padded_regions;
    if constexpr (Shape != ShapeKind::kScalar) {
      bool full = block_start >= start && block_end <= end;
      if constexpr (Shape != ShapeKind::kDenseTiles) {
        full = full && block_end <= num_regions;
      }
      if (full) {
        const int32_t col = block_start + tid * 4;
        store_valid_float4(scores_in, summary_valid, scores_out,
                           in_base + col, out_base + col);
        return;
      }
    }

    for (int32_t col = block_start + tid;
         col < block_end; col += kThreads) {
      if (col >= padded_regions) {
        continue;
      }
      float value = neg_inf();
      if (col < num_regions && col >= start && col < end &&
          summary_valid[in_base + col]) {
        value = scores_in[in_base + col];
      }
      scores_out[out_base + col] = value;
    }
  } else {
    const int64_t in_base =
        static_cast<int64_t>(row) * num_regions;
    const int64_t out_base =
        static_cast<int64_t>(row) * padded_regions;
    if constexpr (Shape == ShapeKind::kDenseTiles) {
      const int32_t col = block_start + tid * 4;
      *reinterpret_cast<float4*>(scores_out + out_base + col) =
          *reinterpret_cast<const float4*>(scores_in + in_base + col);
      return;
    }
    if constexpr (Shape == ShapeKind::kAligned) {
      if (block_end <= num_regions) {
        const int32_t col = block_start + tid * 4;
        *reinterpret_cast<float4*>(scores_out + out_base + col) =
            *reinterpret_cast<const float4*>(scores_in + in_base + col);
        return;
      }
    }

    for (int32_t col = block_start + tid;
         col < block_end; col += kThreads) {
      if (col >= padded_regions) {
        continue;
      }
      scores_out[out_base + col] =
          col < num_regions ? scores_in[in_base + col] : neg_inf();
    }
  }
}

template <bool HasSummaryValid, ShapeKind Shape>
void launch_shape(
    const float* scores_in,
    const uint8_t* summary_valid,
    const int32_t* row_starts,
    const int32_t* row_ends,
    float* scores_out,
    int32_t num_regions,
    int32_t padded_regions,
    dim3 grid,
    cudaStream_t stream) {
  prepare_topk_scores_kernel<HasSummaryValid, Shape>
      <<<grid, kThreads, 0, stream>>>(
          scores_in, summary_valid, row_starts, row_ends, scores_out,
          num_regions, padded_regions);
}

cudaError_t launch_prepare_topk_scores(
    const float* scores_in,
    const uint8_t* summary_valid,
    const int32_t* row_starts,
    const int32_t* row_ends,
    float* scores_out,
    int32_t rows,
    int32_t num_regions,
    int32_t padded_regions,
    int32_t has_summary_valid,
    cudaStream_t stream) {
  const int32_t tiles =
      (padded_regions + kBlock - 1) / kBlock;
  if (rows == 0 || tiles == 0) {
    return cudaSuccess;
  }

  const dim3 grid(rows, tiles);
  const bool dense_tiles =
      num_regions == padded_regions && (num_regions & (kBlock - 1)) == 0;
  const bool aligned = ((num_regions | padded_regions) & 3) == 0;
  if (has_summary_valid) {
    if (dense_tiles) {
      launch_shape<true, ShapeKind::kDenseTiles>(
          scores_in, summary_valid, row_starts, row_ends, scores_out,
          num_regions, padded_regions, grid, stream);
    } else if (aligned) {
      launch_shape<true, ShapeKind::kAligned>(
          scores_in, summary_valid, row_starts, row_ends, scores_out,
          num_regions, padded_regions, grid, stream);
    } else {
      launch_shape<true, ShapeKind::kScalar>(
          scores_in, summary_valid, row_starts, row_ends, scores_out,
          num_regions, padded_regions, grid, stream);
    }
  } else {
    if (dense_tiles) {
      launch_shape<false, ShapeKind::kDenseTiles>(
          scores_in, summary_valid, row_starts, row_ends, scores_out,
          num_regions, padded_regions, grid, stream);
    } else if (aligned) {
      launch_shape<false, ShapeKind::kAligned>(
          scores_in, summary_valid, row_starts, row_ends, scores_out,
          num_regions, padded_regions, grid, stream);
    } else {
      launch_shape<false, ShapeKind::kScalar>(
          scores_in, summary_valid, row_starts, row_ends, scores_out,
          num_regions, padded_regions, grid, stream);
    }
  }
  return cudaGetLastError();
}
}  // namespace

void prepare_topk_scores(
    const torch::Tensor& scores_in,
    const torch::Tensor& summary_valid,
    const torch::Tensor& row_starts,
    const torch::Tensor& row_ends,
    torch::Tensor& scores_out,
    int64_t num_regions,
    int64_t padded_regions,
    bool has_summary_valid) {
  TORCH_CHECK(scores_in.is_cuda(), "scores_in must be a CUDA tensor");
  TORCH_CHECK(summary_valid.is_cuda(),
              "summary_valid must be a CUDA tensor");
  TORCH_CHECK(row_starts.is_cuda(),
              "row_starts must be a CUDA tensor");
  TORCH_CHECK(row_ends.is_cuda(), "row_ends must be a CUDA tensor");
  TORCH_CHECK(scores_out.is_cuda(), "scores_out must be a CUDA tensor");
  TORCH_CHECK(
      scores_in.device() == summary_valid.device() &&
          scores_in.device() == row_starts.device() &&
          scores_in.device() == row_ends.device() &&
          scores_in.device() == scores_out.device(),
      "all tensors must be on the same CUDA device");

  TORCH_CHECK(scores_in.scalar_type() == torch::kFloat32,
              "scores_in must have dtype float32");
  TORCH_CHECK(summary_valid.scalar_type() == torch::kBool,
              "summary_valid must have dtype bool");
  TORCH_CHECK(row_starts.scalar_type() == torch::kInt32,
              "row_starts must have dtype int32");
  TORCH_CHECK(row_ends.scalar_type() == torch::kInt32,
              "row_ends must have dtype int32");
  TORCH_CHECK(scores_out.scalar_type() == torch::kFloat32,
              "scores_out must have dtype float32");

  TORCH_CHECK(scores_in.dim() == 2, "scores_in must be 2D");
  TORCH_CHECK(summary_valid.dim() == 2,
              "summary_valid must be 2D");
  TORCH_CHECK(row_starts.dim() == 1, "row_starts must be 1D");
  TORCH_CHECK(row_ends.dim() == 1, "row_ends must be 1D");
  TORCH_CHECK(scores_out.dim() == 2, "scores_out must be 2D");

  TORCH_CHECK(num_regions >= 0, "num_regions must be non-negative");
  TORCH_CHECK(padded_regions >= 0,
              "padded_regions must be non-negative");
  TORCH_CHECK(padded_regions >= num_regions,
              "padded_regions must be at least num_regions");
  TORCH_CHECK(
      scores_in.size(0) <= std::numeric_limits<int32_t>::max() &&
          num_regions <= std::numeric_limits<int32_t>::max() &&
          padded_regions <= std::numeric_limits<int32_t>::max(),
      "rows, num_regions, and padded_regions must fit in int32");

  const int64_t rows = scores_in.size(0);
  TORCH_CHECK(scores_in.size(1) == num_regions,
              "scores_in width must equal num_regions");
  TORCH_CHECK(summary_valid.sizes() == scores_in.sizes(),
              "summary_valid shape must match scores_in");
  TORCH_CHECK(row_starts.numel() == rows,
              "row_starts size must match scores_in rows");
  TORCH_CHECK(row_ends.numel() == rows,
              "row_ends size must match scores_in rows");
  TORCH_CHECK(scores_out.size(0) == rows,
              "scores_out rows must match scores_in rows");
  TORCH_CHECK(scores_out.size(1) == padded_regions,
              "scores_out width must equal padded_regions");

  TORCH_CHECK(scores_in.is_contiguous(),
              "scores_in must be contiguous");
  TORCH_CHECK(summary_valid.is_contiguous(),
              "summary_valid must be contiguous");
  TORCH_CHECK(row_starts.is_contiguous(),
              "row_starts must be contiguous");
  TORCH_CHECK(row_ends.is_contiguous(),
              "row_ends must be contiguous");
  TORCH_CHECK(scores_out.is_contiguous(),
              "scores_out must be contiguous");

  if (rows == 0 || padded_regions == 0) {
    return;
  }

  static_assert(sizeof(bool) == sizeof(uint8_t),
                "bool storage must be one byte");
  at::cuda::OptionalCUDAGuard const device_guard(scores_in.device());
  const auto* valid_ptr = reinterpret_cast<const uint8_t*>(
      summary_valid.data_ptr<bool>());
  const cudaError_t status = launch_prepare_topk_scores(
      scores_in.data_ptr<float>(), valid_ptr,
      row_starts.data_ptr<int32_t>(), row_ends.data_ptr<int32_t>(),
      scores_out.data_ptr<float>(), static_cast<int32_t>(rows),
      static_cast<int32_t>(num_regions),
      static_cast<int32_t>(padded_regions), static_cast<int32_t>(has_summary_valid),
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cudaSuccess,
              "prepare_topk_scores launch failed: ",
              cudaGetErrorString(status));
}
