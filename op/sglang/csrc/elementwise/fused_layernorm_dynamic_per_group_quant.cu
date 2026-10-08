
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <torch/all.h>

#include "../dispatch_utils.h"
#include "layernorm_utils.cuh"

namespace vllm {

constexpr int32_t kWarpSize = 64;
constexpr int32_t kQuantGroupSize = 128;
constexpr unsigned long long kFullWarpMask = 0xffffffffffffffffULL;

template <typename T, int N, int Alignment = sizeof(T) * N>
struct alignas(Alignment) AlignedArray {
  T data[N];
};

template <int32_t BLOCK_DIM>
__device__ __forceinline__ float block_sum(float value) {
  static_assert(BLOCK_DIM % kWarpSize == 0);
  constexpr int32_t kNumWarps = BLOCK_DIM / kWarpSize;
  __shared__ float warp_sums[kNumWarps];

#pragma unroll
  for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFullWarpMask, value, offset, kWarpSize);
  }

  const int32_t lane = threadIdx.x % kWarpSize;
  const int32_t warp = threadIdx.x / kWarpSize;
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    value = lane < kNumWarps ? warp_sums[lane] : 0.0f;
#pragma unroll
    for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(kFullWarpMask, value, offset, kWarpSize);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads();
  return warp_sums[0];
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value =
        fmaxf(value, __shfl_xor_sync(kFullWarpMask, value, offset, kWarpSize));
  }
  return value;
}

template <typename scalar_out_t> __device__ __forceinline__ float quant_qmax() {
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    return 127.0f;
  } else {
    return static_cast<float>(std::numeric_limits<scalar_out_t>::max());
  }
}

__device__ __forceinline__ float finalize_scale(float amax, float rms,
                                                float qmax, float min_scale,
                                                const float *scale_ub) {
  float scale = fmaxf(amax * rms / qmax, min_scale);
  if (scale_ub != nullptr) {
    scale = fminf(scale, *scale_ub);
  }
  return scale;
}

// Default fallback. One thread owns scalar elements. Two adjacent warps write
// their local amax to shared memory and one thread combines them per group.
template <typename scalar_t, typename scalar_out_t, int32_t VPT,
          int32_t BLOCK_DIM, bool has_residual, bool TRANS_SCATTER>
__global__ void rms_group_quant_default_kernel(
    scalar_out_t *__restrict__ out, scalar_t *__restrict__ out_norm,
    float *__restrict__ scales, const scalar_t *__restrict__ input,
    const scalar_t *__restrict__ weight, const float *scale_ub,
    float variance_epsilon, int32_t hidden_size, int32_t groups_per_token,
    float min_scale, scalar_t *__restrict__ residual) {
  static_assert(BLOCK_DIM % kQuantGroupSize == 0);
  constexpr int32_t kWarps = BLOCK_DIM / kWarpSize;
  constexpr int32_t kMaxGroups = VPT * BLOCK_DIM / kQuantGroupSize;
  __shared__ float warp_amax[VPT * kWarps];
  __shared__ float group_scales[kMaxGroups];

  const int32_t token = blockIdx.x;
  const int64_t row = static_cast<int64_t>(token) * hidden_size;
  float values[VPT];
  float ss = 0.0f;

#pragma unroll
  for (int32_t k = 0; k < VPT; ++k) {
    const int32_t col = threadIdx.x + k * BLOCK_DIM;
    float x = 0.0f;
    if (col < hidden_size) {
      x = static_cast<float>(input[row + col]);
      if constexpr (has_residual) {
        x += static_cast<float>(residual[row + col]);
        residual[row + col] = static_cast<scalar_t>(x);
      }
      ss += x * x;
      values[k] = x * static_cast<float>(weight[col]);
    } else {
      values[k] = 0.0f;
    }
  }

  const float rms =
      rsqrtf(block_sum<BLOCK_DIM>(ss) / static_cast<float>(hidden_size) +
             variance_epsilon);

  const int32_t lane = threadIdx.x % kWarpSize;
  const int32_t warp = threadIdx.x / kWarpSize;
#pragma unroll
  for (int32_t k = 0; k < VPT; ++k) {
    const int32_t col = threadIdx.x + k * BLOCK_DIM;
    float amax = col < hidden_size ? fabsf(values[k]) : 0.0f;
    amax = warp_max(amax);
    if (lane == 0) {
      warp_amax[k * kWarps + warp] = amax;
    }
  }
  __syncthreads();

  const float qmax = quant_qmax<scalar_out_t>();
  for (int32_t group = threadIdx.x; group < groups_per_token;
       group += BLOCK_DIM) {
    const int32_t first_col = group * kQuantGroupSize;
    const int32_t k = first_col / BLOCK_DIM;
    const int32_t first_warp = (first_col % BLOCK_DIM) / kWarpSize;
    const float amax = fmaxf(warp_amax[k * kWarps + first_warp],
                             warp_amax[k * kWarps + first_warp + 1]);
    const float scale = finalize_scale(amax, rms, qmax, min_scale, scale_ub);
    group_scales[group] = scale;
    if constexpr (TRANS_SCATTER) {
      scales[static_cast<int64_t>(group) * gridDim.x + token] = scale;
    } else {
      scales[static_cast<int64_t>(token) * groups_per_token + group] = scale;
    }
  }
  __syncthreads();

#pragma unroll
  for (int32_t k = 0; k < VPT; ++k) {
    const int32_t col = threadIdx.x + k * BLOCK_DIM;
    if (col < hidden_size) {
      const float norm = values[k] * rms;
      const float inv_scale = 1.0f / group_scales[col / kQuantGroupSize];
      out_norm[row + col] = static_cast<scalar_t>(norm);
      out[row + col] =
          ScaledQuant<scalar_out_t, true>::quant_fn(norm, inv_scale);
    }
  }
}

// VEC=2/4/8 maps a 128-element group to a 64/32/16-lane subgroup. The group
// amax and scale remain inside one warp, so this path uses no global atomic.
template <typename scalar_t, typename scalar_out_t, int32_t VEC,
          int32_t TILES_PER_WARP, int32_t BLOCK_DIM, bool has_residual,
          bool TRANS_SCATTER>
__global__ void rms_group_quant_vector_kernel(
    scalar_out_t *__restrict__ out, scalar_t *__restrict__ out_norm,
    float *__restrict__ scales, const scalar_t *__restrict__ input,
    const scalar_t *__restrict__ weight, const float *scale_ub,
    float variance_epsilon, int32_t hidden_size, int32_t groups_per_token,
    float min_scale, scalar_t *__restrict__ residual) {
  static_assert(VEC == 2 || VEC == 4 || VEC == 8);
  static_assert(BLOCK_DIM % kWarpSize == 0);
  constexpr int32_t kWarps = BLOCK_DIM / kWarpSize;
  constexpr int32_t kElementsPerWarp = kWarpSize * VEC;
  constexpr int32_t kSubgroupWidth = kQuantGroupSize / VEC;
  constexpr int32_t kGroupsPerWarp = kWarpSize / kSubgroupWidth;
  using InputVec = AlignedArray<scalar_t, VEC, sizeof(scalar_t) * VEC>;
  using OutputVec = AlignedArray<scalar_out_t, VEC, sizeof(scalar_out_t) * VEC>;

  const int32_t token = blockIdx.x;
  const int32_t warp = threadIdx.x / kWarpSize;
  const int32_t lane = threadIdx.x % kWarpSize;
  const int32_t subgroup_lane = lane % kSubgroupWidth;
  const int32_t subgroup = lane / kSubgroupWidth;
  const int64_t row = static_cast<int64_t>(token) * hidden_size;
  float values[TILES_PER_WARP][VEC];
  float ss = 0.0f;

#pragma unroll
  for (int32_t tile_iter = 0; tile_iter < TILES_PER_WARP; ++tile_iter) {
    const int32_t tile = warp + tile_iter * kWarps;
    const int32_t col = tile * kElementsPerWarp + lane * VEC;
    InputVec input_vec{};
    InputVec weight_vec{};
    InputVec residual_vec{};
    if (col < hidden_size) {
      input_vec = *reinterpret_cast<const InputVec *>(input + row + col);
      weight_vec = *reinterpret_cast<const InputVec *>(weight + col);
      if constexpr (has_residual) {
        residual_vec =
            *reinterpret_cast<const InputVec *>(residual + row + col);
      }
    }

#pragma unroll
    for (int32_t j = 0; j < VEC; ++j) {
      float x =
          col < hidden_size ? static_cast<float>(input_vec.data[j]) : 0.0f;
      if constexpr (has_residual) {
        if (col < hidden_size) {
          x += static_cast<float>(residual_vec.data[j]);
          residual_vec.data[j] = static_cast<scalar_t>(x);
        }
      }
      ss += x * x;
      values[tile_iter][j] =
          col < hidden_size ? x * static_cast<float>(weight_vec.data[j]) : 0.0f;
    }

    if constexpr (has_residual) {
      if (col < hidden_size) {
        *reinterpret_cast<InputVec *>(residual + row + col) = residual_vec;
      }
    }
  }

  const float rms =
      rsqrtf(block_sum<BLOCK_DIM>(ss) / static_cast<float>(hidden_size) +
             variance_epsilon);
  const float qmax = quant_qmax<scalar_out_t>();

#pragma unroll
  for (int32_t tile_iter = 0; tile_iter < TILES_PER_WARP; ++tile_iter) {
    const int32_t tile = warp + tile_iter * kWarps;
    const int32_t col = tile * kElementsPerWarp + lane * VEC;
    const int32_t group = tile * kGroupsPerWarp + subgroup;
    float amax = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < VEC; ++j) {
      amax = fmaxf(amax, fabsf(values[tile_iter][j]));
    }
#pragma unroll
    for (int32_t offset = kSubgroupWidth / 2; offset > 0; offset >>= 1) {
      amax = fmaxf(
          amax, __shfl_xor_sync(kFullWarpMask, amax, offset, kSubgroupWidth));
    }

    float scale = 0.0f;
    if (subgroup_lane == 0 && group < groups_per_token) {
      scale = finalize_scale(amax, rms, qmax, min_scale, scale_ub);
      if constexpr (TRANS_SCATTER) {
        scales[static_cast<int64_t>(group) * gridDim.x + token] = scale;
      } else {
        scales[static_cast<int64_t>(token) * groups_per_token + group] = scale;
      }
    }
    scale = __shfl_sync(kFullWarpMask, scale, 0, kSubgroupWidth);

    if (col < hidden_size) {
      InputVec norm_vec;
      OutputVec quant_vec;
      const float inv_scale = 1.0f / scale;
#pragma unroll
      for (int32_t j = 0; j < VEC; ++j) {
        const float norm = values[tile_iter][j] * rms;
        norm_vec.data[j] = static_cast<scalar_t>(norm);
        quant_vec.data[j] =
            ScaledQuant<scalar_out_t, true>::quant_fn(norm, inv_scale);
      }
      *reinterpret_cast<InputVec *>(out_norm + row + col) = norm_vec;
      *reinterpret_cast<OutputVec *>(out + row + col) = quant_vec;
    }
  }
}

template <typename scalar_t, typename scalar_out_t, int32_t VPT,
          int32_t BLOCK_DIM, bool has_residual>
void launch_default(scalar_out_t *out, scalar_t *out_norm, float *scales,
                    const scalar_t *input, const scalar_t *weight,
                    const float *scale_ub, float epsilon, int32_t hidden,
                    int32_t groups, int32_t tokens, float min_scale,
                    scalar_t *residual, bool trans_scatter,
                    cudaStream_t stream) {
  if (trans_scatter) {
    rms_group_quant_default_kernel<scalar_t, scalar_out_t, VPT, BLOCK_DIM,
                                   has_residual, true>
        <<<tokens, BLOCK_DIM, 0, stream>>>(out, out_norm, scales, input, weight,
                                           scale_ub, epsilon, hidden, groups,
                                           min_scale, residual);
  } else {
    rms_group_quant_default_kernel<scalar_t, scalar_out_t, VPT, BLOCK_DIM,
                                   has_residual, false>
        <<<tokens, BLOCK_DIM, 0, stream>>>(out, out_norm, scales, input, weight,
                                           scale_ub, epsilon, hidden, groups,
                                           min_scale, residual);
  }
}

template <typename scalar_t, typename scalar_out_t, bool has_residual>
void dispatch_default(scalar_out_t *out, scalar_t *out_norm, float *scales,
                      const scalar_t *input, const scalar_t *weight,
                      const float *scale_ub, float epsilon, int32_t hidden,
                      int32_t groups, int32_t tokens, float min_scale,
                      scalar_t *residual, bool trans_scatter,
                      cudaStream_t stream) {
#define LAUNCH_DEFAULT(VPT, BLOCK)                                             \
  launch_default<scalar_t, scalar_out_t, VPT, BLOCK, has_residual>(            \
      out, out_norm, scales, input, weight, scale_ub, epsilon, hidden, groups, \
      tokens, min_scale, residual, trans_scatter, stream)

  if (hidden <= 1024) {
    LAUNCH_DEFAULT(4, 256);
  } else if (hidden <= 2048) {
    LAUNCH_DEFAULT(8, 256);
  } else if (hidden <= 4096) {
    LAUNCH_DEFAULT(8, 512);
  } else if (hidden <= 8192) {
    LAUNCH_DEFAULT(16, 512);
  } else {
    LAUNCH_DEFAULT(32, 512);
  }
#undef LAUNCH_DEFAULT
}

template <typename scalar_t, typename scalar_out_t, int32_t VEC, int32_t TILES,
          int32_t BLOCK, bool has_residual>
void launch_vector(scalar_out_t *out, scalar_t *out_norm, float *scales,
                   const scalar_t *input, const scalar_t *weight,
                   const float *scale_ub, float epsilon, int32_t hidden,
                   int32_t groups, int32_t tokens, float min_scale,
                   scalar_t *residual, bool trans_scatter,
                   cudaStream_t stream) {
  if (trans_scatter) {
    rms_group_quant_vector_kernel<scalar_t, scalar_out_t, VEC, TILES, BLOCK,
                                  has_residual, true>
        <<<tokens, BLOCK, 0, stream>>>(out, out_norm, scales, input, weight,
                                       scale_ub, epsilon, hidden, groups,
                                       min_scale, residual);
  } else {
    rms_group_quant_vector_kernel<scalar_t, scalar_out_t, VEC, TILES, BLOCK,
                                  has_residual, false>
        <<<tokens, BLOCK, 0, stream>>>(out, out_norm, scales, input, weight,
                                       scale_ub, epsilon, hidden, groups,
                                       min_scale, residual);
  }
}

template <typename scalar_t, typename scalar_out_t, int32_t VEC,
          bool has_residual>
void dispatch_vector(scalar_out_t *out, scalar_t *out_norm, float *scales,
                     const scalar_t *input, const scalar_t *weight,
                     const float *scale_ub, float epsilon, int32_t hidden,
                     int32_t groups, int32_t tokens, float min_scale,
                     scalar_t *residual, bool trans_scatter,
                     cudaStream_t stream) {
#define LAUNCH_VECTOR(TILES, BLOCK)                                            \
  launch_vector<scalar_t, scalar_out_t, VEC, TILES, BLOCK, has_residual>(      \
      out, out_norm, scales, input, weight, scale_ub, epsilon, hidden, groups, \
      tokens, min_scale, residual, trans_scatter, stream)

  if constexpr (VEC == 8) {
    if (hidden <= 512) {
      LAUNCH_VECTOR(1, 64);
    } else if (hidden <= 1024) {
      LAUNCH_VECTOR(1, 128);
    } else if (hidden <= 2048) {
      LAUNCH_VECTOR(1, 256);
    } else if (hidden <= 4096) {
      LAUNCH_VECTOR(1, 512);
    } else if (hidden == 5120) {
      LAUNCH_VECTOR(2, 320);
    } else if (hidden == 6144) {
      LAUNCH_VECTOR(2, 384);
    } else if (hidden == 7168) {
      LAUNCH_VECTOR(2, 448);
    } else if (hidden <= 8192) {
      LAUNCH_VECTOR(2, 512);
    } else {
      LAUNCH_VECTOR(4, 512);
    }
  } else if constexpr (VEC == 4) {
    if (hidden <= 256) {
      LAUNCH_VECTOR(1, 64);
    } else if (hidden <= 512) {
      LAUNCH_VECTOR(1, 128);
    } else if (hidden <= 1024) {
      LAUNCH_VECTOR(1, 256);
    } else if (hidden <= 2048) {
      LAUNCH_VECTOR(1, 512);
    } else if (hidden <= 4096) {
      LAUNCH_VECTOR(2, 512);
    } else if (hidden == 5120) {
      LAUNCH_VECTOR(4, 320);
    } else if (hidden == 6144) {
      LAUNCH_VECTOR(3, 512);
    } else if (hidden == 7168) {
      LAUNCH_VECTOR(4, 448);
    } else if (hidden <= 8192) {
      LAUNCH_VECTOR(4, 512);
    } else {
      LAUNCH_VECTOR(8, 512);
    }
  } else {
    if (hidden <= 128) {
      LAUNCH_VECTOR(1, 64);
    } else if (hidden <= 256) {
      LAUNCH_VECTOR(1, 128);
    } else if (hidden <= 512) {
      LAUNCH_VECTOR(1, 256);
    } else if (hidden <= 1024) {
      LAUNCH_VECTOR(1, 512);
    } else if (hidden <= 2048) {
      LAUNCH_VECTOR(2, 512);
    } else if (hidden <= 4096) {
      LAUNCH_VECTOR(4, 512);
    } else if (hidden == 5120) {
      LAUNCH_VECTOR(5, 512);
    } else if (hidden == 6144) {
      LAUNCH_VECTOR(6, 512);
    } else if (hidden == 7168) {
      LAUNCH_VECTOR(7, 512);
    } else if (hidden <= 8192) {
      LAUNCH_VECTOR(8, 512);
    } else {
      LAUNCH_VECTOR(16, 512);
    }
  }
#undef LAUNCH_VECTOR
}

inline bool is_aligned(const void *ptr, size_t alignment) {
  return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

template <typename scalar_t, typename scalar_out_t, int32_t VEC>
bool can_vectorize(const scalar_out_t *out, const scalar_t *out_norm,
                   const scalar_t *input, const scalar_t *weight,
                   const scalar_t *residual, int32_t hidden,
                   bool has_residual) {
  if constexpr (sizeof(scalar_t) != 2) {
    return false;
  }
  constexpr size_t kInputAlignment = sizeof(scalar_t) * VEC;
  constexpr size_t kOutputAlignment = sizeof(scalar_out_t) * VEC;
  return hidden % VEC == 0 && is_aligned(input, kInputAlignment) &&
         is_aligned(weight, kInputAlignment) &&
         is_aligned(out_norm, kInputAlignment) &&
         is_aligned(out, kOutputAlignment) &&
         (!has_residual || is_aligned(residual, kInputAlignment));
}

template <typename scalar_t, typename scalar_out_t, bool has_residual>
void dispatch_selected_kernel(scalar_out_t *out, scalar_t *out_norm,
                              float *scales, const scalar_t *input,
                              const scalar_t *weight, const float *scale_ub,
                              float epsilon, int32_t hidden, int32_t groups,
                              int32_t tokens, float min_scale,
                              scalar_t *residual, bool trans_scatter,
                              cudaStream_t stream) {
  if (can_vectorize<scalar_t, scalar_out_t, 8>(
          out, out_norm, input, weight, residual, hidden, has_residual)) {
    dispatch_vector<scalar_t, scalar_out_t, 8, has_residual>(
        out, out_norm, scales, input, weight, scale_ub, epsilon, hidden, groups,
        tokens, min_scale, residual, trans_scatter, stream);
  } else if (can_vectorize<scalar_t, scalar_out_t, 4>(out, out_norm, input,
                                                      weight, residual, hidden,
                                                      has_residual)) {
    dispatch_vector<scalar_t, scalar_out_t, 4, has_residual>(
        out, out_norm, scales, input, weight, scale_ub, epsilon, hidden, groups,
        tokens, min_scale, residual, trans_scatter, stream);
  } else if (can_vectorize<scalar_t, scalar_out_t, 2>(out, out_norm, input,
                                                      weight, residual, hidden,
                                                      has_residual)) {
    dispatch_vector<scalar_t, scalar_out_t, 2, has_residual>(
        out, out_norm, scales, input, weight, scale_ub, epsilon, hidden, groups,
        tokens, min_scale, residual, trans_scatter, stream);
  } else {
    dispatch_default<scalar_t, scalar_out_t, has_residual>(
        out, out_norm, scales, input, weight, scale_ub, epsilon, hidden, groups,
        tokens, min_scale, residual, trans_scatter, stream);
  }
}

template <typename scalar_in_t, bool has_residual>
void dispatch_output_type(torch::Tensor &out, torch::Tensor &out_norm,
                          float *scales_ptr, const torch::Tensor &input,
                          const torch::Tensor &weight,
                          const std::optional<at::Tensor> &scale_ub,
                          const std::optional<at::Tensor> &residual,
                          float epsilon, int32_t hidden, int32_t groups,
                          int32_t tokens, float min_scale, bool trans_scatter,
                          cudaStream_t stream) {
  scalar_in_t *residual_ptr =
      has_residual ? residual->data_ptr<scalar_in_t>() : nullptr;
  const float *scale_ub_ptr =
      scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr;

  VLLM_DISPATCH_QUANT_TYPES(
      out.scalar_type(), "rms_norm_dynamic_per_group_quant", [&] {
        dispatch_selected_kernel<scalar_in_t, scalar_t, has_residual>(
            out.data_ptr<scalar_t>(), out_norm.data_ptr<scalar_in_t>(),
            scales_ptr, input.data_ptr<scalar_in_t>(),
            weight.data_ptr<scalar_in_t>(), scale_ub_ptr, epsilon, hidden,
            groups, tokens, min_scale, residual_ptr, trans_scatter, stream);
      });
}

template <typename scalar_in_t>
void dispatch_input_type(torch::Tensor &out, torch::Tensor &out_norm,
                         float *scales_ptr, const torch::Tensor &input,
                         const torch::Tensor &weight,
                         const std::optional<at::Tensor> &scale_ub,
                         const std::optional<at::Tensor> &residual,
                         float epsilon, int32_t hidden, int32_t groups,
                         int32_t tokens, float min_scale, bool trans_scatter,
                         cudaStream_t stream) {
  if (residual.has_value()) {
    dispatch_output_type<scalar_in_t, true>(
        out, out_norm, scales_ptr, input, weight, scale_ub, residual, epsilon,
        hidden, groups, tokens, min_scale, trans_scatter, stream);
  } else {
    dispatch_output_type<scalar_in_t, false>(
        out, out_norm, scales_ptr, input, weight, scale_ub, residual, epsilon,
        hidden, groups, tokens, min_scale, trans_scatter, stream);
  }
}

} // namespace vllm

namespace {
__global__ __launch_bounds__(256)
void transpose_scale_kernel(const float* __restrict__ src, float* __restrict__ dst, int m, int G)
{
    __shared__ float tile[32][33];
    const int bx = blockIdx.x * 32, by = blockIdx.y * 32;
    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        int r = by + threadIdx.y + j, c = bx + threadIdx.x;
        if (r < m && c < G) tile[threadIdx.y + j][threadIdx.x] = src[(long)r * G + c];
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        int r = bx + threadIdx.y + j, c = by + threadIdx.x;
        if (r < G && c < m) dst[(long)r * m + c] = tile[threadIdx.x][threadIdx.y + j];
    }
}
}

// Note: only support group 128
void rms_norm_dynamic_per_group_quant(
    torch::Tensor &out, torch::Tensor &out_norm, const torch::Tensor &input,
    const torch::Tensor &weight, torch::Tensor &scales,
    int64_t quant_group_size, double variance_epsilon,
    const std::optional<at::Tensor> &scale_ub,
    const std::optional<at::Tensor> &residual,
    bool trans_scale) {
  const auto fp8_type = is_fp8_ocp()
      ? c10::ScalarType::Float8_e4m3fn
      : c10::ScalarType::Float8_e4m3fnuz;

  TORCH_CHECK(input.defined(), "input must be defined");
  TORCH_CHECK(out.defined(), "out must be defined");
  TORCH_CHECK(out_norm.defined(), "out_norm must be defined");
  TORCH_CHECK(weight.defined(), "weight must be defined");
  TORCH_CHECK(scales.defined(), "scales must be defined");

  TORCH_CHECK(input.is_cuda() && out.is_cuda() && out_norm.is_cuda() &&
                  weight.is_cuda() && scales.is_cuda(),
              "input, out, out_norm, weight and scales must be CUDA tensors");
  TORCH_CHECK(input.device() == out.device() &&
                  input.device() == out_norm.device() &&
                  input.device() == weight.device() &&
                  input.device() == scales.device(),
              "input, out, out_norm, weight and scales must be on the same CUDA device");

  TORCH_CHECK(input.dim() >= 1,
              "input must have rank >= 1 and shape [..., hidden_size]");
  TORCH_CHECK(input.numel() > 0,
              "input must be non-empty");
  const int64_t hidden = input.size(-1);
  TORCH_CHECK(hidden > 0,
              "input.size(-1) must be > 0");
  TORCH_CHECK(input.numel() % hidden == 0,
              "input.numel() must be divisible by hidden_size");
  const int64_t tokens = input.numel() / hidden;
  TORCH_CHECK(tokens > 0,
              "num_tokens must be > 0");

  TORCH_CHECK(quant_group_size == vllm::kQuantGroupSize,
              "quant_group_size must be 128");
  TORCH_CHECK(hidden <= 16384,
              "hidden_size must be <= 16384, got ", hidden);
  const int64_t groups =
      (hidden + quant_group_size - 1) / quant_group_size;

  TORCH_CHECK(input.scalar_type() == torch::kFloat16 ||
                  input.scalar_type() == torch::kBFloat16 ||
                  input.scalar_type() == torch::kFloat32,
              "input dtype must be float16, bfloat16, or float32");
  TORCH_CHECK(out.scalar_type() == torch::kInt8 ||
                  out.scalar_type() == fp8_type,
              "out dtype must be int8 or the platform FP8 e4m3 type");
  TORCH_CHECK(out_norm.scalar_type() == input.scalar_type(),
              "out_norm dtype must match input dtype");
  TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
              "weight dtype must match input dtype");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
              "scales dtype must be float32");

  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(out_norm.is_contiguous(), "out_norm must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  TORCH_CHECK(out.sizes() == input.sizes(),
              "out shape must match input shape, got out=", out.sizes(),
              ", input=", input.sizes());
  TORCH_CHECK(out_norm.sizes() == input.sizes(),
              "out_norm shape must match input shape, got out_norm=",
              out_norm.sizes(), ", input=", input.sizes());
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == hidden,
              "weight shape must be [hidden_size], got ", weight.sizes(),
              ", hidden_size=", hidden);

  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == tokens &&
                  scales.size(1) == groups,
              "scales shape must be [num_tokens, ceil(hidden_size / 128)], got ",
              scales.sizes(), ", expected [", tokens, ", ", groups, "]");

  TORCH_CHECK(std::isfinite(variance_epsilon) && variance_epsilon >= 0.0,
              "variance_epsilon must be finite and non-negative");

  if (scale_ub.has_value()) {
    TORCH_CHECK(scale_ub->defined(), "scale_ub must be defined when provided");
    TORCH_CHECK(out.scalar_type() == fp8_type,
                "scale_ub is only supported for FP8 output");
    TORCH_CHECK(scale_ub->is_cuda(), "scale_ub must be a CUDA tensor");
    TORCH_CHECK(scale_ub->device() == input.device(),
                "scale_ub must be on the same CUDA device as input");
    TORCH_CHECK(scale_ub->is_contiguous(), "scale_ub must be contiguous");
    TORCH_CHECK(scale_ub->scalar_type() == torch::kFloat32,
                "scale_ub dtype must be float32");
    TORCH_CHECK(scale_ub->numel() == 1,
                "scale_ub must contain exactly one element");
  }

  if (residual.has_value()) {
    TORCH_CHECK(residual->defined(), "residual must be defined when provided");
    TORCH_CHECK(residual->is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(residual->device() == input.device(),
                "residual must be on the same CUDA device as input");
    TORCH_CHECK(residual->is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(residual->scalar_type() == input.scalar_type(),
                "residual dtype must match input dtype");
    TORCH_CHECK(residual->sizes() == input.sizes(),
                "residual shape must match input shape, got residual=",
                residual->sizes(), ", input=", input.sizes());
  }

  const float min_scale =
      out.scalar_type() == torch::kInt8
          ? std::numeric_limits<float>::epsilon()
          : 1.0f / (static_cast<float>(
                        std::numeric_limits<c10::Float8_e4m3fn>::max()) *
                    512.0f);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const bool use_ws = trans_scale && tokens >= 512 && groups >= 8;
  torch::Tensor ws;
  float *scales_ptr = scales.data_ptr<float>();
  if (use_ws) {
    ws = torch::empty({tokens, groups},
                      input.options().dtype(torch::kFloat32));
    scales_ptr = ws.data_ptr<float>();
  }
  const bool trans_scatter = trans_scale && !use_ws;

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_dynamic_per_group_quant_dispatch", [&] {
        vllm::dispatch_input_type<scalar_t>(
            out, out_norm, scales_ptr, input, weight, scale_ub, residual,
            static_cast<float>(variance_epsilon), static_cast<int32_t>(hidden),
            static_cast<int32_t>(groups), static_cast<int32_t>(tokens),
            min_scale, trans_scatter, stream);
      });

  if (use_ws) {
    transpose_scale_kernel<<<
        dim3((groups + 31) / 32, (tokens + 31) / 32), dim3(32, 8), 0, stream>>>(
        scales_ptr, scales.data_ptr<float>(), static_cast<int32_t>(tokens),
        static_cast<int32_t>(groups));
  }
}
