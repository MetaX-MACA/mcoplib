#include "core/batch_invariant.hpp"
#include "cub_helpers.h"
#include "dispatch_utils.h"
#include "quantization/vectorization_utils.cuh"
#include "type_convert.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <torch/cuda.h>
#include "ops.h"

#include <cub/cub.cuh>
#include <climits>
#include <cstring>
#include <type_traits>

namespace vllm {
template <typename scalar_t, int NUM_DIMS>
__device__ __forceinline__ const scalar_t *
rms_input_row(const scalar_t *input, int token, int64_t input_stride_d2,
              int64_t input_stride_d3, int64_t input_stride_d4,
              int64_t input_shape_d2, int64_t input_shape_d3) {
  if constexpr (NUM_DIMS == 2) {
    return input + token * input_stride_d2;
  } else if constexpr (NUM_DIMS == 3) {
    const int batch_idx = token / input_shape_d2;
    const int head_idx = token % input_shape_d2;
    return input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
  } else {
    const int batch_idx = token / (input_shape_d3 * input_shape_d2);
    const int remaining = token % (input_shape_d3 * input_shape_d2);
    const int seq_idx = remaining / input_shape_d2;
    const int head_idx = remaining % input_shape_d2;
    return input + batch_idx * input_stride_d4 + seq_idx * input_stride_d3 +
           head_idx * input_stride_d2;
  }
}

// ===========================================================================
// single-barrier block reduction.
// ===========================================================================
template <int BLOCK_SIZE>
__device__ __forceinline__ float rms_block_reduce_sum(float val,
                                                      float *smem) {
  constexpr int NUM_GROUPS = BLOCK_SIZE >> 4;  // 16-lane groups
#pragma unroll
  for (int i = 8; i > 0; i >>= 1) {
    val += __shfl_down_sync_16(0xffffffffffffffffULL, val, i);
  }
  const int lane_id = threadIdx.x & 15;
  const int group_id = threadIdx.x >> 4;
  if (lane_id == 0) {
    smem[group_id] = val;
  }
  __syncthreads();
  float tot = 0.0f;
#pragma unroll
  for (int g = 0; g < NUM_GROUPS; ++g) {
    tot += smem[g];
  }
  return tot;
}


template <bool ZERO_CENTERED, typename scalar_t>
__device__ __forceinline__ float rms_weight(scalar_t w) {
  float wf = static_cast<float>(w);
  if constexpr (ZERO_CENTERED) return (wf + 1.0f);
  return wf;
}

// The float cast in the reference allocates aligned storage, even when the
// original low-precision tensor has a storage offset or padded rows. Preserve
// its logical packs while accessing the actual low-precision storage safely.
template <typename scalar_t, typename weight_t, int VEC_SIZE>
__device__ __forceinline__ vec_n_t<scalar_t, VEC_SIZE>
rms_load_input(const scalar_t *ptr) {
  using Vec = vec_n_t<scalar_t, VEC_SIZE>;
  if constexpr (std::is_same_v<weight_t, float> &&
                !std::is_same_v<scalar_t, float>) {
    if (reinterpret_cast<std::uintptr_t>(ptr) % alignof(Vec) != 0) {
      Vec value;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) value.val[j] = ptr[j];
      return value;
    }
  }
  return *reinterpret_cast<const Vec *>(ptr);
}

template <typename scalar_t, typename weight_t, int VEC_SIZE>
__device__ __forceinline__ void rms_store_output(
    scalar_t *ptr, const vec_n_t<scalar_t, VEC_SIZE> &value) {
  using Vec = vec_n_t<scalar_t, VEC_SIZE>;
  if constexpr (std::is_same_v<weight_t, float> &&
                !std::is_same_v<scalar_t, float>) {
    if (reinterpret_cast<std::uintptr_t>(ptr) % alignof(Vec) != 0) {
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) ptr[j] = value.val[j];
      return;
    }
  }
  *reinterpret_cast<Vec *>(ptr) = value;
}

// MACA's packed float output arithmetic in the default and multirow kernels
// produces positive exact zero. Match that before the final low-precision
// conversion; a negative nonzero float that underflows must still yield -0.
template <typename scalar_t, typename weight_t, int VEC_SIZE>
__device__ __forceinline__ scalar_t rms_packed_result(float value) {
  if constexpr (std::is_same_v<weight_t, float> &&
                !std::is_same_v<scalar_t, float> && VEC_SIZE == 4) {
    if (value == 0.0f) value = 0.0f;
  }
  return static_cast<scalar_t>(value);
}

// ===========================================================================
// Warp-per-row RMSNorm for SMALL hidden sizes.
template <typename scalar_t, typename weight_t, int VEC_SIZE, int NUM_DIMS, bool HAS_WEIGHT,
          bool PER_TOKEN_WEIGHT, int BLOCK_SIZE, int ITEMS_PER_LANE, bool ZERO_CENTERED>
__global__ __launch_bounds__(BLOCK_SIZE) void rms_norm_warp_kernel(
    scalar_t *__restrict__ out, const scalar_t *__restrict__ input,
    int64_t input_stride_d2, int64_t input_stride_d3, int64_t input_stride_d4,
    int64_t input_shape_d2, int64_t input_shape_d3,
    const weight_t *__restrict__ weight, int64_t weight_stride, float epsilon,
    int num_tokens, int hidden_size) {
  using Vec = vec_n_t<scalar_t, VEC_SIZE>;
  using WeightVec = vec_n_t<weight_t, VEC_SIZE>;
  constexpr int WARP = 64;
  constexpr int ROWS_PER_BLOCK = BLOCK_SIZE / WARP;
  const int lane = threadIdx.x & (WARP - 1);
  const int warp_in_block = threadIdx.x / WARP;
  const int token_idx = blockIdx.x * ROWS_PER_BLOCK + warp_in_block;
  if (token_idx >= num_tokens) return;

  const int vec_count = hidden_size / VEC_SIZE;

  // Resolve the row pointer for this token (supports 2/3/4-D layouts).
  int batch_idx = 0;
  if constexpr (NUM_DIMS == 2) {
    batch_idx = token_idx;
  } else if constexpr (NUM_DIMS == 3) {
    batch_idx = token_idx / input_shape_d2;
  } else if constexpr (NUM_DIMS == 4) {
    batch_idx = token_idx / (input_shape_d3 * input_shape_d2);
  }
  const scalar_t *input_row;
  if constexpr (NUM_DIMS == 2) {
    input_row = input + token_idx * input_stride_d2;
  } else if constexpr (NUM_DIMS == 3) {
    const int b = token_idx / input_shape_d2;
    const int h = token_idx % input_shape_d2;
    input_row = input + b * input_stride_d3 + h * input_stride_d2;
  } else {
    const int b = token_idx / (input_shape_d3 * input_shape_d2);
    const int rem = token_idx % (input_shape_d3 * input_shape_d2);
    const int s = rem / input_shape_d2;
    const int h = rem % input_shape_d2;
    input_row = input + b * input_stride_d4 + s * input_stride_d3 +
                h * input_stride_d2;
  }
  const Vec *input_vec = reinterpret_cast<const Vec *>(input_row);

  Vec cached[ITEMS_PER_LANE];
  float variance = 0.0f;
#pragma unroll
  for (int it = 0; it < ITEMS_PER_LANE; ++it) {
    const int vidx = lane + it * WARP;
    Vec src{};
    if (vidx < vec_count) src = rms_load_input<scalar_t, weight_t, VEC_SIZE>(input_row + vidx * VEC_SIZE);
    cached[it] = src;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const float x = static_cast<float>(src.val[j]);
      variance += x * x;
    }
  }

  // Pure warp reduction over 64 lanes -- no smem, no __syncthreads.
#pragma unroll
  for (int offset = WARP >> 1; offset > 0; offset >>= 1) {
    variance += __shfl_xor_sync(0xffffffffffffffffULL, variance, offset);
  }
  const float s_variance =
      __builtin_mxc_rcpf(sqrtf(variance / hidden_size + epsilon));

  scalar_t *out_row = out + token_idx * hidden_size;
  auto *output_vec = reinterpret_cast<Vec *>(out_row);
  const WeightVec *weight_vec = nullptr;
  if constexpr (HAS_WEIGHT) {
    if constexpr (PER_TOKEN_WEIGHT) {
      weight_vec = reinterpret_cast<const WeightVec *>(weight + batch_idx * weight_stride);
    } else {
      weight_vec = reinterpret_cast<const WeightVec *>(weight);
    }
  }
#pragma unroll
  for (int it = 0; it < ITEMS_PER_LANE; ++it) {
    const int vidx = lane + it * WARP;
    if (vidx < vec_count) {
      Vec dst;
      WeightVec w{};
      if constexpr (HAS_WEIGHT) w = weight_vec[vidx];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        float value = static_cast<float>(cached[it].val[j]) * s_variance;
        if constexpr (HAS_WEIGHT) value *= rms_weight<ZERO_CENTERED, weight_t>(w.val[j]);
        dst.val[j] = static_cast<scalar_t>(value);
      }
      rms_store_output<scalar_t, weight_t, VEC_SIZE>(out_row + vidx * VEC_SIZE, dst);
    }
  }
}

// ===========================================================================
// Sub-warp (multi-row-per-warp) RMSNorm for TINY hidden sizes.
//
template <typename scalar_t, typename weight_t, int VEC_SIZE, int NUM_DIMS, bool HAS_WEIGHT,
          bool PER_TOKEN_WEIGHT, int BLOCK_SIZE, int LANES_PER_ROW, int ITEMS, bool ZERO_CENTERED>
__global__ __launch_bounds__(BLOCK_SIZE) void rms_norm_subwarp_kernel(
    scalar_t *__restrict__ out, const scalar_t *__restrict__ input,
    int64_t input_stride_d2, int64_t input_stride_d3, int64_t input_stride_d4,
    int64_t input_shape_d2, int64_t input_shape_d3,
    const weight_t *__restrict__ weight, int64_t weight_stride, float epsilon,
    int num_tokens, int hidden_size) {
  using Vec = vec_n_t<scalar_t, VEC_SIZE>;
  using WeightVec = vec_n_t<weight_t, VEC_SIZE>;
  const int row_in_block = threadIdx.x / LANES_PER_ROW;
  const int lane = threadIdx.x % LANES_PER_ROW;
  const int rows_per_block = BLOCK_SIZE / LANES_PER_ROW;
  const int token_idx = blockIdx.x * rows_per_block + row_in_block;
  if (token_idx >= num_tokens) return;

  const int vec_count = hidden_size / VEC_SIZE;

  int batch_idx = 0;
  const scalar_t *input_row;
  if constexpr (NUM_DIMS == 2) {
    batch_idx = token_idx;
    input_row = input + token_idx * input_stride_d2;
  } else if constexpr (NUM_DIMS == 3) {
    batch_idx = token_idx / input_shape_d2;
    const int b = token_idx / input_shape_d2;
    const int h = token_idx % input_shape_d2;
    input_row = input + b * input_stride_d3 + h * input_stride_d2;
  } else {
    batch_idx = token_idx / (input_shape_d3 * input_shape_d2);
    const int b = token_idx / (input_shape_d3 * input_shape_d2);
    const int rem = token_idx % (input_shape_d3 * input_shape_d2);
    const int s = rem / input_shape_d2;
    const int h = rem % input_shape_d2;
    input_row = input + b * input_stride_d4 + s * input_stride_d3 +
                h * input_stride_d2;
  }
  const Vec *input_vec = reinterpret_cast<const Vec *>(input_row);

  // Each lane owns ITEMS packs (strided by LANES_PER_ROW), cached in registers.
  Vec src[ITEMS];
  float variance = 0.0f;
#pragma unroll
  for (int it = 0; it < ITEMS; ++it) {
    const int vidx = lane + it * LANES_PER_ROW;
    Vec s{};
    if (vidx < vec_count) s = rms_load_input<scalar_t, weight_t, VEC_SIZE>(input_row + vidx * VEC_SIZE);
    src[it] = s;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const float x = static_cast<float>(s.val[j]);
      variance += x * x;
    }
  }

  // Reduction confined to this row's LANES_PER_ROW lane group.
  // For 16-lane rows use the C500/C600U-native __shfl_down_sync_16 primitive
  // (down-shift into lane 0), then broadcast lane 0's total back to the group.
  // Narrower groups (8) must keep the generic xor shuffle -- a 16-lane
  // primitive would cross into the neighbouring row.
  if constexpr (LANES_PER_ROW == 16) {
#pragma unroll
    for (int i = 8; i > 0; i >>= 1) {
      variance += __shfl_down_sync_16(0xffffffffffffffffULL, variance, i);
    }
    variance = __shfl_sync(0xffffffffffffffffULL, variance,
                           (threadIdx.x >> 4) << 4);
  } else {
#pragma unroll
    for (int offset = LANES_PER_ROW >> 1; offset > 0; offset >>= 1) {
      variance += __shfl_xor_sync(0xffffffffffffffffULL, variance, offset);
    }
  }
  const float s_variance =
      __builtin_mxc_rcpf(sqrtf(variance / hidden_size + epsilon));

  scalar_t *out_row = out + token_idx * hidden_size;
  auto *output_vec = reinterpret_cast<Vec *>(out_row);
  const weight_t *wbase = nullptr;
  if constexpr (HAS_WEIGHT) {
    wbase = PER_TOKEN_WEIGHT ? (weight + batch_idx * weight_stride) : weight;
  }
  const WeightVec *weight_vec = reinterpret_cast<const WeightVec *>(wbase);
#pragma unroll
  for (int it = 0; it < ITEMS; ++it) {
    const int vidx = lane + it * LANES_PER_ROW;
    if (vidx < vec_count) {
      Vec dst;
      if constexpr (HAS_WEIGHT) {
        const WeightVec w = weight_vec[vidx];
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
          const float value = static_cast<float>(src[it].val[j]) * s_variance *
                              rms_weight<ZERO_CENTERED, weight_t>(w.val[j]);
          dst.val[j] = static_cast<scalar_t>(value);
        }
      } else {
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
          dst.val[j] = static_cast<scalar_t>(
              static_cast<float>(src[it].val[j]) * s_variance);
        }
      }
      rms_store_output<scalar_t, weight_t, VEC_SIZE>(out_row + vidx * VEC_SIZE, dst);
    }
  }
}

// Existing two-pass implementation retained for unaligned and untested shapes.
template <typename scalar_t, typename weight_t, int VEC_SIZE, int NUM_DIMS, bool HAS_WEIGHT, bool PER_TOKEN_WEIGHT, bool ZERO_CENTERED>
__global__ void rms_norm_default_kernel(
    scalar_t *__restrict__ out, const scalar_t *__restrict__ input,
    int64_t input_stride_d2, int64_t input_stride_d3, int64_t input_stride_d4,
    int64_t input_shape_d2, int64_t input_shape_d3,
    const weight_t *__restrict__ weight, int64_t weight_stride, float epsilon, int num_tokens,
    int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  const int token_idx = blockIdx.x;
  int batch_idx = 0;

  if constexpr(NUM_DIMS == 2) {
      batch_idx = blockIdx.x;
  }
  else if constexpr(NUM_DIMS == 3) {
      batch_idx = blockIdx.x / input_shape_d2;
  }
  else if constexpr(NUM_DIMS == 4) {
      batch_idx = blockIdx.x / (input_shape_d3 * input_shape_d2);
  }
  const scalar_t *input_row = rms_input_row<scalar_t, NUM_DIMS>(
      input, blockIdx.x, input_stride_d2, input_stride_d3, input_stride_d4, input_shape_d2,
      input_shape_d3);

  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE> &vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      const float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };
  auto scalar_op = [&variance](const scalar_t &value) {
    const float x = static_cast<float>(value);
    variance += x * x;
  };
  if constexpr (std::is_same_v<weight_t, float> &&
                !std::is_same_v<scalar_t, float>) {
    for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
      vec_op(rms_load_input<scalar_t, weight_t, VEC_SIZE>(
          input_row + i * VEC_SIZE));
    }
  } else {
    vllm::vectorize_read_with_alignment<VEC_SIZE>(
        input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  variance = BlockReduce(reduce_store).Reduce(variance, CubAddOp{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar_t *out_row = out + blockIdx.x * hidden_size;
  const auto *input_vec =
      reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE> *>(input_row);
  auto *output_vec = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE> *>(out_row);

  if constexpr (HAS_WEIGHT) {
    if constexpr (!PER_TOKEN_WEIGHT) {
      const auto *weight_vec =
          reinterpret_cast<const vec_n_t<weight_t, VEC_SIZE> *>(weight);
      for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
        vec_n_t<scalar_t, VEC_SIZE> dst;
        const auto src = rms_load_input<scalar_t, weight_t, VEC_SIZE>(input_row + i * VEC_SIZE);
        const vec_n_t<weight_t, VEC_SIZE> w = weight_vec[i];
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
          const float x = static_cast<float>(src.val[j]);
          const float wf = rms_weight<ZERO_CENTERED, weight_t>(w.val[j]);
          dst.val[j] = rms_packed_result<scalar_t, weight_t, VEC_SIZE>(x * s_variance * wf);
        }
        rms_store_output<scalar_t, weight_t, VEC_SIZE>(out_row + i * VEC_SIZE, dst);
      }
    } else {
      const weight_t *token_weight = weight + batch_idx * weight_stride;
      const auto *weight_vec =
          reinterpret_cast<const vec_n_t<weight_t, VEC_SIZE> *>(token_weight);
      for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
        vec_n_t<scalar_t, VEC_SIZE> dst;
        const auto src = rms_load_input<scalar_t, weight_t, VEC_SIZE>(input_row + i * VEC_SIZE);
        const vec_n_t<weight_t, VEC_SIZE> w = weight_vec[i];
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
          const float x = static_cast<float>(src.val[j]);
          const float wf = rms_weight<ZERO_CENTERED, weight_t>(w.val[j]);
          dst.val[j] =
              rms_packed_result<scalar_t, weight_t, VEC_SIZE>(x * s_variance * wf);
        }
        rms_store_output<scalar_t, weight_t, VEC_SIZE>(out_row + i * VEC_SIZE, dst);
      }
    }
  }
  else {
    for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
      vec_n_t<scalar_t, VEC_SIZE> dst;
      const auto src = rms_load_input<scalar_t, weight_t, VEC_SIZE>(input_row + i * VEC_SIZE);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const float x = static_cast<float>(src.val[j]);
        dst.val[j] = static_cast<scalar_t>(x * s_variance);
      }
      rms_store_output<scalar_t, weight_t, VEC_SIZE>(out_row + i * VEC_SIZE, dst);
    }
  }
}

// The input packs stay in registers across the reduction. This eliminates the
// second global input read while keeping the cached representation packed.
template <typename scalar_t, typename weight_t, int VEC_SIZE, int NUM_DIMS, bool HAS_WEIGHT, bool PER_TOKEN_WEIGHT,
          int BLOCK_SIZE, int ITEMS_PER_THREAD, bool ZERO_CENTERED>
__global__ __launch_bounds__(BLOCK_SIZE) void rms_norm_cached_kernel(
    scalar_t *__restrict__ out, const scalar_t *__restrict__ input,
    int64_t input_stride_d2, int64_t input_stride_d3, int64_t input_stride_d4,
    int64_t input_shape_d2, int64_t input_shape_d3,
    const weight_t *__restrict__ weight, int64_t weight_stride, float epsilon, int num_tokens,
    int hidden_size) {
  using Vec = vec_n_t<scalar_t, VEC_SIZE>;
  using WeightVec = vec_n_t<weight_t, VEC_SIZE>;
  const int vec_count = hidden_size / VEC_SIZE;
  const int token_idx = blockIdx.x;
  int batch_idx = 0;

  if constexpr(NUM_DIMS == 2) {
      batch_idx = blockIdx.x;
  }
  else if constexpr(NUM_DIMS == 3) {
      batch_idx = blockIdx.x / input_shape_d2;
  }
  else if constexpr(NUM_DIMS == 4) {
      batch_idx = blockIdx.x / (input_shape_d3 * input_shape_d2);
  }
  const scalar_t *input_row = rms_input_row<scalar_t, NUM_DIMS>(
      input, blockIdx.x, input_stride_d2, input_stride_d3, input_stride_d4, input_shape_d2,
      input_shape_d3);
  const auto *input_vec = reinterpret_cast<const Vec *>(input_row);
  Vec cached[ITEMS_PER_THREAD];
  float variance = 0.0f;

#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int vec_idx = threadIdx.x + item * BLOCK_SIZE;
    Vec src{};
    if (vec_idx < vec_count)
      src = rms_load_input<scalar_t, weight_t, VEC_SIZE>(input_row + vec_idx * VEC_SIZE);
    cached[item] = src;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const float x = static_cast<float>(src.val[j]);
      variance += x * x;
    }
  }

  // Single-barrier SIMD-16 block reduction (C500-optimized).
  __shared__ float s_reduce[BLOCK_SIZE >> 4];
  variance = rms_block_reduce_sum<BLOCK_SIZE>(variance, s_reduce);
  const float s_variance =
      __builtin_mxc_rcpf(sqrtf(variance / hidden_size + epsilon));

  scalar_t *out_row = out + blockIdx.x * hidden_size;
  auto *output_vec = reinterpret_cast<Vec *>(out_row);
  const WeightVec *weight_vec = nullptr;
  if constexpr (HAS_WEIGHT) {
    if constexpr (!PER_TOKEN_WEIGHT) {
      weight_vec = reinterpret_cast<const WeightVec *>(weight);
    }
  }
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int vec_idx = threadIdx.x + item * BLOCK_SIZE;
    if (vec_idx < vec_count) { 
      Vec dst; 
      WeightVec w{}; 
      if constexpr (HAS_WEIGHT) { 
        if constexpr (PER_TOKEN_WEIGHT) {
          const weight_t *token_weight = weight + batch_idx * weight_stride;
          const auto *token_weight_vec =
              reinterpret_cast<const WeightVec *>(token_weight); 
          w = token_weight_vec[vec_idx];
        } else { 
          w = weight_vec[vec_idx];
        }
      }
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        float value = static_cast<float>(cached[item].val[j]) * s_variance;
        if constexpr (HAS_WEIGHT) {
          value *= rms_weight<ZERO_CENTERED, weight_t>(w.val[j]);
        }
        dst.val[j] = static_cast<scalar_t>(value);
      }
      rms_store_output<scalar_t, weight_t, VEC_SIZE>(out_row + vec_idx * VEC_SIZE, dst);
    }
  }
}

template <typename scalar_t, typename weight_t, int VEC_SIZE, int NUM_DIMS, bool HAS_WEIGHT,
          bool PER_TOKEN_WEIGHT, int BLOCK_SIZE, int ITEMS_PER_THREAD, bool ZERO_CENTERED>
__global__ __launch_bounds__(BLOCK_SIZE) void rms_norm_multirow_cached_kernel(
    scalar_t *__restrict__ out, const scalar_t *__restrict__ input,
    int64_t input_stride_d2, int64_t input_stride_d3, int64_t input_stride_d4,
    int64_t input_shape_d2, int64_t input_shape_d3,
    const weight_t *__restrict__ weight, int64_t weight_stride, float epsilon,
    int num_tokens, int hidden_size) {
  using Vec = vec_n_t<scalar_t, VEC_SIZE>;
  using WeightVec = vec_n_t<weight_t, VEC_SIZE>;
  constexpr int WARP = 64;
  constexpr int NW = BLOCK_SIZE / WARP;
  constexpr bool W_IN_REG = HAS_WEIGHT && !PER_TOKEN_WEIGHT;
  const int tid = threadIdx.x;
  const float inv_h = 1.0f / static_cast<float>(hidden_size);

  float wreg[W_IN_REG ? ITEMS_PER_THREAD : 1][VEC_SIZE];
  if constexpr (W_IN_REG) {
    const WeightVec *wv = reinterpret_cast<const WeightVec *>(weight);
#pragma unroll
    for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
      const WeightVec w = wv[tid + it * BLOCK_SIZE];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j)
        wreg[it][j] = rms_weight<ZERO_CENTERED, weight_t>(w.val[j]);
    }
  }

  __shared__ float red[2][NW > 1 ? NW : 1];
  int parity = 0;

  for (int token = blockIdx.x; token < num_tokens; token += gridDim.x) {
    const scalar_t *input_row = rms_input_row<scalar_t, NUM_DIMS>(
        input, token, input_stride_d2, input_stride_d3, input_stride_d4,
        input_shape_d2, input_shape_d3);
    const Vec *src = reinterpret_cast<const Vec *>(input_row);

    Vec x[ITEMS_PER_THREAD];
#pragma unroll
    for (int it = 0; it < ITEMS_PER_THREAD; ++it)
      x[it] = rms_load_input<scalar_t, weight_t, VEC_SIZE>(
          input_row + (tid + it * BLOCK_SIZE) * VEC_SIZE);

    float var = 0.f;
#pragma unroll
    for (int it = 0; it < ITEMS_PER_THREAD; ++it)
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const float v = static_cast<float>(x[it].val[j]);
        var += v * v;
      }
#pragma unroll
    for (int off = WARP >> 1; off > 0; off >>= 1)
      var += __shfl_xor_sync(0xffffffffffffffffULL, var, off);
    if constexpr (NW > 1) {
      if ((tid & (WARP - 1)) == 0) red[parity][tid / WARP] = var;
      __syncthreads();
      var = 0.f;
#pragma unroll
      for (int k = 0; k < NW; ++k) var += red[parity][k];
      parity ^= 1;
    }
    const float inv = __builtin_mxc_rcpf(sqrtf(var * inv_h + epsilon));

    const WeightVec *tw = nullptr;
    if constexpr (HAS_WEIGHT && PER_TOKEN_WEIGHT) {
      int batch_idx;
      if constexpr (NUM_DIMS == 2) batch_idx = token;
      else if constexpr (NUM_DIMS == 3) batch_idx = token / input_shape_d2;
      else batch_idx = token / (input_shape_d3 * input_shape_d2);
      tw = reinterpret_cast<const WeightVec *>(weight + batch_idx * weight_stride);
    }

    Vec *dst = reinterpret_cast<Vec *>(out + static_cast<int64_t>(token) * hidden_size);
#pragma unroll
    for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
      const int vidx = tid + it * BLOCK_SIZE;
      Vec d;
      if constexpr (HAS_WEIGHT && PER_TOKEN_WEIGHT) {
        const WeightVec w = tw[vidx];
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j)
          d.val[j] = rms_packed_result<scalar_t, weight_t, VEC_SIZE>(static_cast<float>(x[it].val[j]) * inv *
                                           rms_weight<ZERO_CENTERED, weight_t>(w.val[j]));
      } else if constexpr (W_IN_REG) {
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j)
          d.val[j] = rms_packed_result<scalar_t, weight_t, VEC_SIZE>(static_cast<float>(x[it].val[j]) * inv * wreg[it][j]);
      } else {
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j)
          d.val[j] = static_cast<scalar_t>(static_cast<float>(x[it].val[j]) * inv);
      }
      rms_store_output<scalar_t, weight_t, VEC_SIZE>(
          out + static_cast<int64_t>(token) * hidden_size + vidx * VEC_SIZE, d);
    }
  }
}

// Preserve the fp32 kernel's virtual lanes while packing two of them into one
// physical thread. This restores 16-byte low-precision I/O without changing the
// per-lane accumulation or the order of the 256-thread fp32 reduction.
template <typename scalar_t, int NUM_DIMS, bool ZERO_CENTERED>
__global__ __launch_bounds__(128) void rms_norm_mixed_4096_kernel(
    scalar_t *__restrict__ out, const scalar_t *__restrict__ input,
    int64_t stride2, int64_t stride3, int64_t stride4,
    int64_t shape2, int64_t shape3, const float *__restrict__ weight,
    float epsilon, int num_tokens) {
  using Vec = vec_n_t<scalar_t, 8>;
  const int tid = threadIdx.x;
  float wreg[4][8];
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const auto *wv = reinterpret_cast<const vec_n_t<float, 4> *>(weight);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const auto w = wv[(tid + it * 128) * 2 + k];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        wreg[it][k * 4 + j] = rms_weight<ZERO_CENTERED>(w.val[j]);
    }
  }
  __shared__ float red[2][4];
  int parity = 0;
  for (int token = blockIdx.x; token < num_tokens; token += gridDim.x) {
    const scalar_t *row = rms_input_row<scalar_t, NUM_DIMS>(
        input, token, stride2, stride3, stride4, shape2, shape3);
    Vec x[4];
    float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      x[it] = reinterpret_cast<const Vec *>(row)[tid + it * 128];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float a = static_cast<float>(x[it].val[j]);
        const float b = static_cast<float>(x[it].val[j + 4]);
        v0 += a * a;
        v1 += b * b;
      }
    }
    // Only the leader of each 32-thread group contributes to red. Its
    // downward SIMD-16 reduction has exactly the same addition tree as the
    // xor reduction, without the general shuffle overhead at offsets < 16.
    v0 += __shfl_xor_sync(0xffffffffffffffffULL, v0, 16);
    v1 += __shfl_xor_sync(0xffffffffffffffffULL, v1, 16);
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
      v0 += __shfl_down_sync_16(0xffffffffffffffffULL, v0, off);
      v1 += __shfl_down_sync_16(0xffffffffffffffffULL, v1, off);
    }
    if ((tid & 31) == 0) red[parity][tid / 32] = v0 + v1;
    __syncthreads();
    float var = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) var += red[parity][k];
    parity ^= 1;
    const float inv = __builtin_mxc_rcpf(sqrtf(var * (1.0f / 4096) + epsilon));
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      Vec dst;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        dst.val[j] = rms_packed_result<scalar_t, float, 4>(
            static_cast<float>(x[it].val[j]) * inv * wreg[it][j]);
      reinterpret_cast<Vec *>(out + static_cast<int64_t>(token) * 4096)[tid + it * 128] = dst;
    }
  }
}

// Small batches retain all 256 reference lanes to reduce decode latency.
template <typename scalar_t, int NUM_DIMS, bool ZERO_CENTERED>
__global__ __launch_bounds__(256) void rms_norm_mixed_4096_decode_kernel(
    scalar_t *__restrict__ out, const scalar_t *__restrict__ input,
    int64_t stride2, int64_t stride3, int64_t stride4,
    int64_t shape2, int64_t shape3, const float *__restrict__ weight,
    float epsilon, int num_tokens) {
  using Vec = vec_n_t<scalar_t, 4>;
  const int tid = threadIdx.x;
  float wreg[4][4];
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const auto *wv = reinterpret_cast<const vec_n_t<float, 4> *>(weight);
#pragma unroll
    for (int k = 0; k < 1; ++k) {
      const auto w = wv[tid + it * 256 + k];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        wreg[it][k * 4 + j] = rms_weight<ZERO_CENTERED>(w.val[j]);
    }
  }
  __shared__ float red[2][4];
  int parity = 0;
  for (int token = blockIdx.x; token < num_tokens; token += gridDim.x) {
    const scalar_t *row = rms_input_row<scalar_t, NUM_DIMS>(
        input, token, stride2, stride3, stride4, shape2, shape3);
    Vec x[4];
    float v0 = 0.0f;
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      x[it] = reinterpret_cast<const Vec *>(row)[tid + it * 256];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float a = static_cast<float>(x[it].val[j]);
        v0 += a * a;
      }
    }
    // Match the reference's 64-lane reduction; only group leaders write.
    v0 += __shfl_xor_sync(0xffffffffffffffffULL, v0, 32);
    v0 += __shfl_xor_sync(0xffffffffffffffffULL, v0, 16);
#pragma unroll
    for (int off = 8; off > 0; off >>= 1)
      v0 += __shfl_down_sync_16(0xffffffffffffffffULL, v0, off);
    if ((tid & 63) == 0) red[parity][tid / 64] = v0;
    __syncthreads();
    float var = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) var += red[parity][k];
    parity ^= 1;
    const float inv = __builtin_mxc_rcpf(sqrtf(var * (1.0f / 4096) + epsilon));
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      Vec dst;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        dst.val[j] = rms_packed_result<scalar_t, float, 4>(
            static_cast<float>(x[it].val[j]) * inv * wreg[it][j]);
      reinterpret_cast<Vec *>(out + static_cast<int64_t>(token) * 4096)[tid + it * 256] = dst;
    }
  }
}

// Four consecutive virtual fp32 lanes per physical lane. The reference's
// shuffle offsets 32,16,8,4 become 8,4,2,1; offsets 2,1 are evaluated locally.
// Thus 16 rows fit in a block without using the native lowp reduction tree.
template <typename scalar_t, int NUM_DIMS, int HIDDEN, bool ZERO_CENTERED>
__global__ __launch_bounds__(256) void rms_norm_mixed_small_kernel(
    scalar_t *__restrict__ out, const scalar_t *__restrict__ input,
    int64_t stride2, int64_t stride3, int64_t stride4,
    int64_t shape2, int64_t shape3, const float *__restrict__ weight,
    float epsilon, int num_tokens, int hidden_size) {
  const int token = blockIdx.x * 16 + threadIdx.x / 16;
  if (token >= num_tokens) return;
  const int col = (threadIdx.x & 15) * 16;
  const scalar_t *row = rms_input_row<scalar_t, NUM_DIMS>(
      input, token, stride2, stride3, stride4, shape2, shape3);
  using Vec = vec_n_t<scalar_t, 8>;
  Vec x[2]{};
  if (col < HIDDEN) {
#pragma unroll
    for (int i = 0; i < 2; ++i)
      x[i] = reinterpret_cast<const Vec *>(row + col)[i];
  }
  float v[4]{};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float a = static_cast<float>(x[k / 2].val[(k % 2) * 4 + j]);
      v[k] += a * a;
    }
  }
#pragma unroll
  for (int off = 8; off > 0; off >>= 1) {
#pragma unroll
    for (int k = 0; k < 4; ++k)
      v[k] += __shfl_xor_sync(0xffffffffffffffffULL, v[k], off);
  }
  const float variance = (v[0] + v[2]) + (v[1] + v[3]);
  const float inv = __builtin_mxc_rcpf(sqrtf(variance / hidden_size + epsilon));
  if (col < HIDDEN) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      Vec dst;
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const auto w = reinterpret_cast<const vec_n_t<float, 4> *>(weight + col)[i * 2 + k];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          dst.val[k * 4 + j] = static_cast<scalar_t>(static_cast<float>(x[i].val[k * 4 + j]) * inv *
              rms_weight<ZERO_CENTERED>(w.val[j]));
      }
      reinterpret_cast<Vec *>(out + static_cast<int64_t>(token) * HIDDEN + col)[i] = dst;
    }
  }
}

template <typename scalar_t, typename weight_t, int VEC_SIZE, int NUM_DIMS, bool HAS_WEIGHT, bool PER_TOKEN_WEIGHT, bool ZERO_CENTERED>
bool launch_rms_norm_cached(int block_size, int items_per_thread, dim3 grid,
                            cudaStream_t stream, scalar_t *out,
                            const scalar_t *input, int64_t input_stride_d2,
                            int64_t input_stride_d3, int64_t input_stride_d4,
                            int64_t input_shape_d2, int64_t input_shape_d3,
                            const weight_t *weight, int64_t weight_stride, float epsilon,
                            int num_tokens, int hidden_size) {
#define LAUNCH_RMS_CACHED(BLOCK, ITEMS)                                        \
  rms_norm_cached_kernel<scalar_t, weight_t, VEC_SIZE, NUM_DIMS, HAS_WEIGHT, PER_TOKEN_WEIGHT, BLOCK,      \
                         ITEMS, ZERO_CENTERED><<<grid, BLOCK, 0, stream>>>(    \
      out, input, input_stride_d2, input_stride_d3, input_stride_d4,           \
      input_shape_d2, input_shape_d3, weight, weight_stride, epsilon, num_tokens,             \
      hidden_size)

#define DISPATCH_ITEMS(BLOCK)                                                  \
  switch (items_per_thread) {                                                  \
  case 1:                                                                      \
    LAUNCH_RMS_CACHED(BLOCK, 1);                                               \
    return true;                                                               \
  case 2:                                                                      \
    LAUNCH_RMS_CACHED(BLOCK, 2);                                               \
    return true;                                                               \
  case 3:                                                                      \
    LAUNCH_RMS_CACHED(BLOCK, 3);                                               \
    return true;                                                               \
  case 4:                                                                      \
    LAUNCH_RMS_CACHED(BLOCK, 4);                                               \
    return true;                                                               \
  case 5:                                                                      \
    LAUNCH_RMS_CACHED(BLOCK, 5);                                               \
    return true;                                                               \
  case 6:                                                                      \
    LAUNCH_RMS_CACHED(BLOCK, 6);                                               \
    return true;                                                               \
  case 7:                                                                      \
    LAUNCH_RMS_CACHED(BLOCK, 7);                                               \
    return true;                                                               \
  case 8:                                                                      \
    LAUNCH_RMS_CACHED(BLOCK, 8);                                               \
    return true;                                                               \
  default:                                                                     \
    return false;                                                              \
  }

  if (block_size == 512) {
    DISPATCH_ITEMS(512);
  }
  DISPATCH_ITEMS(256);
#undef DISPATCH_ITEMS
#undef LAUNCH_RMS_CACHED
}


template <typename scalar_t, typename weight_t, int VEC_SIZE, int NUM_DIMS, bool HAS_WEIGHT, bool PER_TOKEN_WEIGHT, bool ZERO_CENTERED>
bool launch_rms_norm_multirow_cached(int block_size, int items_per_thread, dim3 grid,
                                     cudaStream_t stream, scalar_t *out,
                                     const scalar_t *input, int64_t input_stride_d2,
                                     int64_t input_stride_d3, int64_t input_stride_d4,
                                     int64_t input_shape_d2, int64_t input_shape_d3,
                                     const weight_t *weight, int64_t weight_stride, float epsilon,
                                     int num_tokens, int hidden_size) {
#define LAUNCH_RMS_MULTIROW(BLOCK, ITEMS)                                                                \
  rms_norm_multirow_cached_kernel<scalar_t, weight_t, VEC_SIZE, NUM_DIMS, HAS_WEIGHT, PER_TOKEN_WEIGHT,  \
                                  BLOCK, ITEMS, ZERO_CENTERED><<<grid, BLOCK, 0, stream>>>(              \
      out, input, input_stride_d2, input_stride_d3, input_stride_d4,                                     \
      input_shape_d2, input_shape_d3, weight, weight_stride, epsilon, num_tokens,                        \
      hidden_size)

#define DISPATCH_MULTIROW_ITEMS(BLOCK)                                         \
  switch (items_per_thread) {                                                  \
  case 1:                                                                      \
    LAUNCH_RMS_MULTIROW(BLOCK, 1);                                             \
    return true;                                                               \
  case 2:                                                                      \
    LAUNCH_RMS_MULTIROW(BLOCK, 2);                                             \
    return true;                                                               \
  case 3:                                                                      \
    LAUNCH_RMS_MULTIROW(BLOCK, 3);                                             \
    return true;                                                               \
  case 4:                                                                      \
    LAUNCH_RMS_MULTIROW(BLOCK, 4);                                             \
    return true;                                                               \
  default:                                                                     \
    return false;                                                              \
  }

  if (block_size == 128) {
    DISPATCH_MULTIROW_ITEMS(128);
  }
  if (block_size == 256) {
    DISPATCH_MULTIROW_ITEMS(256);
  }
  if (block_size == 512) {
    DISPATCH_MULTIROW_ITEMS(512);
  }
  return false;
}

inline int select_rms_cached_block(int num_tokens, int hidden_size) {
  // block=512 is only usable by the cached path when vec_count >= 512, i.e.
  // hidden >= 4096 (bf16 vec_size 8). For medium hidden (2048..4088) picking
  // 512 would make can_use_cached (which requires vec_count >= cached_block)
  // reject the shape and dump it onto the slow streaming default kernel.
  // Pick 256 there so these rows still ride the cached kernel.
  if (num_tokens < 64) {
    if (hidden_size >= 4096) return 512;
    return hidden_size >= 2048 ? 256 : 0;
  }
  if (num_tokens <= 256) {
    if (hidden_size >= 4096) return 512;
    return hidden_size >= 2048 ? 256 : 0;
  }
  if (num_tokens < 1024) {
    return 256;
  }
  return hidden_size >= 7168 ? 512 : 256;
}

// Launcher for the warp-per-row kernel. Returns false if the shape is not
// handled (caller falls back to another path). BLOCK_SIZE=256 => 4 rows/block.
template <typename scalar_t, typename weight_t, int VEC_SIZE, int NUM_DIMS, bool HAS_WEIGHT,
          bool PER_TOKEN_WEIGHT, bool ZERO_CENTERED>
bool launch_rms_norm_warp(int items_per_lane, cudaStream_t stream,
                          scalar_t *out, const scalar_t *input,
                          int64_t input_stride_d2, int64_t input_stride_d3,
                          int64_t input_stride_d4, int64_t input_shape_d2,
                          int64_t input_shape_d3, const weight_t *weight,
                          int64_t weight_stride, float epsilon, int num_tokens,
                          int hidden_size) {
  constexpr int BLOCK = 512;   // 8 warps (rows) per block
  constexpr int ROWS_PER_BLOCK = BLOCK / 64;
  const dim3 grid((num_tokens + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
#define LAUNCH_RMS_WARP(ITEMS)                                                  \
  rms_norm_warp_kernel<scalar_t, weight_t, VEC_SIZE, NUM_DIMS, HAS_WEIGHT,      \
                       PER_TOKEN_WEIGHT, BLOCK, ITEMS, ZERO_CENTERED>           \
      <<<grid, BLOCK, 0, stream>>>(out, input, input_stride_d2,                 \
                                   input_stride_d3, input_stride_d4,            \
                                   input_shape_d2, input_shape_d3, weight,      \
                                   weight_stride, epsilon, num_tokens,          \
                                   hidden_size);                                \
  return true
  switch (items_per_lane) {
    case 1: LAUNCH_RMS_WARP(1);
    case 2: LAUNCH_RMS_WARP(2);
    case 3: LAUNCH_RMS_WARP(3);
    case 4: LAUNCH_RMS_WARP(4);
    default: return false;
  }
#undef LAUNCH_RMS_WARP
}

// Launcher for the sub-warp kernel (tiny hidden). lanes_per_row is the smallest
// power-of-2 in {16,32} that is >= vec_count. Returns false if unsupported.
template <typename scalar_t, typename weight_t, int VEC_SIZE, int NUM_DIMS, bool HAS_WEIGHT,
          bool PER_TOKEN_WEIGHT, bool ZERO_CENTERED>
bool launch_rms_norm_subwarp(int lanes_per_row, int items,
                             cudaStream_t stream, scalar_t *out,
                             const scalar_t *input, int64_t input_stride_d2,
                             int64_t input_stride_d3, int64_t input_stride_d4,
                             int64_t input_shape_d2, int64_t input_shape_d3,
                             const weight_t *weight, int64_t weight_stride,
                             float epsilon, int num_tokens, int hidden_size) {
  constexpr int BLOCK = 256;
#define LAUNCH_RMS_SUBWARP(LPR, ITEMS)                                          \
  do {                                                                          \
    constexpr int ROWS_PER_BLOCK = BLOCK / (LPR);                              \
    const dim3 grid((num_tokens + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);       \
    rms_norm_subwarp_kernel<scalar_t, weight_t, VEC_SIZE, NUM_DIMS, HAS_WEIGHT,               \
                            PER_TOKEN_WEIGHT, BLOCK, (LPR), (ITEMS), ZERO_CENTERED>           \
        <<<grid, BLOCK, 0, stream>>>(out, input, input_stride_d2,              \
                                     input_stride_d3, input_stride_d4,         \
                                     input_shape_d2, input_shape_d3, weight,   \
                                     weight_stride, epsilon, num_tokens,       \
                                     hidden_size);                             \
    return true;                                                              \
  } while (0)
  if (items == 1) {
    switch (lanes_per_row) {
      case 16: LAUNCH_RMS_SUBWARP(16, 1);
      case 32: LAUNCH_RMS_SUBWARP(32, 1);
      default: return false;
    }
  } else if (items == 2) {
    switch (lanes_per_row) {
      case 8: LAUNCH_RMS_SUBWARP(8, 2);
      case 16: LAUNCH_RMS_SUBWARP(16, 2);
      default: return false;
    }
  } else if (items == 4 && lanes_per_row == 16) {
    LAUNCH_RMS_SUBWARP(16, 4);
  }
  return false;
#undef LAUNCH_RMS_SUBWARP
}

inline bool rms_rows_are_vector_aligned(int num_dims, int vec_size,
                                        int64_t input_stride_d2,
                                        int64_t input_stride_d3,
                                        int64_t input_stride_d4) {
  if (input_stride_d2 % vec_size != 0)
    return false;
  if (num_dims >= 3 && input_stride_d3 % vec_size != 0)
    return false;
  if (num_dims >= 4 && input_stride_d4 % vec_size != 0)
    return false;
  return true;
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, typename weight_t, int width, bool HasWeight, bool ZERO_CENTERED>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,        // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,     // [..., hidden_size]
    const weight_t* __restrict__ weight, // [hidden_size], nullptr if !HasWeight
    const float epsilon,
    const int num_tokens,
    const int hidden_size,
    const int64_t residual_stride) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  const int64_t vec_input_stride = input_stride / width;

  __shared__ float s_variance;
  float variance = 0.0f;

  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<weight_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int64_t id = blockIdx.x * residual_stride / width + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;

    _f16Vec<scalar_t, width> temp = input_v[strided_id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int64_t id = blockIdx.x * residual_stride / width + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;

    _f16Vec<scalar_t, width> res = residual_v[id];
    _f16Vec<scalar_t, width> out;

    using Converter = _typeConvert<scalar_t>;

    if constexpr (HasWeight) {
      _f16Vec<weight_t, width> w = weight_v[idx];

#pragma unroll
      for (int j = 0; j < width; ++j) {
        float x = Converter::convert(res.data[j]);
        float wf = vllm::rms_weight<ZERO_CENTERED, weight_t>(w.data[j]);
        out.data[j] = Converter::convert(x * s_variance * wf);
      }
    } else {
#pragma unroll
      for (int j = 0; j < width; ++j) {
        float x = Converter::convert(res.data[j]);
        out.data[j] = Converter::convert(x * s_variance);
      }
    }

    input_v[strided_id] = out;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, typename weight_t, int width, bool HasWeight, bool ZERO_CENTERED>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,        // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,     // [..., hidden_size]
    const weight_t* __restrict__ weight, // [hidden_size], nullptr if !HasWeight
    const float epsilon,
    const int num_tokens,
    const int hidden_size,
    const int64_t residual_stride) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * residual_stride + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * residual_stride + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * residual_stride + idx];

    if constexpr (HasWeight) {
      float w = vllm::rms_weight<ZERO_CENTERED, weight_t>(weight[idx]);
      input[blockIdx.x * input_stride + idx] =
          (scalar_t)(x * s_variance * w);
    } else {
      input[blockIdx.x * input_stride + idx] =
          (scalar_t)(x * s_variance);
    }
  }
}

} // namespace vllm

inline const cudaDeviceProp& GetDeviceProp() {
  static cudaDeviceProp prop = [] {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    return p;
  }();
  return prop;
}

void rms_norm(torch::Tensor &out, torch::Tensor &input,
              std::optional<torch::Tensor> weight, double epsilon,
              bool zero_centered) {
  TORCH_CHECK(!zero_centered || weight.has_value(),
              "rms_norm zero_centered requires weight");
  TORCH_CHECK(out.is_contiguous());
  if (input.stride(-1) != 1)
    input = input.contiguous();
  TORCH_CHECK(input.stride(-1) == 1);
  const bool has_weight = weight.has_value();
  const bool mixed_fp32_weight =
      has_weight && weight->scalar_type() == at::ScalarType::Float &&
      (input.scalar_type() == at::ScalarType::Half ||
       input.scalar_type() == at::ScalarType::BFloat16);
  const int hidden_size = input.size(-1);
  if (mixed_fp32_weight) {
    TORCH_CHECK(input.dim() >= 2 && input.dim() <= 4,
                "rms_norm input must have 2, 3 or 4 dimensions");
    TORCH_CHECK(hidden_size > 0, "rms_norm hidden size must be positive");
    TORCH_CHECK(out.scalar_type() == input.scalar_type() &&
                    out.sizes() == input.sizes(),
                "rms_norm out must have the same dtype and shape as input");
    TORCH_CHECK(out.device() == input.device() &&
                    weight->device() == input.device(),
                "rms_norm tensors must be on the same device");
  }
  const int num_tokens = input.numel() / hidden_size;
  bool per_token_weight = false;
  int64_t weight_stride = 0;

  if (has_weight) {
    TORCH_CHECK(weight->is_contiguous());
    TORCH_CHECK(weight->scalar_type() == at::ScalarType::Float ||
                weight->scalar_type() == input.scalar_type(),
                "weight dtype must be float or the same as input dtype");
    TORCH_CHECK(weight->dim() <=2 , "rms_norm weight only supports 1D or 2D");
    if (weight->dim() == 1) {
      TORCH_CHECK(weight->size(0) == hidden_size);
      per_token_weight = false;
    } else if (weight->dim() == 2) {
      // 修改：
      // weight shape == [input.size(0), input.size(-1)]
      TORCH_CHECK(weight->size(0) == input.size(0));
      TORCH_CHECK(weight->size(-1) == input.size(-1));
      weight_stride = weight->stride(0);
      per_token_weight = true;
    } else {
      TORCH_CHECK(false,
                  "rms_norm weight only supports 1D or 2D");
    }
  }

  // No kernel may be launched with a zero-sized grid. Keep validation above
  // the early return, including the existing non-contiguous out rejection.
  if (num_tokens == 0) return;

  const int num_dims = input.dim();
  const int64_t input_stride_d2 = input.stride(-2);
  const int64_t input_stride_d3 = num_dims >= 3 ? input.stride(-3) : 0;
  const int64_t input_stride_d4 = num_dims >= 4 ? input.stride(-4) : 0;
  const int64_t input_shape_d2 = num_dims >= 3 ? input.size(-2) : 0;
  const int64_t input_shape_d3 = num_dims >= 4 ? input.size(-3) : 0;

  const dim3 grid(num_tokens);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  
  cudaDeviceProp prop = GetDeviceProp();
  bool is_c600 = prop.sharedMemPerBlock > 65536;

  VLLM_DISPATCH_RANK234(num_dims, [&] {
    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
      auto dispatch_weight = [&](auto weight_tag) {
        using weight_t = decltype(weight_tag);
        constexpr bool mixed_storage = std::is_same_v<weight_t, float> &&
                                       !std::is_same_v<scalar_t, float>;
        const weight_t *weight_ptr =
            has_weight ? weight->data_ptr<weight_t>() : nullptr;
        // Mixed input must reproduce the float-input kernel's arithmetic
        // grouping and reduction tree. Storage stays low precision, but the
        // logical vector width and all launch decisions follow float input.
        constexpr int full_vec_size =
            16 / (std::is_same_v<weight_t, float> ? sizeof(float)
                                                : sizeof(scalar_t));
        const int vec_size = std::gcd(full_vec_size, hidden_size);
        // These specializations emulate exactly the float dispatch below.
        if constexpr (mixed_storage) {
          if (is_c600 && has_weight && !per_token_weight &&
              reinterpret_cast<std::uintptr_t>(weight_ptr) % 16 == 0 &&
              reinterpret_cast<std::uintptr_t>(input.data_ptr()) % 16 == 0 &&
              reinterpret_cast<std::uintptr_t>(out.data_ptr()) % 16 == 0 &&
              vllm::rms_rows_are_vector_aligned(num_dims, 8, input_stride_d2,
                                               input_stride_d3, input_stride_d4)) {
            auto launch_mixed = [&](auto centered_tag) {
              constexpr bool centered = decltype(centered_tag)::value;
              if (hidden_size == 4096) {
                const int blocks = std::min(num_tokens, prop.multiProcessorCount * 32);
                if (num_tokens <= 64) {
                  vllm::rms_norm_mixed_4096_decode_kernel<scalar_t, tensor_rank, centered>
                      <<<blocks, 256, 0, stream>>>(out.data_ptr<scalar_t>(),
                          input.data_ptr<scalar_t>(), input_stride_d2, input_stride_d3,
                          input_stride_d4, input_shape_d2, input_shape_d3, weight_ptr,
                          static_cast<float>(epsilon), num_tokens);
                } else {
                vllm::rms_norm_mixed_4096_kernel<scalar_t, tensor_rank, centered>
                    <<<blocks, 128, 0, stream>>>(out.data_ptr<scalar_t>(),
                        input.data_ptr<scalar_t>(), input_stride_d2, input_stride_d3,
                        input_stride_d4, input_shape_d2, input_shape_d3, weight_ptr,
                        static_cast<float>(epsilon), num_tokens);
                }
                return true;
              }
              if (num_tokens >= 1024 && (hidden_size == 192 || hidden_size == 256)) {
                if (hidden_size == 192) {
                vllm::rms_norm_mixed_small_kernel<scalar_t, tensor_rank, 192, centered>
                    <<<(num_tokens + 15) / 16, 256, 0, stream>>>(out.data_ptr<scalar_t>(),
                        input.data_ptr<scalar_t>(), input_stride_d2, input_stride_d3,
                        input_stride_d4, input_shape_d2, input_shape_d3, weight_ptr,
                        static_cast<float>(epsilon), num_tokens, hidden_size);
                } else {
                vllm::rms_norm_mixed_small_kernel<scalar_t, tensor_rank, 256, centered>
                    <<<(num_tokens + 15) / 16, 256, 0, stream>>>(out.data_ptr<scalar_t>(),
                        input.data_ptr<scalar_t>(), input_stride_d2, input_stride_d3,
                        input_stride_d4, input_shape_d2, input_shape_d3, weight_ptr,
                        static_cast<float>(epsilon), num_tokens, hidden_size);
                }
                return true;
              }
              return false;
            };
            if (zero_centered ? launch_mixed(std::true_type{})
                              : launch_mixed(std::false_type{})) return;
          }
        }
        const bool batch_invariant_launch = vllm::vllm_is_batch_invariant();
        const int fallback_max_block =
            batch_invariant_launch ? 1024 : (num_tokens <= 256 ? 1024 : 256);
        const int fallback_block =
            std::min(hidden_size / vec_size, fallback_max_block);

        VLLM_DISPATCH_VEC_SIZE(vec_size, [&] {
          bool launched = false;
          if constexpr (vec_size == full_vec_size) {
            const int cached_block =
                vllm::select_rms_cached_block(num_tokens, hidden_size);
            const int vec_count = hidden_size / vec_size;
            const int items_per_thread =
                cached_block == 0 ? 0
                                  : (vec_count + cached_block - 1) / cached_block;
            constexpr size_t alignment = sizeof(scalar_t) * vec_size;
            constexpr size_t weight_alignment = sizeof(weight_t) * vec_size;
            const bool pointers_aligned =
                (mixed_storage ||
                 (reinterpret_cast<std::uintptr_t>(input.data_ptr()) % alignment == 0 &&
                  reinterpret_cast<std::uintptr_t>(out.data_ptr()) % alignment == 0)) &&
                (!has_weight ||
                reinterpret_cast<std::uintptr_t>(weight_ptr) % weight_alignment == 0);
            // Route to the cached kernel only when the row fully utilizes the
            // block (vec_count >= cached_block), keeping short rows on default.
            const bool can_use_cached =
                cached_block != 0 && vec_count >= cached_block &&
                items_per_thread >= 1 && items_per_thread <= 8 &&
                pointers_aligned &&
                (mixed_storage || vllm::rms_rows_are_vector_aligned(
                    num_dims, vec_size, input_stride_d2, input_stride_d3,
                    input_stride_d4));

            // Warp-per-row path for SMALL hidden with enough rows to fill the
            // GPU. Each 64-lane warp reduces one row via pure register shuffle
            // (no shared memory / __syncthreads). vec_count must fit in
            // 64 * ITEMS_PER_LANE lanes (ITEMS_PER_LANE <= 4 => vec_count <= 256).
            // The warp path packs 8 rows/block, so it only fills the 104 SMs when
            // there are enough rows (>= ~1024). For fewer rows the one-block-per-
            // row cached path yields a larger grid and higher occupancy, so we
            // restrict the warp path to the many-token regime.
            const int items_per_lane = (vec_count + 63) / 64;
            const bool rows_aligned = mixed_storage ||
                vllm::rms_rows_are_vector_aligned(
                    num_dims, vec_size, input_stride_d2, input_stride_d3,
                    input_stride_d4);
            const bool can_use_warp =
                vec_count > 32 && vec_count <= 256 && items_per_lane <= 4 &&
                num_tokens >= 1024 && pointers_aligned && rows_aligned;

            // Sub-warp (multi-row-per-warp) path for TINY hidden: when a whole row
            // fits in <=32 vec packs a full warp would idle half its lanes, so we
            // pack lane groups (one row each) to keep all lanes busy. Giving each
            // lane 2 packs (ITEMS=2) halves the lanes/row, packing 2x more rows
            // per warp and doubling per-thread bytes for tiny rows.
            // Tiny rows never fill a one-block-per-row grid well, so a lower token
            // threshold is fine here.
            int lanes_per_row = 0;
            int subwarp_items = 1;
            if (vec_count <= 16) {
              // e.g. H=128 (vec_count 16): 8 lanes/row x 2 packs, 8 rows/warp.
              lanes_per_row = 8;
              subwarp_items = 2;
            } else if (vec_count <= 32) {
              // e.g. H=256 (vec_count 32): 16 lanes/row x 2 packs, 4 rows/warp.
              lanes_per_row = 16;
              subwarp_items = 2;
            } else if (vec_count == 64 && num_tokens <= 64) {
              // Tiny-grid H=512: use a native 16-lane subgroup, four 128-bit
              // packs per lane, and four independent rows per 64-lane warp.
              // This removes the block reduction/barrier while retaining one
              // global read of input. Restrict to <=64 rows so the established
              // many-token routing remains unchanged.
              lanes_per_row = 16;
              subwarp_items = 4;
            }
            const bool tiny_subwarp_profitable =
                vec_count >= 32 && num_tokens <= 64 && num_tokens >= 16;
            const bool can_use_subwarp =
                lanes_per_row != 0 &&
                (num_tokens >= 256 || tiny_subwarp_profitable) &&
                pointers_aligned && rows_aligned;

            if (can_use_subwarp) {
              #define LAUNCH_RMS_NORM_SUBWARP(HAS_WEIGHT, PER_TOKEN_WEIGHT, WEIGHT, WEIGHT_STRIDE, ZERO_CENTERED)                                     \
                launched = vllm::launch_rms_norm_subwarp<scalar_t, weight_t, vec_size, tensor_rank, HAS_WEIGHT, PER_TOKEN_WEIGHT, ZERO_CENTERED>(     \
                        lanes_per_row, subwarp_items, stream, out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),                                   \
                        input_stride_d2, input_stride_d3, input_stride_d4, input_shape_d2, input_shape_d3, WEIGHT, WEIGHT_STRIDE,                     \
                        static_cast<float>(epsilon), num_tokens, hidden_size)

              if (!has_weight) {
                if (zero_centered) {
                  LAUNCH_RMS_NORM_SUBWARP(false, false, nullptr, 0, true);
                } else {
                  LAUNCH_RMS_NORM_SUBWARP(false, false, nullptr, 0, false);
                }
              } else if (per_token_weight) {
                if (zero_centered) {
                  LAUNCH_RMS_NORM_SUBWARP(true, true, weight_ptr, weight_stride, true);
                } else {
                  LAUNCH_RMS_NORM_SUBWARP(true, true, weight_ptr, weight_stride, false);
                }
              } else {
                if (zero_centered) {
                  LAUNCH_RMS_NORM_SUBWARP(true, false, weight_ptr, 0, true);
                } else {
                  LAUNCH_RMS_NORM_SUBWARP(true, false, weight_ptr, 0, false);
                }
              }
            }

            if (!launched && can_use_warp) {
              #define LAUNCH_RMS_NORM_WARP(HAS_WEIGHT, PER_TOKEN_WEIGHT, WEIGHT, WEIGHT_STRIDE, ZERO_CENTERED)                                       \
                launched = vllm::launch_rms_norm_warp<scalar_t, weight_t, vec_size, tensor_rank, HAS_WEIGHT, PER_TOKEN_WEIGHT, ZERO_CENTERED>(       \
                        items_per_lane, stream, out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),                                                \
                        input_stride_d2, input_stride_d3, input_stride_d4, input_shape_d2, input_shape_d3, WEIGHT, WEIGHT_STRIDE,                    \
                        static_cast<float>(epsilon), num_tokens, hidden_size)

              if (!has_weight) {
                if (zero_centered) {
                  LAUNCH_RMS_NORM_WARP(false, false, nullptr, 0, true);
                } else {
                  LAUNCH_RMS_NORM_WARP(false, false, nullptr, 0, false);
                }
              } else if (per_token_weight) {
                if (zero_centered) {
                  LAUNCH_RMS_NORM_WARP(true, true, weight_ptr, weight_stride, true);
                } else {
                  LAUNCH_RMS_NORM_WARP(true, true, weight_ptr, weight_stride, false);
                }
              } else {
                if (zero_centered) {
                  LAUNCH_RMS_NORM_WARP(true, false, weight_ptr, 0, true);
                } else {
                  LAUNCH_RMS_NORM_WARP(true, false, weight_ptr, 0, false);
                }
              }
            }

            if (!launched && can_use_cached) {
              int mr_block = vec_count <= 512 ? 128 : (vec_count <= 1024 ? 256 : 512);
              if (is_c600 && vec_count % mr_block == 0) {
                int mr_items = vec_count / mr_block;
                dim3 mr_grid = dim3(std::max(1, std::min(num_tokens, prop.multiProcessorCount * (3072 / mr_block))));
  
                #define LAUNCH_RMS_NORM_MULTIROW(HAS_WEIGHT, PER_TOKEN_WEIGHT, WEIGHT, WEIGHT_STRIDE, ZERO_CENTERED)                                        \
                  launched = vllm::launch_rms_norm_multirow_cached<scalar_t, weight_t, vec_size, tensor_rank, HAS_WEIGHT, PER_TOKEN_WEIGHT, ZERO_CENTERED>( \
                        mr_block, mr_items, mr_grid, stream, out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),                                          \
                        input_stride_d2, input_stride_d3, input_stride_d4, input_shape_d2, input_shape_d3, WEIGHT, WEIGHT_STRIDE,                           \
                        static_cast<float>(epsilon), num_tokens, hidden_size)

                if (!has_weight) {
                  if (zero_centered) {
                    LAUNCH_RMS_NORM_MULTIROW(false, false, nullptr, 0, true);
                  } else {
                    LAUNCH_RMS_NORM_MULTIROW(false, false, nullptr, 0, false);
                  }
                } else if (per_token_weight) {
                  if (zero_centered) {
                    LAUNCH_RMS_NORM_MULTIROW(true, true, weight_ptr, weight_stride, true);
                  } else {
                    LAUNCH_RMS_NORM_MULTIROW(true, true, weight_ptr, weight_stride, false);
                  }
                } else {
                  if (zero_centered) {
                    LAUNCH_RMS_NORM_MULTIROW(true, false, weight_ptr, 0, true);
                  } else {
                    LAUNCH_RMS_NORM_MULTIROW(true, false, weight_ptr, 0, false);
                  }
                }
              }
              else {
                  #define LAUNCH_RMS_NORM_CACHED(HAS_WEIGHT, PER_TOKEN_WEIGHT, WEIGHT, WEIGHT_STRIDE, ZERO_CENTERED)                                     \
                  launched = vllm::launch_rms_norm_cached<scalar_t, weight_t, vec_size, tensor_rank, HAS_WEIGHT, PER_TOKEN_WEIGHT, ZERO_CENTERED>(     \
                        cached_block, items_per_thread, grid, stream, out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),                            \
                        input_stride_d2, input_stride_d3, input_stride_d4, input_shape_d2, input_shape_d3, WEIGHT, WEIGHT_STRIDE,                      \
                        static_cast<float>(epsilon), num_tokens, hidden_size)

                if (!has_weight) {
                  if (zero_centered) {
                    LAUNCH_RMS_NORM_CACHED(false, false, nullptr, 0, true);
                  } else {
                    LAUNCH_RMS_NORM_CACHED(false, false, nullptr, 0, false);
                  }
                } else if (per_token_weight) {
                  if (zero_centered) {
                    LAUNCH_RMS_NORM_CACHED(true, true, weight_ptr, weight_stride, true);
                  } else {
                    LAUNCH_RMS_NORM_CACHED(true, true, weight_ptr, weight_stride, false);
                  }
                } else {
                  if (zero_centered) {
                    LAUNCH_RMS_NORM_CACHED(true, false, weight_ptr, 0, true);
                  } else {
                    LAUNCH_RMS_NORM_CACHED(true, false, weight_ptr, 0, false);
                  }
                }
              }
            }
          }

          if (!launched) {
            const dim3 block(fallback_block);
            #define RMS_NORM_DEFAULT_KERNEL(HAS_WEIGHT, PER_TOKEN_WEIGHT, WEIGHT, WEIGHT_STRIDE, ZERO_CENTERED)                                                       \
              vllm::rms_norm_default_kernel<scalar_t, weight_t, vec_size, tensor_rank, HAS_WEIGHT, PER_TOKEN_WEIGHT, ZERO_CENTERED><<<grid, block, 0, stream>>>(      \
                      out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), input_stride_d2, input_stride_d3,                                                         \
                      input_stride_d4, input_shape_d2, input_shape_d3, WEIGHT, WEIGHT_STRIDE, static_cast<float>(epsilon),                                            \
                      num_tokens, hidden_size)

            if (!has_weight) {
              if (zero_centered) {
                RMS_NORM_DEFAULT_KERNEL(false, false, nullptr, 0, true);
              } else {
                RMS_NORM_DEFAULT_KERNEL(false, false, nullptr, 0, false);
              }
            } else if (per_token_weight) {
              if (zero_centered) {
                RMS_NORM_DEFAULT_KERNEL(true, true, weight_ptr, weight_stride, true);
              } else {
                RMS_NORM_DEFAULT_KERNEL(true, true, weight_ptr, weight_stride, false);
              }
            } else {
              if (zero_centered) {
                RMS_NORM_DEFAULT_KERNEL(true, false, weight_ptr, 0, true);
              } else {
                RMS_NORM_DEFAULT_KERNEL(true, false, weight_ptr, 0, false);
              }
            }
          }
        });
      };
      if (has_weight && weight->scalar_type() == at::ScalarType::Float) {
        dispatch_weight(float{});
      } else {
        dispatch_weight(scalar_t{});
      }
    });
  });
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width, has_weight, zero_centered)                          \
  VLLM_DISPATCH_FLOATING_TYPES(                                                              \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {                                \
        if (has_weight) {                                                                    \
          if (weight->scalar_type() == at::ScalarType::Float) {                              \
            vllm::fused_add_rms_norm_kernel<scalar_t, float, width, true, zero_centered>     \
                <<<grid, block, 0, stream>>>(                                                \
                    input.data_ptr<scalar_t>(), input_stride,                                \
                    residual.data_ptr<scalar_t>(),                                           \
                    weight->data_ptr<float>(),                                               \
                    epsilon, num_tokens, hidden_size, residual_stride);                      \
          } else {                                                                           \
            vllm::fused_add_rms_norm_kernel<scalar_t, scalar_t, width, true, zero_centered>  \
                <<<grid, block, 0, stream>>>(                                                \
                    input.data_ptr<scalar_t>(), input_stride,                                \
                    residual.data_ptr<scalar_t>(),                                           \
                    weight->data_ptr<scalar_t>(),                                            \
                    epsilon, num_tokens, hidden_size, residual_stride);                      \
          }                                                                                  \
        } else {                                                                             \
          vllm::fused_add_rms_norm_kernel<scalar_t, scalar_t, width, false, zero_centered>   \
              <<<grid, block, 0, stream>>>(                                                  \
                  input.data_ptr<scalar_t>(), input_stride,                                  \
                  residual.data_ptr<scalar_t>(),                                             \
                  nullptr,                                                                   \
                  epsilon, num_tokens, hidden_size, residual_stride);                        \
        }                                                                                    \
      })

template <typename T>
static __device__ __forceinline__ T float_to_dstT(float value) {
  return static_cast<T>(value);
}

template <>
static __device__ __forceinline__ maca_bfloat16 float_to_dstT(float value) {
  return __float2bfloat16(value);
}

template <> static __device__ __forceinline__ half float_to_dstT(float value) {
  return __float2half(value);
}

template <int N> __device__ __forceinline__ void copy(void *src, void *dst) {
  int8_t *ptr_src = (int8_t *)src;
  int8_t *ptr_dst = (int8_t *)dst;
#pragma unroll N
  for (int i = 0; i < N; i++) {
    ptr_dst[i] = ptr_src[i];
  }
}

template <> __device__ __forceinline__ void copy<32>(void *src, void *dst) {
  const float4 *ptr_src = (const float4 *)src;
  float4 *ptr_dst = (float4 *)dst;
  ptr_dst[0] = ptr_src[0];
  ptr_dst[1] = ptr_src[1];
}

template <> __device__ __forceinline__ void copy<16>(void *src, void *dst) {
  float4 *ptr_src = (float4 *)src;
  float4 *ptr_dst = (float4 *)dst;
  *ptr_dst = *ptr_src;
}

template <> __device__ __forceinline__ void copy<8>(void *src, void *dst) {
  float2 *ptr_src = (float2 *)src;
  float2 *ptr_dst = (float2 *)dst;
  *ptr_dst = *ptr_src;
}

template <> __device__ __forceinline__ void copy<4>(void *src, void *dst) {
  float *ptr_src = (float *)src;
  float *ptr_dst = (float *)dst;
  *ptr_dst = *ptr_src;
}

template <> __device__ __forceinline__ void copy<2>(void *src, void *dst) {
  half *ptr_src = (half *)src;
  half *ptr_dst = (half *)dst;
  *ptr_dst = *ptr_src;
}

template <> __device__ __forceinline__ void copy<1>(void *src, void *dst) {
  int8_t *ptr_src = (int8_t *)src;
  int8_t *ptr_dst = (int8_t *)dst;
  *ptr_dst = *ptr_src;
}

// Storage alignment must not change the fp32 compatibility reduction tree.
template <typename T, int N>
__device__ __forceinline__ vllm::vec_n_t<T, N>
fused_norm_load(const T* ptr) {
    using Vec = vllm::vec_n_t<T, N>;
    if (reinterpret_cast<std::uintptr_t>(ptr) % alignof(Vec) == 0)
        return *reinterpret_cast<const Vec*>(ptr);
    Vec value;
#pragma unroll
    for (int j = 0; j < N; ++j) value.val[j] = ptr[j];
    return value;
}

template <typename T, int N>
__device__ __forceinline__ void fused_norm_store(
    T* ptr, const vllm::vec_n_t<T, N>& value) {
    using Vec = vllm::vec_n_t<T, N>;
    if (reinterpret_cast<std::uintptr_t>(ptr) % alignof(Vec) == 0) {
        *reinterpret_cast<Vec*>(ptr) = value;
    } else {
#pragma unroll
        for (int j = 0; j < N; ++j) ptr[j] = value.val[j];
    }
}

#include "fused_add_rms_norm_extended.cuh"

template <uint32_t VEC_SIZE,
          uint32_t NUM_REG,
          typename T,
          typename S,
          int NUM_THREADS,
          bool HasWeight,
          bool ZERO_CENTERED>
__global__ void FusedAddRMSNormKernelOpt(
    T *__restrict__ input,
    T *__restrict__ residual,
    S *__restrict__ weight,
    const uint32_t d,
    const uint32_t stride_input,
    const uint32_t stride_residual,
    float weight_bias,
    float eps)
{
    constexpr bool mixed_fp32 = sizeof(T) == 2 && std::is_same_v<S, float>;
    float rms = 0;

    T *ptr_input = input + blockIdx.x * stride_input;
    T *ptr_residual = residual + blockIdx.x * stride_residual;

    float reg_input[NUM_REG][VEC_SIZE];

    // sum of squares
    float ss = 0.0f;

    uint32_t tid = threadIdx.x * VEC_SIZE;
    uint32_t block_stride = NUM_THREADS * VEC_SIZE;
    uint32_t k = 0;

    for (uint32_t i = tid; i < d; i += block_stride) {
        T local[VEC_SIZE];
        if constexpr (mixed_fp32) {
            const auto v = fused_norm_load<T, VEC_SIZE>(ptr_input + i);
#pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) local[j] = v.val[j];
        } else {
            copy<sizeof(T) * VEC_SIZE>((void *)(ptr_input + i), (void *)local);
        }

        T reg_residual[VEC_SIZE];
        if constexpr (mixed_fp32) {
            const auto v = fused_norm_load<T, VEC_SIZE>(ptr_residual + i);
#pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) reg_residual[j] = v.val[j];
        } else {
            copy<sizeof(T) * VEC_SIZE>(
                (void *)(ptr_residual + i), (void *)reg_residual);
        }

#pragma unroll VEC_SIZE
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
            float x = static_cast<float>(local[j]);
            x += static_cast<float>(reg_residual[j]);
            reg_residual[j] = float_to_dstT<T>(x);
            ss += x * x;
            reg_input[k][j] = x;
        }

        if constexpr (mixed_fp32) {
            vllm::vec_n_t<T, VEC_SIZE> v;
#pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) v.val[j] = reg_residual[j];
            fused_norm_store<T, VEC_SIZE>(ptr_residual + i, v);
        } else {
            copy<sizeof(T) * VEC_SIZE>(
                (void *)reg_residual, (void *)(ptr_residual + i));
        }

        k++;
    }

    constexpr int sm_size = NUM_THREADS >> 4;
    constexpr int sm_size2 = sm_size / 2;

    __shared__ float sm_sum[sm_size];

    if constexpr (sm_size == 32) {

        for (int i = 8; i > 0; i >>= 1) {
            ss += __shfl_down_sync_16(0xffffffffffffffff, ss, i);
        }

        int lane_id = threadIdx.x & 15;
        int group_id = threadIdx.x >> 4;

        if (lane_id == 0) {
            sm_sum[group_id] = ss;
        }

        __syncthreads();

        __shared__ float sm_sum2[sm_size >> 4];

        if (threadIdx.x < sm_size) {

            float data = sm_sum[threadIdx.x];

            for (int i = 8; i >= 1; i >>= 1) {
                data += __shfl_down_sync_16(0xffffffffffffffff, data, i);
            }

            if (lane_id == 0) {
                sm_sum2[group_id] = data;
            }
        }

        __syncthreads();

        ss = sm_sum2[0] + sm_sum2[1];

    } else if constexpr (sm_size == 16) {

        for (int i = 8; i > 0; i >>= 1) {
            ss += __shfl_down_sync_16(0xffffffffffffffff, ss, i);
        }

        int lane_id = threadIdx.x & 15;
        int group_id = threadIdx.x >> 4;

        if (lane_id == 0) {
            sm_sum[group_id] = ss;
        }

        __syncthreads();

        if (threadIdx.x < sm_size) {

            float data = sm_sum[threadIdx.x];

            for (int i = 8; i >= 1; i >>= 1) {
                data += __shfl_down_sync_16(0xffffffffffffffff, data, i);
            }

            if (threadIdx.x == 0) {
                sm_sum[0] = data;
            }
        }

        __syncthreads();

        ss = sm_sum[0];

    } else if constexpr (sm_size == 8) {

        for (int i = 8; i > 0; i >>= 1) {
            ss += __shfl_down_sync_16(0xffffffffffffffff, ss, i);
        }

        int lane_id = threadIdx.x & 15;
        int group_id = threadIdx.x >> 4;

        if (lane_id == 0) {
            sm_sum[group_id] = ss;
        }

        __syncthreads();

        if (threadIdx.x < sm_size) {

            float data = sm_sum[threadIdx.x];

            for (int i = 4; i >= 1; i >>= 1) {
                data += __shfl_down_sync_16(0xffffffffffffffff, data, i);
            }

            if (threadIdx.x == 0) {
                sm_sum[0] = data;
            }
        }

        __syncthreads();

        ss = sm_sum[0];

    } else if constexpr (sm_size == 4) {

        for (int i = 8; i > 0; i >>= 1) {
            ss += __shfl_down_sync_16(0xffffffffffffffff, ss, i);
        }

        int lane_id = threadIdx.x & 15;
        int group_id = threadIdx.x >> 4;

        if (lane_id == 0) {
            sm_sum[group_id] = ss;
        }

        __syncthreads();

        if (threadIdx.x < sm_size) {

            float data = sm_sum[threadIdx.x];

            for (int i = 2; i >= 1; i >>= 1) {
                data += __shfl_down_sync_16(0xffffffffffffffff, data, i);
            }

            if (threadIdx.x == 0) {
                sm_sum[0] = data;
            }
        }

        __syncthreads();

        ss = sm_sum[0];
    }

    __shared__ float s_rms;

    if (threadIdx.x == 0) {
        s_rms = rsqrtf(ss / (float)d + eps);
    }

    __syncthreads();

    rms = s_rms;

    S const *ptr_weight = weight;

    k = 0;

    for (uint32_t i = tid; i < d; i += block_stride) {

        T reg_dst[VEC_SIZE];

        if constexpr (HasWeight) {

            S local_weight[VEC_SIZE];

            if constexpr (mixed_fp32) {
                const auto w = fused_norm_load<S, VEC_SIZE>(ptr_weight + i);
#pragma unroll
                for (int j = 0; j < VEC_SIZE; ++j) local_weight[j] = w.val[j];
            } else {
                copy<sizeof(S) * VEC_SIZE>(
                    (void *)(ptr_weight + i), (void *)local_weight);
            }

#pragma unroll VEC_SIZE
            for (uint32_t j = 0; j < VEC_SIZE; j++) {
                reg_dst[j] = float_to_dstT<T>(
                    reg_input[k][j] * rms * vllm::rms_weight<ZERO_CENTERED, S>(local_weight[j]));
            }

        } else {

#pragma unroll VEC_SIZE
            for (uint32_t j = 0; j < VEC_SIZE; j++) {
                reg_dst[j] =
                    float_to_dstT<T>(reg_input[k][j] * rms);
            }
        }

        k++;

        if constexpr (mixed_fp32) {
            vllm::vec_n_t<T, VEC_SIZE> v;
#pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) v.val[j] = reg_dst[j];
            fused_norm_store<T, VEC_SIZE>(ptr_input + i, v);
        } else {
            copy<VEC_SIZE * sizeof(T)>((void *)reg_dst, (void *)(ptr_input + i));
        }
    }
}

template <uint32_t VEC_SIZE,
          uint32_t NUM_REG,
          typename T,
          typename S,
          int NUM_THREADS,
          bool HasWeight,
          bool ZERO_CENTERED,
          bool ALIGNED_MIXED = false>
__global__ __launch_bounds__(NUM_THREADS) void FusedAddRMSNormKernelMultiRow(
    T *__restrict__ input,
    T *__restrict__ residual,
    const S *__restrict__ weight,
    const uint32_t d,
    const uint32_t stride_input,
    const uint32_t stride_residual,
    const uint32_t batch_size,
    float eps)
{
    constexpr bool mixed_fp32 = sizeof(T) == 2 && std::is_same_v<S, float>;
    // Same-dtype instantiations retain their original intermediate rounding.
    // Mixed weights require the unrounded fp32 sum for both variance and output.
    using ResidualT = std::conditional_t<mixed_fp32, float, T>;
    using ResidualVec = vllm::vec_n_t<ResidualT, VEC_SIZE>;
    constexpr int WARP = 64;
    constexpr int NW = NUM_THREADS / WARP;
    using Vec = vllm::vec_n_t<T, VEC_SIZE>;
    using WeightVec = vllm::vec_n_t<S, VEC_SIZE>;

    const int tid = threadIdx.x;
    const float inv_d = 1.0f / static_cast<float>(d);

    float wreg[HasWeight ? NUM_REG : 1][VEC_SIZE];
    if constexpr (HasWeight) {
        const WeightVec *wv = reinterpret_cast<const WeightVec *>(weight);
#pragma unroll
        for (int k = 0; k < NUM_REG; ++k) {
            const WeightVec w = mixed_fp32 && !ALIGNED_MIXED
                ? fused_norm_load<S, VEC_SIZE>(weight + (tid + k * NUM_THREADS) * VEC_SIZE)
                : wv[tid + k * NUM_THREADS];
#pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j)
                wreg[k][j] = vllm::rms_weight<ZERO_CENTERED, S>(w.val[j]);
        }
    }

    __shared__ float red[2][NW > 1 ? NW : 1];
    int parity = 0;

    for (uint32_t row = blockIdx.x; row < batch_size; row += gridDim.x) {
        Vec *in_row  = reinterpret_cast<Vec *>(input + static_cast<int64_t>(row) * stride_input);
        Vec *res_row = reinterpret_cast<Vec *>(residual + static_cast<int64_t>(row) * stride_residual);

        Vec a[NUM_REG];
        ResidualVec r[NUM_REG];
#pragma unroll
        for (int k = 0; k < NUM_REG; ++k) {
            if constexpr (mixed_fp32) {
                const int64_t offset = static_cast<int64_t>(row) * stride_input +
                    (tid + k * NUM_THREADS) * VEC_SIZE;
                const int64_t res_offset = static_cast<int64_t>(row) * stride_residual +
                    (tid + k * NUM_THREADS) * VEC_SIZE;
                Vec rv;
                if constexpr (ALIGNED_MIXED) {
                    a[k] = in_row[tid + k * NUM_THREADS];
                    rv = res_row[tid + k * NUM_THREADS];
                } else {
                    a[k] = fused_norm_load<T, VEC_SIZE>(input + offset);
                    rv = fused_norm_load<T, VEC_SIZE>(residual + res_offset);
                }
#pragma unroll
                for (int j = 0; j < VEC_SIZE; ++j)
                    r[k].val[j] = static_cast<float>(rv.val[j]);
            } else {
                a[k] = in_row[tid + k * NUM_THREADS];
                r[k] = res_row[tid + k * NUM_THREADS];
            }
        }

        float ss = 0.f;
#pragma unroll
        for (int k = 0; k < NUM_REG; ++k)
#pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                const ResidualT z = float_to_dstT<ResidualT>(static_cast<float>(a[k].val[j]) +
                                             static_cast<float>(r[k].val[j]));
                r[k].val[j] = z;
                const float zf = static_cast<float>(z);
                ss += zf * zf;
            }

#pragma unroll
        for (int off = WARP >> 1; off > 0; off >>= 1)
            ss += __shfl_xor_sync(0xffffffffffffffffULL, ss, off);
        if constexpr (NW > 1) {
            if ((tid & (WARP - 1)) == 0) red[parity][tid / WARP] = ss;
            __syncthreads();
            ss = 0.f;
#pragma unroll
            for (int k = 0; k < NW; ++k) ss += red[parity][k];
            parity ^= 1;
        }
        const float rms = __builtin_mxc_rcpf(sqrtf(ss * inv_d + eps));

#pragma unroll
        for (int k = 0; k < NUM_REG; ++k) {
            Vec dst;
#pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                float v;
                if constexpr (mixed_fp32) {
                    // C600 fp32 oracle lowers both multiplies to pk_fma_f32
                    // with a +0 addend. Scalar mul preserves -0 instead, so
                    // explicitly match each operation before the final cast.
                    v = __fmaf_rn(static_cast<float>(r[k].val[j]), rms, 0.0f);
                    if constexpr (HasWeight)
                        v = __fmaf_rn(v, wreg[k][j], 0.0f);
                } else {
                    v = static_cast<float>(r[k].val[j]) * rms;
                    if constexpr (HasWeight) v *= wreg[k][j];
                }
                dst.val[j] = float_to_dstT<T>(v);
            }
            if constexpr (mixed_fp32) {
                Vec residual_dst;
#pragma unroll
                for (int j = 0; j < VEC_SIZE; ++j)
                    residual_dst.val[j] = float_to_dstT<T>(r[k].val[j]);
                if constexpr (ALIGNED_MIXED) {
                    res_row[tid + k * NUM_THREADS] = residual_dst;
                    in_row[tid + k * NUM_THREADS] = dst;
                } else {
                    fused_norm_store<T, VEC_SIZE>(residual + static_cast<int64_t>(row) * stride_residual +
                        (tid + k * NUM_THREADS) * VEC_SIZE, residual_dst);
                    fused_norm_store<T, VEC_SIZE>(input + static_cast<int64_t>(row) * stride_input +
                        (tid + k * NUM_THREADS) * VEC_SIZE, dst);
                }
            } else {
                res_row[tid + k * NUM_THREADS] = r[k];
                in_row[tid + k * NUM_THREADS] = dst;
            }
        }
    }
}

// C600-U decode: paired 16B accesses, late weights and explicit +0 FMA.
// One physical thread owns two adjacent logical C4 threads. This makes
// activation accesses 16B wide while preserving each logical thread's sum order.
template <bool ZERO_CENTERED>
__global__ __launch_bounds__(128) void FusedAddRMSNormKernelPairedScalarFmaC600U(
    maca_bfloat16* __restrict__ input,
    maca_bfloat16* __restrict__ residual,
    const float* __restrict__ weight,
    const uint32_t d,
    float eps)
{
    using WideVec = vllm::vec_n_t<maca_bfloat16, 8>;
    using FloatVec = vllm::vec_n_t<float, 4>;
    const int tid = threadIdx.x;
    const float inv_d = 1.0f / static_cast<float>(d);
    const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * d;
    auto* in_row = reinterpret_cast<WideVec*>(input + row_offset);
    auto* res_row = reinterpret_cast<WideVec*>(residual + row_offset);
    const auto* weights = reinterpret_cast<const FloatVec*>(weight);

    FloatVec r0[4], r1[4];
    float ss0 = 0.f;
    float ss1 = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const WideVec a = in_row[tid + k * 128];
        const WideVec r = res_row[tid + k * 128];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float z0 = static_cast<float>(a.val[j]) + static_cast<float>(r.val[j]);
            const float z1 = static_cast<float>(a.val[j + 4]) + static_cast<float>(r.val[j + 4]);
            r0[k].val[j] = z0;
            r1[k].val[j] = z1;
            ss0 += z0 * z0;
            ss1 += z1 * z1;
        }
    }

    // Logical lanes 2*tid and 2*tid+1 retain independent accumulators.
    // Physical xor 16..1 reproduces logical xor 32..2 within each virtual
    // 64-lane warp. The final register add reproduces logical xor 1 at its
    // representative lane, including operand order.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss0 += __shfl_xor_sync(0xffffffffffffffffULL, ss0, off, 32);
        ss1 += __shfl_xor_sync(0xffffffffffffffffULL, ss1, off, 32);
    }
    __shared__ float red[4];
    if ((tid & 31) == 0) red[tid >> 5] = ss0 + ss1;
    __syncthreads();
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) ss += red[k];
    const float rms = __builtin_mxc_rcpf(sqrtf(ss * inv_d + eps));

#pragma unroll
    for (int k = 0; k < 4; ++k) {
        FloatVec w0 = weights[2 * tid + k * 256];
        FloatVec w1 = weights[2 * tid + 1 + k * 256];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            w0.val[j] = vllm::rms_weight<ZERO_CENTERED, float>(w0.val[j]);
            w1.val[j] = vllm::rms_weight<ZERO_CENTERED, float>(w1.val[j]);
        }
        WideVec dst;
        WideVec residual_dst;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float v0 = __builtin_fmaf(r0[k].val[j], rms, 0.0f);
            v0 = __builtin_fmaf(v0, w0.val[j], 0.0f);
            float v1 = __builtin_fmaf(r1[k].val[j], rms, 0.0f);
            v1 = __builtin_fmaf(v1, w1.val[j], 0.0f);
            dst.val[j] = float_to_dstT<maca_bfloat16>(v0);
            dst.val[j + 4] = float_to_dstT<maca_bfloat16>(v1);
            residual_dst.val[j] = float_to_dstT<maca_bfloat16>(r0[k].val[j]);
            residual_dst.val[j + 4] = float_to_dstT<maca_bfloat16>(r1[k].val[j]);
        }
        res_row[tid + k * 128] = residual_dst;
        in_row[tid + k * 128] = dst;
    }
}

// C600-U prefill: specialize the H4096 persistent launch address contract.
// One physical thread owns two adjacent logical C4 threads. This makes
// activation accesses 16B wide while preserving each logical thread's sum order.
template <bool ZERO_CENTERED>
__global__ __launch_bounds__(128) void FusedAddRMSNormKernelPairedLaunchContractC600U(
    maca_bfloat16* __restrict__ input,
    maca_bfloat16* __restrict__ residual,
    const float* __restrict__ weight,
    const uint32_t d,
    const uint32_t batch_size,
    const uint32_t row_step,
    float eps)
{
    using WideVec = vllm::vec_n_t<maca_bfloat16, 8>;
    using FloatVec = vllm::vec_n_t<float, 4>;
    const int tid = threadIdx.x;
    const float inv_d = 1.0f / static_cast<float>(d);
    const auto* weights = reinterpret_cast<const FloatVec*>(weight);

    FloatVec w0[4], w1[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        w0[k] = weights[2 * tid + k * 256];
        w1[k] = weights[2 * tid + 1 + k * 256];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            w0[k].val[j] = vllm::rms_weight<ZERO_CENTERED, float>(w0[k].val[j]);
            w1[k].val[j] = vllm::rms_weight<ZERO_CENTERED, float>(w1[k].val[j]);
        }
    }

    __shared__ float red[2][4];
    int parity = 0;
    for (uint32_t row = blockIdx.x; row < batch_size; row += row_step) {
        const int64_t row_offset = static_cast<int64_t>(row) * 4096;
        auto* in_row = reinterpret_cast<WideVec*>(input + row_offset);
        auto* res_row = reinterpret_cast<WideVec*>(residual + row_offset);
        FloatVec r0[4], r1[4];
        float ss0 = 0.f;
        float ss1 = 0.f;
    #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const WideVec a = in_row[tid + k * 128];
            const WideVec r = res_row[tid + k * 128];
    #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float z0 = static_cast<float>(a.val[j]) + static_cast<float>(r.val[j]);
                const float z1 = static_cast<float>(a.val[j + 4]) + static_cast<float>(r.val[j + 4]);
                r0[k].val[j] = z0;
                r1[k].val[j] = z1;
                ss0 += z0 * z0;
                ss1 += z1 * z1;
            }
        }

        // Logical lanes 2*tid and 2*tid+1 retain independent accumulators.
        // Physical xor 16..1 reproduces logical xor 32..2 within each virtual
        // 64-lane warp. The final register add reproduces logical xor 1 at its
        // representative lane, including operand order.
    #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            ss0 += __shfl_xor_sync(0xffffffffffffffffULL, ss0, off, 32);
            ss1 += __shfl_xor_sync(0xffffffffffffffffULL, ss1, off, 32);
        }
        if ((tid & 31) == 0) red[parity][tid >> 5] = ss0 + ss1;
        __syncthreads();
        float ss = 0.f;
    #pragma unroll
        for (int k = 0; k < 4; ++k) ss += red[parity][k];
        parity ^= 1;
        const float rms = __builtin_mxc_rcpf(sqrtf(ss * inv_d + eps));

    #pragma unroll
        for (int k = 0; k < 4; ++k) {
            WideVec dst;
            WideVec residual_dst;
    #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float v0 = __builtin_fmaf(r0[k].val[j], rms, 0.0f);
                v0 = __builtin_fmaf(v0, w0[k].val[j], 0.0f);
                float v1 = __builtin_fmaf(r1[k].val[j], rms, 0.0f);
                v1 = __builtin_fmaf(v1, w1[k].val[j], 0.0f);
                dst.val[j] = float_to_dstT<maca_bfloat16>(v0);
                dst.val[j + 4] = float_to_dstT<maca_bfloat16>(v1);
                residual_dst.val[j] = float_to_dstT<maca_bfloat16>(r0[k].val[j]);
                residual_dst.val[j + 4] = float_to_dstT<maca_bfloat16>(r1[k].val[j]);
            }
            res_row[tid + k * 128] = residual_dst;
            in_row[tid + k * 128] = dst;
        }
    }
}

template<typename T, typename S, bool HasWeight, bool ZERO_CENTERED>
int launch_fused_add_rms_norm_opt(
    T* input,
    T* residual,
    S* weight,
    uint32_t batch_size,
    uint32_t d,
    uint32_t stride_input,
    uint32_t stride_residual,
    float eps = 1e-5,
    cudaStream_t stream = 0)
{
    dim3 nblks(batch_size);

    constexpr bool mixed_fp32 = sizeof(T) == 2 && std::is_same_v<S, float>;
    // Compatibility is defined by the fp32 oracle, not the storage byte width.
    constexpr int N = mixed_fp32 ? 4 : 16 / sizeof(T);

    if ((d & (N - 1)) == 0) {
        int blocksize = 64;
        float weight_bias = 0.0f;

        if (d <= blocksize * N) {
            constexpr int NUM_THREADS = 64;
            FusedAddRMSNormKernelOpt<N, 1, T, S, NUM_THREADS, HasWeight, ZERO_CENTERED>
                <<<nblks, NUM_THREADS, 0, stream>>>(
                    input,
                    residual,
                    weight,
                    d,
                    stride_input,
                    stride_residual,
                    weight_bias,
                    eps);
            return 0;
        } else if (d <= blocksize * 2 * N) {
            constexpr int NUM_THREADS = 128;
            FusedAddRMSNormKernelOpt<N, 1, T, S, NUM_THREADS, HasWeight, ZERO_CENTERED>
                <<<nblks, NUM_THREADS, 0, stream>>>(
                    input,
                    residual,
                    weight,
                    d,
                    stride_input,
                    stride_residual,
                    weight_bias,
                    eps);
            return 0;
        } else if (d <= blocksize * 4 * N) {
            constexpr int NUM_THREADS = 256;
            FusedAddRMSNormKernelOpt<N, 1, T, S, NUM_THREADS, HasWeight, ZERO_CENTERED>
                <<<nblks, NUM_THREADS, 0, stream>>>(
                    input,
                    residual,
                    weight,
                    d,
                    stride_input,
                    stride_residual,
                    weight_bias,
                    eps);
            return 0;
        } else if (d <= blocksize * 8 * N) {
            constexpr int NUM_THREADS = 512;
            FusedAddRMSNormKernelOpt<N, 1, T, S, NUM_THREADS, HasWeight, ZERO_CENTERED>
                <<<nblks, NUM_THREADS, 0, stream>>>(
                    input,
                    residual,
                    weight,
                    d,
                    stride_input,
                    stride_residual,
                    weight_bias,
                    eps);
            return 0;
        } else if (d <= blocksize * 16 * N) {
            constexpr int NUM_THREADS = 512;
            FusedAddRMSNormKernelOpt<N, 2, T, S, NUM_THREADS, HasWeight, ZERO_CENTERED>
                <<<nblks, NUM_THREADS, 0, stream>>>(
                    input,
                    residual,
                    weight,
                    d,
                    stride_input,
                    stride_residual,
                    weight_bias,
                    eps);
            return 0;
        }
    }

    return -1;
}

template<typename T, typename S, bool HasWeight, bool ZERO_CENTERED>
int launch_fused_add_rms_norm_multirow(
    T* input,
    T* residual,
    S* weight,
    uint32_t batch_size,
    uint32_t d,
    uint32_t stride_input,
    uint32_t stride_residual,
    int sm_count,
    float eps = 1e-5,
    cudaStream_t stream = 0)
{
    constexpr bool mixed_fp32 = sizeof(T) == 2 && std::is_same_v<S, float>;
    // Compatibility is defined by the fp32 oracle, not the storage byte width.
    constexpr int N = mixed_fp32 ? 4 : 16 / sizeof(T);
    constexpr int MAX_REG = mixed_fp32 ? 4 : (sizeof(T) == 2 ? 8 : 4);

    if (d % N != 0 || stride_input % N != 0 || stride_residual % N != 0) return -1;
    if constexpr (!mixed_fp32) {
        if (reinterpret_cast<std::uintptr_t>(input) % 16 != 0 ||
            reinterpret_cast<std::uintptr_t>(residual) % 16 != 0) return -1;
    }
    if (HasWeight && reinterpret_cast<std::uintptr_t>(weight) % (sizeof(S) * N) != 0) return -1;

    const int vec_count = d / N;
    int num_threads = 0;
    for (int b : {64, 128, 256, 512}) {
        if (vec_count % b == 0 && vec_count / b <= MAX_REG) { num_threads = b; break; }
    }
    if (num_threads == 0) return -1;
    const int num_reg = vec_count / num_threads;

    const int g = !mixed_fp32 && sizeof(T) == 2
                      ? std::min<int>(batch_size, sm_count * (2048 / num_threads))
                      : static_cast<int>(batch_size);
    dim3 nblks(std::max(1, g));

#define FUSED_ADD_RMS_NORM_MULTIROW(NT, NR)                                             \
    FusedAddRMSNormKernelMultiRow<N, NR, T, S, NT, HasWeight, ZERO_CENTERED>            \
        <<<nblks, NT, 0, stream>>>(input, residual, weight, d, stride_input,            \
                                   stride_residual, batch_size, eps);                   \
    return 0
#define LAUNCH_NUM_THREADS(NT)                                                          \
    switch (num_reg) {                                                                  \
        case 1: FUSED_ADD_RMS_NORM_MULTIROW(NT, 1);                                     \
        case 2: FUSED_ADD_RMS_NORM_MULTIROW(NT, 2);                                     \
        case 3: FUSED_ADD_RMS_NORM_MULTIROW(NT, 3);                                     \
        case 4: FUSED_ADD_RMS_NORM_MULTIROW(NT, 4);                                     \
        case 6: FUSED_ADD_RMS_NORM_MULTIROW(NT, 6);                                     \
        case 8: FUSED_ADD_RMS_NORM_MULTIROW(NT, 8);                                     \
        default: return -1;                                                             \
    }

    switch (num_threads) {
        case 64:  LAUNCH_NUM_THREADS(64)
        case 128: LAUNCH_NUM_THREADS(128)
        case 256: LAUNCH_NUM_THREADS(256)
        case 512: LAUNCH_NUM_THREADS(512)
    }
    return -1;
}

// Compatibility fallback: no full-tensor temporary and no intermediate T rounding.
template <typename T, bool ZERO_CENTERED>
__global__ void fused_add_rms_norm_mixed_fallback(
    T* __restrict__ input, T* __restrict__ residual,
    const float* __restrict__ weight, int hidden_size, float epsilon) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * hidden_size;
    float variance = 0.0f;
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float z = static_cast<float>(input[row + idx]);
        z += static_cast<float>(residual[row + idx]);
        variance += z * z;
    }
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    __shared__ float s_variance;
    variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);
    if (threadIdx.x == 0) s_variance = rsqrtf(variance / hidden_size + epsilon);
    __syncthreads();
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float z = static_cast<float>(input[row + idx]);
        z += static_cast<float>(residual[row + idx]);
        const float w = vllm::rms_weight<ZERO_CENTERED, float>(weight[idx]);
        const float out = z * s_variance * w;
        residual[row + idx] = float_to_dstT<T>(z);
        input[row + idx] = float_to_dstT<T>(out);
    }
}

void fused_add_rms_norm_out(
    torch::Tensor& input,
    torch::Tensor& residual,
    std::optional<torch::Tensor> weight,
    double epsilon, bool zero_centered,
    std::optional<torch::Tensor> out, bool input_inplace)
{
    if (!out && input_inplace) {
        // Preserve every legacy dispatch and rounding rule when out is absent.
        fused_add_rms_norm(input, residual, weight, epsilon, zero_centered);
    } else {
        fused_add_rms_norm_extended(input, residual, weight, epsilon, zero_centered,
                                   out, input_inplace);
    }
}

void fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    std::optional<torch::Tensor> weight,
    double epsilon, bool zero_centered)
{

    if (input.scalar_type() != residual.scalar_type()) {
        fused_add_rms_norm_extended(input, residual, weight, epsilon, zero_centered, std::nullopt);
        return;
    }
    const bool mixed_fp32 = weight.has_value() && weight->scalar_type() == at::ScalarType::Float &&
        (input.scalar_type() == at::ScalarType::BFloat16 || input.scalar_type() == at::ScalarType::Half);
    if (mixed_fp32) {
        TORCH_CHECK(input.is_cuda() && residual.is_cuda() && weight->is_cuda() &&
                    input.device() == residual.device() && input.device() == weight->device(),
                    "fused_add_rms_norm mixed dtype requires tensors on the same CUDA device");
        TORCH_CHECK(input.dim() == 2 && residual.sizes() == input.sizes(),
                    "fused_add_rms_norm mixed dtype requires matching [T, H] shapes");
        TORCH_CHECK(input.scalar_type() == residual.scalar_type(),
                    "fused_add_rms_norm residual dtype must match input");
        TORCH_CHECK(input.is_contiguous() && residual.is_contiguous(),
                    "fused_add_rms_norm mixed dtype requires contiguous input and residual");
        TORCH_CHECK(input.size(1) > 0 && input.size(1) <= INT_MAX && input.size(0) <= INT_MAX &&
                    input.numel() <= UINT_MAX,
                    "fused_add_rms_norm mixed dtype requires H > 0, dimensions <= INT_MAX and numel <= UINT_MAX");
        TORCH_CHECK(weight->dim() == 1 && weight->size(0) == input.size(1) && weight->is_contiguous(),
                    "fused_add_rms_norm mixed dtype requires contiguous weight[H]");
        // restrict kernels require disjoint buffers; reject before any write.
        const auto begin = reinterpret_cast<std::uintptr_t>(input.data_ptr());
        const auto res_begin = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
        const auto wt_begin = reinterpret_cast<std::uintptr_t>(weight->data_ptr());
        const auto bytes = input.numel() * input.element_size();
        const auto wt_bytes = weight->numel() * weight->element_size();
        TORCH_CHECK(bytes == 0 || ((begin + bytes <= res_begin || res_begin + bytes <= begin) &&
                    (begin + bytes <= wt_begin || wt_begin + wt_bytes <= begin) &&
                    (res_begin + bytes <= wt_begin || wt_begin + wt_bytes <= res_begin)),
                    "fused_add_rms_norm mixed dtype requires non-overlapping buffers");
        if (input.size(0) == 0) return;
    }
    TORCH_CHECK(!zero_centered || weight.has_value(),
                "rms_norm zero_centered requires weight");
    TORCH_CHECK(input.scalar_type() == residual.scalar_type());
    TORCH_CHECK(residual.stride(-1) == 1);

    if (weight.has_value()) {
        TORCH_CHECK(weight->scalar_type() == at::ScalarType::Float ||
                    weight->scalar_type() == input.scalar_type(),
                    "weight dtype must be float or the same as input dtype");
        TORCH_CHECK(weight->is_contiguous());
    }

    int hidden_size = input.size(-1);
    int64_t input_stride = input.stride(-2);
    int64_t residual_stride = residual.stride(-2);
    int num_tokens = input.numel() / hidden_size;

    dim3 grid(num_tokens);

    /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. In batch-invariant mode the block size must
     not depend on num_tokens, otherwise the same token would use a different
     reduction width (and thus a different floating-point summation order)
     across batches; lock it to 1024 to keep results bit-exact. */
    const bool batch_invariant_launch = vllm::vllm_is_batch_invariant();
    const int max_block_size =
        batch_invariant_launch ? 1024 : ((num_tokens < 256) ? 1024 : 256);
    dim3 block(std::min(hidden_size, max_block_size));

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
    auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());

    int status = -1;
    cudaDeviceProp prop = GetDeviceProp();
    bool is_c600 = prop.sharedMemPerBlock > 65536;

    #define LAUNCH_FUSED_ADD_RMS_NORM_OPT(T, S, HAS_WEIGHT, ZERO_CENTERED)                  \
      launch_fused_add_rms_norm_opt<T, S, HAS_WEIGHT, ZERO_CENTERED>(                       \
          static_cast<T*>(input.data_ptr()),                                                \
          static_cast<T*>(residual.data_ptr()),                                             \
          (HAS_WEIGHT) ? static_cast<S*>(weight->data_ptr()) : nullptr,                     \
          num_tokens,                                                                       \
          hidden_size,                                                                      \
          input_stride,                                                                     \
          residual_stride,                                                                  \
          epsilon,                                                                          \
          stream)

    #define LAUNCH_FUSED_ADD_RMS_NORM_MULTIROW(T, S, HAS_WEIGHT, ZERO_CENTERED)             \
      launch_fused_add_rms_norm_multirow<T, S, HAS_WEIGHT, ZERO_CENTERED>(                  \
          static_cast<T*>(input.data_ptr()),                                                \
          static_cast<T*>(residual.data_ptr()),                                             \
          (HAS_WEIGHT) ? static_cast<S*>(weight->data_ptr()) : nullptr,                     \
          num_tokens,                                                                       \
          hidden_size,                                                                      \
          input_stride,                                                                     \
          residual_stride,                                                                  \
          prop.multiProcessorCount,                                                         \
          epsilon,                                                                          \
          stream)

    bool use_bf16 = (mixed_fp32 || (hidden_size % 8 == 0 && (input_stride & 7) == 0)) &&
                    (input.dtype() == at::ScalarType::BFloat16);
    
    // Mixed H4096 specialization is selected independently of device model.
    // The mixed entry has already checked contiguous, disjoint [T,H] buffers.
    // H=4096 preserves 8B activation and 16B weight alignment for every access.
    if (mixed_fp32 && use_bf16 && hidden_size == 4096 &&
        input_stride == 4096 && residual_stride == 4096 &&
        inp_ptr % 8 == 0 && res_ptr % 8 == 0 &&
        reinterpret_cast<std::uintptr_t>(weight->data_ptr()) % 16 == 0) {
        auto* x = static_cast<maca_bfloat16*>(input.data_ptr());
        auto* r = static_cast<maca_bfloat16*>(residual.data_ptr());
        auto* w = static_cast<float*>(weight->data_ptr());
        // Paired single-row decode and persistent prefill weight reuse.
        // Keep dtype, layout, alignment and workload-range guards.
        if (inp_ptr % 16 == 0 && res_ptr % 16 == 0) {
            if (num_tokens >= 128 && num_tokens <= 256) {
                if (zero_centered) {
                    FusedAddRMSNormKernelPairedScalarFmaC600U<true>
                        <<<grid, 128, 0, stream>>>(x, r, w, hidden_size, epsilon);
                } else {
                    FusedAddRMSNormKernelPairedScalarFmaC600U<false>
                        <<<grid, 128, 0, stream>>>(x, r, w, hidden_size, epsilon);
                }
                return;
            }
            const uint32_t sm = at::cuda::getDeviceProperties(input.get_device())->multiProcessorCount;
            const uint32_t cap = sm * (2048 / 128);
            if (num_tokens >= 2 * cap) {
                if (zero_centered) {
                    FusedAddRMSNormKernelPairedLaunchContractC600U<true>
                        <<<cap, 128, 0, stream>>>(x, r, w, hidden_size, num_tokens, cap, epsilon);
                } else {
                    FusedAddRMSNormKernelPairedLaunchContractC600U<false>
                        <<<cap, 128, 0, stream>>>(x, r, w, hidden_size, num_tokens, cap, epsilon);
                }
                return;
            }
        }
        if (zero_centered) {
            FusedAddRMSNormKernelMultiRow<4, 4, maca_bfloat16, float, 256, true, true, true>
                <<<grid, 256, 0, stream>>>(x, r, w, hidden_size, input_stride,
                                           residual_stride, num_tokens, epsilon);
        } else {
            FusedAddRMSNormKernelMultiRow<4, 4, maca_bfloat16, float, 256, true, false, true>
                <<<grid, 256, 0, stream>>>(x, r, w, hidden_size, input_stride,
                                           residual_stride, num_tokens, epsilon);
        }
        return;
    }

    if (is_c600) {
      if (weight.has_value()) {
        bool weight_fp32 = (weight->dtype() == at::ScalarType::Float);
        if (use_bf16) {
            if (weight_fp32) {
                status = zero_centered ? LAUNCH_FUSED_ADD_RMS_NORM_MULTIROW(maca_bfloat16, float, true, true)
                                       : LAUNCH_FUSED_ADD_RMS_NORM_MULTIROW(maca_bfloat16, float, true, false);
            } else {
                status = zero_centered ? LAUNCH_FUSED_ADD_RMS_NORM_MULTIROW(maca_bfloat16, maca_bfloat16, true, true)
                                       : LAUNCH_FUSED_ADD_RMS_NORM_MULTIROW(maca_bfloat16, maca_bfloat16, true, false);
            }
        } else if (input.dtype() == at::ScalarType::Half) {
            if (weight_fp32) {
                status = zero_centered ? LAUNCH_FUSED_ADD_RMS_NORM_MULTIROW(half, float, true, true)
                                       : LAUNCH_FUSED_ADD_RMS_NORM_MULTIROW(half, float, true, false);
            } else {
                status = zero_centered ? LAUNCH_FUSED_ADD_RMS_NORM_MULTIROW(half, half, true, true)
                                       : LAUNCH_FUSED_ADD_RMS_NORM_MULTIROW(half, half, true, false);
            }
        } else if (input.dtype() == at::ScalarType::Float) {
            status = zero_centered ? LAUNCH_FUSED_ADD_RMS_NORM_MULTIROW(float, float, true, true)
                                   : LAUNCH_FUSED_ADD_RMS_NORM_MULTIROW(float, float, true, false);
        }
      } else {
          if (use_bf16) {
              status = LAUNCH_FUSED_ADD_RMS_NORM_MULTIROW(maca_bfloat16, maca_bfloat16, false, false);
          } else if (input.dtype() == at::ScalarType::Half) {
              status = LAUNCH_FUSED_ADD_RMS_NORM_MULTIROW(half, half, false, false);
          } else if (input.dtype() == at::ScalarType::Float) {
              status = LAUNCH_FUSED_ADD_RMS_NORM_MULTIROW(float, float, false, false);
          }
      }
    } else {
      if (weight.has_value()) {
          bool weight_fp32 = (weight->dtype() == at::ScalarType::Float);
          TORCH_CHECK(weight_fp32 || weight->dtype() == input.dtype(),
                      "weight dtype must be float or the same as input dtype");
          if (use_bf16) {
              if (weight_fp32) {
                  status = zero_centered ? LAUNCH_FUSED_ADD_RMS_NORM_OPT(maca_bfloat16, float, true, true)
                                        : LAUNCH_FUSED_ADD_RMS_NORM_OPT(maca_bfloat16, float, true, false);
              } else {
                  status = zero_centered ? LAUNCH_FUSED_ADD_RMS_NORM_OPT(maca_bfloat16, maca_bfloat16, true, true)
                                        : LAUNCH_FUSED_ADD_RMS_NORM_OPT(maca_bfloat16, maca_bfloat16, true, false);
              }
          } else if (input.dtype() == at::ScalarType::Half) {
              if (weight_fp32) {
                  status = zero_centered ? LAUNCH_FUSED_ADD_RMS_NORM_OPT(half, float, true, true)
                                        : LAUNCH_FUSED_ADD_RMS_NORM_OPT(half, float, true, false);
              } else {
                  status = zero_centered ? LAUNCH_FUSED_ADD_RMS_NORM_OPT(half, half, true, true)
                                        : LAUNCH_FUSED_ADD_RMS_NORM_OPT(half, half, true, false);
              }
          } else if (input.dtype() == at::ScalarType::Float) {
              status = zero_centered ? LAUNCH_FUSED_ADD_RMS_NORM_OPT(float, float, true, true)
                                    : LAUNCH_FUSED_ADD_RMS_NORM_OPT(float, float, true, false);
          }
      } else {
          if (use_bf16) {
              status = LAUNCH_FUSED_ADD_RMS_NORM_OPT(maca_bfloat16, maca_bfloat16, false, false);
          } else if (input.dtype() == at::ScalarType::Half) {
              status = LAUNCH_FUSED_ADD_RMS_NORM_OPT(half, half, false, false);
          } else if (input.dtype() == at::ScalarType::Float) {
              status = LAUNCH_FUSED_ADD_RMS_NORM_OPT(float, float, false, false);
          }
      }
    }

    if (status == 0) {
        return;
    }

    if (mixed_fp32) {
        VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_add_rms_norm_mixed", [&] {
            if (zero_centered) {
                fused_add_rms_norm_mixed_fallback<scalar_t, true><<<grid, block, 0, stream>>>(
                    input.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(),
                    weight->data_ptr<float>(), hidden_size, epsilon);
            } else {
                fused_add_rms_norm_mixed_fallback<scalar_t, false><<<grid, block, 0, stream>>>(
                    input.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(),
                    weight->data_ptr<float>(), hidden_size, epsilon);
            }
        });
        return;
    }

    constexpr int vector_width = 8;
    constexpr int req_alignment_bytes = vector_width * 2;

    bool offsets_are_multiple_of_vector_width =
      hidden_size % vector_width == 0 && input_stride % vector_width == 0 &&
      residual_stride % vector_width == 0;
    if (weight.has_value()) {
        auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight->data_ptr());
        const int wt_req_alignment_bytes = vector_width * weight->element_size();
        bool ptrs_are_aligned =
            inp_ptr % req_alignment_bytes == 0 &&
            res_ptr % req_alignment_bytes == 0 &&
            wt_ptr % wt_req_alignment_bytes == 0;
        if (ptrs_are_aligned &&
            offsets_are_multiple_of_vector_width &&
            !batch_invariant_launch) {
            if (zero_centered) {
              LAUNCH_FUSED_ADD_RMS_NORM(8, true, true);
            } else {
              LAUNCH_FUSED_ADD_RMS_NORM(8, true, false);
            }
        } else {
            if (zero_centered) {
              LAUNCH_FUSED_ADD_RMS_NORM(0, true, true);
            } else {
              LAUNCH_FUSED_ADD_RMS_NORM(0, true, false);
            }
        }
    } else {
        bool ptrs_are_aligned =
            inp_ptr % req_alignment_bytes == 0 &&
            res_ptr % req_alignment_bytes == 0;
        if (ptrs_are_aligned &&
            offsets_are_multiple_of_vector_width &&
            !batch_invariant_launch) {
            LAUNCH_FUSED_ADD_RMS_NORM(8, false, false);
        } else {
            LAUNCH_FUSED_ADD_RMS_NORM(0, false, false);
        }
    }
}
