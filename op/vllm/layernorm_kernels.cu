#include "core/batch_invariant.hpp"
#include "cub_helpers.h"
#include "dispatch_utils.h"
#include "quantization/vectorization_utils.cuh"
#include "type_convert.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <torch/cuda.h>

#include <cub/cub.cuh>

namespace vllm {
template <typename scalar_t, int NUM_DIMS>
__device__ __forceinline__ const scalar_t *
rms_input_row(const scalar_t *input, int64_t input_stride_d2,
              int64_t input_stride_d3, int64_t input_stride_d4,
              int64_t input_shape_d2, int64_t input_shape_d3) {
  if constexpr (NUM_DIMS == 2) {
    return input + blockIdx.x * input_stride_d2;
  } else if constexpr (NUM_DIMS == 3) {
    const int batch_idx = blockIdx.x / input_shape_d2;
    const int head_idx = blockIdx.x % input_shape_d2;
    return input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
  } else {
    const int batch_idx = blockIdx.x / (input_shape_d3 * input_shape_d2);
    const int remaining = blockIdx.x % (input_shape_d3 * input_shape_d2);
    const int seq_idx = remaining / input_shape_d2;
    const int head_idx = remaining % input_shape_d2;
    return input + batch_idx * input_stride_d4 + seq_idx * input_stride_d3 +
           head_idx * input_stride_d2;
  }
}

// Existing two-pass implementation retained for unaligned and untested shapes.
template <typename scalar_t, int VEC_SIZE, int NUM_DIMS, bool HAS_WEIGHT>
__global__ void rms_norm_default_kernel(
    scalar_t *__restrict__ out, const scalar_t *__restrict__ input,
    int64_t input_stride_d2, int64_t input_stride_d3, int64_t input_stride_d4,
    int64_t input_shape_d2, int64_t input_shape_d3,
    const scalar_t *__restrict__ weight, float epsilon, int num_tokens,
    int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  const scalar_t *input_row = rms_input_row<scalar_t, NUM_DIMS>(
      input, input_stride_d2, input_stride_d3, input_stride_d4, input_shape_d2,
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
  vllm::vectorize_read_with_alignment<VEC_SIZE>(
      input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);

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
    const auto *weight_vec =
        reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE> *>(weight);
    for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
      vec_n_t<scalar_t, VEC_SIZE> dst;
      const vec_n_t<scalar_t, VEC_SIZE> src = input_vec[i];
      const vec_n_t<scalar_t, VEC_SIZE> w = weight_vec[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const float x = static_cast<float>(src.val[j]);
        const float wf = static_cast<float>(w.val[j]);
        dst.val[j] = static_cast<scalar_t>(x * s_variance * wf);
      }
      output_vec[i] = dst;
    }
  } else {
    for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
      vec_n_t<scalar_t, VEC_SIZE> dst;
      const vec_n_t<scalar_t, VEC_SIZE> src = input_vec[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const float x = static_cast<float>(src.val[j]);
        dst.val[j] = static_cast<scalar_t>(x * s_variance);
      }
      output_vec[i] = dst;
    }
  }
}

// The input packs stay in registers across the reduction. This eliminates the
// second global input read while keeping the cached representation packed.
template <typename scalar_t, int VEC_SIZE, int NUM_DIMS, bool HAS_WEIGHT,
          int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ __launch_bounds__(BLOCK_SIZE) void rms_norm_cached_kernel(
    scalar_t *__restrict__ out, const scalar_t *__restrict__ input,
    int64_t input_stride_d2, int64_t input_stride_d3, int64_t input_stride_d4,
    int64_t input_shape_d2, int64_t input_shape_d3,
    const scalar_t *__restrict__ weight, float epsilon, int num_tokens,
    int hidden_size) {
  using Vec = vec_n_t<scalar_t, VEC_SIZE>;
  const int vec_count = hidden_size / VEC_SIZE;
  const scalar_t *input_row = rms_input_row<scalar_t, NUM_DIMS>(
      input, input_stride_d2, input_stride_d3, input_stride_d4, input_shape_d2,
      input_shape_d3);
  const auto *input_vec = reinterpret_cast<const Vec *>(input_row);
  Vec cached[ITEMS_PER_THREAD];
  float variance = 0.0f;

#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int vec_idx = threadIdx.x + item * BLOCK_SIZE;
    Vec src{};
    if (vec_idx < vec_count)
      src = input_vec[vec_idx];
    cached[item] = src;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const float x = static_cast<float>(src.val[j]);
      variance += x * x;
    }
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  variance = BlockReduce(reduce_store).Sum(variance);
  __shared__ float s_variance;
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar_t *out_row = out + blockIdx.x * hidden_size;
  auto *output_vec = reinterpret_cast<Vec *>(out_row);
  const auto *weight_vec = reinterpret_cast<const Vec *>(weight);
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int vec_idx = threadIdx.x + item * BLOCK_SIZE;
    if (vec_idx < vec_count) {
      Vec dst;
      Vec w{};
      if constexpr (HAS_WEIGHT)
        w = weight_vec[vec_idx];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        float value = static_cast<float>(cached[item].val[j]) * s_variance;
        if constexpr (HAS_WEIGHT) {
          value *= static_cast<float>(w.val[j]);
        }
        dst.val[j] = static_cast<scalar_t>(value);
      }
      output_vec[vec_idx] = dst;
    }
  }
}

template <typename scalar_t, int VEC_SIZE, int NUM_DIMS, bool HAS_WEIGHT>
bool launch_rms_norm_cached(int block_size, int items_per_thread, dim3 grid,
                            cudaStream_t stream, scalar_t *out,
                            const scalar_t *input, int64_t input_stride_d2,
                            int64_t input_stride_d3, int64_t input_stride_d4,
                            int64_t input_shape_d2, int64_t input_shape_d3,
                            const scalar_t *weight, float epsilon,
                            int num_tokens, int hidden_size) {
#define LAUNCH_RMS_CACHED(BLOCK, ITEMS)                                        \
  rms_norm_cached_kernel<scalar_t, VEC_SIZE, NUM_DIMS, HAS_WEIGHT, BLOCK,      \
                         ITEMS><<<grid, BLOCK, 0, stream>>>(                   \
      out, input, input_stride_d2, input_stride_d3, input_stride_d4,           \
      input_shape_d2, input_shape_d3, weight, epsilon, num_tokens,             \
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

inline int select_rms_cached_block(int num_tokens, int hidden_size) {
  if (num_tokens < 64) {
    return hidden_size >= 4096 ? 512 : 0;
  }
  if (num_tokens <= 256) {
    return hidden_size >= 2560 ? 512 : 0;
  }
  if (num_tokens < 1024) {
    return 256;
  }
  return hidden_size >= 7168 ? 512 : 256;
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
template <typename scalar_t, int width, bool HasWeight>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,        // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,     // [..., hidden_size]
    const scalar_t* __restrict__ weight, // [hidden_size], nullptr if !HasWeight
    const float epsilon,
    const int num_tokens,
    const int hidden_size) {
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
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
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
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;

    _f16Vec<scalar_t, width> res = residual_v[id];
    _f16Vec<scalar_t, width> out;

    using Converter = _typeConvert<scalar_t>;

    if constexpr (HasWeight) {
      _f16Vec<scalar_t, width> w = weight_v[idx];

#pragma unroll
      for (int j = 0; j < width; ++j) {
        float x = Converter::convert(res.data[j]);
        float wf = Converter::convert(w.data[j]);
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
template <typename scalar_t, int width, bool HasWeight>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,        // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,     // [..., hidden_size]
    const scalar_t* __restrict__ weight, // [hidden_size], nullptr if !HasWeight
    const float epsilon,
    const int num_tokens,
    const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];

    if constexpr (HasWeight) {
      float w = (float)weight[idx];
      input[blockIdx.x * input_stride + idx] =
          (scalar_t)(x * s_variance * w);
    } else {
      input[blockIdx.x * input_stride + idx] =
          (scalar_t)(x * s_variance);
    }
  }
}

} // namespace vllm

void rms_norm(torch::Tensor &out, torch::Tensor &input,
              std::optional<torch::Tensor> weight, double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  if (input.stride(-1) != 1)
    input = input.contiguous();
  TORCH_CHECK(input.stride(-1) == 1);

  const bool has_weight = weight.has_value();
  if (has_weight) {
    TORCH_CHECK(weight->is_contiguous());
    TORCH_CHECK(weight->size(0) == input.size(-1));
    TORCH_CHECK(weight->scalar_type() == input.scalar_type());
  }

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  const int num_dims = input.dim();
  const int64_t input_stride_d2 = input.stride(-2);
  const int64_t input_stride_d3 = num_dims >= 3 ? input.stride(-3) : 0;
  const int64_t input_stride_d4 = num_dims >= 4 ? input.stride(-4) : 0;
  const int64_t input_shape_d2 = num_dims >= 3 ? input.size(-2) : 0;
  const int64_t input_shape_d3 = num_dims >= 4 ? input.size(-3) : 0;

  const dim3 grid(num_tokens);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VLLM_DISPATCH_RANK234(num_dims, [&] {
    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
      const scalar_t *weight_ptr =
          has_weight ? weight->data_ptr<scalar_t>() : nullptr;
      const int vec_size =
          std::gcd(static_cast<int>(16 / sizeof(scalar_t)), hidden_size);
      const int fallback_max_block = num_tokens < 256 ? 1024 : 256;
      const int fallback_block =
          std::min(hidden_size / vec_size, fallback_max_block);

      VLLM_DISPATCH_VEC_SIZE(vec_size, [&] {
        bool launched = false;
        if constexpr (vec_size == 16 / sizeof(scalar_t)) {
          const int cached_block =
              vllm::select_rms_cached_block(num_tokens, hidden_size);
          const int vec_count = hidden_size / vec_size;
          const int items_per_thread =
              cached_block == 0 ? 0
                                : (vec_count + cached_block - 1) / cached_block;
          constexpr size_t alignment = sizeof(scalar_t) * vec_size;
          const bool pointers_aligned =
              reinterpret_cast<std::uintptr_t>(input.data_ptr()) % alignment ==
                  0 &&
              reinterpret_cast<std::uintptr_t>(out.data_ptr()) % alignment ==
                  0 &&
              (!has_weight ||
               reinterpret_cast<std::uintptr_t>(weight_ptr) % alignment == 0);
          const bool can_use_cached =
              cached_block != 0 && items_per_thread >= 1 &&
              items_per_thread <= 8 && pointers_aligned &&
              vllm::rms_rows_are_vector_aligned(
                  num_dims, vec_size, input_stride_d2, input_stride_d3,
                  input_stride_d4);

          if (can_use_cached) {
            if (has_weight) {
              launched = vllm::launch_rms_norm_cached<scalar_t, vec_size,
                                                      tensor_rank, true>(
                  cached_block, items_per_thread, grid, stream,
                  out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  input_stride_d2, input_stride_d3, input_stride_d4,
                  input_shape_d2, input_shape_d3, weight_ptr,
                  static_cast<float>(epsilon), num_tokens, hidden_size);
            } else {
              launched = vllm::launch_rms_norm_cached<scalar_t, vec_size,
                                                      tensor_rank, false>(
                  cached_block, items_per_thread, grid, stream,
                  out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  input_stride_d2, input_stride_d3, input_stride_d4,
                  input_shape_d2, input_shape_d3, weight_ptr,
                  static_cast<float>(epsilon), num_tokens, hidden_size);
            }
          }
        }

        if (!launched) {
          const dim3 block(fallback_block);
          if (has_weight) {
            vllm::rms_norm_default_kernel<scalar_t, vec_size, tensor_rank, true>
                <<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                    input_stride_d2, input_stride_d3, input_stride_d4,
                    input_shape_d2, input_shape_d3, weight_ptr,
                    static_cast<float>(epsilon), num_tokens, hidden_size);
          } else {
            vllm::rms_norm_default_kernel<scalar_t, vec_size, tensor_rank,
                                          false><<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                input_stride_d2, input_stride_d3, input_stride_d4,
                input_shape_d2, input_shape_d3, weight_ptr,
                static_cast<float>(epsilon), num_tokens, hidden_size);
          }
        }
      });
    });
  });
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width, has_weight)                     \
  VLLM_DISPATCH_FLOATING_TYPES(                                          \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {            \
        if (has_weight) {                                                \
          vllm::fused_add_rms_norm_kernel<scalar_t, width, true>         \
              <<<grid, block, 0, stream>>>(                              \
                  input.data_ptr<scalar_t>(), input_stride,              \
                  residual.data_ptr<scalar_t>(),                         \
                  weight->data_ptr<scalar_t>(),                          \
                  epsilon, num_tokens, hidden_size);                     \
        } else {                                                         \
          vllm::fused_add_rms_norm_kernel<scalar_t, width, false>        \
              <<<grid, block, 0, stream>>>(                              \
                  input.data_ptr<scalar_t>(), input_stride,              \
                  residual.data_ptr<scalar_t>(),                         \
                  nullptr,                                               \
                  epsilon, num_tokens, hidden_size);                     \
        }                                                                \
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

template <uint32_t VEC_SIZE,
          uint32_t NUM_REG,
          typename T,
          int NUM_THREADS,
          bool HasWeight>
__global__ void FusedAddRMSNormKernelOpt(
    T *__restrict__ input,
    T *__restrict__ residual,
    T *__restrict__ weight,
    const uint32_t d,
    const uint32_t stride_input,
    const uint32_t stride_residual,
    float weight_bias,
    float eps)
{
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
        copy<sizeof(T) * VEC_SIZE>((void *)(ptr_input + i), (void *)local);

        T reg_residual[VEC_SIZE];
        copy<sizeof(T) * VEC_SIZE>(
            (void *)(ptr_residual + i),
            (void *)reg_residual);

#pragma unroll VEC_SIZE
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
            float x = static_cast<float>(local[j]);
            x += static_cast<float>(reg_residual[j]);
            reg_residual[j] = float_to_dstT<T>(x);
            ss += x * x;
            reg_input[k][j] = x;
        }

        copy<sizeof(T) * VEC_SIZE>(
            (void *)reg_residual,
            (void *)(ptr_residual + i));

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

    T const *ptr_weight = weight;

    k = 0;

    for (uint32_t i = tid; i < d; i += block_stride) {

        T reg_dst[VEC_SIZE];

        if constexpr (HasWeight) {

            T local_weight[VEC_SIZE];

            copy<sizeof(T) * VEC_SIZE>(
                (void *)(ptr_weight + i),
                (void *)local_weight);

#pragma unroll VEC_SIZE
            for (uint32_t j = 0; j < VEC_SIZE; j++) {
                reg_dst[j] = float_to_dstT<T>(
                    reg_input[k][j] * rms * float(local_weight[j]));
            }

        } else {

#pragma unroll VEC_SIZE
            for (uint32_t j = 0; j < VEC_SIZE; j++) {
                reg_dst[j] =
                    float_to_dstT<T>(reg_input[k][j] * rms);
            }
        }

        k++;

        copy<VEC_SIZE * sizeof(T)>(
            (void *)reg_dst,
            (void *)(ptr_input + i));
    }
}

template<typename T, bool HasWeight>
int launch_fused_add_rms_norm(
    T* input,
    T* residual,
    T* weight,
    uint32_t batch_size,
    uint32_t d,
    uint32_t stride_input,
    uint32_t stride_residual,
    float eps = 1e-5,
    cudaStream_t stream = 0)
{
    dim3 nblks(batch_size);

    constexpr int N = 16 / sizeof(T);

    if ((d & (N - 1)) == 0) {
        int blocksize = 64;
        float weight_bias = 0.0f;

        if (d <= blocksize * N) {
            constexpr int NUM_THREADS = 64;
            FusedAddRMSNormKernelOpt<N, 1, T, NUM_THREADS, HasWeight>
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
            FusedAddRMSNormKernelOpt<N, 1, T, NUM_THREADS, HasWeight>
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
            FusedAddRMSNormKernelOpt<N, 1, T, NUM_THREADS, HasWeight>
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
        } else if (d < blocksize * 8 * N) {
            constexpr int NUM_THREADS = 512;
            FusedAddRMSNormKernelOpt<N, 1, T, NUM_THREADS, HasWeight>
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
        } else if (d < blocksize * 16 * N) {
            constexpr int NUM_THREADS = 512;
            FusedAddRMSNormKernelOpt<N, 2, T, NUM_THREADS, HasWeight>
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

void fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    std::optional<torch::Tensor> weight,
    double epsilon)
{
    TORCH_CHECK(input.scalar_type() == residual.scalar_type());
    TORCH_CHECK(residual.is_contiguous());

    if (weight.has_value()) {
        TORCH_CHECK(weight->scalar_type() == input.scalar_type());
        TORCH_CHECK(weight->is_contiguous());
    }

    int hidden_size = input.size(-1);
    int64_t input_stride = input.stride(-2);
    int num_tokens = input.numel() / hidden_size;

    dim3 grid(num_tokens);

    const int max_block_size = (num_tokens < 256) ? 1024 : 256;
    dim3 block(std::min(hidden_size, max_block_size));

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
    auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());

    int status = -1;

    if (weight.has_value()) {

        if ((hidden_size % 8 == 0 && (input_stride & 7) == 0) &&
            (input.dtype() == at::ScalarType::BFloat16)) {

            status = launch_fused_add_rms_norm<maca_bfloat16, true>(
                static_cast<maca_bfloat16*>(input.data_ptr()),
                static_cast<maca_bfloat16*>(residual.data_ptr()),
                static_cast<maca_bfloat16*>(weight->data_ptr()),
                num_tokens,
                hidden_size,
                input_stride,
                hidden_size,
                epsilon,
                stream);

        } else if (input.dtype() == at::ScalarType::Half) {

            status = launch_fused_add_rms_norm<half, true>(
                static_cast<half*>(input.data_ptr()),
                static_cast<half*>(residual.data_ptr()),
                static_cast<half*>(weight->data_ptr()),
                num_tokens,
                hidden_size,
                input_stride,
                hidden_size,
                epsilon,
                stream);
        }

    } else {

        if ((hidden_size % 8 == 0 && (input_stride & 7) == 0) &&
            (input.dtype() == at::ScalarType::BFloat16)) {

            status = launch_fused_add_rms_norm<maca_bfloat16, false>(
                static_cast<maca_bfloat16*>(input.data_ptr()),
                static_cast<maca_bfloat16*>(residual.data_ptr()),
                nullptr,
                num_tokens,
                hidden_size,
                input_stride,
                hidden_size,
                epsilon,
                stream);

        } else if (input.dtype() == at::ScalarType::Half) {

            status = launch_fused_add_rms_norm<half, false>(
                static_cast<half*>(input.data_ptr()),
                static_cast<half*>(residual.data_ptr()),
                nullptr,
                num_tokens,
                hidden_size,
                input_stride,
                hidden_size,
                epsilon,
                stream);
        }
    }

    if (status == 0) {
        return;
    }

    constexpr int vector_width = 8;
    constexpr int req_alignment_bytes = vector_width * 2;

    bool offsets_are_multiple_of_vector_width =
        hidden_size % vector_width == 0 &&
        input_stride % vector_width == 0;

    bool batch_invariant_launch = vllm::vllm_is_batch_invariant();

    if (weight.has_value()) {

        auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight->data_ptr());

        bool ptrs_are_aligned =
            inp_ptr % req_alignment_bytes == 0 &&
            res_ptr % req_alignment_bytes == 0 &&
            wt_ptr % req_alignment_bytes == 0;

        if (ptrs_are_aligned &&
            offsets_are_multiple_of_vector_width &&
            !batch_invariant_launch) {
            LAUNCH_FUSED_ADD_RMS_NORM(8, true);
        } else {
            LAUNCH_FUSED_ADD_RMS_NORM(0, true);
        }

    } else {

        bool ptrs_are_aligned =
            inp_ptr % req_alignment_bytes == 0 &&
            res_ptr % req_alignment_bytes == 0;

        if (ptrs_are_aligned &&
            offsets_are_multiple_of_vector_width &&
            !batch_invariant_launch) {
            LAUNCH_FUSED_ADD_RMS_NORM(8, false);
        } else {
            LAUNCH_FUSED_ADD_RMS_NORM(0, false);
        }
    }
}
