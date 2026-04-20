#include "type_convert.cuh"
#include "dispatch_utils.h"
#include "cub_helpers.h"
#include "core/batch_invariant.hpp"
#include "quantization/vectorization_utils.cuh"
#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

#include <cub/cub.cuh>

namespace vllm {
// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t, int VEC_SIZE, int NUM_DIMS>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const int64_t input_stride_d2,        // input.stride(-2)
    const int64_t input_stride_d3,        // input.stride(-3)
    const int64_t input_stride_d4,        // input.stride(-4)
    const int64_t input_shape_d2,         // input.size(-2)
    const int64_t input_shape_d3,         // input.size(-3)
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  const scalar_t* input_row;
  if constexpr (NUM_DIMS == 2) {
    // 2D for layernorm normal case [batch_size, hidden]
    input_row = input + blockIdx.x * input_stride_d2;
  } else if constexpr (NUM_DIMS == 3) {
    // 3D for q/k norm [batch_size, num_heads, head_size]
    int batch_idx = blockIdx.x / input_shape_d2;
    int head_idx = blockIdx.x % input_shape_d2;
    input_row =
        input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
  } else if constexpr (NUM_DIMS == 4) {
    // 4D for transformers model_impl qk norm [batch, seq, head, head_dim]
    int batch_idx = blockIdx.x / (input_shape_d3 * input_shape_d2);
    int remaining = blockIdx.x % (input_shape_d3 * input_shape_d2);
    int seq_idx = remaining / input_shape_d2;
    int head_idx = remaining % input_shape_d2;
    input_row = input + batch_idx * input_stride_d4 +
                seq_idx * input_stride_d3 + head_idx * input_stride_d2;
  }

  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };
  auto scalar_op = [&variance](const scalar_t& val) {
    float x = static_cast<float>(val);
    variance += x * x;
  };
  vllm::vectorize_read_with_alignment<VEC_SIZE>(
      input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar_t* out_row = out + blockIdx.x * hidden_size;
  auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
  auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(weight);
  auto* v_out = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE>*>(out_row);
  for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
    vec_n_t<scalar_t, VEC_SIZE> dst;
    vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[i];
    vec_n_t<scalar_t, VEC_SIZE> src2 = v_w[i];
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float x = static_cast<float>(src1.val[j]);
      dst.val[j] = ((scalar_t)(x * s_variance)) * src2.val[j];
    }
    v_out[i] = dst;
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
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
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    input_v[strided_id] = temp;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
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
    input[blockIdx.x * input_stride + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

}  // namespace vllm

void rms_norm(torch::Tensor& out,     // [..., hidden_size]
              torch::Tensor& input,   // [..., hidden_size]
              torch::Tensor& weight,  // [hidden_size]
              double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  if (input.stride(-1) != 1) {
    input = input.contiguous();
  }
  TORCH_CHECK(input.stride(-1) == 1);
  TORCH_CHECK(weight.is_contiguous());

  int hidden_size = input.size(-1);

  int num_tokens = input.numel() / hidden_size;
  int num_dims = input.dim();
  int64_t input_stride_d2 = input.stride(-2);
  int64_t input_stride_d3 = (num_dims >= 3) ? input.stride(-3) : 0;
  int64_t input_stride_d4 = (num_dims >= 4) ? input.stride(-4) : 0;
  int64_t input_shape_d2 = (num_dims >= 3) ? input.size(-2) : 0;
  int64_t input_shape_d3 = (num_dims >= 4) ? input.size(-3) : 0;

  // For large num_tokens, use smaller blocks to increase SM concurrency.
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 grid(num_tokens);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_RANK234(num_dims, [&] {
    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
      const int calculated_vec_size =
          std::gcd(16 / sizeof(scalar_t), hidden_size);
      const int block_size =
          std::min(hidden_size / calculated_vec_size, max_block_size);
      dim3 block(block_size);
      VLLM_DISPATCH_VEC_SIZE(calculated_vec_size, [&] {
        vllm::rms_norm_kernel<scalar_t, vec_size, tensor_rank>
            <<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                input_stride_d2, input_stride_d3, input_stride_d4,
                input_shape_d2, input_shape_d3, weight.data_ptr<scalar_t>(),
                epsilon, num_tokens, hidden_size);
      });
    });
  });
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                    \
  VLLM_DISPATCH_FLOATING_TYPES(                                             \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {               \
        vllm::fused_add_rms_norm_kernel<scalar_t, width>                    \
            <<<grid, block, 0, stream>>>(                                   \
                input.data_ptr<scalar_t>(), input_stride,                   \
                residual.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), \
                epsilon, num_tokens, hidden_size);                          \
      });

template<typename T>
static __device__ __forceinline__ T float_to_dstT(float value) {
  return static_cast<T>(value);
}

template<>
static __device__ __forceinline__ maca_bfloat16 float_to_dstT(float value) {
  return __float2bfloat16(value);
}

template<>
static __device__ __forceinline__ half float_to_dstT(float value) {
  return __float2half(value);
}

template<int N> 
__device__  __forceinline__ void copy(void * src, void* dst){
    int8_t* ptr_src = (int8_t*)src;
    int8_t* ptr_dst = (int8_t*)dst;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
        ptr_dst[i] = ptr_src[i];
    }
}

template<>
__device__ __forceinline__ void copy<16>(void* src, void* dst) {
    float4 *ptr_src = (float4*)src;
    float4* ptr_dst = (float4*)dst;
    *ptr_dst = *ptr_src;
}

template<>
__device__ __forceinline__ void copy<8>(void* src, void* dst) {
    float2 *ptr_src = (float2*)src;
    float2* ptr_dst = (float2*)dst;
    *ptr_dst = *ptr_src;
}

template<>
__device__ __forceinline__ void copy<4>(void* src, void* dst) {
    float *ptr_src = (float*)src;
    float* ptr_dst = (float*)dst;
    *ptr_dst = *ptr_src;
}

template<>
__device__ __forceinline__ void copy<2>(void* src, void* dst) {
    half *ptr_src = (half*)src;
    half* ptr_dst = (half*)dst;
    *ptr_dst = *ptr_src;
}

template<>
__device__ __forceinline__ void copy<1>(void* src, void* dst) {
    int8_t *ptr_src = (int8_t*)src;
    int8_t* ptr_dst = (int8_t*)dst;
    *ptr_dst = *ptr_src;
}

template<uint32_t VEC_SIZE, uint32_t NUM_REG, typename T, int NUM_THREADS>
__global__ void FusedAddRMSNormKernelOpt(T* __restrict__ input, T* __restrict__ residual,
                                    T* __restrict__ weight, const uint32_t d,
                                    const uint32_t stride_input, const uint32_t stride_residual,
                                    float weight_bias, float eps) {
  float rms = 0;
  T * ptr_input = input + blockIdx.x * stride_input;
  T* ptr_residual = residual + blockIdx.x * stride_residual;

  float reg_input[NUM_REG][VEC_SIZE];
  // sum of squares
  float ss = 0.0f;
  uint32_t tid = threadIdx.x * VEC_SIZE;
  uint32_t block_stride = NUM_THREADS * VEC_SIZE;
  uint32_t k = 0;
  for(uint32_t i = tid; i < d; i += block_stride) {
    T local[VEC_SIZE];
    copy<sizeof(T)*VEC_SIZE>((void*)(ptr_input + i), (void*)local);
    T reg_residual[VEC_SIZE];
    copy<sizeof(T)*VEC_SIZE>((void*)(ptr_residual + i), (void*)reg_residual);
    
    #pragma unroll VEC_SIZE
    for(uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = static_cast<float>(local[j]);
      x += static_cast<float>(reg_residual[j]);
      reg_residual[j] = float_to_dstT<T>(x);
      ss += x * x;
      reg_input[k][j] = x;
    }
    copy<sizeof(T)*VEC_SIZE>((void*)reg_residual, (void*)(ptr_residual + i));
    k++;
  }
  // using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;
  // __shared__ typename BlockReduce::TempStorage reduceStore;
  // ss = BlockReduce(reduceStore).Reduce(ss, cub::Sum{}, NUM_THREADS);
    constexpr int sm_size = NUM_THREADS >> 4;
    constexpr int sm_size2 = sm_size / 2;
    __shared__ float sm_sum[sm_size];
    if constexpr (sm_size == 32) {
        for(int i = 8; i > 0; i >>= 1) {
            ss += __shfl_down_sync_16(0xffffffffffffffff, ss, i);
        }
        int lane_id = threadIdx.x & 15;
        int group_id = threadIdx.x >> 4;
        if(lane_id == 0) {
            sm_sum[group_id] = ss;
        }
        __syncthreads();
        __shared__ float sm_sum2[sm_size>>4];
        if(threadIdx.x < sm_size) {
        float data = sm_sum[threadIdx.x];
        for(int i = 8; i >= 1; i >>= 1) {
            data += __shfl_down_sync_16(0xffffffffffffffff, data, i);
        }

        if(lane_id == 0) {
            sm_sum2[group_id] = data;
        }
        }
        __syncthreads();
        ss = sm_sum2[0] + sm_sum2[1];
    } else if constexpr(sm_size == 16) {
        for(int i = 8; i > 0; i >>=1 ) {
        ss += __shfl_down_sync_16(0xffffffffffffffff, ss, i);
        }
        int lane_id = threadIdx.x & 15;
        int group_id = threadIdx.x >> 4;
        if(lane_id == 0) {
        sm_sum[group_id] = ss;
        }
        __syncthreads();
        if(threadIdx.x < sm_size) {
        float data = sm_sum[threadIdx.x];
        for(int i = 8; i >= 1; i >>= 1) {
            data += __shfl_down_sync_16(0xffffffffffffffff, data, i);
        }
        if(threadIdx.x == 0) {
            sm_sum[0] = data;
        }
        }
        __syncthreads();
        ss = sm_sum[0];
    } else if constexpr(sm_size == 8) {
        for(int i = 8; i > 0; i >>=1 ) {
        ss += __shfl_down_sync_16(0xffffffffffffffff, ss, i);
        }
        int lane_id = threadIdx.x & 15;
        int group_id = threadIdx.x >> 4;
        if(lane_id == 0) {
        sm_sum[group_id] = ss;
        }
        __syncthreads();
        if(threadIdx.x < sm_size) {
        float data = sm_sum[threadIdx.x];
        for(int i = 4; i >= 1; i >>= 1) {
            data += __shfl_down_sync_16(0xffffffffffffffff, data, i);
        }
        if(threadIdx.x == 0) {
            sm_sum[0] = data;
        }
        }
        __syncthreads();
        ss = sm_sum[0];
    } else if constexpr(sm_size == 4) {
        for(int i = 8; i > 0; i >>=1 ) {
        ss += __shfl_down_sync_16(0xffffffffffffffff, ss, i);
        }
        int lane_id = threadIdx.x & 15;
        int group_id = threadIdx.x >> 4;
        if(lane_id == 0) {
        sm_sum[group_id] = ss;
        }
        __syncthreads();
        if(threadIdx.x < sm_size) {
        float data = sm_sum[threadIdx.x];
        for(int i = 2; i >= 1; i >>= 1) {
            data += __shfl_down_sync_16(0xffffffffffffffff, data, i);
        }
        if(threadIdx.x == 0) {
            sm_sum[0] = data;
        }
        }
        __syncthreads();
        ss = sm_sum[0];
    }
  __shared__ float s_rms;
  if(threadIdx.x == 0) {
    s_rms = rsqrtf(ss / (float)d + eps);
  }
  __syncthreads();
  rms = s_rms;
  T const* ptr_weight = weight;
  k = 0;
  for(uint32_t i = tid; i < d; i += block_stride) {
    T local_weight[VEC_SIZE];
    copy<sizeof(T)*VEC_SIZE>((void*)(ptr_weight + i),(void*)local_weight);
    T reg_dst[VEC_SIZE];
    #pragma unroll VEC_SIZE
    for(uint32_t j = 0; j < VEC_SIZE; j++) {
      reg_dst[j] = float_to_dstT<T>(reg_input[k][j] * rms * float(local_weight[j]));
    }
    k++;
    copy<VEC_SIZE*sizeof(T)>((void*)reg_dst, (void*)(ptr_input + i));
  }
}

template<typename T>
int launch_fused_add_rmsnorm(T* input, T* residual, T* weight, uint32_t batch_size, uint32_t d,
                            uint32_t stride_input, uint32_t stride_residual, float eps = 1e-5,
                            cudaStream_t stream = 0)
{
  dim3 nblks(batch_size);
  constexpr int N = 16 / sizeof(T);
  if((d & (N - 1)) == 0) {
    int blocksize = 64;
    float weight_bias = 0.0f;
    if(d <= blocksize * N) {
        constexpr int NUM_THREADS = 64;
        FusedAddRMSNormKernelOpt<N, 1, T, NUM_THREADS><<<nblks, NUM_THREADS, 0, stream>>>(input, residual, weight, d, stride_input, stride_residual, weight_bias, eps);
        return 0;
    } else if(d <= blocksize * 2 * N) {
        constexpr int NUM_THREADS = 128;
        FusedAddRMSNormKernelOpt<N, 1, T, NUM_THREADS><<<nblks, NUM_THREADS, 0, stream>>>(input, residual, weight, d, stride_input, stride_residual, weight_bias, eps);
        return 0;
    } else if(d <= blocksize * 4 * N) {
        constexpr int NUM_THREADS = 256;
        FusedAddRMSNormKernelOpt<N, 1, T, NUM_THREADS><<<nblks, NUM_THREADS, 0, stream>>>(input, residual, weight, d, stride_input, stride_residual, weight_bias, eps);
        return 0;
    } else if(d < blocksize * 8 * N) {
        constexpr int NUM_THREADS = 512;
        FusedAddRMSNormKernelOpt<N, 1, T, NUM_THREADS><<<nblks, NUM_THREADS, 0, stream>>>(input, residual, weight, d, stride_input, stride_residual, weight_bias, eps);
        return 0;
    } else if(d < blocksize * 16 * N) {
        constexpr int NUM_THREADS = 512;
        FusedAddRMSNormKernelOpt<N, 2, T, NUM_THREADS><<<nblks, NUM_THREADS, 0, stream>>>(input, residual, weight, d, stride_input, stride_residual, weight_bias, eps);
        return 0;
    }
  }

  return -1;
}

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon) {
  TORCH_CHECK(weight.scalar_type() == input.scalar_type());
  TORCH_CHECK(input.scalar_type() == residual.scalar_type());
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  int hidden_size = input.size(-1);
  int64_t input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());

  int status = -1;
  if((hidden_size & 7) == 0 && (input_stride & 7) == 0) {
    if (input.dtype() == at::ScalarType::BFloat16) {
      status  = launch_fused_add_rmsnorm<maca_bfloat16>(static_cast<maca_bfloat16*>(input.data_ptr()),
        static_cast<maca_bfloat16*>(residual.data_ptr()),
        static_cast<maca_bfloat16*>(weight.data_ptr()),
        num_tokens,
        hidden_size,
        input_stride,
        hidden_size,
        epsilon,
        stream);
    } else if(input.dtype() == at::ScalarType::Half) {
      status  = launch_fused_add_rmsnorm<half>(static_cast<half*>(input.data_ptr()),
        static_cast<half*>(residual.data_ptr()),
        static_cast<half*>(weight.data_ptr()),
        num_tokens,
        hidden_size,
        input_stride,
        hidden_size,
        epsilon,
        stream);
    }
  }
  if(status == 0) return;

  constexpr int vector_width = 8;
  constexpr int req_alignment_bytes =
      vector_width * 2;  // vector_width * sizeof(bfloat16 or float16) (float32
                         // falls back to non-vectorized version anyway)
  bool ptrs_are_aligned = inp_ptr % req_alignment_bytes == 0 &&
                          res_ptr % req_alignment_bytes == 0 &&
                          wt_ptr % req_alignment_bytes == 0;
  bool offsets_are_multiple_of_vector_width =
      hidden_size % vector_width == 0 && input_stride % vector_width == 0;
  bool batch_invariant_launch = vllm::vllm_is_batch_invariant();
  if (ptrs_are_aligned && offsets_are_multiple_of_vector_width &&
      !batch_invariant_launch) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}
