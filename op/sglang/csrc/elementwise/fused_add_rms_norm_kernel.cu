/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include "utils.h"
#include "norm.cuh"

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

template<uint32_t VEC_SIZE, uint32_t NUM_REG, typename T, int NUM_THREADS, int MIN_BLOCKS = 8>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS)
FusedAddRMSNormKernelOpt(T* __restrict__ input, T* __restrict__ residual,
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

    // Single-barrier reduction: each 16-lane group reduces via native shuffle,
    // lane 0 writes its partial, then ONE __syncthreads, then every thread sums
    // the (NUM_THREADS/16) partials itself and computes rms locally. This removes
    // the second-level shared reduce AND the s_rms broadcast barrier (3 -> 1 barrier).
    constexpr int sm_size = NUM_THREADS >> 4;
    __shared__ float sm_sum[sm_size];
    #pragma unroll
    for(int i = 8; i > 0; i >>= 1) {
        ss += __shfl_down_sync_16(0xffffffffffffffff, ss, i);
    }
    int lane_id = threadIdx.x & 15;
    int group_id = threadIdx.x >> 4;
    if(lane_id == 0) {
        sm_sum[group_id] = ss;
    }
    __syncthreads();
    float tot = 0.0f;
    #pragma unroll
    for(int g = 0; g < sm_size; g++) {
        tot += sm_sum[g];
    }
    rms = __builtin_mxc_rcpf(sqrtf(tot / (float)d + eps));
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
                            uint32_t stride_input, uint32_t stride_residual, int num_ap, float eps = 1e-5,
                            bool enable_pdl = false, cudaStream_t stream = 0)
{
  dim3 nblks(batch_size);
  constexpr int N = 16 / sizeof(T);
  int num_peu = num_ap  * 4;
  int num1 = 1;
  if(batch_size < num_peu) {
    num1 = (num_peu + batch_size - 1) / batch_size;
  }

  if((d & (N - 1)) == 0) {
    int blocksize = 64;
    float weight_bias = 0.0f;
    // C600-U tuned path: hidden=4096, bf16 (N=8). 128 threads x 4 regs = 32 elems/thread;
    // small blocks -> many resident blocks per AP so the reduction barrier of one block
    // is hidden by the streaming loads/stores of others (barrier-bubble hiding).
    if(d == 4096 && N == 8) {
        if(num1 >= 2) {
          constexpr int NUM_THREADS = 1024;
          FusedAddRMSNormKernelOpt<N/2, 1, T, NUM_THREADS><<<nblks, NUM_THREADS, 0, stream>>>(input, residual, weight, d, stride_input, stride_residual, weight_bias, eps);
        } else {
          constexpr int NUM_THREADS = 128;
          FusedAddRMSNormKernelOpt<N, 4, T, NUM_THREADS><<<nblks, NUM_THREADS, 0, stream>>>(input, residual, weight, d, stride_input, stride_residual, weight_bias, eps);
        }
        return 0;
    }
    
    if(d == 5120 && N == 8) {
      if(num1 >= 2) {
          constexpr int NUM_THREADS = 1024;
          FusedAddRMSNormKernelOpt<N/2, 2, T, NUM_THREADS><<<nblks, NUM_THREADS, 0, stream>>>(input, residual, weight, d, stride_input, stride_residual, weight_bias, eps);
      } else {
        constexpr int NUM_THREADS = 256;
        FusedAddRMSNormKernelOpt<N, 3, T, NUM_THREADS><<<nblks, NUM_THREADS, 0, stream>>>(input, residual, weight, d, stride_input, stride_residual, weight_bias, eps);
      }
      return 0;
    }

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

void sgl_fused_add_rmsnorm(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps, bool enable_pdl) {
  CHECK_INPUT(input);
  CHECK_INPUT(residual);
  CHECK_INPUT(weight);
  auto device = input.device();
  CHECK_EQ(residual.device(), device);
  CHECK_EQ(weight.device(), device);
  CHECK_DIM(2, input);     // input: (batch_size, hidden_size)
  CHECK_DIM(2, residual);  // residual: (batch_size, hidden_size)
  CHECK_DIM(1, weight);    // weight: (hidden_size)
  CHECK_EQ(input.size(0), residual.size(0));
  CHECK_EQ(input.size(1), residual.size(1));
  CHECK_EQ(input.size(1), weight.size(0));
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);

  cudaDeviceProp prop;
  int devId = device.index();
  cudaError_t err = cudaGetDeviceProperties(&prop, devId);
  int num_ap = prop.multiProcessorCount;
  cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
  int status = -1;
  if((hidden_size & 7) == 0 && (input.stride(0)& 7) == 0 && (residual.stride(0) & 7) == 0) {
    if (input.dtype() == at::ScalarType::BFloat16) {
      status  = launch_fused_add_rmsnorm<maca_bfloat16>(static_cast<maca_bfloat16*>(input.data_ptr()),
        static_cast<maca_bfloat16*>(residual.data_ptr()),
        static_cast<maca_bfloat16*>(weight.data_ptr()),
        batch_size,
        hidden_size,
        input.stride(0),
        residual.stride(0),
        num_ap,
        eps,
        enable_pdl,
        torch_current_stream);
    } else if(input.dtype() == at::ScalarType::Half) {
      status  = launch_fused_add_rmsnorm<half>(static_cast<half*>(input.data_ptr()),
        static_cast<half*>(residual.data_ptr()),
        static_cast<half*>(weight.data_ptr()),
        batch_size,
        hidden_size,
        input.stride(0),
        residual.stride(0),
        num_ap,
        eps,
        enable_pdl,
        torch_current_stream);
    }
  }
  if(status == 0) return;
  // support float16, bfloat16 and float32
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    cudaError_t status = mcoplib::norm::FusedAddRMSNorm(
        static_cast<c_type*>(input.data_ptr()),
        static_cast<c_type*>(residual.data_ptr()),
        static_cast<c_type*>(weight.data_ptr()),
        batch_size,
        hidden_size,
        input.stride(0),
        residual.stride(0),
        eps,
        enable_pdl,
        torch_current_stream);
    TORCH_CHECK(
        status == cudaSuccess, "FusedAddRMSNorm failed with error code " + std::string(cudaGetErrorString(status)));
    return true;
  });
}
