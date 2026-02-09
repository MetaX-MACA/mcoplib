// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include "../kernel/all_reduce_kernel.cuh"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

template<typename T, int ELTS_PER_LDG>
__global__ void reduce_kernel_max_vec(T* input, T* output, int num_tokens, int hidden_size, int num_vec) {
    T* thread_read_ptr = input + blockIdx.x * hidden_size;
    using AccessType = AlignedArrayI4<T, ELTS_PER_LDG>;
    T row_chunk[ELTS_PER_LDG];
    AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
    T val = __float2bfloat16(-9999.f);

#pragma unroll
    for (int ii = threadIdx.x; ii < num_vec; ii += blockDim.x)
    {
        *row_chunk_vec_ptr = vec_thread_read_ptr[ii];
#pragma unroll
        for (int jj = 0; jj < ELTS_PER_LDG; ++jj) {
            val = val > row_chunk[jj] ? val : row_chunk[jj];
        }
    }

    val = BlockReduceMax<T>(val);
    if(threadIdx.x == 0) {
        output[blockIdx.x] = val;
    }
}

template<typename T, int ELTS_PER_LDG>
__global__ void reduce_kernel_sum_vec(T* input, T* output, int num_tokens, int hidden_size, int num_vec) {
    T* thread_read_ptr = input + blockIdx.x * hidden_size;
    using AccessType = AlignedArrayI4<T, ELTS_PER_LDG>;
    T row_chunk[ELTS_PER_LDG];
    float row_chunk_float[ELTS_PER_LDG];
    AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
    float val_f = 0.0f;

#pragma unroll
    for (int ii = threadIdx.x; ii < num_vec; ii += blockDim.x)
    {
        *row_chunk_vec_ptr = vec_thread_read_ptr[ii];
#pragma unroll
        for (int jj = 0; jj < ELTS_PER_LDG; ++jj) {
            val_f += __bfloat162float(row_chunk[jj]);
        }
    }

    val_f = BlockReduceSum<float>(val_f);
    if(threadIdx.x == 0) {
        output[blockIdx.x] = __float2bfloat16(val_f);
    }
}

template<typename T>
__global__ void reduce_kernel_max (T* input, T* output, int num_tokens, int hidden_size) {
    if (threadIdx.x >= hidden_size) return;
    T* thread_read_ptr = input + blockIdx.x * hidden_size;
    T val = *thread_read_ptr;

#pragma unroll
    for (int ii = threadIdx.x; ii < hidden_size; ii += blockDim.x)
    {
        T tmp = thread_read_ptr[ii];
        val = max_<T>(val, tmp);
    }

    val = BlockReduceMax<T>(val);
    if(threadIdx.x == 0) {
        output[blockIdx.x] = val;
    }
}

template<typename T>
__global__ void reduce_kernel_sum (T* input, T* output, int num_tokens, int hidden_size) {
    T* thread_read_ptr = input + blockIdx.x * hidden_size;
    float val = 0.0f;

#pragma unroll
    for (int ii = threadIdx.x; ii < hidden_size; ii += blockDim.x)
    {
        val += __bfloat162float(thread_read_ptr[ii]);
    }

    val = BlockReduceSum<float>(val);
    if(threadIdx.x == 0) {
        output[blockIdx.x] = __float2bfloat16(val);
    }
}

template<typename T>
void launch_all_reduce_max_kernel(T* input, T* output, int num_tokens, int hidden_size, const cudaStream_t stream) {
    constexpr int WARP_SIZE = 32;
    constexpr int BYTES_PER_LDG = 16;
    constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    constexpr int MAX_DIM = 512;
    int grid_dim = num_tokens;

    if ((hidden_size & (ELTS_PER_LDG - 1)) == 0) {
        const int num_vec = (hidden_size + ELTS_PER_LDG - 1) / ELTS_PER_LDG;
        int block_dim = min(MAX_DIM, num_vec);
        reduce_kernel_max_vec<T, ELTS_PER_LDG><<<grid_dim, block_dim, 0, stream>>>(input, output, num_tokens, hidden_size, num_vec);
        return;
    }
    reduce_kernel_max<T><<<grid_dim, MAX_DIM, 0, stream>>>(input, output, num_tokens, hidden_size);
    return;
}

template<typename T>
void launch_all_reduce_sum_kernel(T* input, T* output, int num_tokens, int hidden_size, const cudaStream_t stream) {
    constexpr int WARP_SIZE = 32;
    constexpr int BYTES_PER_LDG = 16;
    constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    constexpr int MAX_DIM = 512;
    int grid_dim = num_tokens;

    if ((hidden_size & (ELTS_PER_LDG - 1)) == 0) {
        const int num_vec = (hidden_size + ELTS_PER_LDG - 1) / ELTS_PER_LDG;
        int block_dim = min(MAX_DIM, num_vec);
        reduce_kernel_sum_vec<T, ELTS_PER_LDG><<<grid_dim, block_dim, 0, stream>>>(input, output, num_tokens, hidden_size, num_vec);
        return;
    }
    int block_dim = min(MAX_DIM, hidden_size);
    reduce_kernel_sum<T><<<grid_dim, MAX_DIM, 0, stream>>>(input, output, num_tokens, hidden_size);
    return;
}

void all_reduce_max(at::Tensor input,
                at::Tensor output)               // [num_tokens, hidden_size]
{
    DEBUG_TRACE_PARAMS(input, output);
    DEBUG_DUMP_PARAMS(input, output);
    const int hidden_size = input.size(-1);
    const int num_tokens = input.numel() / hidden_size;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (input.dtype() == at::ScalarType::BFloat16) {
        launch_all_reduce_max_kernel<bfloat16>(reinterpret_cast<bfloat16*>(input.data_ptr<at::BFloat16>()), reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>()), num_tokens, hidden_size, stream);
    }else {
        TORCH_CHECK(false, "Only bfloat16 are supported");
    }
}

void all_reduce_sum(at::Tensor input,
                at::Tensor output)               // [num_tokens, hidden_size]
{
    DEBUG_TRACE_PARAMS(input, output);
    DEBUG_DUMP_PARAMS(input, output);
    const int hidden_size = input.size(-1);
    const int num_tokens = input.numel() / hidden_size;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (input.dtype() == at::ScalarType::BFloat16) {
        launch_all_reduce_sum_kernel<bfloat16>(reinterpret_cast<bfloat16*>(input.data_ptr<at::BFloat16>()), reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>()), num_tokens, hidden_size, stream);
    }else {
        TORCH_CHECK(false, "Only bfloat16 are supported");
    }
}