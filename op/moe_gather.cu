// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include "../kernel/utils.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

__host__ __device__ __forceinline__
int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

template<typename scalar_t, int VPT>
__global__ void moe_gather_kernel_vectorized_tuned(const scalar_t* input, const int* token_offset, scalar_t* output, const int hidden_size, const int num_tokens) {
    scalar_t row_chunk[VPT];
    const int target = token_offset[blockIdx.x];
    scalar_t* output_block_ptr = output + target * hidden_size;
    using VecType = AlignedArrayI4<scalar_t, VPT>;
    VecType* row_chunk_vec_ptr = reinterpret_cast<VecType*>(row_chunk);
    const VecType* input_vec_ptr = reinterpret_cast<const VecType*>(input + blockIdx.x * hidden_size + threadIdx.x * VPT);
    *row_chunk_vec_ptr = *input_vec_ptr;
    for (int idx = 0; idx < VPT; ++idx) {
        atomicAdd((output_block_ptr + threadIdx.x * VPT + idx), row_chunk[idx]);
    }
}

template<typename scalar_t, int vec_size>
__global__ void moe_gather_vec(
    const scalar_t* __restrict__ input, 
    const int* __restrict__ token_offset, 
    scalar_t* __restrict__ output, 
    const int hidden_size, 
    const int num_tokens,
    const int totalSize
) {
    using VecType = AlignedArrayI4<scalar_t, vec_size>;
    int start = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
    int step = gridDim.x * blockDim.x * vec_size;

    for (int linearIndex = start;
        linearIndex < totalSize;
        linearIndex += step) {

        int srcIndex, elementInSlice;
        srcIndex = linearIndex / hidden_size;
        elementInSlice = linearIndex % hidden_size;
        int dstIndex = token_offset[srcIndex];
        int Offset = elementInSlice;
        int dstOffset = Offset + dstIndex * hidden_size;
        int srcOffset = Offset + srcIndex * hidden_size;

        VecType SrcVal = *(reinterpret_cast<const VecType*>(input + srcOffset));

        __maca_bfloat162 value2;
        value2.x = *reinterpret_cast<__maca_bfloat16*>(&SrcVal.data[0]);
        value2.y = *reinterpret_cast<__maca_bfloat16*>(&SrcVal.data[1]);
        atomicAdd(reinterpret_cast<__maca_bfloat162*>(output + dstOffset), value2);
    }
}

template<typename scalar_t, int hidden_size, int block_dim, int VPT = hidden_size / block_dim>
__global__ void moe_gather_no_atomic_add(
    const scalar_t* __restrict__ input, 
    const int* __restrict__ token_offset, 
    scalar_t* __restrict__ output, 
    const int num_tokens
) {
    __shared__ int src[40];
    __shared__ int total_rows;
    if (threadIdx.x == 0) {
        total_rows = 0;
        for (int idx = 0; idx < num_tokens; ++idx) {
            int target_val = token_offset[idx];
            if (target_val != blockIdx.x) continue;
            src[total_rows] = idx;
            ++total_rows;
        }
    }
    __syncthreads();
    if (total_rows == 0) return;

    scalar_t sum[VPT];
    scalar_t row_chunk[VPT];
    using VecType = AlignedArrayI4<scalar_t, VPT>;
    VecType* row_chunk_vec_ptr = reinterpret_cast<VecType*>(row_chunk);
    VecType* sum_vec_ptr = reinterpret_cast<VecType*>(sum);
    for (int idx = 0; idx < VPT; ++idx) {
        sum[idx] = __float2bfloat16(0.0f);
    }

    for (int idx = 0; idx < total_rows; ++idx) {
        int src_row = src[idx];
        const VecType* input_vec_ptr = reinterpret_cast<const VecType*>(input + src_row * hidden_size + threadIdx.x * VPT);
        *row_chunk_vec_ptr = *input_vec_ptr;
        for (int idx = 0; idx < VPT; ++idx) {
            sum[idx] += row_chunk[idx];
        }
    }
    VecType* output_vec_ptr = reinterpret_cast<VecType*>(output + blockIdx.x * hidden_size + threadIdx.x * VPT);
    *output_vec_ptr = *sum_vec_ptr;
}

template<typename scalar_t>
__global__ void moe_gather_kernel(const scalar_t* input, const int* token_offset, scalar_t* output, const int hidden_size, const int num_tokens){
    const scalar_t* input_block_ptr = input + blockIdx.x * hidden_size;
    const int target = token_offset[blockIdx.x];
    scalar_t* output_block_ptr = output + target * hidden_size;
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        atomicAdd(output_block_ptr + idx, input_block_ptr[idx]);
    }
}

template<typename scalar_t>
void launch_moe_gather_kernel(const scalar_t* scatter_tokens, const int* scatter_tokens_offset, scalar_t* convergent_tokens,
                                const int hidden_size, const int num_tokens, const cudaStream_t& stream){
    constexpr int vec_size = 2;
    if (hidden_size % vec_size == 0) {
        if (num_tokens <= 40 && hidden_size == 4096) {
            moe_gather_no_atomic_add<scalar_t, 4096, 512><<<num_tokens, 512, 0, stream>>>(scatter_tokens, scatter_tokens_offset, convergent_tokens, num_tokens);
            return;
        }
        const int totalSize = num_tokens * hidden_size;
        const int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        dim3 grid(min(ceil_div(totalSize, 128 * vec_size), (mpc * 8 * 4 * 2)));
        dim3 block(min(totalSize, 128));
        moe_gather_vec<scalar_t, vec_size><<<grid, block, 0, stream>>>(scatter_tokens, scatter_tokens_offset, convergent_tokens, hidden_size, num_tokens, totalSize);
        return;
    }
    
    constexpr int block_size = 512;
    moe_gather_kernel<scalar_t><<<num_tokens, block_size, 0, stream>>>(scatter_tokens, scatter_tokens_offset, convergent_tokens, hidden_size, num_tokens);
}

void moe_gather(at::Tensor scatter_tokens, at::Tensor scatter_tokens_offset, at::Tensor convergent_tokens){
    DEBUG_TRACE_PARAMS(scatter_tokens, scatter_tokens_offset, convergent_tokens);
	DEBUG_DUMP_PARAMS(scatter_tokens, scatter_tokens_offset, convergent_tokens);
    const int hidden_size = scatter_tokens.size(-1);
    const int num_tokens = scatter_tokens.numel() / hidden_size;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (scatter_tokens.dtype() == at::ScalarType::BFloat16) {
        launch_moe_gather_kernel<bfloat16>(reinterpret_cast<bfloat16*>(scatter_tokens.data_ptr<at::BFloat16>()), scatter_tokens_offset.data_ptr<int>(), 
                                                            reinterpret_cast<bfloat16*>(convergent_tokens.data_ptr<at::BFloat16>()), hidden_size, num_tokens, stream);
    } else {
        TORCH_CHECK(false, "Only float16, bfloat16 are supported");
    }
}