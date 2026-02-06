// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include "../kernel/utils.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

inline __device__ int get_batch_idx(const int64_t* accum_q_lens, int64_t bid, const int batch_size) {
    #pragma unroll
    for (int i = 0; i < batch_size; ++i) {
        if (bid < accum_q_lens[i + 1]) {
            return i;
        }
    }
}

template<typename T>
static __device__ __forceinline__ float convert_to_float(T value) {
    return float(value);
}

template<>
static __device__ __forceinline__ float convert_to_float<maca_bfloat16>(maca_bfloat16 value) {
    return __bfloat162float(value);
}

template<>
static __device__ __forceinline__ float convert_to_float<half>(half value) {
    return __half2float(value);
}

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

template<typename scalar_t>
__global__ void apply_rotary_pos_emb_forward_kernel(
    const scalar_t* input, 
    const scalar_t* sin, 
    const scalar_t* cos, 
    scalar_t* out, 
    int head_dim,
    int num_heads,
    int skip_head_dim,
    const int64_t* cumsum_len,
    int batch_size
) {
    int length = num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= length) return;
    int head_dim_rope = head_dim - skip_head_dim;
    int head_dim_half = head_dim_rope / 2;
    
    int batch = 0;
    batch = get_batch_idx(cumsum_len, blockIdx.y, batch_size);

    int local_seq = blockIdx.y - cumsum_len[batch];
    int head_id, head_offset;
    head_id = idx / head_dim;
    head_offset = idx % head_dim;

    int64_t batch_offset = blockIdx.y * length;
    const scalar_t * ptr_input = input + batch_offset;
    scalar_t reg_in = ptr_input[idx];
    float value_in = convert_to_float<scalar_t>(reg_in); 
    scalar_t* ptr_out = out + batch_offset;

    if(head_offset >= skip_head_dim) {
        float rotated;
        if (head_offset - skip_head_dim < head_dim_half) {
            rotated = -value_in;
        } else {
            rotated = value_in;
        }
        ptr_out[idx] = float_to_dstT<scalar_t>(value_in * (convert_to_float<scalar_t>(cos[local_seq * head_dim_rope + head_offset - skip_head_dim])) + 
            rotated * (convert_to_float<scalar_t>(sin[local_seq * head_dim_rope + head_offset - skip_head_dim])));
    } else {
        ptr_out[idx] = reg_in;
    }
}

template<typename scalar_t>
__global__ void apply_rotary_pos_emb_backward_kernel(
    const scalar_t* input, 
    const scalar_t* sin, 
    const scalar_t* cos, 
    scalar_t* out, 
    int head_dim,
    int num_heads,
    int skip_head_dim,
    const int64_t* cumsum_len,
    int batch_size
) {
    int length = num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= length) return;
    int head_dim_rope = head_dim - skip_head_dim;
    int head_dim_half = head_dim_rope / 2;
    
    int batch = 0;
    batch = get_batch_idx(cumsum_len, blockIdx.y, batch_size);

    int local_seq = blockIdx.y - cumsum_len[batch];
    int head_id, head_offset;
    head_id = idx / head_dim;
    head_offset = idx % head_dim;

    int64_t batch_offset = blockIdx.y * length;
    const scalar_t * ptr_input = input + batch_offset;
    scalar_t reg_in = ptr_input[idx];
    float value_in = convert_to_float<scalar_t>(reg_in); 
    scalar_t* ptr_out = out + batch_offset;

    if(head_offset >= skip_head_dim) {
        float rotated;
        if (head_offset - skip_head_dim < head_dim_half) {
            rotated = value_in;
        } else {
            rotated = -value_in;
        }
        ptr_out[idx] = float_to_dstT<scalar_t>(value_in * (convert_to_float<scalar_t>(cos[local_seq * head_dim_rope + head_offset - skip_head_dim])) + 
            rotated * (convert_to_float<scalar_t>(sin[local_seq * head_dim_rope + head_offset - skip_head_dim])));
    } else {
        ptr_out[idx] = reg_in;
    }
}

torch::Tensor rotary_pos_emb_forward(
    torch::Tensor input, 
    torch::Tensor sin, 
    torch::Tensor cos, 
    torch::Tensor cumsum_len, 
    int batch_size,
    int cut_head_dim = 0
) {
	DEBUG_TRACE_PARAMS(input, sin, cos, cumsum_len, batch_size, cut_head_dim);
	DEBUG_DUMP_PARAMS(input, sin, cos, cumsum_len, batch_size, cut_head_dim);
    // 检查输入设备是否为CUDA
    TORCH_CHECK(input.device().is_cuda(), "input must be on CUDA");
    TORCH_CHECK(sin.device().is_cuda(), "sin must be on CUDA");
    TORCH_CHECK(cos.device().is_cuda(), "cos must be on CUDA");
    TORCH_CHECK(cumsum_len.device().is_cuda(), "cumsum_len must be on CUDA");
    
    int num_heads = input.size(1);
    int head_dim = input.size(2);
    int skip_head_dim = head_dim - cut_head_dim;
    int total_seq_len = input.size(0);
    auto out = torch::empty_like(input);
    
    TORCH_CHECK(cut_head_dim <= head_dim, "cur_head_dim must less than head_dm");

    int threads = 512;
    dim3 blocks((num_heads*head_dim + threads - 1) / threads, total_seq_len, 1);
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if(input.dtype() == at::ScalarType::BFloat16) {  
        apply_rotary_pos_emb_forward_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<bfloat16*>(sin.data_ptr<at::BFloat16>()),
            reinterpret_cast<bfloat16*>(cos.data_ptr<at::BFloat16>()),
            reinterpret_cast<bfloat16*>(out.data_ptr<at::BFloat16>()),
            head_dim,
            num_heads,
            skip_head_dim,
            cumsum_len.data_ptr<int64_t>(),
            batch_size
        ); 
    } else if(input.dtype() == at::ScalarType::Half) { 
        apply_rotary_pos_emb_forward_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(sin.data_ptr<at::Half>()),
            reinterpret_cast<half*>(cos.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            head_dim,
            num_heads,
            skip_head_dim,
            cumsum_len.data_ptr<int64_t>(),
            batch_size
        ); 
    } else if(input.dtype() == at::ScalarType::Float) {
        apply_rotary_pos_emb_forward_kernel<<<blocks, threads, 0, stream>>>(
            input.data_ptr<float>(),
            sin.data_ptr<float>(),
            cos.data_ptr<float>(),
            out.data_ptr<float>(),
            head_dim,
            num_heads,
            skip_head_dim,
            cumsum_len.data_ptr<int64_t>(),
            batch_size
        );
    } else {
        TORCH_CHECK(0, "rope forward not support this type");
    }
    return out;
}

torch::Tensor rotary_pos_emb_backward(
    torch::Tensor input, 
    torch::Tensor sin, 
    torch::Tensor cos, 
    torch::Tensor cumsum_len, 
    int batch_size,
    int cut_head_dim = 0
) {
    DEBUG_TRACE_PARAMS(input, sin, cos, cumsum_len, batch_size, cut_head_dim);
	DEBUG_DUMP_PARAMS(input, sin, cos, cumsum_len, batch_size, cut_head_dim);
    // 检查输入设备是否为CUDA
    TORCH_CHECK(input.device().is_cuda(), "input must be on CUDA");
    TORCH_CHECK(sin.device().is_cuda(), "sin must be on CUDA");
    TORCH_CHECK(cos.device().is_cuda(), "cos must be on CUDA");
    TORCH_CHECK(cumsum_len.device().is_cuda(), "cumsum_len must be on CUDA");
    
    int num_heads = input.size(1);
    int head_dim = input.size(2);
    int skip_head_dim = head_dim - cut_head_dim;
    int total_seq_len = input.size(0);
    auto out = torch::empty_like(input);
    
    TORCH_CHECK(cut_head_dim <= head_dim, "cur_head_dim must less than head_dm");

    int threads = 512;
    dim3 blocks((num_heads*head_dim + threads - 1) / threads, total_seq_len, 1);
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if(input.dtype() == at::ScalarType::BFloat16) {  
        apply_rotary_pos_emb_backward_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<bfloat16*>(sin.data_ptr<at::BFloat16>()),
            reinterpret_cast<bfloat16*>(cos.data_ptr<at::BFloat16>()),
            reinterpret_cast<bfloat16*>(out.data_ptr<at::BFloat16>()),
            head_dim,
            num_heads,
            skip_head_dim,
            cumsum_len.data_ptr<int64_t>(),
            batch_size
        ); 
    } else if(input.dtype() == at::ScalarType::Half) { 
        apply_rotary_pos_emb_backward_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(sin.data_ptr<at::Half>()),
            reinterpret_cast<half*>(cos.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            head_dim,
            num_heads,
            skip_head_dim,
            cumsum_len.data_ptr<int64_t>(),
            batch_size
        ); 
    } else if(input.dtype() == at::ScalarType::Float) {
        apply_rotary_pos_emb_backward_kernel<<<blocks, threads, 0, stream>>>(
            input.data_ptr<float>(),
            sin.data_ptr<float>(),
            cos.data_ptr<float>(),
            out.data_ptr<float>(),
            head_dim,
            num_heads,
            skip_head_dim,
            cumsum_len.data_ptr<int64_t>(),
            batch_size
        );
    } else {
        TORCH_CHECK(0, "rope forward not support this type");
    }
    return out;
}
