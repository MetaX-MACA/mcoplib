// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../kernel/fused_repeat_kv_kernel.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

at::Tensor fused_repeat_kv_fwd(at::Tensor input, int q_num_head, int kv_num_head, int head_dim) {

    // mixed_x_layer seq, bs, partition, (q_num_head + kv_num_head * 2) * head_dim
    // output        seq, bs, 3, q_num_head, head_dim
    
	DEBUG_TRACE_PARAMS(input, q_num_head, kv_num_head, head_dim);
	DEBUG_DUMP_PARAMS(input, q_num_head, kv_num_head, head_dim);
    CHECK_DEVICE(input);
    CHECK_DIMS(input, 4);
    CHECK_CONTIGUOUS(input);


    const int seq = input.size(0);
    const int bs = input.size(1);
    const int partition = input.size(2);

    
    at::Tensor output = at::empty({seq, bs, 3, q_num_head, head_dim}, input.options());
    #ifdef USE_MACA
        mcStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #else
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #endif

    const int element_num = 8; 
    const int block_num = seq;
    const int thread_num = kv_num_head * head_dim / element_num;

    if(partition == kv_num_head) {
        int num_elems = 3 * q_num_head * head_dim;
        int inum_elems = (q_num_head + 2*kv_num_head) * head_dim;
        dim3 gridSize((num_elems + 4095)/(4096),seq*bs,1);
        if (input.dtype() == at::ScalarType::Half) {
            fused_repeat_kv_fwd_kernel_opt<half, 512><<<gridSize, 512, 0, stream>>>(
                reinterpret_cast<half *>(input.data_ptr<at::Half>()), 
                reinterpret_cast<half *>(output.data_ptr<at::Half>()),
                q_num_head, kv_num_head, head_dim, num_elems,inum_elems
            );
        } else if (input.dtype() == at::ScalarType::BFloat16) {
            fused_repeat_kv_fwd_kernel_opt<bfloat16, 512><<<gridSize, 512, 0, stream>>>(
                reinterpret_cast<bfloat16 *>(input.data_ptr<at::BFloat16>()), 
                reinterpret_cast<bfloat16 *>(output.data_ptr<at::BFloat16>()),
                q_num_head, kv_num_head, head_dim, num_elems,inum_elems
            );
        } else {
            TORCH_CHECK(false, "Only float16 and bfloat16 are supported");
        }
        return output;  
    }

    dim3 threads = dim3(thread_num);
    dim3 blocks = dim3(seq, bs, 1);
    if (input.dtype() == at::ScalarType::Half) {
        fused_repeat_kv_fwd_kernel<half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<half *>(input.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(output.data_ptr<at::Half>()),
            seq, bs, partition, q_num_head, kv_num_head, head_dim
        );
    } else if (input.dtype() == at::ScalarType::BFloat16) {
        fused_repeat_kv_fwd_kernel<bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<bfloat16 *>(input.data_ptr<at::BFloat16>()), 
            reinterpret_cast<bfloat16 *>(output.data_ptr<at::BFloat16>()),
            seq, bs, partition, q_num_head, kv_num_head, head_dim
        );
    } else {
        TORCH_CHECK(false, "Only float16 and bfloat16 are supported");
    }
    return output;

}



at::Tensor fused_repeat_kv_bwd(at::Tensor input, int q_num_head, int kv_num_head, int partition) {

    // input    seq, bs, partition, (q_num_head + kv_num_head * 2) * head_dim
    // output   seq, bs, 3, q_num_head, head_dim
	DEBUG_TRACE_PARAMS(input, q_num_head, kv_num_head, partition);
	DEBUG_DUMP_PARAMS(input, q_num_head, kv_num_head, partition);
    CHECK_DEVICE(input);
    CHECK_DIMS(input, 5);
    CHECK_CONTIGUOUS(input);


    const int seq = input.size(0);
    const int bs = input.size(1);
    const int head_dim = input.size(4);

    int repeat = q_num_head / kv_num_head;
    
    at::Tensor output = at::empty({seq, bs, partition, (q_num_head + kv_num_head * 2) * head_dim / partition}, input.options());
    #ifdef USE_MACA
        mcStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #else
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #endif

    const int element_num = 8; 
    const int block_num = seq;
    const int thread_num = kv_num_head * head_dim / element_num;

    dim3 threads = dim3(thread_num);
    dim3 blocks = dim3(seq, bs, 1);
    
    if (input.dtype() == at::ScalarType::Half) {
        fused_repeat_kv_bwd_kernel<half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<half *>(input.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(output.data_ptr<at::Half>()),
            seq, bs, partition, q_num_head, kv_num_head, head_dim
        );
    } else if (input.dtype() == at::ScalarType::BFloat16) {
        fused_repeat_kv_bwd_kernel<bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<bfloat16 *>(input.data_ptr<at::BFloat16>()), 
            reinterpret_cast<bfloat16 *>(output.data_ptr<at::BFloat16>()),
            seq, bs, partition, q_num_head, kv_num_head, head_dim
        );
    } else {
        TORCH_CHECK(false, "Only float16 and bfloat16 are supported");
    }
    return output;

}