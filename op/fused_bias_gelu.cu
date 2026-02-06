// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include "../kernel/fused_gelu_kernel.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

at::Tensor fused_gelu_fwd(at::Tensor input, at::Tensor bias) {
    DEBUG_TRACE_PARAMS(input, bias);
	DEBUG_DUMP_PARAMS(input, bias);
    CHECK_DEVICE(input);
    at::Tensor output = at::empty_like(input);
    const int numel = input.numel();
    const int hidden_size = input.size(2);
    #ifdef USE_MACA
        mcStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #else
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #endif

    if(input.dtype() == at::ScalarType::Float) {
        int blocks = (numel + 512 * 4 - 1) / (512 * 4);
        if((hidden_size & 3) == 0) {
            fused_gelu_fwd_align_kernel<float, 512><<<blocks, 512, 0, stream>>>(
                reinterpret_cast<float *>(input.data_ptr<float>()), 
                reinterpret_cast<float *>(bias.data_ptr<float>()),
                reinterpret_cast<float *>(output.data_ptr<float>()),
                numel,
                hidden_size
            );
        }else {
            fused_gelu_fwd_kernel<float, 512><<<blocks, 512, 0, stream>>>(
                reinterpret_cast<float *>(input.data_ptr<float>()),
                reinterpret_cast<float *>(bias.data_ptr<float>()), 
                reinterpret_cast<float *>(output.data_ptr<float>()), 
                numel,
                hidden_size
            );
        }
    } else if (input.dtype() == at::ScalarType::Half) {
        int blocks = (numel + 512 * 8 - 1) / (512 * 8);
        if((hidden_size & 7) == 0) {
            fused_gelu_fwd_align_kernel<half, 512><<<blocks, 512, 0, stream>>>(
                reinterpret_cast<half *>(input.data_ptr<at::Half>()),
                reinterpret_cast<half *>(bias.data_ptr<at::Half>()), 
                reinterpret_cast<half *>(output.data_ptr<at::Half>()), 
                numel,
                hidden_size
            );
        }else {
            fused_gelu_fwd_kernel<half, 512><<<blocks, 512, 0, stream>>>(
                reinterpret_cast<half *>(input.data_ptr<at::Half>()),
                reinterpret_cast<half *>(bias.data_ptr<at::Half>()),  
                reinterpret_cast<half *>(output.data_ptr<at::Half>()), 
                numel,
                hidden_size
            );
        }
    } else if (input.dtype() == at::ScalarType::BFloat16) {
        int blocks = (numel + 512 * 8 - 1) / (512 * 8);
        if((hidden_size & 7) == 0) {
            fused_gelu_fwd_align_kernel<bfloat16, 512><<<blocks, 512, 0, stream>>>(
                reinterpret_cast<bfloat16 *>(input.data_ptr<at::BFloat16>()), 
                reinterpret_cast<bfloat16 *>(bias.data_ptr<at::BFloat16>()),
                reinterpret_cast<bfloat16 *>(output.data_ptr<at::BFloat16>()), 
                numel,
                hidden_size
            );
        }else {
            fused_gelu_fwd_kernel<bfloat16, 512><<<blocks, 512, 0, stream>>>(
                reinterpret_cast<bfloat16 *>(input.data_ptr<at::BFloat16>()),
                reinterpret_cast<bfloat16 *>(bias.data_ptr<at::BFloat16>()),
                reinterpret_cast<bfloat16 *>(output.data_ptr<at::BFloat16>()), 
                numel,
                hidden_size
            );
        }
    } else {
        TORCH_CHECK(false, "Only float,float16 and bfloat16 are supported");
    }
    return output;
}

at::Tensor fused_gelu_bwd(at::Tensor input, at::Tensor input1, at::Tensor bias) {
	DEBUG_TRACE_PARAMS(input, input1, bias);
	DEBUG_DUMP_PARAMS(input, input1, bias);
    CHECK_DEVICE(input);
    at::Tensor output = at::empty_like(input);
    const int numel = input.numel();
    const int hidden_size = input.size(2);
    #ifdef USE_MACA
        mcStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #else
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #endif
    if(input.dtype() == at::ScalarType::Float) {
        int blocks = (numel + 512 * 4 - 1) / (512 * 4);
        if((hidden_size & 3) == 0) {
            fused_gelu_bwd_kernel<float, 512><<<blocks, 512, 0, stream>>>(
                reinterpret_cast<float *>(input.data_ptr<float>()), 
                reinterpret_cast<float *>(input1.data_ptr<float>()),
                reinterpret_cast<float *>(bias.data_ptr<float>()),
                reinterpret_cast<float *>(output.data_ptr<float>()), 
                numel,
                hidden_size
            );
        }else {
            fused_gelu_bwd_kernel<float, 512><<<blocks, 512, 0, stream>>>(
                reinterpret_cast<float *>(input.data_ptr<float>()), 
                reinterpret_cast<float*>(input1.data_ptr<float>()),
                reinterpret_cast<float*>(bias.data_ptr<float>()),
                reinterpret_cast<float *>(output.data_ptr<float>()), 
                numel,
                hidden_size
            );
        }
    } else if (input.dtype() == at::ScalarType::Half) {
        int blocks = (numel + 512 * 8 - 1) / (512 * 8);
        if((hidden_size & 7) == 0) {
            fused_gelu_bwd_kernel<half, 512><<<blocks, 512, 0, stream>>>(
                reinterpret_cast<half *>(input.data_ptr<at::Half>()), 
                reinterpret_cast<half *>(input1.data_ptr<at::Half>()),
                reinterpret_cast<half *>(bias.data_ptr<at::Half>()),
                reinterpret_cast<half *>(output.data_ptr<at::Half>()), 
                numel,
                hidden_size
            );
        }else {
            fused_gelu_bwd_kernel<half, 512><<<blocks, 512, 0, stream>>>(
                reinterpret_cast<half *>(input.data_ptr<at::Half>()), 
                reinterpret_cast<half*>(input1.data_ptr<at::Half>()),
                reinterpret_cast<half*>(bias.data_ptr<at::Half>()),
                reinterpret_cast<half *>(output.data_ptr<at::Half>()), 
                numel,
                hidden_size
            );
        }
    } else if (input.dtype() == at::ScalarType::BFloat16) {
        int blocks = (numel + 512 * 8 - 1) / (512 * 8);
        if((hidden_size & 7) == 0) {
            fused_gelu_bwd_kernel<bfloat16, 512><<<blocks, 512, 0, stream>>>(
                reinterpret_cast<bfloat16 *>(input.data_ptr<at::BFloat16>()), 
                reinterpret_cast<bfloat16*>(input1.data_ptr<at::BFloat16>()),
                reinterpret_cast<bfloat16*>(bias.data_ptr<at::BFloat16>()),
                reinterpret_cast<bfloat16 *>(output.data_ptr<at::BFloat16>()), 
                numel,
                hidden_size
            );
        }else {
            fused_gelu_bwd_kernel<bfloat16, 512><<<blocks, 512, 0, stream>>>(
                reinterpret_cast<bfloat16*>(input.data_ptr<at::BFloat16>()),
                reinterpret_cast<bfloat16*>(input1.data_ptr<at::BFloat16>()), 
                reinterpret_cast<bfloat16*>(bias.data_ptr<at::BFloat16>()),
                reinterpret_cast<bfloat16 *>(output.data_ptr<at::BFloat16>()), 
                numel,
                hidden_size
            );
        }
    } else {
        TORCH_CHECK(false, "Only float float16 and bfloat16 are supported");
    }
    return output;
}