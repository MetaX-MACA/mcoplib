// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../kernel/fused_bias_swiglu_kernel.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"


at::Tensor fused_bias_swiglu_fwd(at::Tensor input) {

	DEBUG_TRACE_PARAMS(input);
	DEBUG_DUMP_PARAMS(input);
    CHECK_DEVICE(input);
    CHECK_DIMS(input, 3);
    CHECK_CONTIGUOUS(input);

    const int stride_in = input.size(2);
    const int stride_out = stride_in / 2;
    const int cols = stride_out;
    const int rows = input.size(0) * input.size(1);

    at::Tensor output = at::empty({input.size(0), input.size(1), cols}, input.options());

    const int tpb = 256;
    dim3 threads = dim3(tpb);
    dim3 blocks = dim3(rows, DIV_UP(cols >> 3, tpb));

    #ifdef USE_MACA
        mcStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #else
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #endif
    // Launch kernel
    if (input.dtype() == at::ScalarType::Half) {
        fused_bias_swiglu_fwd_kernel<half, half2><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<half *>(input.data_ptr<at::Half>()), 
            nullptr, 
            reinterpret_cast<half *>(output.data_ptr<at::Half>()), 
            cols, stride_in, stride_out
        );
    } else if (input.dtype() == at::ScalarType::BFloat16) {
        fused_bias_swiglu_fwd_kernel<bfloat16, bfloat162><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<bfloat16 *>(input.data_ptr<at::BFloat16>()), 
            nullptr, 
            reinterpret_cast<bfloat16 *>(output.data_ptr<at::BFloat16>()), 
            cols, stride_in, stride_out
        );
    } else {
        TORCH_CHECK(false, "Only float16 and bfloat16 are supported");
    }

    return output;
}

at::Tensor fused_bias_swiglu_bwd(at::Tensor input, at::Tensor grad_output) {

	DEBUG_TRACE_PARAMS(input, grad_output);
	DEBUG_DUMP_PARAMS(input, grad_output);
    const int stride_in = input.size(-1);
    const int stride_out = grad_output.size(-1);
    const int numel = grad_output.numel();
    const int cols = grad_output.size(-1);
    const int rows = numel / cols;

    CHECK_DEVICE(input);
    CHECK_DEVICE(grad_output);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(grad_output);

    at::Tensor grad_input = input;

    const int tpb = 256;
    dim3 threads = dim3(tpb);
    dim3 blocks = dim3(rows, DIV_UP(cols >> 3, tpb));

    #ifdef USE_MACA
        mcStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #else
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #endif
    // Launch kernel
    if (input.dtype() == at::ScalarType::Half) {
        fused_bias_swiglu_bwd_kernel<half, half2><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<half *>(input.data_ptr<at::Half>()), 
            nullptr, 
            reinterpret_cast<half *>(grad_output.data_ptr<at::Half>()), 
            reinterpret_cast<half *>(grad_input.data_ptr<at::Half>()), 
            nullptr, 
            cols, stride_in, stride_out
        );
    } else if (input.dtype() == at::ScalarType::BFloat16) {
        fused_bias_swiglu_bwd_kernel<bfloat16, bfloat162><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<bfloat16 *>(input.data_ptr<at::BFloat16>()), 
            nullptr, 
            reinterpret_cast<bfloat16 *>(grad_output.data_ptr<at::BFloat16>()), 
            reinterpret_cast<bfloat16 *>(grad_input.data_ptr<at::BFloat16>()), 
            nullptr, 
            cols, stride_in, stride_out
        );
    } else {
        TORCH_CHECK(false, "Only float16 and bfloat16 are supported");
    }

    return grad_input;
}
