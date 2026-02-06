// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../kernel/fused_bias_dropout_kernel.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"


at::Tensor fused_bias_dropout(at::Tensor input, 
                                at::Tensor residual, 
                                float dropout_prob) {
    DEBUG_TRACE_PARAMS(input, residual,dropout_prob);
    DEBUG_DUMP_PARAMS(input, residual,dropout_prob);
    CHECK_DEVICE(input);
    CHECK_DEVICE(residual);

    at::Tensor output = at::empty_like(input);

    const int64_t numel = input.numel();

    const int threads = 512;

    const int64_t blocks = DIV_UP(numel, threads*8);
    #ifdef USE_MACA
        mcStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #else
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #endif

    // Launch kernel
    if (dropout_prob == 0.0f) {
        if (input.dtype() == at::ScalarType::Half) {
            if((numel & 7) == 0) {
                fused_bias_add_kernel<half, half2><<<blocks, threads, 0, stream>>>(
                    reinterpret_cast<half *>(input.data_ptr<at::Half>()), 
                    nullptr, 
                    reinterpret_cast<half *>(residual.data_ptr<at::Half>()), 
                    reinterpret_cast<half *>(output.data_ptr<at::Half>()), 
                    dropout_prob, 
                    numel
                );
            } else {
                fused_bias_add_kernel_na<half, half2><<<blocks, threads, 0, stream>>>(
                    reinterpret_cast<half *>(input.data_ptr<at::Half>()), 
                    nullptr, 
                    reinterpret_cast<half *>(residual.data_ptr<at::Half>()), 
                    reinterpret_cast<half *>(output.data_ptr<at::Half>()), 
                    dropout_prob, 
                    numel
                );
            }
        } else if (input.dtype() == at::ScalarType::BFloat16) {
            if((numel & 7) == 0) {
                fused_bias_add_kernel<bfloat16, bfloat162><<<blocks, threads, 0, stream>>>(
                    reinterpret_cast<bfloat16 *>(input.data_ptr<at::BFloat16>()), 
                    nullptr, 
                    reinterpret_cast<bfloat16 *>(residual.data_ptr<at::BFloat16>()), 
                    reinterpret_cast<bfloat16 *>(output.data_ptr<at::BFloat16>()), 
                    dropout_prob, 
                    numel
                );
            } else {
                fused_bias_add_kernel_na<bfloat16, bfloat162><<<blocks, threads, 0, stream>>>(
                    reinterpret_cast<bfloat16 *>(input.data_ptr<at::BFloat16>()), 
                    nullptr, 
                    reinterpret_cast<bfloat16 *>(residual.data_ptr<at::BFloat16>()), 
                    reinterpret_cast<bfloat16 *>(output.data_ptr<at::BFloat16>()), 
                    dropout_prob, 
                    numel
                );
            }
        } else {
            TORCH_CHECK(false, "Only float16 and bfloat16 are supported");
        }
    } else {
        TORCH_CHECK(false, "Dropout not implemented yet");
    }

    return output;
}
