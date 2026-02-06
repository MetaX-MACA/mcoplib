// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../kernel/fused_rope_kernel.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"



at::Tensor fused_rope_fwd(at::Tensor qkv, at::Tensor cos, at::Tensor sin, c10::optional<at::Tensor> indexes, bool force_bf16_attn = false){

    DEBUG_TRACE_PARAMS(qkv, cos,sin,indexes,force_bf16_attn);
    DEBUG_DUMP_PARAMS(qkv, cos,sin,indexes,force_bf16_attn);
    CHECK_DEVICE(qkv);
    CHECK_DEVICE(cos);
    CHECK_DEVICE(sin);
    if (indexes.has_value()) CHECK_DEVICE(indexes.value());
    CHECK_CONTIGUOUS(qkv);
    CHECK_CONTIGUOUS(cos);
    CHECK_CONTIGUOUS(sin);
    if (indexes.has_value()) CHECK_CONTIGUOUS(indexes.value());
    if (indexes.has_value()) CHECK_DTYPE(indexes.value(), at::ScalarType::Long);
    
    int seq = qkv.size(0); // 4096
    int qkv_num = qkv.size(1);
    int num_head = qkv.size(2); // 64
    int head_dim = qkv.size(3); // 128

    int head_dim_half = cos.size(1); // 64

    // This is fixed data
    int element_num = 16;

    int thread_num = num_head * head_dim / element_num;

    #ifdef USE_MACA
        mcStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #else
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #endif
    if (qkv.dtype() == at::ScalarType::Half && force_bf16_attn) {
        fused_rope_fwd_kernel<half, bfloat16><<<dim3(seq), dim3(thread_num), 0, stream>>>(
            reinterpret_cast<half *>(qkv.data_ptr<at::Half>()),
            reinterpret_cast<half *>(cos.data_ptr<at::Half>()),
            reinterpret_cast<half *>(sin.data_ptr<at::Half>()),
            indexes.has_value() ? reinterpret_cast<long *>(indexes.value().data_ptr<int64_t>()) : nullptr,
            reinterpret_cast<bfloat16 *>(qkv.data_ptr<at::Half>()),
            seq, qkv_num, num_head, head_dim, head_dim_half, element_num);
        qkv = qkv.view(at::ScalarType::BFloat16);
    } else if (qkv.dtype() == at::ScalarType::Half) {
        fused_rope_fwd_kernel<half, half><<<dim3(seq), dim3(thread_num), 0, stream>>>(
            reinterpret_cast<half *>(qkv.data_ptr<at::Half>()),
            reinterpret_cast<half *>(cos.data_ptr<at::Half>()),
            reinterpret_cast<half *>(sin.data_ptr<at::Half>()),
            indexes.has_value() ? reinterpret_cast<long *>(indexes.value().data_ptr<int64_t>()) : nullptr,
            reinterpret_cast<half *>(qkv.data_ptr<at::Half>()),
            seq, qkv_num, num_head, head_dim, head_dim_half, element_num);
    } else if (qkv.dtype() == at::ScalarType::BFloat16) {
        fused_rope_fwd_kernel<bfloat16, bfloat16><<<dim3(seq), dim3(thread_num), 0, stream>>>(
            reinterpret_cast<bfloat16 *>(qkv.data_ptr<at::BFloat16>()),
            reinterpret_cast<bfloat16 *>(cos.data_ptr<at::BFloat16>()),
            reinterpret_cast<bfloat16 *>(sin.data_ptr<at::BFloat16>()),
            indexes.has_value() ? reinterpret_cast<long *>(indexes.value().data_ptr<int64_t>()) : nullptr,
            reinterpret_cast<bfloat16 *>(qkv.data_ptr<at::BFloat16>()),
            seq, qkv_num, num_head, head_dim, head_dim_half, element_num);
    } else {
            TORCH_CHECK(false, "Only float16 and bfloat16 are supported");
    }

    return qkv;
}


at::Tensor fused_rope_bwd(at::Tensor qkv, at::Tensor cos, at::Tensor sin, c10::optional<at::Tensor> indexes, bool force_bf16_attn = false){

    DEBUG_TRACE_PARAMS(qkv, cos,sin,indexes,force_bf16_attn);
    DEBUG_DUMP_PARAMS(qkv, cos,sin,indexes,force_bf16_attn);
    CHECK_DEVICE(qkv);
    CHECK_DEVICE(cos);
    CHECK_DEVICE(sin);
    if (indexes.has_value()) CHECK_DEVICE(indexes.value());
    CHECK_CONTIGUOUS(qkv);
    CHECK_CONTIGUOUS(cos);
    CHECK_CONTIGUOUS(sin);
    if (indexes.has_value()) CHECK_CONTIGUOUS(indexes.value());
    if (indexes.has_value()) CHECK_DTYPE(indexes.value(), at::ScalarType::Long);
    
    int seq = qkv.size(0); // 4096
    int qkv_num = qkv.size(1);
    int num_head = qkv.size(2); // 64
    int head_dim = qkv.size(3); // 128

    int head_dim_half = cos.size(1); // 64

    // This is fixed data
    int element_num = 16;

    int thread_num = num_head * head_dim / element_num;

    #ifdef USE_MACA
        mcStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #else
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    #endif
    if (qkv.dtype() == at::ScalarType::BFloat16 && force_bf16_attn && cos.dtype() == at::ScalarType::Half) {
        qkv = qkv.view(at::ScalarType::Half);
        fused_rope_bwd_kernel<half, bfloat16><<<dim3(seq), dim3(thread_num), 0, stream>>>(
            reinterpret_cast<bfloat16 *>(qkv.data_ptr<at::Half>()),
            reinterpret_cast<half *>(cos.data_ptr<at::Half>()),
            reinterpret_cast<half *>(sin.data_ptr<at::Half>()),
            indexes.has_value() ? reinterpret_cast<long *>(indexes.value().data_ptr<int64_t>()) : nullptr,
            reinterpret_cast<half *>(qkv.data_ptr<at::Half>()),
            seq, qkv_num, num_head, head_dim, head_dim_half, element_num);
    } else if (qkv.dtype() == at::ScalarType::Half) {
        fused_rope_bwd_kernel<half><<<dim3(seq), dim3(thread_num), 0, stream>>>(
            reinterpret_cast<half *>(qkv.data_ptr<at::Half>()),
            reinterpret_cast<half *>(cos.data_ptr<at::Half>()),
            reinterpret_cast<half *>(sin.data_ptr<at::Half>()),
            indexes.has_value() ? reinterpret_cast<long *>(indexes.value().data_ptr<int64_t>()) : nullptr,
            reinterpret_cast<half *>(qkv.data_ptr<at::Half>()),
            seq, qkv_num, num_head, head_dim, head_dim_half, element_num);
    } else if (qkv.dtype() == at::ScalarType::BFloat16) {
        fused_rope_bwd_kernel<bfloat16><<<dim3(seq), dim3(thread_num), 0, stream>>>(
            reinterpret_cast<bfloat16 *>(qkv.data_ptr<at::BFloat16>()),
            reinterpret_cast<bfloat16 *>(cos.data_ptr<at::BFloat16>()),
            reinterpret_cast<bfloat16 *>(sin.data_ptr<at::BFloat16>()),
            indexes.has_value() ? reinterpret_cast<long *>(indexes.value().data_ptr<int64_t>()) : nullptr,
            reinterpret_cast<bfloat16 *>(qkv.data_ptr<at::BFloat16>()),
            seq, qkv_num, num_head, head_dim, head_dim_half, element_num);
    } else {
            TORCH_CHECK(false, "Only float16 and bfloat16 are supported");
    }

    return qkv;
}