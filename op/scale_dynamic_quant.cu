// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include "../kernel/utils.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float blockReduceMax(float val) {
    constexpr size_t WARP_SIZE = 32;
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    val = warpReduceMax(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        val = shared[lane];
    } else {
        val = 0;
    }
    
    if (wid == 0) val = warpReduceMax(val);
    
    return val;
}


template<typename T1, typename T2>
__global__ void scale_dynamic_quant_bfloat16_int8(const  T1* hidden_states, const float* smooth_scales,
                                                size_t hidden_size,
                                                T2* output, float* scales) {
    const int BLOCK_SIZE = blockDim.x;
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float shared_max;
    float local_max = 0.0f;
    const T1* token_hidden = hidden_states + token_idx * hidden_size;
    T2* token_output = output + token_idx * hidden_size;
    const int elements_per_thread = 2;
    const int stride = BLOCK_SIZE * elements_per_thread;
    int idx = tid * elements_per_thread;

    for (; idx + 1 < hidden_size; idx += stride) {
        float2 smooth2 = *reinterpret_cast<const float2*>(&smooth_scales[idx]);
        float val0 = __bfloat162float(token_hidden[idx]);
        float val1 = __bfloat162float(token_hidden[idx + 1]);
        val0 *= smooth2.x;
        val1 *= smooth2.y;
        float max_val = fmaxf(fabsf(val0), fabsf(val1));
        local_max = fmaxf(local_max, max_val);
    }

    if (idx < hidden_size) {
        float smooth_factor = smooth_scales[idx];
        float val = __bfloat162float(token_hidden[idx]);
        val *= smooth_factor;
        local_max = fmaxf(local_max, fabsf(val));
    }
    local_max = blockReduceMax(local_max);
    if (tid == 0) {
        shared_max = local_max;
        scales[token_idx] = shared_max * 0.00787401574803f; // 1/127.0f
    }
    __syncthreads();

    const float inv_scale = 127.0f / shared_max;
    idx = tid * elements_per_thread;
    for (; idx + 1 < hidden_size; idx += stride) {
        float2 smooth2 = *reinterpret_cast<const float2*>(&smooth_scales[idx]);
        float val0 = __bfloat162float(token_hidden[idx]);
        float val1 = __bfloat162float(token_hidden[idx + 1]);
        val0 *= smooth2.x;
        val1 *= smooth2.y;
        int2 quantized;
        quantized.x = __float2int_rn(val0 * inv_scale);
        quantized.y = __float2int_rn(val1 * inv_scale);
        quantized.x = min(127, max(-128, quantized.x));
        quantized.y = min(127, max(-128, quantized.y));
        token_output[idx] = (int8_t)quantized.x;
        token_output[idx + 1] = (int8_t)quantized.y;
    }

    if (idx < hidden_size) {
        float smooth_factor = smooth_scales[idx];
        float val = __bfloat162float(token_hidden[idx]);
        val *= smooth_factor;
        int quantized = __float2int_rn(val * inv_scale);
        quantized = min(127, max(-128, quantized));
        token_output[idx] = (int8_t)quantized;
    }
}

template<typename T1, typename T2>
void launch_scale_dynamic_quant(const T1* hidden_states, const float* smooth_scales,
                                size_t hidden_size, size_t token_num,
                                T2* output, float* scales,
                                const cudaStream_t stream) {
    constexpr size_t BLOCK = 256;
    dim3 grid = token_num, block = BLOCK;
    if (hidden_size <= 1024) {
        block.x = 128;
    } else if (hidden_size <= 4096) {
        block.x = 256;
    } else {
        block.x = 512;
    }
    
    size_t shared_mem_size = sizeof(float) * (block.x < hidden_size ? block.x : hidden_size);
    scale_dynamic_quant_bfloat16_int8<<<grid, block, shared_mem_size, stream>>>(hidden_states, smooth_scales,
                                                                hidden_size,
                                                                output, scales);
}

std::tuple<at::Tensor, at::Tensor> scale_dynamic_quant(
    const at::Tensor& hidden_states,
    const at::Tensor& smooth_scales,
    at::ScalarType dst_dtype = at::ScalarType::Char
) {
	DEBUG_TRACE_PARAMS(hidden_states, smooth_scales, dst_dtype);
	DEBUG_DUMP_PARAMS(hidden_states, smooth_scales, dst_dtype);
    CHECK_DEVICE(hidden_states);
    CHECK_DEVICE(smooth_scales);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const size_t token_num = hidden_states.numel() / hidden_states.size(-1);
    const size_t hidden_size = hidden_states.size(-1);
    at::Tensor output = at::empty_like(hidden_states, hidden_states.options().dtype(dst_dtype));
    at::Tensor scales = at::empty({hidden_states.numel() / hidden_states.size(-1)}, smooth_scales.options());
    if (token_num != hidden_states.numel() / hidden_size) {
        TORCH_CHECK(false, "token count doesn't match smooth scale count");
    }
    if (hidden_states.dtype() == at::ScalarType::BFloat16 && dst_dtype == at::ScalarType::Char) {
        launch_scale_dynamic_quant(reinterpret_cast<bfloat16*>(hidden_states.data_ptr<at::BFloat16>()), smooth_scales.data_ptr<float>(), hidden_size, token_num,
                                   output.data_ptr<int8_t>(), scales.data_ptr<float>(),
                                   stream);
    } else {
        TORCH_CHECK(false, "Only support bfloat16");
    }
    return std::make_tuple(output, scales);
}