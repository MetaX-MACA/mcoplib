// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include "../kernel/utils.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"


__global__ void set_zero_kernel(uint4* output, const int total_vec){
    constexpr uint4 res = {0, 0, 0, 0};
    const int offset = blockIdx.x * total_vec;
    for (int idx = threadIdx.x; idx < total_vec; idx += 512) {
        output[offset + idx] = res;
    }
}

template<typename scalar_t>
__global__ void move_hidden_status_from_ffn(const scalar_t* moe_hidden_status, const float* deepep_topk_weights, const int* ori_index, const int* new_index, scalar_t* output, const int* valid_idx_size,
                                                const int num_tokens, const int hidden_size, const int num_local_experts)
{
    const int bidx = blockIdx.x;
    if(bidx >= valid_idx_size[0]) return;
    const int block_dim = blockDim.x;
    const int src_batch_idx = new_index[bidx * 2];
    const int src_row_idx = new_index[bidx * 2 + 1];
    const int tar_row_idx = ori_index[bidx * 2 + 1];

    const scalar_t weights = float_to_target<scalar_t>(deepep_topk_weights[src_batch_idx * num_tokens + src_row_idx]);
    const int tar_offset = tar_row_idx * hidden_size;
    const int input_offset = (src_batch_idx * num_tokens + src_row_idx) * hidden_size;

    for (int idx = threadIdx.x; idx < hidden_size; idx += block_dim) {
        atomicAdd(output + tar_offset + idx, moe_hidden_status[input_offset + idx] * weights);
    }
}

template<typename scalar_t>
void launch_send_to_atten_kernel(const scalar_t* moe_hidden_status, const float* deepep_topk_weights, const int* ori_index, const int* new_index, scalar_t* output,
                                    int* valid_idx_size, const int num_tokens, const int hidden_size, const int num_local_experts, const int max_index_size,
                                    const int output_row, const int output_col, const cudaStream_t& stream){
    const int block_size = min(512, hidden_size);
    const int grid_size = max_index_size;
    set_zero_kernel<<<output_row, 512, 0, stream>>>((uint4*)output, (hidden_size * sizeof(scalar_t))/ sizeof(uint4));
    move_hidden_status_from_ffn<scalar_t><<<grid_size, block_size, 0, stream>>>(moe_hidden_status, deepep_topk_weights, ori_index, new_index, (scalar_t*)output, valid_idx_size,
                                                                                    num_tokens, hidden_size, num_local_experts);
}





void send_to_attention_node_pre_process(at::Tensor moe_hidden_status, at::Tensor deepep_topk_weights, at::Tensor ori_index,
                                            at::Tensor new_index, at::Tensor output, at::Tensor valid_idx_size, const int max_index_size)
{
	DEBUG_TRACE_PARAMS(moe_hidden_status, deepep_topk_weights, ori_index, new_index, output, valid_idx_size, max_index_size);
	DEBUG_DUMP_PARAMS(moe_hidden_status, deepep_topk_weights, ori_index, new_index, output, valid_idx_size, max_index_size);
    const int num_tokens = moe_hidden_status.size(-2);
    const int hidden_size = moe_hidden_status.size(-1);
    const int num_local_experts = moe_hidden_status.numel() / (num_tokens * hidden_size);
    const int output_col = output.size(-1);
    const int output_row = output.numel() / output.size(-1);
    if (num_local_experts == 0) return;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (moe_hidden_status.dtype() == at::ScalarType::BFloat16) {
        launch_send_to_atten_kernel<bfloat16>(reinterpret_cast<bfloat16*>(moe_hidden_status.data_ptr<at::BFloat16>()), deepep_topk_weights.data_ptr<float>(),
                                                            ori_index.data_ptr<int>(), new_index.data_ptr<int>(),
                                                            reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>()), valid_idx_size.data_ptr<int>(),
                                                            num_tokens, hidden_size, num_local_experts, max_index_size, output_row, output_col, stream);
    } else {
        TORCH_CHECK(false, "Only bfloat16 are supported");
    }
}
