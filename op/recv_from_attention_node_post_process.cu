// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include "../kernel/utils.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

template<typename scalar_t>
__global__ void move_hidden_status_from_aten(const scalar_t* input, scalar_t* output, const int* ori_index, const int* new_index, 
                                    int* valid_idx_size, const int hidden_size, const int batch_size, const int work_count){
    const int bidx = blockIdx.x;
    if(bidx >= valid_idx_size[0]) return;
    const int src_row = ori_index[bidx * 2 + 1];
    const int tgt_row = new_index[bidx * 2 + 1];
    const int tgt_batch_idx = new_index[bidx * 2];
    const int out_batch_offset = tgt_batch_idx * batch_size * work_count * hidden_size;
    const int output_off = out_batch_offset + tgt_row * hidden_size;
    const int input_off = src_row * hidden_size;

    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        output[output_off + idx] = input[input_off + idx];
    }
}

// __forceinline__ __device__ bool idx_is_valid(int idx, const int batch_size, const int topk, const int* token_count) {
//     int work_idx = idx / (batch_size * topk);
//     int valid_size = token_count[work_idx];
//     int token_idx_in_work = (idx - work_idx * batch_size * topk) / topk;
//     return (token_idx_in_work < valid_size);
// }

// static __device__ __managed__ int valid_index_size = 0;

template<typename scalar_t>
__global__ void aten_ffn_depart_index_set_kernel(const int* topk_idx, const float* topk_weights, int* ori_index, 
                                int* new_index, int* expert_cnt, float* deepep_topk_weights, int* valid_idx_size, const int begin_expert_id, const int num_local_experts,
                                const int batch_size, const int topk, const int work_count) {
    extern __shared__ int smem[];
    int block_dim = blockDim.x;
    int* shared_expert_cnt = smem;
    int* shared_expert_start = shared_expert_cnt + num_local_experts;
    __shared__ int warp_1;
    const int vpt = (batch_size * work_count * topk + block_dim - 1) / block_dim;
    const int target_expert_id = blockIdx.x + begin_expert_id;
    int thread_expert_count = 0;
    int cum_sum = 0;
    int cum_sum_before = 0;

    for (int idx = threadIdx.x; idx < num_local_experts; idx += blockDim.x) {
        shared_expert_cnt[idx] = 0;
    }
    if (threadIdx.x == 0) {
        shared_expert_start[0] = 0;
    }
    __syncthreads();
    
    #pragma unroll
    for (int idx = threadIdx.x * vpt; 
            idx < batch_size * work_count * topk && idx < (threadIdx.x + 1) * vpt; 
            ++idx) {
        const int expert_id = topk_idx[idx];
        // bool is_valid = idx_is_valid(idx, batch_size, topk, token_count);
        if (expert_id >= begin_expert_id && expert_id < begin_expert_id + num_local_experts) {
            atomicAdd(shared_expert_cnt + expert_id - begin_expert_id, 1);
            if (expert_id == target_expert_id) {
                ++thread_expert_count;
            }
        }
    }
    __syncthreads();

    const int vpt_num_experts = (num_local_experts + block_dim - 1) / block_dim;
    #pragma unroll
    for (int idx = 0; idx < vpt_num_experts; ++idx) {
        if (threadIdx.x * vpt_num_experts + idx < num_local_experts) {
            cum_sum += shared_expert_cnt[threadIdx.x * vpt_num_experts + idx];
        }
    }

    int test = ScanBlock(cum_sum, block_dim);
    cum_sum_before = test - cum_sum;
    cum_sum = 0;

    #pragma unroll
    for (int idx = 0; idx < vpt_num_experts; ++idx) {
        if (threadIdx.x * vpt_num_experts + idx < num_local_experts) {
           int res = cum_sum + shared_expert_cnt[threadIdx.x * vpt_num_experts + idx] + cum_sum_before;
           shared_expert_start[threadIdx.x * vpt_num_experts + idx + 1] = res;
           cum_sum += shared_expert_cnt[threadIdx.x * vpt_num_experts + idx];
        }
    }
    __syncthreads();
    int thread_end = ScanWarp64(thread_expert_count);
    if (threadIdx.x == 63) warp_1 = thread_end;
    __syncthreads();
    if (threadIdx.x > 63) thread_end += warp_1;

    if(blockIdx.x == 0) {
        for (int idx = threadIdx.x; idx < num_local_experts; idx += block_dim) {
            expert_cnt[idx] = shared_expert_cnt[idx];
        }
        if (threadIdx.x == 0) {
            // valid_index_size = shared_expert_start[num_local_experts];
            valid_idx_size[0] = shared_expert_start[num_local_experts];
        }
    }

    int thread_start = thread_end - thread_expert_count;
    int writen_idx = 0;
    
    #pragma unrolls
    for (int idx = threadIdx.x * vpt; 
        idx < batch_size * work_count * topk && idx < (threadIdx.x + 1) * vpt; 
        ++idx) {
        const int expert_id = topk_idx[idx];
        // bool is_valid = idx_is_valid(idx, batch_size, topk, token_count);
        if (expert_id == target_expert_id) {
            const int expert_start = shared_expert_start[blockIdx.x];
                // printf("EEEEEEEEEE    threadIdx.x = %d, blockIdx.x = %d, thread_start = %d, expert_start = %d, writen_idx = %d, expert_id = %d, idx = %d\n", threadIdx.x, blockIdx.x, thread_start, expert_start, writen_idx, expert_id, idx);
                // printf("thread_end = %d, thread_expert_count = %d, thread_start = %d, block_dim = %d\n", thread_end, thread_expert_count, thread_start, block_dim);
            ori_index[(expert_start + thread_start + writen_idx) * 2] = blockIdx.x;
            ori_index[(expert_start + thread_start + writen_idx) * 2 + 1] = idx / topk;
            deepep_topk_weights[blockIdx.x * batch_size * work_count + thread_start + writen_idx] = 
                topk_weights[idx];
            ++writen_idx;
        }
    }
    
    #pragma unroll
    for (int idx = threadIdx.x; idx < shared_expert_cnt[blockIdx.x]; idx += block_dim) {
        int write_start = shared_expert_start[blockIdx.x] + idx;
        new_index[write_start * 2] = blockIdx.x;
        new_index[write_start * 2 + 1] = idx;
    }
}

template<typename scalar_t>
void launch_recv_from_attention_kernel(const scalar_t* hidden_status, const int* topk_idx, const float* topk_weights, 
                                    int* ori_index, int* new_index, scalar_t* deepep_hidden_status, float* deepep_topk_weights,
                                    int* expert_cnt, int* valid_idx_size, const int work_count, const int num_local_experts, const int batch_size,
                                    const int hidden_size, const int topk, const int begin_expert_id, const int max_index_size, const cudaStream_t& stream) {
    const int block_size = 128;
    const int smem_size = (num_local_experts * 2 + 1) * sizeof(int);
    aten_ffn_depart_index_set_kernel<scalar_t><<<num_local_experts, block_size, smem_size, stream>>>(topk_idx, topk_weights, ori_index, new_index, expert_cnt, deepep_topk_weights,
                                                                            valid_idx_size, begin_expert_id, num_local_experts, batch_size, topk, work_count);
    // cudaStreamSynchronize(stream);
    move_hidden_status_from_aten<scalar_t><<<max_index_size, 512, 0, stream>>>(hidden_status, deepep_hidden_status, ori_index, new_index, valid_idx_size, hidden_size, batch_size, work_count);
}

void recv_from_attention_node_post_process(at::Tensor hidden_status, at::Tensor topk_idx, at::Tensor topk_weights, 
                                        at::Tensor ori_index, at::Tensor new_index, at::Tensor deepep_hidden_status,
                                        at::Tensor deepep_topk_weights,  at::Tensor expert_cnt, at::Tensor valid_idx_size, const int begin_expert_id, 
                                        const int num_local_experts, const int max_index_size, const int work_count) 
{
	DEBUG_TRACE_PARAMS(hidden_status, topk_idx, topk_weights, ori_index, new_index, deepep_hidden_status, deepep_topk_weights, expert_cnt, valid_idx_size, begin_expert_id, num_local_experts, max_index_size, work_count);
	DEBUG_DUMP_PARAMS(hidden_status, topk_idx, topk_weights, ori_index, new_index, deepep_hidden_status, deepep_topk_weights, expert_cnt, valid_idx_size, begin_expert_id, num_local_experts, max_index_size, work_count);
    const int topk = topk_idx.size(-1);
    const int batch_size = topk_idx.numel() / (topk * work_count);
    const int hidden_size = hidden_status.size(-1);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    if (batch_size == 0) return;

    if (hidden_status.dtype() == at::ScalarType::BFloat16) {
        launch_recv_from_attention_kernel<bfloat16>(reinterpret_cast<bfloat16*>(hidden_status.data_ptr<at::BFloat16>()), 
                                                            topk_idx.data_ptr<int>(), topk_weights.data_ptr<float>(), 
                                                            ori_index.data_ptr<int>(), new_index.data_ptr<int>(), reinterpret_cast<bfloat16*>(deepep_hidden_status.data_ptr<at::BFloat16>()),
                                                            deepep_topk_weights.data_ptr<float>(), expert_cnt.data_ptr<int>(), 
                                                            valid_idx_size.data_ptr<int>(), work_count, 
                                                            num_local_experts, batch_size, hidden_size, topk, begin_expert_id, max_index_size, stream);
    } else {
        TORCH_CHECK(false, "Only bfloat16 are supported");
    }
}
