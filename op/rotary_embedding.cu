// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include "../kernel/utils.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    const scalar_t* __restrict__ qkv, scalar_t* __restrict__ arr, const float* __restrict__ cos_ptr,
    const float* __restrict__ sin_ptr, int rot_offset, int embed_dim) {
    int x_index, y_index;
    float cos, sin;

    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = cos_ptr[x_index];
    sin = sin_ptr[x_index];

    const float x = target_to_float<scalar_t>(qkv[x_index]);
    const float y = target_to_float<scalar_t>(qkv[y_index]);

    arr[x_index] = float_to_target<scalar_t>(x * cos - y * sin);
    arr[y_index] = float_to_target<scalar_t>(y * cos + x * sin);
}

inline __device__ int get_batch_idx(const int* accum_q_lens, int bid, const int batch_size) {
    #pragma unroll
    for (int i = 0; i < batch_size; ++i) {
        if (bid < accum_q_lens[i + 1]) {
            return i;
        }
    }
}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const scalar_t* __restrict__ qkv,
    scalar_t* __restrict__ qk_output,        
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    const int* __restrict__ accum_q_lens,
    const int* __restrict__ q_len,
    const int* __restrict__ cache_lens,
    const int batch_size,
    const int q_head_num,
    const int kv_head_num,
    const int rope_offset,
    const int rope_dim,
    const int head_dim,
    const int total_head_num_per_token) {

    const int qk_head_start = 0;
    const int qk_head_end = q_head_num + kv_head_num;
    const int num_heads_per_token = qk_head_end - qk_head_start;

    const int batch_idx = get_batch_idx(accum_q_lens, blockIdx.x, batch_size);
    const int q_len_ = q_len[batch_idx];                // num_tokens_per_batch
    const int q_offset_ = accum_q_lens[batch_idx];
    if (blockIdx.x >= q_len_ + q_offset_) return;

    const int token_idx_in_batch = blockIdx.x - q_offset_;

    const int dim_start = rope_offset;
    const int dim_end = rope_offset + rope_dim * 2;
    const int embed_dim = rope_dim;

    const int cur_cache_len_ = cache_lens[batch_idx];
    const int cache_start = cur_cache_len_;
    const int cache_idx = cache_start + token_idx_in_batch;

    const float* sin_ptr = sin + cache_idx * rope_dim;
    const float* cos_ptr = cos + cache_idx * rope_dim;
    const scalar_t* qkv_ptr = qkv + blockIdx.x * total_head_num_per_token * head_dim;
    scalar_t* qk_output_ptr = qk_output + blockIdx.x * total_head_num_per_token * head_dim;

    for (int head_id = qk_head_start; head_id < qk_head_end; ++head_id) {
        for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
            apply_token_rotary_embedding<scalar_t, IS_NEOX>(qkv_ptr + head_id * head_dim, qk_output_ptr + head_id * head_dim, 
                                                                cos_ptr, sin_ptr, i, embed_dim);
        }
    }

}

template<typename scalar_t, bool is_neox>
void launch_rotary_embedding_kernel(const scalar_t* qkv, scalar_t* output, const int* q_len, const int* accum_q_lens, const int* cache_lens, const float* cos, const float* sin,
                                    const int num_tokens, const int rope_dim, const int rope_offset, const int batch_size, const int q_head_num, const int kv_head_num,
                                    const int head_dim, const int total_head_num, const cudaStream_t& stream){
    const int block_size = min(512, rope_dim);
    rotary_embedding_kernel<scalar_t, is_neox><<<num_tokens, block_size, 0, stream>>>(qkv, output, cos, sin, accum_q_lens, q_len, cache_lens, batch_size, q_head_num, kv_head_num,
                                                                            rope_offset, rope_dim, head_dim, total_head_num);
}

void rotary_embedding(at::Tensor packed_qkv, // [num_tokens, total_head_num, head_dim]
                        at::Tensor q_len, at::Tensor accum_q_lens, at::Tensor cache_lens, at::Tensor cos,
                        at::Tensor sin,  at::Tensor output, const int q_head_num, const int kv_head_num, 
                        const int rope_offset = 0) {
	DEBUG_TRACE_PARAMS(packed_qkv, q_len, accum_q_lens, cache_lens, cos, sin, output, q_head_num, kv_head_num, rope_offset);
	DEBUG_DUMP_PARAMS(packed_qkv, q_len, accum_q_lens, cache_lens, cos, sin, output, q_head_num, kv_head_num, rope_offset);
    const int head_dim = packed_qkv.size(-1);
    const int total_head_num = packed_qkv.size(-2);
    const int num_tokens = packed_qkv.numel() / (head_dim * total_head_num);
    const int batch_size = q_len.numel();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr bool is_neox = true;
    const int rope_dim = cos.size(-1);

    if (packed_qkv.dtype() == at::ScalarType::BFloat16) {
        launch_rotary_embedding_kernel<bfloat16, is_neox>(reinterpret_cast<bfloat16*>(packed_qkv.data_ptr<at::BFloat16>()), reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>()), 
                                                            q_len.data_ptr<int>(), accum_q_lens.data_ptr<int>(), cache_lens.data_ptr<int>(), 
                                                            (cos.data_ptr<float>()), (sin.data_ptr<float>()), 
                                                            num_tokens, rope_dim, rope_offset, batch_size, q_head_num, kv_head_num, head_dim, total_head_num, stream);
    } else {
        TORCH_CHECK(false, "Only float16, bfloat16 are supported");
    }
}