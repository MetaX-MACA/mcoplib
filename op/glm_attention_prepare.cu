#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "../kernel/dispatch_utils.h"
#include "../kernel/utils.h"
#include "../kernel/utils.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template<typename scalar_t>
__device__ void compute_sin_cos(
    const int64_t pos,
    const int emb_pos,
    float base,
    int rotary_dim,
    scalar_t* cos_ptr,
    scalar_t* sin_ptr
) {
    float exponent = 2.0f * emb_pos / static_cast<float>(rotary_dim);
    float inv_freq = 1.0f / powf(base, exponent);
    float freqs = float(pos) * inv_freq;
    *cos_ptr = static_cast<scalar_t>(cosf(freqs));
    *sin_ptr = static_cast<scalar_t>(sinf(freqs));
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    const int64_t pos,
    const float base,
    scalar_t* __restrict__ x_ptr,
    scalar_t* __restrict__ y_ptr,
    int rot_offset,
    int embed_dim,
    int do_idx) {

    scalar_t cos, sin;
    if constexpr (IS_NEOX) {
        // GPT-NeoX style rotary embedding.
        compute_sin_cos<scalar_t>(pos, rot_offset, base, embed_dim * 2, &cos, &sin);   // rotary_dim = emb_dim * 2
    } 

    const scalar_t x = x_ptr[do_idx];
    const scalar_t y = y_ptr[do_idx];
    x_ptr[do_idx] = x * cos - y * sin;
    y_ptr[do_idx] = y * cos + x * sin;

    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("do_idx = %d\n", do_idx);
    //     printf("x = %f\n", float(x));
    //     printf("y = %f\n", float(y));
    //     printf("cos = %f\n", float(cos));
    //     printf("sin = %f\n", float(sin));
    //     printf("res 0 = %f\n", float(x_ptr[do_idx]));
    //     printf("res 1 = %f\n", float(y_ptr[do_idx]));
    // }
}

template <typename scalar_t, int cg_size, int vec_size>
inline __device__ void apply_rotary_embedding(
    cg::thread_block_tile<cg_size> group,
    scalar_t* __restrict__ query,  // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,    // nullptr or
                                   // [batch_size, seq_len, num_kv_heads,
                                   // head_size] or [num_tokens, num_kv_heads,
                                   // head_size]
    scalar_t* __restrict__ out_q,
    scalar_t* __restrict__ out_k,
    const scalar_t* __restrict__ weight,
    const float q_rms,
    const float k_rms,
    const float base,
    const int64_t pos,
    const int head_idx,
    const int num_kv_heads, 
    const int rot_dim) {

    scalar_t bf_chunk[vec_size];
    scalar_t af_chunk[vec_size];
    scalar_t bw_chunk[vec_size];
    scalar_t aw_chunk[vec_size];

    using AccessType = AlignedArrayI4<scalar_t, vec_size>;
    AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&bf_chunk);
    AccessType* vec_thread_read_ptr = reinterpret_cast<AccessType*>(query);
    *row_chunk_vec_ptr = vec_thread_read_ptr[group.thread_rank()];

    row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&af_chunk);
    *row_chunk_vec_ptr = vec_thread_read_ptr[group.thread_rank() + group.num_threads()];

    AccessType* row_weight_ptr = reinterpret_cast<AccessType*>(&bw_chunk);
    const AccessType* vec_weight_ptr = reinterpret_cast<const AccessType*>(weight);
    *row_weight_ptr = vec_weight_ptr[group.thread_rank()];

    row_weight_ptr = reinterpret_cast<AccessType*>(&aw_chunk);
    *row_weight_ptr = vec_weight_ptr[group.thread_rank() + group.num_threads()];       

    const int embed_dim = rot_dim / 2;
    #pragma unroll 
    for (int i = 0; i < vec_size; i++) {
        bf_chunk[i] = static_cast<scalar_t>(static_cast<float>(bf_chunk[i]) * q_rms) * bw_chunk[i];
        af_chunk[i] = static_cast<scalar_t>(static_cast<float>(af_chunk[i]) * q_rms) * aw_chunk[i];
    }


    for (int i = 0; i < vec_size; i++) {
        const int rot_offset = group.thread_rank() * vec_size + i;
        apply_token_rotary_embedding<scalar_t, true>(
            pos, base, bf_chunk, af_chunk, rot_offset, embed_dim, i);
    }

    // if (blockIdx.x == 3 && threadIdx.x == 0) {
    //     for (int i = 0; i < vec_size; i++) {
    //         printf("bf_chunk[%d] = %f\n", i, float(bf_chunk[i]));
    //     }
    //     for (int i = 0; i < vec_size; i++) {
    //         printf("af_chunk[%d] = %f\n", i, float(af_chunk[i]));
    //     }
    // }

    vec_thread_read_ptr = reinterpret_cast<AccessType*>(out_q); 
    row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&bf_chunk);
    vec_thread_read_ptr[group.thread_rank()] = *row_chunk_vec_ptr;
    row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&af_chunk);
    vec_thread_read_ptr[group.thread_rank() + group.num_threads()] = *row_chunk_vec_ptr;

    vec_thread_read_ptr = reinterpret_cast<AccessType*>(query); 
    row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&bf_chunk);
    *row_chunk_vec_ptr = vec_thread_read_ptr[group.thread_rank() + group.num_threads() * 2];
    row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&af_chunk);
    *row_chunk_vec_ptr = vec_thread_read_ptr[group.thread_rank() + group.num_threads() * 3];

    row_weight_ptr = reinterpret_cast<AccessType*>(&bw_chunk);
    *row_weight_ptr = vec_weight_ptr[group.thread_rank() + group.num_threads() * 2];
    row_weight_ptr = reinterpret_cast<AccessType*>(&aw_chunk);
    *row_weight_ptr = vec_weight_ptr[group.thread_rank() + group.num_threads() * 3];

    #pragma unroll 
    for (int i = 0; i < vec_size; i++) {
        bf_chunk[i] = static_cast<scalar_t>(static_cast<float>(bf_chunk[i]) * q_rms) * bw_chunk[i];
        af_chunk[i] = static_cast<scalar_t>(static_cast<float>(af_chunk[i]) * q_rms) * aw_chunk[i];
    }

    vec_thread_read_ptr = reinterpret_cast<AccessType*>(out_q); 
    row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&bf_chunk);
    vec_thread_read_ptr[group.thread_rank() + group.num_threads() * 2] = *row_chunk_vec_ptr;
    row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&af_chunk);
    vec_thread_read_ptr[group.thread_rank() + group.num_threads() * 3] = *row_chunk_vec_ptr;

    if (head_idx < num_kv_heads) {
        vec_thread_read_ptr = reinterpret_cast<AccessType*>(key);

        row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&bf_chunk);
        *row_chunk_vec_ptr = vec_thread_read_ptr[group.thread_rank()];

        row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&af_chunk);
        *row_chunk_vec_ptr = vec_thread_read_ptr[group.thread_rank() + group.num_threads()];

        row_weight_ptr = reinterpret_cast<AccessType*>(&bw_chunk);
        *row_weight_ptr = vec_weight_ptr[group.thread_rank() + group.num_threads() * 0];
        row_weight_ptr = reinterpret_cast<AccessType*>(&aw_chunk);
        *row_weight_ptr = vec_weight_ptr[group.thread_rank() + group.num_threads() * 1];

        #pragma unroll 
        for (int i = 0; i < vec_size; i++) {
            bf_chunk[i] = static_cast<scalar_t>(static_cast<float>(bf_chunk[i]) * k_rms) * bw_chunk[i];
            af_chunk[i] = static_cast<scalar_t>(static_cast<float>(af_chunk[i]) * k_rms) * aw_chunk[i];
        }

        for (int i = 0; i < vec_size; i++) {
            const int rot_offset = (group.thread_rank() * vec_size + i);
            apply_token_rotary_embedding<scalar_t, true>(
                pos, base, bf_chunk, af_chunk, rot_offset, embed_dim, i);
        }

        vec_thread_read_ptr = reinterpret_cast<AccessType*>(out_k); 
        row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&bf_chunk);
        vec_thread_read_ptr[group.thread_rank()] = *row_chunk_vec_ptr;
        row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&af_chunk);
        vec_thread_read_ptr[group.thread_rank() + group.num_threads()] = *row_chunk_vec_ptr;

        vec_thread_read_ptr = reinterpret_cast<AccessType*>(key); 
        row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&bf_chunk);
        *row_chunk_vec_ptr = vec_thread_read_ptr[group.thread_rank() + group.num_threads() * 2];
        row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&af_chunk);
        *row_chunk_vec_ptr = vec_thread_read_ptr[group.thread_rank() + group.num_threads() * 3];

        row_weight_ptr = reinterpret_cast<AccessType*>(&bw_chunk);
        *row_weight_ptr = vec_weight_ptr[group.thread_rank() + group.num_threads() * 2];
        row_weight_ptr = reinterpret_cast<AccessType*>(&aw_chunk);
        *row_weight_ptr = vec_weight_ptr[group.thread_rank() + group.num_threads() * 3];

        #pragma unroll 
        for (int i = 0; i < vec_size; i++) {
            bf_chunk[i] = static_cast<scalar_t>(static_cast<float>(bf_chunk[i]) * k_rms) * bw_chunk[i];
            af_chunk[i] = static_cast<scalar_t>(static_cast<float>(af_chunk[i]) * k_rms) * aw_chunk[i];
        }

        vec_thread_read_ptr = reinterpret_cast<AccessType*>(out_k); 
        row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&bf_chunk);
        vec_thread_read_ptr[group.thread_rank() + group.num_threads() * 2] = *row_chunk_vec_ptr;
        row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&af_chunk);
        vec_thread_read_ptr[group.thread_rank() + group.num_threads() * 3] = *row_chunk_vec_ptr;
    }
}

template<typename scalar_t, int cg_size, int vec_size>
__device__ float rms_norm_func_one_load(cg::thread_block_tile<cg_size> group,
                                scalar_t* __restrict__ input,
                                const int hidden_size,
                                const float eps)
{
    float ss = 0.0f;
    float s_rms = 0.0f;
    const int num_vec = hidden_size / vec_size;
    using AccessType = AlignedArrayI4<scalar_t, vec_size>;
    scalar_t row_chunk[vec_size];
    
    AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
    AccessType* vec_thread_read_ptr = reinterpret_cast<AccessType*>(input);
    const int lane_idx = group.thread_rank();

    *row_chunk_vec_ptr = vec_thread_read_ptr[lane_idx];

    #pragma unroll vec_size
    for (int i = 0; i < vec_size; i++) {
        float tmp = static_cast<float>(row_chunk[i]);
        ss += tmp * tmp;
    }
    for (int i = 1; i < cg_size; i *= 2) {
        ss += group.shfl_down(ss, i);
    }
    
    s_rms = rsqrtf(ss / hidden_size + eps);
    s_rms = group.shfl(s_rms, 0);
    return s_rms;
}

template<typename scalar_t, int cg_size>
__global__ void fused_attention_prepare(
    scalar_t* __restrict__ qkv,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out_q,
    scalar_t* __restrict__ out_kv,
    const int64_t* __restrict__ positions,
    const float base,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int input_len,
    const int q_stride,
    const int k_stride,
    const int qkv_stride,
    const int rot_dim,
    const float eps
){
    constexpr int vec_size = 16 / sizeof(scalar_t);
    auto block = cg::this_thread_block();
    auto cg = cg::tiled_partition<cg_size>(block);
    const int cg_idx = block.thread_index().x / cg_size + block.group_index().x * block.num_threads() / cg_size;
    if (cg_idx >= input_len * num_heads) return;

    const int token_idx = cg_idx / num_heads;
    const int head_idx = cg_idx % num_heads;
    const int q_start_offset = token_idx * qkv_stride + head_idx * head_size;
    const int k_start_offset = token_idx * qkv_stride + num_heads * head_size + head_idx * head_size;
    const int q_out_offset = token_idx * q_stride + head_idx * head_size;
    const int k_out_offset = token_idx * k_stride + head_idx * head_size;

    float q_rms = 0.0f;
    float k_rms = 0.0f;
    
    q_rms = rms_norm_func_one_load<scalar_t, cg_size, vec_size>(cg, qkv + q_start_offset, head_size, eps);
    if (head_idx < num_kv_heads) {
        k_rms = rms_norm_func_one_load<scalar_t, cg_size, vec_size>(cg, qkv + k_start_offset, head_size, eps);
    }

    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("q_rms = %f\n", q_rms);
    //     printf("k_rms = %f\n", k_rms);
    //     printf("token_idx = %d\n", token_idx);
    //     printf("head_idx = %d\n", head_idx);
    //     printf("cg_idx = %d\n", cg_idx);
    //     printf("q_start_offset = %d\n", q_start_offset);
    //     printf("k_start_offset = %d\n", k_start_offset);
    //     printf("q_stride = %d\n", q_stride);
    //     printf("k_stride = %d\n", k_stride);
    //     printf("q_out_offset = %d\n", q_out_offset);
    //     printf("k_out_offset = %d\n", k_out_offset);
    // }

    apply_rotary_embedding<scalar_t, cg_size, vec_size / 4>(cg, qkv + q_start_offset, qkv + k_start_offset, out_q + q_out_offset, out_kv + k_out_offset, weight, q_rms, k_rms,
                                                        base, positions[token_idx],
                                                        head_idx, num_kv_heads, rot_dim);
}

void FusedAttentionPrepare(torch::Tensor qkv,
                            torch::Tensor weight,
                            torch::Tensor positions,
                            torch::Tensor out_q,
                            torch::Tensor out_kv,
                            const int num_heads,
                            const int num_kv_heads,
                            const int head_dim,
                            const float base,
                            const int max_position_embeddings,
                            const float rms_norm_eps,
                            const float partial_rotary_factor){
    TORCH_CHECK(qkv.dtype() == torch::kBFloat16);
    TORCH_CHECK(positions.dtype() == torch::kInt64);
    int positions_ndim = positions.dim();
    TORCH_CHECK(positions_ndim == 1,
      "only support positions have shape [num_tokens]");
    TORCH_CHECK(qkv.size(0) == positions.size(0),
                "query, key and positions must have the same number of tokens");

    const int input_len = qkv.size(-2);
    TORCH_CHECK(positions.size(-1) == input_len);
    TORCH_CHECK(num_kv_heads <= num_heads);

    const int rot_dim = int(head_dim * partial_rotary_factor);
    const int q_size = num_heads * head_dim;
    const int kv_size = num_kv_heads * head_dim;
    const int qkv_stride = qkv.size(-1);

    const int block_size = 512;
    constexpr int cg_size = 16;
    const int cg_per_block = block_size / cg_size;
    const int grid_size = (input_len * num_heads + cg_per_block - 1) / cg_per_block;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_attention_prepare<bfloat16, cg_size><<<grid_size, block_size, 0, stream>>>
        (reinterpret_cast<bfloat16*>(qkv.data_ptr<at::BFloat16>()), 
        reinterpret_cast<bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16*>(out_q.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16*>(out_kv.data_ptr<at::BFloat16>()), 
        positions.data_ptr<int64_t>(), base, num_heads, num_kv_heads, head_dim, input_len, q_size, 
        kv_size, qkv_stride, rot_dim, rms_norm_eps);
}