// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
// #include "fused_rotary_emb.h"

#include <cuda_bf16.h>
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>
#include "fused_mla_impl.cuh"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"


int64_t fused_mla_absorb_rotary_emb(
    torch::Tensor& q, // [bs, num_local_heads, qk_nope_head_dim+qk_rope_head_dim], dtype=bf16
    torch::Tensor& w_kc, // [num_local_heads, qk_nope_head_dim, kv_lora_rank], dtype=bf16
    torch::Tensor& latent_cache, // [bs, kv_lora_rank+qk_rope_head_dim], dtype=bf16
    torch::Tensor& cos_sin_cache, // [max_position_embeddings, qk_rope_head_dim], dtype=bf16
    torch::Tensor& positions, // [bs], dtype=int64
    torch::Tensor& norm_weight, // [kv_lora_rank], dtype=bf16
    torch::Tensor& q_input, //[bs, qk_nope_head_dim, kv_lora_rank+qk_rope_head_dim], dtype=bf16
    torch::Tensor& k_input, //[bs, 1, kv_lora_rank+qk_rope_head_dim], dtype=bf16
    torch::Tensor& v_input, //[bs, 1, kv_lora_rank]
    int64_t q_len, //16
    int64_t num_local_heads, //128,
    int64_t kv_lora_rank, // 512
    int64_t qk_rope_head_dim, //64
    int64_t qk_nope_head_dim //128
) {
  DEBUG_TRACE_PARAMS(q, w_kc, latent_cache, cos_sin_cache, positions, norm_weight, q_input, k_input, v_input, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim);
  DEBUG_DUMP_PARAMS(q, w_kc, latent_cache, cos_sin_cache, positions, norm_weight, q_input, k_input, v_input, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim);
    //TODO:
    //check all shape
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    int dev = q.get_device();
    // std::cout << "fused_forward_absorb q_len = " << q_len
    //     << ", num_local_heads = " << num_local_heads
    //     << ", kv_lora_rank = " << kv_lora_rank
    //     << ", qk_rope_head_dim = " << qk_rope_head_dim
    //     << ", qk_nope_head_dim = " << qk_nope_head_dim
    //     << std::endl;

    //CHECK ALL SHAPES
    TORCH_CHECK(q_len == q.size(0), "Expected q.size(0) = q_len, but get ", q.size(0), " vs ", q_len);
    TORCH_CHECK(num_local_heads == q.size(1), "Expected q.size(1) = num_local_heads, but get ", q.size(1), " vs ", num_local_heads);
    TORCH_CHECK(qk_nope_head_dim+qk_rope_head_dim == q.size(2), "Expected q.size(2) == qk_nope_head_dim+qk_rope_head_dim but get",
        q.size(2), " vs ", qk_nope_head_dim+qk_rope_head_dim);
    TORCH_CHECK(w_kc.size(0) == num_local_heads && w_kc.size(1) == qk_nope_head_dim && w_kc.size(2) == kv_lora_rank,
        "Invalid shape of w_kc, get (", w_kc.size(0), ", ", w_kc.size(1), ", ", w_kc.size(2), ") but expected (",
        num_local_heads, ", ", qk_nope_head_dim, ", ", kv_lora_rank, ")");
    TORCH_CHECK(latent_cache.size(0) == q_len && latent_cache.size(1) == kv_lora_rank + qk_rope_head_dim,
        "Invalid shape of latent cache, get (", latent_cache.size(0), ", ", latent_cache.size(1), ") but expected (",
        q_len, ", ", kv_lora_rank + qk_rope_head_dim, ")"
    );
    TORCH_CHECK(q_input.size(0) == q_len && q_input.size(1) == num_local_heads && q_input.size(2) == kv_lora_rank + qk_rope_head_dim,
        "Invalid shape of q_input, get (", q_input.size(0), ", ", q_input.size(1), ", ", q_input.size(2), ") but expeced (",
        q_len, ", ", qk_nope_head_dim, ", ", kv_lora_rank+qk_rope_head_dim, ")"
    );
    TORCH_CHECK(k_input.size(0) == q_len && k_input.size(1) == 1 && k_input.size(2) == kv_lora_rank + qk_rope_head_dim,
        "Invalid shape of k_input, get (", k_input.size(0), ", ", k_input.size(1), ", ", k_input.size(2), ") but expeced (",
        q_len, ", ", 1, ", ", kv_lora_rank+qk_rope_head_dim, ")"
    );
    TORCH_CHECK(v_input.size(0) == q_len && v_input.size(1) == 1 && v_input.size(2) == kv_lora_rank,
        "Invalid shape of v_input, get (", v_input.size(0), ", ", v_input.size(1), ", ", v_input.size(2), ") but expeced (",
        q_len, ", ", 1, ", ", kv_lora_rank, ")"
    );
    TORCH_CHECK(cos_sin_cache.size(1) == qk_rope_head_dim,
        "Invalid cos_sin_cache shape, get(MAX_POSITION_EMBEDDINGS, ", cos_sin_cache.size(1),
        "), but expected (MAX_POSITION_EMBEDDINGS, ", qk_rope_head_dim, ")");
    TORCH_CHECK(positions.size(0) == q_len, "Invalid positions shape, get (", positions.size(0), ") but expected (", q_len, ")");
    TORCH_CHECK(norm_weight.size(0) == kv_lora_rank, "Invalid norm_weight shape, get (", norm_weight.size(0), ") but expected (", kv_lora_rank, ")");

    #define LAUNCH_FUSED_ABSORB_MLA(NUM_LOCAL_HEADS, KV_LORA_RANK, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM) \
        else if (q_len > 0 && num_local_heads == NUM_LOCAL_HEADS && kv_lora_rank == KV_LORA_RANK && qk_nope_head_dim == QK_NOPE_HEAD_DIM && qk_rope_head_dim == QK_ROPE_HEAD_DIM) { \
            fused_mla::fused_absorb_mla<scalar_t, NUM_LOCAL_HEADS, KV_LORA_RANK, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM><<<grid, block, 0, at::cuda::getCurrentCUDAStream(dev)>>>(      \
                q_len,                                                                                                                                                              \
                (const nv_bfloat16*)(q.data_ptr<at::BFloat16>()),                                                                                                                   \
                (const nv_bfloat16*)(w_kc.data_ptr<at::BFloat16>()),                                                                                                                \
                (const nv_bfloat16*)(latent_cache.data_ptr<at::BFloat16>()),                                                                                                        \
                latent_cache_stride,                                                                                                                                                \
                (const float*)(cos_sin_cache.data_ptr<float>()),                                                                                                                    \
                (const int64_t*)(positions.data_ptr()),                                                                                                                             \
                (const nv_bfloat16*)(norm_weight.data_ptr<at::BFloat16>()),                                                                                                         \
                (nv_bfloat16*)(q_input.data_ptr<at::BFloat16>()),                                                                                                                   \
                (nv_bfloat16*)(k_input.data_ptr<at::BFloat16>()),                                                                                                                   \
                (nv_bfloat16*)(v_input.data_ptr<at::BFloat16>())                                                                                                                    \
            );                                                                                                                                                                      \
        }



    // dim3 grid = dim3((q_len/4 +4)*(num_local_heads+1)-1, 1, 1);
    dim3 grid = dim3((q_len + 15)/16 * kv_lora_rank/128 * num_local_heads + (q_len+3)/4 * num_local_heads + (q_len+3)/4, 1, 1);
    dim3 block = dim3(256, 1, 1);
    const int latent_cache_stride = latent_cache.stride(0);
    // printf("latent_cache_stride: %d\n", latent_cache_stride);
    using scalar_t = nv_bfloat16;

    // TORCH_CHECK(q_len <= 32, "q_len should be less than 32 , ", "q_len =", q_len, " is not supported.");
    // if (q_len > 0 && num_local_heads == 128 && kv_lora_rank == 512 && qk_rope_head_dim == 64 && qk_nope_head_dim == 128) {
    //     fused_mla::fused_absorb_mla<scalar_t, 128, 512, 128, 64><<<grid, block, 0, at::cuda::getCurrentCUDAStream(dev)>>>(
    //         q_len,
    //         (const nv_bfloat16*)(q.data_ptr<at::BFloat16>()),
    //         (const nv_bfloat16*)(w_kc.data_ptr<at::BFloat16>()),
    //         (const nv_bfloat16*)(latent_cache.data_ptr<at::BFloat16>()),
    //         latent_cache_stride,
    //         (const float*)(cos_sin_cache.data_ptr<float>()),
    //         (const int64_t*)(positions.data_ptr()),
    //         (const nv_bfloat16*)(norm_weight.data_ptr<at::BFloat16>()),
    //         (nv_bfloat16*)(q_input.data_ptr<at::BFloat16>()),
    //         (nv_bfloat16*)(k_input.data_ptr<at::BFloat16>()),
    //         (nv_bfloat16*)(v_input.data_ptr<at::BFloat16>())
    //     );
    //     return 0;
    // }

    if (false) {

    }
    LAUNCH_FUSED_ABSORB_MLA(4, 512, 128, 64)
    LAUNCH_FUSED_ABSORB_MLA(8, 512, 128, 64)
    LAUNCH_FUSED_ABSORB_MLA(16, 512, 128, 64)
    LAUNCH_FUSED_ABSORB_MLA(32, 512, 128, 64)
    LAUNCH_FUSED_ABSORB_MLA(64, 512, 128, 64)
    LAUNCH_FUSED_ABSORB_MLA(128, 512, 128, 64)
    else {
        TORCH_CHECK(false, "Parameters num_local_heads = ", num_local_heads, ", kv_lora_rank = ", kv_lora_rank,
            ", qk_nope_head_dim = ", qk_nope_head_dim, ", qk_rope_head_dim = ", qk_rope_head_dim, " do not supported!");
        return 1;
    }

    return 0;
}
