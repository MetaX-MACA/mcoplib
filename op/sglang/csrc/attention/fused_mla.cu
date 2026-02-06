// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
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

// 对q做rotary_emb，对latent_cache做rms_normal，更新latent_cache和kv_a，之后对latent_cache做rotary_emb。
// 所有的cos_sin_cache都是float.
int64_t fused_mla_RMS_rotary_emb(
    torch::Tensor& q, // [3201, 128, 129] bf16
    torch::Tensor& latent_cache, // [3201, 576], dtype=bf16
    torch::Tensor& cos_sin_cache, // [max, 64], dtype=float
    torch::Tensor& positions, // [bs], dtype=int64
    torch::Tensor& norm_weight, // [512], dtype=bf16
    torch::Tensor& kv_a, // [3201, 512], dtype=bf16
    int64_t q_len, //16
    int64_t num_local_heads, //128,
    int64_t kv_lora_rank, // 512
    int64_t qk_rope_head_dim, //64
    int64_t qk_nope_head_dim //128
) {
    DEBUG_TRACE_PARAMS(q, latent_cache, cos_sin_cache, positions, norm_weight, kv_a, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim);
    DEBUG_DUMP_PARAMS(q, latent_cache, cos_sin_cache, positions, norm_weight, kv_a, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    int dev = q.get_device();

    //CHECK ALL SHAPES
    TORCH_CHECK(q_len == q.size(0), "Expected q.size(0) = q_len, but get ", q.size(0), " vs ", q_len);
    TORCH_CHECK(num_local_heads == q.size(1), "Expected q.size(1) = num_local_heads, but get ", q.size(1), " vs ", num_local_heads);
    TORCH_CHECK(qk_nope_head_dim+qk_rope_head_dim == q.size(2), "Expected q.size(2) == qk_nope_head_dim+qk_rope_head_dim but get",
        q.size(2), " vs ", qk_nope_head_dim+qk_rope_head_dim);
    TORCH_CHECK(latent_cache.size(0) == q_len && latent_cache.size(1) == kv_lora_rank + qk_rope_head_dim,
        "Invalid shape of latent cache, get (", latent_cache.size(0), ", ", latent_cache.size(1), ") but expected (",
        q_len, ", ", kv_lora_rank + qk_rope_head_dim, ")"
    );
    TORCH_CHECK(cos_sin_cache.size(1) == qk_rope_head_dim,
        "Invalid cos_sin_cache shape, get(MAX_POSITION_EMBEDDINGS, ", cos_sin_cache.size(1),
        "), but expected (MAX_POSITION_EMBEDDINGS, ", qk_rope_head_dim, ")");
    TORCH_CHECK(positions.size(0) == q_len, "Invalid positions shape, get (", positions.size(0), ") but expected (", q_len, ")");
    TORCH_CHECK(norm_weight.size(0) == kv_lora_rank, "Invalid norm_weight shape, get (", norm_weight.size(0), ") but expected (", kv_lora_rank, ")");
    TORCH_CHECK(kv_a.size(0) == q_len && kv_a.size(1) == kv_lora_rank ,
        "Invalid shape of kv_a, get (", kv_a.size(0), ", ", kv_a.size(1), ") but expected (",
        q_len, ", ", kv_lora_rank, ")"
    );

    #define LAUNCH_FUSED_MLA_RMS_ROTARY_EMB(NUM_LOCAL_HEADS, KV_LORA_RANK, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM)  \
        else if (q_len > 0 && num_local_heads == NUM_LOCAL_HEADS && kv_lora_rank == KV_LORA_RANK &&             \
            qk_nope_head_dim == QK_NOPE_HEAD_DIM && qk_rope_head_dim == QK_ROPE_HEAD_DIM) {                     \
                fused_mla::fused_mla_RMS_rotary_emb<scalar_t, NUM_LOCAL_HEADS, KV_LORA_RANK,                         \
                    QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM><<<grid, block, 0, at::cuda::getCurrentCUDAStream(dev)>>>(   \
                    q_len,                                                                                          \
                    (nv_bfloat16*)(q.data_ptr<at::BFloat16>()),                                                            \
                    (nv_bfloat16*)(latent_cache.data_ptr<at::BFloat16>()),                                                 \
                    (const float*)(cos_sin_cache.data_ptr<float>()),                                                \
                    (const int64_t*)(positions.data_ptr()),                                                         \
                    (const nv_bfloat16*)(norm_weight.data_ptr<at::BFloat16>()),                                            \
                    (nv_bfloat16*)(kv_a.data_ptr<at::BFloat16>()));                                                        \
            }
            

    dim3 grid = dim3(q_len*num_local_heads/4 + (q_len+3)/4, 1, 1);
    dim3 block = dim3(256, 1, 1);
    using scalar_t = nv_bfloat16;
    if (false) {

    }
    LAUNCH_FUSED_MLA_RMS_ROTARY_EMB(8, 512, 128, 64)
    LAUNCH_FUSED_MLA_RMS_ROTARY_EMB(16, 512, 128, 64)
    LAUNCH_FUSED_MLA_RMS_ROTARY_EMB(32, 512, 128, 64)
    LAUNCH_FUSED_MLA_RMS_ROTARY_EMB(64, 512, 128, 64)
    LAUNCH_FUSED_MLA_RMS_ROTARY_EMB(128, 512, 128, 64)
    else {
        TORCH_CHECK(false, "Parameters num_local_heads = ", num_local_heads, ", kv_lora_rank = ", kv_lora_rank,
            ", qk_nope_head_dim = ", qk_nope_head_dim, ", qk_rope_head_dim = ", qk_rope_head_dim, " do not supported!");
        return 1;
    }

    return 0;

}

// 在prefill阶段做数据拷贝,element_wise操作。
int64_t fused_mla_normal_kv_element_wise(
    torch::Tensor& kv, // [bs, num_local_heads * (qk_nope_head_dim + v_head_dim)] [3201, 32768], dtype=bf16
    torch::Tensor& latent_cache, // [bs, kv_lora_rank + qk_rope_head_dim] [3201, 576], dtype=bf16
    torch::Tensor& k, // [bs, num_local_heads, qk_nope_head_dim + qk_rope_head_dim] [3201, 128, 192], dtype=bf16
    torch::Tensor& v, // [bs, num_local_heads, v_head_dim] [3201, 128, 128], dtype=bf16
    int64_t q_len, // 3201
    int64_t num_local_heads, // 128
    int64_t kv_lora_rank, // 512
    int64_t qk_nope_head_dim, // 128
    int64_t qk_rope_head_dim, // 64
    int64_t v_head_dim // 128
) {
    DEBUG_TRACE_PARAMS(kv, latent_cache, k, v, q_len, num_local_heads, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim);
    DEBUG_DUMP_PARAMS(kv, latent_cache, k, v, q_len, num_local_heads, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(k));
    int dev = k.get_device();

    //CHECK ALL SHAPS
    TORCH_CHECK(q_len == k.size(0), "Expected k.size(0) = q_len, but get ", k.size(0), " vs ", q_len);
    TORCH_CHECK(q_len == v.size(0), "Expected v.size(0) = q_len, but get ", v.size(0), " vs ", q_len);
    TORCH_CHECK(num_local_heads == k.size(1), "Expected k.size(1) = num_local_heads, but get ", k.size(1), " vs ", num_local_heads);
    TORCH_CHECK(num_local_heads == v.size(1), "Expected v.size(1) = num_local_heads, but get ", v.size(1), " vs ", num_local_heads);
    TORCH_CHECK(qk_nope_head_dim+qk_rope_head_dim == k.size(2), "Expected k.size(2) == qk_nope_head_dim+qk_rope_head_dim, but get",
        k.size(2), " vs ", qk_nope_head_dim+qk_rope_head_dim);
    TORCH_CHECK(v_head_dim == v.size(2), "Expected v.size(2) == v_head_dim, but get",
        k.size(2), " vs ", qk_nope_head_dim+qk_rope_head_dim);
    TORCH_CHECK(kv.size(0) == q_len && kv.size(1) == num_local_heads * (qk_nope_head_dim + v_head_dim),
        "Invalid shape of kv, get (", kv.size(0), ", ", kv.size(1), ") but expected (",
        q_len, ", ", num_local_heads * (qk_nope_head_dim + v_head_dim), ")"
    );
    TORCH_CHECK(latent_cache.size(0) == q_len && latent_cache.size(1) == kv_lora_rank + qk_rope_head_dim,
        "Invalid shape of latent cache, get (", latent_cache.size(0), ", ", latent_cache.size(1), ") but expected (",
        q_len, ", ", kv_lora_rank + qk_rope_head_dim, ")"
    );

    #define LAUNCH_FUSED_MLA_NORMAL_KV_ELEMENT_WISE(NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM, V_HEAD_DIM) \
        else if (k_len > 0 && num_local_heads == NUM_LOCAL_HEADS  &&                 \
                 qk_nope_head_dim == QK_NOPE_HEAD_DIM && qk_rope_head_dim == QK_ROPE_HEAD_DIM &&                    \
                 v_head_dim == V_HEAD_DIM) {                                                                          \
                fused_mla::fused_mla_normal_kv_element_wise<scalar_t, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM,   \
                    QK_ROPE_HEAD_DIM, V_HEAD_DIM><<<grid, block, 0, at::cuda::getCurrentCUDAStream(dev)>>>(             \
                    k_len,                                                                                              \
                    (const nv_bfloat16*)(kv.data_ptr<at::BFloat16>()),                                                         \
                    (const nv_bfloat16*)(latent_cache.data_ptr<at::BFloat16>()),                                               \
                    (nv_bfloat16*)(k.data_ptr<at::BFloat16>()),                                                                \
                    (nv_bfloat16*)(v.data_ptr<at::BFloat16>()));                                                               \
                }


    const int k_len = k.size(0);
    dim3 grid = dim3((k_len * num_local_heads * (qk_nope_head_dim + qk_rope_head_dim + v_head_dim)/8 + 255) / 256, 1, 1);
    dim3 block = dim3(256, 1, 1);
    using scalar_t = nv_bfloat16;
    if (false) {

    }
    LAUNCH_FUSED_MLA_NORMAL_KV_ELEMENT_WISE(8, 128, 64, 128)
    LAUNCH_FUSED_MLA_NORMAL_KV_ELEMENT_WISE(16, 128, 64, 128)
    LAUNCH_FUSED_MLA_NORMAL_KV_ELEMENT_WISE(32, 128, 64, 128)
    LAUNCH_FUSED_MLA_NORMAL_KV_ELEMENT_WISE(64, 128, 64, 128)
    LAUNCH_FUSED_MLA_NORMAL_KV_ELEMENT_WISE(128, 128, 64, 128)
    else {
        TORCH_CHECK(false, "Parameters num_local_heads = ", num_local_heads, ", kv_lora_rank = ", kv_lora_rank,
            ", qk_nope_head_dim = ", qk_nope_head_dim, ", qk_rope_head_dim = ", qk_rope_head_dim,
            ", v_head_dim = ", v_head_dim," do not supported!");
        return 1;
    }

    return 0;
}
