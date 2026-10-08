"""
Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""

import torch

try:
    import mcoplib.sgl_kernel as sgl
except ImportError as e:
    print("Failed to import from sgl_kernel with %r", e)


try:
    import mcoplib.sgl_grouped_gemm_cuda
except ImportError as e:
    print("Failed to import from sgl_grouped_gemm_cuda with %r", e)



#功能：mla中，对q做rotary_emb，对latent_cache做rms_normal，更新latent_cache和kv_a，之后对latent_cache做rotary_emb。
#     调用torch的kv_b_proj计算kv，将数据从kv拷贝到k和v，从latent_cache中拷贝数据到k
#输入：
#输出：
#限制：
def fused_mla_normal_rotary_emb(
    kv_a:torch.tensor,
    kv_b_proj,
    q:torch.tensor, # [bs, 128, 192], dtype=bf16
    latent_cache:torch.tensor, # [bs, 576], dtype=bf16
    positions:torch.tensor, # [bs], dtype=int64
    cos_sin_cache:torch.tensor, # [max_position_embeddings, 64], dtype=float
    norm_weight:torch.tensor, # [512], dtype=bf16
    k:torch.tensor, # [bs, 128, 192], dtype=bf16
    v:torch.tensor, # [bs, 128, 192], dtype=bf16
    q_len:int, #bs
    qk_nope_head_dim:int, #128
    qk_rope_head_dim:int, #64
    kv_lora_rank:int, #512
    v_head_dim:int, #128
    num_local_heads:int , #128
):
    out = torch.ops.sgl_kernel.fused_mla_RMS_rotary_emb(q, latent_cache, cos_sin_cache, positions, norm_weight, kv_a, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim)
    if out != 0:
        print("Failed to call mcoplib ops.fused_mla_RMS_rotary_emb")
    kv = kv_b_proj(kv_a)
    kv = kv[0] if isinstance(kv, tuple) else kv
    out = torch.ops.sgl_kernel.fused_mla_normal_kv_element_wise(kv, latent_cache, k, v, q_len, num_local_heads, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim)
    if out != 0:
        print("Failed to call mcoplib ops.fused_mla_normal_kv_element_wise")
    return q, k, v, latent_cache

#功能：对q做rotary_emb，对latent_cache做rms_normal，更新latent_cache和kv_a，之后对latent_cache做rotary_emb。
#输入：kv_a,q,latent_cache,positions,cos_sin_cache,norm_weight
#输出：q, kv_a, latent_cache
#限制：
def fused_mla_RMS_rotary_emb(
    kv_a:torch.tensor,
    q:torch.tensor, # [bs, 128, 192], dtype=bf16
    latent_cache:torch.tensor, # [bs, 576], dtype=bf16
    positions:torch.tensor, # [bs], dtype=int64
    cos_sin_cache:torch.tensor, # [max_position_embeddings, 64], dtype=float
    norm_weight:torch.tensor, # [512], dtype=bf16
    q_len:int, #bs
    qk_nope_head_dim:int, #128
    qk_rope_head_dim:int, #64
    kv_lora_rank:int, #512
    num_local_heads:int , #128
):
    out = torch.ops.sgl_kernel.fused_mla_RMS_rotary_emb(q, latent_cache, cos_sin_cache, positions, norm_weight, kv_a, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim)
    if out != 0:
        print("Failed to call ops.fused_mla_RMS_rotary_emb")
    return q, kv_a, latent_cache

#功能：在prefill阶段做数据拷贝,将数据从kv拷贝到k和v，从latent_cache中拷贝数据到k，element_wise操作。
#输入：kv,latent_cache,k,v
#输出：k,v,latent_cache
#限制：
def fused_mla_normal_kv_element_wise(
    kv:torch.tensor,
    latent_cache:torch.tensor, # [bs, 576], dtype=bf16
    k:torch.tensor, # [bs, 128, 192], dtype=bf16
    v:torch.tensor, # [bs, 128, 192], dtype=bf16
    q_len:int, #bs
    qk_nope_head_dim:int, #128
    qk_rope_head_dim:int, #64
    kv_lora_rank:int, #512
    v_head_dim:int, #128
    num_local_heads:int , #128
):
    out = torch.ops.sgl_kernel.fused_mla_normal_kv_element_wise(kv, latent_cache, k, v, q_len, num_local_heads, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim)
    if out != 0:
        print("Failed to call ops.fused_mla_normal_kv_element_wise")
    return k, v, latent_cache
