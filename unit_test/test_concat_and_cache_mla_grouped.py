import torch
import torch.nn.functional as F
import pytest
from typing import List
import mcoplib._C


def ref_concat_and_cache_mla_grouped(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_caches: List[torch.Tensor],
    slot_mapping: torch.Tensor,
    block_size: int,
    block_stride: int,
    entry_stride: int,
):
    num_layers, num_tokens, kv_lora_rank = kv_c.shape
    pe_dim = k_pe.shape[2]

    for layer_idx in range(num_layers):
        kv_cache_flat = kv_caches[layer_idx].view(-1)
        for token_idx in range(num_tokens):
            slot_idx = slot_mapping[layer_idx, token_idx].item()
            if slot_idx < 0:
                continue

            block_idx = slot_idx // block_size
            block_offset = slot_idx % block_size

            c_vec = kv_c[layer_idx, token_idx]
            pe_vec = k_pe[layer_idx, token_idx]

            dst_base = block_idx * block_stride + block_offset * entry_stride
            
            kv_cache_flat[dst_base : dst_base + kv_lora_rank] = c_vec
            kv_cache_flat[dst_base + kv_lora_rank : dst_base + kv_lora_rank + pe_dim] = pe_vec

# ==============================================================================
# 3. Pytest 单元测试（修复并发冲突与全0比较逻辑）
# ==============================================================================

@pytest.mark.parametrize("num_layers", [2, 8])
@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("pe_dim", [64])
@pytest.mark.parametrize("block_size", [16])
def test_concat_and_cache_mla_grouped(
    num_layers: int,
    num_tokens: int,
    kv_lora_rank: int,
    pe_dim: int,
    block_size: int,
):
    device = "cuda"
    dtype = torch.bfloat16
    num_blocks = max(64, (num_tokens // block_size) + 16)

    entry_stride = kv_lora_rank + pe_dim
    block_stride = block_size * entry_stride
    max_slots = num_blocks * block_size

    # 1. 构造随机输入 Tensor
    kv_c = torch.randn((num_layers, num_tokens, kv_lora_rank), dtype=dtype, device=device)
    k_pe = torch.randn((num_layers, num_tokens, pe_dim), dtype=dtype, device=device)

    # 【修复重点 1】：使用 randperm 确保单层内 slot_mapping 互不重复，消除 CUDA 写竞争
    slot_mapping = torch.full((num_layers, num_tokens), -1, dtype=torch.int64, device=device)
    for l in range(num_layers):
        perm = torch.randperm(max_slots, device=device)[:num_tokens]
        # 随机让大约 15% 的 token 成为 Padding (-1)
        valid_mask = torch.rand(num_tokens, device=device) > 0.15
        slot_mapping[l, valid_mask] = perm[valid_mask]

    # 2. 构造 Kernel 与 Ref 调用的独立 KV Cache 内存
    kernel_kv_caches = [
        torch.zeros((num_blocks, block_size, entry_stride), dtype=dtype, device=device)
        for _ in range(num_layers)
    ]
    ref_kv_caches = [
        torch.zeros((num_blocks, block_size, entry_stride), dtype=dtype, device=device)
        for _ in range(num_layers)
    ]

    kv_cache_ptrs = torch.tensor(
        [cache.data_ptr() for cache in kernel_kv_caches],
        dtype=torch.int64,
        device=device,
    )

    # 3. 执行 PyTorch Reference 实现与 CUDA Kernel
    ref_concat_and_cache_mla_grouped(
        kv_c=kv_c,
        k_pe=k_pe,
        kv_caches=ref_kv_caches,
        slot_mapping=slot_mapping,
        block_size=block_size,
        block_stride=block_stride,
        entry_stride=entry_stride,
    )

    torch.ops._C.concat_and_cache_mla_grouped(
        kv_c,
        k_pe,
        kv_cache_ptrs,
        slot_mapping,
        block_size,
        block_stride,
        entry_stride,
    )

    # 4. 精度校验
    min_cosine_similarity = 1.0

    for layer in range(num_layers):
        kernel_out = kernel_kv_caches[layer].flatten().float()
        ref_out = ref_kv_caches[layer].flatten().float()

        # 【修复重点 2】：对于纯内存 Copy 算子，验证完全一致 (Bitwise Equal)
        assert torch.equal(kernel_out, ref_out), f"Layer {layer} 数据未实现完全一致性写入！"

        # 处理余弦相似度（处理全零层的极端情况）
        if torch.all(kernel_out == 0) and torch.all(ref_out == 0):
            cos_sim = 1.0
        else:
            cos_sim = F.cosine_similarity(kernel_out, ref_out, dim=0, eps=1e-8).item()
            
        min_cosine_similarity = min(min_cosine_similarity, cos_sim)

        assert cos_sim >= 0.9999, (
            f"Layer {layer} 余弦相似度校验失败！期望目标 >= 0.9999，实际达到: {cos_sim:.6f}"
        )

    print(f"\n[PASS] 所有层精度验证通过！最低余弦相似度: {min_cosine_similarity:.6f}")