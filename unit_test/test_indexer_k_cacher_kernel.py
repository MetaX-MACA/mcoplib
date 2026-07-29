import torch
import torch.nn.functional as F

# 导入你注册的算子库
import mcoplib._C 

def indexer_k_cache_ref(k: torch.Tensor, kv_cache: torch.Tensor, slot_mapping: torch.Tensor) -> torch.Tensor:

    kv_cache_ref = kv_cache.clone()
    block_size = kv_cache_ref.size(1)
    head_dim = k.size(1)

    # 正常的地址解码（没有脏数据，不需要过滤拦截）
    block_idx = slot_mapping // block_size
    block_offset = slot_mapping % block_size

    # 将 K 写入 Cache 对应位置
    kv_cache_ref[block_idx, block_offset, :head_dim] = k

    return kv_cache_ref

def test_indexer_k_cache_pure():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    # ==========================================
    # 严格遵照要求的 Shape
    # k: [5163, 128]
    # kv_cache: [1690, 64, 132]
    # slot_mapping: [5163]
    # ==========================================
    num_tokens = 5163
    head_dim = 128
    num_blocks = 1690
    block_size = 64
    cache_stride = 132
    max_slots = num_blocks * block_size

    print(">>> [1/5] 初始化测试数据...")
    # 正常的连续张量，不再使用切片
    k = torch.randn((num_tokens, head_dim), dtype=dtype, device=device)
    kv_cache_init = torch.randn((num_blocks, block_size, cache_stride), dtype=dtype, device=device)
    
    # 使用 randperm 保证生成的 5163 个有效 Slot 互不重复，避免并发写冲突
    unique_slots = torch.randperm(max_slots, dtype=torch.int64, device=device)
    slot_mapping = unique_slots[:num_tokens]

    kv_cache_ref_input = kv_cache_init.clone()
    kv_cache_cuda = kv_cache_init.clone()

    print(">>> [2/5] 运行 Torch Reference...")
    kv_cache_ref_out = indexer_k_cache_ref(k, kv_cache_ref_input, slot_mapping)

    print(">>> [3/5] 运行 CUDA Kernel...")
    # 调用底层 C++ Kernel
    torch.ops._C_cache_ops.indexer_k_cache(k, kv_cache_cuda, slot_mapping)
    
    # 强制同步等待 GPU 写完
    torch.cuda.synchronize()

    print(">>> [4/5] 进行余弦相似度校验...")
    cos_sim = F.cosine_similarity(
        kv_cache_cuda.flatten().float(),
        kv_cache_ref_out.flatten().float(),
        dim=0
    ).item()

    print("\n" + "="*60)
    print(f"🚀 CUDA Kernel vs Torch Ref 余弦相似度: {cos_sim:.8f}")
    print("="*60 + "\n")

    print(">>> [5/5] 抽样数据比对展示...")
    # 随机抽样第 0 个 Token 打印查看
    sample_idx = 0
    slot_idx = slot_mapping[sample_idx].item()
    b_idx = slot_idx // block_size
    b_off = slot_idx % block_size

    print(f"📊 Token ID {sample_idx} 映射到 -> Block: {b_idx}, Offset: {b_off}")
    print(f"\n[1] 原始 K 张量输入前 8 个元素:\n{k[sample_idx, :8]}")
    print(f"\n[2] Torch Ref 写入结果:\n{kv_cache_ref_out[b_idx, b_off, :8]}")
    print(f"\n[3] CUDA Kernel 写入结果:\n{kv_cache_cuda[b_idx, b_off, :8]}")
    print("\n" + "-"*60)

    # 严格断言
    if cos_sim <= 0.9999:
        # 如果依然报错，计算具体的差异数量方便排查
        diff = torch.abs(kv_cache_cuda.float() - kv_cache_ref_out.float())
        mismatch_count = (diff > 1e-4).sum().item()
        print(f"\n⚠️ 发现 {mismatch_count} 个元素不一致！")
        raise AssertionError(f"❌ 精度校验失败！余弦相似度 {cos_sim:.8f} 低于阈值 0.9999")
    
    print("✅ 完美通过！CUDA 算子输出结果与 Torch Ref 100% 一致！")

if __name__ == "__main__":
    test_indexer_k_cache_pure()
