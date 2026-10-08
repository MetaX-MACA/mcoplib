# -*- coding: utf-8 -*-
"""
hc_mix_stats 单元测试 + 主流大模型 MHC(超连接 Hyper-Connection)配置多用例覆盖。

本文件完全自包含: 不加载、不读取、不依赖 ./hc_mix_stats_inputs.pt。
所有入参均由 torch 现场生成(固定 seed 可复现),并覆盖原 .pt 文件对应的
数据 shape 与 dtype: x_flat(1,20480) bf16, hc_fn(24,20480) fp32, eps=1e-20。

hc_mix_stats 语义:
    out = F.linear(x_flat.float(), hc_fn) * rsqrt(mean(x_flat^2, dim=K) + eps)
    x_flat: [M, K] 任意浮点 dtype;   hc_fn: [MIX, K] fp32;   返回: [M, MIX] fp32

维度含义(对应 mhc.py):
    K   = hidden_size(残差流隐藏维)
    M   = num_tokens(batch,推理时的 token 数)
    MIX = hc_mult3 = hc_mult * (2 + hc_mult)   # 超连接展开的混合通道数, 断言 <= 32

Metax C500/C600U 约束: 张量核 dot 的 M-tile 必须 >= 16(见 mhc.py 内 NOTE),
_HC_MIX_BLOCK_M_SMALL 已由 8 提升为 16。本测试同时覆盖 M 的各分支边界与批不变性。
"""
import torch

from mcoplib.triton_sglang_mhc import hc_mix_stats


# ----------------------------------------------------------------------------
# 原 ./hc_mix_stats_inputs.pt 对应的数据规格(仅保留 shape/dtype/eps 语义,不读文件)
#   x_flat: (1, 20480) bfloat16 ;  hc_fn: (24, 20480) float32 ;  eps=1e-20
# ----------------------------------------------------------------------------
ORIG_M, ORIG_K, ORIG_MIX = 1, 20480, 24
ORIG_EPS = 1e-20


def make_inputs(m, k, mix, x_dtype=torch.bfloat16, seed=0, device="cpu"):
    """按给定 shape/dtype 现场生成一组可复现的 hc_mix_stats 入参(不依赖任何文件)。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x_flat = torch.randn(m, k, generator=g, dtype=torch.float32).to(x_dtype)
    # hc_fn 恒为 fp32(kernel 强制要求)。缩放 1/sqrt(k) 让 F.linear 结果量级稳定。
    hc_fn = (torch.randn(mix, k, generator=g, dtype=torch.float32) / (k ** 0.5))
    return x_flat.to(device).contiguous(), hc_fn.to(device).contiguous()


# ----------------------------------------------------------------------------
# torch 参考实现(golden)
# ----------------------------------------------------------------------------
def reference(x_flat, hc_fn, eps):
    xf = x_flat.float()
    inv_k = 1.0 / xf.shape[1]
    rms = torch.rsqrt(xf.pow(2).sum(dim=1) * inv_k + eps)
    return torch.nn.functional.linear(xf, hc_fn) * rms[:, None]


# tf32x3 三遍 TF32 累加 + 长归约(K 可达 20480),放宽到 tf32 量级容差
RTOL, ATOL = 2e-2, 2e-2


def check_case(name, x_flat, hc_fn, eps, device):
    xd = x_flat.to(device)
    hd = hc_fn.to(device)
    out = hc_mix_stats(xd, hd, eps)              # [M, MIX] fp32
    ref = reference(xd, hd, eps)
    max_abs = (out - ref).abs().max().item()
    denom = ref.abs().max().item() + 1e-12
    max_rel = max_abs / denom
    trap = (not torch.isfinite(out).all().item())
    m, k = x_flat.shape
    mix = hc_fn.shape[0]
    status = "TRAP/NaN" if trap else "ok"
    print(f"[{name:34s}] M={m:<5d} K={k:<6d} MIX={mix:<3d} x={str(x_flat.dtype).replace('torch.',''):8s}"
          f" -> out{tuple(out.shape)} max_abs={max_abs:.2e} max_rel={max_rel:.2e} {status}")
    assert not trap, f"{name}: 输出包含 NaN/Inf (Kernel Trap)"
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
    return out


def check_batch_invariance(name, x_flat, hc_fn, eps, device):
    """核心契约: 每一行单独计算 与 批处理中计算 必须逐 bit 一致。"""
    xd = x_flat.to(device)
    hd = hc_fn.to(device)
    batched = hc_mix_stats(xd, hd, eps)
    m = x_flat.shape[0]
    idxs = sorted(set([0, m // 2, m - 1]))
    max_bit_diff = 0.0
    for i in idxs:
        alone = hc_mix_stats(xd[i:i + 1].contiguous(), hd, eps)
        d = (alone[0] - batched[i]).abs().max().item()
        max_bit_diff = max(max_bit_diff, d)
    ok = "bit-identical" if max_bit_diff == 0.0 else f"DIFF={max_bit_diff:.2e}"
    print(f"[{name:34s}] 批不变性 rows={idxs} -> {ok}")
    assert max_bit_diff == 0.0, f"{name}: 批不变性被破坏 (max_bit_diff={max_bit_diff})"


# ----------------------------------------------------------------------------
# 主流大模型 MHC 配置矩阵
#   K = hidden_size;  MIX = hc_mult*(2+hc_mult), hc_mult∈{1,2,3,4} -> {3,8,15,24}
# ----------------------------------------------------------------------------
MAINSTREAM_HIDDEN = [
    ("Qwen2.5-7B",            3584),
    ("Llama3-8B/Mistral-7B",  4096),
    ("Qwen2.5-14B",           5120),
    ("GLM-4-9B",              6144),
    ("DeepSeek-V3/R1",        7168),
    ("Llama3-70B/Qwen2.5-72B",8192),
    ("DeepSeek-V4.1(orig)",   20480),
]
HC_MULTS = [1, 2, 3, 4]  # -> MIX 3/8/15/24
# 覆盖 _block_m_for 的全部分支与边界: SMALL(<=16)/MID(<=2048)/BIG(>2048)
M_VALUES = [1, 8, 16, 17, 128, 2048, 2049, 4096]
X_DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def test_original_spec(device):
    """覆盖原 .pt 对应的数据规格(现场生成,不读文件): M=1,K=20480,MIX=24,bf16,eps=1e-20。"""
    print("\n===== 1) 原 .pt 数据规格(现场生成,不依赖文件) =====")
    x_flat, hc_fn = make_inputs(ORIG_M, ORIG_K, ORIG_MIX,
                                x_dtype=torch.bfloat16, seed=1234)
    assert tuple(x_flat.shape) == (1, 20480) and x_flat.dtype == torch.bfloat16
    assert tuple(hc_fn.shape) == (24, 20480) and hc_fn.dtype == torch.float32
    check_case("orig-spec(1,20480)x24 bf16", x_flat, hc_fn, ORIG_EPS, device)
    print("✅ 原始数据规格用例精度校验通过")


def test_mainstream_matrix(device):
    print("\n===== 2) 主流大模型 MHC 配置矩阵(K × MIX,bf16) =====")
    for tag, k in MAINSTREAM_HIDDEN:
        for hc in HC_MULTS:
            mix = hc * (2 + hc)
            x_flat, hc_fn = make_inputs(1, k, mix, x_dtype=torch.bfloat16, seed=k + mix)
            check_case(f"{tag}|hc={hc}", x_flat, hc_fn, ORIG_EPS, device)
    print("✅ 主流配置矩阵全部通过")


def test_batch_shapes(device):
    print("\n===== 3) batch(M)分支/边界 覆盖(小 shape + 大 shape) =====")
    K, MIX = 4096, 24  # Llama-3-8B 隐藏维 + hc_mult=4
    for m in M_VALUES:
        x_flat, hc_fn = make_inputs(m, K, MIX, x_dtype=torch.bfloat16, seed=100 + m)
        check_case(f"M-sweep M={m}", x_flat, hc_fn, ORIG_EPS, device)
    print("✅ M 分支/边界全部通过")


def test_dtypes(device):
    print("\n===== 4) x_flat 多 dtype 覆盖(bf16/fp16/fp32) =====")
    K, MIX = 7168, 24  # DeepSeek 隐藏维
    for dt in X_DTYPES:
        x_flat, hc_fn = make_inputs(128, K, MIX, x_dtype=dt, seed=7)
        check_case(f"dtype {str(dt).replace('torch.','')}", x_flat, hc_fn, ORIG_EPS, device)
    print("✅ 多 dtype 全部通过")


def test_batch_invariance(device):
    print("\n===== 5) 批不变性(逐 bit)=====")
    for (tag, k, mix, m) in [
        ("Llama3-8B", 4096, 24, 129),
        ("DeepSeek",  7168, 15, 2049),
        ("Qwen2.5-7B",3584, 8,  512),
    ]:
        x_flat, hc_fn = make_inputs(m, k, mix, x_dtype=torch.bfloat16, seed=k)
        check_batch_invariance(f"{tag} M={m}", x_flat, hc_fn, ORIG_EPS, device)
    print("✅ 批不变性全部通过")


def test_edge(device):
    print("\n===== 6) 边界: eps 极端值 / M=0 空张量 =====")
    K, MIX = 4096, 24
    # 大 eps
    x_flat, hc_fn = make_inputs(16, K, MIX, seed=1)
    check_case("eps=1e-2", x_flat, hc_fn, 1e-2, device)
    # M=0
    x0, hc0 = make_inputs(0, K, MIX, seed=2)
    out0 = hc_mix_stats(x0.to(device), hc0.to(device), ORIG_EPS)
    assert tuple(out0.shape) == (0, MIX), out0.shape
    print(f"[M=0 空张量] -> out{tuple(out0.shape)} ok")
    print("✅ 边界用例通过")


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    test_original_spec(device)
    test_mainstream_matrix(device)
    test_batch_shapes(device)
    test_dtypes(device)
    test_batch_invariance(device)
    test_edge(device)
    print("\n🎉 全部测试通过 (rtol=atol=2e-2, 批不变性逐 bit);无需 ./hc_mix_stats_inputs.pt")
