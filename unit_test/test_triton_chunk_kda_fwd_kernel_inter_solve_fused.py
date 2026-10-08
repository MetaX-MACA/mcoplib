"""
Accuracy + performance benchmark for chunk_kda_fwd_kernel_inter_solve_fused.

用法:
  python test_triton_chunk_kda_fwd_kernel_inter_solve_fused.py                 # 精度 + 性能
  python test_triton_chunk_kda_fwd_kernel_inter_solve_fused.py --acc-only
  python test_triton_chunk_kda_fwd_kernel_inter_solve_fused.py --perf-only --fuse-recompute --fuse-diagonal

说明:
  - 参考实现全部在 float64 下完成，chunk 内先减去首行 g 做数值稳定化。
  - 输入采用真实 KDA 工作点：q/k 做 L2 归一化、门控每步衰减很小、beta 收缩，
    保证 M = tril(beta*K K^T, -1) 谱半径远小于 1，(I - M)^{-1} 良态。
  - 比对前有条件数护栏：输入病态时标记 SKIP，而不是误判 kernel 失败。
"""

import argparse
import functools
import itertools
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import triton

from mcoplib.triton_chunk_kda_fwd_kernel_inter_solve_fused import (
    chunk_kda_fwd_kernel_inter_solve_fused,
)

BT = 64
BC = 16
K = 128
V = 128
NC = BT // BC  # 4

# 下三角块 (row_sub, col_sub)，Akk_inv 实际写出的 10 个块
TRIL_BLOCKS = [(i, j) for i in range(NC) for j in range(i + 1)]

# bf16 输入 + tf32 求逆下的默认判据
ATOL_DEFAULT = 2e-2
RTOL_DEFAULT = 5e-3
RTOL_RECOMPUTE = 1e-2  # u/w 多一层 Ainv 累积

# 条件数护栏阈值
AINV_MAX_LIMIT = 1e3


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _tensor_cache(fn: Callable[..., Any]) -> Callable[..., Any]:
    cache_entries = []

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries
        for i, (old_args, old_kwargs, result) in enumerate(cache_entries):
            if len(args) == len(old_args) and len(kwargs) == len(old_kwargs):
                if all(a is b for a, b in zip(args, old_args)) and all(
                    key in old_kwargs and value is old_kwargs[key]
                    for key, value in kwargs.items()
                ):
                    cache_entries.append(cache_entries.pop(i))
                    return result
        result = fn(*args, **kwargs)
        if len(cache_entries) >= 4:
            cache_entries.pop(0)
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


@_tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    indices = torch.cat(
        [torch.arange(int(n)) for n in triton.cdiv(lengths, chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def _chunkify(x: torch.Tensor) -> torch.Tensor:
    """[B, T, H, D] -> [B, H, NT, BT, D]"""
    B, T, H, D = x.shape
    assert T % BT == 0, f"T={T} 必须是 BT={BT} 的整数倍"
    return x.view(B, T // BT, BT, H, D).permute(0, 3, 1, 2, 4).contiguous()


def _unchunkify(x: torch.Tensor) -> torch.Tensor:
    """[B, H, NT, BT, D] -> [B, T, H, D]"""
    B, H, NT, bt, D = x.shape
    return x.permute(0, 2, 3, 1, 4).reshape(B, NT * bt, H, D).contiguous()


# --------------------------------------------------------------------------- #
# reference (float64)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def reference(q, k, v, g, beta, scale, fuse_recompute, fuse_diagonal, use_safe_gate):
    """
    返回 dict:
      Aqk      [B, T, H, BT]  下三角块（FUSE_DIAGONAL=False 时对角块无效）
      Ainv     [B, T, H, BT]  下三角块 = (I - tril(beta*K K^T * gate, -1))^{-1}
      Akkd_in  [B, T, H, BC]  kernel 的输入（非 fuse_diagonal 时）
      Akkd_out [B, T, H, BC]  kernel 的输出（fuse_diagonal 时）
      M_max    float          max|M|，用于条件数护栏
      u, w, kg [B, T, H, V/K] （fuse_recompute 时）
    """
    f64 = torch.float64
    qc = _chunkify(q.to(f64))
    kc = _chunkify(k.to(f64))
    vc = _chunkify(v.to(f64))
    gc = _chunkify(g.to(f64))
    bc = _chunkify(beta.to(f64).unsqueeze(-1)).squeeze(-1)  # [B,H,NT,BT]

    # 以 chunk 首行为基准做数值稳定化
    grel = gc - gc[..., 0:1, :]  # <= 0

    qg = qc * torch.exp2(grel)
    kg_pos = kc * torch.exp2(grel)
    kg_neg = kc * torch.exp2(-grel)

    i_idx = torch.arange(BT, device=q.device)
    causal = i_idx[:, None] >= i_idx[None, :]
    strict = i_idx[:, None] > i_idx[None, :]
    zero = torch.zeros((), dtype=f64, device=q.device)

    Aqk = (qg @ kg_neg.transpose(-1, -2)) * scale
    Aqk = torch.where(causal, Aqk, zero)

    Mkk = (kg_pos @ kg_neg.transpose(-1, -2)) * bc[..., None]
    Mkk = torch.where(strict, Mkk, zero)

    eye = torch.eye(BT, dtype=f64, device=q.device)
    Ainv = torch.linalg.solve_triangular(eye + Mkk, eye.expand_as(Mkk), upper=False)
    Ainv = torch.tril(Ainv)

    # Akkd 布局 [T, BC]：每行存该 sub-chunk 内的 BC 个值
    eyec = torch.eye(BC, dtype=f64, device=q.device)
    diag_in = torch.empty(*Mkk.shape[:-1], BC, dtype=f64, device=q.device)
    diag_out = torch.empty_like(diag_in)
    for c in range(NC):
        sl = slice(c * BC, (c + 1) * BC)
        blk = Mkk[..., sl, sl]  # strictly lower
        diag_out[..., sl, :] = blk
        if use_safe_gate and not fuse_diagonal:
            # 此路径下 kernel 跳过前代回代，直接把 Akkd 当作"已求逆"的对角块
            diag_in[..., sl, :] = torch.linalg.solve_triangular(
                eyec + blk, eyec.expand_as(blk), upper=False
            )
        else:
            diag_in[..., sl, :] = blk

    out: Dict[str, Any] = {
        "Aqk": _unchunkify(Aqk),
        "Ainv": _unchunkify(Ainv),
        "Akkd_in": _unchunkify(diag_in),
        "Akkd_out": _unchunkify(diag_out),
        "M_max": Mkk.abs().max().item(),
    }

    if fuse_recompute:
        out["u"] = _unchunkify(Ainv @ (vc * bc[..., None]))
        out["w"] = _unchunkify(Ainv @ (kc * bc[..., None] * torch.exp2(gc)))
        out["kg"] = _unchunkify(kc * torch.exp2(gc[..., -1:, :] - gc))
    return out


# --------------------------------------------------------------------------- #
# inputs / runner
# --------------------------------------------------------------------------- #
def make_inputs(B, T, H, varlen, fuse_diagonal, device="cuda", dtype=torch.bfloat16,
                ref_akkd: Optional[torch.Tensor] = None, seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)

    # q/k 做 L2 归一化（真实 KDA 工作点），保证 |k_i . k_j| <= 1
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    q = torch.nn.functional.normalize(q, dim=-1).to(dtype)
    k = torch.nn.functional.normalize(k, dim=-1).to(dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)

    # 门控每步衰减放小，使 chunk(BT=64) 内 |dg| <= ~1.5，避免 exp2 动态范围爆炸
    g = -torch.rand(B, T, H, K, device=device, dtype=torch.float32) * 0.02
    g = g.cumsum(dim=1).clamp_min(-60.0).contiguous()

    # beta 收缩，保证 I - tril(beta*K K^T, -1) 良态
    beta = (torch.rand(B, T, H, device=device, dtype=torch.float32) * 0.5).contiguous()

    Aqk = (torch.zeros if fuse_diagonal else torch.empty)(
        B, T, H, BT, device=device, dtype=dtype
    )
    if ref_akkd is not None:
        Akkd = ref_akkd.to(torch.float32).contiguous()
    else:
        Akkd = torch.randn(B, T, H, BC, device=device, dtype=torch.float32) * 0.01
    Akk = torch.zeros(B, T, H, BT, device=device, dtype=dtype)

    if varlen:
        assert B == 1, "varlen 模式下把 batch 打平成 B=1"
        cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
    else:
        cu_seqlens, chunk_indices = None, None
        NT = triton.cdiv(T, BT)

    return dict(q=q, k=k, v=v, g=g, beta=beta, Aqk=Aqk, Akkd=Akkd, Akk=Akk,
                cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, NT=NT)


def build_runner(B, T, H, x, safe_gate, fuse_recompute, fuse_diagonal):
    grid = (x["NT"], B * H)
    scale = K ** -0.5

    if fuse_recompute:
        x.setdefault("w", torch.empty_like(x["k"]))
        x.setdefault("u", torch.empty_like(x["v"]))
        x.setdefault("kg", torch.empty_like(x["k"]))
        kwargs = dict(v_in=x["v"], w_out=x["w"], u_out=x["u"], kg_out=x["kg"],
                      Akk=x["k"], V=V, FUSE_RECOMPUTE=True)
    else:
        kwargs = dict(v_in=x["k"], w_out=x["k"], u_out=x["k"], kg_out=x["k"],
                      Akk=x["Akk"], V=0, FUSE_RECOMPUTE=False)

    def run():
        chunk_kda_fwd_kernel_inter_solve_fused[grid](
            q=x["q"], k=x["k"], g=x["g"], beta=x["beta"],
            Aqk=x["Aqk"], Akkd=x["Akkd"], scale=scale,
            cu_seqlens=x["cu_seqlens"], chunk_indices=x["chunk_indices"],
            T=T, H=H, K=K, BT=BT, BC=BC,
            USE_SAFE_GATE=safe_gate, FUSE_DIAGONAL=fuse_diagonal, **kwargs,
        )

    return run


# --------------------------------------------------------------------------- #
# accuracy
# --------------------------------------------------------------------------- #
def _stat(got: torch.Tensor, ref: torch.Tensor) -> Tuple[float, float]:
    """张量级归一的相对误差（逐元素相对误差在大动态范围下无意义）。"""
    got = got.to(torch.float64)
    ref = ref.to(torch.float64)
    diff = (got - ref).abs()
    scale = ref.abs().max().clamp_min(1e-6)
    return diff.max().item(), (diff.max() / scale).item()


def _report(name: str, got: torch.Tensor, ref: torch.Tensor,
            atol: float, rtol: float) -> bool:
    if got.numel() == 0:
        return True
    a, r = _stat(got, ref)
    ok = (a <= atol) or (r <= rtol)
    print(f"    {name:<18} max_abs={a:.4e}  max_rel={r:.4e}  "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def _conditioning(m_max: float, ainv_max: float) -> bool:
    ok = ainv_max < AINV_MAX_LIMIT
    print(f"    {'conditioning':<18} max|M|={m_max:.3e}  max|Ainv|={ainv_max:.3e}  "
          f"[{'OK' if ok else 'ILL-CONDITIONED'}]")
    return ok


def _block_mask(T: int, device, with_diagonal: bool) -> torch.Tensor:
    """[T, BT] bool mask，标出 Aqk/Akk 中被写出的 sub-chunk 块。"""
    m = torch.zeros(BT, BT, dtype=torch.bool, device=device)
    for i, j in TRIL_BLOCKS:
        if i == j and not with_diagonal:
            continue
        m[i * BC:(i + 1) * BC, j * BC:(j + 1) * BC] = True
    return m.repeat(T // BT, 1)


@torch.no_grad()
def run_accuracy(B, T, H, varlen, safe_gate, fuse_recompute, fuse_diagonal,
                 atol=ATOL_DEFAULT, rtol=RTOL_DEFAULT) -> bool:
    scale = K ** -0.5
    seed = 1234

    # 先用同一 seed 生成一份输入算参考
    x0 = make_inputs(B, T, H, varlen, fuse_diagonal, seed=seed)
    ref = reference(x0["q"], x0["k"], x0["v"], x0["g"], x0["beta"], scale,
                    fuse_recompute, fuse_diagonal, safe_gate)

    # 再生成完全相同的输入喂给 kernel（非 fuse_diagonal 时 Akkd 用参考值）
    x = make_inputs(B, T, H, varlen, fuse_diagonal, seed=seed,
                    ref_akkd=None if fuse_diagonal else ref["Akkd_in"])

    print(f"[ACC] T={T}  H={H}  varlen={varlen}  safe_gate={safe_gate}  "
          f"fuse_recompute={fuse_recompute}  fuse_diagonal={fuse_diagonal}")

    ainv_max = ref["Ainv"].abs().max().item()
    if not _conditioning(ref["M_max"], ainv_max):
        print("    -> SKIP (输入病态，非 kernel 问题)\n")
        return True

    run = build_runner(B, T, H, x, safe_gate, fuse_recompute, fuse_diagonal)
    run()
    torch.cuda.synchronize()

    ok = True
    dev = x["q"].device

    m_aqk = _block_mask(T, dev, with_diagonal=fuse_diagonal)
    m_aqk = m_aqk.view(1, T, 1, BT).expand(B, T, H, BT)
    ok &= _report("Aqk", x["Aqk"][m_aqk], ref["Aqk"][m_aqk], atol, rtol)

    if fuse_diagonal:
        ok &= _report("Akkd (diag out)", x["Akkd"], ref["Akkd_out"], atol, rtol)

    if fuse_recompute:
        ok &= _report("u", x["u"], ref["u"], atol, RTOL_RECOMPUTE)
        ok &= _report("w", x["w"], ref["w"], atol, RTOL_RECOMPUTE)
        ok &= _report("kg", x["kg"], ref["kg"], atol, rtol)
    else:
        m_akk = _block_mask(T, dev, with_diagonal=True)
        m_akk = m_akk.view(1, T, 1, BT).expand(B, T, H, BT)
        ok &= _report("Akk_inv", x["Akk"][m_akk], ref["Ainv"][m_akk],
                      atol, RTOL_RECOMPUTE)

    print(f"    -> {'PASS' if ok else 'FAIL'}\n")
    return ok


# --------------------------------------------------------------------------- #
# perf model
# --------------------------------------------------------------------------- #
def kernel_flops(B, T, H, fuse_recompute, fuse_diagonal):
    NT = triton.cdiv(T, BT)
    f = 6 * 2 * (2 * BC * BC * K)                 # inter：6 个严格下三角对 x (Aqk+Akk)
    if fuse_diagonal:
        f += 4 * 2 * (2 * BC * BC * K)            # 4 个对角块 x (Aqk+Akk)
    f += 16 * (2 * BC * BC * BC)                  # block 下三角求逆
    if fuse_recompute:
        f += 10 * (2 * BC * BC * V)               # u = Ainv @ (beta*v)
        f += 10 * (2 * BC * BC * K)               # w = Ainv @ (beta*k*exp(g))
    return f * NT * B * H


def kernel_bytes(B, T, H, fuse_recompute):
    e2, e4 = 2, 4
    n = B * T * H
    rd = n * K * e2 * 2 + n * K * e4 + n * e4     # q, k, g, beta
    wr = n * BT * e2                              # Aqk
    if fuse_recompute:
        rd += n * V * e2                          # v
        wr += n * V * e2 + n * K * e2 * 2         # u, w, kg
    else:
        wr += n * BT * e2                         # Akk
        rd += n * BC * e4                         # Akkd
    return rd + wr                                # compulsory traffic 下界


@torch.no_grad()
def torch_baseline(x, scale, fuse_recompute):
    """fp32 PyTorch 基线（同数学）。"""
    qc = _chunkify(x["q"].float())
    kc = _chunkify(x["k"].float())
    vc = _chunkify(x["v"].float())
    gc = _chunkify(x["g"])
    bc = _chunkify(x["beta"].unsqueeze(-1)).squeeze(-1)
    grel = gc - gc[..., 0:1, :]
    qg = qc * torch.exp2(grel)
    kp = kc * torch.exp2(grel)
    kn = kc * torch.exp2(-grel)
    Aqk = torch.tril(qg @ kn.transpose(-1, -2)) * scale
    M = torch.tril((kp @ kn.transpose(-1, -2)) * bc[..., None], -1)
    eye = torch.eye(BT, device=qc.device)
    Ainv = torch.linalg.solve_triangular(eye - M, eye.expand_as(M), upper=False)
    if fuse_recompute:
        _ = Ainv @ (vc * bc[..., None])
        _ = Ainv @ (kc * bc[..., None] * torch.exp2(gc))
        _ = kc * torch.exp2(gc[..., -1:, :] - gc)
    return Aqk, Ainv


def run_perf(args):
    header = (f"{'TP':>4} {'H':>4} {'tokens':>7} {'NT':>5} {'ms':>9} "
              f"{'TFLOP/s':>9} {'GB/s':>8}")
    print(header)
    print("-" * len(header))
    scale = K ** -0.5
    combos = [
            # (fuse_recompute, fuse_diagonal, safe_gate)
            (False, False, False),
            (False, False, True),   # Akkd 为"已求逆"的对角块
            (True,  True,  False),
            (False, True,  False),
        ]
    for (fr, fd, sg) in combos:
        print(fr,fd,sg)
        for tp, tokens in itertools.product([8, 4], [2048, 3072, 4096, 8192, 16384]):
            H = 96 // tp
            B, T = args.batch, tokens
            x = make_inputs(B, T, H, args.varlen, fd)
            run = build_runner(B, T, H, x, sg,
                            fr, fd)
            run()  # warmup + autotune
            torch.cuda.synchronize()

            ms = triton.testing.do_bench(run, warmup=10, rep=50, quantiles=None)
            base_ms = triton.testing.do_bench(
                lambda: torch_baseline(x, scale, fr),
                warmup=3, rep=10, quantiles=None,
            )

            fl = kernel_flops(B, T, H, fr, fd)
            by = kernel_bytes(B, T, H, fr)
            print(f"{'TP'+str(tp):>4} {H:>4} {tokens:>7} {x['NT']:>5} "
                f"{ms:>9.4f} {fl/(ms*1e-3)/1e12:>9.2f} {by/(ms*1e-3)/1e9:>8.1f} ")

            del x, run
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--varlen", action="store_true")
    p.add_argument("--fuse-recompute", action="store_true")
    p.add_argument("--fuse-diagonal", action="store_true")
    p.add_argument("--safe-gate", action="store_true")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--acc-only", action="store_true")
    p.add_argument("--perf-only", action="store_true")
    args = p.parse_args()

    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"varlen={args.varlen} fuse_recompute={args.fuse_recompute} "
          f"fuse_diagonal={args.fuse_diagonal} safe_gate={args.safe_gate}\n")

    if not args.perf_only:
        all_ok = True
        combos = [
            # (fuse_recompute, fuse_diagonal, safe_gate)
            (False, False, False),
            (False, False, True),   # Akkd 为"已求逆"的对角块
            (True,  True,  False),
            (False, True,  False),
        ]
        for T, (fr, fd, sg) in itertools.product([256, 1024, 4096], combos):
            all_ok &= run_accuracy(B=1, T=T, H=4, varlen=args.varlen,
                                   safe_gate=sg, fuse_recompute=fr,
                                   fuse_diagonal=fd)
        print(f"=== ACCURACY: {'ALL PASS' if all_ok else 'SOME FAILED'} ===\n")
        if not all_ok and not args.acc_only:
            print("精度未通过，跳过性能测试。")
            return

    if not args.acc_only:
        run_perf(args)


if __name__ == "__main__":
    main()