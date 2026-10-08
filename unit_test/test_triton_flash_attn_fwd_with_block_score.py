"""
Benchmark & accuracy test for `_flash_attn_fwd_with_block_score_kernel`
and `_topk_index_kernel`, with a PyTorch baseline (标杆).

Usage:
    python test_triton_flash_attn_fwd_with_block_score.py --mode perf
    python test_triton_flash_attn_fwd_with_block_score.py --mode acc
    python test_triton_flash_attn_fwd_with_block_score.py --mode both \
        --chunk-sizes 2048 4096 8192 --score-type max --no-baseline-perf
"""

import argparse
import torch
import torch.nn.functional as F
import triton

from mcoplib.triton_flash_attn_fwd_with_block_score import (  # noqa: F401
    _flash_attn_fwd_with_block_score_kernel,
    _topk_index_kernel,
    flash_prefill_with_topk_index,
    get_cu_seqblocks,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
LOG2E = 1.4426950409

CHUNK_SIZES = [2048, 4096, 8192, 16384, 32768]
NUM_HEADS, NUM_KV_HEADS = 1, 1
QK_HEAD_DIM, V_HEAD_DIM = 128, 128
BLOCK_SIZE_K = 32       # block_size (score 粒度)
BLOCK_SIZE_Q = 64       # sample_interval
TOPK, INIT_BLOCKS, LOCAL_BLOCKS = 64, 1, 2


# ===========================================================================
# 1. 输入构造
# ===========================================================================
def make_inputs(chunk_size, prefix_len=None, disable_index_value=True, seed=0):
    torch.manual_seed(seed)
    prefix_len = chunk_size if prefix_len is None else prefix_len
    total_q, seq_len = chunk_size, prefix_len + chunk_size
    max_slots = seq_len + 1024

    q = torch.randn(total_q, NUM_HEADS, QK_HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k_cache = torch.randn(max_slots, NUM_KV_HEADS, QK_HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v_cache = None if disable_index_value else torch.randn(
        max_slots, NUM_KV_HEADS, V_HEAD_DIM, device=DEVICE, dtype=DTYPE
    )

    req_to_token = torch.zeros(2, max_slots, device=DEVICE, dtype=torch.int32)
    req_to_token[0, :seq_len] = torch.randperm(max_slots, device=DEVICE)[:seq_len].to(torch.int32)

    return dict(
        q=q, k_cache=k_cache, v_cache=v_cache, sink=None,
        req_to_token=req_to_token,
        slot_ids=torch.tensor([0], device=DEVICE, dtype=torch.int32),
        cu_seqlens=torch.tensor([0, total_q], device=DEVICE, dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], device=DEVICE, dtype=torch.int32),
        prefix_lens=torch.tensor([prefix_len], device=DEVICE, dtype=torch.int32),
        total_q=total_q, seq_len=seq_len, prefix_len=prefix_len, max_slots=max_slots,
        max_seqlen_q=chunk_size, max_seqlen_k=seq_len,
        disable_index_value=disable_index_value,
    )


def alloc_outputs(inp):
    nblk_k = triton.cdiv(inp["max_seqlen_k"], BLOCK_SIZE_K)
    score = torch.full((NUM_HEADS, nblk_k, inp["total_q"]), float("-inf"),
                       dtype=torch.float32, device=DEVICE)
    o = None if inp["disable_index_value"] else torch.empty(
        inp["total_q"], NUM_HEADS, V_HEAD_DIM, dtype=DTYPE, device=DEVICE)
    return score, o


# ===========================================================================
# 2. Triton kernel launchers
# ===========================================================================
def launch_flash_kernel(inp, score, o, score_type="max"):
    q, k_cache, v_cache = inp["q"], inp["k_cache"], inp["v_cache"]
    batch_size = inp["cu_seqlens"].numel() - 1
    sm_scale = QK_HEAD_DIM ** -0.5

    def grid(META):
        return (triton.cdiv(inp["max_seqlen_q"], META["BLOCK_SIZE_Q"]),
                batch_size * NUM_HEADS)

    _flash_attn_fwd_with_block_score_kernel[grid](
        q, k_cache, v_cache, None, o, score, inp["req_to_token"],
        inp["cu_seqlens"], inp["seq_lens"], inp["prefix_lens"], inp["slot_ids"],
        inp["max_slots"], NUM_HEADS, NUM_HEADS // NUM_KV_HEADS, QK_HEAD_DIM,
        QK_HEAD_DIM if inp["disable_index_value"] else V_HEAD_DIM,
        BLOCK_SIZE_K, sm_scale, False, 1,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        *(v_cache.stride() if v_cache is not None else (0, 0, 0)),
        0, 0,
        *(o.stride() if o is not None else (0, 0, 0)),
        score.stride(0), score.stride(2), score.stride(1),
        inp["req_to_token"].stride(0),
        SCORE_TYPE=score_type,
        DISABLE_INDEX_VALUE=inp["disable_index_value"],
    )


def launch_topk_kernel(inp, score, topk_idx, cu_seqblocks_q, max_seqblock_q):
    grid = (max_seqblock_q, inp["cu_seqlens"].numel() - 1, NUM_HEADS)
    _topk_index_kernel[grid](
        score, topk_idx, BLOCK_SIZE_Q, BLOCK_SIZE_K,
        inp["cu_seqlens"], cu_seqblocks_q, inp["prefix_lens"],
        TOPK, INIT_BLOCKS, LOCAL_BLOCKS,
        score.stride(0), score.stride(2), score.stride(1),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        MASK_INIT=False, MASK_LOCAL=False,
    )


# ===========================================================================
# 3. PyTorch 标杆 (baseline)
# ===========================================================================
# ===========================================================================
# 3. PyTorch 标杆 (baseline)
# ===========================================================================
def _gather_kv(inp):
    slots = inp["req_to_token"][0, : inp["seq_len"]].long()
    k = inp["k_cache"][slots, 0, :]
    v = inp["v_cache"][slots, 0, :] if inp["v_cache"] is not None else None
    return k, v


def baseline_score_and_out(inp, score_type="max", q_tile=1024):
    """
    标杆：分块 (q_tile) 计算 QK^T，得到
      - block score [1, nblk_k, n]  (max 或 lse，log2 域，与 kernel 对齐)
      - attention output [n, 1, vd]  (disable_index_value=False 时)
    """
    n, p, s = inp["total_q"], inp["prefix_len"], inp["seq_len"]
    sm = QK_HEAD_DIM ** -0.5 * LOG2E
    k, v = _gather_kv(inp)
    kf = k.float()
    vf = None if v is None else v.float()
    nblk = triton.cdiv(s, BLOCK_SIZE_K)
    pad = nblk * BLOCK_SIZE_K - s

    score = torch.full((nblk, n), float("-inf"), device=DEVICE, dtype=torch.float32)
    out = None if vf is None else torch.empty(n, V_HEAD_DIM, device=DEVICE, dtype=torch.float32)
    k_abs = torch.arange(s, device=DEVICE)
    NEG = float("-inf")

    for st in range(0, n, q_tile):
        ed = min(st + q_tile, n)
        qf = inp["q"][st:ed, 0, :].float()
        qk = (qf @ kf.T) * sm                                    # [t, s]，log2 域
        q_abs = torch.arange(st, ed, device=DEVICE) + p
        qk = qk.masked_fill(q_abs[:, None] < k_abs[None, :], NEG)

        qkp = F.pad(qk, (0, pad), value=NEG) if pad else qk
        tiles = qkp.view(ed - st, nblk, BLOCK_SIZE_K)
        m = tiles.max(dim=-1).values                             # [t, nblk]
        if score_type == "max":
            blk = m
        else:  # lse (log2 域)
            lse = m + torch.log2(torch.exp2(tiles - m[..., None]).sum(-1))
            blk = torch.where(m == NEG, m, lse)                  # 全 -inf 的块保持 -inf
        score[:, st:ed] = blk.T

        if vf is not None:
            row_max = qk.max(dim=-1, keepdim=True).values
            pmat = torch.exp2(qk - row_max)
            pmat = pmat / pmat.sum(dim=-1, keepdim=True)
            out[st:ed] = pmat @ vf                               # 全 fp32，避免 dtype 不匹配

    return score.unsqueeze(0), (None if out is None else out.unsqueeze(1).to(DTYPE))


def baseline_topk(score, inp, all_seqblock_q):
    """
    标杆 topk：完整复现 _topk_index_kernel 的语义
      - 只在 valid_blocks 内取
      - MASK_INIT=False  -> init_blocks 的分数改写为 1e30（强制入选）
      - MASK_LOCAL=False -> local_blocks 的分数改写为 1e29（强制入选，覆盖 init）
      - 存储列数为 min(topk, valid_blocks)，其余为 -1
    """
    n, p = inp["total_q"], inp["prefix_len"]
    rows = torch.arange(0, n, BLOCK_SIZE_Q, device=DEVICE)[:all_seqblock_q]
    s = score[0][:, rows].T.float().clone()                      # [nq, nblk]
    s = torch.nan_to_num(s, nan=-1e30, neginf=-1e30)

    valid = (p + rows + BLOCK_SIZE_K) // BLOCK_SIZE_K            # 与 kernel valid_blocks 一致
    blk = torch.arange(s.shape[1], device=DEVICE)[None, :]
    causal = blk < valid[:, None]

    s = torch.where(causal, s, torch.full_like(s, -1e30))
    # 与 kernel 中的写入顺序一致：先 init，后 local
    init_mask = causal & (blk < INIT_BLOCKS)
    s = torch.where(init_mask, torch.full_like(s, 1e30), s)
    local_lo = torch.clamp(valid - LOCAL_BLOCKS, min=0)[:, None]
    local_mask = causal & (blk >= local_lo)
    s = torch.where(local_mask, torch.full_like(s, 1e29), s)

    kk = min(TOPK, s.shape[1])
    idx = s.topk(kk, dim=-1).indices.to(torch.int32)
    out = torch.full((NUM_HEADS, all_seqblock_q, TOPK), -1, device=DEVICE, dtype=torch.int32)
    out[0, :, :kk] = idx

    # kernel 的存储 mask：arange(BLOCK_SIZE_T) < min(topk, valid_blocks)
    lim = torch.clamp(valid, max=TOPK)
    col = torch.arange(TOPK, device=DEVICE)
    out[0] = torch.where(col[None, :] < lim[:, None], out[0], torch.full_like(out[0], -1))
    return out


def baseline_sdpa(inp):
    """额外标杆：torch SDPA causal attention（仅 disable_index_value=False 时有意义）。"""
    k, v = _gather_kv(inp)
    n, p, s = inp["total_q"], inp["prefix_len"], inp["seq_len"]
    q = inp["q"][:, 0, :].unsqueeze(0).unsqueeze(0)              # [1,1,n,d]
    kk, vv = k.unsqueeze(0).unsqueeze(0), v.unsqueeze(0).unsqueeze(0)
    q_abs = torch.arange(n, device=DEVICE) + p
    mask = q_abs[:, None] >= torch.arange(s, device=DEVICE)[None, :]
    return F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask).squeeze(0).squeeze(0)


# ===========================================================================
# 4. FLOPs / 带宽模型
# ===========================================================================
def kv_pairs(inp):
    n, p = inp["total_q"], inp["prefix_len"]
    return n * p + n * (n + 1) // 2


def flash_flops(inp):
    f = 2 * NUM_HEADS * QK_HEAD_DIM * kv_pairs(inp)              # QK^T
    if not inp["disable_index_value"]:
        f += 2 * NUM_HEADS * V_HEAD_DIM * kv_pairs(inp)          # P@V
    return f


def flash_bytes(inp):
    elem = torch.finfo(DTYPE).bits // 8
    mult = 1 if inp["disable_index_value"] else 2
    return kv_pairs(inp) * QK_HEAD_DIM * elem * mult


# ===========================================================================
# 5. 精度模式
# ===========================================================================
def _report(name, got, ref, atol, mask=None):
    g, r = got.float(), ref.float()
    if mask is not None:
        g, r = g[mask], r[mask]
    d = torch.nan_to_num((g - r).abs(), nan=0.0, posinf=0.0)
    err = d.max().item()
    rel = (d / (r.abs() + 1e-6)).max().item()
    ok = err < atol
    print(f"    {name:<18} max_abs={err:.4e}  max_rel={rel:.4e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def run_accuracy(chunk_sizes, score_type, disable_index_value):
    all_ok = True
    for cs in chunk_sizes:
        print(f"\n[ACC] chunk_size={cs}  score_type={score_type}  "
              f"disable_index_value={disable_index_value}")
        inp = make_inputs(cs, disable_index_value=disable_index_value)
        score, o = alloc_outputs(inp)
        cu_sb, max_sb, all_sb, *_ = get_cu_seqblocks(
            inp["cu_seqlens"], inp["max_seqlen_q"], BLOCK_SIZE_Q, BLOCK_SIZE_K)
        topk_idx = torch.full((NUM_HEADS, all_sb, TOPK), -1, device=DEVICE, dtype=torch.int32)

        launch_flash_kernel(inp, score, o, score_type)
        launch_topk_kernel(inp, score, topk_idx, cu_sb, max_sb)
        torch.cuda.synchronize()
        ref_score, ref_o = baseline_score_and_out(inp, score_type)

        finite = torch.isfinite(ref_score)
        bad = (torch.isfinite(score) != finite).sum().item()
        print(f"    {'-inf pattern':<18} mismatch={bad}  [{'PASS' if bad == 0 else 'FAIL'}]")
        all_ok &= bad == 0
        all_ok &= _report("score", score, ref_score, 2e-2, finite)

        if ref_o is not None:
            all_ok &= _report("output", o, ref_o, 5e-2)
            all_ok &= _report("output vs sdpa", o[:, 0, :], baseline_sdpa(inp), 5e-2)

        ref_ti = baseline_topk(ref_score, inp, all_sb)
        # 顺序无关 + 并列分数容忍：按集合比较
        a = torch.sort(topk_idx[0], dim=-1).values
        b = torch.sort(ref_ti[0], dim=-1).values
        mism = (a != b).any(dim=-1).sum().item()
        rate = mism / max(a.shape[0], 1)
        ok = rate < 0.01
        print(f"    {'topk_idx set':<18} mismatch_rows={mism}/{a.shape[0]} "
              f"({rate:.2%})  [{'PASS' if ok else 'FAIL'}]")
        all_ok &= ok

        del inp, score, o, topk_idx, ref_score, ref_o, ref_ti, a, b
        torch.cuda.empty_cache()

    print(f"\n=== ACCURACY: {'ALL PASS' if all_ok else 'FAILED'} ===")
    return all_ok


# ===========================================================================
# 6. 性能模式
# ===========================================================================
def run_perf(chunk_sizes, score_type, disable_index_value):
    cols = f"{'chunk':>7} {'seq_len':>8} {'flash ms':>10} {'TFLOPS':>9} {'topk ms':>9} {'total ms':>9}"
    print("\n" + cols)
    print("-" * len(cols))

    for cs in chunk_sizes:
        inp = make_inputs(cs, disable_index_value=disable_index_value)
        score, o = alloc_outputs(inp)
        cu_sb, max_sb, all_sb, *_ = get_cu_seqblocks(
            inp["cu_seqlens"], inp["max_seqlen_q"], BLOCK_SIZE_Q, BLOCK_SIZE_K)
        topk_idx = torch.full((NUM_HEADS, all_sb, TOPK), -1, device=DEVICE, dtype=torch.int32)

        # warmup / autotune
        launch_flash_kernel(inp, score, o, score_type)
        launch_topk_kernel(inp, score, topk_idx, cu_sb, max_sb)
        torch.cuda.synchronize()

        ms_flash = triton.testing.do_bench(
            lambda: launch_flash_kernel(inp, score, o, score_type), warmup=25, rep=100)
        ms_topk = triton.testing.do_bench(
            lambda: launch_topk_kernel(inp, score, topk_idx, cu_sb, max_sb), warmup=25, rep=100)

        flops = flash_flops(inp)
        row = (f"{cs:>7} {inp['seq_len']:>8} {ms_flash:>10.3f} "
               f"{flops / ms_flash / 1e9:>9.2f} "
               f"{ms_topk:>9.3f} {ms_flash + ms_topk:>9.3f}")

        print(row)
        del inp, score, o, topk_idx
        torch.cuda.empty_cache()


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["perf", "acc", "both"], default="both")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=CHUNK_SIZES)
    ap.add_argument("--score-type", choices=["max", "lse"], default="max")
    ap.add_argument("--index-value", action="store_true",
                    help="启用 V/output 计算 (disable_index_value=False)")
    ap.add_argument("--no-baseline-perf", action="store_true", help="性能模式下不跑标杆")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA"
    print(torch.cuda.get_device_name(0))
    div = not args.index_value

    if args.mode in ("acc", "both"):
        run_accuracy(args.chunk_sizes, args.score_type, div)
    if args.mode in ("perf", "both"):
        run_perf(args.chunk_sizes, args.score_type, div)


if __name__ == "__main__":
    main()


# python bench_flash_prefill_with_topk_index.py --mode acc --acc-chunks 2048 4096
# python bench_flash_prefill_with_topk_index.py --mode perf --no-baseline-perf --e2e
# python bench_flash_prefill_with_topk_index.py --mode both --index-value --score-type lse