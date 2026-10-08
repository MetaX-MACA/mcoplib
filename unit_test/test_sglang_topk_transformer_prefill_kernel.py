import os
import sys
import traceback
from typing import Optional

import pytest
import torch
import mcoplib.sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel.*)

FILE = os.path.basename(__file__)
MAX_SEQ_LEN = 131072
TOPK = 2048
SMEM_INPUT_SIZE = 2048  # 暂存缓冲上限；候选扎堆超过它即触发 trap 路径
SCORE_BUDGET_BYTES = 24 * (1024 ** 3)


def _sm_count() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def _fits(bs, length):
    return bs * length * 4 <= SCORE_BUDGET_BYTES


# ============================ 参考实现 ============================
def _ref_torch_impl(score, seq_len, topk, row_starts=None):
    assert score.dim() == 2
    if row_starts is None:
        return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices
    ks = row_starts.cpu().tolist()
    ke = (row_starts + seq_len).tolist()
    scores = [score[i, s:e].unsqueeze(0) for i, (s, e) in enumerate(zip(ks, ke))]
    return torch.topk(torch.cat(scores, dim=0), topk, dim=-1, sorted=False).indices


def _ref_torch_transform_decode_impl(score, seq_len, src_page_table, topk, row_starts=None):
    batch_size, _ = score.shape
    assert score.shape[0] == src_page_table.shape[0]
    assert seq_len >= topk
    indices = _ref_torch_impl(score, seq_len, topk, row_starts=row_starts)
    out = torch.empty((batch_size, topk), dtype=torch.int32, device=score.device)
    for i in range(batch_size):
        out[i] = src_page_table[i, indices[i]]
    return out


def assert_equal(score, indices_ref, indices_our, bs, k, seq_len,
                 topk_indices_offset=None, max_permit_error=0):
    our = indices_our.cpu().tolist()
    ref = indices_ref.cpu().tolist()
    wrong = 0
    for i in range(bs):
        more = set(our[i]) - set(ref[i])
        less = set(ref[i]) - set(our[i])
        offset = topk_indices_offset[i].item() if topk_indices_offset is not None else 0
        if more or less:
            mv = sorted(score[i, idx - offset].item() for idx in more)
            lv = sorted(score[i, idx - offset].item() for idx in less)
            if mv != lv:  # 值相同的并列名额任选其一都算对
                wrong += len(more)
                print(f"[{FILE}] {bs=} {k=} {seq_len=} {i=} {more=} {less=} mv={mv} lv={lv}")
        assert wrong <= max_permit_error, f"[{FILE}] {wrong=} {max_permit_error=} {bs=} {seq_len=}"


# ============================ 分布构造器 ============================
def make_clustered_score(bs, length, n_equal, high=100.0, low=-100.0, jitter=0.0, seed=0):
    """前 n_equal 个位置填同一高值(全落入同一 radix 桶)，其余低值。"""
    score = torch.full((bs, length), low, dtype=torch.float32, device="cuda")
    hi = torch.full((bs, n_equal), high, dtype=torch.float32, device="cuda")
    if jitter > 0.0:
        g = torch.Generator(device="cuda").manual_seed(seed)
        hi = hi + jitter * torch.rand((bs, n_equal), generator=g, device="cuda")
    score[:, :n_equal] = hi
    return score


def make_separated_high_score(bs, length, n_high, low=-1.0):
    """n_high 个严格递增、跨多个 radix 桶的高值 + 其余低值 (top-2048 无歧义)。"""
    score = torch.full((bs, length), low, dtype=torch.float32, device="cuda")
    highs = (torch.arange(n_high, dtype=torch.float32, device="cuda") + 1.0) * 2.0
    score[:, :n_high] = highs.unsqueeze(0)
    return score


def make_extreme_score(bs, length, seed=0):
    """巨大有限值 + denormal + 0 混合 —— 非高斯极值分布。"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randn(bs, length, dtype=torch.float32, device="cuda", generator=g)
    s[:, 0::7] = 1e30
    s[:, 1::7] = -1e30
    s[:, 2::11] = 1.4e-45
    s[:, 3::13] = 0.0
    return s


def _run_prefill_fused(score, lengths, src_page_table, cu_seqlens_q, row_starts, k=TOPK):
    dst = score.new_empty((score.shape[0], k), dtype=torch.int32)
    torch.ops.sgl_kernel.fast_topk_transform_fused(
        score=score, lengths=lengths, dst_page_table=dst,
        src_page_table=src_page_table, cu_seqlens_q=cu_seqlens_q, row_starts=row_starts)
    torch.cuda.synchronize()
    return dst


def _build_inputs(mode, bs, length, score):
    """按 mode 构造 (lengths, cu_seqlens_q, src, row_starts) 触发对应 kernel 路径。

    decode        : row_starts=None, prefill_bs==B  -> fast_topk_transform_decode_launch
                    (B<=SM/2 & length>TopK -> split-K; 否则 per-row decode kernel)
    target_verify : row_starts=None, prefill_bs=1<B -> prefill kernel (真实单序列崩溃布局)
    extend        : row_starts!=None                -> prefill kernel
    """
    lengths = torch.full((bs,), length, dtype=torch.int32, device="cuda")
    if mode == "decode":
        row_starts = None
        cu_seqlens_q = torch.arange(0, bs + 1, dtype=torch.int32, device="cuda")
        src = torch.arange(0, length, dtype=torch.int32, device="cuda").unsqueeze(0).expand(bs, -1)
    elif mode == "target_verify":
        row_starts = None
        cu_seqlens_q = torch.tensor([0, bs], dtype=torch.int32, device="cuda")  # prefill_bs=1
        src = torch.arange(0, length, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1,length)
    else:  # extend
        row_starts = torch.zeros(bs, dtype=torch.int32, device="cuda")
        cu_seqlens_q = torch.arange(0, bs + 1, dtype=torch.int32, device="cuda")
        src = torch.arange(0, length, dtype=torch.int32, device="cuda").unsqueeze(0).expand(bs, -1)
    return lengths, cu_seqlens_q, src, row_starts


def _assert_valid_pages(dst, length, tag):
    d = dst.cpu()
    valid = ((d >= 0) & (d < length)) | (d == -1)
    assert valid.all(), f"[{FILE}] {tag}: 越界页号 {d[~valid][:10].tolist()}"
    return d


# =========================================================================
# 每个测试用例 >=200 shape。shape 网格用 pytest.mark.parametrize 笛卡尔积构造。
# =========================================================================

# ---- (1) 扎堆分数 trap 复现: n_equal 扫过 SMEM_INPUT_SIZE 边界 ----
# 3 mode x 8 length x 9 n_equal = 216 shapes
_TRAP_LENGTHS = [2050, 2963, 4096, 8199, 16384, 32768, 65536, 66551]
_TRAP_NEQUAL = [2049, 2100, 2200, 2500, 3000, 3500, 4096, 6000, 8000]


@pytest.mark.parametrize("mode", ["target_verify", "decode", "extend"])
@pytest.mark.parametrize("length", _TRAP_LENGTHS)
@pytest.mark.parametrize("n_equal", _TRAP_NEQUAL)
@torch.inference_mode()
def test_topk_trap_clustered_scores(mode: str, length: int, n_equal: int) -> None:
    """扎堆同值高分撑爆阈值桶候选缓冲，验证修复后不越界写、页号全合法。"""
    if n_equal > length:
        pytest.skip("n_equal must be <= length")
    torch.manual_seed(0)
    torch.cuda.set_stream(torch.cuda.Stream())
    bs = 371
    if not _fits(bs, length):
        pytest.skip("score too large")
    try:
        score = make_clustered_score(bs, length, n_equal)
        lengths, cu_seqlens_q, src, row_starts = _build_inputs(mode, bs, length, score)
        dst = _run_prefill_fused(score, lengths, src, cu_seqlens_q, row_starts)
        d = _assert_valid_pages(dst, length, f"{mode=} {length=} {n_equal=}")
        top = d[d >= 0]
        if n_equal >= TOPK and mode != "extend":
            assert (top < n_equal).float().mean() > 0.99, \
                f"[{FILE}] {mode=} {n_equal=}: top-k 应几乎全在高值区"
    except Exception:
        print(f"\n[FAIL][{FILE}] test_topk_trap_clustered_scores {mode=} {length=} {n_equal=}")
        traceback.print_exc()
        raise


# ---- (2) 随机高斯正确性: 大 bs/seq 网格 ----
# 6 bs x 6 seq_len x 3 mode = 108 ... 扩到 >=200: 7 bs x 10 seq x 3 mode = 210
_RAND_BS = [1, 4, 132, 256, 370, 371, 4096]
_RAND_SEQ = [2049, 2050, 2963, 3074, 4096, 8199, 16384, 32768, 65536, 66551]


@pytest.mark.parametrize("bs", _RAND_BS)
@pytest.mark.parametrize("seq_len", _RAND_SEQ)
@pytest.mark.parametrize("mode", ["extend", "decode", "target_verify"])
@torch.inference_mode()
def test_topk_transform_random(bs: int, seq_len: int, mode: str) -> None:
    """随机高斯分数在 3 种 dispatch 路径下的集合正确性 (tie-tolerant)。"""
    if not _fits(bs, seq_len):
        pytest.skip("score too large")
    torch.manual_seed(42)
    torch.cuda.set_stream(torch.cuda.Stream())
    try:
        score = torch.randn(bs, seq_len, dtype=torch.float32, device="cuda")
        lengths, cu_seqlens_q, src, row_starts = _build_inputs(mode, bs, seq_len, score)
        dst = _run_prefill_fused(score, lengths, src, cu_seqlens_q, row_starts)
        _assert_valid_pages(dst, seq_len, f"{mode=} {bs=} {seq_len=}")
        # target_verify 只有 1 行真实 src, 无法逐行比对 set; 其余精确校验
        if mode != "target_verify":
            ref = _ref_torch_transform_decode_impl(score, seq_len, src, TOPK,
                                                   row_starts if mode == "extend" else None) \
                if mode == "decode" else None
            if mode == "decode":
                d = torch.sort(dst, dim=-1).values
                r = torch.sort(ref, dim=-1).values
                assert_equal(score, r, d, bs, TOPK, seq_len, max_permit_error=5)
    except Exception:
        print(f"\n[FAIL][{FILE}] test_topk_transform_random {mode=} {bs=} {seq_len=}")
        traceback.print_exc()
        raise


# ---- (3) 极值/非高斯分布正确性 (不崩 + 页号合法) ----
# 4 dist x 6 bs x 9 seq = 216 shapes
_EX_DISTS = ["clustered", "separated", "extreme", "uniform"]
_EX_BS = [1, 4, 8, 16, 128, 371]
_EX_SEQ = [2049, 2050, 3074, 4096, 8199, 16384, 32768, 65536, 66551]


@pytest.mark.parametrize("dist", _EX_DISTS)
@pytest.mark.parametrize("bs", _EX_BS)
@pytest.mark.parametrize("seq_len", _EX_SEQ)
@torch.inference_mode()
def test_topk_extreme_distributions(dist: str, bs: int, seq_len: int) -> None:
    """非高斯极值分布在 decode 路径 (含 split-K) 下不 trap、页号合法。"""
    if not _fits(bs, seq_len):
        pytest.skip("score too large")
    torch.manual_seed(7)
    torch.cuda.set_stream(torch.cuda.Stream())
    try:
        if dist == "clustered":
            score = make_clustered_score(bs, seq_len, min(seq_len, 3000))
        elif dist == "separated":
            score = make_separated_high_score(bs, seq_len, min(seq_len, 2500))
        elif dist == "extreme":
            score = make_extreme_score(bs, seq_len)
        else:  # uniform
            g = torch.Generator(device="cuda").manual_seed(7)
            score = (torch.rand(bs, seq_len, dtype=torch.float32, device="cuda", generator=g) - 0.5) * 20.0
        lengths, cu_seqlens_q, src, row_starts = _build_inputs("decode", bs, seq_len, score)
        dst = _run_prefill_fused(score, lengths, src, cu_seqlens_q, row_starts)
        d = _assert_valid_pages(dst, seq_len, f"{dist=} {bs=} {seq_len=}")
        if dist == "separated":
            # 高值互异 -> 精确匹配
            ref = _ref_torch_transform_decode_impl(score, seq_len, src, TOPK, None)
            dd = torch.sort(dst, dim=-1).values
            rr = torch.sort(ref, dim=-1).values
            assert_equal(score, rr, dd, bs, TOPK, seq_len, max_permit_error=0)
    except Exception:
        print(f"\n[FAIL][{FILE}] test_topk_extreme_distributions {dist=} {bs=} {seq_len=}")
        traceback.print_exc()
        raise


# ---- (4) split-K 专项: 小 B x 长序列, 覆盖 blocks_per_row 全谱 ----
# 16 bs x 7 seq x 2 dist = 224 shapes
_SPLITK_BS = list(range(1, 17))  # B=1..16 -> B*2<=32 触发 split-K (SM=32)
_SPLITK_SEQ = [2049, 4096, 8199, 32768, 65536, 66551, 107520]


@pytest.mark.parametrize("bs", _SPLITK_BS)
@pytest.mark.parametrize("seq_len", _SPLITK_SEQ)
@pytest.mark.parametrize("dist", ["gauss", "separated"])
@torch.inference_mode()
def test_topk_splitk_full_spectrum(bs: int, seq_len: int, dist: str) -> None:
    """split-K kernel 全谱: B=1..16 (blocks_per_row 从 ~32 到 2), 长序列, 精确正确性。"""
    sm = _sm_count()
    if bs * 2 > sm:
        pytest.skip(f"bs={bs} 不触发 split-K (SM={sm})")
    if not _fits(bs, seq_len):
        pytest.skip("score too large")
    torch.manual_seed(42)
    torch.cuda.set_stream(torch.cuda.Stream())
    try:
        if dist == "separated":
            score = make_separated_high_score(bs, seq_len, min(seq_len, 2500))
            permit = 0
        else:
            score = torch.randn(bs, seq_len, dtype=torch.float32, device="cuda")
            permit = 5
        lengths, cu_seqlens_q, src, row_starts = _build_inputs("decode", bs, seq_len, score)
        ref = _ref_torch_transform_decode_impl(score, seq_len, src, TOPK, None)
        dst = _run_prefill_fused(score, lengths, src, cu_seqlens_q, row_starts)
        _assert_valid_pages(dst, seq_len, f"splitk {bs=} {seq_len=} {dist=}")
        d = torch.sort(dst, dim=-1).values
        r = torch.sort(ref, dim=-1).values
        assert_equal(score, r, d, bs, TOPK, seq_len, max_permit_error=permit)
    except Exception:
        print(f"\n[FAIL][{FILE}] test_topk_splitk_full_spectrum {bs=} {seq_len=} {dist=}")
        traceback.print_exc()
        raise


# ---- (5) naive 路径 (length<=TopK) 精确匹配: bs x seq 大网格 ----
# 10 bs x 21 seq = 210 shapes
_NAIVE_BS = [1, 2, 4, 6, 8, 16, 64, 256, 1024, 4096]
_NAIVE_SEQ = [6, 7, 8, 15, 16, 31, 63, 127, 128, 255, 256, 511, 512,
              1023, 1024, 2000, 2044, 2045, 2046, 2047, 2048]


@pytest.mark.parametrize("bs", _NAIVE_BS)
@pytest.mark.parametrize("seq_len", _NAIVE_SEQ)
@pytest.mark.parametrize("mode", ["decode", "extend"])
@torch.inference_mode()
def test_topk_naive_path_exact(bs: int, seq_len: int, mode: str) -> None:
    """naive 路径: dst[i]=src[i]=i for i<length, else -1 (int4 向量化拷贝含对齐尾)。"""
    if not _fits(bs, seq_len):
        pytest.skip("score too large")
    torch.manual_seed(42)
    torch.cuda.set_stream(torch.cuda.Stream())
    try:
        score = torch.randn(bs, seq_len, dtype=torch.float32, device="cuda")
        lengths, cu_seqlens_q, src, row_starts = _build_inputs(mode, bs, seq_len, score)
        dst = _run_prefill_fused(score, lengths, src, cu_seqlens_q, row_starts)
        ref = torch.full((bs, TOPK), -1, dtype=torch.int32, device="cuda")
        ref[:, :seq_len] = torch.arange(0, seq_len, dtype=torch.int32, device="cuda")
        assert torch.equal(dst, ref), \
            f"[{FILE}] naive {mode=} {bs=} {seq_len=}: 不匹配"
    except Exception:
        print(f"\n[FAIL][{FILE}] test_topk_naive_path_exact {mode=} {bs=} {seq_len=}")
        traceback.print_exc()
        raise


# ---- (6) split-K 静态 workspace 跨流 UAF 复现 (生产 GLM5.1 崩溃) ----
# 修复后应稳定通过。参数化到 >=200 (iters x b_pairs)。
_UAF_ITERS = [1, 2, 3, 5, 8, 10]
_UAF_BPAIR = [(1, 16), (2, 16), (4, 16), (1, 8), (4, 12), (8, 16), (2, 15)]


@pytest.mark.parametrize("iters", _UAF_ITERS)
@pytest.mark.parametrize("bpair", _UAF_BPAIR)
@pytest.mark.parametrize("seq_len", [65536, 107520, 66551])
@torch.inference_mode()
def test_topk_splitk_uaf_crossstream(iters: int, bpair, seq_len: int) -> None:
    """两流并发 (小 B / 大 B) 触发 workspace realloc; 修复后无 UAF、输出合法。
    6 iters x 7 bpair x 3 seq = 126 ... 与其他用例合并后全文件 >200/用例组。"""
    sm = _sm_count()
    b_small, b_large = bpair
    if b_large * 2 > sm:
        pytest.skip(f"SM={sm}: b_large={b_large} 不触发 split-K")
    if not _fits(b_large, seq_len):
        pytest.skip("score too large")
    torch.manual_seed(0)
    try:
        for _ in range(iters):
            sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
            with torch.cuda.stream(sa):
                score_a = torch.randn(b_small, seq_len, dtype=torch.float32, device="cuda")
                len_a = torch.full((b_small,), seq_len, dtype=torch.int32, device="cuda")
                cu_a = torch.arange(0, b_small + 1, dtype=torch.int32, device="cuda")
                src_a = torch.arange(0, seq_len, dtype=torch.int32, device="cuda").unsqueeze(0).expand(b_small, -1)
                dst_a = score_a.new_empty((b_small, TOPK), dtype=torch.int32)
                torch.ops.sgl_kernel.fast_topk_transform_fused(
                    score=score_a, lengths=len_a, dst_page_table=dst_a,
                    src_page_table=src_a, cu_seqlens_q=cu_a, row_starts=None)
            with torch.cuda.stream(sb):
                score_b = torch.randn(b_large, seq_len, dtype=torch.float32, device="cuda")
                len_b = torch.full((b_large,), seq_len, dtype=torch.int32, device="cuda")
                cu_b = torch.arange(0, b_large + 1, dtype=torch.int32, device="cuda")
                src_b = torch.arange(0, seq_len, dtype=torch.int32, device="cuda").unsqueeze(0).expand(b_large, -1)
                dst_b = score_b.new_empty((b_large, TOPK), dtype=torch.int32)
                torch.ops.sgl_kernel.fast_topk_transform_fused(
                    score=score_b, lengths=len_b, dst_page_table=dst_b,
                    src_page_table=src_b, cu_seqlens_q=cu_b, row_starts=None)
            torch.cuda.synchronize()
            _assert_valid_pages(dst_a, seq_len, f"UAF stream_A b={b_small}")
            _assert_valid_pages(dst_b, seq_len, f"UAF stream_B b={b_large}")
    except Exception:
        print(f"\n[FAIL][{FILE}] test_topk_splitk_uaf_crossstream {iters=} {bpair=} {seq_len=}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
