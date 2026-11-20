#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
from mcoplib.op import moe_finalize_routing
import pytest

# =============================================================================
# (1) 输入校验（ref 与 cuda 路径共用，对齐 aclnn 文档约束）
# =============================================================================

def _validate(expanded_x, expanded_row_idx, x1, x2, bias, scales, expert_idx,
              drop_pad_mode, k):
    if not (0 <= int(drop_pad_mode) <= 3):
        raise ValueError("drop_pad_mode must be in [0,3], got %d" % drop_pad_mode)
    drop_pad = int(drop_pad_mode) & 1
    if expanded_x.dim() != (3 if drop_pad else 2):
        raise ValueError(
            "expandedX must be (num_rows*K, H) for drop-less or (E, C, H) "
            "for drop-pad, got dim=%d with mode=%d" % (expanded_x.dim(), drop_pad_mode))
    if x2 is not None and x1 is None:
        raise ValueError("x2 requires x1")
    if bias is not None and expert_idx is None:
        raise ValueError("bias requires expert_idx")

    K = int(scales.shape[1]) if scales is not None else int(k)
    if K < 1:
        raise ValueError("K must be >= 1")
    if scales is not None and int(k) > 1 and int(k) != K:
        raise ValueError("k must match scales.shape[1] when k>1")
    num_rows = (int(scales.shape[0]) if scales is not None
                else int(expanded_row_idx.numel()) // K)
    H = int(expanded_x.shape[-1])
    if num_rows <= 0 or H <= 0:
        raise ValueError("bad num_rows/H")
    if int(expanded_row_idx.numel()) != num_rows * K:
        raise ValueError("expandedRowIdx must have num_rows*K elements")
    if not drop_pad and int(expanded_x.shape[0]) != num_rows * K:
        raise ValueError("drop-less expandedX must have num_rows*K rows")
    if expert_idx is not None and tuple(expert_idx.shape) != (num_rows, K):
        raise ValueError("expert_idx must be (num_rows, K)")
    if scales is not None and tuple(scales.shape) != (num_rows, K):
        raise ValueError("scales must be (num_rows, K)")
    if x1 is not None and tuple(x1.shape) != (num_rows, H):
        raise ValueError("x1 must be (num_rows, H)")
    if x2 is not None and tuple(x2.shape) != (num_rows, H):
        raise ValueError("x2 must be (num_rows, H)")
    if bias is not None and (bias.dim() != 2 or int(bias.shape[1]) != H):
        raise ValueError("bias must be (E, H)")
    if bias is not None and drop_pad and int(bias.shape[0]) != int(expanded_x.shape[0]):
        raise ValueError("bias E must match expandedX.shape[0] in drop-pad mode")

    dt = expanded_x.dtype
    if dt not in (torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn):
        raise ValueError("expandedX dtype must be fp8/fp16/bf16/fp32")
    for name, t in (("x1", x1), ("x2", x2), ("bias", bias)):
        if t is not None and t.dtype != dt:
            raise ValueError("%s dtype must match expandedX" % name)
    if scales is not None and scales.dtype not in (dt, torch.float32):
        raise ValueError("scales dtype must match expandedX or be fp32")
    return K, num_rows, H


# =============================================================================
# (2) 纯 PyTorch 参考实现（golden，逐 k 循环，fp32 累加，显存占用 ~num_rows*H）
# =============================================================================


def moe_finalize_routing_v2_ref(expanded_x, expanded_row_idx, *, x1=None, x2=None,
                                bias=None, scales=None, expert_idx=None,
                                drop_pad_mode=0, k=1):
    """纯 PyTorch 实现，CPU/GPU 均可运行。"""
    _validate(expanded_x, expanded_row_idx, x1, x2, bias, scales, expert_idx,
              drop_pad_mode, k)
    row_major = (int(drop_pad_mode) >> 1) & 1
    K = int(scales.shape[1]) if scales is not None else int(k)
    num_rows = (int(scales.shape[0]) if scales is not None
                else int(expanded_row_idx.numel()) // K)
    H = int(expanded_x.shape[-1])

    idx = expanded_row_idx.reshape(-1)
    if row_major:
        ridx = idx.reshape(num_rows, K)
    else:  # 列主序: (i, kk) -> idx[i + kk*num_rows]
        ridx = idx.reshape(K, num_rows).T
    x2d = expanded_x.reshape(-1, H)
    eidx2 = expert_idx.reshape(num_rows, K) if expert_idx is not None else None
    sc2 = scales.reshape(num_rows, K) if scales is not None else None

    acc = torch.zeros(num_rows, H, dtype=torch.float32, device=expanded_x.device)
    for kk in range(K):
        r = ridx[:, kk]
        valid = (r >= 0)
        rs = r.clamp_min(0).to(torch.long)
        t = x2d.index_select(0, rs).to(torch.float32)
        if bias is not None:
            t = t + bias.index_select(0, eidx2[:, kk].to(torch.long)).to(torch.float32)
        if sc2 is not None:
            t = t * sc2[:, kk].to(torch.float32).unsqueeze(1)
        acc += t * valid.unsqueeze(1).to(torch.float32)  # -1 丢弃：整项 0（含 bias）
    if x1 is not None:
        acc = acc + x1.to(torch.float32)
    if x2 is not None:
        acc = acc + x2.to(torch.float32)
    return acc.to(expanded_x.dtype).reshape(num_rows, H)

def moe_finalize_routing_v2_cuda(expanded_x, expanded_row_idx,  x1=None, x2=None,
                                 bias=None, scales=None, expert_idx=None,
                                 drop_pad_mode=0, k=1):
    num_rows = (int(scales.shape[0]) if scales is not None
                else int(expanded_row_idx.numel()) // k)
    H = int(expanded_x.shape[-1])
    output = torch.zeros(num_rows, H, dtype=expanded_x.dtype, device=expanded_x.device)
    moe_finalize_routing(output, expanded_x, expanded_row_idx, x1, x2, bias, scales, expert_idx, drop_pad_mode,k)
    return output


_TOL = {
    torch.float32: (1e-3, 1e-4),
    torch.float16: (2e-2, 1e-2),
    torch.bfloat16: (0.15, 2e-2),
    torch.float8_e4m3fn: (0.15,2e-2)
}

# nr, k, h, e, c, mode, x1, x2, bias, scales, eidx, scfp32, dtype, seed
_CASES = [
    ("mode0 col-major doc-example-shape", 3, 2, 4, 2, 0, 0, 1, 1, 1, 1, 1, 0, torch.float16, 1),
    ("mode2 row-major scalar-path", 3, 2, 4, 2, 0, 2, 1, 1, 1, 1, 1, 0, torch.float16, 2),
    ("mode1 drop-pad col-major", 17, 3, 256, 8, 64, 1, 1, 1, 1, 1, 1, 0, torch.float16, 3),
    ("mode3 drop-pad row-major", 33, 2, 1024, 4, 32, 3, 1, 1, 1, 1, 1, 0, torch.float16, 4),
    ("mode0 minimal k=1 no-optional", 5, 1, 64, 2, 0, 0, 0, 0, 0, 0, 0, 0, torch.float16, 5),
    ("mode2 x1+scales only", 9, 4, 128, 8, 0, 2, 1, 0, 0, 1, 0, 0, torch.float16, 6),
    ("mode0 mixed scales-fp32", 13, 2, 512, 4, 0, 0, 1, 1, 1, 1, 1, 1, torch.float16, 7),
    ("mode2 large NR=1024 K=8", 1024, 8, 2048, 16, 0, 2, 1, 1, 1, 1, 1, 0, torch.float16, 10),
    ("mode0 single-token row", 1, 1, 2048, 2, 0, 0, 1, 1, 1, 1, 1, 0, torch.float16, 11),
    ("mode1 drop-pad no-bias", 25, 3, 512, 8, 32, 1, 1, 1, 0, 1, 0, 0, torch.float16, 14),
    ("mode3 drop-pad fp32", 21, 2, 768, 4, 16, 3, 1, 1, 1, 1, 1, 0, torch.float32, 8),
    ("mode0 odd-H scalar-path fp32", 4, 2, 6, 3, 0, 0, 1, 1, 1, 1, 1, 0, torch.float32, 9),
    ("mode0 bf16", 7, 2, 256, 4, 0, 0, 1, 1, 1, 1, 1, 0, torch.bfloat16, 12),
    ("mode3 drop-pad bf16 scales-fp32", 11, 2, 512, 4, 8, 3, 1, 1, 1, 1, 1, 1, torch.bfloat16, 13),
    ("mode3 drop-pad float8_e4m3fn scales-fp32", 11, 2, 512, 4, 8, 3, 1, 1, 1, 1, 1, 1, torch.float8_e4m3fn, 13),
]

def _gen_case(c):
    (name, nr, K, H, E, C, mode, hx1, hx2, hbias, hscales, heidx, scfp32,
     dtype, seed) = c
    g = torch.Generator().manual_seed(seed)
    drop_pad = mode & 1
    ex_rows = E * C if drop_pad else nr * K

    def rnd(shape):
        return (torch.rand(shape, generator=g, dtype=torch.float32) * 2 - 1).to(dtype)

    exX = rnd((E, C, H)) if drop_pad else rnd((ex_rows, H))  # drop-pad 要求 3D
    if drop_pad:
        m = torch.rand(nr * K, generator=g) < 0.15
        ridx = torch.where(m, torch.full((nr * K,), -1, dtype=torch.int64),
                           torch.randint(0, ex_rows, (nr * K,), generator=g)).to(torch.int32)
    else:
        ridx = torch.randperm(ex_rows, generator=g).to(torch.int32)
    eidx = torch.randint(0, E, (nr, K), generator=g, dtype=torch.int32)
    x1 = rnd((nr, H)) if hx1 else None
    x2 = rnd((nr, H)) if hx2 else None
    bias = rnd((E, H)) if hbias else None
    scales = None
    if hscales:
        scales = (0.5 + 1.5 * torch.rand((nr, K), generator=g)).to(
            torch.float32 if scfp32 else dtype)
    return dict(name=name, nr=nr, k=K, h=H, e=E, c=C, mode=mode,
                exX=exX, ridx=ridx, eidx=eidx, x1=x1, x2=x2, bias=bias,
                scales=scales, dtype=dtype)


def _compare(got, ref):
    atol, rtol = _TOL[ref.dtype]
    got = got.to(torch.float64).cpu()
    ref = ref.to(torch.float64).cpu()
    diff = (got - ref).abs()
    tol = atol + rtol * ref.abs()
    bad = int((diff > tol).sum())
    return bad, float(diff.max())


def _run_case(dev, c):
    # import pdb
    # pdb.set_trace()
    t = _gen_case(c)
    ref = moe_finalize_routing_v2_ref(
        t["exX"], t["ridx"], x1=t["x1"], x2=t["x2"], bias=t["bias"],
        scales=t["scales"], expert_idx=t["eidx"], drop_pad_mode=t["mode"], k=t["k"])
    to_dev = lambda x: x.to(dev) if x is not None else None
    got = moe_finalize_routing_v2_cuda(
        to_dev(t["exX"]), to_dev(t["ridx"]), to_dev(t["x1"]), to_dev(t["x2"]),
        to_dev(t["bias"]), to_dev(t["scales"]), to_dev(t["eidx"]),
        t["mode"], t["k"])
    
    bad, maxdiff = _compare(got, ref)

    ok = bad == 0
    print("  [%s] %-36s %-6s nr=%d k=%d h=%d E=%d C=%d mode=%d max_diff=%.5g"
          % ("PASS" if ok else "FAIL", t["name"], str(t["dtype"]).replace("torch.", ""),
             t["nr"], t["k"], t["h"], t["e"], t["c"], t["mode"], maxdiff))
    return ok


def _test_doc_example(dev):
    """aclnn 文档官方示例（fp32, dropPadMode=0, 列主序 rowIdx={2,1,4,3,0,5}）。
    期望值按公式手算：
      row0: 2*x1 + 1.3*(exX[2]+bias[0]) + 1.6*(exX[3]+bias[1])
      row1: 2*x1 + 1.2*(exX[1]+bias[0]) + 1.8*(exX[0]+bias[1])
      row2: 2*x1 + 1.2*(exX[4]+bias[0]) + 2.3*(exX[5]+bias[1])
    """
    exX = torch.tensor([[0.1, 1.1, 2.1, 3.1], [4.1, 5.1, 6.1, 7.1],
                        [8.1, 9.1, 10.1, 11.1], [0.1, 1.1, 2.1, 3.1],
                        [4.1, 5.1, 6.1, 7.1], [8.1, 9.1, 10.1, 11.1]],
                       dtype=torch.float32)
    x1 = torch.tensor([[0.2, 1.2, 2.2, 3.2], [4.2, 5.2, 6.2, 7.2],
                       [8.2, 9.2, 10.2, 11.2]], dtype=torch.float32)
    x2 = x1.clone()
    bias = torch.tensor([[0.2, 0.4, 0.2, 0.4], [0.2, 0.4, 0.2, 0.4]],
                        dtype=torch.float32)
    scales = torch.tensor([[1.3, 1.6], [1.2, 1.8], [1.2, 2.3]], dtype=torch.float32)
    eidx = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.int32)
    ridx = torch.tensor([2, 1, 4, 3, 0, 5], dtype=torch.int32)
    expected = torch.tensor([[11.67, 17.15, 21.47, 26.95],
                             [14.10, 19.70, 24.10, 29.70],
                             [40.65, 46.85, 51.65, 57.85]], dtype=torch.float32)

    ref = moe_finalize_routing_v2_ref(exX, ridx, x1=x1, x2=x2, bias=bias,
                                      scales=scales, expert_idx=eidx,
                                      drop_pad_mode=0, k=2)
    got = moe_finalize_routing_v2_cuda(exX.to(dev), ridx.to(dev), x1.to(dev),
                                           x2.to(dev), bias.to(dev), scales.to(dev),
                                           eidx.to(dev), 0, 2).cpu()
    bad_ref = int(((ref - expected).abs() > 1e-3).sum())
    bad_cuda = int(((got - expected).abs() > 1e-3).sum())
    ok = bad_ref == 0 and bad_cuda == 0
    print("  [%s] doc-example (fp32 mode0) ref_vs_expected=%d err, "
          "cuda_vs_expected=%d err" % ("PASS" if ok else "FAIL", bad_ref, bad_cuda))
    print("    cuda out:\n%s" % got)
    return ok


def _test_error_paths(dev):
    n = 0
    t = _gen_case(_CASES[0])
    exX, ridx = t["exX"], t["ridx"]
    x2 = t["x2"]

    def expect_raise(fn, tag):
        nonlocal n
        try:
            fn()
        except Exception:
            return
        print("  [FAIL] %s not rejected" % tag)
        n += 1

    expect_raise(lambda: moe_finalize_routing_v2_ref(exX, ridx, x2=x2),
                 "ref: x2-without-x1")
    expect_raise(lambda: moe_finalize_routing_v2_ref(exX, ridx, x1=t["x1"],
                                                     bias=t["bias"]),
                 "ref: bias-without-expertIdx")
    expect_raise(lambda: moe_finalize_routing_v2_ref(exX, ridx, drop_pad_mode=4),
                 "ref: mode=4")

    expect_raise(lambda: moe_finalize_routing_v2_cuda(
        exX.to(dev), ridx.to(dev), None, x2.to(dev), None, None, None, 0, 2),
        "cuda: x2-without-x1")
    expect_raise(lambda: moe_finalize_routing_v2_cuda(
        exX.to(dev), ridx.to(dev), None, None, t["bias"].to(dev), None, None, 0, 2),
        "cuda: bias-without-expertIdx")
    expect_raise(lambda: moe_finalize_routing_v2_cuda(
        exX.to(dev), ridx.to(dev), None, None, None, None, None, 4, 2),
        "cuda: mode=4")
    ok = n == 0
    print("  [%s] error-path validation" % ("PASS" if ok else "FAIL"))
    return ok


# @pytest.mark.parametrize("seq", [10240, 20480, 40960, 81920])
# @pytest.mark.parametrize("topk", [15, 20, 25, 30])
# @pytest.mark.parametrize("hiddensize", [4096])
# @pytest.mark.parametrize("expert_num", [800, 1000, 1200])
# @pytest.mark.parametrize("drop_pad", [0,1,2,3])
# @pytest.mark.parametrize("dytpe",[torch.bfloat16, torch.float16])
# def test_moefinalize_routingv2(seq: int,
#      topk: int,
#      hiddensize: int, 
#      expert_num: int,
#      drop_pad: int):
#      generator = torch.Generator().manual_seed(42)
#      nr = seq
#      K = topk
#      H = hiddensize
#      E = expert_num
#      mode = drop_pad
#      C = 32  # drop-pad模式下每个专家的capacity，和前面case保持一致
#      drop_pad_flag = mode & 1
#      ex_rows = E * C if drop_pad_flag else nr * K
#      dtype = torch.bfloat16

#      # 构造输入
#      if drop_pad_flag:
#          exX = (torch.rand((E, C, H), generator=generator, dtype=torch.float32) * 2 - 1).to(dtype)
#      else:
#         exX = (torch.rand((ex_rows, H), generator=generator, dtype=torch.float32) * 2 - 1).to(dtype)

#      # ridx：部分置-1无效
#      ridx = torch.randint(0, ex_rows, (nr * K,), generator=generator, dtype=torch.int32)
#      mask_invalid = torch.rand(nr * K, generator=generator) < 0.08
#      ridx[mask_invalid] = -1

#      expert_idx = torch.randint(0, E, (nr, K), generator=generator, dtype=torch.int32)
#      x1 = (torch.rand((nr, H), generator=generator, dtype=torch.float32) * 2 - 1).to(dtype)
#      x2 = (torch.rand((nr, H), generator=generator, dtype=torch.float32) * 2 - 1).to(dtype)
#      bias = (torch.rand((E, H), generator=generator, dtype=torch.float32) * 2 - 1).to(dtype)
#      scales = (0.5 + 1.5 * torch.rand((nr, K), generator=generator, dtype=torch.float32)).to(dtype)

#      # 计算golden ref
#      ref = moe_finalize_routing_v2_ref(
#          exX, ridx,
#          x1=x1, x2=x2, bias=bias, scales=scales, expert_idx=expert_idx,
#          drop_pad_mode=mode, k=K
#          )

#      dev = torch.device("cuda")
#      to_dev = lambda x: x.to(dev) if x is not None else None
#      got = moe_finalize_routing_v2_cuda(
#         exx, to_dev(ridx),
#         to_dev(x1), to_dev(x2), to_dev(bias), to_dev(scales), to_dev(expert_idx),
#         mode, K
#         )

#      # 误差校验，复用全局TOL
#      atol, rtol = _TOL[dtype]
#      got = got.cpu()
#      diff = torch.abs(got - ref)
#      tol = atol + rtol * torch.abs(ref)
#      bad_count = int((diff > tol).sum())
#      max_diff = float(diff.max())
#      assert bad_count == 0, (
#         f"Failed case: seq={nr}, topk={K}, H={H}, E={E}, mode={mode}, "
#         f"bad={bad_count}, max_diff={max_diff:.5g}"
#         )

def main():
    print("== MoeFinalizeRoutingV2 PyTorch test ==")
    print("torch %s, cuda_available=%s" % (torch.__version__, torch.cuda.is_available()))
    dev = None
    
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("extension built OK")
    else:
        print("no CUDA device: kernel PyTorch 参考实现")

    npass = nfail = 0

    if _test_doc_example(dev):
        npass += 1
    else:
        nfail += 1

    for c in _CASES:
        ref_ok = True
        # 纯参考实现自洽（CPU 上直接可跑）
        t = _gen_case(c)
        try:
            ref_ok = _run_case(dev, c)
        except Exception as e:
            print("  [FAIL] %s ref raised: %s" % (t["name"], e))
            ok = False
        else:
            ok = ref_ok
            print("  [%s] %-36s check " % ("PASS" if ok else "FAIL", t["name"]))
        npass, nfail = (npass + 1, nfail) if ok else (npass, nfail + 1)

    if _test_error_paths(dev):
        npass += 1
    else:
        nfail += 1

    print("\n== SUMMARY: %d pass, %d fail ==" % (npass, nfail))
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
