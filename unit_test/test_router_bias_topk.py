# SPDX-License-Identifier: Apache-2.0
"""Step5 router_bias_topk 单测：torch 实现 vs triton 实现。

* 参考实现：纯 PyTorch ``torch_router_bias_topk``（README.md「计算语义」的公式）。
* 候选实现：默认内联的当前 Triton 实现（从
  ``vLLM-metax/vllm_metax/models/step5/router_bias.py`` 原样拷入），可用
  ``IMPL=mcoplib`` 换成 ``mcoplib.op.router_bias_topk``。

两者在同一组输入上逐元素对比：ids 完全相等，weights 在容差内。

运行（需要 GPU + Triton）：

    python test_router_bias_topk.py
    IMPL=mcoplib python test_router_bias_topk.py
"""

from __future__ import annotations

import os
import sys

import torch
import mcoplib.op as ops  # noqa: PLC0415
from mcoplib.profiler import profiler

try:
    import triton
    import triton.language as tl
except ImportError:  # 无 triton 时无法跑 triton 对比
    triton = None
    tl = None
@profiler(output_dir="./profiles", warmup=2, repeat=3)
def mcoplib_router_bias_topk(
    gating_output: torch.Tensor,
    router_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    check_nan: bool = True,
    routed_scaling_factor: float = 1.0,
    nan_row_i_out: int = 0,
    indices_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert gating_output.is_cuda, "gating_output must be on CUDA"
    assert gating_output.ndim == 2
    assert router_bias is not None, "router_bias must be provided"
    num_tokens, num_experts = gating_output.shape
    assert 0 < topk <= num_experts
    if indices_dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"indices_dtype must be torch.int32 or torch.int64, got {indices_dtype}"
        )

    bias = router_bias.to(gating_output.device)
    assert bias.numel() == num_experts

    topk_weights = torch.empty(
        (num_tokens, topk), device=gating_output.device, dtype=torch.float32
    )
    topk_ids = torch.empty(
        (num_tokens, topk), device=gating_output.device, dtype=indices_dtype
    )
    ops.router_bias_topk(gating_output, router_bias, topk_weights, topk_ids, topk, renormalize, check_nan, routed_scaling_factor, nan_row_i_out)

    return topk_weights, topk_ids


# --------------------------------------------------------------------------
# 参考实现（torch）：README.md 第 6 节的公式
# --------------------------------------------------------------------------
def torch_router_bias_topk(
    gating_output: torch.Tensor,
    router_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    check_nan: bool = True,
    routed_scaling_factor: float = 1.0,
    nan_row_i_out: int = 0,
    indices_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = gating_output.shape
    probs = gating_output.to(torch.float32).sigmoid()
    scores = probs + router_bias.to(torch.float32).reshape(-1)

    # descending + stable => 平局时更小的 expert id 在前
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :topk]
    weights = probs.gather(1, order)

    if renormalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * routed_scaling_factor

    ids = order
    if check_nan:
        bad = gating_output.isnan().any(dim=-1, keepdim=True)
        weights = torch.where(bad, torch.zeros_like(weights), weights)
        ids = torch.where(bad, torch.full_like(ids, nan_row_i_out), ids)
    return weights.to(torch.float32), ids.to(indices_dtype)


# --------------------------------------------------------------------------
# 内联 Triton 实现（从 router_bias.py 原样拷入，改动请同步回源文件）
# --------------------------------------------------------------------------
if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=num_warps, num_stages=num_stages)
            for num_warps in [2, 4, 8]
            for num_stages in [2, 3, 4, 5]
        ],
        key=["E", "TOPK"],
    )
    @triton.jit
    def _router_bias_topk_kernel(
        gating_ptr,
        bias_ptr,
        out_w_ptr,
        out_i_ptr,
        stride_gm,
        stride_om,
        E: tl.constexpr,
        TOPK: tl.constexpr,
        RENORM: tl.constexpr,
        CHECK_NAN: tl.constexpr,
        BLOCK_E: tl.constexpr,
        ROUTED_SCALING_FACTOR: tl.constexpr,
        NAN_ROW_I_OUT: tl.constexpr,
    ):
        pid = tl.program_id(0)

        offs_e = tl.arange(0, BLOCK_E)
        mask_e = offs_e < E

        row_ptr = gating_ptr + pid * stride_gm + offs_e
        tl.multiple_of(row_ptr, 8)
        tl.max_contiguous(offs_e, 128)

        gating = tl.load(row_ptr, mask=mask_e, other=0)
        gate_prob = tl.sigmoid(gating.to(tl.float32))

        bias = tl.load(bias_ptr + offs_e, mask=mask_e, other=0).to(tl.float32)
        bias = tl.where(mask_e, bias, -float("inf"))
        scores = tl.where(mask_e, gate_prob + bias, -float("inf"))

        if CHECK_NAN:
            gating_nan = gating != gating
            has_bad = tl.max(gating_nan.to(tl.int32), axis=0) > 0

        weights = tl.zeros((TOPK,), dtype=tl.float32)
        indices = tl.zeros((TOPK,), dtype=tl.int32)
        weight_sum = 0.0
        topk_offsets = tl.arange(0, TOPK)

        for k in tl.static_range(TOPK):
            _, max_index = tl.max(scores, axis=0, return_indices=True)
            max_index = max_index.to(tl.int32)

            # gate_prob 已是 in-range expert 的 sigmoid；在这里按 max_index
            # gather。不能重读 gating[max_index]：当 E 非 2 的幂、argmax 落到
            # BLOCK_E 尾部时（尾部 score=-inf 会赢过 NaN）会越界。
            selected_lane = (offs_e == max_index) & mask_e
            selected_prob = tl.sum(tl.where(selected_lane, gate_prob, 0.0), axis=0)

            weights = tl.where(topk_offsets == k, selected_prob, weights)
            indices = tl.where(topk_offsets == k, max_index, indices)

            weight_sum += selected_prob
            scores = tl.where(offs_e == max_index, -float("inf"), scores)

        if RENORM:
            weights = weights / (weight_sum + 1e-20)

        if ROUTED_SCALING_FACTOR != 1.0:
            weights = weights * ROUTED_SCALING_FACTOR

        if CHECK_NAN:
            weights = tl.where(has_bad, 0.0, weights)
            indices = tl.where(has_bad, NAN_ROW_I_OUT, indices)

        offsets = tl.arange(0, TOPK)
        tl.store(out_w_ptr + pid * stride_om + offsets, weights, mask=offsets < TOPK)
        # store 会把 int32 expert id 转成输出指针的元素类型（int32/int64）。
        tl.store(out_i_ptr + pid * stride_om + offsets, indices, mask=offsets < TOPK)
    @profiler(output_dir="./profiles", warmup=2, repeat=3)
    def router_bias_triton_func(
        gating_output: torch.Tensor,
        router_bias: torch.Tensor,
        topk: int,
        renormalize: bool,
        check_nan: bool = True,
        routed_scaling_factor: float = 1.0,
        nan_row_i_out: int = 0,
        indices_dtype: torch.dtype = torch.int32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Router-bias top-k，id 直接写成 ``indices_dtype``。"""
        assert gating_output.is_cuda, "gating_output must be on CUDA"
        assert gating_output.ndim == 2
        assert router_bias is not None, "router_bias must be provided"
        num_tokens, num_experts = gating_output.shape
        assert 0 < topk <= num_experts
        if indices_dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"indices_dtype must be torch.int32 or torch.int64, got {indices_dtype}"
            )

        bias = router_bias.to(gating_output.device)
        assert bias.numel() == num_experts

        topk_weights = torch.empty(
            (num_tokens, topk), device=gating_output.device, dtype=torch.float32
        )
        topk_ids = torch.empty(
            (num_tokens, topk), device=gating_output.device, dtype=indices_dtype
        )
        if num_tokens == 0:
            return topk_weights, topk_ids

        block_experts = 1 << (num_experts - 1).bit_length()
        if block_experts > 1024:
            raise ValueError(
                f"num_experts={num_experts} is too large for single-block top-k "
                f"(block_experts={block_experts})"
            )

        _router_bias_topk_kernel[(num_tokens,)](
            gating_output,
            bias,
            topk_weights,
            topk_ids,
            gating_output.stride(0),
            topk_weights.stride(0),
            E=num_experts,
            TOPK=topk,
            RENORM=renormalize,
            CHECK_NAN=check_nan,
            BLOCK_E=block_experts,
            ROUTED_SCALING_FACTOR=routed_scaling_factor,
            NAN_ROW_I_OUT=nan_row_i_out,
        )
        return topk_weights, topk_ids


# --------------------------------------------------------------------------
# 候选实现解析（参考实现恒为 torch_router_bias_topk）
# --------------------------------------------------------------------------
def _resolve_impl(name: str):
    name = name.lower()
    if name == "triton":
        if triton is None:
            raise RuntimeError("IMPL=triton 但当前环境无法 import triton")
        return "triton", router_bias_triton_func
    if name == "mcoplib":
        return "mcoplib", mcoplib_router_bias_topk
    raise ValueError(f"未知 IMPL={name}，应为 triton 或 mcoplib")


# --------------------------------------------------------------------------
# 用例
# --------------------------------------------------------------------------
# 生产配置，取自 config_step5_text.json
E = 352                # moe_num_experts
TOPK = 8               # moe_top_k
RENORMALIZE = True     # norm_expert_weight
SCALING = 3.0          # moe_router_scaling_factor

# token 数模拟
CHUNK_PREFILL = [4096, 8192]              # chunked prefill
DECODE_CONCURRENCY = [1, 2, 4, 8, 16, 32, 64]
MTP = 3                                   # num_nextn_predict_layers


def _make_case(
    num_tokens: int,
    num_experts: int,
    topk: int,
    *,
    tie: bool = False,
    nan_rows: tuple[int, ...] = (),
    seed: int = 0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gating = torch.randn(num_tokens, num_experts, generator=gen, dtype=torch.float32)
    bias = torch.randn(num_experts, generator=gen, dtype=torch.float32) * 0.1
    if tie and num_tokens > 0:
        # 让每行前若干列分数完全相同，逼出平局（gating 与 bias 同时相等）
        m = min(num_experts, 4)
        gating[0, :m] = 0.5
        bias[:m] = 0.0
    for r in nan_rows:
        if num_tokens > 0:
            gating[r, 0] = float("nan")
    return gating.to(device), bias.to(device)


def _cases():
    # token 数模拟：prefill chunk 4k/8k；decode 并发 1..64，普通 decode 1 token/条，
    # MTP3 每条排成 1 + 3 个连续 position，故 token 数为并发的 4 倍。
    for chunk in CHUNK_PREFILL:
        yield dict(name=f"prefill chunk={chunk}", T=chunk, E=E, topk=TOPK,
                   renorm=RENORMALIZE, scale=SCALING)
    for conc in DECODE_CONCURRENCY:
        yield dict(name=f"decode c={conc}", T=conc, E=E, topk=TOPK,
                   renorm=RENORMALIZE, scale=SCALING)
    for conc in DECODE_CONCURRENCY:
        yield dict(name=f"decode mtp3 c={conc}", T=conc * (1 + MTP), E=E, topk=TOPK,
                   renorm=RENORMALIZE, scale=SCALING)

    # 语义/边界用例，仍用生产 E=352、topk=8
    yield dict(name="empty T=0", T=0, E=E, topk=TOPK, renorm=RENORMALIZE, scale=SCALING)
    yield dict(name="no renorm", T=8, E=E, topk=TOPK, renorm=False, scale=1.0)
    yield dict(name="renorm no scale", T=8, E=E, topk=TOPK, renorm=True, scale=1.0)
    yield dict(name="check_nan off", T=8, E=E, topk=TOPK, renorm=RENORMALIZE, scale=SCALING,
               check_nan=False)
    yield dict(name="nan row", T=8, E=E, topk=TOPK, renorm=RENORMALIZE, scale=SCALING,
               nan_rows=(0, 3))
    yield dict(name="nan row int64", T=8, E=E, topk=TOPK, renorm=RENORMALIZE, scale=SCALING,
               nan_rows=(0,), idtype=torch.int64)
    yield dict(name="int64 ids", T=8, E=E, topk=TOPK, renorm=RENORMALIZE, scale=SCALING,
               idtype=torch.int64)
    yield dict(name="ties", T=8, E=E, topk=TOPK, renorm=RENORMALIZE, scale=SCALING, tie=True)
    yield dict(name="ties no renorm", T=8, E=E, topk=TOPK, renorm=False, scale=1.0, tie=True)


def run(device: str = "cuda") -> int:
    impl_name, impl = _resolve_impl(os.environ.get("IMPL", "triton"))
    print(f"reference = torch | candidate = {impl_name}")
    print(
        f"config: E={E} topk={TOPK} renorm={RENORMALIZE} scale={SCALING} | "
        f"prefill chunk={CHUNK_PREFILL} | decode c={DECODE_CONCURRENCY} "
        f"mtp={MTP}"
    )
    failures = 0
    for i, case in enumerate(_cases()):
        name = case.pop("name")
        T = case.pop("T")
        E_c = case.pop("E")
        topk = case.pop("topk")
        renorm = case.pop("renorm")
        scale = case.pop("scale")
        idtype = case.pop("idtype", torch.int32)
        check_nan = case.pop("check_nan", True)
        gating, bias = _make_case(T, E_c, topk, device=device, **case)

        ref_w, ref_i = torch_router_bias_topk(
            gating, bias, topk, renorm,
            check_nan=check_nan,
            routed_scaling_factor=scale,
            nan_row_i_out=0,
            indices_dtype=idtype,
        )
        out_w, out_i = impl(
            gating, bias, topk, renorm,
            check_nan=check_nan,
            routed_scaling_factor=scale,
            nan_row_i_out=0,
            indices_dtype=idtype,
        )

        ok_shape = out_w.shape == (T, topk) and out_i.shape == (T, topk)
        ok_dtype = out_w.dtype == torch.float32 and out_i.dtype == idtype
        ok_ids = bool(torch.equal(out_i.to(torch.int64), ref_i.to(torch.int64)))
        
        ok_w = bool(
            torch.allclose(out_w, ref_w, rtol=1e-5, atol=1e-6, equal_nan=True)
        )
        ok = ok_shape and ok_dtype and ok_ids and ok_w
        status = "ok  " if ok else "FAIL"
        print(f"[{status}] {i:2d} {name:22s} T={T:5d} E={E_c:4d} k={topk:2d}")
        if not ok:
            failures += 1
            print(f"        shape={ok_shape} dtype={ok_dtype} ids={ok_ids} weights={ok_w}")
            if not ok_ids:
                bad = (out_i.to(torch.int64) != ref_i.to(torch.int64)).nonzero()
                print(f"        first id mismatch row={bad[0].tolist()}")
            if not ok_w:
                diff = (out_w - ref_w).abs()
                print(f"        max weight diff={float(diff.max()):.3e}")

    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: {failures} failed")
    return failures


def main() -> None:
    if not torch.cuda.is_available() or triton is None:
        print(
            "[error] 本单测为 torch vs triton 对比，需要 CUDA + Triton",
            file=sys.stderr,
        )
        raise SystemExit(2)
    raise SystemExit(run("cuda"))


if __name__ == "__main__":
    main()
