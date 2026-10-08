# SPDX-License-Identifier: Apache-2.0
"""Baseline-parity test for the Step5-shaped per-token-group fp8 quant ops.
Baseline is pure python reference per_token_group_quant_8bit_py,
candidate is custom CUDA op (per_token_group_fp8_quant generic smem path).
Covered entries, on the Step5 geometry (`step4-hf/inference/config.json`):
hidden 4096, MoE intermediate 1536, w13 3072, dense intermediate 13824, e4m3,
`weight_block_size [128, 128]`):
* `per_token_group_fp8_quant`     -- generic smem path
Usage
-----
Test candidate CUDA op against python ref::
    python unit_test/test_per_token_group_fp8.py --candidate-generic per_token_group_fp8_quant
Self-check (candidate=python ref, just verify harness)::

    python unit_test/test_per_token_group_fp8.py --candidate-generic pyref
"""
from __future__ import annotations
import argparse
import importlib
import os
import torch
import mcoplib._C  # noqa: F401

# ======================================
# Python Reference Implementation
# ======================================
FP8_E4M3 = torch.float8_e4m3fn
def compute_group_scale(group_x: torch.Tensor, eps: float, max_8bit: float, SCALE_UE8M0: bool):
    """
    等价CUDA ComputeGroupScale<T, SCALE_UE8M0>
    group_x: shape [..., group_size]，一组或多组原始输入（bf16/fp16）
    return: scale (float32)，shape [...]

    【修复】必须全程 fp32。CUDA 侧是 static_cast<float>(src) 之后在 fp32
    里做 fmaxf 和除法；本函数原先用 bf16 的 group_x 直接做 abs/max（虽
    精确，但依赖隐式提升），这里显式落到 fp32，并补上 CUDA 的 UE8M0
    保护 fmaxf(fabsf(y_s), 1e-10f)。
    """
    x_f32 = group_x.to(torch.float32)
    # CUDA: local_absmax 初值为 eps，随后逐元素 fmaxf → max(eps, |x|)
    abs_max = x_f32.abs().amax(dim=-1)
    abs_max = torch.maximum(abs_max, torch.full_like(abs_max, eps))
    y_s = abs_max / max_8bit
    if SCALE_UE8M0:
        # CUDA: exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))))
        y_s = torch.exp2(torch.ceil(torch.log2(y_s.abs().clamp_min(1e-10))))
    return y_s

def quantize_group(group_x: torch.Tensor, scale: torch.Tensor, min_8bit: float, max_8bit: float):
    """
    等价CUDA QuantizeGroup<T, DST_DTYPE>
    group_x: [..., group_size]
    scale: [...]，float32
    return: fp8量化后的tensor [..., group_size]

    【修复】原实现 `group_x / scale` 中，scale 是 0-dim fp32 tensor，
    PyTorch 类型提升（dim>0 tensor 优先于同 category 的 0-dim tensor）
    使结果落回 bf16/fp16：scale 先被舍入到 bf16，商再舍入到 bf16，
    最后才转 fp8 —— 双重舍入，与 kernel 的一次 fp32→fp8 舍入不一致，
    非 UE8M0 路径大面积偏 1+ ulp。修复：显式 .to(torch.float32) 后再除，
    fp32 除法 + fp32 clamp + 一次 fp8 转换，与 CUDA 逐 bit 对齐。
    （UE8M0 的 scale 是 2 的幂、bf16 可精确表示，所以原先 UE8M0 case
    恰好能对上，只有非 UE8M0 爆炸。）
    """
    x_scaled = group_x.to(torch.float32) / scale.unsqueeze(-1)
    x_clamped = torch.clamp(x_scaled, min_8bit, max_8bit)
    return x_clamped.to(FP8_E4M3)

def per_token_group_quant_8bit_py(
    x: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    IS_COLUMN_MAJOR: bool = False,
    SCALE_UE8M0: bool = False,
    scale_num_rows: int = 0,
    scale_stride: int = 0,
):
    """
    Python reference implementation for per_token_group_quant_8bit_kernel
    x: [M, K], bf16 / fp16
    return: output_q(fp8), output_s(float32)

    【修复】向量化实现，与逐组循环数值完全等价（abs/max/div/clamp 都是
    逐元素或精确可结合的操作，分组与否不影响结果），但快几个数量级
    （原逐组 Python 循环在整个测试里要跑 ~214 万组）。
    """
    M, K = x.shape
    assert K % group_size == 0
    groups_per_row = K // group_size
    num_groups = M * groups_per_row

    # global_group_id = m * groups_per_row + g，与 kernel 的展平顺序一致
    xg = x.reshape(num_groups, group_size)
    scales_flat = compute_group_scale(xg, eps, max_8bit, SCALE_UE8M0)
    output_q = quantize_group(xg, scales_flat, min_8bit, max_8bit).reshape(M, K)

    if IS_COLUMN_MAJOR:
        # CUDA column-major 分支（scale_packed_t=float → num_elems_per_pack=1）:
        #   row_idx = gid / groups_per_row (m), col_idx = gid % groups_per_row (g)
        #   addr = col_idx * scale_stride + row_idx = g * M + m
        # 即存储层 flat[g * M + m] = scales_flat[m * groups_per_row + g]，
        # 等价于 stride=(1, M) 的 [M, groups_per_row] 视图（copy_ 按逻辑
        # 下标写入，与本 harness 的 _scales_alloc 布局一致）。
        buf = scales_flat.new_empty(num_groups)
        output_s = buf.as_strided((M, groups_per_row), (1, M))
        output_s.copy_(scales_flat.reshape(M, groups_per_row))
    else:
        # row-major: output_s shape [M, groups_per_row]
        output_s = scales_flat.reshape(M, groups_per_row)
    return output_q, output_s

# ======================================
# Test Harness Config
# ======================================
FP8_MIN = float(torch.finfo(FP8_E4M3).min)
FP8_MAX = float(torch.finfo(FP8_E4M3).max)
EPS = 1e-10
DEVICE = "cuda"
GROUP_SIZE = 128
HIDDEN = 4096
MOE_INTERMEDIATE = 1536
W13 = 2 * MOE_INTERMEDIATE
DENSE_INTERMEDIATE = 13824
SHAPES = [
    (1, HIDDEN),
    (6, MOE_INTERMEDIATE),
    (127, MOE_INTERMEDIATE),
    (512, W13),
    (1024, HIDDEN),
    (2048, DENSE_INTERMEDIATE),
]
# Baseline = python ref, candidate = your CUDA op
BASELINE_GENERIC = "pyref"
# ---------------------------------------------------------------------------- #
# Op lookup (add pyref special case)
# ---------------------------------------------------------------------------- #
def _resolve_op(spec: str):
    """
    spec == "pyref" -> return python reference function
    else resolve torch.ops cuda op
    """
    if spec == "pyref":
        # wrap pyref to match cuda op signature
        def pyref_op(x, q_out, s_out, group_size, eps, fp8_min, fp8_max, scale_ue8m0, column_major, dummy):
            M, K = x.shape
            q_ref, s_ref = per_token_group_quant_8bit_py(
                x, group_size, eps, fp8_min, fp8_max,
                IS_COLUMN_MAJOR=column_major,
                SCALE_UE8M0=scale_ue8m0,
                scale_num_rows=M,
                scale_stride=M
            )
            q_out.copy_(q_ref)
            s_out.copy_(s_ref)
        return pyref_op

    ns, _, name = spec.rpartition(".")
    ns = ns or "_C"
    obj = torch.ops
    try:
        for part in ns.split("."):
            obj = getattr(obj, part)
        return getattr(obj, name)
    except AttributeError as exc:
        try:
            known = sorted(m for m in dir(obj) if "per_token" in m)
        except Exception:
            known = []
        raise AssertionError(
            f"op {spec!r} is not registered; per_token ops found: {known}"
        ) from exc

# ---------------------------------------------------------------------------- #
# Buffers
# ---------------------------------------------------------------------------- #
def _scales_alloc(m: int, gpr: int, column_major: bool) -> torch.Tensor:
    if column_major:
        # stride(0) < stride(1) is what the op reads as column-major.
        tensor = torch.empty((m, gpr), dtype=torch.float32, device=DEVICE)
        # 制造列主序stride: stride(0)=1, stride(1)=m
        tensor = tensor.as_strided(size=(m, gpr), stride=(1, m))
        return tensor
    return torch.empty((m, gpr), dtype=torch.float32, device=DEVICE)

# ---------------------------------------------------------------------------- #
# Gates 【FIXED：按指数缩放的真实 fp8 ulp 比对，允许 1 ulp；
#         超出 1 ulp 的元素受 max_mismatch_frac 比例约束】
# ---------------------------------------------------------------------------- #
def _fp8_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    fp8 e4m3fn 逐元素 ulp 距离。
    把 sign-magnitude 的 fp8 位型映射成单调整数格点：
    正数 bits+128、负数 127-(bits&0x7F)，差值即随指数缩放的真实 ulp 距离。
    （原先用固定的 finfo.eps=0.125 当绝对阈值，|v|>=16 处 1 个真实 ulp
    就有 >=2 的绝对差，会被误判成 mismatch，小 shape 下造成偶发 fail。）
    两侧同为 NaN 视为相等，单侧 NaN 记为大距离。
    """
    ai = a.contiguous().view(torch.uint8).to(torch.int16)
    bi = b.contiguous().view(torch.uint8).to(torch.int16)

    def key(t: torch.Tensor) -> torch.Tensor:
        return torch.where((t & 0x80) != 0, 127 - (t & 0x7F), t + 128)

    dist = (key(ai) - key(bi)).abs()
    a_nan = (ai & 0x7F) == 0x7F
    b_nan = (bi & 0x7F) == 0x7F
    dist = torch.where(a_nan & b_nan, torch.zeros_like(dist), dist)
    dist = torch.where(a_nan ^ b_nan, torch.full_like(dist, 255), dist)
    return dist

def _check_q(
    case: str, candidate: torch.Tensor, baseline: torch.Tensor, max_mismatch_frac: float
) -> int:
    """
    fp8 e4m3 允许 1 ulp（随指数缩放的真实 ulp）误差，
    超过 1 ulp 的元素数量受 max_mismatch_frac 约束。
    返回超出 1 ulp 的元素数量（含在限额内的）。
    """
    dist = _fp8_ulp_distance(candidate, baseline)
    bad = dist > 1
    n_bad = int(bad.sum().item())
    limit = max_mismatch_frac * candidate.numel()
    if n_bad <= limit:
        return n_bad
    w = int(dist.reshape(-1).argmax())
    print(f"[FAIL] quantized_output: {case}")
    print(f" elements beyond 1 fp8 ulp: {n_bad} / {candidate.numel()} (allowed {limit:.1f})")
    print(
        f" worst: candidate={float(candidate.float().reshape(-1)[w])} "
        f"baseline={float(baseline.float().reshape(-1)[w])} "
        f"ulp_distance={int(dist.reshape(-1)[w])}"
    )
    raise AssertionError(f"quantized output mismatch for {case}")

def _check_scale(
    name: str, case: str, candidate: torch.Tensor, baseline: torch.Tensor, atol: float, rtol: float
) -> None:
    c, b = candidate.float(), baseline.float()
    diff = (c - b).abs()
    allowed = atol + rtol * b.abs()
    bad = diff > allowed
    if not bool(bad.any()):
        return
    w = int(diff.reshape(-1).argmax())
    print(f"[FAIL] {name}: {case}")
    print(f" mismatched: {int(bad.sum().item())} / {c.numel()} (atol={atol}, rtol={rtol})")
    print(
        f" worst: candidate={float(c.reshape(-1)[w])} baseline={float(b.reshape(-1)[w])} "
        f"abs_diff={float(diff.reshape(-1)[w])} allowed={float(allowed.reshape(-1)[w])}"
    )
    raise AssertionError(f"{name} mismatch for {case}")

# ---------------------------------------------------------------------------- #
# Case runners
# ---------------------------------------------------------------------------- #
def _run_generic_pair(
    base_op, cand_op, seed: int, m: int, n: int, dtype: torch.dtype, scale_ue8m0: bool,
    column_major: bool, scale_atol: float, scale_rtol: float, max_q_mismatch_frac: float,
) -> int:
    gpr = n // GROUP_SIZE
    case = (
        f"per_token_group_fp8_quant shape=({m}, {n}) dtype={dtype} "
        f"group_size={GROUP_SIZE} scale_ue8m0={scale_ue8m0} column_major={column_major}"
    )
    torch.manual_seed(seed)
    x = torch.randn(m, n, dtype=dtype, device=DEVICE)
    args = (GROUP_SIZE, EPS, FP8_MIN, FP8_MAX, scale_ue8m0, column_major, False)
    q_b, s_b = torch.empty_like(x, dtype=FP8_E4M3), _scales_alloc(m, gpr, column_major)
    q_c, s_c = torch.empty_like(x, dtype=FP8_E4M3), _scales_alloc(m, gpr, column_major)
    base_op(x, q_b, s_b, *args)
    cand_op(x, q_c, s_c, *args)
    n_bad = _check_q(case, q_c, q_b, max_q_mismatch_frac)
    _check_scale("scale", case, s_c, s_b, scale_atol, scale_rtol)
    return n_bad

# ---------------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------------- #
def run(
    baseline_generic: str, candidate_generic: str,
    scale_atol: float, scale_rtol: float, max_q_mismatch_frac: float,
    extra_imports: tuple[str, ...] = (),
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("no cuda device visible to torch")
    for module in extra_imports:
        importlib.import_module(module)
    base_g = _resolve_op(baseline_generic)
    cand_g = _resolve_op(candidate_generic)

    if candidate_generic == baseline_generic:
        print(
            "[NOTE] candidate == baseline: running self-check only. "
            "Point --candidate-generic at your CUDA op to compare against python ref."
        )
    failures: list[str] = []
    q_mismatch_total = 0
    seed = 0
    for dtype in (torch.bfloat16, torch.float16):
        for m, n in SHAPES:
            for scale_ue8m0 in (False, True):
                for column_major in (False, True):
                    seed += 1
                    try:
                        q_mismatch_total += _run_generic_pair(
                            base_g, cand_g, seed, m, n, dtype, scale_ue8m0, column_major,
                            scale_atol, scale_rtol, max_q_mismatch_frac,
                        )
                    except AssertionError as exc:
                        failures.append(str(exc))
    if failures:
        print(f"\n{len(failures)} case(s) failed:")
        for i, failure in enumerate(failures, start=1):
            print(f"  {i}. {failure}")
        raise AssertionError(f"{len(failures)} case(s) failed")
    print(
        f"per_token_group_fp8_quant parity against python ref: all cases passed "
        f"(scale rtol={scale_rtol}, max q mismatch frac={max_q_mismatch_frac}, "
        f"elements beyond 1 fp8 ulp={q_mismatch_total})"
    )

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline-generic", default=BASELINE_GENERIC)
    parser.add_argument(
        "--candidate-generic",
        default=os.environ.get("PER_TOKEN_GROUP_QUANT_CANDIDATE_GENERIC", BASELINE_GENERIC),
    )
    parser.add_argument(
        "--import",
        dest="imports",
        action="append",
        metavar="MODULE",
        help="import a module so its ops register (repeatable)",
    )
    parser.add_argument("--scale-atol", type=float, default=0.0)
    parser.add_argument("--scale-rtol", type=float, default=1e-2)
    # 允许少量超过1ulp的元素，你可以按需调大
    parser.add_argument("--max-q-mismatch-frac", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if args.imports is None:
        env = os.environ.get("PER_TOKEN_GROUP_QUANT_IMPORTS", "")
        args.imports = [m.strip() for m in env.split(",") if m.strip()]
    return args

def _main(argv) -> None:
    args = _parse_args(argv)
    run(
        args.baseline_generic,
        args.candidate_generic,
        args.scale_atol,
        args.scale_rtol,
        args.max_q_mismatch_frac,
        tuple(args.imports),
    )

def test_per_token_group_fp8_quant_parity() -> None:
    _main([])

if __name__ == "__main__":
    _main(None)
