#!/usr/bin/env python3

import argparse

import torch

import mcoplib.sgl_kernel  # noqa: F401: registers torch.ops.sgl_kernel


GROUP_SIZE = 128
COS_EPS = 1.0e-12


def get_input_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported input dtype: {name}")


def get_quant_dtype(name: str) -> torch.dtype:
    if name == "int8":
        return torch.int8
    if name == "fp8":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("this PyTorch build does not provide float8_e4m3fn")
        return torch.float8_e4m3fn
    raise ValueError(f"unsupported quant dtype: {name}")


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def qmax_and_min_scale(quant_dtype: torch.dtype) -> tuple[float, float]:
    if quant_dtype == torch.int8:
        qmax = 127.0
        min_absmax = qmax * torch.finfo(torch.float32).eps
        return qmax, min_absmax / qmax

    qmax = float(torch.finfo(quant_dtype).max)
    # Matches the standalone/interface implementation pattern:
    # absmax = max(absmax, 1 / 512), scale = absmax / qmax.
    return qmax, (1.0 / 512.0) / qmax


def make_input(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    # Keep values moderate so SiLU*mul avoids pathological FP8 saturation-heavy
    # distributions while still exercising quantization.
    return (torch.randn(shape, device="cuda", dtype=torch.float32) * 0.5).to(dtype)


def reference(
    input_tensor: torch.Tensor,
    quant_dtype: torch.dtype,
    swiglu_limit: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_hidden = input_tensor.size(-1)
    hidden = input_hidden // 2
    tokens = input_tensor.numel() // input_hidden
    groups = hidden // GROUP_SIZE

    x = input_tensor.reshape(tokens, input_hidden).float()
    gate = x[:, :hidden]
    up = x[:, hidden:]

    y = torch.nn.functional.silu(gate) * up
    # Match the kernel: when swiglu_limit is provided, the SiLU(gate)*up output
    # is symmetrically clamped to [-swiglu_limit, +swiglu_limit] BEFORE the
    # per-group absmax reduction and quantization.
    if swiglu_limit is not None:
        y = torch.clamp(y, -swiglu_limit, swiglu_limit)

    grouped = y.view(tokens, groups, GROUP_SIZE)
    amax = grouped.abs().amax(dim=-1)

    qmax, min_scale = qmax_and_min_scale(quant_dtype)
    scales_ref = torch.clamp(amax / qmax, min=min_scale)
    scaled = grouped / scales_ref.unsqueeze(-1)

    if quant_dtype == torch.int8:
        out_ref = torch.round(scaled).clamp(-127, 127).to(torch.int8)
    else:
        out_ref = scaled.to(quant_dtype)

    return (
        out_ref.view(*input_tensor.shape[:-1], hidden),
        scales_ref,
        y.view(*input_tensor.shape[:-1], hidden),
    )


def dequantize(
    out: torch.Tensor,
    scales: torch.Tensor,
    tokens: int,
    groups: int,
) -> torch.Tensor:
    return (
        out.float().reshape(tokens, groups, GROUP_SIZE)
        * scales.unsqueeze(-1)
    )


def cosine_similarity(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    actual_flat = actual.float().reshape(-1)
    expected_flat = expected.float().reshape(-1)
    numerator = torch.dot(actual_flat, expected_flat)
    denominator = actual_flat.norm() * expected_flat.norm()
    if denominator.item() < COS_EPS:
        return 1.0 if actual_flat.norm().item() < COS_EPS else 0.0
    return (numerator / denominator).item()


def call_op(
    out: torch.Tensor,
    scales: torch.Tensor,
    input_tensor: torch.Tensor,
    swiglu_limit: float | None = None,
) -> None:
    if swiglu_limit is None:
        torch.ops.sgl_kernel.fused_silu_mul_per_group_quant(
            out,
            scales,
            input_tensor,
        )
    else:
        torch.ops.sgl_kernel.fused_silu_mul_per_group_quant(
            out,
            scales,
            input_tensor,
            float(swiglu_limit),
        )


def check_functional_case(
    shape_prefix: tuple[int, ...],
    hidden: int,
    input_dtype: torch.dtype,
    quant_dtype: torch.dtype,
    cos_threshold: float,
    swiglu_limit: float | None = None,
) -> None:
    input_shape = (*shape_prefix, hidden * 2)
    tokens = 1
    for dim in shape_prefix:
        tokens *= dim
    groups = hidden // GROUP_SIZE

    input_tensor = make_input(input_shape, input_dtype)
    out = torch.empty((*shape_prefix, hidden), device="cuda", dtype=quant_dtype)
    scales = torch.empty((tokens, groups), device="cuda", dtype=torch.float32)

    out_ref, scales_ref, y_ref = reference(
        input_tensor, quant_dtype, swiglu_limit=swiglu_limit
    )

    call_op(out, scales, input_tensor, swiglu_limit=swiglu_limit)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        scales,
        scales_ref,
        rtol=2.0e-4,
        atol=2.0e-6,
    )

    # Main output correctness check: compare dequantized output with the
    # floating-point SiLU(gate) * up reference by cosine similarity.
    dequant = dequantize(out, scales, tokens, groups).reshape_as(y_ref)
    cos = cosine_similarity(dequant, y_ref)
    if cos < cos_threshold:
        max_abs_err = (dequant - y_ref.float()).abs().max().item()
        raise AssertionError(
            f"cosine similarity is {cos}, threshold={cos_threshold}, "
            f"max_abs_err={max_abs_err}"
        )

    print(
        f"PASS functional shape={input_shape} hidden={hidden} "
        f"tokens={tokens} input={dtype_name(input_dtype)} "
        f"quant={dtype_name(quant_dtype)} cos={cos:.8f}"
    )


def expect_error(name: str, fn, pattern: str | None = None) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - this is a negative test helper.
        msg = str(exc)
        if pattern is not None and pattern not in msg:
            raise AssertionError(
                f"{name}: error message does not contain {pattern!r}: {msg}"
            ) from exc
        print(f"PASS shape_check {name}: {msg.splitlines()[0]}")
        return
    raise AssertionError(f"{name}: expected an exception, but op succeeded")


def valid_tensors(
    input_shape: tuple[int, ...] = (2, 256),
    input_dtype: torch.dtype = torch.float16,
    quant_dtype: torch.dtype = torch.int8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_hidden = input_shape[-1]
    hidden = input_hidden // 2
    tokens = 1
    for dim in input_shape[:-1]:
        tokens *= dim
    groups = hidden // GROUP_SIZE

    input_tensor = make_input(input_shape, input_dtype)
    out = torch.empty((*input_shape[:-1], hidden), device="cuda", dtype=quant_dtype)
    scales = torch.empty((tokens, groups), device="cuda", dtype=torch.float32)
    return out, scales, input_tensor


def run_shape_checks() -> None:
    out, scales, input_tensor = valid_tensors()
    call_op(out, scales, input_tensor, swiglu_limit=None)
    torch.cuda.synchronize()
    print("PASS shape_check valid")

    expect_error(
        "input_dim_lt_2",
        lambda: call_op(
            torch.empty((128,), device="cuda", dtype=torch.int8),
            torch.empty((1, 1), device="cuda", dtype=torch.float32),
            torch.empty((256,), device="cuda", dtype=torch.float16),
        ),
        "at least 2 dimensions",
    )

    expect_error(
        "empty_input",
        lambda: call_op(
            torch.empty((0, 128), device="cuda", dtype=torch.int8),
            torch.empty((0, 1), device="cuda", dtype=torch.float32),
            torch.empty((0, 256), device="cuda", dtype=torch.float16),
        ),
        "non-empty",
    )

    expect_error(
        "odd_last_dim",
        lambda: call_op(
            torch.empty((2, 128), device="cuda", dtype=torch.int8),
            torch.empty((2, 1), device="cuda", dtype=torch.float32),
            torch.empty((2, 257), device="cuda", dtype=torch.float16),
        ),
        "even",
    )

    expect_error(
        "hidden_not_multiple_of_128",
        lambda: call_op(
            torch.empty((2, 192), device="cuda", dtype=torch.int8),
            torch.empty((2, 1), device="cuda", dtype=torch.float32),
            torch.empty((2, 384), device="cuda", dtype=torch.float16),
        ),
        "divisible by 128",
    )

    expect_error(
        "out_dim_mismatch",
        lambda: call_op(
            torch.empty((2, 1, 128), device="cuda", dtype=torch.int8),
            valid_tensors()[1],
            valid_tensors()[2],
        ),
        "out dim",
    )

    expect_error(
        "out_leading_mismatch",
        lambda: call_op(
            torch.empty((3, 128), device="cuda", dtype=torch.int8),
            valid_tensors()[1],
            valid_tensors()[2],
        ),
        "out shape",
    )

    expect_error(
        "out_last_dim_mismatch",
        lambda: call_op(
            torch.empty((2, 256), device="cuda", dtype=torch.int8),
            valid_tensors()[1],
            valid_tensors()[2],
        ),
        "out last dimension",
    )

    expect_error(
        "scales_dim_mismatch",
        lambda: call_op(
            valid_tensors()[0],
            torch.empty((2, 1, 1), device="cuda", dtype=torch.float32),
            valid_tensors()[2],
        ),
        "scales must be 2D",
    )

    expect_error(
        "scales_tokens_mismatch",
        lambda: call_op(
            valid_tensors()[0],
            torch.empty((3, 1), device="cuda", dtype=torch.float32),
            valid_tensors()[2],
        ),
        "scales.size(0)",
    )

    expect_error(
        "scales_groups_mismatch",
        lambda: call_op(
            valid_tensors((2, 512))[0],
            torch.empty((2, 3), device="cuda", dtype=torch.float32),
            valid_tensors((2, 512))[2],
        ),
        "scales.size(1)",
    )

    expect_error(
        "scales_dtype_mismatch",
        lambda: call_op(
            valid_tensors()[0],
            torch.empty((2, 1), device="cuda", dtype=torch.float16),
            valid_tensors()[2],
        ),
        "scales dtype",
    )

    expect_error(
        "input_dtype_mismatch",
        lambda: call_op(
            torch.empty((2, 128), device="cuda", dtype=torch.int8),
            torch.empty((2, 1), device="cuda", dtype=torch.float32),
            torch.empty((2, 256), device="cuda", dtype=torch.float64),
        ),
        "input dtype",
    )

    expect_error(
        "out_dtype_mismatch",
        lambda: call_op(
            torch.empty((2, 128), device="cuda", dtype=torch.float16),
            torch.empty((2, 1), device="cuda", dtype=torch.float32),
            torch.empty((2, 256), device="cuda", dtype=torch.float16),
        ),
        "out dtype",
    )

    expect_error(
        "input_not_contiguous",
        lambda: call_op(
            torch.empty((2, 128), device="cuda", dtype=torch.int8),
            torch.empty((2, 1), device="cuda", dtype=torch.float32),
            torch.empty((2, 256, 2), device="cuda", dtype=torch.float16)[:, :, 0],
        ),
        "contiguous",
    )

    expect_error(
        "out_not_contiguous",
        lambda: call_op(
            torch.empty((2, 128, 2), device="cuda", dtype=torch.int8)[:, :, 0],
            torch.empty((2, 1), device="cuda", dtype=torch.float32),
            torch.empty((2, 256), device="cuda", dtype=torch.float16),
        ),
        "contiguous",
    )

    expect_error(
        "scales_not_contiguous",
        lambda: call_op(
            torch.empty((2, 128), device="cuda", dtype=torch.int8),
            torch.empty((2, 1, 2), device="cuda", dtype=torch.float32)[:, :, 0],
            torch.empty((2, 256), device="cuda", dtype=torch.float16),
        ),
        "contiguous",
    )


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["functional", "shape", "all"], default="all")
    parser.add_argument("--input-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--quant", choices=["int8", "fp8", "all"], default="all")
    parser.add_argument("--tokens", type=parse_int_list, default=parse_int_list("1,16,32"))
    parser.add_argument("--hidden", type=parse_int_list, default=parse_int_list("128,512,1024"))
    parser.add_argument("--cos-threshold", type=float, default=0.999)
    parser.add_argument("--include-prod-shapes", action="store_true")
    parser.add_argument("--rank3", action="store_true")
    parser.add_argument(
        "--swiglu_limit",
        type=float,
        default=100.0,
        help="SwiGLU clamp limit applied to SiLU(gate)*up before per-group "
        "quantization. Default 100 for tests (kernel C++ default is 10.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    #torch.manual_seed(66)
    #torch.cuda.manual_seed_all(66)

    if args.mode in ("shape", "all"):
        run_shape_checks()

    if args.mode in ("functional", "all"):
        input_dtype = get_input_dtype(args.input_dtype)

        quant_dtypes: list[torch.dtype] = []
        if args.quant in ("int8", "all"):
            quant_dtypes.append(torch.int8)
        if args.quant in ("fp8", "all"):
            quant_dtypes.append(get_quant_dtype("fp8"))

        hidden_list = list(args.hidden)
        if args.include_prod_shapes:
            hidden_list.extend([24576, 36864])

        for quant_dtype in quant_dtypes:
            for tokens in args.tokens:
                for hidden in hidden_list:
                    if args.rank3 and tokens > 1:
                        shape_prefix = (2, tokens // 2) if tokens % 2 == 0 else (1, tokens)
                    else:
                        shape_prefix = (tokens,)
                    check_functional_case(
                        shape_prefix,
                        hidden,
                        input_dtype,
                        quant_dtype,
                        args.cos_threshold,
                        swiglu_limit=args.swiglu_limit,
                    )


if __name__ == "__main__":
    main()

