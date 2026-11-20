"""MHC interfaces for the fixed HC=4, hidden-size=5120 implementation.

``mhc_pre`` replaces the decomposed Triton pre path with::

    DeepGEMM tf32_hc_prenorm_gemm
      -> mcoplib.op.mhc_pre_big_fuse_out

``mhc_post`` replaces the TileLang post path with
``mcoplib.op.mhc_post_cuda``.
"""

from __future__ import annotations

from functools import cache
from typing import Callable

import torch

from mcoplib import op as mcoplib_ops


HC_MULT = 4
HIDDEN_SIZE = 5120
PACKED_INT8_ROW_ELEMENTS = ((HIDDEN_SIZE + 4 + 511) // 512) * 256
MIX_SIZE = HC_MULT * (2 + HC_MULT)
HC_HIDDEN_SIZE = HC_MULT * HIDDEN_SIZE
SUPPORTED_NUM_SPLITS = (1, 4, 16)


def _select_num_splits(num_tokens: int) -> int:
    """Choose a native-supported DeepGEMM split count without exposing it.

    C500 measurements show that the MHC GEMM is severely under-filled with
    four K splits at decode sizes: for N=1/4/8, 16 splits reduce the steady
    DeepGEMM latency from about 0.199 ms to 0.053 ms.  The native big-fuse
    reduction changes by less than 0.001 ms, so use the maximum supported
    split parallelism for every non-empty H=5120 workload.
    """
    if num_tokens == 0:
        return 1
    return 16


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _residual_layout(
    residual: torch.Tensor,
) -> tuple[tuple[int, ...], int, torch.Tensor]:
    if not isinstance(residual, torch.Tensor):
        raise TypeError("residual must be a torch.Tensor")
    if residual.ndim < 3 or tuple(residual.shape[-2:]) != (
        HC_MULT,
        HIDDEN_SIZE,
    ):
        raise ValueError(
            "residual must have shape [..., 4, 5120], "
            f"got {tuple(residual.shape)}"
        )
    if residual.dtype != torch.bfloat16:
        raise TypeError(
            f"residual must have dtype {torch.bfloat16}, got {residual.dtype}"
        )
    if not residual.is_cuda:
        raise ValueError("residual must be a CUDA tensor")
    if not residual.is_contiguous():
        raise ValueError("residual must be contiguous")

    outer_shape = tuple(residual.shape[:-2])
    num_tokens = residual.numel() // HC_HIDDEN_SIZE
    residual_flat = residual.view(num_tokens, HC_MULT, HIDDEN_SIZE)
    return outer_shape, num_tokens, residual_flat


def _prepare_pre_mix(
    name: str,
    tensor: torch.Tensor | None,
    outer_shape: tuple[int, ...],
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None")
    expected = (*outer_shape, HC_MULT)
    flat_expected = (num_tokens, HC_MULT)
    if tuple(tensor.shape) not in (expected, flat_expected):
        raise ValueError(
            f"{name} must have shape {expected} or {flat_expected}, "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor.view(num_tokens, HC_MULT)


@cache
def _load_deepgemm_prenorm() -> Callable[..., None]:
    try:
        import deep_gemm
    except ImportError as exc:
        raise RuntimeError(
            "MHC requires DeepGEMM and its tf32_hc_prenorm_gemm API"
        ) from exc

    op = getattr(deep_gemm, "tf32_hc_prenorm_gemm", None)
    if op is None:
        raise RuntimeError("DeepGEMM does not expose tf32_hc_prenorm_gemm")
    return op


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
    pre_mix_in: torch.Tensor | None = None,
    pre_mix_out: torch.Tensor | None = None,
    quant_mode: str = "none",
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
):
    """Run the complete MHC pre path and return ``post, comb, layer_input``.

    The output order follows ``triton_sglang_mhc.mhc_pre``. DeepGEMM's split
    count is an internal implementation detail selected from the values
    supported by the native big-fuse kernel.

    Delayed-pre modes are selected by the optional tensors:

    * ``pre_mix_out`` without ``pre_mix_in`` saves the current pre weights and
      collapses the residual with the ``[1, 0, 0, 0]`` first-layer sentinel.
    * Supplying both tensors collapses with ``pre_mix_in`` and saves the newly
      computed weights into ``pre_mix_out`` for the next layer.
    * ``norm_weight`` fuses RMSNorm into the layer-input write and requires
      ``norm_eps``.

    ``pre_mix_out`` is an out parameter and is not included in the return
    tuple; the caller retains the same Tensor and can pass it to the next call.

    ``quant_mode='int8'`` returns an INT8 layer input and a separate FP32
    per-token scale plus the BF16 RMSNorm output. ``quant_mode='pack_int8'``
    returns a BF16 storage container whose bytes hold 5120 INT8 values, one
    FP32 scale, and zero padding per token; it additionally returns the separate
    scale, BF16 RMSNorm output and unpacked INT8 output. Both modes require
    fused RMSNorm. The default mode retains the original three-result BF16
    interface.
    """
    outer_shape, num_tokens, residual_flat = _residual_layout(residual)
    device = residual.device
    _require_tensor("fn", fn, (MIX_SIZE, HC_HIDDEN_SIZE), torch.float32, device)
    _require_tensor("hc_scale", hc_scale, (3,), torch.float32, device)
    _require_tensor("hc_base", hc_base, (MIX_SIZE,), torch.float32, device)

    n_splits = _select_num_splits(num_tokens)
    assert n_splits in SUPPORTED_NUM_SPLITS
    if sinkhorn_repeat != 20:
        raise ValueError("the native H=5120 path requires sinkhorn_repeat=20")

    pre_mix_in_flat = _prepare_pre_mix(
        "pre_mix_in", pre_mix_in, outer_shape, num_tokens, device
    )
    pre_mix_out_flat = _prepare_pre_mix(
        "pre_mix_out", pre_mix_out, outer_shape, num_tokens, device
    )
    if quant_mode not in ("none", "int8", "pack_int8"):
        raise ValueError("quant_mode must be 'none', 'int8', or 'pack_int8'")
    if quant_mode != "none" and norm_weight is None:
        raise ValueError("quantized mhc_pre requires norm_weight")

    norm_weight_bf: torch.Tensor | None = None
    if norm_weight is not None:
        if not isinstance(norm_weight, torch.Tensor):
            raise TypeError("norm_weight must be a torch.Tensor or None")
        if tuple(norm_weight.shape) != (HIDDEN_SIZE,):
            raise ValueError(
                f"norm_weight must have shape ({HIDDEN_SIZE},), "
                f"got {tuple(norm_weight.shape)}"
            )
        if norm_weight.device != device:
            raise ValueError(
                f"norm_weight must be on {device}, got {norm_weight.device}"
            )
        if norm_eps is None:
            raise ValueError("norm_eps is required when norm_weight is provided")
        norm_weight_bf = norm_weight.to(dtype=torch.bfloat16)
        if not norm_weight_bf.is_contiguous():
            norm_weight_bf = norm_weight_bf.contiguous()

    post_mix_flat = torch.empty(
        (num_tokens, HC_MULT), device=device, dtype=torch.float32
    )
    comb_mix_flat = torch.empty(
        (num_tokens, HC_MULT * HC_MULT), device=device, dtype=torch.float32
    )
    if quant_mode == "int8":
        layer_input_flat = torch.empty(
            (num_tokens, HIDDEN_SIZE), device=device, dtype=torch.int8
        )
    elif quant_mode == "pack_int8":
        layer_input_flat = torch.empty(
            (num_tokens, PACKED_INT8_ROW_ELEMENTS),
            device=device,
            dtype=torch.bfloat16,
        )
    else:
        layer_input_flat = torch.empty(
            (num_tokens, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
        )
    quant_scales_flat = (
        torch.empty((num_tokens,), device=device, dtype=torch.float32)
        if quant_mode != "none"
        else None
    )
    output_rms_flat = (
        torch.empty(
            (num_tokens, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
        )
        if quant_mode != "none"
        else None
    )
    output_quant_int8_flat = (
        torch.empty((num_tokens, HIDDEN_SIZE), device=device, dtype=torch.int8)
        if quant_mode == "pack_int8"
        else None
    )

    if num_tokens != 0:
        part_mix = torch.empty(
            (n_splits, num_tokens, MIX_SIZE),
            device=device,
            dtype=torch.float32,
        )
        part_sq = torch.empty(
            (n_splits, num_tokens), device=device, dtype=torch.float32
        )

        deepgemm_prenorm = _load_deepgemm_prenorm()
        deepgemm_prenorm(
            residual_flat.view(num_tokens, HC_HIDDEN_SIZE),
            fn,
            part_mix,
            part_sq,
            num_splits=n_splits,
        )

        mcoplib_ops.mhc_pre_big_fuse_out(
            part_mix,
            part_sq,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix_flat,
            comb_mix_flat,
            layer_input_flat,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight=norm_weight_bf,
            pre_mix_in=pre_mix_in_flat,
            pre_mix_out=pre_mix_out_flat,
            aux_out=None,
            norm_eps=1e-6 if norm_eps is None else norm_eps,
            rms_numel=HC_HIDDEN_SIZE,
            quant_scales=quant_scales_flat,
            output_rms=output_rms_flat,
            output_quant_int8=output_quant_int8_flat,
        )

    post_mix = post_mix_flat.view(*outer_shape, HC_MULT, 1)
    comb_mix = comb_mix_flat.view(*outer_shape, HC_MULT, HC_MULT)
    layer_input = layer_input_flat.view(*outer_shape, layer_input_flat.shape[-1])
    if quant_mode == "pack_int8":
        assert quant_scales_flat is not None
        assert output_rms_flat is not None
        assert output_quant_int8_flat is not None
        return (
            post_mix,
            comb_mix,
            layer_input,
            quant_scales_flat.view(*outer_shape),
            output_rms_flat.view(*outer_shape, HIDDEN_SIZE),
            output_quant_int8_flat.view(*outer_shape, HIDDEN_SIZE),
        )
    if quant_scales_flat is not None:
        assert output_rms_flat is not None
        return (
            post_mix,
            comb_mix,
            layer_input,
            quant_scales_flat.view(*outer_shape),
            output_rms_flat.view(*outer_shape, HIDDEN_SIZE),
        )
    return post_mix, comb_mix, layer_input


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Run MHC post with ``mcoplib.op.mhc_post_cuda``.

    Shapes follow the existing TileLang interface: ``x`` is ``[..., 5120]``,
    ``residual`` is ``[..., 4, 5120]``, ``post_layer_mix`` may be
    ``[..., 4]`` or ``[..., 4, 1]``, and ``comb_res_mix`` is ``[..., 4, 4]``.
    """
    # Production calls already use the native [N, ...] layout.  Delegate that
    # layout directly: mhc_post_cuda performs the same shape, dtype, device,
    # contiguity and alias checks in C++, while avoiding duplicated Python
    # validation and no-op views that are visible at decode-scale latency.
    if (
        isinstance(residual, torch.Tensor)
        and residual.ndim == 3
        and residual.size(1) == HC_MULT
        and residual.size(2) == HIDDEN_SIZE
    ):
        return mcoplib_ops.mhc_post_cuda(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )

    # Preserve support for arbitrary outer dimensions by flattening them to
    # the native [N, ...] contract and restoring the caller-visible layout.
    outer_shape, num_tokens, residual_flat = _residual_layout(residual)
    device = residual.device
    _require_tensor(
        "x", x, (*outer_shape, HIDDEN_SIZE), torch.bfloat16, device
    )
    _require_tensor(
        "comb_res_mix",
        comb_res_mix,
        (*outer_shape, HC_MULT, HC_MULT),
        torch.float32,
        device,
    )

    if not isinstance(post_layer_mix, torch.Tensor):
        raise TypeError("post_layer_mix must be a torch.Tensor")
    post_shape = tuple(post_layer_mix.shape)
    expected_post = (*outer_shape, HC_MULT)
    expected_post_3d = (*outer_shape, HC_MULT, 1)
    if post_shape not in (expected_post, expected_post_3d):
        raise ValueError(
            "post_layer_mix must have shape "
            f"{expected_post} or {expected_post_3d}, got {post_shape}"
        )
    if post_layer_mix.dtype != torch.float32:
        raise TypeError(
            "post_layer_mix must have dtype torch.float32, "
            f"got {post_layer_mix.dtype}"
        )
    if post_layer_mix.device != device:
        raise ValueError(
            f"post_layer_mix must be on {device}, got {post_layer_mix.device}"
        )
    if not post_layer_mix.is_contiguous():
        raise ValueError("post_layer_mix must be contiguous")

    if num_tokens == 0:
        return torch.empty_like(residual)

    output = mcoplib_ops.mhc_post_cuda(
        x.view(num_tokens, HIDDEN_SIZE),
        residual_flat,
        post_layer_mix.view(num_tokens, HC_MULT),
        comb_res_mix.view(num_tokens, HC_MULT, HC_MULT),
    )
    return output.view(*outer_shape, HC_MULT, HIDDEN_SIZE)


__all__ = [
    "mhc_pre",
    "mhc_post",
]
