from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.library
import mcoplib.op as ops

def _prepare_add_rms_norm_args(input, weight,  epsilon):

    if input is None:
        raise ValueError("`input` is required and must not be None")
    assert input.is_contiguous(), "`input` must be contiguous"
    hidden = input.shape[-1]

    if weight is None:
        raise ValueError("`weight` is required and must not be None")
    assert weight.is_contiguous(), "`weight` must be contiguous"
    assert weight.shape[-1] == hidden, \
        f"weight last dim({weight.shape[-1]}) must equal hidden({hidden})"
    assert weight.device == input.device, "`weight` and `input` must be on the same device"


    epsilon = 1e-6 if epsilon is None else float(epsilon)


    return epsilon


def fused_attention_prepare(
    qkv: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    base: float,
    max_position_embedding: int,
    rms_norm_eps: Optional[float] = None,
    partial_rotary_factor: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:                   
    out_q = torch.zeros((qkv.size(0), num_heads * head_dim), dtype=qkv.dtype, device=qkv.device)
    out_kv = torch.zeros((qkv.size(0), num_kv_heads * head_dim), dtype=qkv.dtype, device=qkv.device)
    eps = 1e-6 if rms_norm_eps is None else rms_norm_eps
    partial_rotary_factor = 1.0 if partial_rotary_factor is None else partial_rotary_factor
    ops.FusedAttentionPrepare(qkv, weight, positions, out_q, out_kv, num_heads, num_kv_heads,
                              head_dim, base, max_position_embedding, eps, partial_rotary_factor)
    return out_q, out_kv


def fused_add_rms_norm_dynamic_per_token_quant_padding_output(
    input: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    pad_size: Optional[int] = None,
    epsilon: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:  
    
    epsilon = _prepare_add_rms_norm_args(input, weight, epsilon)

    if residual is not None:
        assert residual.is_contiguous(), "`residual` must be contiguous"
        assert residual.shape == input.shape, \
            f"residual shape{tuple(residual.shape)} must match input{tuple(input.shape)}"
            
    shape = list(input.shape)
    shape[-1] = pad_size
    output = torch.empty(shape, dtype=torch.bfloat16, device=input.device)
    assert weight.is_contiguous()
    C_output = output.view(dtype=torch.int8)

    output_rms = torch.empty_like(input)
    output_quant_int8 = torch.empty_like(input, dtype=torch.int8)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                         device=input.device, dtype=torch.float32)
    ops.fused_add_rms_norm_dynamic_per_token_quant_padding_output(
        C_output, output_rms, output_quant_int8, scales, input, residual, weight, pad_size, epsilon)

    residual_out = residual                                
    return output, residual_out, output_rms, output_quant_int8, scales


def fused_fp8_add_rms_norm_dynamic_per_token_quant_padding_output(
    input: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    pad_size: Optional[int] = None,
    epsilon: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]: 
    
    epsilon = _prepare_add_rms_norm_args(input, weight, epsilon)

    if residual is not None:
        assert residual.is_contiguous(), "`residual` must be contiguous"
        assert residual.shape == input.shape, \
            f"residual shape{tuple(residual.shape)} must match input{tuple(input.shape)}"
            
    shape = list(input.shape)
    shape[-1] = pad_size
    output = torch.empty(shape, dtype=torch.bfloat16, device=input.device)
    assert weight.is_contiguous()
    C_output = output.view(dtype=torch.float8_e4m3fn)

    output_rms = torch.empty_like(input)
    output_quant_fp8 = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                         device=input.device, dtype=torch.float32)
    ops.fused_add_rms_norm_dynamic_per_token_quant_padding_output(
        C_output, output_rms, output_quant_fp8, scales, input, residual, weight, pad_size, epsilon)

    residual_out = residual
    return output, residual_out, output_rms, output_quant_fp8, scales


def rms_norm_dynamic_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    scale_ub: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input, dtype=quant_dtype)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                         device=input.device, dtype=torch.float32)
    ops.rms_norm_dynamic_per_token_quant_custom(
        output, input, weight, scales, epsilon, scale_ub, residual)
    return output, scales