from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.library
import mcoplib.op as ops

def fused_attention_prepare(qkv :torch.tensor,
                            weight :torch.tensor,
                            positions :torch.tensor,
                            num_heads :int,
                            num_kv_heads :int,
                            head_dim :int,
                            base :float,
                            max_position_embedding :int,
                            rms_norm_eps :Optional[float] = None,
                            partial_rotary_factor :Optional[float] = None):
    out_q = torch.zeros((qkv.size(0), num_heads * head_dim), dtype=qkv.dtype, device=qkv.device)
    out_kv = torch.zeros((qkv.size(0), num_kv_heads * head_dim), dtype=qkv.dtype, device=qkv.device)
    ops.FusedAttentionPrepare(qkv, weight, positions, out_q, out_kv, num_heads, num_kv_heads, head_dim, base,
                              max_position_embedding, rms_norm_eps, partial_rotary_factor)
    return out_q, out_kv

def fused_add_rms_norm_dynamic_per_token_quant_padding_output(input: torch.tensor, 
                                                              residual: torch.tensor,
                                                              weight: torch.tensor,
                                                              pad_size: int,
                                                              epsilon: float
                                                              ) -> tuple[torch.tensor, torch.tensor]:
    assert input.is_contiguous()
    shape = list(input.shape)
    shape[-1] = pad_size
    output = torch.empty(shape, dtype=torch.bfloat16, device=input.device)
    assert weight.is_contiguous()
    C_output = output.view(dtype=torch.int8)

    output_rms = torch.empty_like(input)
    output_quant_int8 = torch.empty_like(input, dtype=torch.int8)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                         device=input.device,
                         dtype=torch.float32)
    ops.fused_add_rms_norm_dynamic_per_token_quant_padding_output(C_output, output_rms, output_quant_int8, scales, input, residual,
                                                  weight, pad_size, epsilon)
    return output, residual, output_rms, output_quant_int8, scales

def fused_fp8_add_rms_norm_dynamic_per_token_quant_padding_output(input: torch.tensor, 
                                                              residual: torch.tensor,
                                                              weight: torch.tensor,
                                                              pad_size: int,
                                                              epsilon: float
                                                              ) -> tuple[torch.tensor, torch.tensor]:
    assert input.is_contiguous()
    shape = list(input.shape)
    shape[-1] = pad_size
    output = torch.empty(shape, dtype=torch.bfloat16, device=input.device)
    assert weight.is_contiguous()
    C_output = output.view(dtype=torch.float8_e4m3fn)

    output_rms = torch.empty_like(input)
    output_quant_fp8 = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                         device=input.device,
                         dtype=torch.float32)
    ops.fused_add_rms_norm_dynamic_per_token_quant_padding_output(C_output, output_rms, output_quant_fp8, scales, input, residual,
                                                  weight, pad_size, epsilon)
    return output, residual, output_rms, output_quant_fp8, scales

def rms_norm_dynamic_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    scale_ub: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input, dtype=quant_dtype)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                         device=input.device,
                         dtype=torch.float32)

    ops.rms_norm_dynamic_per_token_quant_custom(output, input, weight,
                                                  scales, epsilon, scale_ub,
                                                  residual)
    return output, scales