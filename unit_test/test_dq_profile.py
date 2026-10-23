import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import mcoplib._C
from typing import Optional, Union

def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (
            azp
            is None), "azp must only be provided for asymmetric quantization."
        torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, azp

    # dynamic-per-token quantization.
    input_scales = torch.empty((input.numel() // input.shape[-1], 1),
                               device=input.device,
                               dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales,
                                                        dtype=torch.int32)
    # import pdb 
    # pdb.set_trace()
    torch.ops._C.dynamic_scaled_int8_quant(output, input.contiguous(),
                                           input_scales, input_azp)
    return output, input_scales, input_azp

# ======================
# 2. 造输入
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
attn_output = torch.randn(32, 128, 512, device=device, dtype=torch.float16)  # 你常用的shape
print(attn_output.transpose(0, 1).contiguous().shape)
# ======================
# 3. 预热（必须）
# ======================
for _ in range(5):
    attn_output_quant_val, attn_output_quant_scale, _ = scaled_int8_quant(attn_output.transpose(0, 1).contiguous())
torch.cuda.synchronize()
print(f"预热完成")

# ======================
# 4. Profiler 测速
# ======================
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(100):
        attn_output_quant_val, attn_output_quant_scale, _ = scaled_int8_quant(attn_output.transpose(0, 1).contiguous())
        torch.cuda.synchronize()

# ======================
# 5. 打印结果（按CUDA耗时排）
# ======================
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))