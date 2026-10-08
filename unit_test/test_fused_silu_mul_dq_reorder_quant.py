"""Test for `fused_silu_mul_dq_reorder_quant` (op/int8_quant_kernels.cu).

Fused kernel: for every token, silu(gate) * up (with an optional per-expert
w2 scale), then dynamic per-token int8 quantization of the result.

Signature:
    fused_silu_mul_dq_reorder_quant(out, scale, input, reorder_topk_ids,
                                    w2_scale, start_expert_id, end_expert_id)

- input:            (num_tokens, hidden_size)  bfloat16, [gate | up] concatenated
- out:              (num_tokens, hidden_size // 2)  int8
- scale:            (num_tokens, 1)  float32
- reorder_topk_ids: (num_tokens,)  int64, expert id of each token
- w2_scale:         empty, or (num_experts,) float32 per-expert scale

Only the empty ``w2_scale`` path (no per-expert scaling) is exercised here;
the op is validated against a triton reference that reproduces the same
silu-mul + per-token int8 quant.
"""

import torch
import triton
import triton.language as tl

from mcoplib.op import fused_silu_mul_dq_reorder_quant


@triton.jit
def _silu_and_mul_triton_kernel(
    gateup_output,
    down_input,
    hidden_size,
    reorder_topk_ids,
    scales,
    start_expert_id,
    end_expert_id,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2

    pid = tl.program_id(0)
    expert_id = tl.load(reorder_topk_ids + pid)
    if expert_id >= start_expert_id and expert_id <= end_expert_id:
        gateup_output_ptr = gateup_output + pid * hidden_size
        gate_output_ptr = gateup_output_ptr
        up_output_ptr = gateup_output_ptr + half_hidden_size
        down_input_ptr = down_input + pid * half_hidden_size

        if scales is not None:
            scale = tl.load(scales + expert_id - start_expert_id)
            scale = (1 / scale).to(InDtype)
        else:
            scale = 1

        for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
            offset = start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offset < half_hidden_size

            gate_output = tl.load(gate_output_ptr + offset, mask=mask).to(tl.float32)
            up_output = tl.load(up_output_ptr + offset, mask=mask)

            gate_output = gate_output * tl.sigmoid(gate_output)
            gate_output = gate_output.to(InDtype)

            silu_mul_output = gate_output * up_output * scale
            silu_mul_output = silu_mul_output.to(OutDtype)
            tl.store(down_input_ptr + offset, silu_mul_output, mask=mask)


def _ref_dynamic_per_token_quant(x, quant_dtype=torch.int8):
    """Reference per-token int8 quantization matching the CUDA kernel."""
    qmax = torch.iinfo(quant_dtype).max  # 127
    x_token_max = x.abs().max(dim=-1).values.float()
    scales = (x_token_max / qmax)[:, None]
    iscales = (1.0 / scales)
    out = (x.float() * iscales).round().clamp(-128, 127).to(torch.int8)
    return out, scales


def run_case(num_tokens, hidden_size, num_experts):
    torch.manual_seed(0)
    device = "cuda"

    gateup_output = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    reorder_topk_ids = torch.randint(0, num_experts, (num_tokens,), device=device, dtype=torch.int64)
    # reference uses per-expert ones (1/1 = 1); op is called with empty w2_scale
    # (no scaling) — both effectively scale = 1.
    w2_input_scale = torch.ones(num_experts, device=device, dtype=torch.float32)

    # --- reference: triton silu-mul, then per-token int8 quant ---
    down_input = torch.empty(num_tokens, hidden_size // 2, device=device, dtype=torch.bfloat16)
    _silu_and_mul_triton_kernel[(num_tokens,)](
        gateup_output,
        down_input,
        hidden_size,
        reorder_topk_ids,
        w2_input_scale,
        0,
        num_experts - 1,
        BLOCK_SIZE=512,
    )
    ref_out, ref_scale = _ref_dynamic_per_token_quant(down_input, torch.int8)

    # --- op under test ---
    out = torch.empty(num_tokens, hidden_size // 2, device=device, dtype=torch.int8)
    scale = torch.empty(num_tokens, 1, device=device, dtype=torch.float32)
    fused_silu_mul_dq_reorder_quant(
        out,
        scale,
        gateup_output.clone(),
        reorder_topk_ids,
        torch.empty(0, device=device, dtype=torch.float32),
        0,
        num_experts - 1,
    )
    torch.cuda.synchronize()

    # The op fuses silu-mul + quant in one kernel while the reference goes
    # through a bf16 intermediate, so int8 can differ by a couple of units.
    out_diff = (out.to(torch.float32) - ref_out.to(torch.float32)).abs()
    assert out_diff.max().item() <= 4, f"int8 output max diff {out_diff.max().item()}"
    assert out_diff.mean().item() <= 0.1, f"int8 output mean diff {out_diff.mean().item()}"

    scale_diff = (scale - ref_scale).abs()
    assert scale_diff.max().item() <= 1e-2, f"scale max diff {scale_diff.max().item()}"


def test_fused_silu_mul_dq_reorder_quant():
    # hidden_size must be even (gate/up split); keep 4096 as in the original script.
    for num_tokens in (512, 1024, 2048, 3200):
        run_case(num_tokens=num_tokens, hidden_size=4096, num_experts=8)


if __name__ == "__main__":
    test_fused_silu_mul_dq_reorder_quant()
    print("test_fused_silu_mul_dq_reorder_quant PASS")
