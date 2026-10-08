import triton
import triton.language as tl
import torch
import mcoplib
from mcoplib.op import step4_weighted_topk_gather

@triton.jit
def _weighted_topk_gather_kernel(
    contributions_ptr,
    weights_ptr,
    output_ptr,
    hidden_size,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce one ``[top_k, hidden]`` slice in deterministic slot order."""
    token = tl.program_id(0).to(tl.int64)
    hidden_block = tl.program_id(1).to(tl.int64)
    hidden_offsets = hidden_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    # TOP_K is constexpr.  Keeping this loop serial in slot order is
    # required: a tree reduction can change the final BF16 rounding.
    for slot in range(0, TOP_K):
        contribution_offset = (
            (token * TOP_K + slot) * hidden_size + hidden_offsets
        )
        contribution = tl.load(
            contributions_ptr + contribution_offset,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(weights_ptr + token * TOP_K + slot).to(tl.float32)
        accumulator += contribution * weight
    output_offset = token * hidden_size + hidden_offsets
    tl.store(
        output_ptr + output_offset,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


def weighted_topk_gather(contributions: torch.Tensor, weights: torch.Tensor, BLOCK_SIZE=128):
    """
    Triton kernel wrapper
    contributions: [num_tokens, TOP_K, hidden_size], bfloat16, cuda
    weights: [num_tokens, TOP_K], bfloat16, cuda
    returns output: [num_tokens, hidden_size], bfloat16
    """
    assert contributions.is_cuda and weights.is_cuda, "Inputs must be CUDA tensors"
    assert contributions.dtype == torch.bfloat16, "contributions must be bfloat16"
    # assert weights.dtype == torch.bfloat16, "weights must be bfloat16"

    num_tokens, TOP_K, hidden_size = contributions.shape
    assert weights.shape == (num_tokens, TOP_K), "weights shape mismatch"

    output = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16, device=contributions.device)
    grid = lambda meta: (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))

    _weighted_topk_gather_kernel[grid](
        contributions,
        weights,
        output,
        hidden_size,
        TOP_K=TOP_K,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# CASES: (num_tokens, TOP_K, hidden_size)
CASES = (
    (0, 8, 4096),
    (1, 1, 1),
    (1, 3, 191),
    (2, 7, 192),
    (7, 8, 1536),
    (47, 8, 4096),
    (48, 8, 4096),
    (49, 8, 4096),
    (64, 16, 4096),
    (5, 63, 257),
    (3, 64, 4097),
    (256, 8, 4096),
)


def run_test():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # BF16 误差容忍
    atol = 1e-2
    rtol = 1e-2

    passed_count = 0
    failed_cases = []

    for idx, (num_tokens, topk, hidden) in enumerate(CASES):
        print(f"\n======== Test[{idx}] num_tokens={num_tokens}, TOP_K={topk}, hidden={hidden} ========")
        # 构造 bfloat16 cuda input
        contrib = torch.randn(num_tokens, topk, hidden, dtype=torch.bfloat16, device="cuda")
        
        w = torch.randn((num_tokens, topk), dtype=torch.float32, device="cuda")
        
        # triton kernel result
        out_triton = weighted_topk_gather(contrib, w)

        out_check = torch.zeros_like(out_triton, dtype=torch.bfloat16, device="cuda")
        step4_weighted_topk_gather(contrib, w, out_check)
        
        if num_tokens == 0:
            continue
        # compare
        abs_diff = (out_triton.float() - out_check.float()).abs()
        max_diff = float(abs_diff.max())
        is_ok = torch.allclose(out_triton, out_check, atol=atol, rtol=rtol)

        print(f"max_abs_diff={max_diff:.6f}, PASS={is_ok}")
        if is_ok:
            passed_count += 1
        else:
            failed_cases.append((num_tokens, topk, hidden))

    print("\n================ Test Summary ================")
    print(f"Total cases: {len(CASES)}, Passed: {passed_count}")
    if failed_cases:
        print(f"Failed cases: {failed_cases}")
    else:
        print("✅ All test cases passed!")


if __name__ == "__main__":
    run_test()
