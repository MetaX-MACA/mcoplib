import pytest
import torch
import mcoplib._moe_C


NUM_TOKENS = [
    1,
    8,
    32,
    128,
    1024,
]

HIDDEN_DIMS = [
    512,
    2048,
    7168,
]

NUM_EXPERTS = [
    16,
    64,
    256,
]

TOP_KS = [
    2,
    4,
    8,
]

DTYPES = [
    torch.bfloat16,
]


# =====================================================
# reference
# =====================================================

def torch_permute_reference(
    hidden_states,
    topk_ids,
    token_expert_indices,
    topk,
    n_expert,
):

    sorted_topk_ids, sorted_indices = torch.sort(
        topk_ids.flatten(),
        stable=True,
    )

    dst_row_id2src_row_id_map = (
        token_expert_indices.flatten()[sorted_indices]
    )


    offset = torch.zeros(
        n_expert + 1,
        dtype=torch.int64,
        device=hidden_states.device,
    )

    idx = 0

    for i in range(n_expert):
        count = 0

        while (
            idx < sorted_topk_ids.numel()
            and sorted_topk_ids[idx] == i
        ):
            count += 1
            idx += 1

        offset[i + 1] = offset[i] + count

    permuted = hidden_states[
        dst_row_id2src_row_id_map // topk
    ]

    # original index -> permuted index
    _, src2dst = torch.sort(
        dst_row_id2src_row_id_map
    )

    inv = torch.arange(
        hidden_states.shape[0] * topk,
        dtype=torch.int32,
        device=hidden_states.device,
    )[src2dst]

    inv = inv.reshape(
        hidden_states.shape[0],
        topk,
    )

    return (
        permuted,
        offset,
        inv,
        dst_row_id2src_row_id_map,
    )

# =====================================================
# basic permute
# =====================================================

@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden", HIDDEN_DIMS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_moe_permute(
    num_tokens,
    hidden,
    num_experts,
    topk,
    dtype,
):
    device="cuda"
    torch.manual_seed(0)

    hidden_states = torch.randn(
        num_tokens,
        hidden,
        dtype=dtype,
        device=device,
    )

    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        dtype=torch.int32,
        device=device,
    )

    token_expert_indices = torch.arange(
        num_tokens * topk,
        dtype=torch.int32,
        device=device,
    ).reshape(
        num_tokens,
        topk,
    )

    total = num_tokens * topk

    output = torch.empty(
        total,
        hidden,
        dtype=dtype,
        device=device,
    )

    offset = torch.empty(
        num_experts + 1,
        dtype=torch.int64,
        device=device,
    )

    inv = torch.empty(
        num_tokens,
        topk,
        dtype=torch.int32,
        device=device,
    )

    idx = torch.empty(
        total,
        dtype=torch.int32,
        device=device,
    )

    torch.ops._moe_C.moe_permute(
        hidden_states,
        topk_ids,
        token_expert_indices,
        None,
        num_experts,
        num_experts,
        topk,
        output,
        offset,
        inv,
        idx,
    )

    ref_output, ref_offset, ref_inv, ref_idx = (
        torch_permute_reference(
            hidden_states,
            topk_ids,
            token_expert_indices,
            topk,
            num_experts,
        )
    )

    torch.testing.assert_close(
        output,
        ref_output,
    )

    torch.testing.assert_close(
        offset,
        ref_offset,
    )

    torch.testing.assert_close(
        inv,
        ref_inv,
    )

    torch.testing.assert_close(
        idx,
        ref_idx,
    )

    print(
        "permute PASS",
        num_tokens,
        hidden,
        num_experts,
        topk,
    )

# =====================================================
# one expert
# =====================================================

def test_moe_permute_one_expert():
    device="cuda"
    tokens=64
    hidden=2048
    experts=256
    topk=8

    hidden_states=torch.randn(
        tokens,
        hidden,
        device=device,
        dtype=torch.bfloat16,
    )

    topk_ids=torch.zeros(
        tokens,
        topk,
        dtype=torch.int32,
        device=device,
    )

    token_expert_indices=torch.arange(
        tokens * topk,
        dtype=torch.int32,
        device=device,
    ).reshape(
        tokens,
        topk,
    )

    output=torch.empty(
        tokens*topk,
        hidden,
        dtype=torch.bfloat16,
        device=device,
    )

    offset=torch.empty(
        experts+1,
        dtype=torch.int64,
        device=device,
    )

    inv=torch.empty(
        tokens,
        topk,
        dtype=torch.int32,
        device=device,
    )

    idx=torch.empty(
        tokens*topk,
        dtype=torch.int32,
        device=device,
    )

    torch.ops._moe_C.moe_permute(
        hidden_states,
        topk_ids,
        token_expert_indices,
        None,
        experts,
        experts,
        topk,
        output,
        offset,
        inv,
        idx,
    )
    assert offset[1].item() == tokens*topk
    print("one expert PASS")

# =====================================================
# permute + unpermute
# =====================================================

def test_moe_permute_unpermute():
    device="cuda"
    tokens=32
    hidden=1024
    experts=64
    topk=4

    hidden_states=torch.randn(
        tokens,
        hidden,
        dtype=torch.bfloat16,
        device=device,
    )

    topk_ids=torch.randint(
        0,
        experts,
        (tokens,topk),
        dtype=torch.int32,
        device=device,
    )

    token_expert_indices=torch.arange(
        tokens*topk,
        dtype=torch.int32,
        device=device,
    ).reshape(
        tokens,
        topk,
    )

    permuted=torch.empty(
        tokens*topk,
        hidden,
        dtype=torch.bfloat16,
        device=device,
    )

    offset=torch.empty(
        experts+1,
        dtype=torch.int64,
        device=device,
    )

    inv=torch.empty(
        tokens,
        topk,
        dtype=torch.int32,
        device=device,
    )

    idx=torch.empty(
        tokens*topk,
        dtype=torch.int32,
        device=device,
    )

    torch.ops._moe_C.moe_permute(
        hidden_states,
        topk_ids,
        token_expert_indices,
        None,
        experts,
        experts,
        topk,
        permuted,
        offset,
        inv,
        idx,
    )

    # fake expert output
    expert_output = permuted + 1

    topk_weights=torch.ones(
        tokens,
        topk,
        dtype=torch.float32,
        device=device,
    )

    output=torch.empty_like(
        hidden_states
    )

    torch.ops._moe_C.moe_unpermute(
        expert_output,
        topk_weights,
        inv,
        offset,
        topk,
        output,
    )
    assert output.shape == hidden_states.shape
    print("permute+unpermute PASS")

if __name__=="__main__":

    test_moe_permute_one_expert()
    test_moe_permute_unpermute()