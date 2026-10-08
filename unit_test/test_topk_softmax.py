import itertools
import sys

import pytest
import torch
import mcoplib._moe_C


# ============================================================
# Test scenarios
#
# (shared_experts, num_experts, num_expert_group, group_topk, topk)
# ============================================================
TOPK_SOFTMAX_SCENARIOS = [
    (0, 160, 1, 1, 8),
    (0, 256, 8, 4, 8),
    (0, 256, 1, 1, 8),
    (0, 320, 1, 1, 8),
    (0, 384, 1, 1, 8),
    (0, 448, 1, 1, 8),
    (0, 288, 1, 1, 8),

    # TopK=16, 无共享专家配置 (kimi-k3 896 专家)
    (0, 896, 1, 1, 16),

    # TopK=9, 1个共享专家配置
    (1, 160, 1, 1, 9),
    (1, 256, 1, 1, 9),
    (1, 256, 8, 4, 9),
    (1, 320, 1, 1, 9),
    (1, 384, 1, 1, 9),
    (1, 448, 1, 1, 9),

    # TopK=9, 2个共享专家配置
    (2, 256, 8, 4, 9),
    (2, 320, 1, 1, 9),
    (2, 384, 1, 1, 9),
    (2, 448, 1, 1, 9),
]


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    bias=None,
) -> None:
    """
    调用 MCOPLIB topk_softmax。

    C++:
        topk_softmax(
            Tensor! topk_weights,
            Tensor! topk_indices,
            Tensor! token_expert_indices,
            Tensor gating_output,
            bool renormalize,
            Tensor? bias)
    """

    torch.ops._moe_C.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        bias,
    )


def compare_topk_values(
    gating_output,
    topk_indices_ref,
    topk_indices,
):
    values_ref = torch.gather(
        gating_output,
        1,
        topk_indices_ref,
    )

    values = torch.gather(
        gating_output,
        1,
        topk_indices,
    )

    return torch.equal(
        values_ref,
        values,
    )


@pytest.mark.parametrize(
    "shared_experts,num_experts,num_expert_group,group_topk,topk",
    TOPK_SOFTMAX_SCENARIOS,
)
def test_topk_softmax(
    shared_experts,
    num_experts,
    num_expert_group,
    group_topk,
    topk,
):
    """
    测试 topk_softmax 基础 FP32 路径。

    注意：
        shared_experts / num_expert_group / group_topk
        是模型路由场景参数，目前 topk_softmax 本身不使用。
    """

    num_tokens = 4096
    dtype = torch.float32

    print(
        f"\n=== test_topk_softmax ===\n"
        f"shared_experts={shared_experts}, "
        f"num_experts={num_experts}, "
        f"num_expert_group={num_expert_group}, "
        f"group_topk={group_topk}, "
        f"topk={topk}, "
        f"num_tokens={num_tokens}, "
        f"dtype={dtype}"
    )

    assert topk <= num_experts, (
        f"TOP_K={topk} > NUM_EXPERTS={num_experts}"
    )

    gating_output = torch.randn(
        (num_tokens, num_experts),
        dtype=dtype,
        device="cuda",
    )

    topk_weights = torch.empty(
        (num_tokens, topk),
        dtype=torch.float32,
        device="cuda",
    )

    topk_indices = torch.empty(
        (num_tokens, topk),
        dtype=torch.int32,
        device="cuda",
    )

    token_expert_indices = torch.empty(
        (num_tokens, topk),
        dtype=torch.int32,
        device="cuda",
    )

    # --------------------------------------------------------
    # CUDA op
    # --------------------------------------------------------
    topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
        bias=None,
    )

    torch.cuda.synchronize()

    # --------------------------------------------------------
    # PyTorch reference
    # --------------------------------------------------------
    softmax_output = torch.softmax(
        gating_output,
        dim=-1,
    )

    topk_weights_ref, topk_indices_ref = torch.topk(
        softmax_output,
        topk,
        dim=-1,
        sorted=True,
    )

    # --------------------------------------------------------
    # Weight comparison
    # --------------------------------------------------------
    assert torch.allclose(
        topk_weights_ref,
        topk_weights,
        atol=1e-3,
        rtol=1e-3,
    ), (
        f"Weights mismatch:\n"
        f"torch={topk_weights_ref}\n"
        f"mcop={topk_weights}"
    )

    # --------------------------------------------------------
    # Index comparison
    # --------------------------------------------------------
    assert compare_topk_values(
        gating_output,
        topk_indices_ref.int(),
        topk_indices,
    ), (
        f"Values at the two indices are not equal:\n"
        f"torch={topk_indices_ref}\n"
        f"mcop={topk_indices}"
    )

    print("✅ topk_softmax 精度测试通过。")


@pytest.mark.parametrize(
    "shared_experts,num_experts,num_expert_group,group_topk,topk",
    TOPK_SOFTMAX_SCENARIOS,
)
def test_topk_softmax_renormalize(
    shared_experts,
    num_experts,
    num_expert_group,
    group_topk,
    topk,
):
    """
    测试 renormalize=True。
    """

    num_tokens = 4096
    dtype = torch.bfloat16

    print(
        f"\n=== test_topk_softmax_renormalize ===\n"
        f"shared_experts={shared_experts}, "
        f"num_experts={num_experts}, "
        f"num_expert_group={num_expert_group}, "
        f"group_topk={group_topk}, "
        f"topk={topk}, "
        f"num_tokens={num_tokens}, "
        f"dtype={dtype}"
    )

    assert topk <= num_experts, (
        f"TOP_K={topk} > NUM_EXPERTS={num_experts}"
    )

    gating_output = torch.randn(
        (num_tokens, num_experts),
        dtype=dtype,
        device="cuda",
    )

    topk_weights = torch.empty(
        (num_tokens, topk),
        dtype=torch.float32,
        device="cuda",
    )

    topk_indices = torch.empty(
        (num_tokens, topk),
        dtype=torch.int32,
        device="cuda",
    )

    token_expert_indices = torch.empty(
        (num_tokens, topk),
        dtype=torch.int32,
        device="cuda",
    )

    # --------------------------------------------------------
    # CUDA: renormalize=True
    # --------------------------------------------------------
    topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=True,
        bias=None,
    )

    torch.cuda.synchronize()

    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------
    softmax_output = torch.softmax(
        gating_output.float(),
        dim=-1,
    )

    topk_weights_ref, topk_indices_ref = torch.topk(
        softmax_output,
        topk,
        dim=-1,
        sorted=True,
    )

    topk_weights_ref = (
        topk_weights_ref
        / topk_weights_ref.sum(
            dim=-1,
            keepdim=True,
        )
    )

    # --------------------------------------------------------
    # Weight comparison
    # --------------------------------------------------------
    assert torch.allclose(
        topk_weights_ref,
        topk_weights,
        atol=1e-3,
        rtol=1e-3,
    ), (
        f"Renormalized weights mismatch:\n"
        f"torch={topk_weights_ref}\n"
        f"mcop={topk_weights}"
    )

    # --------------------------------------------------------
    # Index comparison
    # --------------------------------------------------------
    assert compare_topk_values(
        gating_output.float(),
        topk_indices_ref.int(),
        topk_indices,
    )

    # --------------------------------------------------------
    # Verify sum ~= 1
    # --------------------------------------------------------
    weight_sum = topk_weights.sum(
        dim=-1
    )

    assert torch.allclose(
        weight_sum,
        torch.ones_like(weight_sum),
        atol=1e-4,
        rtol=1e-4,
    ), (
        f"Renormalized weight sum != 1:\n"
        f"min={weight_sum.min().item()}\n"
        f"max={weight_sum.max().item()}"
    )

    print(
        "✅ topk_softmax renormalize "
        "测试通过。"
    )


@pytest.mark.parametrize(
    "shared_experts,num_experts,num_expert_group,group_topk,topk",
    TOPK_SOFTMAX_SCENARIOS,
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
def test_topk_softmax_dtype(
    shared_experts,
    num_experts,
    num_expert_group,
    group_topk,
    topk,
    dtype,
):
    """
    测试 FP16 / BF16 / FP32 输入。
    输出统一为 FP32。
    """

    num_tokens = 4096

    print(
        f"\n=== test_topk_softmax_dtype ===\n"
        f"shared_experts={shared_experts}, "
        f"num_experts={num_experts}, "
        f"num_expert_group={num_expert_group}, "
        f"group_topk={group_topk}, "
        f"topk={topk}, "
        f"num_tokens={num_tokens}, "
        f"dtype={dtype}"
    )

    assert topk <= num_experts

    gating_output = torch.randn(
        (num_tokens, num_experts),
        dtype=dtype,
        device="cuda",
    )

    topk_weights = torch.empty(
        (num_tokens, topk),
        dtype=torch.float32,
        device="cuda",
    )

    topk_indices = torch.empty(
        (num_tokens, topk),
        dtype=torch.int32,
        device="cuda",
    )

    token_expert_indices = torch.empty(
        (num_tokens, topk),
        dtype=torch.int32,
        device="cuda",
    )

    topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
        bias=None,
    )

    torch.cuda.synchronize()

    # Reference
    gating_fp32 = gating_output.float()

    softmax_output = torch.softmax(
        gating_fp32,
        dim=-1,
    )

    topk_weights_ref, topk_indices_ref = torch.topk(
        softmax_output,
        topk,
        dim=-1,
        sorted=True,
    )

    assert torch.allclose(
        topk_weights_ref,
        topk_weights,
        atol=1e-3,
        rtol=1e-3,
    ), (
        f"dtype={dtype} weights mismatch:\n"
        f"torch={topk_weights_ref}\n"
        f"mcop={topk_weights}"
    )

    assert compare_topk_values(
        gating_fp32,
        topk_indices_ref.int(),
        topk_indices,
    )

    print(
        f"✅ dtype={dtype} 测试通过。"
    )


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [
                __file__,
                "-s",
                "-v",
            ]
        )
    )