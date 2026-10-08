# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn.functional as F

import mcoplib._moe_C


# ============================================================
# Test scenarios
#
# (shared_experts, num_experts, num_expert_group, group_topk, topk)
# ============================================================
TEST_SCENARIOS = [
    (0, 160, 1, 1, 8),      # SKIP: num_experts=160 不支持
    (0, 256, 8, 4, 8),      # PASS
    (0, 256, 1, 1, 8),      # PASS
    (0, 320, 1, 1, 8),      # PASS
    (0, 384, 1, 1, 8),      # PASS
    (0, 448, 1, 1, 8),      # PASS
    (0, 288, 1, 1, 8),      # SKIP: num_experts=288 不支持
    (0, 896, 1, 1, 16),     # SKIP: num_experts=896 不支持

    (1, 160, 1, 1, 9),      # SKIP: num_experts=160 不支持
    (1, 256, 1, 1, 9),      # PASS
    (1, 256, 8, 4, 9),      # PASS
    (1, 320, 1, 1, 9),      # PASS
    (1, 384, 1, 1, 9),      # PASS
    (1, 448, 1, 1, 9),      # PASS

    (2, 256, 8, 4, 9),      # PASS
    (2, 320, 1, 1, 9),      # PASS
    (2, 384, 1, 1, 9),      # PASS
    (2, 448, 1, 1, 9),      # PASS
]


# ============================================================
# topk_softplus_sqrt 当前 kernel 明确支持的 expert 数
#
# 来源于 C++:
#
# switch (num_experts) {
#     case 1:
#     case 2:
#     case 4:
#     case 8:
#     case 16:
#     case 32:
#     case 64:
#     case 128:
#     case 256:
#     case 512:
#     case 192:
#     case 320:
#     case 384:
#     case 448:
#     case 576:
# }
#
# 因此 160 / 288 / 896 当前会触发：
#
# RuntimeError: Unsupported expert number: xxx
# ============================================================
SUPPORTED_EXPERTS = {
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    192,
    256,
    320,
    384,
    448,
    512,
    576,
}


def topk_softplus_sqrt_reference(gating_output, topk, renormalize, routed_scaling_factor, correction_bias=None):
    """
    PyTorch reference implementation of topk_softplus_sqrt.

    scores = sqrt(softplus(gating_output))
    bias 只参与 TopK selection
    topk_weights 使用原始 scores
    renormalize 后乘 routed_scaling_factor
    """

    scores = torch.sqrt(
        F.softplus(gating_output.float())
    )

    if correction_bias is not None:
        scores_for_choice = (
            scores
            + correction_bias.float().unsqueeze(0)
        )
    else:
        scores_for_choice = scores

    _, topk_indices = torch.topk(
        scores_for_choice,
        k=topk,
        dim=-1,
        sorted=True,
    )

    topk_weights = torch.gather(
        scores,
        dim=-1,
        index=topk_indices,
    )

    if renormalize:
        row_sum = topk_weights.sum(
            dim=-1,
            keepdim=True,
        )

        row_sum = torch.where(
            row_sum > 0.0,
            row_sum,
            torch.ones_like(row_sum),
        )

        topk_weights = (
            topk_weights / row_sum
        )

    topk_weights = (
        topk_weights
        * float(routed_scaling_factor)
    )

    return (
        topk_weights.float(),
        topk_indices.int(),
    )


def run_single_test(scenario):
    shared_experts, num_experts, num_expert_group, group_topk, topk = scenario

    print("\n" + "=" * 80)
    print(
        f"场景: shared_experts={shared_experts}, "
        f"num_experts={num_experts}, "
        f"num_expert_group={num_expert_group}, "
        f"group_topk={group_topk}, "
        f"topk={topk}"
    )

    # ---------------------------------------------------------
    # Skip 不支持的 expert number
    # ---------------------------------------------------------
    if num_experts not in SUPPORTED_EXPERTS:
        print(
            f"⏭️ SKIP: num_experts={num_experts} "
            f"当前 topk_softplus_sqrt kernel 不支持。"
        )
        print(
            f"   支持的 num_experts: "
            f"{sorted(SUPPORTED_EXPERTS)}"
        )
        print(
            "   原因: C++ switch(num_experts) "
            "没有对应的 LAUNCH_SOFTPLUS_SQRT case。"
        )
        print("=" * 80)
        return None

    num_tokens = 4096
    dtype = torch.bfloat16
    renormalize = True
    routed_scaling_factor = 1.0

    print(
        f"Tokens={num_tokens}, "
        f"Dtype={dtype}, "
        f"Renormalize={renormalize}, "
        f"RoutedScaling={routed_scaling_factor}"
    )
    print("=" * 80)

    assert topk <= num_experts, (
        f"TOP_K={topk} > NUM_EXPERTS={num_experts}"
    )

    # =========================================================
    # 1. Input
    # =========================================================
    torch.manual_seed(42)

    gating_output = torch.randn(
        (num_tokens, num_experts),
        dtype=dtype,
        device="cuda",
    )

    correction_bias = torch.randn(
        (num_experts,),
        dtype=torch.float32,
        device="cuda",
    )

    # =========================================================
    # 2. Output
    # =========================================================
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

    # =========================================================
    # 3. Reference
    # =========================================================
    ref_weights, ref_indices = topk_softplus_sqrt_reference(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        correction_bias=correction_bias,
    )

    # =========================================================
    # 4. CUDA operator
    # =========================================================
    torch.ops._moe_C.topk_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        correction_bias,
        None,
        None,
    )

    torch.cuda.synchronize()

    # =========================================================
    # 5. Index check
    # =========================================================
    sorted_ref_indices, ref_order = ref_indices.sort(
        dim=-1
    )

    sorted_cuda_indices, cuda_order = topk_indices.sort(
        dim=-1
    )

    torch.testing.assert_close(
        sorted_ref_indices,
        sorted_cuda_indices,
        atol=0,
        rtol=0,
    )

    print("✅ Top-K expert indices 对比通过。")

    # =========================================================
    # 6. Weight check
    # =========================================================
    sorted_ref_weights = ref_weights.gather(
        1,
        ref_order,
    )

    sorted_cuda_weights = topk_weights.gather(
        1,
        cuda_order,
    )

    cosine = F.cosine_similarity(
        sorted_ref_weights.flatten(),
        sorted_cuda_weights.flatten(),
        dim=0,
    ).item()

    max_diff = torch.max(
        torch.abs(
            sorted_ref_weights
            - sorted_cuda_weights
        )
    ).item()

    mean_diff = torch.mean(
        torch.abs(
            sorted_ref_weights
            - sorted_cuda_weights
        )
    ).item()

    print(
        f"📊 Weight cosine similarity: "
        f"{cosine:.8f}"
    )

    print(
        f"📊 Weight max diff: "
        f"{max_diff:.8e}"
    )

    print(
        f"📊 Weight mean diff: "
        f"{mean_diff:.8e}"
    )

    assert not math.isnan(
        cosine
    ), "❌ Weight cosine similarity 为 NaN"

    assert 1.0 - cosine < 1e-5, (
        f"❌ Weight 精度失败: "
        f"1-cosine={1.0 - cosine:.8e}"
    )

    print("✅ Weight 精度校验通过。")

    # =========================================================
    # 7. Performance
    # =========================================================
    WARMUP = 10
    RUNS = 100

    for _ in range(WARMUP):
        torch.ops._moe_C.topk_softplus_sqrt(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            routed_scaling_factor,
            correction_bias,
            None,
            None,
        )

    torch.cuda.synchronize()

    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )

    start.record()

    for _ in range(RUNS):
        torch.ops._moe_C.topk_softplus_sqrt(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            routed_scaling_factor,
            correction_bias,
            None,
            None,
        )

    end.record()

    torch.cuda.synchronize()

    elapsed_ms = (
        start.elapsed_time(end)
        / RUNS
    )

    print(
        f"⏱️ topk_softplus_sqrt 平均耗时: "
        f"{elapsed_ms:.4f} ms"
    )

    return elapsed_ms


def run_test():
    print(
        "=== 开始测试 topk_softplus_sqrt ==="
    )

    print(
        f"总测试场景数: "
        f"{len(TEST_SCENARIOS)}"
    )

    results = []
    skipped = []

    for index, scenario in enumerate(
        TEST_SCENARIOS,
        start=1,
    ):
        print(
            f"\n[{index}/{len(TEST_SCENARIOS)}] "
            f"开始测试"
        )

        elapsed_ms = run_single_test(
            scenario
        )

        if elapsed_ms is None:
            skipped.append(
                scenario
            )
        else:
            results.append(
                (
                    scenario,
                    elapsed_ms,
                )
            )

    # =========================================================
    # Final summary
    # =========================================================
    print("\n" + "=" * 80)
    print(
        "=== 所有 topk_softplus_sqrt 测试完成 ==="
    )
    print("=" * 80)

    print(
        f"总场景数: {len(TEST_SCENARIOS)}"
    )

    print(
        f"通过测试: {len(results)}"
    )

    print(
        f"跳过测试: {len(skipped)}"
    )

    if skipped:
        print("\n⏭️ 跳过的场景:")

        for scenario in skipped:
            print(
                f"  {scenario} "
                f"-> num_experts={scenario[1]} "
                f"当前 kernel 不支持"
            )

    if results:
        print("\n✅ 已执行场景性能:")

        print(
            f"{'Scenario':<35} "
            f"{'Time(ms)':>12}"
        )

        for scenario, elapsed_ms in results:
            print(
                f"{str(scenario):<35} "
                f"{elapsed_ms:>12.4f}"
            )


if __name__ == "__main__":
    run_test()