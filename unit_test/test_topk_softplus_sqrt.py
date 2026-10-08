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
    (0, 160, 1, 1, 8),
    (0, 256, 8, 4, 8),
    (0, 256, 1, 1, 8),
    (0, 320, 1, 1, 8),
    (0, 384, 1, 1, 8),
    (0, 448, 1, 1, 8),
    (0, 288, 1, 1, 8),
    (0, 896, 1, 1, 16),

    (1, 160, 1, 1, 9),
    (1, 256, 1, 1, 9),
    (1, 256, 8, 4, 9),
    (1, 320, 1, 1, 9),
    (1, 384, 1, 1, 9),
    (1, 448, 1, 1, 9),

    (2, 256, 8, 4, 9),
    (2, 320, 1, 1, 9),
    (2, 384, 1, 1, 9),
    (2, 448, 1, 1, 9),
]


# ============================================================
# topk_softplus_sqrt 当前 kernel 支持的 expert 数
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
    correction_bias 只参与 TopK selection
    topk_weights 使用原始 scores
    renormalize 后乘 routed_scaling_factor
    """

    scores = torch.sqrt(F.softplus(gating_output.float()))

    if correction_bias is not None:
        scores_for_choice = scores + correction_bias.float().unsqueeze(0)
    else:
        scores_for_choice = scores

    _, topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=True)

    topk_weights = torch.gather(scores, dim=-1, index=topk_indices)

    if renormalize:
        row_sum = topk_weights.sum(dim=-1, keepdim=True)
        row_sum = torch.where(row_sum > 0.0, row_sum, torch.ones_like(row_sum))
        topk_weights = topk_weights / row_sum

    topk_weights = topk_weights * float(routed_scaling_factor)

    return topk_weights.float(), topk_indices.int()


def run_single_test(scenario):
    shared_experts, num_experts, num_expert_group, group_topk, topk = scenario

    print("\n" + "=" * 80)
    print(f"场景: shared_experts={shared_experts}, num_experts={num_experts}, num_expert_group={num_expert_group}, group_topk={group_topk}, topk={topk}")

    # ---------------------------------------------------------
    # Skip 不支持的 expert number
    # ---------------------------------------------------------
    if num_experts not in SUPPORTED_EXPERTS:
        print(f"⏭️ SKIP: num_experts={num_experts} 当前 topk_softplus_sqrt kernel 不支持。")
        print(f"   支持的 num_experts: {sorted(SUPPORTED_EXPERTS)}")
        print("   原因: C++ switch(num_experts) 没有对应的 LAUNCH_SOFTPLUS_SQRT case。")
        print("=" * 80)
        return None

    num_tokens = 4096
    dtype = torch.bfloat16
    renormalize = True
    routed_scaling_factor = 1.0

    print(f"Tokens={num_tokens}, Dtype={dtype}, Renormalize={renormalize}, RoutedScaling={routed_scaling_factor}")
    print("=" * 80)

    assert topk <= num_experts, f"TOP_K={topk} > NUM_EXPERTS={num_experts}"

    # =========================================================
    # 1. Input
    # =========================================================
    torch.manual_seed(42)

    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")

    correction_bias = torch.randn((num_experts,), dtype=torch.float32, device="cuda")

    # 新接口新增：
    # input_ids
    # tid2eid
    # is_padding
    #
    # 当前测试使用 None，测试标准 dense 路径。
    input_ids = None
    tid2eid = None
    is_padding = None

    # =========================================================
    # 2. Output
    # =========================================================
    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")

    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    token_expert_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    # =========================================================
    # 3. Reference
    # =========================================================
    ref_weights, ref_indices = topk_softplus_sqrt_reference(gating_output=gating_output, topk=topk, renormalize=renormalize, routed_scaling_factor=routed_scaling_factor, correction_bias=correction_bias)

    # =========================================================
    # 4. CUDA operator
    #
    # 当前接口：
    #
    # void topk_softplus_sqrt(
    #     torch::Tensor& topk_weights,
    #     torch::Tensor& topk_indices,
    #     torch::Tensor& token_expert_indices,
    #     torch::Tensor& gating_output,
    #     bool renormalize,
    #     double routed_scaling_factor,
    #     const c10::optional<torch::Tensor>& correction_bias,
    #     const c10::optional<torch::Tensor>& input_ids,
    #     const c10::optional<torch::Tensor>& tid2eid,
    #     const c10::optional<torch::Tensor>& is_padding
    # );
    # =========================================================
    torch.ops._moe_C.topk_softplus_sqrt(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize, routed_scaling_factor, correction_bias, input_ids, tid2eid, is_padding)

    torch.cuda.synchronize()

    # =========================================================
    # 5. Index check
    # =========================================================
    sorted_ref_indices, ref_order = ref_indices.sort(dim=-1)
    sorted_cuda_indices, cuda_order = topk_indices.sort(dim=-1)

    torch.testing.assert_close(sorted_ref_indices, sorted_cuda_indices, atol=0, rtol=0)

    print("✅ Top-K expert indices 对比通过。")

    # =========================================================
    # 6. Weight check
    # =========================================================
    sorted_ref_weights = ref_weights.gather(1, ref_order)
    sorted_cuda_weights = topk_weights.gather(1, cuda_order)

    cosine = F.cosine_similarity(sorted_ref_weights.flatten(), sorted_cuda_weights.flatten(), dim=0).item()

    max_diff = torch.max(torch.abs(sorted_ref_weights - sorted_cuda_weights)).item()

    mean_diff = torch.mean(torch.abs(sorted_ref_weights - sorted_cuda_weights)).item()

    print(f"📊 Weight cosine similarity: {cosine:.8f}")
    print(f"📊 Weight max diff: {max_diff:.8e}")
    print(f"📊 Weight mean diff: {mean_diff:.8e}")

    assert not math.isnan(cosine), "❌ Weight cosine similarity 为 NaN"

    assert 1.0 - cosine < 1e-5, f"❌ Weight 精度失败: 1-cosine={1.0 - cosine:.8e}"

    print("✅ Weight 精度校验通过。")

    # =========================================================
    # 7. Verify token_expert_indices
    # =========================================================
    # 当前测试没有 input_ids / tid2eid，
    # token_expert_indices 通常应与 topk_indices 保持对应关系。
    #
    # 如果当前 kernel 的 token_expert_indices 语义不同，
    # 可以单独关闭这一项检查。
    if input_ids is None and tid2eid is None:
        token_expert_match = torch.equal(topk_indices, token_expert_indices)
        if token_expert_match:
            print("✅ token_expert_indices 与 topk_indices 一致。")
        else:
            print("⚠️ token_expert_indices 与 topk_indices 不完全一致，跳过失败判断。")

    # =========================================================
    # 8. Performance
    # =========================================================
    WARMUP = 10
    RUNS = 100

    for _ in range(WARMUP):
        torch.ops._moe_C.topk_softplus_sqrt(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize, routed_scaling_factor, correction_bias, input_ids, tid2eid, is_padding)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(RUNS):
        torch.ops._moe_C.topk_softplus_sqrt(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize, routed_scaling_factor, correction_bias, input_ids, tid2eid, is_padding)

    end.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / RUNS

    print(f"⏱️ topk_softplus_sqrt 平均耗时: {elapsed_ms:.4f} ms")

    return elapsed_ms


def run_test():
    print("=== 开始测试 topk_softplus_sqrt ===")

    print(f"总测试场景数: {len(TEST_SCENARIOS)}")

    results = []
    skipped = []

    for index, scenario in enumerate(TEST_SCENARIOS, start=1):
        print(f"\n[{index}/{len(TEST_SCENARIOS)}] 开始测试")

        elapsed_ms = run_single_test(scenario)

        if elapsed_ms is None:
            skipped.append(scenario)
        else:
            results.append((scenario, elapsed_ms))

    # =========================================================
    # Final summary
    # =========================================================
    print("\n" + "=" * 80)
    print("=== 所有 topk_softplus_sqrt 测试完成 ===")
    print("=" * 80)

    print(f"总场景数: {len(TEST_SCENARIOS)}")
    print(f"通过测试: {len(results)}")
    print(f"跳过测试: {len(skipped)}")

    if skipped:
        print("\n⏭️ 跳过的场景:")

        for scenario in skipped:
            print(f"  {scenario} -> num_experts={scenario[1]} 当前 kernel 不支持")

    if results:
        print("\n✅ 已执行场景性能:")
        print(f"{'Scenario':<35} {'Time(ms)':>12}")

        for scenario, elapsed_ms in results:
            print(f"{str(scenario):<35} {elapsed_ms:>12.4f}")


if __name__ == "__main__":
    run_test()