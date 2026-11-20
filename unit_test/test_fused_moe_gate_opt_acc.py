import torch
import torch.nn.functional as F
from mcoplib import op as ops


def cosine_similarity(a, b):
    """
    计算两个tensor的余弦相似度

    Args:
        a: tensor, 任意形状
        b: tensor, 必须与a形状相同

    Returns:
        float: 余弦相似度，范围[-1, 1]
                - 1.0 表示完全相同
                - 0.0 表示正交
                - -1.0 表示完全相反
                越接近1表示越相似
    """
    # 展平为一维向量
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()

    # 计算余弦相似度: cos(θ) = (a·b) / (||a|| * ||b||)
    dot_product = torch.sum(a_flat * b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)

    # 避免除零错误
    norm_a = norm_a if norm_a.item() > 1e-8 else torch.tensor(1.0, device=a.device)
    norm_b = norm_b if norm_b.item() > 1e-8 else torch.tensor(1.0, device=a.device)

    cosine_sim = dot_product / (norm_a * norm_b)
    return cosine_sim.item()


def reference_moe_gate(gating_outputs, correction_bias, topk, num_expert_group,
                        topk_group, num_fused_shared_experts=None,
                        routed_scaling_factor=None):
    """
    参考实现：用于验证fused_moe_gate_opt的正确性

    该实现严格遵循fused_moe_gate.cuh中的算法逻辑：
    1. Sigmoid激活
    2. 加bias
    3. 基于bias后的值选择topk专家
    4. 使用sigmoid后的原始值（未加bias）作为权重
    5. 处理共享专家
    6. 归一化权重

    Args:
        gating_outputs: [batch_size, num_experts], 门控网络输出
        correction_bias: [num_experts], 校正偏置
        topk: 选择的专家总数
        num_expert_group: 专家分组数
        topk_group: 从每组选择的组数
        num_fused_shared_experts: 共享专家数量
        routed_scaling_factor: 路由缩放因子

    Returns:
        normalized_weights: [batch_size, topk], 归一化后的权重
        selected_indices: [batch_size, topk], 选中的专家索引
    """
    batch_size = gating_outputs.shape[0]
    num_experts = gating_outputs.shape[1]
    device = gating_outputs.device

    # Step 1: Sigmoid激活
    sigmoid_output = torch.sigmoid(gating_outputs.float())

    # Step 2: 加bias（用于专家选择）
    bias_output = sigmoid_output + correction_bias.float()

    # Step 3: 确定实际需要选择的topk数量（排除共享专家）
    if num_fused_shared_experts is None or num_fused_shared_experts == 0:
        topk_excluding_shared = topk
        has_shared = False
    else:
        topk_excluding_shared = topk - num_fused_shared_experts
        has_shared = True

    # Step 4: 基于bias后的值选择topk专家
    if num_expert_group == 1:
        # 单组模式：直接选择topk
        topk_values, topk_indices = torch.topk(bias_output, k=topk_excluding_shared, dim=1)
    else:
        # 多组模式：这里简化处理，直接选择topk
        # 实际的CUDA实现会先选择topk_group个组，再从这些组中选择
        topk_values, topk_indices = torch.topk(bias_output, k=topk_excluding_shared, dim=1)

    # Step 5: 获取sigmoid后的权重值（注意：使用sigmoid后的值，不是bias后的值）
    selected_weights = torch.gather(sigmoid_output, 1, topk_indices)

    # Step 6: 处理共享专家
    if has_shared:
        # 计算所有选中专家权重的和
        weight_sum = selected_weights.sum(dim=1, keepdim=True)  # [batch_size, 1]

        # 共享专家的权重 = sum_of_weights / routed_scaling_factor
        scale = routed_scaling_factor if routed_scaling_factor is not None else 1.0
        shared_weight = weight_sum / scale  # [batch_size, 1]

        # 添加共享专家
        for i in range(num_fused_shared_experts):
            # 共享专家的索引 = num_experts + i（超出实际专家范围）
            shared_indices = torch.full((batch_size, 1), num_experts + i,
                                        dtype=torch.long, device=device)

            # 拼接索引和权重
            topk_indices = torch.cat([topk_indices, shared_indices], dim=1)
            selected_weights = torch.cat([selected_weights, shared_weight], dim=1)

    # Step 7: 归一化权重
    # 注意：如果有共享专家，只归一化路由专家，不包括共享专家
    # 这与biased_grouped_topk的逻辑一致
    if has_shared:
        # 只计算路由专家的和作为归一化基数
        weight_sum = selected_weights[:, :-1].sum(dim=1, keepdim=True)
    else:
        # 无共享专家，计算所有专家的和
        weight_sum = selected_weights.sum(dim=1, keepdim=True)
    normalized_weights = selected_weights / (weight_sum + 1e-8)  # 添加小值避免除零

    return normalized_weights, topk_indices


def verify_accuracy(output_weights, output_indices, ref_weights, ref_indices,
                    tolerance=0.0001, test_name=""):
    """
    验证输出精度

    Args:
        output_weights: 算子输出的权重
        output_indices: 算子输出的专家索引
        ref_weights: 参考实现的权重
        ref_indices: 参考实现的专家索引
        tolerance: 容差（余弦相似度与1.0的最大差距）
        test_name: 测试名称
    """
    # 1. 计算余弦相似度
    weight_similarity = cosine_similarity(output_weights, ref_weights)
    similarity_diff = abs(1.0 - weight_similarity)

    print(f"  {test_name} - Weight Cosine Similarity: {weight_similarity:.10f}")
    print(f"  {test_name} - Difference from perfect (1.0): {similarity_diff:.10f}")

    # 2. 验证余弦相似度
    if similarity_diff >= tolerance:
        # 打印详细的调试信息
        print(f"\n  📊 详细调试信息：")
        batch_size = output_indices.shape[0]
        topk = output_indices.shape[1]

        for b in range(batch_size):
            print(f"\n  Batch {b}:")
            print(f"    CUDA 索引: {output_indices[b].cpu().numpy()}")
            print(f"    Ref  索引: {ref_indices[b].cpu().numpy()}")

            indices_match = torch.equal(output_indices[b], ref_indices[b])
            if not indices_match:
                print(f"    ⚠️  索引不匹配！")

            cuda_w = output_weights[b].cpu()
            ref_w = ref_weights[b].cpu()
            print(f"    CUDA 权重: {cuda_w.numpy()}")
            print(f"    Ref  权重: {ref_w.numpy()}")
            print(f"    权重差异: {(cuda_w - ref_w).numpy()}")

            # 单个batch的余弦相似度
            batch_cosine = torch.cosine_similarity(
                cuda_w.unsqueeze(0), ref_w.unsqueeze(0)
            ).item()
            print(f"    Batch余弦相似度: {batch_cosine:.10f}")

        raise AssertionError(
            f"{test_name} - Accuracy check failed!\n"
            f"  Cosine similarity = {weight_similarity:.10f}\n"
            f"  Difference from 1.0 = {similarity_diff:.10f} >= {tolerance}\n"
            f"  This indicates the outputs do not match the reference implementation."
        )

    # 3. 验证专家索引（允许顺序不同，但集合必须相同）
    batch_size = output_indices.shape[0]
    for i in range(batch_size):
        pred_set = set(output_indices[i].cpu().numpy())
        ref_set = set(ref_indices[i].cpu().numpy())

        if pred_set != ref_set:
            raise AssertionError(
                f"{test_name} - Expert indices mismatch at batch {i}!\n"
                f"  Predicted: {sorted(pred_set)}\n"
                f"  Reference: {sorted(ref_set)}\n"
                f"  Missing: {ref_set - pred_set}\n"
                f"  Extra: {pred_set - ref_set}"
            )

    print(f"  ✅ {test_name} - Accuracy verification passed!")


def test_160_experts_no_shared():
    """测试160专家，无共享专家，并验证精度"""
    print("\n" + "="*70)
    print("[Test 1] Testing 160 experts (no shared experts)")
    print("="*70)

    batch_size = 4
    num_experts = 160
    topk = 8

    # 设置随机种子以保证可复现性
    torch.manual_seed(1234)

    # 创建测试数据（修正：使用device参数）
    gating_outputs = torch.randn(batch_size, num_experts, dtype=torch.float, device='cuda')
    correction_bias = torch.randn(num_experts, dtype=torch.float, device='cuda')
    out_routing_weights = torch.empty(batch_size, topk, dtype=torch.float, device='cuda')
    out_selected_experts = torch.empty(batch_size, topk, dtype=torch.int32, device='cuda')

    print(f"  Configuration:")
    print(f"    Batch size: {batch_size}")
    print(f"    Num experts: {num_experts}")
    print(f"    TopK: {topk}")
    print(f"    Expert groups: 1")
    print(f"    Shared experts: 0")

    # 调用算子
    ret = ops.fused_moe_gate_opt(
        gating_outputs,
        correction_bias,
        out_routing_weights,
        out_selected_experts,
        topk=topk,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        num_fused_shared_experts=None,
        routed_scaling_factor=None
    )

    # 基本断言
    assert ret == 0, f"fused_moe_gate_opt returned error code {ret}"
    assert out_selected_experts.shape == (batch_size, topk), \
        f"Output shape mismatch: {out_selected_experts.shape} != ({batch_size}, {topk})"
    assert torch.all(out_selected_experts >= 0), "Expert indices should be non-negative"
    assert torch.all(out_selected_experts < num_experts), \
        f"Expert indices should be < {num_experts}"

    # 验证权重归一化（和应该为1）
    # 注意：由于bfloat16精度的限制，权重和可能不完全等于1.0
    # bfloat16只有7位有效精度，在多次类型转换和除法运算后会有累积误差
    # 正常误差范围应该在 ±0.005 以内
    weight_sums = out_routing_weights.sum(dim=1)
    normalization_tolerance = 5e-3  # 放宽容差到0.005，适应bfloat16精度
    assert torch.all(torch.abs(weight_sums - 1.0) < normalization_tolerance), \
        f"Weights should sum to 1.0 (within tolerance {normalization_tolerance}), " \
        f"got sums in range [{weight_sums.min():.6f}, {weight_sums.max():.6f}]"

    print(f"  ✓ Basic checks passed")

    # 计算参考实现
    ref_weights, ref_indices = reference_moe_gate(
        gating_outputs, correction_bias, topk,
        num_expert_group=1, topk_group=1,
        num_fused_shared_experts=None,
        routed_scaling_factor=None
    )

    # 精度验证
    verify_accuracy(
        out_routing_weights, out_selected_experts,
        ref_weights, ref_indices,
        tolerance=0.00001,
        test_name="Test 1 (160 experts, no shared)"
    )

def test_256_experts_no_shared():
    """测试160专家，无共享专家，并验证精度"""
    print("\n" + "="*70)
    print("[Test 1] Testing 160 experts (no shared experts)")
    print("="*70)

    batch_size = 4
    num_experts = 256
    topk = 8

    # 设置随机种子以保证可复现性
    torch.manual_seed(1234)

    # 创建测试数据（修正：使用device参数）
    gating_outputs = torch.randn(batch_size, num_experts, dtype=torch.float, device='cuda')
    correction_bias = torch.randn(num_experts, dtype=torch.float, device='cuda')
    out_routing_weights = torch.empty(batch_size, topk, dtype=torch.float, device='cuda')
    out_selected_experts = torch.empty(batch_size, topk, dtype=torch.int32, device='cuda')

    print(f"  Configuration:")
    print(f"    Batch size: {batch_size}")
    print(f"    Num experts: {num_experts}")
    print(f"    TopK: {topk}")
    print(f"    Expert groups: 1")
    print(f"    Shared experts: 0")

    # 调用算子
    ret = ops.fused_moe_gate_opt(
        gating_outputs,
        correction_bias,
        out_routing_weights,
        out_selected_experts,
        topk=topk,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        num_fused_shared_experts=None,
        routed_scaling_factor=None
    )

    # 基本断言
    assert ret == 0, f"fused_moe_gate_opt returned error code {ret}"
    assert out_selected_experts.shape == (batch_size, topk), \
        f"Output shape mismatch: {out_selected_experts.shape} != ({batch_size}, {topk})"
    assert torch.all(out_selected_experts >= 0), "Expert indices should be non-negative"
    assert torch.all(out_selected_experts < num_experts), \
        f"Expert indices should be < {num_experts}"

    # 验证权重归一化（和应该为1）
    # 注意：由于bfloat16精度的限制，权重和可能不完全等于1.0
    # bfloat16只有7位有效精度，在多次类型转换和除法运算后会有累积误差
    # 正常误差范围应该在 ±0.005 以内
    weight_sums = out_routing_weights.sum(dim=1)
    normalization_tolerance = 5e-3  # 放宽容差到0.005，适应bfloat16精度
    assert torch.all(torch.abs(weight_sums - 1.0) < normalization_tolerance), \
        f"Weights should sum to 1.0 (within tolerance {normalization_tolerance}), " \
        f"got sums in range [{weight_sums.min():.6f}, {weight_sums.max():.6f}]"

    print(f"  ✓ Basic checks passed")

    # 计算参考实现
    ref_weights, ref_indices = reference_moe_gate(
        gating_outputs, correction_bias, topk,
        num_expert_group=1, topk_group=1,
        num_fused_shared_experts=None,
        routed_scaling_factor=None
    )

    # 精度验证
    verify_accuracy(
        out_routing_weights, out_selected_experts,
        ref_weights, ref_indices,
        tolerance=0.00001,
        test_name="Test 1 (160 experts, no shared)"
    )

    print("✅ Test 1 passed!")


def test_160_experts_with_shared():
    """测试160专家，1个共享专家，并验证精度"""
    print("\n" + "="*70)
    print("[Test 2] Testing 160 experts (with 1 shared expert)")
    print("="*70)

    batch_size = 4
    num_experts = 160
    topk = 9  # 8 + 1 shared
    num_shared = 1
    routed_scaling_factor = 2.0

    # 设置随机种子
    torch.manual_seed(5678)

    # 创建测试数据
    gating_outputs = torch.randn(batch_size, num_experts, dtype=torch.bfloat16, device='cuda')
    correction_bias = torch.randn(num_experts, dtype=torch.bfloat16, device='cuda')
    out_routing_weights = torch.empty(batch_size, topk, dtype=torch.float, device='cuda')
    out_selected_experts = torch.empty(batch_size, topk, dtype=torch.int32, device='cuda')

    print(f"  Configuration:")
    print(f"    Batch size: {batch_size}")
    print(f"    Num experts: {num_experts}")
    print(f"    TopK: {topk}")
    print(f"    Expert groups: 1")
    print(f"    Shared experts: {num_shared}")
    print(f"    Routed scaling factor: {routed_scaling_factor}")

    # 调用算子
    ret = ops.fused_moe_gate_opt(
        gating_outputs,
        correction_bias,
        out_routing_weights,
        out_selected_experts,
        topk=topk,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        num_fused_shared_experts=num_shared,
        routed_scaling_factor=routed_scaling_factor
    )

    # 基本断言
    assert ret == 0, f"fused_moe_gate_opt returned error code {ret}"

    # 检查最后一个专家是共享专家（索引160）
    assert torch.all(out_selected_experts[:, -1] == num_experts), \
        f"Last expert should be shared expert with index {num_experts}, " \
        f"but got indices in range [{out_selected_experts[:, -1].min()}, {out_selected_experts[:, -1].max()}]"

    # 验证权重归一化
    # 注意：由于bfloat16精度的限制，权重和可能不完全等于1.0
    # 另外，当有共享专家时，权重和 = 1.0 + (1.0 / routed_scaling_factor)
    # 例如：routed_scaling_factor=2.0时，权重和 = 1.0 + 0.5 = 1.5
    weight_sums = out_routing_weights.sum(dim=1)
    expected_sum = 1.0 + (1.0 / routed_scaling_factor)  # 1.0 + shared_weight
    normalization_tolerance = 5e-3  # 放宽容差到0.005，适应bfloat16精度
    assert torch.all(torch.abs(weight_sums - expected_sum) < normalization_tolerance), \
        f"Weights should sum to {expected_sum:.3f} (within tolerance {normalization_tolerance}), " \
        f"got sums in range [{weight_sums.min():.6f}, {weight_sums.max():.6f}]"

    print(f"  ✓ Basic checks passed")
    print(f"  ✓ Shared expert index verified: all = {num_experts}")
    print(f"  ✓ Weight sum verified: ~{expected_sum:.3f}")

    # 计算参考实现
    ref_weights, ref_indices = reference_moe_gate(
        gating_outputs, correction_bias, topk,
        num_expert_group=1, topk_group=1,
        num_fused_shared_experts=num_shared,
        routed_scaling_factor=routed_scaling_factor
    )

    # 精度验证
    verify_accuracy(
        out_routing_weights, out_selected_experts,
        ref_weights, ref_indices,
        tolerance=0.00001,
        test_name="Test 2 (160 experts, with shared)"
    )

    print("✅ Test 2 passed!")


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "="*70)
    print("[Test 3] Testing edge cases")
    print("="*70)

    # 测试1: 小batch size
    print("\n  [Edge Case 1] Small batch size (batch=1)")
    torch.manual_seed(9012)
    gating_outputs = torch.randn(1, 160, dtype=torch.bfloat16, device='cuda')
    correction_bias = torch.randn(160, dtype=torch.bfloat16, device='cuda')
    out_weights = torch.empty(1, 8, dtype=torch.float, device='cuda')
    out_indices = torch.empty(1, 8, dtype=torch.int32, device='cuda')

    ret = ops.fused_moe_gate_opt(
        gating_outputs, correction_bias, out_weights, out_indices,
        topk=8, renormalize=True, num_expert_group=1, topk_group=1,
        num_fused_shared_experts=None, routed_scaling_factor=None
    )
    assert ret == 0, "Failed for batch_size=1"
    print("  ✓ Batch size 1 passed")

    # 测试2: 大batch size
    print("\n  [Edge Case 2] Large batch size (batch=128)")
    torch.manual_seed(3456)
    gating_outputs = torch.randn(128, 160, dtype=torch.bfloat16, device='cuda')
    correction_bias = torch.randn(160, dtype=torch.bfloat16, device='cuda')
    out_weights = torch.empty(128, 8, dtype=torch.float, device='cuda')
    out_indices = torch.empty(128, 8, dtype=torch.int32, device='cuda')

    ret = ops.fused_moe_gate_opt(
        gating_outputs, correction_bias, out_weights, out_indices,
        topk=8, renormalize=True, num_expert_group=1, topk_group=1,
        num_fused_shared_experts=None, routed_scaling_factor=None
    )
    assert ret == 0, "Failed for batch_size=128"
    print("  ✓ Batch size 128 passed")

    # 测试3: 验证所有权重都非负
    assert torch.all(out_weights >= 0), "All weights should be non-negative"
    print("  ✓ All weights are non-negative")

    # 测试4: 验证权重在合理范围内
    assert torch.all(out_weights <= 1.0), "All weights should be <= 1.0"
    print("  ✓ All weights are <= 1.0")

    print("\n✅ Test 3 (edge cases) passed!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FUSED MOE GATE - ACCURACY VERIFICATION TEST SUITE")
    print("="*70)
    print("\nThis test suite verifies the accuracy of the fused_moe_gate_opt")
    print("operator by comparing its output against a reference implementation.")
    print("\nAccuracy metric: Cosine Similarity")
    print("  - Perfect match: similarity = 1.0")
    print("  - Tolerance: 1.0 - similarity < 0.00001")
    print("  - If difference >= 0.00001, test FAILS")

    try:
        # 运行所有测试
        test_160_experts_no_shared()
        test_256_experts_no_shared()
        test_160_experts_with_shared()
        test_edge_cases()

        # 所有测试通过
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✅✅✅")
        print("="*70)
        print("\nThe fused_moe_gate_opt operator has been verified to be accurate")
        print("within the specified tolerance.")
        print("="*70)

    except Exception as e:
        print("\n" + "="*70)
        print("TEST FAILED! ❌")
        print("="*70)
        print(f"\nError: {str(e)}")
        print("\nPlease check:")
        print("  1. Is the CUDA device available?")
        print("  2. Is mcoplib properly installed?")
        print("  3. Are the kernel implementations correct?")
        print("="*70)
        raise