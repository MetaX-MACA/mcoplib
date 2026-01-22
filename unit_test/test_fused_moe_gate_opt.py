"""
Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""

import os
import sys
import torch
import unittest
from typing import Callable, NamedTuple, Optional
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
from mcoplib import op as ops
from measure_cuda import measure_cuda

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

def verify_accuracy(output_weights,  ref_weights, tolerance=0.001, test_name=""):
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
        raise AssertionError(
            f"{test_name} - Accuracy check failed!\n"
            f"  Cosine similarity = {weight_similarity:.10f}\n"
            f"  Difference from 1.0 = {similarity_diff:.10f} >= {tolerance}\n"
            f"  This indicates the outputs do not match the reference implementation."
        )
    else:
        print(f"  ✅ {test_name} - Accuracy verification passed!")

def biased_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False) #in sglang, sorted=num_fused_shared_experts > 0
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    # topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    # _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids

def show_error(golden, v, tag="DIFF ERROR"):
    errors = torch.abs(golden - v)

    errors_max = torch.max(errors)
    errors_ave = torch.sum(errors) / errors.numel()

    max_idx_flat = torch.argmax(errors)
    max_idx = torch.unravel_index(max_idx_flat, errors.shape)

    golden_val = golden[max_idx]
    v_val = v[max_idx]

    print(f"{tag}: error_max={errors_max}, error_ave={errors_ave}, max_error_idx={max_idx}")

def moe_gate_func(q_len, num_experts, topk, num_expert_group, top_k_group, renormalize=True, num_shared_experts=1, test_dtype=torch.bfloat16, scale_factor=1.0, test_name=""):
    gating_output = torch.rand(q_len, num_experts, dtype=test_dtype).cuda()
    correction_bias = torch.rand(num_experts, dtype=test_dtype).cuda()
    out_routing_weights = torch.zeros(q_len, topk, dtype=torch.float).cuda()
    out_selected_experts = torch.zeros(q_len, topk, dtype=torch.int32).cuda()
    device = 0
    torch.cuda.set_device(device)
    golden_routing_weights, golden_selected_experts = biased_grouped_topk(gating_output, gating_output, correction_bias, topk, renormalize, num_expert_group, top_k_group, num_shared_experts, scale_factor)

    def kernel():
        ops.fused_moe_gate_opt(gating_output, correction_bias, out_routing_weights, out_selected_experts, topk, renormalize, num_expert_group, top_k_group, num_shared_experts, scale_factor)
        #torch.ops.sgl_kernel.fused_moe_gate_opt(gating_output, correction_bias, out_routing_weights, out_selected_experts, topk, renormalize, num_expert_group, top_k_group, num_shared_experts, scale_factor)
    stats = measure_cuda(kernel, iters=200, warmup=20, device=device)
    print("PyTorch matmul stats (microseconds):")
    print(f"mean={stats['mean_us']:.3f}µs median={stats['median_us']:.3f}µs min={stats['min_us']:.3f}µs stdev={stats['stdev_us']:.3f}µs")
    routing_weights = out_routing_weights
    selected_experts = out_selected_experts
    sorted_golden_routing = torch.sort(golden_routing_weights, dim=1).values  # 按行排序
    sorted_routing = torch.sort(routing_weights, dim=1).values
    #show_error(sorted_golden_routing, sorted_routing, "DIFF ERROR OF SORTED ROUTING_WEIGHTS")
    #assert torch.allclose(sorted_golden_routing, sorted_routing, rtol=1e-03, atol=1e-03, equal_nan=False)
    verify_accuracy(sorted_golden_routing, sorted_routing, 0.0001, test_name)
    sorted_golden_experts = torch.sort(golden_selected_experts, dim=1).values  # 按行排序
    sorted_selected_experts = torch.sort(selected_experts, dim=1).values
    #show_error(sorted_golden_experts, sorted_selected_experts, "DIFF ERROR OF SORTED SELECTED_EXPERTS")
    assert torch.equal(sorted_golden_experts, sorted_selected_experts)

def moe_gate_func_more_share_experts(q_len, num_experts, topk, num_expert_group, top_k_group, renormalize=True, num_shared_experts=2, test_dtype=torch.bfloat16, scale_factor=1.0):
    gating_output = torch.rand(q_len, num_experts, dtype=test_dtype).cuda()
    correction_bias = torch.rand(num_experts, dtype=test_dtype).cuda()
    out_routing_weights = torch.zeros(q_len, topk, dtype=torch.float).cuda()
    out_selected_experts = torch.zeros(q_len, topk, dtype=torch.int32).cuda()

    golden_routing_weights, golden_selected_experts = biased_grouped_topk(gating_output, gating_output, correction_bias, topk, renormalize, num_expert_group, top_k_group, num_shared_experts, scale_factor)
    #torch.ops.sgl_kernel.fused_moe_gate_opt(gating_output, correction_bias, out_routing_weights, out_selected_experts, topk, renormalize, num_expert_group, top_k_group, num_shared_experts, scale_factor)
    ops.fused_moe_gate_opt(gating_output, correction_bias, out_routing_weights, out_selected_experts, topk, renormalize, num_expert_group, top_k_group, num_shared_experts, scale_factor)
    routing_weights = out_routing_weights
    selected_experts = out_selected_experts
    sorted_golden_routing = torch.sort(golden_routing_weights, dim=1).values  # 按行排序
    sorted_routing = torch.sort(routing_weights, dim=1).values
    #show_error(sorted_golden_routing, sorted_routing, "DIFF ERROR OF SORTED ROUTING_WEIGHTS")
    assert torch.allclose(sorted_golden_routing, sorted_routing, rtol=1e-03, atol=1e-03, equal_nan=False)
    sorted_golden_experts = torch.sort(golden_selected_experts, dim=1).values  # 按行排序
    sorted_selected_experts = torch.sort(selected_experts, dim=1).values
    #show_error(sorted_golden_experts, sorted_selected_experts, "DIFF ERROR OF SORTED SELECTED_EXPERTS")
    assert torch.equal(sorted_golden_experts[:, :-1], sorted_selected_experts[:, :-1])

class TestMoeGate(unittest.TestCase):
    def test_moe_gate_256_experts_bfloat16_t_0(self):
        moe_gate_func(q_len=16, num_experts=256, topk=8, num_expert_group=8, top_k_group=4, renormalize=True, num_shared_experts=0, test_dtype=torch.bfloat16, scale_factor=1.0, test_name="test_moe_gate_256_experts_bfloat16_t_0")

    def test_moe_gate_160_experts_bfloat16_f_0(self):
        moe_gate_func(q_len=16, num_experts=160, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=0, test_dtype=torch.bfloat16, scale_factor=1.0, test_name="test_moe_gate_160_experts_bfloat16_f_0")
    
    def test_moe_gate_160_experts_bfloat16_f_1(self):
        moe_gate_func(q_len=16, num_experts=160, topk=9, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=1, test_dtype=torch.bfloat16, scale_factor=1.0, test_name="test_moe_gate_160_experts_bfloat16_f_1")

    # def test_moe_gate_256_experts_bfloat16_t_1(self):
    #     moe_gate_func(q_len=16, num_experts=256, topk=9, num_expert_group=8, top_k_group=4, renormalize=True, num_shared_experts=1, test_dtype=torch.bfloat16, scale_factor=1.0)

    # def test_moe_gate_256_experts_bfloat16_f_1(self):
    #     moe_gate_func(q_len=16, num_experts=256, topk=9, num_expert_group=8, top_k_group=4, renormalize=True, num_shared_experts=1, test_dtype=torch.bfloat16, scale_factor=1.0)

    # def test_moe_gate_256_experts_float16_t_0(self):
    #     moe_gate_func(q_len=16, num_experts=256, topk=8, num_expert_group=8, top_k_group=4, renormalize=True, num_shared_experts=0, test_dtype=torch.float16, scale_factor=1.0)

    # def test_moe_gate_256_experts_float16_f_0(self):
    #     moe_gate_func(q_len=16, num_experts=256, topk=8, num_expert_group=8, top_k_group=4, renormalize=True, num_shared_experts=0, test_dtype=torch.float16, scale_factor=1.0)

    # def test_moe_gate_256_experts_float16_t_1(self):
    #     moe_gate_func(q_len=16, num_experts=256, topk=9, num_expert_group=8, top_k_group=4, renormalize=True, num_shared_experts=1, test_dtype=torch.float16, scale_factor=1.0)

    # def test_moe_gate_256_experts_float16_f_1(self):
    #     moe_gate_func(q_len=16, num_experts=256, topk=9, num_expert_group=8, top_k_group=4, renormalize=True, num_shared_experts=1, test_dtype=torch.float16, scale_factor=1.0)

    # def test_moe_gate_384_experts_bfloat16_t_0(self):
    #     moe_gate_func(q_len=16, num_experts=384, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=0, test_dtype=torch.bfloat16, scale_factor=1.0)

    # def test_moe_gate_384_experts_bfloat16_f_0(self):
    #     moe_gate_func(q_len=16, num_experts=384, topk=8, num_expert_group=1, top_k_group=1, renormalize=False, num_shared_experts=0, test_dtype=torch.bfloat16, scale_factor=1.0)

    # def test_moe_gate_384_experts_bfloat16_t_1(self):
    #     moe_gate_func(q_len=16, num_experts=384, topk=9, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=1, test_dtype=torch.bfloat16, scale_factor=1.0)

    # def test_moe_gate_384_experts_bfloat16_f_1(self): #this case will failed, max weights diff 1e-2
    #     moe_gate_func(q_len=16, num_experts=384, topk=9, num_expert_group=1, top_k_group=1, renormalize=False, num_shared_experts=1, test_dtype=torch.bfloat16, scale_factor=1.0)

    # def test_moe_gate_384_experts_float16_t_0(self):
    #     moe_gate_func(q_len=16, num_experts=384, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=0, test_dtype=torch.float16, scale_factor=1.0)

    # def test_moe_gate_384_experts_float16_f_0(self):
    #     moe_gate_func(q_len=16, num_experts=384, topk=8, num_expert_group=1, top_k_group=1, renormalize=False, num_shared_experts=0, test_dtype=torch.float16, scale_factor=1.0)

    # def test_moe_gate_384_experts_float16_t_1(self):
    #     moe_gate_func(q_len=16, num_experts=384, topk=9, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=1, test_dtype=torch.float16, scale_factor=1.0)

    # def test_moe_gate_384_experts_float16_f_1(self):
    #     moe_gate_func(q_len=16, num_experts=384, topk=9, num_expert_group=1, top_k_group=1, renormalize=False, num_shared_experts=1, test_dtype=torch.float16, scale_factor=1.0)

    # def test_moe_gate_384_experts_float32_t_2(self):
    #     moe_gate_func_more_share_experts(q_len=16, num_experts=384, topk=9, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=2, test_dtype=torch.float16, scale_factor=1.0)

    # def test_moe_gate_384_experts_float32_f_2(self):
    #     moe_gate_func_more_share_experts(q_len=16, num_experts=384, topk=9, num_expert_group=1, top_k_group=1, renormalize=False, num_shared_experts=2, test_dtype=torch.float16, scale_factor=1.0)

    # def test_moe_gate_384_experts_float32_t_3(self):
    #     moe_gate_func_more_share_experts(q_len=16, num_experts=384, topk=9, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=3, test_dtype=torch.bfloat16, scale_factor=1.0)

    # def test_moe_gate_384_experts_float32_t_4(self):
    #     moe_gate_func_more_share_experts(q_len=16, num_experts=384, topk=9, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=4, test_dtype=torch.bfloat16, scale_factor=1.0)

if __name__ == '__main__':
    unittest.main()
