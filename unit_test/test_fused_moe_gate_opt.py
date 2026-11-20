"""
Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""

import os
import sys
import random
import torch
import unittest
from typing import Callable, NamedTuple, Optional
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
#from mcoplib import op as ops
import mcoplib.sgl_kernel
from measure_cuda import measure_cuda

def set_seed(seed=42):
    # Python
    random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)   # 多GPU

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

    # Reference is the model-correct FP32 routing: sglang's upstream single-group
    # topk kernel consumes FP32 scores/bias, and GLM/ceval quality is defined by
    # FP32 gating. Compute the whole gate (sigmoid + bias + top-k selection) in
    # FP32 here regardless of the input dtype, so this reference is the ground
    # truth the bf16 kernel is compared against. (Previously this reference ran in
    # the input dtype, i.e. bf16, which masked the kernel's bf16 selection bug.)
    gating_output = gating_output.float()
    correction_bias = correction_bias.float()

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
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=(True if num_fused_shared_experts >0 else False)) #in sglang, sorted=num_fused_shared_experts > 0
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

    return topk_weights, topk_ids

def moe_gate_func(q_len, num_experts, topk, num_expert_group, top_k_group, renormalize=True, num_shared_experts=1, test_dtype=torch.bfloat16, scale_factor=1.0, test_name="", near_tie=False):
    print(f"moe_gate_func test_function_name:{test_name} test_dtype:{test_dtype} q_len:{q_len} num_experts:{num_experts} topk:{topk} near_tie:{near_tie}")
    set_seed(42)
    if near_tie:
        # Adversarial GLM-style input: make sigmoid(logit)+bias cluster tightly
        # around the top-k boundary so that the difference between the k-th and
        # (k+1)-th expert is smaller than the bf16 ULP at that magnitude
        # (~2**-8 ≈ 0.004 near 0.7). An FP32 gate resolves the ordering; a bf16
        # gate rounds neighbours to the same/incorrectly-ordered value and flips
        # the expert selection — exactly the GLM5.2/ceval failure mode.
        base = torch.full((q_len, num_experts), 0.85, dtype=torch.float32)
        # tiny per-expert perturbations well below the bf16 step near 0.7
        jitter = (torch.arange(num_experts, dtype=torch.float32) - num_experts / 2) * 1e-3
        gating_output = (base + jitter.unsqueeze(0)).to(test_dtype).cuda()
        correction_bias = (torch.arange(num_experts, dtype=torch.float32) * 1e-3).to(test_dtype).cuda()
    else:
        gating_output = torch.rand(q_len, num_experts, dtype=test_dtype).cuda()
        correction_bias = torch.rand(num_experts, dtype=test_dtype).cuda()
    out_routing_weights = torch.zeros(q_len, topk, dtype=torch.float).cuda()
    out_selected_experts = torch.zeros(q_len, topk, dtype=torch.int32).cuda()
    device = 0
    torch.cuda.set_device(device)
    golden_routing_weights, golden_selected_experts = biased_grouped_topk(gating_output, gating_output, correction_bias, topk, renormalize, num_expert_group, top_k_group, num_shared_experts, scale_factor)

    def kernel():
        torch.ops.sgl_kernel.fused_moe_gate_opt(gating_output, correction_bias, out_routing_weights, out_selected_experts, topk, renormalize, num_expert_group, top_k_group, num_shared_experts, scale_factor)
        #torch.ops.sgl_kernel.fused_moe_gate_opt(gating_output, correction_bias, out_routing_weights, out_selected_experts, topk, renormalize, num_expert_group, top_k_group, num_shared_experts, scale_factor)
    stats = measure_cuda(kernel, iters=200, warmup=20, device=device)
    print("PyTorch matmul stats (microseconds):")
    print(f"mean={stats['mean_us']:.3f}µs median={stats['median_us']:.3f}µs min={stats['min_us']:.3f}µs stdev={stats['stdev_us']:.3f}µs")
    routing_weights = out_routing_weights
    selected_experts = out_selected_experts
    sorted_golden_routing = torch.sort(golden_routing_weights, dim=1).values  # 按行排序
    sorted_routing = torch.sort(routing_weights, dim=1).values
    verify_accuracy(sorted_golden_routing, sorted_routing, 0.0001, test_name)
    sorted_golden_experts = torch.sort(golden_selected_experts, dim=1).values  # 按行排序
    sorted_selected_experts = torch.sort(selected_experts, dim=1).values
    #print(f"sorted_golden_experts:{sorted_golden_experts} \n \t sorted_selected_experts:{sorted_selected_experts}")
    assert torch.equal(sorted_golden_experts, sorted_selected_experts)


class TestMoeGate(unittest.TestCase):
    
    def test_moe_gate_160_experts_bfloat16_f_0(self):
        moe_gate_func(q_len=16, num_experts=160, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=0, test_dtype=torch.bfloat16, scale_factor=2.5, test_name="test_moe_gate_160_experts_bfloat16_f_0")

    # GLM5.2 config (topk=8, num_expert_group=1, topk_group=1, 160 experts, bf16).
    # Adversarial near-tie inputs at the top-k boundary: reproduces the bf16
    # gating precision bug (wrong expert selection) that caused the ~3% ceval
    # accuracy drop. With the FP32-gate kernel fix this must pass; with the old
    # bf16-gate kernel the sorted expert-id equality check fails.
    def test_moe_gate_160_experts_bfloat16_glm_near_tie(self):
        moe_gate_func(q_len=16, num_experts=160, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=0, test_dtype=torch.bfloat16, scale_factor=2.5, test_name="test_moe_gate_160_experts_bfloat16_glm_near_tie", near_tie=True)

    # Same GLM config at 256 experts, near-tie adversarial input.
    def test_moe_gate_256_experts_bfloat16_glm_near_tie(self):
        moe_gate_func(q_len=16, num_experts=256, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=0, test_dtype=torch.bfloat16, scale_factor=2.5, test_name="test_moe_gate_256_experts_bfloat16_glm_near_tie", near_tie=True)

    # ---- Grouped routing (num_expert_group>1) coverage ----
    # These exercise the moe_topk_block_phase1 device path (fused_topk with
    # NUM_GROUPS>1), which the single-group tests above never touch. This is the
    # DeepSeek-V3 style config dispatched by LAUNCH_MOE_GATE(0,256,8,4,8):
    # 256 experts, 8 groups (32 experts/group), top-4 groups, top-8 experts.
    # The fp32 rewrite of moe_topk_block_phase1 (sigmoid, score+bias sort key,
    # per-group top1+top2 group score, and all three warp sorts done in fp32)
    # is validated here: the near-tie case fails on the old bf16-internal grouped
    # code (wrong group/expert selection) and passes with the fp32 fix.
    def test_moe_gate_256_experts_group8_bfloat16_deepseek(self):
        moe_gate_func(q_len=16, num_experts=256, topk=8, num_expert_group=8, top_k_group=4, renormalize=True, num_shared_experts=0, test_dtype=torch.bfloat16, scale_factor=2.5, test_name="test_moe_gate_256_experts_group8_bfloat16_deepseek")

    def test_moe_gate_256_experts_group8_bfloat16_near_tie(self):
        moe_gate_func(q_len=16, num_experts=256, topk=8, num_expert_group=8, top_k_group=4, renormalize=True, num_shared_experts=0, test_dtype=torch.bfloat16, scale_factor=2.5, test_name="test_moe_gate_256_experts_group8_bfloat16_near_tie", near_tie=True)

    def test_moe_gate_256_experts_group8_float32_deepseek(self):
        moe_gate_func(q_len=16, num_experts=256, topk=8, num_expert_group=8, top_k_group=4, renormalize=True, num_shared_experts=0, test_dtype=torch.float32, scale_factor=1.0, test_name="test_moe_gate_256_experts_group8_float32_deepseek")

    def test_moe_gate_160_experts_bfloat16_f_1(self):
        moe_gate_func(q_len=16, num_experts=160, topk=9, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=1, test_dtype=torch.float32, scale_factor=1.0, test_name="test_moe_gate_160_experts_bfloat16_f_1")

    def test_moe_gate_256_experts_float32_t_1(self):
        moe_gate_func(q_len=16, num_experts=256, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=0, test_dtype=torch.float32, scale_factor=1.0,test_name="test_moe_gate_256_experts_float32_t_1")

    def test_moe_gate_256_experts_float16_t_1(self):
        moe_gate_func(q_len=6, num_experts=256, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=0, test_dtype=torch.bfloat16, scale_factor=2.5,test_name="test_moe_gate_256_experts_float16_t_1")

    def test_moe_gate_288_experts_float32_t_0(self):
        moe_gate_func(q_len=32, num_experts=288, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=0, test_dtype=torch.float32, scale_factor=1.0,test_name="test_moe_gate_288_experts_float32_t_0")

    def test_moe_gate_288_experts_float16_t_0(self):
        moe_gate_func(q_len=32, num_experts=288, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=0, test_dtype=torch.bfloat16, scale_factor=1.0,test_name="test_moe_gate_288_experts_float16_t_0")

    def test_moe_gate_896_experts_bfloat16_topk16(self):
        # kimi-k3 配置: 896 路由专家, 16 选, 无共享专家, 单组
        moe_gate_func(q_len=16, num_experts=896, topk=16, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=0, test_dtype=torch.bfloat16, scale_factor=1.0, test_name="test_moe_gate_896_experts_bfloat16_topk16")

    def test_moe_gate_896_experts_float32_topk16(self):
        moe_gate_func(q_len=16, num_experts=896, topk=16, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=0, test_dtype=torch.float32, scale_factor=2.5, test_name="test_moe_gate_896_experts_float32_topk16")

    def test_moe_gate_896_experts_float16_topk16(self):
        moe_gate_func(q_len=32, num_experts=896, topk=16, num_expert_group=1, top_k_group=1, renormalize=True, num_shared_experts=0, test_dtype=torch.float16, scale_factor=1.0, test_name="test_moe_gate_896_experts_float16_topk16")

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
