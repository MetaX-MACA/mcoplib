import torch
import torch.nn.functional as F
import math

# 尝试导入自定义的 CUDA 算子库
try:
    import mcoplib._moe_C
except ImportError:
    print("Warning: 无法导入 mcoplib._moe_C，请确保算子已正确编译并安装。")
    print("测试脚本将继续，但在调用 CUDA 算子时会触发异常。")

def cpu_grouped_topk_reference(
    scores: torch.Tensor, 
    n_group: int, 
    topk_group: int, 
    topk: int, 
    renormalize: bool, 
    routed_scaling_factor: float, 
    bias: torch.Tensor, 
    scoring_func: int
):
    """
    纯 PyTorch 实现的 Grouped Top-K 路由逻辑，作为精度校验的基准 (Baseline)
    scoring_func: 0 -> Softmax, 1 -> Sigmoid
    """
    # 强制转为 float32 防止累加过程中的精度溢出
    scores_fp32 = scores.float()
    bias_fp32 = bias.float() if bias is not None else None

    # 1. 激活函数
    if scoring_func == 0:
        act_scores = torch.softmax(scores_fp32, dim=-1)
    elif scoring_func == 1:
        act_scores = torch.sigmoid(scores_fp32)
    else:
        raise ValueError("Unsupported scoring_func")

    num_token = act_scores.size(0)
    num_experts = act_scores.size(-1)
    experts_per_group = num_experts // n_group
    
    # 保存一份未加 bias 的原始分数值用于最后的权重提取
    original_scores = act_scores

    # 2. 计算 Group Scores
    if bias_fp32 is not None:
        act_scores = act_scores + bias_fp32.unsqueeze(0)
        # DeepSeek 逻辑：每组的得分为组内 Top-2 的专家得分之和
        group_scores = act_scores.view(num_token, n_group, experts_per_group).topk(2, dim=-1)[0].sum(dim=-1)
    else:
        group_scores = act_scores.view(num_token, n_group, experts_per_group).max(dim=-1).values

    # 3. 选出 Top-K 组
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=True)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    
    # 展开 Mask 到专家维度
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, n_group, experts_per_group)
        .reshape(num_token, -1)
    )

    # 4. 未被选中组内的专家置为 -inf
    tmp_scores = act_scores.masked_fill(~score_mask.bool(), float("-inf"))

    # 5. 选出最终的 Top-K 专家
    topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=True)[1]

    # 6. 从【原始分数(未加bias)】中提取路由权重
    topk_weights = original_scores.gather(1, topk_ids)

    # 7. 归一化与缩放
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    # 返回与 C++ 算子签名一致的数据类型
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def run_grouped_topk_test():
    # ==========================================
    # 1. 设定测试参数 (模拟 DeepSeek 配置)
    # ==========================================
    NUM_TOKENS = 64
    NUM_EXPERTS = 256
    N_GROUP = 8
    TOPK_GROUP = 4
    TOPK = 8
    RENORMALIZE = True
    ROUTED_SCALING_FACTOR = 2.0
    SCORING_FUNC = 1  # 1 代表 Sigmoid
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"=== 开始测试 Grouped TopK ===")
    print(f"配置: Tokens={NUM_TOKENS}, Experts={NUM_EXPERTS}, Groups={N_GROUP}, TopK={TOPK}")

    # 构造随机输入
    torch.manual_seed(42)
    scores = torch.randn((NUM_TOKENS, NUM_EXPERTS), dtype=torch.bfloat16, device=device)
    bias = torch.randn((NUM_EXPERTS,), dtype=torch.float32, device=device)

    # ==========================================
    # 2. 运行 Torch 基准实现
    # ==========================================
    ref_weights, ref_indices = cpu_grouped_topk_reference(
        scores, N_GROUP, TOPK_GROUP, TOPK, RENORMALIZE, ROUTED_SCALING_FACTOR, bias, SCORING_FUNC
    )

    # ==========================================
    # 3. 运行 CUDA 算子实现
    # ==========================================
    try:
        # 调用对外声明的 C++ 接口
        cuda_weights, cuda_indices = torch.ops._moe_C.grouped_topk(
            scores, N_GROUP, TOPK_GROUP, TOPK, RENORMALIZE, ROUTED_SCALING_FACTOR, bias, SCORING_FUNC
        )
    except Exception as e:
        print(f"❌ 调用 CUDA 算子失败: {e}")
        return

    # ==========================================
    # 4. 精度对比 (Cosine Similarity)
    # ==========================================
    print("\n[1/2] 正在进行精度校验...")
    
    # (a) 检查索引是否完全一致
    indices_match = torch.equal(ref_indices, cuda_indices)
    if indices_match:
        print("✅ 专家索引 (Indices) 完全匹配")
    else:
        print("❌ 错误: 专家索引 (Indices) 不一致!")
        mismatch_count = (ref_indices != cuda_indices).sum().item()
        print(f"   => 共有 {mismatch_count} 个索引不匹配。")

    # (b) 使用余弦相似度检查权重精度
    # flatten 后计算一维张量的余弦相似度
    cos_sim = F.cosine_similarity(ref_weights.flatten().float(), cuda_weights.flatten().float(), dim=0).item()
    precision_error = 1.0 - cos_sim
    max_diff = torch.max(torch.abs(ref_weights - cuda_weights)).item()

    print(f"📊 权重余弦相似度: {cos_sim:.8f}")
    print(f"📊 余弦相似度误差 (1 - cos_sim): {precision_error:.8e}")
    print(f"📊 最大绝对误差 (Max Diff): {max_diff:.8e}")

    # 判断精度是否达标 (阈值设为 1e-5)
    if precision_error < 1e-5 and not math.isnan(precision_error):
        print("✅ 权重精度校验通过 (误差 < 1e-5)。")
    else:
        print(f"❌ 错误: 权重精度不达标！请检查 CUDA 核函数内部的精度溢出或隐式转换问题。")

    # ==========================================
    # 5. 耗时统计 (Torch vs CUDA)
    # ==========================================
    print("\n[2/2] 正在进行耗时统计分析...")
    WARMUP_ITERS = 10
    TEST_ITERS = 100

    # 声明 CUDA 事件计时器
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    # (a) Profile Torch 实现
    for _ in range(WARMUP_ITERS):
        _ = cpu_grouped_topk_reference(scores, N_GROUP, TOPK_GROUP, TOPK, RENORMALIZE, ROUTED_SCALING_FACTOR, bias, SCORING_FUNC)
    torch.cuda.synchronize()

    start_evt.record()
    for _ in range(TEST_ITERS):
        _ = cpu_grouped_topk_reference(scores, N_GROUP, TOPK_GROUP, TOPK, RENORMALIZE, ROUTED_SCALING_FACTOR, bias, SCORING_FUNC)
    end_evt.record()
    torch.cuda.synchronize()
    torch_avg_time = start_evt.elapsed_time(end_evt) / TEST_ITERS

    # (b) Profile CUDA 算子
    for _ in range(WARMUP_ITERS):
        _ = torch.ops._moe_C.grouped_topk(scores, N_GROUP, TOPK_GROUP, TOPK, RENORMALIZE, ROUTED_SCALING_FACTOR, bias, SCORING_FUNC)
    torch.cuda.synchronize()

    start_evt.record()
    for _ in range(TEST_ITERS):
        _ = torch.ops._moe_C.grouped_topk(scores, N_GROUP, TOPK_GROUP, TOPK, RENORMALIZE, ROUTED_SCALING_FACTOR, bias, SCORING_FUNC)
    end_evt.record()
    torch.cuda.synchronize()
    cuda_avg_time = start_evt.elapsed_time(end_evt) / TEST_ITERS

    print(f"⏱️  Torch 基准平均耗时: {torch_avg_time:.4f} ms")
    print(f"⏱️  CUDA 算子平均耗时 : {cuda_avg_time:.4f} ms")
    if cuda_avg_time > 0:
        print(f"🚀 CUDA 算子加速比  : {torch_avg_time / cuda_avg_time:.2f}x")
    
    print("\n=== 测试圆满结束 ===")

if __name__ == "__main__":
    run_grouped_topk_test()