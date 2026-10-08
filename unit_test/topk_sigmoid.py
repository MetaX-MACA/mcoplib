import torch
import torch.nn.functional as F
import time
import math

try:
    import mcoplib._moe_C
except ImportError:
    print("Warning: 无法导入 mcoplib._moe_C，请确保算子已正确编译安装在环境中。")


def cpu_topk_sigmoid_reference(gating_output, bias, topk, renormalize, routed_scaling_factor):
    gating_fp32 = gating_output.float()
    bias_fp32 = bias.float() if bias is not None else None

    sigmoid_scores = torch.sigmoid(gating_fp32)

    if bias_fp32 is not None:
        routing_scores = sigmoid_scores + bias_fp32
    else:
        routing_scores = sigmoid_scores

    _, topk_indices = torch.topk(routing_scores, k=topk, dim=-1)

    topk_weights = torch.gather(sigmoid_scores, dim=-1, index=topk_indices)

    if renormalize:
        row_sum = topk_weights.sum(dim=-1, keepdim=True)
        row_sum = torch.where(row_sum > 0.0, row_sum, torch.ones_like(row_sum))
        topk_weights = topk_weights / row_sum

    topk_weights = topk_weights * float(routed_scaling_factor)

    return topk_weights, topk_indices


def check_accuracy(gating_output, bias, cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices, topk, renormalize, routed_scaling_factor):
    ref_weights, ref_indices = cpu_topk_sigmoid_reference(gating_output, bias, topk, renormalize, routed_scaling_factor)

    torch.ops._moe_C.topk_sigmoid(cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices, gating_output, renormalize, bias, routed_scaling_factor)

    torch.cuda.synchronize()

    ref_indices_cpu = ref_indices.to(torch.int32).cpu()
    cuda_indices_cpu = cuda_topk_indices.cpu()

    # Top-K 输出顺序允许在 score 完全相等时不同，因此比较排序后的 expert 集合。
    ref_sorted_indices, ref_order = torch.sort(ref_indices_cpu, dim=-1)
    cuda_sorted_indices, cuda_order = torch.sort(cuda_indices_cpu, dim=-1)

    indices_match = torch.equal(ref_sorted_indices, cuda_sorted_indices)

    if not indices_match:
        mismatch = ref_sorted_indices != cuda_sorted_indices
        mismatch_count = mismatch.any(dim=-1).sum().item()

        print()
        print("❌ Top-K expert set mismatch")
        print(f"mismatch_rows={mismatch_count}")
        print(f"total_rows={ref_indices_cpu.shape[0]}")

        rows = torch.nonzero(mismatch.any(dim=-1)).flatten()[:10]

        for row in rows.tolist():
            print(f"row={row}")
            print(f"gating={gating_output[row].float().cpu().tolist()}")
            print(f"reference_indices={ref_indices_cpu[row].tolist()}")
            print(f"cuda_indices={cuda_indices_cpu[row].tolist()}")
            print(f"reference_sorted={ref_sorted_indices[row].tolist()}")
            print(f"cuda_sorted={cuda_sorted_indices[row].tolist()}")

        raise AssertionError(f"❌ routed_scaling_factor={routed_scaling_factor}: CUDA 与 Reference 选出的 Top-K expert 集合不一致，mismatch_rows={mismatch_count}/{ref_indices_cpu.shape[0]}")

    # 按 expert index 对齐 weight，避免 Top-K tie 时输出顺序不同导致 weight 错位比较。
    ref_weights_sorted = torch.gather(ref_weights, dim=-1, index=ref_order.to(ref_weights.device))
    cuda_weights_sorted = torch.gather(cuda_topk_weights, dim=-1, index=cuda_order.to(cuda_topk_weights.device))

    cos_sim = F.cosine_similarity(ref_weights_sorted.flatten(), cuda_weights_sorted.flatten(), dim=0)

    cos_sim_val = cos_sim.item()
    precision_error = 1.0 - cos_sim_val
    max_diff = torch.max(torch.abs(ref_weights_sorted - cuda_weights_sorted)).item()

    assert not math.isnan(precision_error), f"❌ routed_scaling_factor={routed_scaling_factor}: 遇到 NaN"

    assert precision_error < 0.00001, f"❌ routed_scaling_factor={routed_scaling_factor}: 余弦相似度误差 {precision_error} >= 0.00001"

    return cos_sim_val, precision_error, max_diff


def benchmark_cuda(gating_output, cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices, bias, renormalize, routed_scaling_factor, warmup, runs):
    for _ in range(warmup):
        torch.ops._moe_C.topk_sigmoid(cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices, gating_output, renormalize, bias, routed_scaling_factor)

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    for _ in range(runs):
        torch.ops._moe_C.topk_sigmoid(cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices, gating_output, renormalize, bias, routed_scaling_factor)

    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / runs


def benchmark_reference(gating_output, bias, topk, renormalize, routed_scaling_factor, warmup, runs):
    for _ in range(warmup):
        _ = cpu_topk_sigmoid_reference(gating_output, bias, topk, renormalize, routed_scaling_factor)

    torch.cuda.synchronize()

    start_time = time.time()

    for _ in range(runs):
        _ = cpu_topk_sigmoid_reference(gating_output, bias, topk, renormalize, routed_scaling_factor)

    torch.cuda.synchronize()

    return (time.time() - start_time) / runs * 1000.0


def get_test_cases():
    return [
        ("S01_LAUNCH_TOPK_1", 1, 1),
        ("S02_LAUNCH_TOPK_2", 2, 2),
        ("S03_LAUNCH_TOPK_4", 4, 4),
        ("S04_LAUNCH_TOPK_8", 8, 8),
        ("S05_LAUNCH_TOPK_16", 16, 8),
        ("S06_LAUNCH_TOPK_32", 32, 8),
        ("S07_LAUNCH_TOPK_64", 64, 8),

        # SIGMOID -> topkGatingSigmoidCommonOpt
        ("S08_COMMON_OPT_128", 128, 8),
        ("S09_COMMON_OPT_192", 192, 8),
        ("S10_COMMON_OPT_256_TOP8", 256, 8),
        ("S11_COMMON_OPT_256_TOP16", 256, 16),

        # SIGMOID -> topkGatingSigmoid288Opt
        ("S12_SIGMOID_288_OPT", 288, 8),

        # CUDA multiples of 64
        ("S13_LAUNCH_TOPK_320", 320, 8),
        ("S14_LAUNCH_TOPK_384", 384, 8),
        ("S15_LAUNCH_TOPK_448", 448, 8),
        ("S16_LAUNCH_TOPK_512", 512, 8),
        ("S17_LAUNCH_TOPK_576", 576, 8),

        # default -> moeSigmoid + moeTopK
        ("S18_FALLBACK_896", 896, 8),
    ]


def get_token_candidates(num_experts):
    if num_experts <= 8:
        return [4096]
    elif num_experts <= 576:
        return [1024, 4096]
    else:
        return [512, 2048]


def run_test():
    TEST_CASES = get_test_cases()

    DTYPE_LIST = [
        torch.float16,
        torch.bfloat16,
    ]

    ROUTED_SCALING_FACTORS = [
        0.1,
        0.2,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
    ]

    BIAS_LIST = [
        False,
        True,
    ]

    RENORMALIZE_LIST = [
        False,
        True,
    ]

    WARMUP = 10
    RUNS = 100

    TOTAL_TOKEN_CASES = sum(len(get_token_candidates(num_experts)) for _, num_experts, _ in TEST_CASES)

    TOTAL_CASES = TOTAL_TOKEN_CASES * len(DTYPE_LIST) * len(BIAS_LIST) * len(RENORMALIZE_LIST) * len(ROUTED_SCALING_FACTORS)

    print("======================================================================")
    print("              topk_sigmoid Launcher 全场景测试")
    print("======================================================================")
    print(f"Test Cases          : {len(TEST_CASES)}")
    print(f"Token Cases         : {TOTAL_TOKEN_CASES}")
    print(f"Dtypes              : {DTYPE_LIST}")
    print(f"Bias                : {BIAS_LIST}")
    print(f"Renormalize         : {RENORMALIZE_LIST}")
    print(f"Scaling Factors     : {ROUTED_SCALING_FACTORS}")
    print(f"Total Accuracy Test : {TOTAL_CASES}")
    print("======================================================================")

    torch.manual_seed(42)

    results = []
    case_id = 0

    for case_name, num_experts, topk in TEST_CASES:
        token_candidates = get_token_candidates(num_experts)

        for dtype in DTYPE_LIST:
            for num_tokens in token_candidates:
                torch.manual_seed(42)

                gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")

                for use_bias in BIAS_LIST:
                    bias = torch.randn(num_experts, dtype=torch.float32, device="cuda") if use_bias else None

                    for renormalize in RENORMALIZE_LIST:
                        cuda_topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
                        cuda_topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
                        cuda_token_expert_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

                        for routed_scaling_factor in ROUTED_SCALING_FACTORS:
                            case_id += 1

                            print()
                            print("======================================================================")
                            print(f"[{case_id}/{TOTAL_CASES}] {case_name}")
                            print(f"experts={num_experts}, topk={topk}, dtype={dtype}, tokens={num_tokens}")
                            print(f"bias={use_bias}, renormalize={renormalize}, routed_scaling_factor={routed_scaling_factor}")
                            print("----------------------------------------------------------------------")

                            cos_sim_val, precision_error, max_diff = check_accuracy(gating_output, bias, cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices, topk, renormalize, routed_scaling_factor)

                            print(f"✅ Accuracy PASS | CosSim={cos_sim_val:.10f} | 1-CosSim={precision_error:.8e} | MaxDiff={max_diff:.8e}")

                            ref_avg_time = benchmark_reference(gating_output, bias, topk, renormalize, routed_scaling_factor, WARMUP, RUNS)

                            cuda_avg_time = benchmark_cuda(gating_output, cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices, bias, renormalize, routed_scaling_factor, WARMUP, RUNS)

                            speedup = ref_avg_time / cuda_avg_time if cuda_avg_time > 0 else 0.0

                            print(f"⏱ Reference={ref_avg_time:.4f} ms | CUDA={cuda_avg_time:.4f} ms | Speedup={speedup:.2f}x")

                            results.append(
                                {
                                    "case": case_name,
                                    "experts": num_experts,
                                    "topk": topk,
                                    "dtype": str(dtype),
                                    "tokens": num_tokens,
                                    "bias": use_bias,
                                    "renormalize": renormalize,
                                    "scale": routed_scaling_factor,
                                    "cos_sim": cos_sim_val,
                                    "error": precision_error,
                                    "max_diff": max_diff,
                                    "ref_ms": ref_avg_time,
                                    "cuda_ms": cuda_avg_time,
                                    "speedup": speedup,
                                }
                            )

    print()
    print("======================================================================")
    print("                         Final Summary")
    print("======================================================================")

    print(f"{'Case':<28} {'Experts':>8} {'TopK':>6} {'Dtype':>12} {'Bias':>6} {'Norm':>6} {'Scale':>7} {'CosSim':>12} {'MaxDiff':>12} {'CUDA(ms)':>10}")
    print("-" * 125)

    for result in results:
        print(f"{result['case']:<28} {result['experts']:>8} {result['topk']:>6} {result['dtype'].replace('torch.', ''):>12} {str(result['bias']):>6} {str(result['renormalize']):>6} {result['scale']:>7.2f} {result['cos_sim']:>12.8f} {result['max_diff']:>12.4e} {result['cuda_ms']:>10.4f}")

    print("======================================================================")
    print(f"=== 全部 {len(results)} 个测试完成 ===")
    print(f"=== Expected Total Cases: {TOTAL_CASES} ===")
    print("======================================================================")


if __name__ == "__main__":
    run_test()