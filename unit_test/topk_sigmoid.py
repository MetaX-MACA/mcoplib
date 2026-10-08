import torch
import time

try:
    import mcoplib._moe_C
except ImportError:
    print("Warning: 无法导入 mcoplib._moe_C，请确保算子已正确编译安装在环境中。")


def cpu_topk_sigmoid_reference(gating_output, bias, topk, renormalize, routed_scaling_factor, is_padding=None):
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

    if is_padding is not None:
        padding_mask = is_padding.to(torch.bool)
        num_experts = gating_output.shape[-1]
        topk_weights = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(topk_weights), topk_weights)
        padding_indices = torch.full_like(topk_indices, num_experts)
        topk_indices = torch.where(padding_mask.unsqueeze(-1), padding_indices, topk_indices)

    return topk_weights, topk_indices


def check_accuracy(gating_output, bias, cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices, topk, renormalize, routed_scaling_factor, is_padding):
    num_experts = gating_output.shape[-1]

    ref_weights, ref_indices = cpu_topk_sigmoid_reference(gating_output, bias, topk, renormalize, routed_scaling_factor, is_padding)
    ref_indices = ref_indices.to(torch.int32)

    torch.ops._moe_C.topk_sigmoid(cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices, gating_output, renormalize, bias, routed_scaling_factor, is_padding)
    torch.cuda.synchronize()

    # ================================================================
    # 1. TopK expert 集合必须完全一致
    #
    # TopK score 完全相等时，CUDA 和 torch.topk 的内部排序可能不同。
    # 因此先按照 expert index 排序，再做严格 torch.equal()。
    # ================================================================
    ref_sorted_indices, ref_order = torch.sort(ref_indices, dim=-1)
    cuda_sorted_indices, cuda_order = torch.sort(cuda_topk_indices, dim=-1)

    indices_match = torch.equal(cuda_sorted_indices, ref_sorted_indices)

    if not indices_match:
        mismatch = cuda_sorted_indices != ref_sorted_indices
        mismatch_count = mismatch.any(dim=-1).sum().item()

        first_row = torch.nonzero(mismatch.any(dim=-1), as_tuple=False)[0].item()

        print()
        print("❌ topk_indices mismatch")
        print(f"mismatch_rows={mismatch_count}")
        print(f"total_rows={ref_sorted_indices.shape[0]}")
        print(f"first_mismatch_row={first_row}")

        print(f"gating={gating_output[first_row].float().cpu().tolist()}")
        print(f"is_padding={is_padding[first_row].item() if is_padding is not None else None}")
        print(f"reference_indices={ref_indices[first_row].cpu().tolist()}")
        print(f"cuda_indices={cuda_topk_indices[first_row].cpu().tolist()}")
        print(f"reference_sorted={ref_sorted_indices[first_row].cpu().tolist()}")
        print(f"cuda_sorted={cuda_sorted_indices[first_row].cpu().tolist()}")

        raise AssertionError(f"❌ TopK expert 集合不一致，mismatch_rows={mismatch_count}/{ref_sorted_indices.shape[0]}")

    # ================================================================
    # 2. TopK weights
    #
    # 按 expert index 对齐，避免 tie 时 TopK 顺序不同导致 weight 错位。
    # 只允许正常浮点误差。
    # ================================================================
    ref_weights_sorted = torch.gather(ref_weights, dim=-1, index=ref_order.to(ref_weights.device))
    cuda_weights_sorted = torch.gather(cuda_topk_weights, dim=-1, index=cuda_order.to(cuda_topk_weights.device))

    weight_diff = torch.abs(cuda_weights_sorted - ref_weights_sorted)
    max_diff = weight_diff.max().item()

    weights_match = torch.allclose(cuda_weights_sorted, ref_weights_sorted, rtol=1e-5, atol=1e-5)

    if not weights_match:
        mismatch = weight_diff > (1e-5 + 1e-5 * torch.abs(ref_weights_sorted))
        mismatch_count = mismatch.sum().item()

        first_index = torch.nonzero(mismatch, as_tuple=False)[0]
        index = tuple(first_index.tolist())

        print()
        print("❌ topk_weights mismatch")
        print(f"mismatch_count={mismatch_count}")
        print(f"total_count={cuda_weights_sorted.numel()}")
        print(f"max_diff={max_diff}")
        print(f"first_mismatch_index={index}")
        print(f"actual={cuda_weights_sorted[index].item()}")
        print(f"expected={ref_weights_sorted[index].item()}")

        row = index[0]
        print(f"gating={gating_output[row].float().cpu().tolist()}")
        print(f"reference_indices={ref_indices[row].cpu().tolist()}")
        print(f"cuda_indices={cuda_topk_indices[row].cpu().tolist()}")

        raise AssertionError("❌ topk_weights 超出允许浮点误差")

    # ================================================================
    # 3. token_expert_indices
    #
    # 不参与 accuracy 检查。
    # ================================================================

    # ================================================================
    # 4. Padding token
    #
    # padding 的 topk_indices 仍然严格检查。
    # padding weights 必须为 0。
    # token_expert_indices 不检查。
    # ================================================================
    if is_padding is not None:
        padding_rows = torch.nonzero(is_padding.cpu()).flatten()

        if padding_rows.numel() > 0:
            padding_weights = cuda_topk_weights[padding_rows]
            padding_indices = cuda_topk_indices[padding_rows]

            expected_padding_weights = torch.zeros_like(padding_weights)
            expected_padding_indices = torch.full_like(padding_indices, num_experts)

            if not torch.equal(padding_indices, expected_padding_indices):
                raise AssertionError(f"❌ is_padding=True 的 token indices 不是严格 num_experts={num_experts}")

            if not torch.allclose(padding_weights, expected_padding_weights, rtol=1e-5, atol=1e-5):
                max_padding_weight = torch.max(torch.abs(padding_weights)).item()
                raise AssertionError(f"❌ is_padding=True 的 token 权重不是 0，max_abs={max_padding_weight}")

    return max_diff

    # ---------------------------------------------------------
    # 4. Performance Test
    # ---------------------------------------------------------
    print("\n[2/2] 正在进行耗时统计分析 (Profile Reference vs CUDA)...")

def benchmark_cuda(gating_output, cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices, bias, renormalize, routed_scaling_factor, is_padding, warmup, runs):
    for _ in range(warmup):
        torch.ops._moe_C.topk_sigmoid(cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices, gating_output, renormalize, bias, routed_scaling_factor, is_padding)

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    for _ in range(runs):
        torch.ops._moe_C.topk_sigmoid(cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices, gating_output, renormalize, bias, routed_scaling_factor, is_padding)

    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / runs


def benchmark_reference(gating_output, bias, topk, renormalize, routed_scaling_factor, is_padding, warmup, runs):
    for _ in range(warmup):
        _ = cpu_topk_sigmoid_reference(gating_output, bias, topk, renormalize, routed_scaling_factor, is_padding)

    torch.cuda.synchronize()

    start_time = time.time()

    for _ in range(runs):
        _ = cpu_topk_sigmoid_reference(gating_output, bias, topk, renormalize, routed_scaling_factor, is_padding)

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
        ("S08_COMMON_OPT_128", 128, 8),
        ("S09_COMMON_OPT_192", 192, 8),
        ("S10_COMMON_OPT_256_TOP8", 256, 8),
        ("S11_COMMON_OPT_256_TOP16", 256, 16),
        ("S12_SIGMOID_288_OPT", 288, 8),
        ("S13_LAUNCH_TOPK_320", 320, 8),
        ("S14_LAUNCH_TOPK_384", 384, 8),
        ("S15_LAUNCH_TOPK_448", 448, 8),
        ("S16_LAUNCH_TOPK_512", 512, 8),
        ("S17_LAUNCH_TOPK_576", 576, 8),
        ("S18_FALLBACK_896", 896, 8),
    ]


def get_token_candidates(num_experts):
    if num_experts <= 8:
        return [4096]
    elif num_experts <= 576:
        return [1024, 4096]
    else:
        return [512, 2048]


def get_is_padding_cases(num_tokens):
    cases = []

    # None：保持原始调用方式
    cases.append(("None", None))

    # AllFalse：显式传入 is_padding
    cases.append(("AllFalse", torch.zeros(num_tokens, dtype=torch.bool, device="cuda")))

    # Mixed：部分 token 为 padding
    mixed_padding = torch.zeros(num_tokens, dtype=torch.bool, device="cuda")
    segment = max(num_tokens // 8, 1)

    mixed_padding[:segment] = True

    middle_start = num_tokens // 2
    middle_end = min(middle_start + segment, num_tokens)
    mixed_padding[middle_start:middle_end] = True

    mixed_padding[-segment:] = True

    cases.append(("Mixed", mixed_padding))

    return cases


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
    IS_PADDING_CASES = 3
    TOTAL_CASES = TOTAL_TOKEN_CASES * len(DTYPE_LIST) * len(BIAS_LIST) * len(RENORMALIZE_LIST) * len(ROUTED_SCALING_FACTORS) * IS_PADDING_CASES

    print("======================================================================")
    print("              topk_sigmoid Launcher 全场景测试")
    print("======================================================================")
    print(f"Test Cases          : {len(TEST_CASES)}")
    print(f"Token Cases         : {TOTAL_TOKEN_CASES}")
    print(f"Dtypes              : {DTYPE_LIST}")
    print(f"Bias                : {BIAS_LIST}")
    print(f"Renormalize         : {RENORMALIZE_LIST}")
    print(f"Scaling Factors     : {ROUTED_SCALING_FACTORS}")
    print(f"is_padding Cases    : {IS_PADDING_CASES}")
    print("  - None")
    print("  - AllFalse")
    print("  - Mixed")
    print(f"Total Accuracy Test : {TOTAL_CASES}")
    print("======================================================================")

    torch.manual_seed(42)

    results = []
    case_id = 0
    passed_cases = 0
    failed_cases = 0

    for case_name, num_experts, topk in TEST_CASES:
        token_candidates = get_token_candidates(num_experts)

        for dtype in DTYPE_LIST:
            for num_tokens in token_candidates:
                torch.manual_seed(42)
                gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")

                for use_bias in BIAS_LIST:
                    bias = torch.randn(num_experts, dtype=torch.float32, device="cuda") if use_bias else None
                    is_padding_cases = get_is_padding_cases(num_tokens)

                    for is_padding_name, is_padding in is_padding_cases:
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
                                print(f"bias={use_bias}, renormalize={renormalize}, is_padding={is_padding_name}, routed_scaling_factor={routed_scaling_factor}")
                                print("----------------------------------------------------------------------")

                                try:
                                    max_diff = check_accuracy(gating_output, bias, cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices, topk, renormalize, routed_scaling_factor, is_padding)
                                    passed_cases += 1
                                    accuracy_result = "PASS"
                                    print(f"✅ Accuracy PASS | topk_indices=torch.equal() | MaxWeightDiff={max_diff:.8e}")
                                except Exception as e:
                                    failed_cases += 1
                                    accuracy_result = "FAIL"
                                    print(f"❌ Accuracy FAIL | {e}")
                                    raise

                                ref_avg_time = benchmark_reference(gating_output, bias, topk, renormalize, routed_scaling_factor, is_padding, WARMUP, RUNS)
                                cuda_avg_time = benchmark_cuda(gating_output, cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices, bias, renormalize, routed_scaling_factor, is_padding, WARMUP, RUNS)

                                speedup = ref_avg_time / cuda_avg_time if cuda_avg_time > 0 else 0.0

                                print(f"⏱ Reference={ref_avg_time:.4f} ms | CUDA={cuda_avg_time:.4f} ms | Speedup={speedup:.2f}x")

                                results.append({
                                    "case": case_name,
                                    "experts": num_experts,
                                    "topk": topk,
                                    "dtype": str(dtype),
                                    "tokens": num_tokens,
                                    "bias": use_bias,
                                    "renormalize": renormalize,
                                    "is_padding": is_padding_name,
                                    "scale": routed_scaling_factor,
                                    "accuracy": accuracy_result,
                                    "max_diff": max_diff,
                                    "ref_ms": ref_avg_time,
                                    "cuda_ms": cuda_avg_time,
                                    "speedup": speedup,
                                })

    print()
    print("======================================================================")
    print("                         Final Summary")
    print("======================================================================")
    print(f"{'Case':<28} {'Experts':>8} {'TopK':>6} {'Dtype':>12} {'Bias':>6} {'Norm':>6} {'Padding':>10} {'Scale':>7} {'Accuracy':>10} {'MaxDiff':>12} {'CUDA(ms)':>10}")
    print("-" * 140)

    for result in results:
        print(f"{result['case']:<28} {result['experts']:>8} {result['topk']:>6} {result['dtype'].replace('torch.', ''):>12} {str(result['bias']):>6} {str(result['renormalize']):>6} {result['is_padding']:>10} {result['scale']:>7.2f} {result['accuracy']:>10} {result['max_diff']:>12.4e} {result['cuda_ms']:>10.4f}")

    print("======================================================================")
    print(f"=== PASS: {passed_cases} ===")
    print(f"=== FAIL: {failed_cases} ===")
    print(f"=== Completed: {len(results)} ===")
    print(f"=== Expected Total Cases: {TOTAL_CASES} ===")
    print("======================================================================")

    if failed_cases != 0 or len(results) != TOTAL_CASES:
        raise AssertionError(f"topk_sigmoid strict accuracy test failed: PASS={passed_cases}, FAIL={failed_cases}, TOTAL={TOTAL_CASES}")

    print("=== FINAL RESULT: PASS ===")



if __name__ == "__main__":
    run_test()