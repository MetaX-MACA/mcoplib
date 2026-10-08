# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import mcoplib._moe_C


TOPK_SOFTMAX_SCENARIOS = [
    (1, 1),
    (2, 1),
    (4, 1),
    (8, 2),
    (8, 4),
    (8, 8),
    (16, 1),
    (16, 4),
    (16, 8),
    (16, 16),
    (32, 1),
    (32, 8),
    (32, 16),
    (32, 32),
    (64, 1),
    (64, 8),
    (64, 16),
    (64, 32),
    (64, 64),
    (96, 1),
    (96, 2),
    (96, 4),
    (96, 8),
    (96, 16),
    (128, 1),
    (128, 2),
    (128, 4),
    (128, 8),
    (128, 16),
    (160, 1),
    (160, 4),
    (160, 8),
    (160, 16),
    (192, 1),
    (192, 4),
    (192, 8),
    (192, 16),
    (224, 1),
    (224, 2),
    (224, 4),
    (224, 8),
    (224, 16),
    (256, 1),
    (256, 2),
    (256, 4),
    (256, 8),
    (256, 16),
    (288, 1),
    (288, 4),
    (288, 8),
    (288, 16),
    (320, 1),
    (320, 4),
    (320, 8),
    (320, 16),
    (384, 1),
    (384, 4),
    (384, 8),
    (384, 16),
    (448, 1),
    (448, 4),
    (448, 8),
    (448, 16),
    (512, 1),
    (512, 4),
    (512, 8),
    (512, 16),
    (576, 1),
    (576, 4),
    (576, 8),
    (576, 16),
    (896, 1),
    (896, 4),
    (896, 8),
    (896, 16),
]


COMMON_OPT_SCENARIOS = [
    (96, 1),
    (96, 2),
    (96, 4),
    (96, 8),
    (96, 16),
    (224, 1),
    (224, 2),
    (224, 4),
    (224, 8),
    (224, 16),
]


GENERIC_SCENARIOS = [
    (520, 1),
    (520, 4),
    (520, 8),
    (520, 16),
    (640, 1),
    (640, 4),
    (640, 8),
    (640, 16),
    (768, 1),
    (768, 4),
    (768, 8),
    (768, 16),
    (1000, 1),
    (1000, 4),
    (1000, 8),
    (1000, 16),
]


DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]


OPTIMIZED_PADDING_SCENARIOS = [
    (96, 8),
    (128, 8),
    (160, 8),
    (192, 8),
    (224, 8),
    (256, 8),
    (288, 8),
    (320, 8),
    (384, 8),
    (448, 8),
    (512, 8),
    (576, 8),
    (896, 16),
]


GENERIC_PADDING_SCENARIOS = [
    (520, 8),
    (640, 8),
    (768, 8),
    (1000, 8),
]


def _topk_softmax(gating_output, topk_weights, topk_indices, token_expert_indices, renormalize=False, bias=None, is_padding=None):
    return torch.ops._moe_C.topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize, bias, is_padding)


def _make_input(num_tokens, num_experts, dtype, seed=1234):
    torch.manual_seed(seed)
    return torch.randn((num_tokens, num_experts), device="cuda", dtype=dtype)


def _reference_topk(gating_output, topk, renormalize=False, bias=None, is_padding=None):
    probs = torch.softmax(gating_output.float(), dim=-1)

    if bias is None:
        score = probs
    else:
        score = probs + bias.float()

    _, indices = torch.topk(score, k=topk, dim=-1)
    indices = indices.to(torch.int32)

    weights = torch.gather(probs, dim=-1, index=indices.long())

    if renormalize:
        denom = weights.sum(dim=-1, keepdim=True)
        denom = torch.where(denom > 0.0, denom, torch.ones_like(denom))
        weights = weights / denom

    if is_padding is not None:
        mask = is_padding.unsqueeze(-1)
        weights = torch.where(mask, torch.zeros_like(weights), weights)
        indices = torch.where(mask, torch.full_like(indices, gating_output.shape[1]), indices)

    return weights, indices


def _sort_topk_indices_and_weights(indices, weights):
    sorted_indices, order = torch.sort(indices, dim=-1)
    sorted_weights = torch.gather(weights, dim=-1, index=order.long())
    return sorted_indices, sorted_weights


def _check_topk_indices(actual, expected, is_padding=None, num_experts=None, padding_index=None):
    if actual.shape != expected.shape:
        raise AssertionError(f"topk_indices shape mismatch: actual={actual.shape}, expected={expected.shape}")

    if is_padding is None:
        actual_sorted = torch.sort(actual, dim=-1).values
        expected_sorted = torch.sort(expected, dim=-1).values

        if torch.equal(actual_sorted, expected_sorted):
            return

        mismatch = actual_sorted != expected_sorted
        mismatch_count = mismatch.sum().item()
        total_count = actual.numel()

        first_index = torch.nonzero(mismatch, as_tuple=False)[0]
        index = tuple(first_index.tolist())
        row = index[0]

        print()
        print("❌ topk_indices expert set mismatch")
        print(f"mismatch_count={mismatch_count}")
        print(f"total_count={total_count}")
        print(f"first_mismatch_index={index}")
        print(f"actual={actual[index].item()}")
        print(f"expected={expected[index].item()}")
        print(f"actual_row={actual[row].cpu().tolist()}")
        print(f"expected_row={expected[row].cpu().tolist()}")
        print(f"actual_sorted_row={actual_sorted[row].cpu().tolist()}")
        print(f"expected_sorted_row={expected_sorted[row].cpu().tolist()}")

        raise AssertionError(f"❌ topk_indices expert set 不完全相等，mismatch_count={mismatch_count}/{total_count}")

    active = ~is_padding

    if active.any():
        actual_active = actual[active]
        expected_active = expected[active]

        actual_sorted = torch.sort(actual_active, dim=-1).values
        expected_sorted = torch.sort(expected_active, dim=-1).values

        if not torch.equal(actual_sorted, expected_sorted):
            mismatch = actual_sorted != expected_sorted
            mismatch_count = mismatch.sum().item()
            total_count = actual_active.numel()

            first_index = torch.nonzero(mismatch, as_tuple=False)[0]
            index = tuple(first_index.tolist())
            row = index[0]

            print()
            print("❌ active topk_indices expert set mismatch")
            print(f"mismatch_count={mismatch_count}")
            print(f"total_count={total_count}")
            print(f"first_mismatch_index={index}")
            print(f"actual={actual_sorted[index].item()}")
            print(f"expected={expected_sorted[index].item()}")
            print(f"actual_sorted_row={actual_sorted[row].cpu().tolist()}")
            print(f"expected_sorted_row={expected_sorted[row].cpu().tolist()}")

            raise AssertionError(f"❌ active topk_indices expert set 不完全相等，mismatch_count={mismatch_count}/{total_count}")

    if is_padding.any():
        if padding_index is None:
            padding_index = num_experts

        actual_padding = actual[is_padding]

        if not torch.all(actual_padding == padding_index):
            print()
            print("❌ padding topk_indices mismatch")
            print(f"expected_padding_index={padding_index}")
            print(f"actual_padding_indices={actual_padding.cpu().tolist()}")

            raise AssertionError(f"❌ padding topk_indices 不正确，expected={padding_index}")


def _check_active_weights(actual, expected, is_padding, rtol=8e-3, atol=8e-3):
    if is_padding is None:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        return

    active = ~is_padding

    if active.any():
        torch.testing.assert_close(actual[active], expected[active], rtol=rtol, atol=atol)


def _reference_source_rows(num_tokens, topk):
    rows = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    return torch.stack([k * num_tokens + rows for k in range(topk)], dim=1)


def _check_source_rows(actual, num_tokens, topk):
    expected = _reference_source_rows(num_tokens, topk)
    torch.testing.assert_close(actual, expected)


def _check_padding_weights(actual, is_padding, atol=0.0):
    if is_padding is not None and is_padding.any():
        if atol == 0.0:
            assert torch.all(actual[is_padding] == 0)
        else:
            torch.testing.assert_close(actual[is_padding], torch.zeros_like(actual[is_padding]), rtol=0.0, atol=atol)


def _check_padding_source_rows(actual, is_padding, num_tokens, topk):
    if is_padding is None or not is_padding.any():
        return

    expected = _reference_source_rows(num_tokens, topk)
    torch.testing.assert_close(actual[is_padding], expected[is_padding])


def _run_case(num_tokens, num_experts, topk, dtype=torch.float16, renormalize=False, bias=None, is_padding=None, rtol=8e-3, atol=8e-3, check_source_rows=False, padding_index=None, check_padding_weights=True, check_padding_source_rows=True):
    gating_output = _make_input(num_tokens, num_experts, dtype)

    if bias is not None:
        bias = bias.to(device="cuda", dtype=torch.float32)

    topk_weights = torch.empty((num_tokens, topk), device="cuda", dtype=torch.float32)
    topk_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)
    token_expert_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)

    _topk_softmax(gating_output, topk_weights, topk_indices, token_expert_indices, renormalize, bias, is_padding)

    torch.cuda.synchronize()

    ref_weights, ref_indices = _reference_topk(gating_output, topk, renormalize, bias, is_padding)

    _check_topk_indices(topk_indices, ref_indices, is_padding, num_experts, padding_index)

    # Top-K expert 集合严格一致，但是 kernel 与 torch.topk 的顺序可能不同。
    # 因此按照 expert index 排序后再比较 weight，避免仅仅因为顺序不同导致 weight 错位。
    actual_sorted_indices, actual_sorted_weights = _sort_topk_indices_and_weights(topk_indices, topk_weights)
    ref_sorted_indices, ref_sorted_weights = _sort_topk_indices_and_weights(ref_indices, ref_weights)

    if is_padding is not None:
        if check_padding_weights:
            _check_padding_weights(topk_weights, is_padding)

        if check_padding_source_rows:
            _check_padding_source_rows(token_expert_indices, is_padding, num_tokens, topk)

        active = ~is_padding

        if active.any():
            torch.testing.assert_close(actual_sorted_weights[active], ref_sorted_weights[active], rtol=rtol, atol=atol)
    else:
        torch.testing.assert_close(actual_sorted_weights, ref_sorted_weights, rtol=rtol, atol=atol)

    if check_source_rows:
        _check_source_rows(token_expert_indices, num_tokens, topk)

    return topk_weights, topk_indices, token_expert_indices


# ============================================================
# Launcher paths
# ============================================================

@pytest.mark.parametrize("num_experts,topk", TOPK_SOFTMAX_SCENARIOS)
def test_topk_softmax_launcher_paths(num_experts, topk):
    _run_case(8, num_experts, topk, dtype=torch.float16)


# ============================================================
# Generic/default paths
# ============================================================

@pytest.mark.parametrize("num_experts,topk", GENERIC_SCENARIOS)
def test_topk_softmax_generic_paths(num_experts, topk):
    _run_case(8, num_experts, topk, dtype=torch.float16)


# ============================================================
# CommonOpt
# ============================================================

@pytest.mark.parametrize("num_experts,topk", COMMON_OPT_SCENARIOS)
def test_topk_softmax_common_opt(num_experts, topk):
    _run_case(8, num_experts, topk, dtype=torch.float16)


# ============================================================
# CommonOpt token boundaries
# ============================================================

@pytest.mark.parametrize("num_tokens", [1, 2, 3, 7, 8, 16, 31, 32, 63, 64, 127, 128])
@pytest.mark.parametrize("num_experts,topk", [(96, 8), (224, 8)])
def test_topk_softmax_common_opt_token_boundaries(num_experts, topk, num_tokens):
    _run_case(num_tokens, num_experts, topk, dtype=torch.float16)


# ============================================================
# Dtype
# ============================================================

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_experts,topk", [(96, 8), (128, 8), (224, 8), (256, 8), (512, 8), (896, 16)])
def test_topk_softmax_dtype(num_experts, topk, dtype):
    _run_case(8, num_experts, topk, dtype=dtype, rtol=8e-3, atol=8e-3)


# ============================================================
# Renormalize
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(96, 1), (96, 8), (96, 16), (128, 8), (224, 1), (224, 8), (224, 16), (256, 8), (512, 8), (896, 16)])
def test_topk_softmax_renormalize(num_experts, topk):
    _run_case(8, num_experts, topk, dtype=torch.float16, renormalize=True, rtol=8e-3, atol=8e-3)


# ============================================================
# Bias
# ============================================================

@pytest.mark.parametrize("bias_type", ["zero", "positive", "negative", "random"])
@pytest.mark.parametrize("num_experts,topk", [(96, 8), (128, 8), (160, 8), (192, 8), (224, 8), (256, 8), (288, 8), (320, 8), (384, 8), (448, 8), (512, 8), (576, 8), (896, 16)])
def test_topk_softmax_bias(num_experts, topk, bias_type):
    if bias_type == "zero":
        bias = torch.zeros(num_experts, device="cuda", dtype=torch.float32)
    elif bias_type == "positive":
        bias = torch.linspace(0.0, 1.0, num_experts, device="cuda", dtype=torch.float32)
    elif bias_type == "negative":
        bias = torch.linspace(0.0, -1.0, num_experts, device="cuda", dtype=torch.float32)
    else:
        torch.manual_seed(5678)
        bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)

    _run_case(8, num_experts, topk, dtype=torch.float16, bias=bias, rtol=8e-3, atol=8e-3)


# ============================================================
# Strong bias
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(96, 8), (128, 8), (160, 8), (192, 8), (224, 8), (256, 8), (288, 8), (320, 8), (384, 8), (448, 8), (512, 8)])
def test_topk_softmax_strong_bias(num_experts, topk):
    bias = torch.zeros(num_experts, device="cuda", dtype=torch.float32)
    bias[0] = 100.0
    bias[1] = 50.0
    bias[2] = -50.0

    _run_case(8, num_experts, topk, dtype=torch.float16, bias=bias, rtol=8e-3, atol=8e-3)


# ============================================================
# Bias + renormalize
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(96, 8), (128, 8), (160, 8), (224, 8), (256, 8), (320, 8), (512, 8)])
def test_topk_softmax_bias_renormalize(num_experts, topk):
    torch.manual_seed(5678)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)

    _run_case(8, num_experts, topk, dtype=torch.float16, renormalize=True, bias=bias, rtol=8e-3, atol=8e-3)


# ============================================================
# Bias + padding optimized paths
# ============================================================

@pytest.mark.parametrize("num_experts,topk", OPTIMIZED_PADDING_SCENARIOS)
def test_topk_softmax_bias_is_padding(num_experts, topk):
    torch.manual_seed(9999)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)

    is_padding = torch.zeros(8, device="cuda", dtype=torch.bool)
    is_padding[1] = True
    is_padding[4] = True
    is_padding[7] = True

    _run_case(8, num_experts, topk, dtype=torch.float16, bias=bias, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=num_experts)


# ============================================================
# Bias + padding + renormalize optimized paths
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(96, 8), (128, 8), (160, 8), (224, 8), (256, 8), (320, 8), (512, 8)])
def test_topk_softmax_bias_is_padding_renormalize(num_experts, topk):
    torch.manual_seed(9999)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)

    is_padding = torch.zeros(8, device="cuda", dtype=torch.bool)
    is_padding[1] = True
    is_padding[4] = True
    is_padding[7] = True

    _run_case(8, num_experts, topk, dtype=torch.float16, renormalize=True, bias=bias, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=num_experts)


# ============================================================
# Generic padding
# ============================================================

@pytest.mark.parametrize("num_experts,topk", GENERIC_PADDING_SCENARIOS)
@pytest.mark.parametrize("padding_type", ["head", "tail", "mixed", "all"])
def test_topk_softmax_generic_padding(num_experts, topk, padding_type):
    is_padding = torch.zeros(16, device="cuda", dtype=torch.bool)

    if padding_type == "head":
        is_padding[:4] = True
    elif padding_type == "tail":
        is_padding[-4:] = True
    elif padding_type == "mixed":
        is_padding[1] = True
        is_padding[5] = True
        is_padding[9] = True
        is_padding[14] = True
    elif padding_type == "all":
        is_padding[:] = True

    _run_case(16, num_experts, topk, dtype=torch.float16, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=-1, check_padding_weights=False, check_padding_source_rows=False)


# ============================================================
# Generic padding + renormalize
# ============================================================

@pytest.mark.parametrize("num_experts,topk", GENERIC_PADDING_SCENARIOS)
@pytest.mark.parametrize("padding_type", ["head", "tail", "mixed", "all"])
def test_topk_softmax_generic_padding_renormalize(num_experts, topk, padding_type):
    is_padding = torch.zeros(16, device="cuda", dtype=torch.bool)

    if padding_type == "head":
        is_padding[:4] = True
    elif padding_type == "tail":
        is_padding[-4:] = True
    elif padding_type == "mixed":
        is_padding[1] = True
        is_padding[5] = True
        is_padding[9] = True
        is_padding[14] = True
    elif padding_type == "all":
        is_padding[:] = True

    _run_case(16, num_experts, topk, dtype=torch.float16, renormalize=True, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=-1, check_padding_weights=False, check_padding_source_rows=False)


# ============================================================
# CommonOpt + padding
# ============================================================

@pytest.mark.parametrize("num_experts,topk", COMMON_OPT_SCENARIOS)
def test_topk_softmax_common_opt_padding(num_experts, topk):
    is_padding = torch.tensor([False, True, False, True, False, False, True, False], device="cuda", dtype=torch.bool)

    _run_case(8, num_experts, topk, dtype=torch.float16, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=num_experts)


# ============================================================
# CommonOpt + bias + padding + renormalize
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(96, 1), (96, 8), (96, 16), (224, 1), (224, 8), (224, 16)])
def test_topk_softmax_common_opt_all_features(num_experts, topk):
    torch.manual_seed(1357)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)

    is_padding = torch.tensor([False, True, False, True, False, False, True, False], device="cuda", dtype=torch.bool)

    _run_case(8, num_experts, topk, dtype=torch.float16, renormalize=True, bias=bias, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=num_experts)


# ============================================================
# Optimized path padding
# ============================================================

@pytest.mark.parametrize("num_experts,topk", OPTIMIZED_PADDING_SCENARIOS)
@pytest.mark.parametrize("padding_type", ["head", "tail", "mixed", "all"])
def test_topk_softmax_optimized_padding(num_experts, topk, padding_type):
    is_padding = torch.zeros(16, device="cuda", dtype=torch.bool)

    if padding_type == "head":
        is_padding[:4] = True
    elif padding_type == "tail":
        is_padding[-4:] = True
    elif padding_type == "mixed":
        is_padding[1] = True
        is_padding[5] = True
        is_padding[9] = True
        is_padding[14] = True
    elif padding_type == "all":
        is_padding[:] = True

    _run_case(16, num_experts, topk, dtype=torch.float16, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=num_experts)


# ============================================================
# Optimized path padding + renormalize
# ============================================================

@pytest.mark.parametrize("num_experts,topk", OPTIMIZED_PADDING_SCENARIOS)
@pytest.mark.parametrize("padding_type", ["head", "tail", "mixed", "all"])
def test_topk_softmax_optimized_padding_renormalize(num_experts, topk, padding_type):
    is_padding = torch.zeros(16, device="cuda", dtype=torch.bool)

    if padding_type == "head":
        is_padding[:4] = True
    elif padding_type == "tail":
        is_padding[-4:] = True
    elif padding_type == "mixed":
        is_padding[1] = True
        is_padding[5] = True
        is_padding[9] = True
        is_padding[14] = True
    elif padding_type == "all":
        is_padding[:] = True

    _run_case(16, num_experts, topk, dtype=torch.float16, renormalize=True, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=num_experts)


# ============================================================
# Topk edges
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(1, 1), (2, 1), (4, 1), (8, 1), (8, 2), (8, 4), (8, 8), (16, 1), (16, 8), (16, 16), (32, 1), (32, 16), (32, 32), (64, 1), (64, 8), (64, 32), (64, 64), (96, 1), (96, 16), (128, 1), (128, 16), (224, 1), (224, 16), (256, 1), (256, 8), (512, 1), (512, 8), (896, 1), (896, 16)])
def test_topk_softmax_topk_edges(num_experts, topk):
    _run_case(8, num_experts, topk, dtype=torch.float16, rtol=8e-3, atol=8e-3)


# ============================================================
# Small tokens
# ============================================================

@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5, 7, 8, 16, 32])
@pytest.mark.parametrize("num_experts,topk", [(96, 8), (128, 8), (224, 8), (256, 8), (512, 8)])
def test_topk_softmax_small_tokens(num_experts, topk, num_tokens):
    _run_case(num_tokens, num_experts, topk, dtype=torch.float16, rtol=8e-3, atol=8e-3)


# ============================================================
# Token boundaries
# ============================================================

@pytest.mark.parametrize("num_tokens", [31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257])
@pytest.mark.parametrize("num_experts,topk", [(96, 8), (128, 8), (224, 8), (256, 8)])
def test_topk_softmax_token_boundaries(num_experts, topk, num_tokens):
    _run_case(num_tokens, num_experts, topk, dtype=torch.float16, rtol=8e-3, atol=8e-3)


# ============================================================
# 128 decode boundary
# ============================================================

@pytest.mark.parametrize("num_tokens", [1, 2, 3, 7, 8, 16, 31, 32, 63, 64, 127, 128, 255, 256, 512, 1023])
def test_topk_softmax_128_decode_boundaries(num_tokens):
    _run_case(num_tokens, 128, 8, dtype=torch.float16, rtol=8e-3, atol=8e-3)


@pytest.mark.parametrize("num_tokens", [1023, 1024, 1025])
def test_topk_softmax_128_decode_threshold(num_tokens):
    _run_case(num_tokens, 128, 8, dtype=torch.float16, rtol=8e-3, atol=8e-3)


# ============================================================
# 128 decode bias
# ============================================================

@pytest.mark.parametrize("bias_type", ["positive", "negative", "random"])
def test_topk_softmax_128_decode_bias(bias_type):
    num_experts = 128
    topk = 8

    if bias_type == "positive":
        bias = torch.linspace(0.0, 1.0, num_experts, device="cuda", dtype=torch.float32)
    elif bias_type == "negative":
        bias = torch.linspace(0.0, -1.0, num_experts, device="cuda", dtype=torch.float32)
    else:
        torch.manual_seed(5678)
        bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)

    _run_case(8, num_experts, topk, dtype=torch.float16, bias=bias, rtol=8e-3, atol=8e-3)


# ============================================================
# 128 decode strong bias
# ============================================================

def test_topk_softmax_128_decode_strong_bias():
    num_experts = 128
    topk = 8

    bias = torch.zeros(num_experts, device="cuda", dtype=torch.float32)
    bias[0] = 100.0
    bias[1] = 50.0
    bias[2] = -50.0

    _run_case(8, num_experts, topk, dtype=torch.float16, bias=bias, rtol=8e-3, atol=8e-3)


# ============================================================
# 128 decode bias + renormalize
# ============================================================

def test_topk_softmax_128_decode_bias_renormalize():
    num_experts = 128
    topk = 8

    torch.manual_seed(5678)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)

    _run_case(8, num_experts, topk, dtype=torch.float16, renormalize=True, bias=bias, rtol=8e-3, atol=8e-3)


# ============================================================
# 128 decode padding
# ============================================================

@pytest.mark.parametrize("padding_type", ["head", "tail", "mixed", "all"])
def test_topk_softmax_128_decode_padding(padding_type):
    is_padding = torch.zeros(16, device="cuda", dtype=torch.bool)

    if padding_type == "head":
        is_padding[:4] = True
    elif padding_type == "tail":
        is_padding[-4:] = True
    elif padding_type == "mixed":
        is_padding[1] = True
        is_padding[5] = True
        is_padding[9] = True
        is_padding[14] = True
    elif padding_type == "all":
        is_padding[:] = True

    _run_case(16, 128, 8, dtype=torch.float16, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=128)


# ============================================================
# 128 decode bias + padding
# ============================================================

@pytest.mark.parametrize("padding_type", ["head", "tail", "mixed", "all"])
def test_topk_softmax_128_decode_bias_padding(padding_type):
    num_experts = 128
    topk = 8

    torch.manual_seed(6789)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)

    is_padding = torch.zeros(16, device="cuda", dtype=torch.bool)

    if padding_type == "head":
        is_padding[:4] = True
    elif padding_type == "tail":
        is_padding[-4:] = True
    elif padding_type == "mixed":
        is_padding[1] = True
        is_padding[5] = True
        is_padding[9] = True
        is_padding[14] = True
    elif padding_type == "all":
        is_padding[:] = True

    _run_case(16, num_experts, topk, dtype=torch.float16, bias=bias, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=128)


# ============================================================
# 128 decode bias + padding + renormalize
# ============================================================

@pytest.mark.parametrize("padding_type", ["head", "tail", "mixed", "all"])
def test_topk_softmax_128_decode_bias_padding_renormalize(padding_type):
    num_experts = 128
    topk = 8

    torch.manual_seed(6789)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)

    is_padding = torch.zeros(16, device="cuda", dtype=torch.bool)

    if padding_type == "head":
        is_padding[:4] = True
    elif padding_type == "tail":
        is_padding[-4:] = True
    elif padding_type == "mixed":
        is_padding[1] = True
        is_padding[5] = True
        is_padding[9] = True
        is_padding[14] = True
    elif padding_type == "all":
        is_padding[:] = True

    _run_case(16, num_experts, topk, dtype=torch.float16, renormalize=True, bias=bias, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=128)


# ============================================================
# 256 decode boundaries
# ============================================================

@pytest.mark.parametrize("num_tokens", [1, 2, 3, 7, 8, 16, 31, 32, 63, 64, 127, 128, 255, 256, 512, 1023])
def test_topk_softmax_256_decode_boundaries(num_tokens):
    _run_case(num_tokens, 256, 8, dtype=torch.float16, rtol=8e-3, atol=8e-3)


# ============================================================
# 256 decode padding
# ============================================================

@pytest.mark.parametrize("padding_type", ["head", "tail", "mixed", "all"])
def test_topk_softmax_256_decode_padding(padding_type):
    is_padding = torch.zeros(16, device="cuda", dtype=torch.bool)

    if padding_type == "head":
        is_padding[:4] = True
    elif padding_type == "tail":
        is_padding[-4:] = True
    elif padding_type == "mixed":
        is_padding[1] = True
        is_padding[5] = True
        is_padding[9] = True
        is_padding[14] = True
    elif padding_type == "all":
        is_padding[:] = True

    _run_case(16, 256, 8, dtype=torch.float16, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=256)


# ============================================================
# 256 decode padding + renormalize
# ============================================================

@pytest.mark.parametrize("padding_type", ["head", "tail", "mixed", "all"])
def test_topk_softmax_256_decode_padding_renormalize(padding_type):
    is_padding = torch.zeros(16, device="cuda", dtype=torch.bool)

    if padding_type == "head":
        is_padding[:4] = True
    elif padding_type == "tail":
        is_padding[-4:] = True
    elif padding_type == "mixed":
        is_padding[1] = True
        is_padding[5] = True
        is_padding[9] = True
        is_padding[14] = True
    elif padding_type == "all":
        is_padding[:] = True

    _run_case(16, 256, 8, dtype=torch.float16, renormalize=True, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=256)


# ============================================================
# All padding optimized
# ============================================================

@pytest.mark.parametrize("num_experts,topk", OPTIMIZED_PADDING_SCENARIOS)
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_softmax_all_padding_optimized(num_experts, topk, renormalize):
    is_padding = torch.ones(8, device="cuda", dtype=torch.bool)

    _run_case(8, num_experts, topk, dtype=torch.float16, renormalize=renormalize, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=num_experts)


# ============================================================
# All padding generic
# ============================================================

@pytest.mark.parametrize("num_experts,topk", GENERIC_PADDING_SCENARIOS)
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_softmax_all_padding_generic(num_experts, topk, renormalize):
    is_padding = torch.ones(8, device="cuda", dtype=torch.bool)

    _run_case(8, num_experts, topk, dtype=torch.float16, renormalize=renormalize, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=-1, check_padding_weights=False, check_padding_source_rows=False)


# ============================================================
# Zero bias
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(96, 8), (128, 8), (160, 8), (192, 8), (224, 8), (256, 8), (288, 8), (320, 8), (384, 8), (448, 8), (512, 8), (576, 8), (896, 16)])
def test_topk_softmax_zero_bias(num_experts, topk):
    bias = torch.zeros(num_experts, device="cuda", dtype=torch.float32)

    _run_case(8, num_experts, topk, dtype=torch.float16, bias=bias, rtol=8e-3, atol=8e-3)


# ============================================================
# Generic bias
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(520, 8), (640, 8), (768, 8), (1000, 8)])
def test_topk_softmax_generic_bias(num_experts, topk):
    torch.manual_seed(2468)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)

    _run_case(8, num_experts, topk, dtype=torch.float16, bias=bias, rtol=8e-3, atol=8e-3)


# ============================================================
# Generic bias + renormalize
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(520, 8), (640, 8), (768, 8), (1000, 8)])
def test_topk_softmax_generic_bias_renormalize(num_experts, topk):
    torch.manual_seed(2468)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)

    _run_case(8, num_experts, topk, dtype=torch.float16, renormalize=True, bias=bias, rtol=8e-3, atol=8e-3)


# ============================================================
# Generic bias + padding
# ============================================================

@pytest.mark.parametrize("num_experts,topk", GENERIC_PADDING_SCENARIOS)
def test_topk_softmax_generic_bias_padding(num_experts, topk):
    torch.manual_seed(2468)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)

    is_padding = torch.zeros(16, device="cuda", dtype=torch.bool)
    is_padding[1] = True
    is_padding[5] = True
    is_padding[9] = True
    is_padding[14] = True

    _run_case(16, num_experts, topk, dtype=torch.float16, bias=bias, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=-1, check_padding_weights=False, check_padding_source_rows=False)


# ============================================================
# Generic bias + padding + renormalize
# ============================================================

@pytest.mark.parametrize("num_experts,topk", GENERIC_PADDING_SCENARIOS)
def test_topk_softmax_generic_bias_padding_renormalize(num_experts, topk):
    torch.manual_seed(2468)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)

    is_padding = torch.zeros(16, device="cuda", dtype=torch.bool)
    is_padding[1] = True
    is_padding[5] = True
    is_padding[9] = True
    is_padding[14] = True

    _run_case(16, num_experts, topk, dtype=torch.float16, renormalize=True, bias=bias, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=-1, check_padding_weights=False, check_padding_source_rows=False)


# ============================================================
# Renormalize sum
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(96, 8), (128, 8), (224, 8), (256, 8), (512, 8), (896, 16)])
def test_topk_softmax_renormalize_sum(num_experts, topk):
    num_tokens = 16

    gating_output = _make_input(num_tokens, num_experts, torch.float16)

    topk_weights = torch.empty((num_tokens, topk), device="cuda", dtype=torch.float32)
    topk_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)
    token_expert_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)

    _topk_softmax(gating_output, topk_weights, topk_indices, token_expert_indices, True, None, None)

    torch.cuda.synchronize()

    sums = topk_weights.sum(dim=-1)

    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=8e-3, atol=8e-3)


# ============================================================
# Renormalize sum with bias
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(96, 8), (128, 8), (224, 8), (256, 8), (512, 8)])
def test_topk_softmax_renormalize_sum_bias(num_experts, topk):
    num_tokens = 16

    torch.manual_seed(12345)
    gating_output = _make_input(num_tokens, num_experts, torch.float16)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)

    topk_weights = torch.empty((num_tokens, topk), device="cuda", dtype=torch.float32)
    topk_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)
    token_expert_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)

    _topk_softmax(gating_output, topk_weights, topk_indices, token_expert_indices, True, bias, None)

    torch.cuda.synchronize()

    sums = topk_weights.sum(dim=-1)

    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=8e-3, atol=8e-3)


# ============================================================
# token_expert_indices
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(96, 8), (128, 8), (224, 8), (256, 8), (512, 8), (896, 16), (520, 8), (640, 8), (1000, 8)])
def test_topk_softmax_token_expert_indices(num_experts, topk):
    num_tokens = 17

    gating_output = _make_input(num_tokens, num_experts, torch.float16)

    topk_weights = torch.empty((num_tokens, topk), device="cuda", dtype=torch.float32)
    topk_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)
    token_expert_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)

    _topk_softmax(gating_output, topk_weights, topk_indices, token_expert_indices, False, None, None)

    torch.cuda.synchronize()

    _check_source_rows(token_expert_indices, num_tokens, topk)


# ============================================================
# Padding source rows
# ============================================================

@pytest.mark.parametrize("num_experts,topk,padding_index", [(96, 8, 96), (128, 8, 128), (224, 8, 224), (256, 8, 256), (512, 8, 512), (520, 8, -1), (640, 8, -1), (768, 8, -1), (1000, 8, -1)])
def test_topk_softmax_padding_source_rows(num_experts, topk, padding_index):
    num_tokens = 16

    gating_output = _make_input(num_tokens, num_experts, torch.float16)

    topk_weights = torch.empty((num_tokens, topk), device="cuda", dtype=torch.float32)
    topk_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)
    token_expert_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)

    is_padding = torch.zeros(num_tokens, device="cuda", dtype=torch.bool)
    is_padding[2] = True
    is_padding[7] = True
    is_padding[13] = True

    _topk_softmax(gating_output, topk_weights, topk_indices, token_expert_indices, True, None, is_padding)

    torch.cuda.synchronize()

    _check_source_rows(token_expert_indices, num_tokens, topk)

    if padding_index == -1:
        _check_padding_source_rows(token_expert_indices, is_padding, num_tokens, topk)
        assert torch.all(topk_indices[is_padding] == -1)
    else:
        _check_padding_weights(topk_weights, is_padding)
        assert torch.all(topk_indices[is_padding] == padding_index)


# ============================================================
# Large tokens
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(96, 8), (128, 8), (224, 8), (256, 8), (512, 8)])
@pytest.mark.parametrize("num_tokens", [256, 512, 1023, 1024, 1025])
def test_topk_softmax_large_tokens(num_experts, topk, num_tokens):
    _run_case(num_tokens, num_experts, topk, dtype=torch.float16, rtol=8e-3, atol=8e-3)


# ============================================================
# Large tokens + padding optimized
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(96, 8), (128, 8), (224, 8), (256, 8), (512, 8)])
@pytest.mark.parametrize("num_tokens", [256, 512, 1024])
def test_topk_softmax_large_tokens_padding(num_experts, topk, num_tokens):
    is_padding = torch.zeros(num_tokens, device="cuda", dtype=torch.bool)
    is_padding[::7] = True

    _run_case(num_tokens, num_experts, topk, dtype=torch.float16, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=num_experts)


# ============================================================
# Large tokens + padding generic
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(520, 8), (640, 8), (768, 8), (1000, 8)])
@pytest.mark.parametrize("num_tokens", [256, 512, 1024])
def test_topk_softmax_large_tokens_padding_generic(num_experts, topk, num_tokens):
    is_padding = torch.zeros(num_tokens, device="cuda", dtype=torch.bool)
    is_padding[::7] = True

    _run_case(num_tokens, num_experts, topk, dtype=torch.float16, is_padding=is_padding, rtol=8e-3, atol=8e-3, padding_index=-1, check_padding_weights=False, check_padding_source_rows=False)


# ============================================================
# Generic token boundaries
# ============================================================

@pytest.mark.parametrize("num_tokens", [1, 2, 3, 7, 8, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257])
@pytest.mark.parametrize("num_experts,topk", [(520, 8), (640, 8), (768, 8), (1000, 8)])
def test_topk_softmax_generic_token_boundaries(num_experts, topk, num_tokens):
    _run_case(num_tokens, num_experts, topk, dtype=torch.float16, rtol=8e-3, atol=8e-3)


# ============================================================
# Generic large topk
# ============================================================

@pytest.mark.parametrize("num_experts,topk", [(520, 16), (640, 16), (768, 16), (1000, 16)])
def test_topk_softmax_generic_topk16(num_experts, topk):
    _run_case(16, num_experts, topk, dtype=torch.float16, rtol=8e-3, atol=8e-3)


# ============================================================
# Generic all padding + source rows
# ============================================================

@pytest.mark.parametrize("num_experts,topk", GENERIC_PADDING_SCENARIOS)
def test_topk_softmax_generic_all_padding_source_rows(num_experts, topk):
    num_tokens = 17
    is_padding = torch.ones(num_tokens, device="cuda", dtype=torch.bool)

    gating_output = _make_input(num_tokens, num_experts, torch.float16)

    topk_weights = torch.empty((num_tokens, topk), device="cuda", dtype=torch.float32)
    topk_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)
    token_expert_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)

    _topk_softmax(gating_output, topk_weights, topk_indices, token_expert_indices, True, None, is_padding)

    torch.cuda.synchronize()

    assert torch.all(topk_indices == -1)

    assert topk_weights.shape == (num_tokens, topk)
    assert token_expert_indices.shape == (num_tokens, topk)
    assert torch.isfinite(topk_weights).all()


# ============================================================
# API
# ============================================================

def test_topk_softmax_api():
    num_tokens = 4
    num_experts = 96
    topk = 8

    gating_output = _make_input(num_tokens, num_experts, torch.float16)

    topk_weights = torch.empty((num_tokens, topk), device="cuda", dtype=torch.float32)
    topk_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)
    token_expert_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)

    _topk_softmax(gating_output, topk_weights, topk_indices, token_expert_indices, False, None, None)

    torch.cuda.synchronize()

    assert topk_weights.shape == (num_tokens, topk)
    assert topk_indices.shape == (num_tokens, topk)
    assert token_expert_indices.shape == (num_tokens, topk)

    assert topk_weights.dtype == torch.float32
    assert topk_indices.dtype == torch.int32
    assert token_expert_indices.dtype == torch.int32

    assert torch.isfinite(topk_weights).all()


# ============================================================
# API with bias and padding
# ============================================================

def test_topk_softmax_api_full_arguments():
    num_tokens = 8
    num_experts = 128
    topk = 8

    gating_output = _make_input(num_tokens, num_experts, torch.float16)

    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)
    is_padding = torch.tensor([False, True, False, False, True, False, False, True], device="cuda", dtype=torch.bool)

    topk_weights = torch.empty((num_tokens, topk), device="cuda", dtype=torch.float32)
    topk_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)
    token_expert_indices = torch.empty((num_tokens, topk), device="cuda", dtype=torch.int32)

    _topk_softmax(gating_output, topk_weights, topk_indices, token_expert_indices, True, bias, is_padding)

    torch.cuda.synchronize()

    assert topk_weights.shape == (num_tokens, topk)
    assert topk_indices.shape == (num_tokens, topk)
    assert token_expert_indices.shape == (num_tokens, topk)

    assert topk_weights.dtype == torch.float32
    assert topk_indices.dtype == torch.int32
    assert token_expert_indices.dtype == torch.int32

    _check_padding_weights(topk_weights, is_padding)
    assert torch.all(topk_indices[is_padding] == num_experts)