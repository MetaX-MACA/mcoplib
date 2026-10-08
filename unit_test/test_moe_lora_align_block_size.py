# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

import mcoplib._moe_C  # noqa: F401


DEVICE = "cuda"

SENTINEL_EXPERT = -2
SENTINEL_TOKEN = -7
SENTINEL_NPAD = -13


def round_up(x, base):
    return ((x + base - 1) // base) * base


def ceil_div(x, y):
    return (x + y - 1) // y


def _require_op():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    if not hasattr(torch.ops._moe_C, "moe_lora_align_block_size"):
        pytest.skip("moe_lora_align_block_size was not built")


def sample_data(num_experts, max_loras, num_tokens, topk_num, seed=1):
    random.seed(seed)

    topk_ids = torch.empty((num_tokens, topk_num), dtype=torch.int32)
    token_lora_mapping = torch.empty((num_tokens,), dtype=torch.int32)

    for i in range(num_tokens):
        pool = list(range(num_experts))
        random.shuffle(pool)

        for j in range(topk_num):
            topk_ids[i, j] = pool[j]

        token_lora_mapping[i] = random.randint(0, max_loras - 1)

    return topk_ids.to(DEVICE), token_lora_mapping.to(DEVICE)


def get_max_num_tokens_padded(total_elements, num_experts, block_size):
    max_num_tokens_padded = total_elements + num_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)

    if total_elements < num_experts:
        max_num_tokens_padded = total_elements * block_size

    return max_num_tokens_padded


def _run_op(topk_ids, token_lora_mapping, num_experts, block_size, max_loras, max_num_tokens_padded, max_num_m_blocks, sorted_token_ids, expert_ids, num_tokens_post_pad, adapter_enabled, lora_ids):
    torch.ops._moe_C.moe_lora_align_block_size(topk_ids, token_lora_mapping, num_experts, block_size, max_loras, max_num_tokens_padded, max_num_m_blocks, sorted_token_ids, expert_ids, num_tokens_post_pad, adapter_enabled, lora_ids, None)
    torch.cuda.synchronize()


def _make_lora_ids(max_loras, active_ids=None):
    if active_ids is None:
        active_ids = list(range(max_loras))

    lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32, device=DEVICE)

    if active_ids:
        values = torch.tensor(active_ids, dtype=torch.int32, device=DEVICE)
        lora_ids[1:1 + len(active_ids)] = values

    return lora_ids


def _make_case(num_tokens, topk_num, num_experts, max_loras, block_size, seed=1, active_lora_ids=None, disabled_slots=()):
    topk_ids, token_lora_mapping = sample_data(num_experts, max_loras, num_tokens, topk_num, seed)

    total_elements = topk_ids.numel()

    max_num_tokens_padded = get_max_num_tokens_padded(total_elements, num_experts, block_size)
    max_num_m_blocks = ceil_div(max_num_tokens_padded, block_size)

    sorted_token_ids = torch.full((max_loras * max_num_tokens_padded,), total_elements, dtype=torch.int32, device=DEVICE)
    expert_ids = torch.full((max_loras * max_num_m_blocks,), -1, dtype=torch.int32, device=DEVICE)
    num_tokens_post_pad = torch.zeros((max_loras,), dtype=torch.int32, device=DEVICE)

    adapter_enabled = torch.ones((max_loras + 1,), dtype=torch.int32, device=DEVICE)

    for slot in disabled_slots:
        adapter_enabled[slot] = 0

    lora_ids = _make_lora_ids(max_loras, active_lora_ids)

    _run_op(topk_ids, token_lora_mapping, num_experts, block_size, max_loras, max_num_tokens_padded, max_num_m_blocks, sorted_token_ids, expert_ids, num_tokens_post_pad, adapter_enabled, lora_ids)

    return {
        "topk_ids": topk_ids,
        "token_lora_mapping": token_lora_mapping,
        "sorted_token_ids": sorted_token_ids.view(max_loras, max_num_tokens_padded),
        "expert_ids": expert_ids.view(max_loras, max_num_m_blocks),
        "num_tokens_post_pad": num_tokens_post_pad,
        "adapter_enabled": adapter_enabled,
        "lora_ids": lora_ids,
        "max_num_tokens_padded": max_num_tokens_padded,
        "max_num_m_blocks": max_num_m_blocks,
        "block_size": block_size,
        "max_loras": max_loras,
        "num_experts": num_experts,
    }


def _make_sentinel_case(*, num_lora_tokens, num_base_tokens, max_loras, num_experts=64, topk_num=6, block_size=16, lora_ids_override=None, disabled_slots=(), seed=1):
    random.seed(seed)

    num_tokens = num_lora_tokens + num_base_tokens
    assert num_tokens > 0

    topk_ids = torch.empty((num_tokens, topk_num), dtype=torch.int32)
    token_lora_mapping = torch.empty((num_tokens,), dtype=torch.int32)

    for i in range(num_tokens):
        pool = list(range(num_experts))
        random.shuffle(pool)

        for j in range(topk_num):
            topk_ids[i, j] = pool[j]

        token_lora_mapping[i] = 0 if i < num_lora_tokens else -1

    topk_ids = topk_ids.to(DEVICE)
    token_lora_mapping = token_lora_mapping.to(DEVICE)

    total_elements = topk_ids.numel()

    max_num_tokens_padded = get_max_num_tokens_padded(total_elements, num_experts, block_size)
    max_num_m_blocks = ceil_div(max_num_tokens_padded, block_size)

    if lora_ids_override is None:
        lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32, device=DEVICE)
        unique_ids = torch.unique(token_lora_mapping, sorted=True)
        lora_ids[:unique_ids.numel()] = unique_ids.to(dtype=torch.int32)
    else:
        assert lora_ids_override.numel() == max_loras + 1
        lora_ids = lora_ids_override.to(dtype=torch.int32, device=DEVICE)

    adapter_enabled = torch.ones((max_loras + 1,), dtype=torch.int32, device=DEVICE)

    for slot in disabled_slots:
        adapter_enabled[slot] = 0

    sorted_token_ids = torch.full((max_loras * max_num_tokens_padded,), SENTINEL_TOKEN, dtype=torch.int32, device=DEVICE)
    expert_ids = torch.full((max_loras * max_num_m_blocks,), SENTINEL_EXPERT, dtype=torch.int32, device=DEVICE)
    num_tokens_post_pad = torch.full((max_loras,), SENTINEL_NPAD, dtype=torch.int32, device=DEVICE)

    _run_op(topk_ids, token_lora_mapping, num_experts, block_size, max_loras, max_num_tokens_padded, max_num_m_blocks, sorted_token_ids, expert_ids, num_tokens_post_pad, adapter_enabled, lora_ids)

    return {
        "topk_ids": topk_ids,
        "token_lora_mapping": token_lora_mapping,
        "sorted_token_ids": sorted_token_ids.view(max_loras, max_num_tokens_padded),
        "expert_ids": expert_ids.view(max_loras, max_num_m_blocks),
        "num_tokens_post_pad": num_tokens_post_pad,
        "adapter_enabled": adapter_enabled,
        "lora_ids": lora_ids,
        "max_num_tokens_padded": max_num_tokens_padded,
        "max_num_m_blocks": max_num_m_blocks,
        "block_size": block_size,
        "max_loras": max_loras,
        "num_experts": num_experts,
    }


def _check_lora_result(out, lora_id):
    topk_ids = out["topk_ids"]
    token_lora_mapping = out["token_lora_mapping"]
    sorted_token_ids = out["sorted_token_ids"]
    expert_ids = out["expert_ids"]
    num_tokens_post_pad = out["num_tokens_post_pad"]

    block_size = out["block_size"]
    num_experts = out["num_experts"]

    total_elements = topk_ids.numel()

    mapping_mask = token_lora_mapping == lora_id
    expected_topk = topk_ids[mapping_mask].flatten()

    expert_counts = torch.bincount(expected_topk.cpu(), minlength=num_experts)

    expected_post_pad = int(round_up(expert_counts, block_size).sum().item())
    actual_post_pad = int(num_tokens_post_pad[lora_id].item())

    assert actual_post_pad == expected_post_pad, f"LoRA {lora_id}: expected num_tokens_post_pad {expected_post_pad}, got {actual_post_pad}"

    assert actual_post_pad % block_size == 0

    num_blocks = actual_post_pad // block_size

    if expected_topk.numel() == 0:
        assert actual_post_pad == 0
        return

    flat_topk = topk_ids.flatten()

    for block_idx in range(num_blocks):
        expert_id = int(expert_ids[lora_id, block_idx].item())

        assert 0 <= expert_id < num_experts, f"LoRA {lora_id}: invalid expert_id={expert_id} at block={block_idx}"

        start = block_idx * block_size
        end = start + block_size

        block_tokens = sorted_token_ids[lora_id, start:end]

        # topk_ids.numel() is the official padding token index.
        # Padding entries are legal and must not be dereferenced.
        valid_tokens = block_tokens[block_tokens != total_elements]

        if valid_tokens.numel() == 0:
            continue

        assert torch.all(valid_tokens >= 0), f"LoRA {lora_id}: negative sorted_token_ids found"

        assert torch.all(valid_tokens < total_elements), f"LoRA {lora_id}: sorted_token_ids contains out-of-range token index"

        assert torch.all(flat_topk[valid_tokens] == expert_id), f"LoRA {lora_id}: block {block_idx} contains tokens belonging to different experts"

    if num_blocks < expert_ids.size(1):
        tail = expert_ids[lora_id, num_blocks:]
        assert torch.all(tail == -1), f"LoRA {lora_id}: inactive expert_ids tail was modified"

    used_tokens = sorted_token_ids[lora_id, :actual_post_pad]

    # Legal values are either a real token index [0, total_elements)
    # or the padding index total_elements.
    assert torch.all((used_tokens >= 0) & (used_tokens <= total_elements)), f"LoRA {lora_id}: sorted_token_ids contains invalid token index"


@pytest.mark.parametrize("num_tokens", [100, 200, 1024, 4096])
@pytest.mark.parametrize("topk_num", [6])
@pytest.mark.parametrize("num_experts", [64, 128, 256, 512])
@pytest.mark.parametrize("max_loras", [2, 32])
@pytest.mark.parametrize("block_size", [16])
def test_moe_lora_align_block_size(num_tokens, topk_num, num_experts, max_loras, block_size):
    _require_op()

    out = _make_case(num_tokens, topk_num, num_experts, max_loras, block_size)

    for lora_id in range(max_loras):
        _check_lora_result(out, lora_id)


def test_moe_lora_align_block_size_large_path_counts():
    _require_op()

    out = _make_case(100, 12, 64, 4, 16)

    for lora_id in range(4):
        _check_lora_result(out, lora_id)


@pytest.mark.parametrize("max_loras", [1, 2])
def test_moe_lora_align_block_size_mixed_base_and_lora(max_loras):
    _require_op()

    out = _make_sentinel_case(num_lora_tokens=8, num_base_tokens=8, max_loras=max_loras)

    assert out["lora_ids"][0].item() == -1

    real_slot = 0

    post_pad = int(out["num_tokens_post_pad"][real_slot].item())

    assert post_pad != SENTINEL_NPAD, "num_tokens_post_pad[0] was never written; the real LoRA slot was skipped"

    assert 0 < post_pad <= out["max_num_tokens_padded"]
    assert post_pad % out["block_size"] == 0

    expert_row = out["expert_ids"][real_slot]

    assert torch.all(expert_row != SENTINEL_EXPERT), "expert_ids row for the real LoRA slot contains unwritten sentinel values"

    sorted_row = out["sorted_token_ids"][real_slot]

    assert torch.all(sorted_row != SENTINEL_TOKEN), "sorted_token_ids row for the real LoRA slot contains unwritten sentinel values"


def test_moe_lora_align_block_size_disabled_adapter_untouched():
    _require_op()

    max_loras = 1

    out = _make_sentinel_case(num_lora_tokens=16, num_base_tokens=0, max_loras=max_loras, disabled_slots=(0,))

    assert (out["lora_ids"] == 0).any().item()

    assert out["num_tokens_post_pad"][0].item() == SENTINEL_NPAD, "num_tokens_post_pad[0] was modified for a disabled adapter"

    expert_row = out["expert_ids"][0]

    assert torch.all(expert_row == SENTINEL_EXPERT), "expert_ids for disabled adapter was modified"

    sorted_row = out["sorted_token_ids"][0]

    assert torch.all(sorted_row == SENTINEL_TOKEN), "sorted_token_ids for disabled adapter was modified"


def test_moe_lora_align_block_size_nontrivial_lora_ids():
    _require_op()

    max_loras = 4

    out = _make_case(256, 6, 64, max_loras, 16, active_lora_ids=[0, 2, 3])

    assert out["lora_ids"].tolist() == [-1, 0, 2, 3, -1]

    _check_lora_result(out, 0)
    _check_lora_result(out, 2)
    _check_lora_result(out, 3)

    assert out["num_tokens_post_pad"][1].item() == 0


def test_moe_lora_align_block_size_empty_lora_slots():
    _require_op()

    max_loras = 4

    out = _make_case(256, 6, 64, max_loras, 16, active_lora_ids=[0])

    assert out["lora_ids"].tolist() == [-1, 0, -1, -1, -1]

    _check_lora_result(out, 0)

    assert out["num_tokens_post_pad"][1].item() == 0
    assert out["num_tokens_post_pad"][2].item() == 0
    assert out["num_tokens_post_pad"][3].item() == 0


def test_moe_lora_align_block_size_topk_2():
    _require_op()

    max_loras = 4

    out = _make_case(4096, 2, 64, max_loras, 16)

    for lora_id in range(max_loras):
        _check_lora_result(out, lora_id)


def test_moe_lora_align_block_size_topk_4():
    _require_op()

    max_loras = 4

    out = _make_case(4096, 4, 64, max_loras, 16)

    for lora_id in range(max_loras):
        _check_lora_result(out, lora_id)


def test_moe_lora_align_block_size_large():
    _require_op()

    max_loras = 4

    out = _make_case(4096, 6, 256, max_loras, 16)

    for lora_id in range(max_loras):
        _check_lora_result(out, lora_id)


def test_moe_lora_align_block_size_large_topk_2():
    _require_op()

    max_loras = 4

    out = _make_case(4096, 2, 64, max_loras, 16)

    for lora_id in range(max_loras):
        _check_lora_result(out, lora_id)


def test_moe_lora_align_block_size_block_size_1():
    _require_op()

    out = _make_case(256, 6, 64, 4, 1)

    for lora_id in range(4):
        _check_lora_result(out, lora_id)


def test_moe_lora_align_block_size_block_size_32():
    _require_op()

    out = _make_case(1024, 6, 64, 4, 32)

    for lora_id in range(4):
        _check_lora_result(out, lora_id)


def test_moe_lora_align_block_size_multiple_disabled_adapters():
    _require_op()

    max_loras = 4

    lora_ids_override = torch.tensor([0, 1, 2, 3, -1], dtype=torch.int32, device=DEVICE)

    out = _make_sentinel_case(num_lora_tokens=256, num_base_tokens=0, max_loras=max_loras, lora_ids_override=lora_ids_override, disabled_slots=(1, 3))

    _check_lora_result(out, 0)
    _check_lora_result(out, 2)

    assert out["num_tokens_post_pad"][1].item() == SENTINEL_NPAD
    assert out["num_tokens_post_pad"][3].item() == SENTINEL_NPAD

    assert torch.all(out["expert_ids"][1] == SENTINEL_EXPERT)
    assert torch.all(out["expert_ids"][3] == SENTINEL_EXPERT)

    assert torch.all(out["sorted_token_ids"][1] == SENTINEL_TOKEN)
    assert torch.all(out["sorted_token_ids"][3] == SENTINEL_TOKEN)


def test_moe_lora_align_block_size_lora_id_oob_guard():
    _require_op()

    max_loras = 1
    num_experts = 64
    topk_num = 6
    block_size = 16
    num_tokens = 16

    topk_ids, token_lora_mapping = sample_data(num_experts, max_loras, num_tokens, topk_num)

    max_num_tokens_padded = get_max_num_tokens_padded(topk_ids.numel(), num_experts, block_size)
    max_num_m_blocks = ceil_div(max_num_tokens_padded, block_size)

    sorted_token_ids = torch.full((max_loras * max_num_tokens_padded,), SENTINEL_TOKEN, dtype=torch.int32, device=DEVICE)
    expert_ids = torch.full((max_loras * max_num_m_blocks,), SENTINEL_EXPERT, dtype=torch.int32, device=DEVICE)
    num_tokens_post_pad = torch.full((max_loras,), SENTINEL_NPAD, dtype=torch.int32, device=DEVICE)

    adapter_enabled = torch.ones((max_loras + 1,), dtype=torch.int32, device=DEVICE)

    # Slot 0 is a valid LoRA ID.
    # Slot 1 intentionally uses an invalid LoRA ID to test the OOB guard.
    lora_ids = torch.tensor([0, 5], dtype=torch.int32, device=DEVICE)

    _run_op(topk_ids, token_lora_mapping, num_experts, block_size, max_loras, max_num_tokens_padded, max_num_m_blocks, sorted_token_ids, expert_ids, num_tokens_post_pad, adapter_enabled, lora_ids)

    assert num_tokens_post_pad[0].item() != SENTINEL_NPAD, "real LoRA slot 0 was skipped by the align kernel"


def test_moe_lora_align_block_size_empty_token_set():
    _require_op()

    max_loras = 2
    num_tokens = 8
    num_experts = 64
    topk_num = 6
    block_size = 16

    topk_ids = torch.zeros((num_tokens, topk_num), dtype=torch.int32, device=DEVICE)

    # Every token belongs to the base model.
    token_lora_mapping = torch.full((num_tokens,), -1, dtype=torch.int32, device=DEVICE)

    max_num_tokens_padded = get_max_num_tokens_padded(topk_ids.numel(), num_experts, block_size)
    max_num_m_blocks = ceil_div(max_num_tokens_padded, block_size)

    sorted_token_ids = torch.full((max_loras * max_num_tokens_padded,), SENTINEL_TOKEN, dtype=torch.int32, device=DEVICE)
    expert_ids = torch.full((max_loras * max_num_m_blocks,), SENTINEL_EXPERT, dtype=torch.int32, device=DEVICE)
    num_tokens_post_pad = torch.full((max_loras,), SENTINEL_NPAD, dtype=torch.int32, device=DEVICE)

    adapter_enabled = torch.ones((max_loras + 1,), dtype=torch.int32, device=DEVICE)

    lora_ids = _make_lora_ids(max_loras)

    _run_op(topk_ids, token_lora_mapping, num_experts, block_size, max_loras, max_num_tokens_padded, max_num_m_blocks, sorted_token_ids, expert_ids, num_tokens_post_pad, adapter_enabled, lora_ids)

    sorted_token_ids = sorted_token_ids.view(max_loras, max_num_tokens_padded)
    expert_ids = expert_ids.view(max_loras, max_num_m_blocks)

    for lora_id in range(max_loras):
        assert num_tokens_post_pad[lora_id].item() == 0
        assert torch.all(expert_ids[lora_id] == -1)
        assert torch.all(sorted_token_ids[lora_id] == topk_ids.numel())


if __name__ == "__main__":
    pytest.main([__file__])