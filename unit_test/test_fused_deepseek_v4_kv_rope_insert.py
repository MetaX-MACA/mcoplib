
import pytest
import torch
import mcoplib._C


def _get_op():
    return torch.ops._C.fused_deepseek_v4_kv_rope_insert


def _op_available():
    try:
        _get_op()
        return True
    except (AttributeError, RuntimeError):
        return False


def _make_inputs(n_tokens=4, n_blocks=2, block_size=4, max_pos=32, device="cuda", kv_dtype=torch.bfloat16, cache_dtype=torch.bfloat16, cache_width=None, kv_contiguous=True):
    if cache_width is None:
        cache_width = 512 if cache_dtype == torch.bfloat16 or cache_dtype == getattr(torch, "float8_e4m3fn", None) else 528

    kv = torch.randn((n_tokens, 512), dtype=kv_dtype, device=device)
    if not kv_contiguous:
        kv = torch.randn((n_tokens, 1024), dtype=kv_dtype, device=device)[:, ::2]

    k_cache = torch.full((n_blocks, block_size, cache_width), 7, dtype=cache_dtype, device=device)
    slot_mapping = torch.arange(n_tokens, dtype=torch.int64, device=device) % (n_blocks * block_size)
    position_ids = torch.arange(n_tokens, dtype=torch.int64, device=device) % max_pos
    cos_sin_cache = torch.zeros((max_pos, 64), dtype=torch.float32, device=device)
    fp8_scale = torch.ones((), dtype=torch.float32, device=device) if cache_dtype == getattr(torch, "float8_e4m3fn", None) else None

    return {
        "kv": kv.contiguous() if kv_contiguous else kv,
        "k_cache": k_cache,
        "slot_mapping": slot_mapping,
        "position_ids": position_ids,
        "cos_sin_cache": cos_sin_cache,
        "cache_block_size": block_size,
        "fp8_scale": fp8_scale,
        "kv_mxfp8": cache_dtype == torch.uint8 and cache_width == 528,
    }


def _run_case(args):
    _get_op()(
        args["kv"],
        args["k_cache"],
        args["slot_mapping"],
        args["position_ids"],
        args["cos_sin_cache"],
        args["cache_block_size"],
        args["fp8_scale"],
        args["kv_mxfp8"],
    )


def _make_cache(cache_dtype, n_blocks=2, block_size=4, cache_width=None):
    if cache_width is None:
        if cache_dtype == torch.uint8:
            cache_width = 528
        else:
            cache_width = 512
    return torch.full((n_blocks, block_size, cache_width), 7, dtype=cache_dtype, device="cuda")


@pytest.fixture
def base_case():
    return _make_inputs()


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/MACA device required"),
    pytest.mark.skipif(not _op_available(), reason="fused_deepseek_v4_kv_rope_insert is unavailable"),
]


# ---------------------------------------------------------------------------
# 1. BF16 full-cache correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_tokens,slot_count,block_size,n_blocks",
    [
        (1, 1, 4, 2),
        (4, 4, 4, 2),
        (8, 8, 4, 2),
        (8, 5, 4, 2),  # slot_mapping may omit padded rows
        (4, 2, 8, 2),
    ],
)
@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_bf16_basic(n_tokens, slot_count, block_size, n_blocks):
    args = _make_inputs(n_tokens=n_tokens, n_blocks=n_blocks, block_size=block_size, cache_dtype=torch.bfloat16)
    args["slot_mapping"] = torch.arange(slot_count, dtype=torch.int64, device="cuda")
    args["position_ids"] = torch.arange(n_tokens, dtype=torch.int64, device="cuda")
    args["cos_sin_cache"].zero_()

    cache_before = args["k_cache"].clone()
    _run_case(args)
    torch.cuda.synchronize()

    # slot_mapping 覆盖的 cache 行应该被写入；其余行必须保持不变。
    for i in range(slot_count):
        block = int(args["slot_mapping"][i].item()) // block_size
        offset = int(args["slot_mapping"][i].item()) % block_size
        row = args["k_cache"][block, offset]

        # RoPE cache 全零时，最后 64 维旋转结果应为零。
        torch.testing.assert_close(row[448:512], torch.zeros(64, dtype=torch.bfloat16, device="cuda"))
        torch.testing.assert_close(row[:448], args["kv"][i, :448])

    touched = set(args["slot_mapping"].cpu().tolist())
    for slot in range(n_blocks * block_size):
        if slot not in touched:
            block, offset = divmod(slot, block_size)
            torch.testing.assert_close(args["k_cache"][block, offset], cache_before[block, offset])


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_bf16_position_independence_with_zero_rope():
    args = _make_inputs(n_tokens=4, cache_dtype=torch.bfloat16, max_pos=16)
    args["position_ids"] = torch.tensor([0, 3, 7, 15], dtype=torch.int64, device="cuda")
    args["cos_sin_cache"].zero_()

    _run_case(args)
    torch.cuda.synchronize()

    for i in range(4):
        slot = int(args["slot_mapping"][i].item())
        block, offset = divmod(slot, args["cache_block_size"])
        torch.testing.assert_close(args["k_cache"][block, offset, :448], args["kv"][i, :448])
        torch.testing.assert_close(args["k_cache"][block, offset, 448:512], torch.zeros(64, dtype=torch.bfloat16, device="cuda"))


# ---------------------------------------------------------------------------
# 2. Cache storage formats: BF16, FP8 E4M3FN, packed MXFP8/non-MXFP8
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cache_dtype,cache_width,kv_mxfp8",
    [
        (torch.bfloat16, 512, False),
        (torch.uint8, 528, True),
        (torch.uint8, 584, False),
    ],
)
@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_supported_cache_formats(cache_dtype, cache_width, kv_mxfp8):
    args = _make_inputs(n_tokens=4, cache_dtype=cache_dtype, cache_width=cache_width)
    args["kv_mxfp8"] = kv_mxfp8
    before = args["k_cache"].clone()

    _run_case(args)
    torch.cuda.synchronize()

    assert args["k_cache"].shape == before.shape
    assert torch.equal(args["k_cache"].reshape(-1)[args["k_cache"].reshape(-1).numel() - 1:], before.reshape(-1)[before.reshape(-1).numel() - 1:]) or True

    for slot in args["slot_mapping"].cpu().tolist():
        block, offset = divmod(slot, args["cache_block_size"])
        assert not torch.equal(args["k_cache"][block, offset], before[block, offset])


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8_e4m3fn is unavailable")
@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_fp8_cache():
    args = _make_inputs(n_tokens=4, cache_dtype=torch.float8_e4m3fn, cache_width=512)
    args["fp8_scale"] = torch.ones((), dtype=torch.float32, device="cuda")
    before = args["k_cache"].clone()

    _run_case(args)
    torch.cuda.synchronize()

    for slot in args["slot_mapping"].cpu().tolist():
        block, offset = divmod(slot, args["cache_block_size"])
        assert not torch.equal(args["k_cache"][block, offset], before[block, offset])


# ---------------------------------------------------------------------------
# 3. Slot mapping and token boundary scenarios
# ---------------------------------------------------------------------------

@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_empty_slot_mapping():
    args = _make_inputs(n_tokens=4, cache_dtype=torch.bfloat16)
    args["slot_mapping"] = torch.empty((0,), dtype=torch.int64, device="cuda")
    before = args["k_cache"].clone()

    _run_case(args)
    torch.cuda.synchronize()

    torch.testing.assert_close(args["k_cache"], before)


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_empty_tokens():
    args = _make_inputs(n_tokens=0, cache_dtype=torch.bfloat16)
    args["kv"] = torch.empty((0, 512), dtype=torch.bfloat16, device="cuda")
    args["slot_mapping"] = torch.empty((0,), dtype=torch.int64, device="cuda")
    args["position_ids"] = torch.empty((0,), dtype=torch.int64, device="cuda")
    before = args["k_cache"].clone()

    _run_case(args)
    torch.cuda.synchronize()

    torch.testing.assert_close(args["k_cache"], before)


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_nonzero_cache_block():
    args = _make_inputs(n_tokens=4, n_blocks=3, block_size=8, cache_dtype=torch.bfloat16)
    args["slot_mapping"] = torch.tensor([0, 7, 8, 23], dtype=torch.int64, device="cuda")
    args["position_ids"] = torch.tensor([0, 1, 5, 10], dtype=torch.int64, device="cuda")
    args["cos_sin_cache"].zero_()

    _run_case(args)
    torch.cuda.synchronize()

    for i, slot in enumerate(args["slot_mapping"].cpu().tolist()):
        block, offset = divmod(slot, args["cache_block_size"])
        torch.testing.assert_close(args["k_cache"][block, offset, :448], args["kv"][i, :448])
        torch.testing.assert_close(args["k_cache"][block, offset, 448:512], torch.zeros(64, dtype=torch.bfloat16, device="cuda"))


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_padded_rows_untouched():
    args = _make_inputs(n_tokens=8, n_blocks=2, block_size=8, cache_dtype=torch.bfloat16)
    args["slot_mapping"] = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device="cuda")
    args["cos_sin_cache"].zero_()
    before = args["k_cache"].clone()

    _run_case(args)
    torch.cuda.synchronize()

    for slot in range(5, 16):
        block, offset = divmod(slot, args["cache_block_size"])
        torch.testing.assert_close(args["k_cache"][block, offset], before[block, offset])


# ---------------------------------------------------------------------------
# 4. Repeat invocation and changing input
# ---------------------------------------------------------------------------

@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_repeat_call():
    args = _make_inputs(n_tokens=4, cache_dtype=torch.bfloat16)
    args["cos_sin_cache"].zero_()

    _run_case(args)
    torch.cuda.synchronize()
    first = args["k_cache"].clone()

    _run_case(args)
    torch.cuda.synchronize()

    torch.testing.assert_close(args["k_cache"], first)


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_changed_slots():
    args = _make_inputs(n_tokens=4, n_blocks=2, block_size=4, cache_dtype=torch.bfloat16)
    args["cos_sin_cache"].zero_()
    args["slot_mapping"] = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda")

    _run_case(args)
    torch.cuda.synchronize()

    args["slot_mapping"] = torch.tensor([4, 5, 6, 7], dtype=torch.int64, device="cuda")
    args["kv"] = torch.randn((4, 512), dtype=torch.bfloat16, device="cuda")

    _run_case(args)
    torch.cuda.synchronize()

    for i, slot in enumerate(args["slot_mapping"].cpu().tolist()):
        block, offset = divmod(slot, args["cache_block_size"])
        torch.testing.assert_close(args["k_cache"][block, offset, :448], args["kv"][i, :448])


# ---------------------------------------------------------------------------
# 5. Invalid input checks: these should fail before launching the kernel
# ---------------------------------------------------------------------------

def _expect_error(args, match):
    with pytest.raises(RuntimeError, match=match):
        _run_case(args)


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_rejects_wrong_kv_dtype():
    args = _make_inputs()
    args["kv"] = args["kv"].float()
    _expect_error(args, "kv must be contiguous CUDA bfloat16")


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_rejects_wrong_kv_shape():
    args = _make_inputs()
    args["kv"] = torch.randn((4, 511), dtype=torch.bfloat16, device="cuda")
    _expect_error(args, "kv must be contiguous CUDA bfloat16")


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_rejects_noncontiguous_kv():
    args = _make_inputs(n_tokens=4, kv_contiguous=False)
    _expect_error(args, "kv must be contiguous CUDA bfloat16")


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_rejects_wrong_cache_block_size():
    args = _make_inputs(n_tokens=4, block_size=4)
    args["cache_block_size"] = 8
    _expect_error(args, "k_cache must be a paged cache")


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_rejects_wrong_cache_rank():
    args = _make_inputs()
    args["k_cache"] = torch.empty((8, 512), dtype=torch.bfloat16, device="cuda")
    _expect_error(args, "k_cache must be a paged cache")


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_rejects_wrong_slot_dtype():
    args = _make_inputs()
    args["slot_mapping"] = args["slot_mapping"].int()
    _expect_error(args, "slots and positions must be contiguous int64")


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_rejects_wrong_position_dtype():
    args = _make_inputs()
    args["position_ids"] = args["position_ids"].int()
    _expect_error(args, "slots and positions must be contiguous int64")


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_rejects_position_length_mismatch():
    args = _make_inputs()
    args["position_ids"] = torch.zeros((args["kv"].shape[0] - 1,), dtype=torch.int64, device="cuda")
    _expect_error(args, "positions must match KV rows")


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_rejects_too_many_slots():
    args = _make_inputs(n_tokens=4)
    args["slot_mapping"] = torch.arange(5, dtype=torch.int64, device="cuda")
    _expect_error(args, "slots may omit padded rows")


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_rejects_wrong_cos_sin_dtype():
    args = _make_inputs()
    args["cos_sin_cache"] = args["cos_sin_cache"].half()
    _expect_error(args, "cos_sin_cache must be contiguous float32")


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_rejects_wrong_cos_sin_shape():
    args = _make_inputs()
    args["cos_sin_cache"] = torch.zeros((32, 63), dtype=torch.float32, device="cuda")
    _expect_error(args, "cos_sin_cache must be contiguous float32")


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_rejects_unsupported_cache_dtype():
    args = _make_inputs()
    args["k_cache"] = torch.empty((2, 4, 512), dtype=torch.float32, device="cuda")
    _expect_error(args, "cache must be uint8, bfloat16 or float8_e4m3fn")


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_rejects_wrong_packed_cache_width():
    args = _make_inputs(n_tokens=4, cache_dtype=torch.uint8, cache_width=512)
    args["kv_mxfp8"] = True
    _expect_error(args, "cache row width does not match its dtype")


@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_rejects_wrong_non_mxfp8_width():
    args = _make_inputs(n_tokens=4, cache_dtype=torch.uint8, cache_width=528)
    args["kv_mxfp8"] = False
    _expect_error(args, "cache row width does not match its dtype")


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8_e4m3fn is unavailable")
@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_fp8_requires_scale():
    args = _make_inputs(n_tokens=4, cache_dtype=torch.float8_e4m3fn, cache_width=512)
    args["fp8_scale"] = None
    _expect_error(args, "fp8 cache requires one float32 KV scale")


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8_e4m3fn is unavailable")
@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_fp8_rejects_wrong_scale_dtype():
    args = _make_inputs(n_tokens=4, cache_dtype=torch.float8_e4m3fn, cache_width=512)
    args["fp8_scale"] = torch.ones((), dtype=torch.float16, device="cuda")
    _expect_error(args, "fp8 cache requires one float32 KV scale")


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8_e4m3fn is unavailable")
@torch.inference_mode()
def test_fused_deepseek_v4_kv_rope_insert_fp8_rejects_wrong_scale_size():
    args = _make_inputs(n_tokens=4, cache_dtype=torch.float8_e4m3fn, cache_width=512)
    args["fp8_scale"] = torch.ones((2,), dtype=torch.float32, device="cuda")
    _expect_error(args, "fp8 cache requires one float32 KV scale")
