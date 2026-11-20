"""C600-U explicit-layout coverage for fallback and unaligned BF16 views."""

import pytest
import torch

from mcoplib.triton_quantize_k_cache import quantize_k_cache_separate_optimized


@pytest.mark.parametrize("num_tokens", [0, 1, 2, 7, 32, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 127, 128, 129, 255, 256, 257, 511, 512, 513, 767, 768, 769, 1023, 1024, 1025, 1536, 2047, 2048, 2049, 4097, 8191, 8192, 8193, 65537])
@pytest.mark.parametrize("offset", [0, 1])
def test_c600u_output_storage(num_tokens, offset):
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0).upper() != "METAX C600-U":
        pytest.skip("C600-U only")
    inputs = []
    for dim in (512, 64):
        storage = torch.randn(num_tokens * dim + offset, device="cuda", dtype=torch.bfloat16)
        inputs.append(storage[offset:].reshape(num_tokens, 1, dim))
    first = quantize_k_cache_separate_optimized(*inputs)
    saved = [part.clone() for part in first]
    second = quantize_k_cache_separate_optimized(inputs[0] + 1, inputs[1] + 1)
    for part, width in zip((*first, *second), (528, 128, 528, 128)):
        assert part.shape == (num_tokens, 1, width)
        assert part.dtype == torch.uint8 and part.is_contiguous()
    if num_tokens:
        assert len({part.untyped_storage().data_ptr() for part in (*first, *second)}) == 4
    assert all(torch.equal(part, copy) for part, copy in zip(first, saved))


@pytest.mark.parametrize("num_tokens", [1024, 1025, 1536, 2047, 2048, 2049, 4097, 8192, 8193, 16384, 65537])
@pytest.mark.parametrize("offset", [0, 1])
def test_c600u_layout_fallback(monkeypatch, num_tokens, offset):
    _check_fallback(monkeypatch, num_tokens, offset, "_quantize_k_cache_c600u_kernel")




@pytest.mark.parametrize("num_tokens", [65, 66, 67, 68, 69, 70, 71, 72, 127, 128, 129, 255, 256, 257, 511, 512, 513, 767, 768, 769, 1023])
@pytest.mark.parametrize("code_offset", [0, 64])
def test_c600u_raw_output_guards_and_rope_bits(monkeypatch, num_tokens, code_offset):
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0).upper() != "METAX C600-U":
        pytest.skip("C600-U only")
    k_nope = torch.randn((num_tokens, 1, 512), device="cuda", dtype=torch.bfloat16)
    bits = ((torch.arange(num_tokens * 64, device="cuda", dtype=torch.int32) + code_offset) % 65536).to(torch.int16)
    k_rope = bits.view(torch.bfloat16).reshape(num_tokens, 1, 64)
    storages = [torch.full((num_tokens * width + 32,), mark, dtype=torch.uint8, device="cuda")
                for width, mark in [(528, 0xA5), (128, 0x5A)]]
    parts = tuple(storage[16:-16].reshape(num_tokens, 1, width)
                  for storage, width in zip(storages, (528, 128)))
    monkeypatch.setitem(quantize_k_cache_separate_optimized.__globals__,
                        "_allocate_outputs_c600u_raw", lambda *args: parts)
    actual = quantize_k_cache_separate_optimized(k_nope, k_rope)
    assert all(left.data_ptr() == right.data_ptr() for left, right in zip(actual, parts))
    assert torch.equal(actual[1], k_rope.view(torch.uint8))
    for storage, mark in zip(storages, (0xA5, 0x5A)):
        assert bool((storage[:16] == mark).all())
        assert bool((storage[-16:] == mark).all())


@pytest.mark.parametrize("num_tokens", [65, 66, 127, 128, 129, 1023, 1024, 1025, 1536, 2047])
@pytest.mark.parametrize("offset", [0, 1])
def test_c600u_group_layout_fallback(monkeypatch, num_tokens, offset):
    _check_fallback(monkeypatch, num_tokens, offset, "_quantize_k_cache_group_c600u_kernel")




@pytest.mark.parametrize("num_tokens", [31, 32, 33, 47, 48, 49, 63, 64, 65])
@pytest.mark.parametrize("offset", [0, 1])
def test_c600u_decode_inline_bytes_fallback(monkeypatch, num_tokens, offset):
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0).upper() != "METAX C600-U":
        pytest.skip("C600-U only")
    scope = quantize_k_cache_separate_optimized.__globals__
    if "_quantize_k_cache_decode_four_group_c600u_kernel" in scope:
        monkeypatch.setitem(scope, "_quantize_k_cache_decode_four_group_c600u_kernel", None)
    if scope.get("_quantize_k_cache_decode_inline_c600u_kernel") is None:
        pytest.skip("C600-U inline decode kernel unavailable")
    bits = (torch.arange(num_tokens * 64 + offset, device="cuda", dtype=torch.int32)
            % 65536).to(torch.int16)
    rope = bits[offset:].view(torch.bfloat16).reshape(num_tokens, 1, 64)
    storage = torch.randn(num_tokens * 512 + offset, device="cuda", dtype=torch.bfloat16)
    nope = storage[offset:].reshape(num_tokens, 1, 512)
    nope[..., :128] = 0
    nope[..., 128:256] = -0.0
    actual = quantize_k_cache_separate_optimized(nope, rope)
    assert torch.equal(actual[1], rope.view(torch.uint8))
    monkeypatch.setitem(scope, "_quantize_k_cache_decode_inline_c600u_kernel", None)
    expected = quantize_k_cache_separate_optimized(nope, rope)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_c600u_decode_inline_all_uniform_bf16_codes(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0).upper() != "METAX C600-U":
        pytest.skip("C600-U only")
    scope = quantize_k_cache_separate_optimized.__globals__
    if "_quantize_k_cache_decode_four_group_c600u_kernel" in scope:
        monkeypatch.setitem(scope, "_quantize_k_cache_decode_four_group_c600u_kernel", None)
    if scope.get("_quantize_k_cache_decode_inline_c600u_kernel") is None:
        pytest.skip("C600-U inline decode kernel unavailable")
    codes = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    rope = torch.zeros((64, 1, 64), device="cuda", dtype=torch.bfloat16)
    for start in range(0, 65536, 256):
        nope = codes[start:start + 256].reshape(64, 1, 4, 1).expand(64, 1, 4, 128).reshape(64, 1, 512).contiguous()
        actual = quantize_k_cache_separate_optimized(nope, rope)
        with monkeypatch.context() as fallback:
            fallback.setitem(scope, "_quantize_k_cache_decode_inline_c600u_kernel", None)
            expected = quantize_k_cache_separate_optimized(nope, rope)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected)), start


@pytest.mark.parametrize("num_tokens", [31, 32, 33, 55, 56, 57, 58, 63, 64, 65])
@pytest.mark.parametrize("offset", [0, 1])
def test_c600u_decode_four_group_bytes_fallback(monkeypatch, num_tokens, offset):
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0).upper() != "METAX C600-U":
        pytest.skip("C600-U only")
    scope = quantize_k_cache_separate_optimized.__globals__
    if scope.get("_quantize_k_cache_decode_four_group_c600u_kernel") is None:
        pytest.skip("C600-U four-group decode kernel unavailable")
    rope_bits = torch.arange(num_tokens * 64 + offset, device="cuda", dtype=torch.int32).to(torch.int16)
    rope = rope_bits[offset:].view(torch.bfloat16).reshape(num_tokens, 1, 64)
    storage = torch.randn(num_tokens * 512 + offset, device="cuda", dtype=torch.bfloat16)
    nope = storage[offset:].reshape(num_tokens, 1, 512)
    nope[..., :128] = 0.0
    nope[..., 128:256] = -0.0
    nope[..., 256:384:4] = -0.0
    nope[..., 257:384:4] = -(2**-133)
    actual = quantize_k_cache_separate_optimized(nope, rope)
    monkeypatch.setitem(scope, "_quantize_k_cache_decode_four_group_c600u_kernel", None)
    expected = quantize_k_cache_separate_optimized(nope, rope)
    assert torch.equal(actual[1], rope.view(torch.uint8))
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_c600u_decode_four_group_all_uniform_bf16_codes(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0).upper() != "METAX C600-U":
        pytest.skip("C600-U only")
    scope = quantize_k_cache_separate_optimized.__globals__
    if scope.get("_quantize_k_cache_decode_four_group_c600u_kernel") is None:
        pytest.skip("C600-U four-group decode kernel unavailable")
    codes = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    rope = torch.zeros((64, 1, 64), device="cuda", dtype=torch.bfloat16)
    for start in range(0, 65536, 256):
        nope = codes[start:start + 256].reshape(64, 1, 4, 1).expand(64, 1, 4, 128).reshape(64, 1, 512).contiguous()
        actual = quantize_k_cache_separate_optimized(nope, rope)
        with monkeypatch.context() as fallback:
            fallback.setitem(scope, "_quantize_k_cache_decode_four_group_c600u_kernel", None)
            expected = quantize_k_cache_separate_optimized(nope, rope)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected)), start


@pytest.mark.parametrize("num_tokens", [56, 57, 63, 64])
def test_c600u_decode_four_group_output_guards(monkeypatch, num_tokens):
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0).upper() != "METAX C600-U":
        pytest.skip("C600-U only")
    scope = quantize_k_cache_separate_optimized.__globals__
    if scope.get("_quantize_k_cache_decode_four_group_c600u_kernel") is None:
        pytest.skip("C600-U four-group decode kernel unavailable")
    nope = torch.randn((num_tokens, 1, 512), device="cuda", dtype=torch.bfloat16)
    bits = (torch.arange(num_tokens * 64, device="cuda", dtype=torch.int32) + 32640).to(torch.int16)
    rope = bits.view(torch.bfloat16).reshape(num_tokens, 1, 64)
    storage = [torch.full((num_tokens * width + 32,), mark, device="cuda", dtype=torch.uint8)
               for width, mark in [(528, 0xA5), (128, 0x5A)]]
    outputs = tuple(part[16:-16].reshape(num_tokens, 1, width)
                    for part, width in zip(storage, (528, 128)))
    monkeypatch.setitem(scope, "_allocate_outputs_c600u_raw", lambda *args: outputs)
    actual = quantize_k_cache_separate_optimized(nope, rope)
    saved = tuple(part.clone() for part in actual)
    assert all(a.data_ptr() == b.data_ptr() for a, b in zip(actual, outputs))
    assert torch.equal(saved[1], rope.view(torch.uint8))
    monkeypatch.setitem(scope, "_quantize_k_cache_decode_four_group_c600u_kernel", None)
    expected = quantize_k_cache_separate_optimized(nope, rope)
    assert all(torch.equal(a, b) for a, b in zip(saved, expected))
    for part, mark in zip(storage, (0xA5, 0x5A)):
        assert bool((part[:16] == mark).all())
        assert bool((part[-16:] == mark).all())


def _check_fallback(monkeypatch, num_tokens, offset, kernel_name):
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0).upper() != "METAX C600-U":
        pytest.skip("C600-U only")
    scope = quantize_k_cache_separate_optimized.__globals__
    if scope.get(kernel_name) is None:
        pytest.skip("C600-U Gluon kernel unavailable")
    torch.manual_seed(47)
    inputs = []
    for dim in (512, 64):
        storage = torch.randn(num_tokens * dim + offset, device="cuda", dtype=torch.bfloat16)
        inputs.append(storage[offset:].reshape(num_tokens, 1, dim))
    actual = quantize_k_cache_separate_optimized(*inputs)
    monkeypatch.setitem(scope, kernel_name, None)
    fallback = quantize_k_cache_separate_optimized(*inputs)
    for left, right in zip(actual, fallback):
        assert torch.equal(left, right)
