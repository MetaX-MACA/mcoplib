"""Public GLM paged-dequant API tests, independent of the DSV4 layout."""

import pytest
import torch

pytest.importorskip("triton")

from mcoplib.triton_dequantize_k_cache import dequantize_k_cache_paged_optimized
from mcoplib.triton_quantize_k_cache import quantize_k_cache_separate_optimized


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
FP8 = torch.float8_e4m3fn

# M is the flattened gather length. It is not necessarily the query length.
GLM52_DEQUANT_TEST_SET = {
    "gather_rows": [1, 2, 7, 64, 127, 128, 1024, 2048, 4096, 8192, 10240,
                    16384, 32768, 65536],
    "decode_128_topk_2048": [128 * 2048],
    "rows4_dispatch_boundary": [32767, 32769, 40961, 61441],
    "c600u_dispatch_boundaries": [1023, 1025, 1537, 8191, 8193, 10003],
    "c600u_rows4_tail": [65538, 65539],
}


def _make_cache(capacity, dim_rope=64, padding=0):
    width = 528 + dim_rope * 2
    raw = torch.randint(0, 256, (capacity, 1, width + padding),
                        dtype=torch.uint8, device="cuda")[..., :width]
    raw[..., 512:528].view(torch.float32).copy_(
        torch.rand((capacity, 1, 4), device="cuda") + 0.01
    )
    return raw.view(FP8)


def _make_topk_overlap_ids(capacity, m, overlap_percent, topk=2048):
    """Synthetic exact pairwise overlap; each query selects unique positions."""
    assert m % topk == 0 and 0 <= overlap_percent <= 100
    queries = m // topk
    shared = topk * overlap_percent // 100
    private = topk - shared
    needed = shared + queries * private
    assert needed <= capacity
    pool = torch.randperm(capacity, device="cuda")[:needed]
    ids = torch.cat((pool[:shared].expand(queries, shared),
                     pool[shared:].reshape(queries, private)), dim=1)
    order = torch.rand((queries, topk), device="cuda").argsort(dim=1)
    return ids.gather(1, order).reshape(-1)


def _reference(cache, indices):
    width = cache.shape[-1]
    rows = cache.view(-1, width).view(torch.uint8).index_select(0, indices.long())
    m = indices.numel()
    values = rows[:, :512].view(FP8).float().reshape(m, 4, 128)
    scales = rows[:, 512:528].view(torch.float32).reshape(m, 4, 1)
    dim_rope = (width - 528) // 2
    result = torch.empty((m, 1, 512 + dim_rope), device="cuda", dtype=torch.bfloat16)
    result[:, 0, :512] = (values * scales).reshape(m, 512).to(torch.bfloat16)
    if dim_rope:
        result[:, 0, 512:] = rows[:, 528:].view(torch.bfloat16)
    return result


def _assert_equal(actual, expected):
    torch.testing.assert_close(actual[..., :512], expected[..., :512],
                               rtol=0, atol=0, equal_nan=True)
    if actual.shape[-1] > 512:
        torch.testing.assert_close(actual[..., 512:].view(torch.uint8),
                                   expected[..., 512:].view(torch.uint8), rtol=0, atol=0)


@pytest.fixture(scope="module", params=[(200_000, 64), (1_000_000, 64),
                                      (200_000, 0), (1_000_000, 0)])
def target_cache(request):
    capacity, dim_rope = request.param
    return _make_cache(capacity, dim_rope)


@pytest.mark.parametrize("m", sorted({m for cases in GLM52_DEQUANT_TEST_SET.values() for m in cases}))
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_glm52_dequant_target_shapes(target_cache, m, index_dtype):
    capacity = target_cache.shape[0]
    ids = torch.randint(capacity, (m,), device="cuda", dtype=index_dtype)
    if m > 1:
        ids[0], ids[-1] = capacity - 1, 0
    actual = dequantize_k_cache_paged_optimized(target_cache, ids)
    assert actual.shape == (m, 1, 512 + (target_cache.shape[-1] - 528) // 2)
    assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
    _assert_equal(actual, _reference(target_cache, ids))


@pytest.mark.parametrize("dim_rope", [0, 64])
@pytest.mark.parametrize("layout", ["2d", "4d", "padded", "offset", "expanded"])
@pytest.mark.parametrize("gather_rows", [65, 2049, 4097, 8193, 9217, 12289])
def test_dequant_cache_layouts(dim_rope, layout, gather_rows):
    cache = _make_cache(32, dim_rope, padding=16 if layout == "padded" else 0)
    width = cache.shape[-1]
    if layout == "2d":
        cache = cache.view(32, width)
    elif layout == "4d":
        cache = cache.view(4, 8, 1, width)
    elif layout == "offset":
        storage = torch.empty(32 * width + 4, device="cuda", dtype=torch.uint8)
        shifted = storage[4:].reshape(32, 1, width)
        shifted.copy_(cache.view(torch.uint8))
        cache = shifted.view(FP8)
    elif layout == "expanded":
        cache = cache[:1].expand(32, 1, width)
    # Prefix sharing, including multi-row dispatch and partial final tiles.
    ids = (torch.arange(gather_rows, device="cuda") * 7 % 32).to(torch.int64)
    _assert_equal(dequantize_k_cache_paged_optimized(cache, ids), _reference(cache, ids))


@pytest.mark.parametrize("dim_rope", [0, 64])
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.int32, torch.float32])
@pytest.mark.parametrize("m", [65, 1024, 1025, 1536, 2047, 2048, 2049, 4096, 4097, 8191, 16383, 16384, 16385, 65537])
def test_dequant_mask_skips_invalid_ids_and_preserves_out(dim_rope, mask_dtype, m):
    cache = _make_cache(17, dim_rope)
    active = torch.arange(m, device="cuda") % 3 != 1
    mask = active.to(mask_dtype)
    if mask_dtype == torch.float32:
        mask[active] = -2.0  # Any nonzero real value enables a row.
        mask[0] = float("nan")
    ids = (torch.arange(m, device="cuda") % 17).to(torch.int64)
    ids[~active] = 2**40  # Inactive IDs must never be dereferenced or truncated.
    out = torch.full((m, 1, 512 + dim_rope), -17.25, device="cuda", dtype=torch.bfloat16)
    before = out.clone()
    result = dequantize_k_cache_paged_optimized(cache, ids, 128, mask, out=out)
    assert result is out
    _assert_equal(result[active], _reference(cache, ids[active]))
    torch.testing.assert_close(out[~active].view(torch.uint8), before[~active].view(torch.uint8),
                               rtol=0, atol=0)
    allocated = dequantize_k_cache_paged_optimized(cache, ids, row_mask=mask)
    _assert_equal(allocated[active], _reference(cache, ids[active]))


@pytest.mark.parametrize("dim_rope", [0, 64])
@pytest.mark.parametrize("m", [7, 1024, 16384, 65537])
def test_dequant_all_masked_and_empty(dim_rope, m):
    cache = _make_cache(3, dim_rope)
    ids = torch.full((m,), -1, device="cuda", dtype=torch.int32)
    mask = torch.zeros(m, device="cuda", dtype=torch.bool)
    out = torch.full((m, 1, 512 + dim_rope), 23.0, device="cuda", dtype=torch.bfloat16)
    dequantize_k_cache_paged_optimized(cache, ids, row_mask=mask, out=out)
    assert torch.all(out == 23)
    empty_cache = _make_cache(0, dim_rope)
    empty_ids = torch.empty(0, device="cuda", dtype=torch.int32)
    empty = dequantize_k_cache_paged_optimized(empty_cache, empty_ids)
    assert empty.shape == (0, 1, 512 + dim_rope)
    assert dequantize_k_cache_paged_optimized(empty_cache, empty_ids, out=empty) is empty


@pytest.mark.parametrize("gather_rows", [5, 1023, 1024, 1025, 2048, 4097, 8191, 8192, 8193, 9217, 12289, 32767, 32768, 32769, 65537, 65538, 65539])
@pytest.mark.parametrize("masked", [False, True])
def test_dequant_all_fp8_codes_and_special_scales(gather_rows, masked, monkeypatch):
    cache = _make_cache(4)
    raw = cache.view(torch.uint8)
    raw[..., :512] = (torch.arange(512, device="cuda") % 256).to(torch.uint8)
    raw[..., 512:528].view(torch.float32).copy_(torch.tensor(
        [[0.0, -1.0, float("inf"), float("nan")],
         [1e-30, -1e30, 1.0, 1 / 3],
         [-0.0, 2.0, -2.0, 0.125],
         [0.01, 0.1, 10.0, 1000.0]], device="cuda")[:, None, :])
    ids = (torch.arange(gather_rows, device="cuda") * 3 % 4).to(torch.int32)
    mask = torch.arange(gather_rows, device="cuda") % 3 != 1 if masked else None
    out = torch.full((gather_rows, 1, 576), -17.25, device="cuda", dtype=torch.bfloat16)
    expected = out.clone()
    active = mask if masked else torch.ones(gather_rows, device="cuda", dtype=torch.bool)
    expected[active] = _reference(cache, ids[active])
    ids[~active] = -1
    _assert_equal(dequantize_k_cache_paged_optimized(cache, ids, row_mask=mask, out=out), expected)
    # The retained MACA Triton multiply canonicalizes some signed zeros.
    # Compare bits to the retained path; PyTorch's zero sign is different.
    import importlib
    module = importlib.import_module("mcoplib.triton_dequantize_k_cache")
    if module._is_c600u(cache.device.index):
        monkeypatch.setattr(module, "_is_c600u", lambda device_index: False)
        retained = torch.full_like(out, -17.25)
        module.dequantize_k_cache_paged_optimized(cache, ids, row_mask=mask, out=retained)
        non_nan = ~torch.isnan(retained[..., :512])
        torch.testing.assert_close(out[..., :512].view(torch.int16)[non_nan],
                                   retained[..., :512].view(torch.int16)[non_nan], rtol=0, atol=0)


@pytest.mark.parametrize("repeats", [1, 4, 9, 13])
def test_dequant_rope_preserves_all_bf16_bits(repeats):
    cache = _make_cache(1024)
    cache.view(torch.uint8)[..., 528:].view(torch.int16).copy_(
        torch.arange(65536, device="cuda").to(torch.int16).reshape(1024, 1, 64))
    ids = torch.arange(1023, -1, -1, device="cuda", dtype=torch.int64).repeat(repeats)
    _assert_equal(dequantize_k_cache_paged_optimized(cache, ids), _reference(cache, ids))


def test_dequant_quantize_round_trip_and_graph():
    n = 4097
    nope = torch.randn((n, 1, 512), device="cuda", dtype=torch.bfloat16)
    rope = torch.randn((n, 1, 64), device="cuda", dtype=torch.bfloat16)
    nope_part, rope_part = quantize_k_cache_separate_optimized(nope, rope)
    cache = torch.cat((nope_part, rope_part), dim=-1).view(FP8)
    ids = torch.tensor([4096, 0, 17, 17, 65], device="cuda", dtype=torch.int32)
    expected = _reference(cache, ids)
    out = dequantize_k_cache_paged_optimized(cache, ids)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dequantize_k_cache_paged_optimized(cache, ids, out=out)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    _assert_equal(out, expected)


@pytest.mark.parametrize("dim_rope", [0, 64])
def test_dequant_without_gluon(monkeypatch, dim_rope):
    import mcoplib.triton_dequantize_k_cache as module
    monkeypatch.setattr(module, "_dequantize_k_cache_gluon_rows2_kernel", None)
    monkeypatch.setattr(module, "_dequantize_k_cache_gluon_kernel", None)
    cache = _make_cache(32, dim_rope)
    ids = (torch.arange(12289, device="cuda") % 32).to(torch.int32)
    _assert_equal(module.dequantize_k_cache_paged_optimized(cache, ids), _reference(cache, ids))


def test_dequant_rejects_invalid_inputs():
    cache = _make_cache(8)
    ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    op = dequantize_k_cache_paged_optimized
    for bad, error, match in [(cache.view(torch.uint8), TypeError, "float8"),
                              (torch.empty((2, 544), device="cuda", dtype=FP8), ValueError, "528 or 656"),
                              (torch.empty((2, 1312), device="cuda", dtype=FP8)[:, ::2], ValueError, "column stride"),
                              (torch.empty(2 * 656 + 1, device="cuda", dtype=FP8)[1:].reshape(2, 656), ValueError, "aligned")]:
        with pytest.raises(error, match=match):
            op(bad, ids)
    with pytest.raises(ValueError, match="group_size"):
        op(cache, ids, group_size=64)
    for bad in (ids.reshape(1, 2), torch.arange(4, device="cuda", dtype=torch.int32)[::2]):
        with pytest.raises(ValueError, match="contiguous 1D"):
            op(cache, bad)
    with pytest.raises(TypeError, match="int32 or int64"):
        op(cache, ids.float())
    with pytest.raises(ValueError, match="cache device"):
        op(cache, ids.cpu())
    for mask in (torch.ones((2, 1), device="cuda"), torch.ones(3, device="cuda"),
                 torch.ones(4, device="cuda")[::2]):
        with pytest.raises(ValueError, match="row_mask"):
            op(cache, ids, row_mask=mask)
    with pytest.raises(ValueError, match="row_mask"):
        op(cache, ids, row_mask=torch.ones(2))
    with pytest.raises(TypeError, match="real numeric"):
        op(cache, ids, row_mask=torch.ones(2, device="cuda", dtype=torch.complex64))
    for out in (torch.empty((2, 1, 575), device="cuda", dtype=torch.bfloat16),
                torch.empty((2, 1, 1152), device="cuda", dtype=torch.bfloat16)[..., ::2]):
        with pytest.raises(ValueError, match="out must be contiguous"):
            op(cache, ids, out=out)
    with pytest.raises(TypeError, match="bfloat16"):
        op(cache, ids, out=torch.empty((2, 1, 576), device="cuda"))
    aliased = cache.view(torch.bfloat16).reshape(-1)[:2 * 576].reshape(2, 1, 576)
    with pytest.raises(ValueError, match="separate storage"):
        op(cache, ids, out=aliased)


@pytest.mark.parametrize("dim_rope", [0, 64])
@pytest.mark.parametrize("m", [2049, 8193])
def test_dequant_masked_gluon_unavailable_fallback(monkeypatch, dim_rope, m):
    import mcoplib.triton_dequantize_k_cache as module
    monkeypatch.setattr(module, "_dequantize_k_cache_gluon_masked_kernel", None)
    cache = _make_cache(17, dim_rope)
    active = torch.arange(m, device="cuda") % 3 != 1
    ids = (torch.arange(m, device="cuda") % 17).to(torch.int64)
    ids[~active] = 2**40
    out = torch.full((m, 1, 512 + dim_rope), -17.25, device="cuda", dtype=torch.bfloat16)
    expected = out.clone()
    expected[active] = _reference(cache, ids[active])
    _assert_equal(module.dequantize_k_cache_paged_optimized(cache, ids, row_mask=active, out=out), expected)


@pytest.mark.parametrize("dim_rope", [0, 64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("m", [1024, 2049, 4097, 8193, 16383, 16384, 16385, 65537])
def test_dequant_masked_bits_with_unaligned_output(monkeypatch, dim_rope, index_dtype, m):
    width = 528 + 2 * dim_rope
    storage = torch.empty(4 + 32 * (width + 16), device="cuda", dtype=torch.uint8)
    raw = storage[4:].reshape(32, 1, width + 16)[..., :width]
    raw[..., :512] = (torch.arange(32 * 512, device="cuda") % 256).to(torch.uint8).reshape(32, 1, 512)
    scales = [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan"),
              1e-30, -1e30, 1 / 3, 0.1, 448.0, 2**-126, 2**-149, 3.4e38, -3.4e38]
    raw[..., 512:528].view(torch.float32).copy_(torch.tensor(scales * 8, device="cuda").reshape(32, 1, 4))
    if dim_rope:
        raw[..., 528:].view(torch.int16).copy_((torch.arange(32 * dim_rope, device="cuda") * 31).to(torch.int16).reshape(32, 1, dim_rope))
    cache = raw.view(FP8)
    active = torch.arange(m, device="cuda") % 3 != 1
    mask = active.to(torch.float32 if index_dtype == torch.int64 else torch.bool)
    if index_dtype == torch.int64:
        mask[active] = -2.0
        mask[0] = float("nan")
    ids = (torch.arange(m, device="cuda") * 7 % 32).to(index_dtype)
    ids[~active] = 2**40 if index_dtype == torch.int64 else -1
    flat = torch.full((m * (512 + dim_rope) + 1,), -17.25, device="cuda", dtype=torch.bfloat16)
    out = flat[1:].reshape(m, 1, 512 + dim_rope)
    assert out.data_ptr() % 4 == 2
    reference_flat = flat.clone()
    expected = reference_flat[1:].reshape_as(out)
    dequantize_k_cache_paged_optimized(cache, ids, row_mask=mask, out=out)
    # Preserve the original scalar kernel's bit semantics. PyTorch's multiply
    # differs in zero sign on MACA; it remains the numerical reference below.
    import mcoplib.triton_dequantize_k_cache as module
    with monkeypatch.context() as patch:
        patch.setattr(module, "_dequantize_k_cache_gluon_masked_kernel", None)
        module.dequantize_k_cache_paged_optimized(cache, ids, row_mask=mask, out=expected)
    _assert_equal(out[active], _reference(cache, ids[active]))
    actual_nope, expected_nope = out[..., :512], expected[..., :512]
    both_nan = actual_nope.isnan() & expected_nope.isnan()
    assert torch.equal(actual_nope.view(torch.int16)[~both_nan], expected_nope.view(torch.int16)[~both_nan])
    if dim_rope:
        assert torch.equal(out[..., 512:].view(torch.uint8), expected[..., 512:].view(torch.uint8))


@pytest.mark.parametrize("dim_rope", [0, 64])
@pytest.mark.parametrize("m", [8193, 12288, 12289, 16385, 32767, 32768, 32769,
                              40961, 49153, 61441, 65535, 65536, 65537])
def test_dequant_unmasked_multirow_bits_with_unaligned_output(monkeypatch, dim_rope, m):
    import mcoplib.triton_dequantize_k_cache as module
    width = 528 + 2 * dim_rope
    original = _make_cache(32, dim_rope, padding=16)
    storage = torch.empty(4 + 32 * (width + 16), device="cuda", dtype=torch.uint8)
    raw = storage[4:].reshape(32, 1, width + 16)[..., :width]
    raw.copy_(original.view(torch.uint8))
    raw[..., :512] = (torch.arange(512, device="cuda") % 256).to(torch.uint8)
    raw[..., 512:528].view(torch.float32).copy_(torch.tensor([-0.0, -1.0, float("inf"), float("nan")], device="cuda"))
    cache = raw.view(FP8)
    ids = (torch.arange(m, device="cuda") * 7 % 32).to(torch.int32)
    flat = torch.full((m * (512 + dim_rope) + 1,), -17.25, device="cuda", dtype=torch.bfloat16)
    out = flat[1:].reshape(m, 1, 512 + dim_rope)
    expected = flat.clone()[1:].reshape_as(out)
    module.dequantize_k_cache_paged_optimized(cache, ids, out=out)
    with monkeypatch.context() as patch:
        patch.setattr(module, "_dequantize_k_cache_gluon_rows2_kernel", None)
        module.dequantize_k_cache_paged_optimized(cache, ids, out=expected)
    _assert_equal(out, _reference(cache, ids))
    actual_nope, expected_nope = out[..., :512], expected[..., :512]
    both_nan = actual_nope.isnan() & expected_nope.isnan()
    assert torch.equal(actual_nope.view(torch.int16)[~both_nan], expected_nope.view(torch.int16)[~both_nan])
    if dim_rope:
        assert torch.equal(out[..., 512:].view(torch.uint8), expected[..., 512:].view(torch.uint8))


@pytest.mark.parametrize("capacity", [200000, 1000000])
@pytest.mark.parametrize("dim_rope", [0, 64])
@pytest.mark.parametrize("overlap_percent", [0, 25, 50, 75])
def test_dequant_topk_overlap_test_set(capacity, dim_rope, overlap_percent):
    m, topk = 10240, 2048
    ids = _make_topk_overlap_ids(capacity, m, overlap_percent)
    query_ids = ids.reshape(-1, topk)
    ordered = query_ids.sort(dim=1).values
    assert bool((ordered[:, 1:] > ordered[:, :-1]).all())
    shared = topk * overlap_percent // 100
    assert torch.unique(ids).numel() == shared + (m // topk) * (topk - shared)
    assert len(set(query_ids[0].tolist()) & set(query_ids[1].tolist())) == shared
    cache = _make_cache(capacity, dim_rope)
    _assert_equal(dequantize_k_cache_paged_optimized(cache, ids.to(torch.int32)),
                  _reference(cache, ids))


@pytest.mark.parametrize("dim_rope", [0, 64])
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
def test_dequant_full_pairs_and_active_tail_with_output_guards(dim_rope, mask_dtype):
    # Exercise both row owners and a valid odd tail in the large i32 path,
    # without writes to skipped rows or the surrounding output guards.
    m, width = 16385, 512 + dim_rope
    pattern = torch.tensor([False, False, False, True, True, False, True, True], device="cuda")
    active = pattern.repeat((m + 7) // 8)[:m].clone()
    active[-1] = True
    mask = active.to(mask_dtype)
    if mask_dtype == torch.float32:
        mask[active] = -2.0
        mask[-1] = float("nan")
    ids = (torch.arange(m, device="cuda") % 17).to(torch.int32)
    ids[~active] = -(2**31)
    cache = _make_cache(17, dim_rope)
    flat = torch.full((m * width + 2,), -17.25, device="cuda", dtype=torch.bfloat16)
    out = flat[1:-1].reshape(m, 1, width)
    dequantize_k_cache_paged_optimized(cache, ids, row_mask=mask, out=out)
    _assert_equal(out[active], _reference(cache, ids[active]))
    assert bool((out[~active] == -17.25).all())
    assert flat[0].item() == -17.25 and flat[-1].item() == -17.25
