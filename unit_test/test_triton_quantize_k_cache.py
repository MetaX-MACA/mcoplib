import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl

from mcoplib.triton_quantize_k_cache import (  # noqa: E402
    _bounded_fp8_e4m3,
    _bounded_fp8_e4m3_magic,
    quantize_k_cache_separate_optimized,
)

try:
    import triton.experimental.gluon as gluon
    import triton.experimental.gluon.language as gl
    from mcoplib.triton_quantize_k_cache_gluon import _bounded_fp8_e4m3_gluon
except ImportError:
    gluon = None


DIM_NOPE = 512
DIM_ROPE = 64
GROUP_SIZE = 128
FP8_E4M3_RTOL = 1 / 8
FP8_E4M3_ATOL = 2**-9


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires a CUDA device"
)


PREFILL_SHAPES = [2 * 1024, 4 * 1024, 8 * 1024, 16 * 1024]
DECODING_SHAPES = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    12,
    14,
    16,
    20,
    24,
    28,
    32,
    40,
    48,
    56,
    64,
]
DEFAULT_SHAPES = sorted([*PREFILL_SHAPES, *DECODING_SHAPES])

# The GLM-5.2 baseline set is selectable independently with
# pytest -k glm52_target_shapes. Keep it separate from historical coverage.
GLM52_QUANTIZATION_TEST_SET = {
    "decode": [1, 2, 4, 8, 16, 32, 64, 65, 96, 127, 128],
    "prefill": [1024, 2048, 4096, 8192, 16384, 32768, 65536],
}


def _unpack(nope_part, rope_part):
    nope_bytes = nope_part.squeeze(1)
    rope_bytes = rope_part.squeeze(1)
    nope_q = nope_bytes[:, :DIM_NOPE].view(torch.float8_e4m3fn)
    nope_s = nope_bytes[:, DIM_NOPE:].view(torch.float32)
    rope = rope_bytes.view(torch.bfloat16)
    return nope_q, nope_s, rope


@pytest.mark.parametrize(
    "num_tokens,block_m,num_warps",
    [(shape, None, None) for shape in DEFAULT_SHAPES] + [(7, 4, 1)],
)
def test_quantize_k_cache_separate_optimized(num_tokens, block_m, num_warps):
    _check_quantization(num_tokens, block_m, num_warps)


@pytest.mark.parametrize(
    "num_tokens",
    [
        pytest.param(n, id=f"glm52-{phase}-{n}")
        for phase, shapes in GLM52_QUANTIZATION_TEST_SET.items()
        for n in shapes
    ],
)
def test_quantize_k_cache_glm52_target_shapes(num_tokens):
    _check_quantization(num_tokens, block_m=None, num_warps=None)


@pytest.mark.parametrize("num_tokens", [24575, 24576, 24577, 28673, 32769])
def test_quantize_k_cache_magic_dispatch_boundaries(num_tokens):
    _check_quantization(num_tokens, block_m=None, num_warps=None)


def _check_quantization(num_tokens, block_m, num_warps):
    torch.manual_seed(7)
    device = torch.device("cuda")
    k_nope = torch.randn(
        (num_tokens, 1, DIM_NOPE), dtype=torch.bfloat16, device=device
    )
    k_rope = torch.randn(
        (num_tokens, 1, DIM_ROPE), dtype=torch.bfloat16, device=device
    )

    nope_part, rope_part = quantize_k_cache_separate_optimized(
        k_nope, k_rope, block_m=block_m, num_warps=num_warps
    )

    assert nope_part.shape == (num_tokens, 1, 528)
    assert rope_part.shape == (num_tokens, 1, 128)
    assert nope_part.dtype == torch.uint8
    assert rope_part.dtype == torch.uint8

    nope_q, nope_s, rope = _unpack(nope_part, rope_part)
    grouped = k_nope.squeeze(1).float().reshape(-1, 4, GROUP_SIZE)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    expected_s = grouped.abs().amax(dim=-1) / fp8_max
    if block_m is None and num_warps is None and num_tokens <= 64:
        expected_q_float = grouped * (1.0 / expected_s).unsqueeze(-1)
    else:
        safe_s = torch.where(expected_s > 0, expected_s, 1.0)
        expected_q_float = grouped / safe_s.unsqueeze(-1)
    expected_q = expected_q_float.clamp(-fp8_max, fp8_max).to(
        torch.float8_e4m3fn
    ).reshape(-1, DIM_NOPE)

    torch.testing.assert_close(nope_s, expected_s, rtol=1e-5, atol=1e-7)
    # PyTorch and Triton may choose adjacent E4M3 code points at rounding
    # boundaries. One E4M3 quantization interval is therefore allowed here.
    torch.testing.assert_close(
        nope_q.float(),
        expected_q.float(),
        rtol=FP8_E4M3_RTOL,
        atol=FP8_E4M3_ATOL,
    )
    torch.testing.assert_close(rope, k_rope.squeeze(1), rtol=0, atol=0)

def test_quantize_k_cache_rejects_invalid_configuration():
    device = torch.device("cuda")
    k_nope = torch.empty((1, DIM_NOPE), dtype=torch.bfloat16, device=device)
    k_rope = torch.empty((1, DIM_ROPE), dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="tile_size=128"):
        quantize_k_cache_separate_optimized(k_nope, k_rope, tile_size=64)
    with pytest.raises(ValueError, match="Unsupported block_m"):
        quantize_k_cache_separate_optimized(k_nope, k_rope, block_m=3)
    with pytest.raises(TypeError, match="torch.bfloat16"):
        quantize_k_cache_separate_optimized(k_nope.float(), k_rope)
    with pytest.raises(ValueError, match="Expected k_nope shape"):
        quantize_k_cache_separate_optimized(k_nope[:, None, :].expand(-1, 2, -1), k_rope)


@pytest.mark.parametrize("num_tokens", [0, 7, 65, 4097, 65537])
@pytest.mark.parametrize("with_head,strided", [(False, False), (True, False), (False, True), (True, True)])
def test_quantize_k_cache_layouts(num_tokens, with_head, strided):
    torch.manual_seed(31)
    step = 2 if strided else 1
    nope = torch.randn((num_tokens, 512 * step), device="cuda", dtype=torch.bfloat16)[:, ::step]
    rope = torch.randn((num_tokens, 64 * step), device="cuda", dtype=torch.bfloat16)[:, ::step]
    reference = quantize_k_cache_separate_optimized(nope.contiguous(), rope.contiguous())
    if with_head:
        nope, rope = nope[:, None, :], rope[:, None, :]
    actual = quantize_k_cache_separate_optimized(nope, rope)
    assert actual[0].shape == (num_tokens, 1, 528)
    assert actual[1].shape == (num_tokens, 1, 128)
    for left, right in zip(reference, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_quantize_k_cache_rope_preserves_all_bf16_bits():
    # A partial large-prefill tile also checks NaN payloads and signed zeros
    # survive the byte-preserving RoPE copy.
    num_tokens = 32769
    nope = torch.zeros((num_tokens, 1, DIM_NOPE), device="cuda", dtype=torch.bfloat16)
    rope = torch.arange(num_tokens * DIM_ROPE, device="cuda", dtype=torch.int32)
    rope = rope.to(torch.int16).view(torch.bfloat16).reshape(num_tokens, 1, DIM_ROPE)
    _, rope_part = quantize_k_cache_separate_optimized(nope, rope)
    torch.testing.assert_close(rope_part, rope.view(torch.uint8), rtol=0, atol=0)


def test_quantize_k_cache_gluon_fallback(monkeypatch):
    if quantize_k_cache_separate_optimized.__globals__.get("_quantize_k_cache_gluon_kernel") is None:
        pytest.skip("Gluon kernel is not available")
    torch.manual_seed(37)
    nope = torch.randn((32769, 1, 512), device="cuda", dtype=torch.bfloat16)
    rope = torch.randn((32769, 1, 64), device="cuda", dtype=torch.bfloat16)
    reference = quantize_k_cache_separate_optimized(nope, rope)
    monkeypatch.setitem(quantize_k_cache_separate_optimized.__globals__, "_quantize_k_cache_gluon_kernel", None)
    actual = quantize_k_cache_separate_optimized(nope, rope)
    for left, right in zip(reference, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@triton.jit
def _check_bounded_fp8_kernel(src, dst, N: tl.constexpr, BLOCK: tl.constexpr, MAGIC: tl.constexpr = False):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(src + offsets, offsets < N, other=0.0)
    x = tl.clamp(x, -448.0, 448.0)
    if MAGIC:
        encoded = _bounded_fp8_e4m3_magic(x).to(tl.uint8, bitcast=True)
    else:
        encoded = _bounded_fp8_e4m3(x).to(tl.uint8, bitcast=True)
    tl.store(dst + offsets, encoded, offsets < N)


if gluon is not None:
    @gluon.jit
    def _check_bounded_fp8_gluon_kernel(src, dst, N: gl.constexpr, BLOCK: gl.constexpr, WARP_SIZE: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1], [WARP_SIZE], [1], [0])
        offsets = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=layout)
        x = gl.load(src + offsets, offsets < N, other=0.0)
        x = gl.minimum(gl.maximum(x, -448.0), 448.0)
        encoded = _bounded_fp8_e4m3_gluon(x).to(gl.uint8, bitcast=True)
        gl.store(dst + offsets, encoded, offsets < N)


@pytest.mark.parametrize("backend", ["triton", "triton_magic", "gluon"])
def test_bounded_fp8_all_rounding_boundaries(backend):
    if backend == "gluon" and gluon is None:
        pytest.skip("Gluon is not available")
    # All finite positive E4M3 codes, every midpoint and the adjacent FP32
    # values on either side; both signs also exercise signed zero.
    codes = torch.arange(127, device="cuda", dtype=torch.uint8)
    values = codes.view(torch.float8_e4m3fn).float()
    midpoints = (values[:-1] + values[1:]) / 2
    positive = torch.cat((values, midpoints,
                          torch.nextafter(midpoints, torch.full_like(midpoints, float("-inf"))),
                          torch.nextafter(midpoints, torch.full_like(midpoints, float("inf")))))
    torch.manual_seed(19)
    random_values = torch.cat((torch.randn(65536, device="cuda") * 200,
                               torch.randn(65536, device="cuda") * 0.01))
    data = torch.cat((positive, -positive, random_values)).clamp(-448, 448)
    output = torch.empty(data.shape, device="cuda", dtype=torch.uint8)
    if backend == "gluon":
        warp_size = triton.runtime.driver.active.get_current_target().warp_size
        _check_bounded_fp8_gluon_kernel[(triton.cdiv(data.numel(), 256),)](
            data, output, N=data.numel(), BLOCK=256, WARP_SIZE=warp_size, num_warps=1)
    else:
        _check_bounded_fp8_kernel[(triton.cdiv(data.numel(), 256),)](
            data, output, N=data.numel(), BLOCK=256, MAGIC=backend == "triton_magic", num_warps=1)
    expected = data.to(torch.float8_e4m3fn).view(torch.uint8)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
