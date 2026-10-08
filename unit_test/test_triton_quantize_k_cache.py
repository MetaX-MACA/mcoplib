import pytest
import torch

pytest.importorskip("triton")

from mcoplib.triton_quantize_k_cache import (  # noqa: E402
    quantize_k_cache_separate_optimized,
)


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
