"""Mixed RMSNorm must match the existing float-input kernel bit for bit."""
import pytest
import torch
import os
if os.environ.get('RMS_NORM_EXTENSION'):
    torch.ops.load_library(os.environ['RMS_NORM_EXTENSION'])
else:
    import mcoplib._C  # noqa: F401

DTYPES = (torch.bfloat16, torch.float16)
SHAPES = [(1, 1), (7, 7), (7, 8), (7, 9), (17, 64), (257, 128),
          (1, 192), (64, 192), (255, 192), (256, 192), (257, 192),
          (1023, 192), (1024, 192), (1025, 192), (17, 193),
          (16, 256), (64, 256), (65, 256), (1024, 256), (17, 257),
          (64, 512), (1024, 512), (257, 2048), (7, 4095),
          (1, 4096), (64, 4096), (65, 4096), (2, 32, 4096), (2, 4, 8, 4096), (257, 4096), (1024, 4096),
          (257, 4100), (257, 7168), (7, 4097), (4, 8, 192), (2, 4, 8, 256)]


def reference(x, weight, eps, zero_centered):
    compute = x.float()
    out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    torch.ops._C.rms_norm(out, compute, weight, eps, zero_centered)
    return out.to(x.dtype)


def assert_bits(actual, expected):
    a = actual.contiguous().view(torch.uint8).flatten()
    b = expected.contiguous().view(torch.uint8).flatten()
    bad = (a != b).nonzero().flatten()
    assert bad.numel() == 0, f"{bad.numel()} differing bytes; first positions {bad[:8].tolist()}"


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('zero_centered', (False, True))
@pytest.mark.parametrize('shape', SHAPES)
def test_mixed_dispatch(shape, dtype, zero_centered):
    torch.manual_seed(731)
    x = torch.randn(shape, device='cuda', dtype=dtype)
    weight = torch.randn(shape[-1], device='cuda', dtype=torch.float32)
    expected = reference(x, weight, 1e-6, zero_centered)
    actual = torch.empty_like(x)
    torch.ops._C.rms_norm(actual, x, weight, 1e-6, zero_centered)
    assert_bits(actual, expected)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('zero_centered', (False, True))
@pytest.mark.parametrize('hidden', (192, 4096))
@pytest.mark.parametrize('layout', ('input_offset', 'row_padding', 'inner_stride', 'out_offset', 'weight_offset'))
def test_mixed_layout(hidden, layout, dtype, zero_centered):
    torch.manual_seed(947)
    n = 1024
    x = torch.randn((n, hidden), device='cuda', dtype=dtype)
    weight = torch.randn(hidden, device='cuda', dtype=torch.float32)
    out = torch.empty_like(x)
    if layout == 'input_offset':
        x = torch.randn(n * hidden + 1, device='cuda', dtype=dtype)[1:].view(n, hidden)
    elif layout == 'row_padding':
        x = torch.randn((n, hidden + 1), device='cuda', dtype=dtype)[:, :hidden]
    elif layout == 'inner_stride':
        x = torch.randn((n, hidden * 2), device='cuda', dtype=dtype)[:, ::2]
    elif layout == 'out_offset':
        out = torch.empty(n * hidden + 1, device='cuda', dtype=dtype)[1:].view(n, hidden)
    elif layout == 'weight_offset':
        weight = torch.randn(hidden + 1, device='cuda', dtype=torch.float32)[1:]
    expected = reference(x, weight, 1e-6, zero_centered)
    torch.ops._C.rms_norm(out, x, weight, 1e-6, zero_centered)
    assert_bits(out, expected)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', ((0, 192), (0, 4096), (3, 0, 193)))
@pytest.mark.parametrize('zero_centered', (False, True))
def test_mixed_empty(shape, dtype, zero_centered):
    x = torch.empty(shape, device='cuda', dtype=dtype)
    out = torch.empty_like(x)
    weight = torch.ones(shape[-1], device='cuda', dtype=torch.float32)
    torch.ops._C.rms_norm(out, x, weight, 1e-6, zero_centered)
    torch.cuda.synchronize()
    assert out.shape == x.shape and out.dtype == dtype


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', ((17, 192), (3, 8, 192), (2, 4, 8, 256)))
@pytest.mark.parametrize('zero_centered', (False, True))
def test_mixed_per_batch_weight(shape, dtype, zero_centered):
    torch.manual_seed(74)
    x = torch.randn(shape, device='cuda', dtype=dtype)
    weight = torch.randn((shape[0], shape[-1]), device='cuda', dtype=torch.float32)
    expected = reference(x, weight, 1e-6, zero_centered)
    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out, x, weight, 1e-6, zero_centered)
    assert_bits(out, expected)


@pytest.mark.parametrize('dtype', DTYPES)
def test_mixed_validation(dtype):
    x = torch.ones((7, 192), device='cuda', dtype=dtype)
    weight = torch.ones(192, device='cuda', dtype=torch.float32)
    storage = torch.full((7, 384), 123, device='cuda', dtype=dtype)
    with pytest.raises(RuntimeError, match='contiguous'):
        torch.ops._C.rms_norm(storage[:, ::2], x, weight, 1e-6, False)
    assert bool((storage == 123).all())
    with pytest.raises(RuntimeError, match='same dtype and shape'):
        torch.ops._C.rms_norm(torch.empty((7, 191), device='cuda', dtype=dtype), x, weight, 1e-6, False)
    with pytest.raises(RuntimeError, match='same dtype and shape'):
        torch.ops._C.rms_norm(torch.empty_like(x, dtype=torch.float32), x, weight, 1e-6, False)
    empty_hidden = torch.empty((7, 0), device='cuda', dtype=dtype)
    with pytest.raises(RuntimeError, match='hidden size must be positive'):
        torch.ops._C.rms_norm(empty_hidden, empty_hidden, weight[:0], 1e-6, False)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('zero_centered', (False, True))
@pytest.mark.parametrize('pattern', ('signed_zero', 'range', 'small', 'nonfinite'))
@pytest.mark.parametrize('shape', ((64, 192), (1024, 192), (1024, 128), (64, 4096), (7, 193), (257, 4100), (17, 6)))
def test_mixed_rounding_patterns(dtype, zero_centered, pattern, shape):
    n, h = shape
    if pattern == 'signed_zero':
        values = [0.0, -0.0]
    elif pattern == 'nonfinite':
        values = [float('inf'), -float('inf'), float('nan'), 1.0, -1.0, 0.0]
    elif pattern == 'small':
        tiny = torch.finfo(dtype).tiny
        values = [tiny, -tiny, tiny / 2, -tiny / 2, 0.0, 1.0]
    else:
        values = [2.0 ** power for power in (-10, -5, 0, 5, 10)] + [-0.5]
    x = torch.tensor(values, device='cuda', dtype=dtype).repeat((n * h + len(values) - 1) // len(values))[:n * h].view(n, h)
    weight = torch.tensor([-1.0, -0.99999994, -1.00000012, 0.0, 1.0, 0.1], device='cuda', dtype=torch.float32).repeat((h + 5) // 6)[:h]
    expected = reference(x, weight, 1e-6, zero_centered)
    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out, x, weight, 1e-6, zero_centered)
    assert_bits(out, expected)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', ((1025, 192), (8192, 4096)))
def test_mixed_graph_replay(shape, dtype):
    torch.manual_seed(312)
    x = torch.randn(shape, device='cuda', dtype=dtype)
    weight = torch.randn(shape[-1], device='cuda', dtype=torch.float32)
    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out, x, weight, 1e-5, True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.rms_norm(out, x, weight, 1e-5, True)
    for _ in range(3):
        x.normal_()
        graph.replay()
        assert_bits(out, reference(x, weight, 1e-5, True))
