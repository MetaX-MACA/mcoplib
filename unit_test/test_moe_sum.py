import pytest
import torch
import torch.nn.functional as F
from torch.library import opcheck
import mcoplib._moe_C

TOP_KS = [1, 2, 4]


def cosine_similarity(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def calc_bandwidth(m, topk, k, dtype, time_ms):
    bytes_size = torch.tensor([], dtype=dtype).element_size()
    read_bytes = m * topk * k * bytes_size
    write_bytes = m * k * bytes_size
    return (read_bytes + write_bytes) / (time_ms * 1e-3) / 1e9


def benchmark(func, warmup=10, repeat=100):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    total = 0
    for _ in range(repeat):
        start.record()
        func()
        end.record()
        torch.cuda.synchronize()
        total += start.elapsed_time(end)

    return total / repeat


@pytest.mark.parametrize("m", [1, 33, 222])
@pytest.mark.parametrize("topk", [*TOP_KS, 8])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_moe_sum(m, topk, k, dtype):

    input = torch.randn(
        (m, topk, k),
        device="cuda",
        dtype=dtype
    )

    assert input.is_contiguous()

    actual = torch.empty(
        (m, k),
        device="cuda",
        dtype=dtype
    )

    expected = input.float().sum(dim=1).to(dtype)

    torch.ops._moe_C.moe_sum(input, actual)

    torch.cuda.synchronize()

    cos = cosine_similarity(actual, expected)

    print("\n" + "=" * 60)
    print(
        f"m={m}, topk={topk}, k={k}, dtype={dtype}"
    )
    print(f"cos={cos:.8f}")

    assert cos > 0.9999

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=6e-2,
        rtol=1e-3
    )

    print("Precision PASS")

    def run():
        torch.ops._moe_C.moe_sum(input, actual)

    time_ms = benchmark(run)

    bw = calc_bandwidth(
        m,
        topk,
        k,
        dtype,
        time_ms
    )

    print(
        f"time={time_ms:.4f} ms "
        f"bandwidth={bw:.2f} GB/s"
    )

    print("PASS")