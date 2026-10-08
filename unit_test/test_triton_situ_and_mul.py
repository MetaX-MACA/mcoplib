"""Fixed accuracy and performance comparison for MUXI C500."""

import torch
from mcoplib.triton_situ_and_mul import situ_and_mul, situ_and_mul_baseline


TOKEN_COUNTS = (2048, 4096, 8192, 16384)
TOPK = 16
WIDTH = 768
DTYPE = torch.bfloat16
SITU_BETA = 4.0
LINEAR_BETA = 4.0
WARMUP = 20
REPETITIONS = 100
REFERENCE_CHUNK_ROWS = 4096
ATOL = 3.0e-2
RTOL = 3.0e-2


def torch_reference(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.float().chunk(2, dim=-1)
    gate = SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
    up = LINEAR_BETA * torch.tanh(up / LINEAR_BETA)
    return (gate * up).to(x.dtype)


@torch.no_grad()
def check_accuracy(x: torch.Tensor, baseline: torch.Tensor, optimized: torch.Tensor):
    unchanged = torch.equal(baseline, optimized)
    max_error = 0.0
    accurate = True
    for start in range(0, x.shape[0], REFERENCE_CHUNK_ROWS):
        end = min(start + REFERENCE_CHUNK_ROWS, x.shape[0])
        reference = torch_reference(x[start:end])
        result = optimized[start:end]
        max_error = max(
            max_error, (result.float() - reference.float()).abs().max().item()
        )
        accurate &= torch.allclose(result, reference, atol=ATOL, rtol=RTOL)
    return unchanged and accurate, max_error


@torch.no_grad()
def benchmark(operation, x: torch.Tensor, out: torch.Tensor):
    for _ in range(WARMUP):
        operation(x, SITU_BETA, LINEAR_BETA, out=out)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPETITIONS):
        operation(x, SITU_BETA, LINEAR_BETA, out=out)
    end.record()
    torch.cuda.synchronize()

    milliseconds = start.elapsed_time(end) / REPETITIONS
    bytes_moved = (x.numel() + out.numel()) * x.element_size()
    bandwidth = bytes_moved / milliseconds / 1.0e6
    return milliseconds, bandwidth


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch未检测到C500设备")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    print(f"设备：{torch.cuda.get_device_name(0)}；BF16；Top-K={TOPK}")
    print(
        "Token数 | 精度 | 最大误差 | 原耗时(ms) | 新耗时(ms) | 加速比 | "
        "原带宽(GB/s) | 新带宽(GB/s) | 带宽提升"
    )

    all_passed = True
    for tokens in TOKEN_COUNTS:
        rows = tokens * TOPK
        x = torch.randn((rows, WIDTH), device="cuda", dtype=DTYPE)
        baseline_out = torch.empty((rows, WIDTH // 2), device="cuda", dtype=DTYPE)
        optimized_out = torch.empty_like(baseline_out)

        situ_and_mul_baseline(x, SITU_BETA, LINEAR_BETA, out=baseline_out)
        situ_and_mul(x, SITU_BETA, LINEAR_BETA, out=optimized_out)
        torch.cuda.synchronize()
        passed, max_error = check_accuracy(x, baseline_out, optimized_out)

        old_ms, old_bw = benchmark(
            situ_and_mul_baseline, x, baseline_out
        )
        new_ms, new_bw = benchmark(situ_and_mul, x, optimized_out)
        speedup = old_ms / new_ms
        bandwidth_gain = (new_bw / old_bw - 1.0) * 100.0
        all_passed &= passed

        print(
            f"{tokens:7d} | {'通过' if passed else '失败'} | {max_error:.2e} | "
            f"{old_ms:10.3f} | {new_ms:10.3f} | {speedup:6.2f}x | "
            f"{old_bw:12.1f} | {new_bw:12.1f} | {bandwidth_gain:7.1f}%"
        )

        del x, baseline_out, optimized_out
        torch.cuda.empty_cache()

    print("测试结果：", "通过" if all_passed else "失败", sep="")
    return all_passed


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
