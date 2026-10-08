"""Unit test + bandwidth benchmark for torch.ops.sgl_kernel.moe_sum_reduce on MetaX C600-U.

Op:
  moe_sum_reduce(Tensor input, Tensor output, float routed_scaling_factor) -> ()
Semantics:
  input  : [token_num, topk_num, hidden_dim]  bf16
  output : [token_num, hidden_dim]            bf16
  output[t, :] = ( sum_k input[t, k, :].float() ) * routed_scaling_factor  -> bf16

Roofline (bf16, topk=4):
  read  = topk * 2 B  = 8 B / out-elem
  write = 1    * 2 B  = 2 B / out-elem
  total = (topk+1) * 2 B = 10 B / out-elem   (4R:1W streaming, AI~0.4 op/B -> memory-bound)

Accuracy: cosine similarity vs torch fp32 reference, threshold >= 0.9999.
Bandwidth printed per shape. Free GPU auto-selected via mx-smi (or honor preset CUDA_VISIBLE_DEVICES).

Run:
  python unit_test/test_op_sglang_moe_sum_reduce.py
"""

import gc
import os
import subprocess
import torch


# ---------------------------------------------------------------------------
# Free-GPU auto-selection (respect preset CUDA_VISIBLE_DEVICES if given)
# ---------------------------------------------------------------------------
def _select_free_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    try:
        out = subprocess.check_output(["mx-smi"], stderr=subprocess.DEVNULL).decode()
    except Exception:
        return
    # Find physical GPU indices whose usage line shows 0% and Available.
    free = []
    cur = None
    for line in out.splitlines():
        # a GPU header line looks like: | 5  MetaX C600-UL | 10  On | ... | 0%  Disabled |
        parts = line.split("|")
        if len(parts) > 2 and "MetaX" in parts[1]:
            toks = parts[1].split()
            if toks and toks[0].isdigit():
                cur = int(toks[0])
        if cur is not None and "Available" in line and "0%" in line:
            if cur not in free:
                free.append(cur)
    if free:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(free[0])
        print(f"[gpu-select] mx-smi free GPUs {free} -> CUDA_VISIBLE_DEVICES={free[0]}")


_select_free_gpu()

import mcoplib.sgl_kernel  # noqa: E402  registers torch.ops.sgl_kernel.*

DEVICE = "cuda"
DTYPE = torch.bfloat16
ELEM_BYTES = 2  # bf16
COS_THRESHOLD = 0.9999

DISTRIBUTED_TOPOLOGIES = ((4, 8), (8, 4))
DISTRIBUTED_BATCH_SIZES = tuple(2 ** i for i in range(0, 9, 4))

TEST_CONFIGS = (
    # name, topk, hidden, scale, token cases, exact, PP/TP topologies, batch sizes
    (
        "baseline",
        4,
        6144,
        1.0,
        (2048, 3072, 4096, 8192, 16384, 32768),
        False,
        (),
        (),
    ),
    (
        "topk16-hidden3584",
        16,
        3584,
        0.3,
        (2048, 3072, 4096, 8192, 16384),
        True,
        DISTRIBUTED_TOPOLOGIES,
        DISTRIBUTED_BATCH_SIZES,
    ),
    (
        "topk32-hidden1024",
        32,
        1024,
        0.3,
        (3072, 8192, 16384),
        True,
        (),
        (),
    ),
    (
        "dynamic-topk17-hidden1024",
        17,
        1024,
        0.3,
        (3072, 8192),
        True,
        (),
        (),
    ),
    (
        "dynamic-topk24-hidden2048",
        24,
        2048,
        0.3,
        (3072, 8192),
        True,
        (),
        (),
    ),
    (
        "dynamic-topk-boundary",
        5,
        3584,
        0.3,
        (1, 128, 129, 256, 257),
        True,
        (),
        (),
    ),
    (
        "topk16-boundary",
        16,
        3584,
        0.3,
        (1, 128, 129, 256, 257),
        True,
        (),
        (),
    ),
)

NATIVE_WARP_SIZE = 64
BALANCED_WARPS_PER_BLOCK = 4
WIDE_WARPS_PER_BLOCK = 8
SCHEDULING_WAVES = 2
MIN_TOPK_FOR_PACKED_BLOCKS = 16
MIN_REDUCTION_ELEMENTS_FOR_BALANCED_BLOCKS = 32768
MAX_BALANCED_SCHEDULING_WAVES_FOR_MODERATE_FAN_IN = 64

# HBM walls measured on this die (fp8-probe calibration, same die):
#   pure read ~1630, pure write ~1626, 1R1W copy ~1524, 2R1W ~1537 GB/s.
# 4R:1W streaming wall is measured separately; datasheet single-die = 1800.
SINGLE_DIE_DATASHEET = 1800.0
DUAL_DIE_DATASHEET = 3600.0


def moe_sum_reduce_ref(x, scale):
    """PyTorch reference: fp32 accumulate over topk, scale, cast back to bf16."""
    return (x.float().sum(dim=1) * scale).to(x.dtype)


def cosine_sim(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    dot = torch.dot(a, b).double()
    na = torch.dot(a, a).double().sqrt()
    nb = torch.dot(b, b).double().sqrt()
    return (dot / (na * nb + 1e-30)).item()


def bench(fn, warmup=15, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms = torch.tensor([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return ms.min().item()  # best-of, matches roofline-probe methodology


def split_full_chunk(batch_size, token_count):
    """Create a deterministic full-chunk request-length distribution."""
    quotient, remainder = divmod(token_count, batch_size)
    return tuple(
        quotient + (request_index < remainder)
        for request_index in range(batch_size)
    )


def dispatch_geometry(token_count, topk, hidden):
    """Mirror the host dispatch calculation and return grid/block geometry."""
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    grid_x = min(
        ((hidden // 8) + NATIVE_WARP_SIZE - 1) // NATIVE_WARP_SIZE,
        65535,
    )

    reduction_elements_per_token = topk * hidden
    packed_warps_per_block = (
        BALANCED_WARPS_PER_BLOCK
        if reduction_elements_per_token
        >= MIN_REDUCTION_ELEMENTS_FOR_BALANCED_BLOCKS
        else WIDE_WARPS_PER_BLOCK
    )
    if packed_warps_per_block == BALANCED_WARPS_PER_BLOCK and topk < 32:
        balanced_threads = BALANCED_WARPS_PER_BLOCK * NATIVE_WARP_SIZE
        resident_balanced_blocks_per_sm = max(
            properties.max_threads_per_multi_processor // balanced_threads,
            1,
        )
        balanced_grid_y = (
            token_count + BALANCED_WARPS_PER_BLOCK - 1
        ) // BALANCED_WARPS_PER_BLOCK
        balanced_block_count = grid_x * balanced_grid_y
        balanced_full_device_wave_blocks = (
            properties.multi_processor_count
            * resident_balanced_blocks_per_sm
        )
        if balanced_block_count > (
            MAX_BALANCED_SCHEDULING_WAVES_FOR_MODERATE_FAN_IN
            * balanced_full_device_wave_blocks
        ):
            packed_warps_per_block = WIDE_WARPS_PER_BLOCK
    packed_threads = packed_warps_per_block * NATIVE_WARP_SIZE
    resident_packed_blocks_per_sm = max(
        properties.max_threads_per_multi_processor // packed_threads,
        1,
    )
    packed_grid_y = (
        token_count + packed_warps_per_block - 1
    ) // packed_warps_per_block
    packed_block_count = grid_x * packed_grid_y
    full_device_wave_blocks = (
        properties.multi_processor_count * resident_packed_blocks_per_sm
    )
    use_packed_warps = (
        topk >= MIN_TOPK_FOR_PACKED_BLOCKS
        and packed_block_count > SCHEDULING_WAVES * full_device_wave_blocks
    )

    warps_per_block = packed_warps_per_block if use_packed_warps else 1
    if token_count > warps_per_block * 65535:
        warps_per_block = WIDE_WARPS_PER_BLOCK
    grid_y = min(
        (token_count + warps_per_block - 1) // warps_per_block,
        65535,
    )
    return (
        (grid_x, grid_y, 1),
        (warps_per_block * NATIVE_WARP_SIZE, 1, 1),
        use_packed_warps,
    )


def main():
    dev_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES='{dev_env}'  device_count={torch.cuda.device_count()}")
    torch.manual_seed(1234)
    all_pass = True
    config_matrix_checked = 0
    expected_config_matrix = sum(
        len(token_cases) * len(topologies) * len(batch_sizes)
        for (
            _,
            _,
            _,
            _,
            token_cases,
            _,
            topologies,
            batch_sizes,
        ) in TEST_CONFIGS
    )

    assert dispatch_geometry(3072, 16, 1024)[1] == (512, 1, 1)
    assert dispatch_geometry(3072, 16, 3584)[1] == (256, 1, 1)
    assert dispatch_geometry(8192, 16, 3584)[1] == (256, 1, 1)
    assert dispatch_geometry(8192, 16, 4096)[1] == (512, 1, 1)
    assert dispatch_geometry(16384, 16, 3584)[1] == (512, 1, 1)
    assert dispatch_geometry(3072, 32, 1024)[1] == (256, 1, 1)
    assert dispatch_geometry(16384, 32, 1024)[1] == (256, 1, 1)

    for (
        TEST_NAME,
        TOPK,
        HIDDEN,
        SCALE,
        TOKEN_CASES,
        REQUIRE_EXACT,
        TOPOLOGIES,
        BATCH_SIZES,
    ) in TEST_CONFIGS:
        print("=" * 92)
        print(
            f"config={TEST_NAME} topk={TOPK} hidden={HIDDEN} "
            f"dtype=bf16->bf16 scale={SCALE}"
        )
        print(
            f"traffic = (topk+1)*{ELEM_BYTES} = "
            f"{(TOPK + 1) * ELEM_BYTES} B / out-elem "
            f"({TOPK}R:1W)"
        )
        print("=" * 92)

        config_pass = True
        peak = 0.0

        for T in TOKEN_CASES:
            x = torch.randn(T, TOPK, HIDDEN, dtype=DTYPE, device=DEVICE)
            y = torch.empty(T, HIDDEN, dtype=DTYPE, device=DEVICE)

            # correctness
            torch.ops.sgl_kernel.moe_sum_reduce(x, y, SCALE)
            torch.cuda.synchronize()
            ref = moe_sum_reduce_ref(x, SCALE)
            cs = cosine_sim(y, ref)
            ok = cs >= COS_THRESHOLD
            if REQUIRE_EXACT:
                torch.testing.assert_close(y, ref, rtol=0, atol=0)
            config_pass &= ok
            all_pass &= ok

            # bandwidth: total HBM bytes = read topk + write 1, all bf16
            total_bytes = T * HIDDEN * (TOPK + 1) * ELEM_BYTES
            ms = bench(lambda: torch.ops.sgl_kernel.moe_sum_reduce(x, y, SCALE))
            gbps = total_bytes / (ms * 1e-3) / 1e9
            peak = max(peak, gbps)

            if TOPOLOGIES:
                grid, block, uses_packed_warps = dispatch_geometry(
                    T,
                    TOPK,
                    HIDDEN,
                )
                assert uses_packed_warps
                expected_warps_per_block = (
                    BALANCED_WARPS_PER_BLOCK
                    if T <= 8192
                    else WIDE_WARPS_PER_BLOCK
                )
                assert block == (
                    expected_warps_per_block * NATIVE_WARP_SIZE,
                    1,
                    1,
                )

                for pp_size, tp_size in TOPOLOGIES:
                    assert pp_size * tp_size == 32
                    for batch_size in BATCH_SIZES:
                        request_lengths = split_full_chunk(batch_size, T)
                        assert len(request_lengths) == batch_size
                        assert min(request_lengths) > 0
                        assert sum(request_lengths) == T
                        config_matrix_checked += 1

                        print(
                            f"[moe_sum_reduce] T={T:6d} "
                            f"pp={pp_size} tp={tp_size} "
                            f"batch={batch_size:3d} "
                            f"input={tuple(x.shape)} "
                            f"output={tuple(y.shape)} "
                            f"grid={grid} block={block}  "
                            f"cos_sim={cs:.6f} "
                            f"{'OK  ' if ok else 'FAIL'}"
                            f"  {ms:8.4f} ms  {gbps:8.1f} GB/s"
                        )
            else:
                print(
                    f"[moe_sum_reduce] T={T:6d}  "
                    f"cos_sim={cs:.6f} "
                    f"{'OK  ' if ok else 'FAIL'}"
                    f"  {ms:8.4f} ms  {gbps:8.1f} GB/s"
                )

            del x, y, ref
            gc.collect()
            torch.cuda.empty_cache()

        print("=" * 92)
        print(
            f"{TEST_NAME} accuracy: "
            f"{'ALL PASS' if config_pass else 'FAILURE'} "
            f"(threshold cos_sim >= {COS_THRESHOLD})"
        )
        tgt = SINGLE_DIE_DATASHEET
        reached = "reached" if peak >= 0.85 * tgt else "NOT reached"
        print(
            f"{TEST_NAME} peak bandwidth: {peak:.1f} GB/s "
            f"(single-die datasheet {tgt:.0f}, "
            f"85% target {0.85*tgt:.0f} -> {reached}; "
            f"dual-die datasheet {DUAL_DIE_DATASHEET:.0f})"
        )

    assert config_matrix_checked == expected_config_matrix, (
        f"checked {config_matrix_checked} distributed combinations, "
        f"expected {expected_config_matrix}"
    )
    print(
        f"Distributed config matrix: combinations={config_matrix_checked} "
        f"topologies={DISTRIBUTED_TOPOLOGIES} "
        f"batch_sizes={DISTRIBUTED_BATCH_SIZES}"
    )
    return all_pass


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
