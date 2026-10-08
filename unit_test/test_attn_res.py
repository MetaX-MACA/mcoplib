# SPDX-License-Identifier: Apache-2.0

import math
import time

import pytest
import torch
import torch.nn.functional as F

import mcoplib._C


HIDDEN_SIZE = 7168
MAX_BLOCKS = 8
EPS = 1e-5

# Metax C500 HBM2e theoretical peak bandwidth (per the C500 optimization
# guide reference shipped with the optimized-cuda-kernels skill).
C500_THEORETICAL_BW_GBPS = 1550.0  # 1.55 TB/s


def randn(*shape):

    return torch.randn(
        *shape,
        device="cuda",
        dtype=torch.bfloat16,
    )



def reference(
    prefix,
    delta,
    blocks,
    norm_weight,
    qk_weight,
    output_norm_weight,
    num_blocks,
):

    if delta.numel() != 0:
        prefix = prefix + delta


    values = torch.cat(
        [
            blocks[:, :num_blocks],
            prefix.unsqueeze(1),
        ],
        dim=1,
    )


    keys = F.rms_norm(
        values,
        (HIDDEN_SIZE,),
        norm_weight,
        EPS,
    )


    probs = (
        keys @ qk_weight
    ).softmax(dim=-1)


    output = torch.matmul(
        probs.unsqueeze(1),
        values,
    ).squeeze(1)


    if output_norm_weight.numel() != 0:
        output = F.rms_norm(
            output,
            (HIDDEN_SIZE,),
            output_norm_weight,
            EPS,
        )


    return output



def _bytes_per_token(num_blocks, has_delta):

    # Reads per token: num_blocks block rows + 1 prefix row (+ delta if any).
    read_rows = num_blocks + 1 + (1 if has_delta else 0)
    read_bytes = read_rows * HIDDEN_SIZE * 2  # bf16
    # Writes per token: output row (+ prefix row write-back if delta present).
    write_rows = 1 + (1 if has_delta else 0)
    write_bytes = write_rows * HIDDEN_SIZE * 2
    return read_bytes + write_bytes



def _measure_kernel(
    prefix,
    delta,
    blocks,
    norm_weight,
    qk_weight,
    output_norm_weight,
    output,
    num_blocks,
    iters=100,
    warmup=10,
):

    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        torch.ops._C.kimi_k3_attn_res(
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            output,
            num_blocks,
            EPS,
            EPS,
        )
    stream.synchronize()

    start.record(stream)
    for _ in range(iters):
        torch.ops._C.kimi_k3_attn_res(
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            output,
            num_blocks,
            EPS,
            EPS,
        )
    end.record(stream)
    stream.synchronize()
    return start.elapsed_time(end) / iters  # ms per call



def _print_benchmark(
    num_tokens,
    num_blocks,
    has_delta,
    avg_time_ms,
):

    total_bytes = _bytes_per_token(num_blocks, has_delta) * num_tokens
    total_gb = total_bytes / 1e9
    avg_time_sec = avg_time_ms / 1000.0
    achieved_gbps = total_gb / avg_time_sec if avg_time_sec > 0 else 0.0
    efficiency = (achieved_gbps / C500_THEORETICAL_BW_GBPS) * 100.0
    print(
        "\n===================== kimi_k3_attn_res benchmark ===================="
    )
    print(f"  num_tokens       : {num_tokens}")
    print(f"  num_blocks       : {num_blocks}")
    print(f"  has_delta        : {has_delta}")
    print(f"  hidden_size      : {HIDDEN_SIZE}")
    print(f"  N (value rows)   : {num_blocks + 1}")
    print(f"  avg latency      : {avg_time_ms:.4f} ms")
    print(f"  total data moved : {total_gb:.6f} GB")
    print(f"  achieved bw      : {achieved_gbps:.2f} GB/s")
    print(f"  theoretical bw   : {C500_THEORETICAL_BW_GBPS:.2f} GB/s (C500)")
    print(f"  bw efficiency    : {efficiency:.2f}%")
    print("=====================================================================\n")



def _cosine_similarity(a, b):

    a_f = a.flatten().to(torch.float32)
    b_f = b.flatten().to(torch.float32)
    dot = (a_f * b_f).sum()
    na = a_f.norm()
    nb = b_f.norm()
    return float((dot / (na * nb + 1e-12)).item())



@pytest.mark.parametrize(
    (
        "num_tokens",
        "num_blocks",
        "has_delta",
    ),
    [
        (1,0,False),
        (1,0,True),
        (17,5,False),
        (17,5,True),
        (3,8,False),
        (3,8,True),
        (320,1,True),
        (320,4,True),
    ],
)
def test_kimi_k3_attn_res(
    num_tokens,
    num_blocks,
    has_delta,
):


    prefix = randn(
        num_tokens,
        HIDDEN_SIZE,
    )


    if has_delta:

        delta = randn(
            num_tokens,
            HIDDEN_SIZE,
        )

    else:

        delta = torch.empty(
            0,
            device="cuda",
            dtype=torch.bfloat16,
        )


    blocks = randn(
        num_tokens,
        MAX_BLOCKS,
        HIDDEN_SIZE,
    )


    norm_weight = (
        1
        +
        0.1 *
        torch.randn(
            HIDDEN_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
    )


    qk_weight = (
        torch.randn(
            HIDDEN_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        /
        HIDDEN_SIZE**0.5
    )


    output_norm_weight = (
        1
        +
        0.1 *
        torch.randn(
            HIDDEN_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
    )


    expected = reference(
        prefix.clone(),
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
    )


    # 必须提前分配
    output = torch.empty(
        num_tokens,
        HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )


    torch.ops._C.kimi_k3_attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        output,
        num_blocks,
        EPS,
        EPS,
    )


    torch.testing.assert_close(
        output,
        expected,
        atol=8e-2,
        rtol=3e-2,
    )

    # Cosine-similarity gate: kernel output vs torch reference must be >=0.999.
    cos = _cosine_similarity(output, expected)
    print(
        f"[cosine-sim] tokens={num_tokens} blocks={num_blocks} "
        f"delta={has_delta} cos={cos:.6f}"
    )
    assert cos >= 0.999, (
        f"cosine similarity {cos:.6f} below 0.999 threshold "
        f"(tokens={num_tokens}, blocks={num_blocks}, delta={has_delta})"
    )

    assert output.is_contiguous()

    # Performance: latency + bandwidth measurement.
    avg_ms = _measure_kernel(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        output,
        num_blocks,
    )
    _print_benchmark(num_tokens, num_blocks, has_delta, avg_ms)



@pytest.mark.parametrize(
    "num_blocks",
    range(MAX_BLOCKS+1),
)
def test_kimi_k3_attn_res_block_counts(
    num_blocks,
):


    prefix = randn(
        1,
        HIDDEN_SIZE,
    )


    delta = torch.empty(
        0,
        device="cuda",
        dtype=torch.bfloat16,
    )


    blocks = randn(
        1,
        MAX_BLOCKS,
        HIDDEN_SIZE,
    )


    norm_weight = torch.ones(
        HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )


    qk_weight = (
        torch.randn(
            HIDDEN_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        /
        HIDDEN_SIZE**0.5
    )


    output_norm_weight = torch.ones_like(
        norm_weight
    )


    expected = reference(
        prefix.clone(),
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
    )


    output = torch.empty(
        1,
        HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )


    torch.ops._C.kimi_k3_attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        output,
        num_blocks,
        EPS,
        EPS,
    )


    torch.testing.assert_close(
        output,
        expected,
        atol=8e-2,
        rtol=3e-2,
    )

    cos = _cosine_similarity(output, expected)
    print(
        f"[cosine-sim block_counts] blocks={num_blocks} cos={cos:.6f}"
    )
    assert cos >= 0.999, (
        f"cosine similarity {cos:.6f} below 0.999 threshold "
        f"(blocks={num_blocks})"
    )
