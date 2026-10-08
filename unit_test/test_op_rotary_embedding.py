"""
Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

Unit test for mcoplib.op.rotary_embedding (default op).

Covers the exact parameter configuration specified:
  q_head_num.kv_head_num.head_dim = [80, 8, 128]
  rope_offset.rope_dim = [0, 128]
  prefill batch_size.cache_len.q_len: 12 configs
  decode batch_size.cache_len.q_len: 13 configs
"""

import os
import sys
import random
import torch
import unittest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from mcoplib.op import rotary_embedding
from measure_cuda import measure_cuda


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def cosine_similarity(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    dot = torch.sum(a_flat * b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    norm_a = norm_a if norm_a.item() > 1e-8 else torch.tensor(1.0, device=a.device)
    norm_b = norm_b if norm_b.item() > 1e-8 else torch.tensor(1.0, device=b.device)
    return (dot / (norm_a * norm_b)).item()


def build_rope_table(cache_len, rope_dim, head_dim, device):
    """Build cos/sin tables for Neox-style RoPE.

    The kernel accesses cos_base + cache_idx * ROPE_DIM + i for i in [0, ROPE_DIM/2).
    So the cos and sin tables each have ROPE_DIM elements per position, but only
    the first ROPE_DIM/2 elements are used — both halves of the rotation read from
    the same first-half slice.
    """
    half = rope_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / head_dim))
    pos = torch.arange(cache_len, dtype=torch.float32)
    angles = pos.unsqueeze(1) * inv_freq.unsqueeze(0)  # [cache_len, half]
    cos = torch.zeros(cache_len, rope_dim, dtype=torch.float32)
    sin = torch.zeros(cache_len, rope_dim, dtype=torch.float32)
    cos[:, :half] = torch.cos(angles)
    sin[:, :half] = torch.sin(angles)
    return cos.to(device).contiguous(), sin.to(device).contiguous()


def reference_rotary_embedding(packed_qkv, accum_q_lens, cache_lens, cos, sin,
                               q_head_num, kv_head_num, rope_offset):
    num_tokens, total_head_num, head_dim = packed_qkv.shape
    rope_dim = cos.size(-1)
    half = rope_dim // 2

    out = packed_qkv.clone().float()
    accum = accum_q_lens.to(device=packed_qkv.device)
    cache = cache_lens.to(device=packed_qkv.device)
    slice_idx = torch.arange(num_tokens, device=packed_qkv.device)
    bins = torch.searchsorted(accum, slice_idx, right=True) - 1
    cache_idx = cache[bins] + (slice_idx - accum[bins])

    cos_c = cos[cache_idx]
    sin_c = sin[cache_idx]

    # Kernel uses ONLY the first half [0, ROPE_DIM/2) of the cos/sin table for
    # BOTH rotation halves. So c and s below are the first-half slices.
    c = cos_c[:, :half].unsqueeze(1)  # [num_tokens, 1, half]
    s = sin_c[:, :half].unsqueeze(1)

    x = out[:, :, rope_offset:rope_offset + half]
    y = out[:, :, rope_offset + half:rope_offset + rope_dim]
    new_x = x * c - y * s
    new_y = y * c + x * s
    out[:, :, rope_offset:rope_offset + half] = new_x
    out[:, :, rope_offset + half:rope_offset + rope_dim] = new_y
    return out.to(packed_qkv.dtype)


def run_rotary_case(q_head_num, kv_head_num, head_dim, rope_dim, rope_offset,
                    batch_size, cache_len, q_len, test_name):
    """Run one rotary_embedding config; verify cos-sim and print bandwidth."""
    set_seed(42)
    device = 0
    torch.cuda.set_device(device)

    total_head_num = q_head_num + kv_head_num
    num_tokens = q_len

    packed_qkv = (torch.randn(num_tokens, total_head_num, head_dim, dtype=torch.bfloat16) * 0.5).cuda()
    accum_q_lens = torch.zeros(batch_size + 1, dtype=torch.int32).cuda()
    cum = 0
    for b in range(batch_size):
        cum += q_len
        accum_q_lens[b + 1] = cum
    cache_lens_t = torch.full((batch_size,), cache_len, dtype=torch.int32).cuda()
    # cos/sin table must cover cache_len + q_len positions (kernel accesses cache_idx = cache_len + local_token_id)
    cos, sin = build_rope_table(cache_len + q_len, rope_dim, head_dim, 'cuda')

    golden = reference_rotary_embedding(
        packed_qkv, accum_q_lens, cache_lens_t, cos, sin,
        q_head_num, kv_head_num, rope_offset)

    def kernel():
        rotary_embedding(packed_qkv, accum_q_lens, accum_q_lens, cache_lens_t,
                         cos, sin, q_head_num, kv_head_num, rope_offset)

    kernel()
    torch.cuda.synchronize()

    out = packed_qkv
    sim = cosine_similarity(out, golden)
    byte_count = 2 * num_tokens * total_head_num * head_dim * 2
    # Arithmetic intensity: each rotated element position does 3 FLOPs
    # (new_x = x*c - y*s, new_y = y*c + x*s -> 2 mul + 1 add/sub per output,
    #  applied to rope_dim elements per token-head).
    flop_count = num_tokens * total_head_num * rope_dim * 3
    arithmetic_intensity = flop_count / byte_count

    stats = measure_cuda(kernel, iters=200, warmup=20, device=device)
    time_us = stats['mean_us']
    bw_gbps = byte_count / (time_us * 1e-6) / 1e9
    flop_gflops = flop_count / (time_us * 1e-6) / 1e9

    print(f"[{test_name}] tokens={num_tokens} total_heads={total_head_num} "
          f"head_dim={head_dim} rope_dim={rope_dim} rope_offset={rope_offset} "
          f"batch={batch_size} cache_len={cache_len} q_len={q_len} dtype=bf16")
    print(f"  mean={time_us:.3f}us median={stats['median_us']:.3f}us "
          f"min={stats['min_us']:.3f}us stdev={stats['stdev_us']:.3f}us")
    print(f"  bandwidth={bw_gbps:.2f} GB/s  (bytes moved={byte_count})")
    print(f"  arithmetic_intensity={arithmetic_intensity:.4f} FLOP/B  "
          f"(FLOPs={flop_count}, throughput={flop_gflops:.2f} GFLOPS)")
    print(f"  cosine_similarity={sim:.10f}  diff_from_1={abs(1.0 - sim):.2e}")
    assert sim >= 0.9999, f"{test_name} cosine similarity {sim:.6f} < 0.9999 FAILED"
    print(f"  PASS {test_name}\n")


class TestRotaryEmbedding(unittest.TestCase):

    def test_prefill(self):
        # prefill: batch_size=1, rope_offset=0, rope_dim=128
        # batch_size.cache_len.q_len configs
        configs = [
            (1, 0, 1024), (1, 0, 2048), (1, 0, 4096), (1, 0, 8192),
            (1, 0, 10240), (1, 0, 16384), (1, 0, 32768),
            (1, 4096, 4096), (1, 5120, 5120), (1, 8192, 8192),
            (1, 16384, 4096), (1, 28672, 4096),
        ]
        for batch_size, cache_len, q_len in configs:
            run_rotary_case(80, 8, 128, 128, 0, batch_size, cache_len, q_len,
                            f"prefill_bs{batch_size}_cl{cache_len}_ql{q_len}")

    def test_decode(self):
        # decode: batch_size=16, rope_offset=0, rope_dim=128
        # batch_size.cache_len.q_len configs
        configs = [
            (16, 1024, 1), (16, 2048, 1), (16, 4096, 1), (16, 8192, 1),
            (16, 10240, 1), (16, 16384, 1), (16, 32768, 1),
            (16, 1024, 4), (16, 4096, 4), (16, 8192, 4),
            (16, 10240, 4), (16, 16384, 4), (16, 32768, 4),
        ]
        for batch_size, cache_len, q_len in configs:
            run_rotary_case(80, 8, 128, 128, 0, batch_size, cache_len, q_len,
                            f"decode_bs{batch_size}_cl{cache_len}_ql{q_len}")


if __name__ == '__main__':
    unittest.main()