# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import mcoplib._C

DTYPE = torch.bfloat16
NUM_TOKENS = 3
NUM_HEADS = 4
BLOCK_SIZE = 8
POSITIONS = (1, 7, 13)
SLOTS = (0, 3, 9)

def randn(*shape):
    return torch.randn(
        *shape,
        device="cuda",
        dtype=DTYPE,
    )

def make_rope_cache(max_position=32):
    inv_freq = 1.0 / (50000 ** (
            torch.arange(0, 64, 2, device="cuda", dtype=torch.float32,) / 64
        )
    )

    positions = torch.arange(
        max_position,
        device="cuda",
        dtype=torch.float32,
    )

    freqs = torch.outer(
        positions,
        inv_freq,
    )

    return torch.cat(
        (
            freqs.cos(),
            freqs.sin(),
        ),
        dim=-1,
    )

def apply_gptj_rope(x, position_ids, cache,):
    cos, sin = (
        cache.index_select(
            0,
            position_ids,
        )
        .chunk(2, dim=-1)
    )
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    x1 = x[..., ::2].float()
    x2 = x[..., 1::2].float()

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

    return (
        torch.stack(
            [out1, out2,],
            dim=-1,
        )
        .flatten(-2)
        .to(x.dtype)
    )

def cache_rows(cache, slots):
    return (
        cache
        .reshape(-1, cache.shape[-1])
        .index_select(0, slots)
    )

@pytest.fixture
def mla_inputs():
    positions = torch.tensor(
        POSITIONS,
        device="cuda",
        dtype=torch.int64,
    )

    slots = torch.tensor(
        SLOTS,
        device="cuda",
        dtype=torch.int64,
    )

    rope_cache = make_rope_cache()

    q = randn(
        NUM_TOKENS,
        NUM_HEADS,
        192,
    )

    k_nope = randn(
        NUM_TOKENS,
        NUM_HEADS,
        128,
    )

    k_pe = randn(
        NUM_TOKENS,
        64,
    )

    kv_c = torch.randn(
        NUM_TOKENS,
        512,
        device="cuda",
        dtype=torch.float32,
    ).to(torch.bfloat16)

    v = randn(
        NUM_TOKENS,
        NUM_HEADS,
        128,
    )

    return (
        q,
        k_nope,
        k_pe,
        kv_c,
        v,
        slots,
        positions,
        rope_cache,
    )

def test_fused_kimi_k3_mla_key_concat_kv_cache_insert(
    mla_inputs,
):

    (
        q,
        k_nope,
        k_pe,
        kv_c,
        _,
        slots,
        positions,
        rope_cache,
    ) = mla_inputs

    k_out = torch.empty(
        NUM_TOKENS,
        NUM_HEADS,
        192,
        device="cuda",
        dtype=DTYPE,
    )

    k_cache = torch.zeros(
        2,
        BLOCK_SIZE,
        576,
        device="cuda",
        dtype=DTYPE,
    )

    torch.ops._C.fused_kimi_k3_mla_key_concat_kv_cache_insert(
        q,
        k_nope,
        k_pe,
        kv_c,
        k_out,
        k_cache,
        slots,
        BLOCK_SIZE,
        positions,
        rope_cache,
    )

    k_pe_expected = apply_gptj_rope(
        k_pe,
        positions,
        rope_cache,
    )
    expected_k = torch.cat(
        [
            k_nope,
            k_pe_expected[:,None,:].expand(
                -1,
                NUM_HEADS,
                -1,
            ),
        ],
        dim=-1,
    )

    torch.testing.assert_close(
        k_out,
        expected_k,
        atol=2e-2,
        rtol=2e-2,
    )

    expected_cache = torch.cat(
        [
            kv_c,
            k_pe_expected,
        ],
        dim=-1,
    )

    torch.testing.assert_close(
        cache_rows(
            k_cache,
            slots,
        ),
        expected_cache,
        atol=2e-2,
        rtol=2e-2,
    )

@torch.inference_mode()
def test_fused_kimi_k3_mla_key_concat_ds_mla_insert(
    mla_inputs,
):
    (
        q,
        k_nope,
        k_pe,
        kv_c,
        _,
        slots,
        positions,
        rope_cache,
    ) = mla_inputs

    k_out = torch.empty(
        NUM_TOKENS,
        NUM_HEADS,
        192,
        device="cuda",
        dtype=DTYPE,
    )

    k_cache = torch.zeros(
        2,
        BLOCK_SIZE,
        656,
        device="cuda",
        dtype=torch.uint8,
    )

    torch.ops._C.fused_kimi_k3_mla_key_concat_ds_mla_insert(
        q,
        k_nope,
        k_pe,
        kv_c,
        k_out,
        k_cache,
        slots,
        BLOCK_SIZE,
        positions,
        rope_cache,
    )

    k_pe_expected = apply_gptj_rope(
        k_pe,
        positions,
        rope_cache,
    )

    expected_k = torch.cat(
        (
            k_nope,
            k_pe_expected[:, None, :]
            .expand(-1, NUM_HEADS, -1),
        ),
        dim=-1,
    )

    torch.testing.assert_close(
        k_out,
        expected_k,
        atol=2e-2,
        rtol=2e-2,
    )

def test_fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(
    mla_inputs,
):
    (
        _,
        _,
        k_pe,
        kv_c,
        _,
        slots,
        positions,
        rope_cache,
    ) = mla_inputs

    ql_nope = randn(
        NUM_TOKENS,
        NUM_HEADS,
        512,
    )

    q_pe = randn(
        NUM_TOKENS,
        NUM_HEADS,
        64,
    )

    mqa_q = torch.empty(
        NUM_TOKENS,
        NUM_HEADS,
        576,
        device="cuda",
        dtype=DTYPE,
    )

    k_cache = torch.zeros(
        2,
        BLOCK_SIZE,
        576,
        device="cuda",
        dtype=DTYPE,
    )

    torch.ops._C.fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        mqa_q,
        k_cache,
        slots,
        BLOCK_SIZE,
        positions,
        rope_cache,
    )

    q_pe_expected = apply_gptj_rope(
        q_pe,
        positions,
        rope_cache,
    )

    expected_q = torch.cat(
        [
            ql_nope,
            q_pe_expected,
        ],
        dim=-1,
    )

    torch.testing.assert_close(
        mqa_q,
        expected_q,
        atol=2e-2,
        rtol=2e-2,
    )

    k_pe_expected = apply_gptj_rope(
        k_pe,
        positions,
        rope_cache,
    )

    expected_cache = torch.cat(
        [
            kv_c,
            k_pe_expected,
        ],
        dim=-1,
    )

    torch.testing.assert_close(
        cache_rows(
            k_cache,
            slots,
        ),
        expected_cache,
        atol=2e-2,
        rtol=2e-2,
    )

@torch.inference_mode()
def test_fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert(
    mla_inputs,
):

    (
        q,
        k_nope,
        k_pe,
        kv_c,
        v,
        slots,
        positions,
        rope_cache,
    ) = mla_inputs


    #
    # output
    #

    q_fp8 = torch.empty(
        NUM_TOKENS,
        NUM_HEADS,
        192,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )


    k_fp8 = torch.empty(
        NUM_TOKENS,
        NUM_HEADS,
        192,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )


    v_fp8 = torch.empty(
        NUM_TOKENS,
        NUM_HEADS,
        128,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )


    #
    # fp8 cache
    #
    # kv_c 512 + k_pe 64 = 576
    #

    k_cache = torch.zeros(
        2,
        BLOCK_SIZE,
        576,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )


    q_scale_inv = torch.ones(
        1,
        device="cuda",
        dtype=torch.float32,
    )

    k_scale_inv = torch.ones(
        1,
        device="cuda",
        dtype=torch.float32,
    )

    v_scale_inv = torch.ones(
        1,
        device="cuda",
        dtype=torch.float32,
    )

    cache_scale_inv = torch.ones(
        1,
        device="cuda",
        dtype=torch.float32,
    )


    #
    # run kernel
    #

    torch.ops._C.fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert(
        q,
        k_nope,
        k_pe,
        kv_c,
        v,
        q_fp8,
        k_fp8,
        v_fp8,
        k_cache,
        slots,
        q_scale_inv,
        k_scale_inv,
        v_scale_inv,
        cache_scale_inv,
        BLOCK_SIZE,
        positions,
        rope_cache,
    )


    #
    # reference
    #

    q_expected = q.clone()


    q_expected[...,128:] = apply_gptj_rope(
        q_expected[...,128:],
        positions,
        rope_cache,
    )


    k_pe_expected = apply_gptj_rope(
        k_pe,
        positions,
        rope_cache,
    )


    k_expected = torch.cat(
        [
            k_nope,
            k_pe_expected[:,None,:].expand(
                -1,
                NUM_HEADS,
                -1,
            ),
        ],
        dim=-1,
    )


    cache_expected = torch.cat(
        [
            kv_c,
            k_pe_expected,
        ],
        dim=-1,
    )


    #
    # fp8 output check
    #

    torch.testing.assert_close(
        q_fp8.float(),
        q_expected.to(torch.float8_e4m3fn).float(),
        atol=0.05,
        rtol=0.15,
    )


    torch.testing.assert_close(
        k_fp8.float(),
        k_expected.to(torch.float8_e4m3fn).float(),
        atol=0.05,
        rtol=0.15,
    )


    torch.testing.assert_close(
        v_fp8.float(),
        v.to(torch.float8_e4m3fn).float(),
        atol=0.05,
        rtol=0.15,
    )


    #
    # cache check
    #

    cache_out = cache_rows(
        k_cache,
        slots,
    )


    torch.testing.assert_close(
        cache_out.float(),
        cache_expected.to(torch.float8_e4m3fn).float(),
        atol=0.05,
        rtol=0.15,
    )

@torch.inference_mode()
def test_fused_kimi_k3_mla_decode_q_concat_ds_mla_insert(
    mla_inputs,
):

    (
        _,
        _,
        _,
        kv_c,
        _,
        slots,
        positions,
        rope_cache,
    ) = mla_inputs


    ql_nope = randn(
        NUM_TOKENS,
        NUM_HEADS,
        512,
    )


    q_pe = randn(
        NUM_TOKENS,
        NUM_HEADS,
        64,
    )


    k_pe = randn(
        NUM_TOKENS,
        64,
    )


    mqa_q = torch.empty(
        NUM_TOKENS,
        NUM_HEADS,
        576,
        device="cuda",
        dtype=DTYPE,
    )


    k_cache = torch.zeros(
        2,
        BLOCK_SIZE,
        656,
        device="cuda",
        dtype=torch.uint8,
    )


    torch.ops._C.fused_kimi_k3_mla_decode_q_concat_ds_mla_insert(
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        mqa_q,
        k_cache,
        slots,
        BLOCK_SIZE,
        positions,
        rope_cache,
    )


    q_pe_expected = apply_gptj_rope(
        q_pe,
        positions,
        rope_cache,
    )


    expected_q = torch.cat(
        [
            ql_nope,
            q_pe_expected,
        ],
        dim=-1,
    )


    torch.testing.assert_close(
        mqa_q,
        expected_q,
        atol=2e-2,
        rtol=2e-2,
    )

@torch.inference_mode()
def test_fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert(
    mla_inputs,
):

    (
        _,
        _,
        _,
        kv_c,
        _,
        slots,
        positions,
        rope_cache,
    ) = mla_inputs


    ql_nope = randn(
        NUM_TOKENS,
        NUM_HEADS,
        512,
    )

    q_pe = randn(
        NUM_TOKENS,
        NUM_HEADS,
        64,
    )

    k_pe = randn(
        NUM_TOKENS,
        64,
    )


    mqa_q = torch.empty(
        NUM_TOKENS,
        NUM_HEADS,
        576,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )


    k_cache = torch.zeros(
        2,
        BLOCK_SIZE,
        576,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )


    one = torch.ones(
        1,
        device="cuda",
        dtype=torch.float32,
    )


    torch.ops._C.fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert(
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        mqa_q,
        k_cache,
        slots,
        one,
        one,
        BLOCK_SIZE,
        positions,
        rope_cache,
    )


    q_pe_expected = apply_gptj_rope(
        q_pe,
        positions,
        rope_cache,
    )


    expected = torch.cat(
        [
            ql_nope,
            q_pe_expected,
        ],
        dim=-1,
    )


    torch.testing.assert_close(
        mqa_q.float(),
        expected.to(torch.float8_e4m3fn).float(),
        atol=0.05,
        rtol=0.15,
    )