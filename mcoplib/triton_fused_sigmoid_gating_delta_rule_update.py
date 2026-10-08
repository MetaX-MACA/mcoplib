from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    lower_bound,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    stride_h0_source,
    cu_seqlens,
    # Parameters for target_verify support (unused for decode)
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    retrieve_parent_token_ptr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    # ================================================
    scale,
    T,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    NP2_T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    # Optional flags for target_verify support (default False for decode)
    DISABLE_STATE_UPDATE: tl.constexpr = False,
    CACHE_INTERMEDIATE_STATES: tl.constexpr = False,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,
    # ReplaySSM fused ring-write. Pointers stay None and CACHE_RING False for
    # decode / flag-off -> byte-identical. The gate ring layout follows IS_KDA
    # (see the store below).
    replayssm_rawv=None,
    replayssm_rawk=None,
    replayssm_g=None,
    replayssm_beta=None,
    stride_rawv_slot: tl.constexpr = 0,
    stride_rawk_slot: tl.constexpr = 0,
    stride_g_slot: tl.constexpr = 0,
    stride_beta_slot: tl.constexpr = 0,
    MAX_CACHE_LEN: tl.constexpr = 0,
    CACHE_RING: tl.constexpr = False,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + bos * stride_a + i_hv * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        # Slot stride comes from the caller (h0_source.stride(0)): the state pool
        # may be an envelope-strided view (page-major / unified memory), where the
        # per-slot pitch spans ALL layers' state, not HV*K*V. int64: envelope
        # pitches overflow an int32 index product.
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * stride_h0_source
                + i_hv * K * V
                + o_v[None, :] * K
                + o_k[:, None]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Preload tree attention data if needed
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        retrieve_parent_token_base = (
            retrieve_parent_token_ptr
            + (i_n * stride_retrieve_parent_token_seq)
            + token_indices * stride_retrieve_parent_token_token
        )
        parent_idx_tokens = tl.load(
            retrieve_parent_token_base, mask=mask_retrieve, other=0
        )

    # Prepare intermediate state cache index if enabled. int64: the buffer is
    # contiguous but `cache_idx * cache_steps * HV * K * V` can exceed int32 for
    # large slot counts.
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n).to(tl.int64)

    step_idx = 0
    for _ in range(0, T):
        # Tree attention: load parent's cached state
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # step_idx == 0 uses b_h from USE_INITIAL_STATE
            if step_idx != 0 and cache_idx >= 0:
                parent_step_idx = tl.sum(
                    tl.where(token_indices == step_idx, parent_idx_tokens, 0)
                )
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)

        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        if IS_KDA:
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0).to(tl.float32)
        else:
            b_a = tl.load(p_a).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        x = b_a + b_dt_bias
        if USE_LOWER_BOUND:
            # KDA safe gate: lower_bound * sigmoid(exp(A_log) * (a + dt_bias))
            b_g = lower_bound * tl.sigmoid(tl.exp(b_A_log) * x)
        else:
            # Compute g = -exp(A_log) * softplus(a + dt_bias)
            beta_x = softplus_beta * x
            # Apply softplus with numerical stability
            softplus_x = tl.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                x,
            )
            b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # fused ring-write: stash this step's raw inputs + in-kernel gate/beta
        # into the per-slot ring for the commit fold to replay. Must sit here --
        # b_k is still pre-l2norm, b_v still pre-delta, b_g/b_beta are formed,
        # so the fold's replay is bit-identical to the update below. rawk uses
        # the k-head i_h (shared across a GQA group); rawv/g/beta use the v-head
        # i_hv. step_idx < MAX_CACHE_LEN: absorb-inflated rows can exceed the
        # ring; the overflow steps are past the committable prefix, so drop them
        # (writing them would smash the next slot's ring).
        if CACHE_RING:
            ring_slot = tl.load(h0_indices + i_n).to(tl.int64)
            if ring_slot >= 0 and step_idx < MAX_CACHE_LEN:
                tl.store(
                    replayssm_rawv
                    + ring_slot * stride_rawv_slot
                    + i_hv * MAX_CACHE_LEN * V
                    + step_idx * V
                    + o_v,
                    b_v.to(replayssm_rawv.dtype.element_ty),
                    mask=mask_v,
                )
                if i_v == 0:
                    tl.store(
                        replayssm_rawk
                        + ring_slot * stride_rawk_slot
                        + i_h * MAX_CACHE_LEN * K
                        + step_idx * K
                        + o_k,
                        b_k.to(replayssm_rawk.dtype.element_ty),
                        mask=mask_k,
                    )
                    # b_g follows IS_KDA: KDA loads a/dt_bias with mask_k, so the
                    # gate is a per-K vector and the ring row is K wide; GDN's is
                    # a scalar per (head, step). The two layouts are not
                    # interchangeable -- storing one into the other's stride is a
                    # shape error, not a slow path -- and memory_pool.py sizes
                    # replayssm_g off the same is_kda test.
                    if IS_KDA:
                        tl.store(
                            replayssm_g
                            + ring_slot * stride_g_slot
                            + i_hv * MAX_CACHE_LEN * K
                            + step_idx * K
                            + o_k,
                            b_g,
                            mask=mask_k,
                        )
                    else:
                        tl.store(
                            replayssm_g
                            + ring_slot * stride_g_slot
                            + i_hv * MAX_CACHE_LEN
                            + step_idx,
                            b_g,
                        )
                    if i_k == 0:
                        tl.store(
                            replayssm_beta
                            + ring_slot * stride_beta_slot
                            + i_hv * MAX_CACHE_LEN
                            + step_idx,
                            b_beta,
                        )

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Cache intermediate states if enabled
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                step_offset = step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        step_idx += 1

        # Update pointers for next timestep
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += HV * V
        p_a += stride_a

    # Store final state back to h0_source with bounds checking
    if not DISABLE_STATE_UPDATE:
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n).to(tl.int64)
            if idx >= 0:
                p_h0 = (
                    h0_source
                    + idx * stride_h0_source
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


@triton.jit
def sigmoid_gating_precompute_kernel(
    A_log,
    a,
    dt_bias,
    b,
    q,
    k,
    gate_decay,
    gate_values,
    beta_values,
    q_inv_norm,
    k_inv_norm,
    softplus_beta,
    softplus_threshold,
    lower_bound,
    stride_a,
    stride_b,
    stride_q,
    stride_k,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    IS_KDA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    STORE_GATE_VALUES: tl.constexpr,
    PRECOMPUTE_QK_NORMS: tl.constexpr,
):
    """并行预计算每个 token/head 的 exp(g) 与 sigmoid(beta raw)。"""
    i_t = tl.program_id(0)
    i_hv = tl.program_id(1)
    o_k = tl.arange(0, BK)
    mask_k = o_k < K

    b_A = tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    if IS_KDA:
        p_a = a + i_t * stride_a + i_hv * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
        b_a = tl.load(p_a, mask=mask_k, other=0.0).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0.0).to(tl.float32)
    else:
        p_a = a + i_t * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv
        b_a = tl.load(p_a).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

    x = b_a + b_dt_bias
    if USE_LOWER_BOUND:
        b_g = lower_bound * tl.sigmoid(b_A * x)
    else:
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            tl.log(1.0 + tl.exp(beta_x)) / softplus_beta,
            x,
        )
        b_g = -b_A * softplus_x

    gate_width: tl.constexpr = K if IS_KDA else 1
    p_gate = gate_decay + (i_t * HV + i_hv) * gate_width
    if IS_KDA:
        tl.store(p_gate + o_k, tl.exp(b_g), mask=mask_k)
        if STORE_GATE_VALUES:
            tl.store(
                gate_values + (i_t * HV + i_hv) * K + o_k,
                b_g,
                mask=mask_k,
            )
    else:
        tl.store(p_gate, tl.exp(b_g))
        if STORE_GATE_VALUES:
            tl.store(gate_values + i_t * HV + i_hv, b_g)

    b_beta = tl.sigmoid(tl.load(b + i_t * stride_b + i_hv).to(tl.float32))
    tl.store(beta_values + i_t * HV + i_hv, b_beta)

    if PRECOMPUTE_QK_NORMS:
        qk_group_size: tl.constexpr = HV // H
        if i_hv % qk_group_size == 0:
            i_h = i_hv // qk_group_size
            b_q = tl.load(
                q + i_t * stride_q + i_h * K + o_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            b_k = tl.load(
                k + i_t * stride_k + i_h * K + o_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            norm_offset = i_t * H + i_h
            tl.store(
                q_inv_norm + norm_offset,
                tl.rsqrt(tl.sum(b_q * b_q) + 1e-6),
            )
            tl.store(
                k_inv_norm + norm_offset,
                tl.rsqrt(tl.sum(b_k * b_k) + 1e-6),
            )


@triton.jit
def fused_sigmoid_gating_delta_rule_update_kernel_n1_static6(
    gate_decay,
    beta_values,
    q_inv_norm,
    k_inv_norm,
    q,
    k,
    v,
    o,
    h0_source,
    h0_indices,
    stride_h0_source,
    intermediate_states_buffer,
    intermediate_state_indices,
    scale,
    stride_q,
    stride_k,
    stride_v,
):
    """Strict N=1/T=6/H=HV=64/K=V=128 target-verify recurrent kernel."""
    K: tl.constexpr = 128
    V: tl.constexpr = 128
    HV: tl.constexpr = 64
    STEPS: tl.constexpr = 6
    BV: tl.constexpr = 16

    i_hv = tl.program_id(0)
    i_v = tl.program_id(1)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)

    state_idx = tl.load(h0_indices).to(tl.int64)
    p_h0 = (
        h0_source
        + state_idx * stride_h0_source
        + i_hv * K * V
        + o_v[:, None] * K
        + o_k[None, :]
    )
    b_h = tl.load(
        p_h0,
        mask=state_idx >= 0,
        other=0.0,
    ).to(tl.float32)

    cache_idx = tl.load(intermediate_state_indices).to(tl.int64)
    for step in tl.static_range(0, STEPS):
        token_head = step * HV + i_hv
        b_q = tl.load(
            q + step * stride_q + i_hv * K + o_k
        ).to(tl.float32)
        b_k = tl.load(
            k + step * stride_k + i_hv * K + o_k
        ).to(tl.float32)
        b_v = tl.load(
            v + step * stride_v + i_hv * V + o_v
        ).to(tl.float32)
        b_q *= tl.load(q_inv_norm + token_head).to(tl.float32)
        b_k *= tl.load(k_inv_norm + token_head).to(tl.float32)
        b_q *= scale

        b_gate_decay = tl.load(
            gate_decay + token_head * K + o_k
        ).to(tl.float32)
        b_beta = tl.load(beta_values + token_head).to(tl.float32)

        b_h *= b_gate_decay[None, :]
        b_v -= tl.sum(b_h * b_k[None, :], axis=1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], axis=1)
        tl.store(
            o + token_head * V + o_v,
            b_o.to(o.dtype.element_ty),
        )

        if cache_idx >= 0:
            cache_ptr = (
                intermediate_states_buffer
                + cache_idx * STEPS * HV * K * V
                + step * HV * K * V
                + i_hv * K * V
                + o_v[:, None] * K
                + o_k[None, :]
            )
            tl.store(
                cache_ptr,
                b_h.to(cache_ptr.dtype.element_ty),
            )


@triton.jit
def fused_sigmoid_gating_delta_rule_update_kernel_n2_static6(
    gate_decay,
    beta_values,
    q_inv_norm,
    k_inv_norm,
    q,
    k,
    v,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    stride_h0_source,
    intermediate_states_buffer,
    intermediate_state_indices,
    scale,
    stride_q,
    stride_k,
    stride_v,
):
    """Strict N=2/T=12/H=HV=64/K=V=128 target-verify recurrent kernel."""
    K: tl.constexpr = 128
    V: tl.constexpr = 128
    HV: tl.constexpr = 64
    STEPS: tl.constexpr = 6
    BV: tl.constexpr = 32

    i_nh = tl.program_id(0)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_v = tl.program_id(1)
    bos = tl.load(cu_seqlens + i_n)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)

    state_idx = tl.load(h0_indices + i_n).to(tl.int64)
    p_h0 = (
        h0_source
        + state_idx * stride_h0_source
        + i_hv * K * V
        + o_v[:, None] * K
        + o_k[None, :]
    )
    b_h = tl.load(
        p_h0,
        mask=state_idx >= 0,
        other=0.0,
    ).to(tl.float32)

    cache_idx = tl.load(intermediate_state_indices + i_n).to(tl.int64)
    for step in tl.static_range(0, STEPS):
        token = bos + step
        token_head = token * HV + i_hv
        b_q = tl.load(
            q + token * stride_q + i_hv * K + o_k
        ).to(tl.float32)
        b_k = tl.load(
            k + token * stride_k + i_hv * K + o_k
        ).to(tl.float32)
        b_v = tl.load(
            v + token * stride_v + i_hv * V + o_v
        ).to(tl.float32)
        b_q *= tl.load(q_inv_norm + token_head).to(tl.float32)
        b_k *= tl.load(k_inv_norm + token_head).to(tl.float32)
        b_q *= scale

        b_gate_decay = tl.load(
            gate_decay + token_head * K + o_k
        ).to(tl.float32)
        b_beta = tl.load(beta_values + token_head).to(tl.float32)

        b_h *= b_gate_decay[None, :]
        b_v -= tl.sum(b_h * b_k[None, :], axis=1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], axis=1)
        tl.store(
            o + token_head * V + o_v,
            b_o.to(o.dtype.element_ty),
        )

        if cache_idx >= 0:
            cache_ptr = (
                intermediate_states_buffer
                + cache_idx * STEPS * HV * K * V
                + step * HV * K * V
                + i_hv * K * V
                + o_v[:, None] * K
                + o_k[None, :]
            )
            tl.store(
                cache_ptr,
                b_h.to(cache_ptr.dtype.element_ty),
            )


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel_opt(
    A_log,
    a,
    dt_bias,
    b,
    gate_decay,
    gate_values,
    beta_values,
    q_inv_norm,
    k_inv_norm,
    softplus_beta,
    softplus_threshold,
    lower_bound,
    q,
    k,
    v,
    o,
    h0_source,
    h0_indices,
    stride_h0_source,
    cu_seqlens,
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    retrieve_parent_token_ptr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    replayssm_rawv,
    replayssm_rawk,
    replayssm_g,
    replayssm_beta,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    scale,
    T,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    NP2_T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    PRECOMPUTED_GATES: tl.constexpr,
    PRECOMPUTED_QK_NORMS: tl.constexpr,
    CACHE_RING: tl.constexpr,
    STATIC_STEPS: tl.constexpr,
    DISABLE_STATE_UPDATE: tl.constexpr = False,
    CACHE_INTERMEDIATE_STATES: tl.constexpr = False,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,
):

    NV: tl.constexpr = (V + BV - 1) // BV
    pid = tl.program_id(0)
    i_nh, i_v = pid // NV, pid % NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        if STATIC_STEPS > 0:
            T = STATIC_STEPS
        else:
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = eos - bos
        all_tokens = T
    else:
        bos = i_n * T
        all_tokens = B * T

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_o = o + (bos * HV + i_hv) * V + o_v
    gate_width: tl.constexpr = K if IS_KDA else 1
    if PRECOMPUTED_GATES:
        p_gate_decay = gate_decay + (bos * HV + i_hv) * gate_width
        if CACHE_RING:
            p_gate_values = gate_values + (bos * HV + i_hv) * gate_width
        p_beta_values = beta_values + bos * HV + i_hv
        if PRECOMPUTED_QK_NORMS:
            p_q_inv_norm = q_inv_norm + bos * H + i_h
            p_k_inv_norm = k_inv_norm + bos * H + i_h
    else:
        p_b = b + bos * stride_b + i_hv
        if IS_KDA:
            p_a = a + bos * stride_a + i_hv * K + o_k
            p_dt_bias = dt_bias + i_hv * K + o_k
        else:
            p_a = a + bos * stride_a + i_hv
            p_dt_bias = dt_bias + i_hv

    state_idx = -1
    if USE_INITIAL_STATE or CACHE_RING:
        state_idx = tl.load(h0_indices + i_n).to(tl.int64)

    b_h = tl.zeros([BV, K], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = (
            h0_source
            + state_idx * stride_h0_source
            + i_hv * K * V
            + o_v[:, None] * K
            + o_k[None, :]
        )
        state_mask = (state_idx >= 0) & mask_v[:, None]
        b_h += tl.load(p_h0, mask=state_mask, other=0.0).to(tl.float32)

    ring_slot = state_idx
    if not PRECOMPUTED_GATES:
        b_A = tl.exp(tl.load(A_log + i_hv).to(tl.float32))
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        parent_idx_tokens = tl.load(
            retrieve_parent_token_ptr
            + i_n * stride_retrieve_parent_token_seq
            + token_indices * stride_retrieve_parent_token_token,
            mask=mask_retrieve,
            other=0,
        )

    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n).to(tl.int64)

    step_idx = 0
    for _ in range(0, T):
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            if step_idx != 0 and cache_idx >= 0:
                parent_step_idx = tl.sum(
                    tl.where(token_indices == step_idx, parent_idx_tokens, 0)
                )
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + parent_step_idx * HV * K * V
                    + i_hv * K * V
                    + o_v[:, None] * K
                    + o_k[None, :]
                )
                b_h = tl.load(
                    cache_ptr, mask=mask_v[:, None], other=0.0
                ).to(tl.float32)

        b_q = tl.load(p_q).to(tl.float32)
        b_k = tl.load(p_k).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        if PRECOMPUTED_GATES:
            b_beta = tl.load(p_beta_values).to(tl.float32)
            if IS_KDA:
                b_gate_decay = tl.load(p_gate_decay + o_k).to(tl.float32)
                if CACHE_RING:
                    b_g = tl.load(p_gate_values + o_k).to(tl.float32)
            else:
                b_gate_decay = tl.load(p_gate_decay).to(tl.float32)
                if CACHE_RING:
                    b_g = tl.load(p_gate_values).to(tl.float32)
        else:
            b_gate_raw = tl.load(p_a).to(tl.float32)
            gate_x = b_gate_raw + b_dt_bias
            if USE_LOWER_BOUND:
                b_g = lower_bound * tl.sigmoid(b_A * gate_x)
            else:
                beta_x = softplus_beta * gate_x
                softplus_x = tl.where(
                    beta_x <= softplus_threshold,
                    tl.log(1.0 + tl.exp(beta_x)) / softplus_beta,
                    gate_x,
                )
                b_g = -b_A * softplus_x
            b_gate_decay = tl.exp(b_g)
            b_beta = tl.sigmoid(tl.load(p_b).to(tl.float32))

        if CACHE_RING:
            ring_mask = (ring_slot >= 0) & (step_idx < MAX_CACHE_LEN)
            tl.store(
                replayssm_rawv
                + ring_slot * stride_rawv_slot
                + i_hv * MAX_CACHE_LEN * V
                + step_idx * V
                + o_v,
                b_v.to(replayssm_rawv.dtype.element_ty),
                mask=ring_mask & mask_v,
            )
            if i_v == 0:
                tl.store(
                    replayssm_rawk
                    + ring_slot * stride_rawk_slot
                    + i_h * MAX_CACHE_LEN * K
                    + step_idx * K
                    + o_k,
                    b_k.to(replayssm_rawk.dtype.element_ty),
                    mask=ring_mask,
                )
                if IS_KDA:
                    tl.store(
                        replayssm_g
                        + ring_slot * stride_g_slot
                        + i_hv * MAX_CACHE_LEN * K
                        + step_idx * K
                        + o_k,
                        b_g,
                        mask=ring_mask,
                    )
                else:
                    tl.store(
                        replayssm_g
                        + ring_slot * stride_g_slot
                        + i_hv * MAX_CACHE_LEN
                        + step_idx,
                        b_g,
                        mask=ring_mask,
                    )
                tl.store(
                    replayssm_beta
                    + ring_slot * stride_beta_slot
                    + i_hv * MAX_CACHE_LEN
                    + step_idx,
                    b_beta,
                    mask=ring_mask,
                )

        if USE_QK_L2NORM_IN_KERNEL:
            if PRECOMPUTED_QK_NORMS:
                b_q *= tl.load(p_q_inv_norm).to(tl.float32)
                b_k *= tl.load(p_k_inv_norm).to(tl.float32)
            else:
                b_q *= tl.rsqrt(tl.sum(b_q * b_q, axis=0) + 1e-6)
                b_k *= tl.rsqrt(tl.sum(b_k * b_k, axis=0) + 1e-6)
        b_q *= scale

        if IS_KDA:
            b_h *= b_gate_decay[None, :]
        else:
            b_h *= b_gate_decay
        b_v -= tl.sum(b_h * b_k[None, :], axis=1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_idx * HV * K * V
                    + i_hv * K * V
                    + o_v[:, None] * K
                    + o_k[None, :]
                )
                tl.store(
                    cache_ptr,
                    b_h.to(cache_ptr.dtype.element_ty),
                    mask=mask_v[:, None],
                )

        step_idx += 1
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        if PRECOMPUTED_GATES:
            p_gate_decay += HV * gate_width
            if CACHE_RING:
                p_gate_values += HV * gate_width
            p_beta_values += HV
            if PRECOMPUTED_QK_NORMS:
                p_q_inv_norm += H
                p_k_inv_norm += H
        else:
            p_a += stride_a
            p_b += stride_b
        p_o += HV * V

    if not DISABLE_STATE_UPDATE:
        if USE_INITIAL_STATE:
            p_h0 = (
                h0_source
                + state_idx * stride_h0_source
                + i_hv * K * V
                + o_v[:, None] * K
                + o_k[None, :]
            )
            tl.store(
                p_h0,
                b_h.to(p_h0.dtype.element_ty),
                mask=(state_idx >= 0) & mask_v[:, None],
            )


def _select_execution_strategy(
    *, num_sequences: int, value_heads: int, tokens: int, key_dim: int, value_dim: int
) -> str:
    """根据 C600-U 实测结果选择 kernel 路径。"""
    if key_dim != 128 or value_dim != 128:
        return "original"
    if num_sequences == 1 and value_heads <= 6:
        return "original"
    if num_sequences * value_heads < 104:
        return "single_opt"
    return "dual_opt"


def _select_precompute_num_warps(
    *, token_heads: int, key_dim: int, is_kda: bool, use_static_six: bool
) -> int:
    """Choose precompute CTA width from the workload shape."""
    if not is_kda or use_static_six:
        return 1
    return 4


def _fused_sigmoid_gating_delta_rule_update_with_strategy(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
    lower_bound: Optional[float] = None,
    # Optional parameters for target_verify support
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[
        int
    ] = None,  # kept for API compat; stride is derived from ``intermediate_states_buffer.shape[1]``
    retrieve_parent_token: Optional[torch.Tensor] = None,
    # fused ReplaySSM ring-write (spec verify). When cache_ring, each draft step
    # stores pre-norm k / raw v / gate / beta into these per-slot rings,
    # replacing the eager ring-write. Off by default -> decode unchanged.
    cache_ring: bool = False,
    replayssm_rawv: Optional[torch.Tensor] = None,
    replayssm_rawk: Optional[torch.Tensor] = None,
    replayssm_g: Optional[torch.Tensor] = None,
    replayssm_beta: Optional[torch.Tensor] = None,
    *,
    strategy: str,
):
    """
    Triton implementation of sigmoid gating delta rule update.

    The baseline path uses the original fused recurrent kernel.  The optimized
    path selects among the original kernel for very small workloads, an inline-
    gate opt kernel for small grids, and a gate-precompute + recurrent pair for
    larger grids.

    Supports both decode and target_verify modes:
    - decode: standard single-step update with state write-back
    - target_verify: multi-step with intermediate state caching, optional tree attention,
                     and optional state update disable
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    stride_q = q.stride()[1]
    stride_k = k.stride()[1]
    stride_v = v.stride()[1]
    stride_b = b.stride()[-2]
    # Both paths (KDA/GDN) advance p_a once per token, so use the token-axis stride.
    # For 2D a ([T, ...]) this is stride(0); for 3D a ([B, T, ...]) this is stride(1).
    # Using stride()[-2] covers GDN [T, HV] and KDA layouts ([T, HV*K] / [B, T, HV*K]).
    # KDA decode also passes 4-D [B, T, H, K], where [-2] is the head stride, not the
    # token stride; take dim 1 explicitly for that layout.
    stride_a = a.stride()[1] if a.ndim == 4 else a.stride()[-2]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)

    # Prepare retrieve_parent_token strides
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq = retrieve_parent_token.stride(0)
        stride_retrieve_parent_token_token = retrieve_parent_token.stride(1)
    else:
        stride_retrieve_parent_token_seq = 0
        stride_retrieve_parent_token_token = 0

    NP2_T = triton.next_power_of_2(T)

    grid = (NK, NV, N * HV)

    # Per-req stride must match the buffer's allocated dim, not runtime steps
    # (they can differ under --speculative-adaptive).
    cache_stride_steps = (
        intermediate_states_buffer.shape[1]
        if intermediate_states_buffer is not None
        else 0
    )

    # ring strides (per-slot rings are contiguous [num_slots, heads, L, dim];
    # the kernel offsets within a slot with MAX_CACHE_LEN and the dim extents).
    if cache_ring:
        # stride(0) is used as the slot pitch, so a tensor still carrying the
        # layer dim would scribble outside its slot. The gate ring is the one
        # whose rank depends on the model: per-K vector for KDA, per-head scalar
        # for GDN, matching g_shape in memory_pool.py and the IS_KDA branch in
        # the store above.
        assert (
            replayssm_rawv.dim() == 4
            and replayssm_rawk.dim() == 4
            and replayssm_g.dim() == (4 if is_kda else 3)
            and replayssm_beta.dim() == 3
        ), "cache_ring expects per-layer ring views"
        max_cache_len = replayssm_rawv.shape[-2]
        stride_rawv_slot = replayssm_rawv.stride(0)
        stride_rawk_slot = replayssm_rawk.stride(0)
        stride_g_slot = replayssm_g.stride(0)
        stride_beta_slot = replayssm_beta.stride(0)
    else:
        max_cache_len = 0
        stride_rawv_slot = stride_rawk_slot = stride_g_slot = stride_beta_slot = 0

    if strategy not in (
        "auto",
        "original",
        "single_opt",
        "dual_opt",
        "dual_opt_norms",
        "dual_opt_norms_dims",
        "dual_opt_norms_dims_bv32",
        "dual_opt_static6_norms",
        "dual_opt_static6_norms_bv32",
        "dual_opt_static6_norms_n1",
        "dual_opt_static6_norms_n2",
    ):
        raise ValueError(f"unsupported internal strategy: {strategy!r}")

    if strategy == "auto":
        strategy = _select_execution_strategy(
            num_sequences=N,
            value_heads=HV,
            tokens=T,
            key_dim=K,
            value_dim=V,
        )
        # KDA target-verify hierarchy measured on C600-UL.  Precompute the
        # nonlinear gates and Q/K inverse norms only when enough recurrent work
        # amortizes the extra launch.  K=256 uses BV=32 below to stay spill-free.
        norm_precompute_contract = (
            B == 1
            and N >= 1
            and HV == 64
            and HV % H == 0
            and K in (64, 128, 256)
            and V in (64, 128, 256)
            and 0 < T <= N * cache_stride_steps
            and cache_stride_steps == 6
            and is_kda
            and use_qk_l2norm_in_kernel
            and lower_bound is not None
            and disable_state_update
            and intermediate_states_buffer is not None
            and cu_seqlens is not None
            and retrieve_parent_token is None
            and not cache_ring
        )
        dimension_profitable = (
            K == 256
            or (K == 128 and V == 64 and T >= 3)
            or (K == 128 and V in (128, 256) and T > N)
            or (K == 64 and V == 64 and T >= 6)
            or (K == 64 and V == 128 and T >= N + 3)
            or (K == 64 and V == 256 and T >= N + 2)
        )
        if norm_precompute_contract:
            if K == 128 and V == 128 and T == N * 6:
                if N == 1:
                    strategy = (
                        "dual_opt_static6_norms_n1"
                        if H == HV == 64
                        else "dual_opt_static6_norms"
                    )
                elif N == 2 and H == HV == 64:
                    strategy = "dual_opt_static6_norms_n2"
                else:
                    strategy = "dual_opt_static6_norms_bv32"
            elif dimension_profitable:
                if K == 256:
                    strategy = "dual_opt_norms_dims_bv32"
                elif K == 128 and V == 128:
                    strategy = "dual_opt_norms"
                else:
                    strategy = "dual_opt_norms_dims"
    flexible_dims = strategy in (
        "dual_opt_norms_dims",
        "dual_opt_norms_dims_bv32",
    )
    if strategy != "original" and not flexible_dims and (K != 128 or V != 128):
        raise ValueError("opt strategies require K == V == 128")
    if flexible_dims and (K not in (64, 128, 256) or V not in (64, 128, 256)):
        raise ValueError("dual_opt_norms_dims requires K,V in {64,128,256}")
    use_fast_path = strategy != "original"
    use_two_kernels = strategy in (
        "dual_opt",
        "dual_opt_norms",
        "dual_opt_norms_dims",
        "dual_opt_norms_dims_bv32",
        "dual_opt_static6_norms",
        "dual_opt_static6_norms_n1",
        "dual_opt_static6_norms_n2",
        "dual_opt_static6_norms_bv32",
    )
    use_static_six = strategy in (
        "dual_opt_static6_norms",
        "dual_opt_static6_norms_n1",
        "dual_opt_static6_norms_n2",
        "dual_opt_static6_norms_bv32",
    )
    precompute_qk_norms = strategy in (
        "dual_opt_norms",
        "dual_opt_norms_dims",
        "dual_opt_norms_dims_bv32",
        "dual_opt_static6_norms",
        "dual_opt_static6_norms_n1",
        "dual_opt_static6_norms_n2",
        "dual_opt_static6_norms_bv32",
    )
    use_n1_static6_recurrent = strategy == "dual_opt_static6_norms_n1"
    if use_n1_static6_recurrent and not (
        B == 1
        and N == 1
        and T == 6
        and H == 64
        and HV == 64
        and K == 128
        and V == 128
        and cache_stride_steps == 6
        and is_kda
        and use_qk_l2norm_in_kernel
        and lower_bound is not None
        and disable_state_update
        and initial_state_source is not None
        and initial_state_indices is not None
        and intermediate_states_buffer is not None
        and intermediate_state_indices is not None
        and cu_seqlens is not None
        and retrieve_parent_token is None
        and not cache_ring
        # Q/K/V may be views into an interleaved packed-QKV allocation.  The
        # recurrent kernels accept an arbitrary token stride; only the
        # per-token [head, dim] payload must be dense.
        and q.stride(-1) == 1
        and q.stride(-2) == K
        and k.stride(-1) == 1
        and k.stride(-2) == K
        and v.stride(-1) == 1
        and v.stride(-2) == V
        and intermediate_states_buffer.is_contiguous()
        and initial_state_source.stride(-1) == 1
        and initial_state_source.stride(-2) == K
        and initial_state_source.stride(-3) == V * K
    ):
        raise ValueError(
            "dual_opt_static6_norms_n1 requires the exact dense-per-token "
            "N=1/T=6/H=HV=64/K=V=128 target-verify contract"
        )
    use_n2_static6_recurrent = strategy == "dual_opt_static6_norms_n2"
    if use_n2_static6_recurrent and not (
        B == 1
        and N == 2
        and T == 12
        and H == 64
        and HV == 64
        and K == 128
        and V == 128
        and cache_stride_steps == 6
        and is_kda
        and use_qk_l2norm_in_kernel
        and lower_bound is not None
        and disable_state_update
        and initial_state_source is not None
        and initial_state_indices is not None
        and intermediate_states_buffer is not None
        and intermediate_state_indices is not None
        and cu_seqlens is not None
        and retrieve_parent_token is None
        and not cache_ring
        # Q/K/V may be views into an interleaved packed-QKV allocation.  The
        # recurrent kernels accept an arbitrary token stride; only the
        # per-token [head, dim] payload must be dense.
        and q.stride(-1) == 1
        and q.stride(-2) == K
        and k.stride(-1) == 1
        and k.stride(-2) == K
        and v.stride(-1) == 1
        and v.stride(-2) == V
        and intermediate_states_buffer.is_contiguous()
        and initial_state_source.stride(-1) == 1
        and initial_state_source.stride(-2) == K
        and initial_state_source.stride(-3) == V * K
    ):
        raise ValueError(
            "dual_opt_static6_norms_n2 requires the exact dense-per-token "
            "N=2/T=12/H=HV=64/K=V=128 target-verify contract"
        )
    if use_static_six and not (
        cu_seqlens is not None
        and cache_stride_steps == 6
        and T == N * 6
        and intermediate_states_buffer is not None
    ):
        raise ValueError(
            "dual_opt_static6_norms requires six steps for every packed sequence"
        )
    if precompute_qk_norms and not (
        use_qk_l2norm_in_kernel and is_kda and HV % H == 0
    ):
        raise ValueError(
            "Q/K norm precompute requires KDA normalization and HV divisible by H"
        )

    if use_fast_path:
        if strategy in (
            "dual_opt_norms_dims_bv32",
            "dual_opt_static6_norms_bv32",
        ):
            opt_bv = min(V, 32)
        else:
            opt_bv = min(V, 128 if N * HV >= 104 else 64)
        if use_static_six and opt_bv == 32:
            opt_num_warps = 4
        else:
            opt_num_warps = (
                8 if use_static_six and opt_bv == 128 else (4 if opt_bv == 128 else 2)
            )
        # 实测表明 N*HV<104 时额外 kernel launch/中间张量流量得不偿失；
        # 大规模时将非线性门计算移出串行递推更快。
        # Gate/beta are independent across tokens.  Materialize them once so
        # the recurrent kernel only keeps the true state dependency in its T
        # loop.  This also removes duplicate nonlinear evaluation across V
        # tiles.  Keep g itself only when ReplaySSM must write it to the ring.
        if use_two_kernels:
            gate_width = K if is_kda else 1
            all_tokens = B * T
            gate_numel = all_tokens * HV * gate_width
            gate_values_numel = gate_numel if cache_ring else 0
            beta_numel = all_tokens * HV
            norm_numel = 2 * all_tokens * H if precompute_qk_norms else 0
            workspace = torch.empty(
                gate_numel + gate_values_numel + beta_numel + norm_numel,
                device=q.device,
                dtype=torch.float32,
            )
            workspace_offset = 0
            gate_decay = workspace[
                workspace_offset : workspace_offset + gate_numel
            ].view(all_tokens, HV, gate_width)
            workspace_offset += gate_numel
            if cache_ring:
                gate_values = workspace[
                    workspace_offset : workspace_offset + gate_values_numel
                ].view(all_tokens, HV, gate_width)
                workspace_offset += gate_values_numel
            else:
                gate_values = None
            beta_values = workspace[
                workspace_offset : workspace_offset + beta_numel
            ].view(all_tokens, HV)
            workspace_offset += beta_numel
            if precompute_qk_norms:
                qk_inv_norm = workspace[
                    workspace_offset : workspace_offset + norm_numel
                ].view(2, all_tokens, H)
                workspace_offset += norm_numel
                q_inv_norm, k_inv_norm = qk_inv_norm[0], qk_inv_norm[1]
            else:
                q_inv_norm = k_inv_norm = None
            assert workspace_offset == workspace.numel()
            sigmoid_gating_precompute_kernel[(all_tokens, HV)](
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                b=b,
                q=q,
                k=k,
                gate_decay=gate_decay,
                gate_values=gate_values,
                beta_values=beta_values,
                q_inv_norm=q_inv_norm,
                k_inv_norm=k_inv_norm,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                lower_bound=lower_bound if lower_bound is not None else 0.0,
                stride_a=stride_a,
                stride_b=stride_b,
                stride_q=stride_q,
                stride_k=stride_k,
                H=H,
                HV=HV,
                K=K,
                BK=BK,
                IS_KDA=is_kda,
                USE_LOWER_BOUND=lower_bound is not None,
                STORE_GATE_VALUES=cache_ring,
                PRECOMPUTE_QK_NORMS=precompute_qk_norms,
                num_warps=_select_precompute_num_warps(
                    token_heads=all_tokens * HV,
                    key_dim=K,
                    is_kda=is_kda,
                    use_static_six=use_static_six,
                ),
                num_stages=1,
            )
        else:
            gate_decay = gate_values = beta_values = None
            q_inv_norm = k_inv_norm = None
        if use_n1_static6_recurrent:
            fused_sigmoid_gating_delta_rule_update_kernel_n1_static6[(64, 8)](
                gate_decay=gate_decay,
                beta_values=beta_values,
                q_inv_norm=q_inv_norm,
                k_inv_norm=k_inv_norm,
                q=q,
                k=k,
                v=v,
                o=o,
                h0_source=initial_state_source,
                h0_indices=initial_state_indices,
                stride_h0_source=initial_state_source.stride(0),
                intermediate_states_buffer=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices,
                scale=scale,
                stride_q=stride_q,
                stride_k=stride_k,
                stride_v=stride_v,
                num_warps=4,
                num_stages=1,
            )
            return o.squeeze(0)
        if use_n2_static6_recurrent:
            fused_sigmoid_gating_delta_rule_update_kernel_n2_static6[(128, 4)](
                gate_decay=gate_decay,
                beta_values=beta_values,
                q_inv_norm=q_inv_norm,
                k_inv_norm=k_inv_norm,
                q=q,
                k=k,
                v=v,
                o=o,
                h0_source=initial_state_source,
                h0_indices=initial_state_indices,
                cu_seqlens=cu_seqlens,
                stride_h0_source=initial_state_source.stride(0),
                intermediate_states_buffer=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices,
                scale=scale,
                stride_q=stride_q,
                stride_k=stride_k,
                stride_v=stride_v,
                num_warps=4,
                num_stages=1,
            )
            return o.squeeze(0)
        fused_sigmoid_gating_delta_rule_update_kernel_opt[
            (N * HV * triton.cdiv(V, opt_bv),)
        ](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            gate_decay=gate_decay,
            gate_values=gate_values,
            beta_values=beta_values,
            q_inv_norm=q_inv_norm,
            k_inv_norm=k_inv_norm,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            lower_bound=lower_bound if lower_bound is not None else 0.0,
            q=q,
            k=k,
            v=v,
            o=o,
            h0_source=initial_state_source,
            h0_indices=initial_state_indices,
            stride_h0_source=(
                initial_state_source.stride(0)
                if initial_state_source is not None
                else 0
            ),
            cu_seqlens=cu_seqlens,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_stride_steps,
            retrieve_parent_token_ptr=retrieve_parent_token,
            stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq,
            stride_retrieve_parent_token_token=stride_retrieve_parent_token_token,
            replayssm_rawv=replayssm_rawv,
            replayssm_rawk=replayssm_rawk,
            replayssm_g=replayssm_g,
            replayssm_beta=replayssm_beta,
            stride_rawv_slot=stride_rawv_slot,
            stride_rawk_slot=stride_rawk_slot,
            stride_g_slot=stride_g_slot,
            stride_beta_slot=stride_beta_slot,
            MAX_CACHE_LEN=max_cache_len,
            scale=scale,
            T=T,
            stride_a=stride_a,
            stride_q=stride_q,
            stride_k=stride_k,
            stride_v=stride_v,
            stride_b=stride_b,
            NP2_T=NP2_T,
            B=B,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BV=opt_bv,
            USE_INITIAL_STATE=initial_state_source is not None,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            IS_VARLEN=cu_seqlens is not None,
            IS_KDA=is_kda,
            USE_LOWER_BOUND=lower_bound is not None,
            PRECOMPUTED_GATES=use_two_kernels,
            PRECOMPUTED_QK_NORMS=precompute_qk_norms,
            DISABLE_STATE_UPDATE=disable_state_update,
            CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
            HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
            CACHE_RING=cache_ring,
            STATIC_STEPS=6 if use_static_six else 0,
            num_warps=opt_num_warps,
            num_stages=1,
        )
    else:
        fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        lower_bound=lower_bound if lower_bound is not None else 0.0,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        # Envelope-strided state pools (page-major / unified memory) have a
        # per-slot pitch != HV*K*V; contiguous pools pass exactly HV*K*V.
        stride_h0_source=(
            initial_state_source.stride(0) if initial_state_source is not None else 0
        ),
        cu_seqlens=cu_seqlens,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_stride_steps,
        retrieve_parent_token_ptr=retrieve_parent_token,
        stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token=stride_retrieve_parent_token_token,
        scale=scale,
        T=T,
        stride_a=stride_a,
        stride_q=stride_q,
        stride_k=stride_k,
        stride_v=stride_v,
        stride_b=stride_b,
        NP2_T=NP2_T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=is_kda,
        USE_LOWER_BOUND=lower_bound is not None,
        DISABLE_STATE_UPDATE=disable_state_update,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
        replayssm_rawv=replayssm_rawv,
        replayssm_rawk=replayssm_rawk,
        replayssm_g=replayssm_g,
        replayssm_beta=replayssm_beta,
        stride_rawv_slot=stride_rawv_slot,
        stride_rawk_slot=stride_rawk_slot,
        stride_g_slot=stride_g_slot,
        stride_beta_slot=stride_beta_slot,
        MAX_CACHE_LEN=max_cache_len,
        CACHE_RING=cache_ring,
        num_warps=num_warps,
            num_stages=num_stages,
        )
    o = o.squeeze(0)
    return o


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
    lower_bound: Optional[float] = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    cache_ring: bool = False,
    replayssm_rawv: Optional[torch.Tensor] = None,
    replayssm_rawk: Optional[torch.Tensor] = None,
    replayssm_g: Optional[torch.Tensor] = None,
    replayssm_beta: Optional[torch.Tensor] = None,
):
    """根据输入规模自动选择原始、单-kernel opt 或双-kernel opt。"""
    return _fused_sigmoid_gating_delta_rule_update_with_strategy(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        is_kda=is_kda,
        lower_bound=lower_bound,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
        retrieve_parent_token=retrieve_parent_token,
        cache_ring=cache_ring,
        replayssm_rawv=replayssm_rawv,
        replayssm_rawk=replayssm_rawk,
        replayssm_g=replayssm_g,
        replayssm_beta=replayssm_beta,
        strategy="auto",
    )


__all__ = [
    "fused_sigmoid_gating_delta_rule_update",
    "fused_sigmoid_gating_delta_rule_update_kernel",
]
