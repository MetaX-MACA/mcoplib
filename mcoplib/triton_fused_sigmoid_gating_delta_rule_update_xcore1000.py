"""GLM5.3-flash candidates isolated to MACA xcore1000/xcore1008.

The original kernel and other platform dispatches live in the parent module.
Architecture probing is cached per logical device and fails closed.
"""

import ctypes
from functools import lru_cache
import os

import triton
import triton.language as tl


@lru_cache(maxsize=None)
def get_xcore_arch(device_index: int):
    try:
        if triton.runtime.driver.active.get_current_target().backend != "maca":
            return None
        library = ctypes.CDLL(os.path.join(
            os.environ.get("MACA_PATH", "/opt/maca"), "lib", "libmcruntime.so"))
        query = library.mcExtDeviceGetAttribute
        query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        query.restype = ctypes.c_int
        # mcExtDeviceAttributeIsaMajorVersion / IsaMinorVersion, MACA runtime API.
        major, minor = ctypes.c_int(), ctypes.c_int()
        if query(ctypes.byref(major), 6, device_index) != 0:
            return None
        if query(ctypes.byref(minor), 7, device_index) != 0:
            return None
        return f"xcore{major.value:02d}{minor.value:02d}"
    except (OSError, AttributeError, RuntimeError):
        return None


def is_supported_xcore(device_index: int) -> bool:
    return get_xcore_arch(device_index) in ("xcore1000", "xcore1008")


@triton.jit
def glm53_flash_recurrent_kernel(
    A_log, a, dt_bias, b, q, k, v, o,
    h0_source, h0_indices, cu_seqlens,
    intermediate_states_buffer, intermediate_state_indices,
    gate_decay, beta_values, q_inv_norm, k_inv_norm,
    stride_h0_source, stride_a, stride_b, stride_q, stride_k, stride_v, scale,
    BV: tl.constexpr, PRECOMPUTED: tl.constexpr, UNROLL: tl.constexpr,
    PROJECT_BEFORE_UPDATE: tl.constexpr = False,
):
    """Six-step KDA update with a contiguous K axis in the FP32 state tile."""
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // 64, i_nh % 64
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    state_idx = tl.load(h0_indices + i_n).to(tl.int64)
    cache_idx = tl.load(intermediate_state_indices + i_n).to(tl.int64)
    ks = tl.arange(0, 128)
    vs = i_v * BV + tl.arange(0, BV)
    state_offsets = i_h * 128 * 128 + vs[:, None] * 128 + ks[None, :]
    state = tl.load(h0_source + state_idx * stride_h0_source + state_offsets,
                    mask=state_idx >= 0, other=0.0).to(tl.float32)
    if not PRECOMPUTED:
        A = tl.exp(tl.load(A_log + i_h).to(tl.float32))
        bias = tl.load(dt_bias + i_h * 128 + ks).to(tl.float32)

    for step in tl.range(0, 6, loop_unroll_factor=UNROLL):
        token = bos + step
        token_head = token * 64 + i_h
        qt = tl.load(q + token * stride_q + i_h * 128 + ks).to(tl.float32)
        kt = tl.load(k + token * stride_k + i_h * 128 + ks).to(tl.float32)
        vt = tl.load(v + token * stride_v + i_h * 128 + vs).to(tl.float32)
        if PRECOMPUTED:
            decay = tl.load(gate_decay + token_head * 128 + ks).to(tl.float32)
            beta = tl.load(beta_values + token_head).to(tl.float32)
            qt *= tl.load(q_inv_norm + token_head).to(tl.float32)
            kt *= tl.load(k_inv_norm + token_head).to(tl.float32)
        else:
            raw_a = tl.load(a + token * stride_a + i_h * 128 + ks).to(tl.float32)
            decay = tl.exp(-5.0 * tl.sigmoid(A * (raw_a + bias)))
            beta = tl.sigmoid(tl.load(b + token * stride_b + i_h).to(tl.float32))
            qt *= tl.rsqrt(tl.sum(qt * qt, 0) + 1e-6)
            kt *= tl.rsqrt(tl.sum(kt * kt, 0) + 1e-6)
        qt *= scale
        state *= decay[None, :]
        delta = (vt - tl.sum(state * kt[None, :], 1)) * beta
        if PROJECT_BEFORE_UPDATE:
            output = (tl.sum(state * qt[None, :], 1)
                      + delta * tl.sum(kt * qt, 0))
            state += delta[:, None] * kt[None, :]
        else:
            state += delta[:, None] * kt[None, :]
            output = tl.sum(state * qt[None, :], 1)
        tl.store(o + token_head * 128 + vs, output.to(o.dtype.element_ty))
        if cache_idx >= 0:
            tl.store(intermediate_states_buffer
                     + cache_idx * 6 * 64 * 128 * 128
                     + step * 64 * 128 * 128 + state_offsets, state)


@triton.jit
def glm53_flash_n2_recurrent_kernel(
    gate_decay, beta_values, q_inv_norm, k_inv_norm, q, k, v, o,
    h0_source, h0_indices, cu_seqlens, stride_h0_source,
    intermediate_states_buffer, intermediate_state_indices, scale,
    stride_q, stride_k, stride_v,
):
    """N2 specialization using the existing precompute workspace and signature."""
    glm53_flash_recurrent_kernel(
        None, None, None, None, q, k, v, o,
        h0_source, h0_indices, cu_seqlens,
        intermediate_states_buffer, intermediate_state_indices,
        gate_decay, beta_values, q_inv_norm, k_inv_norm,
        stride_h0_source, 0, 0, stride_q, stride_k, stride_v, scale,
        BV=32, PRECOMPUTED=True, UNROLL=1,
    )


@lru_cache(maxsize=None)
def workspace_base_kernels(precompute, n1_recurrent, n2_recurrent):
    """Bind existing JIT math to a single workspace pointer; no tensor views."""

    @triton.jit
    def precompute_base(
        A_log, a, dt_bias, b, q, k, workspace, lower_bound,
        stride_a, stride_b, stride_q, stride_k, TOKENS: tl.constexpr,
    ):
        gate_size: tl.constexpr = TOKENS * 64 * 128
        scalar_size: tl.constexpr = TOKENS * 64
        precompute(
            A_log=A_log, a=a, dt_bias=dt_bias, b=b, q=q, k=k,
            gate_decay=workspace, gate_values=None,
            beta_values=workspace + gate_size,
            q_inv_norm=workspace + gate_size + scalar_size,
            k_inv_norm=workspace + gate_size + 2 * scalar_size,
            softplus_beta=1.0, softplus_threshold=20.0, lower_bound=lower_bound,
            stride_a=stride_a, stride_b=stride_b, stride_q=stride_q, stride_k=stride_k,
            H=64, HV=64, K=128, BK=128, IS_KDA=True, USE_LOWER_BOUND=True,
            STORE_GATE_VALUES=False, PRECOMPUTE_QK_NORMS=True,
        )

    @triton.jit
    def recurrent_base(
        workspace, q, k, v, o, h0_source, h0_indices, cu_seqlens,
        stride_h0_source, intermediate_states_buffer, intermediate_state_indices,
        scale, stride_q, stride_k, stride_v, TOKENS: tl.constexpr,
    ):
        gate_size: tl.constexpr = TOKENS * 64 * 128
        scalar_size: tl.constexpr = TOKENS * 64
        if TOKENS == 6:
            n1_recurrent(
                gate_decay=workspace, beta_values=workspace + gate_size,
                q_inv_norm=workspace + gate_size + scalar_size,
                k_inv_norm=workspace + gate_size + 2 * scalar_size,
                q=q, k=k, v=v, o=o, h0_source=h0_source, h0_indices=h0_indices,
                stride_h0_source=stride_h0_source,
                intermediate_states_buffer=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices,
                scale=scale, stride_q=stride_q, stride_k=stride_k, stride_v=stride_v,
            )
        else:
            n2_recurrent(
                gate_decay=workspace, beta_values=workspace + gate_size,
                q_inv_norm=workspace + gate_size + scalar_size,
                k_inv_norm=workspace + gate_size + 2 * scalar_size,
                q=q, k=k, v=v, o=o, h0_source=h0_source, h0_indices=h0_indices,
                cu_seqlens=cu_seqlens, stride_h0_source=stride_h0_source,
                intermediate_states_buffer=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices,
                scale=scale, stride_q=stride_q, stride_k=stride_k, stride_v=stride_v,
            )

    return precompute_base, recurrent_base
