
import torch
import triton
import triton.language as tl
from functools import lru_cache
import os
import inspect
from typing import Any, Callable, Dict, Literal, Optional, Tuple

FLA_CACHE_RESULTS = os.getenv("FLA_CACHE_RESULTS", "1") == "1"


SUPPORTS_AUTOTUNE_CACHE = (
    "cache_results" in inspect.signature(triton.autotune).parameters
)

autotune_cache_kwargs = (
    {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}
)

@lru_cache(maxsize=None)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        _cpu_device_warning()
        return "cpu"


@lru_cache(maxsize=None)
def _check_platform() -> Literal["nvidia", "amd", "intel", "musa"]:
    device = get_available_device()

    if device == "cuda":
        return "nvidia"
    elif device == "hip":
        return "amd"
    elif device == "xpu":
        return "intel"
    else:
        return device


# For AMD GPUs, the triton backend is 'hip', while for Nvidia GPUs, the triton backend is 'cuda'.
# However, the torch backend is 'cuda' for both Nvidia and AMD GPUs.
# Therefore, we need to check the triton backend to determine the actual GPU vendor.
device = get_available_device() if get_available_device() != "hip" else "cuda"
if device == "maca":
    device = "cuda"

device_torch_lib = getattr(torch, device)
device_platform = _check_platform()

is_amd = device_platform == "amd"
is_intel = device_platform == "intel"
is_nvidia = device_platform == "nvidia"
is_intel_alchemist = is_intel and "Intel(R) Arc(TM) A" in torch.xpu.get_device_name(0)
is_nvidia_hopper = is_nvidia and (
    "NVIDIA H" in torch.cuda.get_device_name(0)
    or torch.cuda.get_device_capability()[0] >= 9
)
use_cuda_graph = is_nvidia and os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"

# Nvidia Ampere or newer, haven't check AMD and intel yet.
is_tf32_supported = is_nvidia and torch.cuda.get_device_capability(0)[0] >= 8
is_gather_supported = hasattr(triton.language, "gather")

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = tldevice.fast_expf
    exp2 = tldevice.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    exp = tl.exp
    exp2 = tl.math.exp2
    log = tl.log
    log2 = tl.log2

if not is_gather_supported:
    @triton.jit
    def gather(src, index, axis, _builder=None):
        """
        Gather operation that works when tl.gather is not supported.
        This is a fallback implementation that returns None.
        Just to make triton compiler happy.
        """
        return None
else:
    gather = tl.gather

if is_tf32_supported:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32")
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")

@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config(
            {"BK": 32, "BV": 128, "pipeline": "cpasync", "scenario": "unroll"}, num_warps=1, num_stages=4
        ),
        triton.Config(
            {"BK": 32, "BV": 16, "pipeline": "cpasync", "scenario": "unroll"}, num_warps=1, num_stages=4
        ),
        triton.Config(
            {"BK": 64, "BV": 64, "pipeline": "cpasync", "scenario": "unroll"}, num_warps=1, num_stages=4
        ),
        triton.Config(
            {"BK": 64, "BV": 32, "pipeline": "cpasync", "scenario": "unroll"}, num_warps=1, num_stages=4
        ),
    ],
    key=["H", "K", "BC", "V", "FUSE_RECOMPUTE", "FUSE_DIAGONAL"],
    **autotune_cache_kwargs,
)

@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_inter_solve_fused(
    q,
    k,
    g,
    beta,
    Aqk,
    Akkd,
    Akk,
    scale,
    v_in,
    w_out,
    u_out,
    kg_out,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
    FUSE_RECOMPUTE: tl.constexpr,
    FUSE_DIAGONAL: tl.constexpr,
):
    """
    Fused kernel: compute inter-subchunk Akk + solve_tril in one pass.
    Prerequisite: token_parallel has already computed diagonal Akk blocks in Akkd.

    This kernel:
    1. Computes off-diagonal Aqk blocks -> writes to global
    2. Computes off-diagonal Akk blocks -> keeps in registers
    3. Loads diagonal Akk blocks from Akkd (fp32)
    4. Does forward substitution on diagonals
    5. Computes merged Akk_inv
    6. Writes Akk_inv to Akk
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    if FUSE_DIAGONAL:
        b_Aqk_d0 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk_d0 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Aqk_d1 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk_d1 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Aqk_d2 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk_d2 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Aqk_d3 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk_d3 = tl.zeros([BC, BC], dtype=tl.float32)
        m_tc0 = (i_tc0 + o_i) < T

    ################################################################################
    # off-diagonal blocks (+ optional diagonal blocks)
    ################################################################################
    NK: tl.constexpr = tl.cdiv(K, BK)
    for i_k in range(NK):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        p_g0 = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        if FUSE_DIAGONAL:
            p_q0 = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
            )
            b_q0 = tl.load(p_q0, boundary_check=(0, 1)).to(tl.float32)
            b_gn0 = tl.load(g + i_tc0 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gm0 = tl.clamp(b_g0 - b_gn0[None, :], -126.0, 126.0)
            b_gq0 = tl.where(m_tc0[:, None], exp2(b_gm0), 0.0)
            b_gk0 = tl.where(m_tc0[:, None], exp2(-b_gm0), 0.0)
            b_kgt_d0 = tl.trans(b_k0 * b_gk0)
            b_Aqk_d0 += tl.dot(b_q0 * b_gq0, b_kgt_d0)
            b_Akk_d0 += tl.dot(b_k0 * b_gq0, b_kgt_d0)

        if i_tc1 < T:
            p_q1 = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_k1 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_g1 = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            # [BC, BK]
            b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            # [BK]
            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            # [BK, BC]
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0)).to(tl.bfloat16)
            # [BC, BC]
            b_qg1 = (b_q1 * b_gqn).to(tl.bfloat16)
            b_kg1 = (b_k1 * b_gqn).to(tl.bfloat16)
            b_Aqk10 += tl.dot(b_qg1, b_kgt)
            b_Akk10 += tl.dot(b_kg1, b_kgt)

            if FUSE_DIAGONAL:
                b_gm1_d = tl.clamp(b_gn1[None, :] - b_g1, -126.0, 126.0)
                b_gk1_d = tl.where(m_tc1[:, None], exp2(b_gm1_d), 0.0)
                b_kgt_d1 = tl.trans(b_k1 * b_gk1_d)
                b_Aqk_d1 += tl.dot(b_q1 * b_gqn, b_kgt_d1)
                b_Akk_d1 += tl.dot(b_k1 * b_gqn, b_kgt_d1)

            if i_tc2 < T:
                p_q2 = tl.make_block_ptr(
                    q, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_k2 = tl.make_block_ptr(
                    k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_g2 = tl.make_block_ptr(
                    g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                # [BC, BK]
                b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
                b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                # [BK]
                b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0).to(
                    tl.float32
                )
                # [BC, BK]
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = (b_q2 * b_gqn2).to(tl.bfloat16)
                b_kg2 = (b_k2 * b_gqn2).to(tl.bfloat16)
                # [BK, BC]
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0)).to(tl.bfloat16)
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_kg2, b_kgt)
                # [BC, BC]
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1)).to(tl.bfloat16)
                # [BC, BC]
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_kg2, b_kgt)

                if FUSE_DIAGONAL:
                    b_gm2_d = tl.clamp(b_gn2[None, :] - b_g2, -126.0, 126.0)
                    b_gk2_d = tl.where(m_tc2[:, None], exp2(b_gm2_d), 0.0)
                    b_kgt_d2 = tl.trans(b_k2 * b_gk2_d)
                    b_Aqk_d2 += tl.dot(b_q2 * b_gqn2, b_kgt_d2)
                    b_Akk_d2 += tl.dot(b_k2 * b_gqn2, b_kgt_d2)

                if i_tc3 < T:
                    p_q3 = tl.make_block_ptr(
                        q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    p_k3 = tl.make_block_ptr(
                        k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    p_g3 = tl.make_block_ptr(
                        g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    # [BC, BK]
                    b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                    b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                    # [BK]
                    b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0).to(
                        tl.float32
                    )
                    # [BC, BK]
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = (b_q3 * b_gqn3).to(tl.bfloat16)
                    b_kg3 = (b_k3 * b_gqn3).to(tl.bfloat16)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0)).to(tl.bfloat16)
                    # [BC, BC]
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1)).to(tl.bfloat16)
                    # [BC, BC]
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2)).to(tl.bfloat16)
                    # [BC, BC]
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_kg3, b_kgt)

                    if FUSE_DIAGONAL:
                        b_gm3_d = tl.clamp(b_gn3[None, :] - b_g3, -126.0, 126.0)
                        b_gk3_d = tl.where(m_tc3[:, None], exp2(b_gm3_d), 0.0)
                        b_kgt_d3 = tl.trans(b_k3 * b_gk3_d)
                        b_Aqk_d3 += tl.dot(b_q3 * b_gqn3, b_kgt_d3)
                        b_Akk_d3 += tl.dot(b_k3 * b_gqn3, b_kgt_d3)

    ################################################################################
    # save off-diagonal Aqk blocks and prepare Akk
    ################################################################################
    if i_tc1 < T:
        p_Aqk10 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        p_b1 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,)
        )
        b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
        b_Akk10 = b_Akk10 * b_b1[:, None]
    if i_tc2 < T:
        p_Aqk20 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Aqk21 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        p_b2 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,)
        )
        b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if i_tc3 < T:
        p_Aqk30 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        p_Aqk31 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0)
        )
        p_Aqk32 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        p_b3 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,)
        )
        b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    if FUSE_DIAGONAL:
        m_Aqk_diag = o_i[:, None] >= o_i[None, :]
        m_Akk_diag = o_i[:, None] > o_i[None, :]

        b_Aqk_d0 = tl.where(m_Aqk_diag, b_Aqk_d0, 0.0)
        b_Akk_d0 = tl.where(m_Akk_diag, b_Akk_d0, 0.0)
        b_Aqk_d1 = tl.where(m_Aqk_diag, b_Aqk_d1, 0.0)
        b_Akk_d1 = tl.where(m_Akk_diag, b_Akk_d1, 0.0)
        b_Aqk_d2 = tl.where(m_Aqk_diag, b_Aqk_d2, 0.0)
        b_Akk_d2 = tl.where(m_Akk_diag, b_Akk_d2, 0.0)
        b_Aqk_d3 = tl.where(m_Aqk_diag, b_Aqk_d3, 0.0)
        b_Akk_d3 = tl.where(m_Akk_diag, b_Akk_d3, 0.0)

        p_Aqk_d0 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0)
        )
        p_Aqk_d1 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0)
        )
        p_Aqk_d2 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0)
        )
        p_Aqk_d3 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk_d0, (b_Aqk_d0 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk_d1, (b_Aqk_d1 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk_d2, (b_Aqk_d2 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk_d3, (b_Aqk_d3 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        p_bd0 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,)
        )
        p_bd1 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,)
        )
        p_bd2 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,)
        )
        p_bd3 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,)
        )
        b_bd0 = tl.load(p_bd0, boundary_check=(0,)).to(tl.float32)
        b_bd1 = tl.load(p_bd1, boundary_check=(0,)).to(tl.float32)
        b_bd2 = tl.load(p_bd2, boundary_check=(0,)).to(tl.float32)
        b_bd3 = tl.load(p_bd3, boundary_check=(0,)).to(tl.float32)
        b_Akk_d0 = b_Akk_d0 * b_bd0[:, None]
        b_Akk_d1 = b_Akk_d1 * b_bd1[:, None]
        b_Akk_d2 = b_Akk_d2 * b_bd2[:, None]
        b_Akk_d3 = b_Akk_d3 * b_bd3[:, None]

        p_Akkd00 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0)
        )
        p_Akkd11 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        p_Akkd22 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Akkd33 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        tl.store(p_Akkd00, b_Akk_d0.to(Akkd.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akkd11, b_Akk_d1.to(Akkd.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akkd22, b_Akk_d2.to(Akkd.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akkd33, b_Akk_d3.to(Akkd.dtype.element_ty), boundary_check=(0, 1))

        b_Ai00 = b_Akk_d0
        b_Ai11 = b_Akk_d1
        b_Ai22 = b_Akk_d2
        b_Ai33 = b_Akk_d3
    else:
        p_Akk00 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0)
        )
        p_Akk11 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        p_Akk22 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Akk33 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
        b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
        b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    ################################################################################
    # forward substitution on diagonals
    # Diagonal blocks are RAW (need substitution) when:
    #   - FUSE_DIAGONAL=True: blocks were computed fresh above as gated k·k.
    #   - FUSE_DIAGONAL=False with USE_SAFE_GATE=False: token_parallel wrote raw.
    # They are pre-inverted only by the safe_gate diagonal kernel
    # (USE_SAFE_GATE=True, FUSE_DIAGONAL=False).
    ################################################################################

    if FUSE_DIAGONAL or not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.0)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2 * BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
            b_a22 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.0)
            b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
        for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
            b_a33 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.0)
            b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        b_Ai22 += m_I
        b_Ai33 += m_I

    ################################################################################
    # compute merged inverse using off-diagonals
    ################################################################################

    # we used tf32 to maintain matrix inverse's precision whenever possible.
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    ################################################################################
    # Output: store Akk_inv OR compute w, u, kg from registers
    ################################################################################

    if FUSE_RECOMPUTE:
        # Cast A-inverse sub-blocks to input dtype for dot products
        b_Ai00_h = b_Ai00.to(k.dtype.element_ty)
        b_Ai10_h = b_Ai10.to(k.dtype.element_ty)
        b_Ai11_h = b_Ai11.to(k.dtype.element_ty)
        b_Ai20_h = b_Ai20.to(k.dtype.element_ty)
        b_Ai21_h = b_Ai21.to(k.dtype.element_ty)
        b_Ai22_h = b_Ai22.to(k.dtype.element_ty)
        b_Ai30_h = b_Ai30.to(k.dtype.element_ty)
        b_Ai31_h = b_Ai31.to(k.dtype.element_ty)
        b_Ai32_h = b_Ai32.to(k.dtype.element_ty)
        b_Ai33_h = b_Ai33.to(k.dtype.element_ty)

        # Load beta for all 4 sub-chunks
        p_b0 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,)
        )
        b_b0 = tl.load(p_b0, boundary_check=(0,)).to(tl.float32)
        p_b1r = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,)
        )
        b_b1r = tl.load(p_b1r, boundary_check=(0,)).to(tl.float32)
        p_b2r = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,)
        )
        b_b2r = tl.load(p_b2r, boundary_check=(0,)).to(tl.float32)
        p_b3r = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,)
        )
        b_b3r = tl.load(p_b3r, boundary_check=(0,)).to(tl.float32)

        # ---- u = A_inv @ (v * beta) ----
        v_base = v_in + (bos * H + i_h) * V
        u_base = u_out + (bos * H + i_h) * V
        NV: tl.constexpr = tl.cdiv(V, BV)
        for i_v in range(NV):
            p_v0 = tl.make_block_ptr(
                v_base, (T, V), (H * V, 1), (i_tc0, i_v * BV), (BC, BV), (1, 0)
            )
            p_v1 = tl.make_block_ptr(
                v_base, (T, V), (H * V, 1), (i_tc1, i_v * BV), (BC, BV), (1, 0)
            )
            p_v2 = tl.make_block_ptr(
                v_base, (T, V), (H * V, 1), (i_tc2, i_v * BV), (BC, BV), (1, 0)
            )
            p_v3 = tl.make_block_ptr(
                v_base, (T, V), (H * V, 1), (i_tc3, i_v * BV), (BC, BV), (1, 0)
            )

            b_v0 = tl.load(p_v0, boundary_check=(0, 1))
            b_v1 = tl.load(p_v1, boundary_check=(0, 1))
            b_v2 = tl.load(p_v2, boundary_check=(0, 1))
            b_v3 = tl.load(p_v3, boundary_check=(0, 1))

            b_vb0 = (b_v0 * b_b0[:, None]).to(b_v0.dtype)
            b_vb1 = (b_v1 * b_b1r[:, None]).to(b_v1.dtype)
            b_vb2 = (b_v2 * b_b2r[:, None]).to(b_v2.dtype)
            b_vb3 = (b_v3 * b_b3r[:, None]).to(b_v3.dtype)

            b_u0 = tl.dot(b_Ai00_h, b_vb0)
            b_u1 = tl.dot(b_Ai10_h, b_vb0) + tl.dot(b_Ai11_h, b_vb1)
            b_u2 = (
                tl.dot(b_Ai20_h, b_vb0)
                + tl.dot(b_Ai21_h, b_vb1)
                + tl.dot(b_Ai22_h, b_vb2)
            )
            b_u3 = (
                tl.dot(b_Ai30_h, b_vb0)
                + tl.dot(b_Ai31_h, b_vb1)
                + tl.dot(b_Ai32_h, b_vb2)
                + tl.dot(b_Ai33_h, b_vb3)
            )

            p_u0 = tl.make_block_ptr(
                u_base, (T, V), (H * V, 1), (i_tc0, i_v * BV), (BC, BV), (1, 0)
            )
            p_u1 = tl.make_block_ptr(
                u_base, (T, V), (H * V, 1), (i_tc1, i_v * BV), (BC, BV), (1, 0)
            )
            p_u2 = tl.make_block_ptr(
                u_base, (T, V), (H * V, 1), (i_tc2, i_v * BV), (BC, BV), (1, 0)
            )
            p_u3 = tl.make_block_ptr(
                u_base, (T, V), (H * V, 1), (i_tc3, i_v * BV), (BC, BV), (1, 0)
            )
            tl.store(p_u0, b_u0.to(p_u0.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_u1, b_u1.to(p_u1.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_u2, b_u2.to(p_u2.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_u3, b_u3.to(p_u3.dtype.element_ty), boundary_check=(0, 1))

        # ---- w = A_inv @ (k * beta * exp2(gk)), kg = k * exp2(gn - gk) ----
        w_base = w_out + (bos * H + i_h) * K
        kg_base = kg_out + (bos * H + i_h) * K
        last_idx = min(i_t * BT + BT, T) - 1

        for i_k in range(NK):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(g + last_idx * H * K + o_k, mask=m_k, other=0.0).to(
                tl.float32
            )

            p_k0 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
            )
            p_k1 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_k2 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
            )
            p_k3 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
            )

            p_gk0 = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
            )
            p_gk1 = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_gk2 = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
            )
            p_gk3 = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
            )

            b_k0r = tl.load(p_k0, boundary_check=(0, 1))
            b_k1r = tl.load(p_k1, boundary_check=(0, 1))
            b_k2r = tl.load(p_k2, boundary_check=(0, 1))
            b_k3r = tl.load(p_k3, boundary_check=(0, 1))

            b_gk0r = tl.load(p_gk0, boundary_check=(0, 1)).to(tl.float32)
            b_gk1r = tl.load(p_gk1, boundary_check=(0, 1)).to(tl.float32)
            b_gk2r = tl.load(p_gk2, boundary_check=(0, 1)).to(tl.float32)
            b_gk3r = tl.load(p_gk3, boundary_check=(0, 1)).to(tl.float32)

            b_kb0 = (b_k0r * b_b0[:, None] * exp2(b_gk0r)).to(b_k0r.dtype)
            b_kb1 = (b_k1r * b_b1r[:, None] * exp2(b_gk1r)).to(b_k1r.dtype)
            b_kb2 = (b_k2r * b_b2r[:, None] * exp2(b_gk2r)).to(b_k2r.dtype)
            b_kb3 = (b_k3r * b_b3r[:, None] * exp2(b_gk3r)).to(b_k3r.dtype)

            b_w0 = tl.dot(b_Ai00_h, b_kb0)
            b_w1 = tl.dot(b_Ai10_h, b_kb0) + tl.dot(b_Ai11_h, b_kb1)
            b_w2 = (
                tl.dot(b_Ai20_h, b_kb0)
                + tl.dot(b_Ai21_h, b_kb1)
                + tl.dot(b_Ai22_h, b_kb2)
            )
            b_w3 = (
                tl.dot(b_Ai30_h, b_kb0)
                + tl.dot(b_Ai31_h, b_kb1)
                + tl.dot(b_Ai32_h, b_kb2)
                + tl.dot(b_Ai33_h, b_kb3)
            )

            p_w0 = tl.make_block_ptr(
                w_base, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
            )
            p_w1 = tl.make_block_ptr(
                w_base, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_w2 = tl.make_block_ptr(
                w_base, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
            )
            p_w3 = tl.make_block_ptr(
                w_base, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
            )
            tl.store(p_w0, b_w0.to(p_w0.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_w1, b_w1.to(p_w1.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_w2, b_w2.to(p_w2.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_w3, b_w3.to(p_w3.dtype.element_ty), boundary_check=(0, 1))

            b_kg0 = b_k0r * exp2(b_gn[None, :] - b_gk0r)
            b_kg1 = b_k1r * exp2(b_gn[None, :] - b_gk1r)
            b_kg2 = b_k2r * exp2(b_gn[None, :] - b_gk2r)
            b_kg3 = b_k3r * exp2(b_gn[None, :] - b_gk3r)

            p_kg0 = tl.make_block_ptr(
                kg_base, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
            )
            p_kg1 = tl.make_block_ptr(
                kg_base, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_kg2 = tl.make_block_ptr(
                kg_base, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
            )
            p_kg3 = tl.make_block_ptr(
                kg_base, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
            )
            tl.store(p_kg0, b_kg0.to(p_kg0.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_kg1, b_kg1.to(p_kg1.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_kg2, b_kg2.to(p_kg2.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_kg3, b_kg3.to(p_kg3.dtype.element_ty), boundary_check=(0, 1))
    else:
        p_Akk00 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0)
        )
        p_Akk10 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        p_Akk11 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0)
        )
        p_Akk20 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Akk21 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0)
        )
        p_Akk22 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0)
        )
        p_Akk30 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        p_Akk31 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0)
        )
        p_Akk32 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)
        )
        p_Akk33 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0)
        )

        tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))
