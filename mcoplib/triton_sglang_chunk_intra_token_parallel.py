# Adapted from flash-linear-attention project.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# KDA intra-chunk diagonal-block kernel for MetaX C600-U (MACA / Triton).
#
# Sub-chunk-parallel, read-once restructure of the original token-parallel
# kernel. Each program owns one <=BC-token diagonal sub-chunk (for a head
# group) and reads every q/k/g/beta row exactly once (vs the token-parallel
# version's ~8.5x redundant re-read of k_j/g_j, which was 86% of its traffic).
#
# The pairwise decay 2^(g_i - g_j) is split, per channel, around the sub-chunk's
# first row g0:  2^(g_i-g_j) = 2^(g_i-g0) * 2^(-(g_j-g0)).  This is exact and
# lets the two intra-sub-chunk blocks be written as fp32 matmuls:
#     Aqk[i,j] = scale * <q_i * 2^(g_i-g0),  k_j * 2^(-(g_j-g0))>      (j <= i)
#     Akk[i,j] =         <beta_i*k_i * 2^(g_i-g0), k_j * 2^(-(g_j-g0))> (j <  i)
# g is a chunk-local cumsum of non-positive gates, so within a BC=16 sub-chunk
# g_i-g0 is non-positive. Large negative differences can still overflow the
# reciprocal factor; numerical range validation remains necessary.

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T", "N"])
def chunk_kda_fwd_kernel_intra_subchunk(
    q, k, g, beta, Aqk, Akk, scale,
    cu_seqlens, T, N,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
    BK: tl.constexpr, BH: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_slot = tl.program_id(0)
    i_hg = tl.program_id(1)

    if IS_VARLEN:
        # Request r owns virtual slots [floor(bos/BC)+r,
        # floor(eos/BC)+r+1). This interval has at least ceil(length/BC)
        # entries and at most one extra. Its boundaries are strictly increasing
        # even for empty requests. No prefix-sum table or CPU readback is needed.
        left, right = 0, N
        while left < right:
            mid = (left + right) // 2
            end = tl.load(cu_seqlens + mid + 1).to(tl.int32)
            if i_slot < end // BC + mid + 1:
                right = mid
            else:
                left = mid + 1
        if left >= N:
            return
        bos = tl.load(cu_seqlens + left).to(tl.int32)
        eos = tl.load(cu_seqlens + left + 1).to(tl.int32)
        local_token = (i_slot - (bos // BC + left)) * BC
    else:
        slots_per_seq = tl.cdiv(T, BC)
        bos = (i_slot // slots_per_seq) * T
        eos = bos + T
        local_token = (i_slot % slots_per_seq) * BC

    off = bos + local_token
    if off >= eos:
        return
    n = tl.minimum(BC, eos - off)
    col0 = local_token % BT

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    row_valid = o_i < n
    m_k = o_k < K
    tri_le = o_i[:, None] >= o_i[None, :]     # j <= i  (Aqk written region)
    tri_lt = o_i[:, None] > o_i[None, :]      # j <  i  (Akk nonzero region)

    for ho in tl.static_range(BH):
        head = i_hg * BH + ho
        if head < H:
            base = off * H * K + head * K
            p_q = tl.make_block_ptr(q + base, (n, K), (H * K, 1), (0, 0), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k + base, (n, K), (H * K, 1), (0, 0), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g + base, (n, K), (H * K, 1), (0, 0), (BC, BK), (1, 0))
            p_g0 = tl.make_block_ptr(g + base, (1, K), (H * K, 1), (0, 0), (1, BK), (1, 0))
            p_beta = tl.make_block_ptr(beta + off * H + head, (n,), (H,), (0,), (BC,), (0,))

            b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
            b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)   # [1, BK]
            b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)  # [BC]

            b_gr = b_g - b_g0                       # [BC, BK], <= 0 within sub-chunk
            b_dp = tl.math.exp2(b_gr)               # 2^(g_i - g0)
            b_dm = 1.0 / b_dp                        # 2^(-(g_j - g0)) = 1 / 2^(g_j-g0)
            # zero padded-K columns so they don't enter the reductions
            b_qg = tl.where(m_k[None, :], b_q * b_dp, 0.0)
            b_kg = tl.where(m_k[None, :], b_k * b_dm, 0.0)
            b_kbg = tl.where(m_k[None, :], b_k * b_beta[:, None] * b_dp, 0.0)

            # stacked dot: qg and kbg share the kg contraction, so pack them along
            # M into one [2*BC, K] tile -> a single [32,128]x[128,16] MMA instead
            # of two [16,128]x[128,16] (better M-utilization, half the launches).
            b_j = tl.join(b_qg, b_kbg)                       # [BC, BK, 2]
            b_qkbg = tl.reshape(tl.permute(b_j, (2, 0, 1)), (2 * BC, BK))
            b_kg_t = tl.trans(b_kg)                          # [BK, BC]
            # Aqk and FP32 Akk share this dot. Do not reduce multiplication
            # precision to TF32: validate elementwise Akk error on C600 before
            # reintroducing any faster precision mode.
            b_A = tl.dot(b_qkbg, b_kg_t, allow_tf32=False)   # [2*BC, BC]
            # unpack: rows [0:BC) -> Aqk, rows [BC:2BC) -> Akk
            b_A = tl.reshape(b_A, (2, BC, BC))               # [2, BC, BC]
            b_Aqk, b_Akk = tl.split(tl.permute(b_A, (1, 2, 0)))  # each [BC, BC]
            b_Aqk = b_Aqk * scale

            # masked coalesced block stores (leave j>i untouched for inter_solve)
            aqk_base = Aqk + off * H * BT + head * BT + col0
            akk_base = Akk + off * H * BC + head * BC
            aqk_ptr = aqk_base + o_i[:, None] * (H * BT) + o_i[None, :]
            akk_ptr = akk_base + o_i[:, None] * (H * BC) + o_i[None, :]
            mask_a = tri_le & row_valid[:, None]
            mask_k = tri_le & row_valid[:, None]
            tl.store(aqk_ptr, b_Aqk.to(Aqk.dtype.element_ty), mask=mask_a)
            tl.store(akk_ptr, tl.where(tri_lt, b_Akk, 0.0).to(Akk.dtype.element_ty), mask=mask_k)


def chunk_kda_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens=None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
) -> None:
    """Sub-chunk-parallel KDA intra-chunk diagonal-block kernel (read-once).

    Public signature is unchanged from the token-parallel version. q,k,gk are
    [B,T,H,K] (gk = chunk-local cumsum of gates, fp32); beta [B,T,H]; Aqk
    [B,T,H,BT]; Akk [B,T,H,BC]. Writes Aqk/Akk in place, diagonal blocks only.
    """
    B, T, H, K = q.shape
    BT = chunk_size
    BC = sub_chunk_size
    if BC < 16 or BC & (BC - 1) or BT < BC or BT % BC:
        raise ValueError("BC must be a power of two >=16 and divide BT")
    if B * T == 0 or H == 0:
        return Aqk, Akk
    if cu_seqlens is not None and cu_seqlens.device != q.device:
        raise ValueError("cu_seqlens must be on the same device as q")
    BK = triton.next_power_of_2(K)
    N = cu_seqlens.numel() - 1 if cu_seqlens is not None else B
    if N < 1:
        raise ValueError("nonempty input requires at least one sequence")

    def grid(meta):
        num_slots = B * T // BC + N if cu_seqlens is not None else B * triton.cdiv(T, BC)
        return (num_slots, triton.cdiv(H, meta["BH"]))

    chunk_kda_fwd_kernel_intra_subchunk[grid](
        q, k, gk, beta, Aqk, Akk, scale,
        cu_seqlens, T, N,
        H=H, K=K, BT=BT, BC=BC, BK=BK,
        IS_VARLEN=cu_seqlens is not None,
        BH=1, num_warps=4,
    )
    return Aqk, Akk
