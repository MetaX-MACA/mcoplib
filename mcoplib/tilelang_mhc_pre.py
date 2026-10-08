"""Standalone production TileLang kernel for the MHC pre operator."""

import math

import tilelang
import tilelang.language as T
import torch

try:
    from mcoplib import op as _default_op
except ImportError:
    _default_op = None


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
}


@tilelang.jit(pass_configs=PASS_CONFIGS)
def mhc_pre_tilelang_optimized(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    mhc_mult: int = 4,
    hidden_block_size: int = 512,
    num_stages: int = 1,
    sinkhorn_unroll_factor: int = 19,
    rms_lanes: int = 8,
    mix_lanes_per_value: int = 2,
    residual_use_shared: bool = True,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic("num_tokens")
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    hidden_block = math.gcd(hidden_block_size, hidden_size)

    @T.prim_func
    def mhc_pre_big_fuse_kernel(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, threads=128) as pid:
            rms_shared = T.alloc_shared(8, T.float32)
            mix_shared = T.alloc_shared(
                mhc_mult3 * mix_lanes_per_value, T.float32
            )

            if rms_lanes == 1:
                if T.get_thread_binding() == 0:
                    rms = T.alloc_fragment(1, T.float32)
                    rms[0] = 0
                    for i_split in T.serial(n_splits):
                        rms[0] += gemm_out_sqrsum[i_split, pid]
                    rms_shared[0] = T.rsqrt(
                        rms[0] / (mhc_mult * hidden_size) + rms_eps
                    )
            else:
                if T.get_thread_binding() < rms_lanes:
                    rms = T.alloc_fragment(1, T.float32)
                    rms[0] = 0
                    for i_part in T.serial(
                        (n_splits + rms_lanes - 1) // rms_lanes
                    ):
                        i_split = i_part * rms_lanes + T.get_thread_binding()
                        if i_split < n_splits:
                            rms[0] += gemm_out_sqrsum[i_split, pid]
                    rms_shared[T.get_thread_binding()] = rms[0]

                T.sync_threads()

                if T.get_thread_binding() == 0:
                    rms = T.alloc_fragment(1, T.float32)
                    rms[0] = 0
                    for i_lane in T.serial(rms_lanes):
                        rms[0] += rms_shared[i_lane]
                    rms_shared[0] = T.rsqrt(
                        rms[0] / (mhc_mult * hidden_size) + rms_eps
                    )

            T.sync_threads()

            if mix_lanes_per_value == 1:
                if T.get_thread_binding() < mhc_mult3:
                    mixes = T.alloc_fragment(mhc_mult3, T.float32)
                    T.clear(mixes)
                    for j in T.Parallel(mhc_mult3):
                        mixes[j] = 0
                        for i_split in T.serial(n_splits):
                            mixes[j] += gemm_out_mul[i_split, pid, j]
                        mixes[j] *= rms_shared[0]
                    T.copy(mixes, mix_shared, disable_tma=True)
            else:
                if (
                    T.get_thread_binding()
                    < mhc_mult3 * mix_lanes_per_value
                ):
                    mix_partial = T.alloc_fragment(1, T.float32)
                    mix_partial[0] = 0
                    mix_lane = T.get_thread_binding() // mhc_mult3
                    mix_index = T.get_thread_binding() % mhc_mult3
                    for i_part in T.serial(
                        (n_splits + mix_lanes_per_value - 1)
                        // mix_lanes_per_value
                    ):
                        i_split = (
                            i_part * mix_lanes_per_value + mix_lane
                        )
                        if i_split < n_splits:
                            mix_partial[0] += gemm_out_mul[
                                i_split, pid, mix_index
                            ]
                    mix_shared[T.get_thread_binding()] = mix_partial[0]

                T.sync_threads()

                if T.get_thread_binding() < mhc_mult3:
                    mix_value = T.alloc_fragment(1, T.float32)
                    mix_value[0] = 0
                    for i_lane in T.serial(mix_lanes_per_value):
                        mix_value[0] += mix_shared[
                            i_lane * mhc_mult3 + T.get_thread_binding()
                        ]
                    mix_shared[T.get_thread_binding()] = (
                        mix_value[0] * rms_shared[0]
                    )

            if T.get_thread_binding() < 64:
                cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)

                for j in T.Parallel(mhc_mult):
                    post_mix[pid, j] = (
                        T.sigmoid(
                            mix_shared[j + mhc_mult] * mhc_scale[1]
                            + mhc_base[j + mhc_mult]
                        )
                        * mhc_post_mult_value
                    )

                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = (
                        mix_shared[j * mhc_mult + k + mhc_mult * 2]
                        * mhc_scale[2]
                        + mhc_base[j * mhc_mult + k + mhc_mult * 2]
                    )

                row_sum = T.alloc_fragment(mhc_mult, T.float32)
                col_sum = T.alloc_fragment(mhc_mult, T.float32)
                row_max = T.alloc_fragment(mhc_mult, T.float32)

                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (
                        col_sum[k] + mhc_sinkhorn_eps
                    )

                for _ in T.unroll(
                    sinkhorn_repeat - 1,
                    unroll_factor=sinkhorn_unroll_factor,
                ):
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (
                            row_sum[j] + mhc_sinkhorn_eps
                        )
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (
                            col_sum[k] + mhc_sinkhorn_eps
                        )

                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    comb_mix[pid, j * mhc_mult + k] = cm[j, k]
            else:
                for j in T.Parallel(mhc_mult):
                    rms_shared[j] = (
                        T.sigmoid(
                            mix_shared[j] * mhc_scale[0]
                            + mhc_base[j]
                        )
                        + mhc_pre_eps
                    )

                for i0_h in T.Pipelined(
                    hidden_size // hidden_block,
                    num_stages=num_stages,
                ):
                    xl = T.alloc_fragment(
                        (mhc_mult, hidden_block), T.float32
                    )
                    if residual_use_shared:
                        xs = T.alloc_shared(
                            (mhc_mult, hidden_block), T.bfloat16
                        )
                        T.copy(
                            residual[pid, 0, i0_h * hidden_block],
                            xs,
                            disable_tma=True,
                        )
                        T.copy(xs, xl, disable_tma=True)
                    else:
                        T.copy(
                            residual[pid, 0, i0_h * hidden_block],
                            xl,
                            disable_tma=True,
                        )

                    ol = T.alloc_fragment((hidden_block,), T.float32)
                    T.clear(ol)
                    for i_mhc in T.serial(mhc_mult):
                        pre = rms_shared[i_mhc]
                        for i1_h in T.Parallel(hidden_block):
                            ol[i1_h] += pre * xl[i_mhc, i1_h]

                    T.copy(
                        ol,
                        layer_input[pid, i0_h * hidden_block],
                        disable_tma=True,
                    )

    return mhc_pre_big_fuse_kernel


def mhc_pre(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    residual: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert gemm_out_mul.dtype == torch.float32
    assert gemm_out_sqrsum.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    assert gemm_out_mul.shape == (n_splits, num_tokens, mhc_mult3)
    assert gemm_out_sqrsum.shape == (n_splits, num_tokens)
    assert mhc_scale.shape == (3,)
    assert mhc_base.shape == (mhc_mult3,)

    post_mix = torch.empty(
        num_tokens, mhc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, mhc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    use_cuda_specialization = (
        _default_op is not None
        and hasattr(_default_op, "mhc_pre_big_fuse_out")
        and hidden_size == 7168
        and mhc_mult == 4
        and n_splits in (16, 64)
        and rms_eps == 1e-6
        and mhc_pre_eps == 1e-6
        and mhc_sinkhorn_eps == 1e-6
        and mhc_post_mult_value == 2.0
        and sinkhorn_repeat == 20
    )
    if use_cuda_specialization:
        _default_op.mhc_pre_big_fuse_out(
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_scale,
            mhc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            rms_eps,
            mhc_pre_eps,
            mhc_sinkhorn_eps,
            mhc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
        )
    else:
        kernel = mhc_pre_tilelang_optimized(
            hidden_size,
            rms_eps,
            mhc_pre_eps,
            mhc_sinkhorn_eps,
            mhc_post_mult_value,
            sinkhorn_repeat,
            n_splits=n_splits,
            mhc_mult=mhc_mult,
            hidden_block_size=512,
            num_stages=1,
            sinkhorn_unroll_factor=19,
            mix_lanes_per_value=4 if n_splits == 64 else 2,
            residual_use_shared=not (32 <= num_tokens <= 280),
        )
        kernel(
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_scale,
            mhc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
        )

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


__all__ = [
    "mhc_pre",
    "mhc_pre_tilelang_optimized",
]
