import math
from functools import cache

import torch

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
from mcoplib.tilelang_mhc_pre import mhc_pre


RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT_VALUE = 2.0
SINKHORN_REPEAT = 20


@cache
def _compute_num_splits(num_tokens: int) -> int:
    if num_tokens < 512:
        return 64
    if num_tokens < 1024:
        return 32
    if num_tokens < 1536:
        return 16
    if num_tokens < 3072:
        return 32
    if num_tokens < 4096:
        return 8
    if num_tokens < 8192:
        return 16
    return 8


class Mhc_pre_big_fuse_kernel_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = int(config.get("num_tokens", 96))
        self.hidden_size = int(config.get("hidden_size", 4096))
        self.mhc_mult = int(config.get("mhc_mult", 4))
        self.mhc_mult2 = self.mhc_mult * self.mhc_mult
        self.mhc_mult3 = self.mhc_mult * (2 + self.mhc_mult)
        self.n_splits = _compute_num_splits(self.num_tokens)
        self.seed = config.get("seed", None)

    def _make_inputs(self, dev):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        residual = torch.randn(
            (self.num_tokens, self.mhc_mult, self.hidden_size),
            device=dev,
            dtype=torch.bfloat16,
        )
        fn = torch.randn(
            (self.mhc_mult3, self.mhc_mult * self.hidden_size),
            device=dev,
            dtype=torch.float32,
        ) / math.sqrt(self.mhc_mult * self.hidden_size)
        hc_scale = torch.randn((3,), device=dev, dtype=torch.float32) * 0.2 + 1.0
        hc_base = torch.randn(
            (self.mhc_mult3,), device=dev, dtype=torch.float32
        ) * 0.2

        k_chunk = (self.mhc_mult * self.hidden_size) // self.n_splits
        residual_2d = residual.view(self.num_tokens, self.mhc_mult * self.hidden_size).float()
        gemm_out_mul = torch.empty(
            (self.n_splits, self.num_tokens, self.mhc_mult3),
            device=dev,
            dtype=torch.float32,
        )
        gemm_out_sqrsum = torch.empty(
            (self.n_splits, self.num_tokens), device=dev, dtype=torch.float32
        )
        for s in range(self.n_splits):
            a = residual_2d[:, s * k_chunk : (s + 1) * k_chunk]
            b = fn[:, s * k_chunk : (s + 1) * k_chunk]
            gemm_out_mul[s] = a @ b.t()
            gemm_out_sqrsum[s] = a.square().sum(-1)

        return residual, hc_scale, hc_base, gemm_out_mul, gemm_out_sqrsum

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", "bfloat16"))
        state.add_summary("Shape", f"({self.num_tokens} {self.mhc_mult} {self.hidden_size})")

        read_bytes = (
            self.num_tokens * self.mhc_mult * self.hidden_size * 2
            + self.n_splits * self.num_tokens * self.mhc_mult3 * 4
            + self.n_splits * self.num_tokens * 4
            + (3 + self.mhc_mult3) * 4
        )
        write_bytes = (
            self.num_tokens * self.mhc_mult * 4
            + self.num_tokens * self.mhc_mult2 * 4
            + self.num_tokens * self.hidden_size * 2
        )
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

        total_elements = (
            self.num_tokens * self.mhc_mult * self.hidden_size
            + self.n_splits * self.num_tokens * self.mhc_mult3
            + self.n_splits * self.num_tokens
            + self.num_tokens * self.mhc_mult
            + self.num_tokens * self.mhc_mult2
            + self.num_tokens * self.hidden_size
        )
        state.add_element_count(total_elements)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = f"cuda:{dev_id}"
        with torch.cuda.stream(tc_s):
            residual, hc_scale, hc_base, gemm_out_mul, gemm_out_sqrsum = self._make_inputs(dev)
        return self.make_launcher(
            dev_id,
            mhc_pre,
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            SINKHORN_REPEAT,
            self.n_splits,
        )

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        residual, hc_scale, hc_base, gemm_out_mul, gemm_out_sqrsum = self._make_inputs(dev)

        post_mix, comb_mix, layer_input = mhc_pre(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            SINKHORN_REPEAT,
            self.n_splits,
        )

        rms = torch.rsqrt(
            gemm_out_sqrsum.sum(0) / (self.mhc_mult * self.hidden_size) + RMS_EPS
        )
        mixes = gemm_out_mul.sum(0) * rms.unsqueeze(-1)

        pre_mix = (
            torch.sigmoid(
                mixes[:, : self.mhc_mult] * hc_scale[0]
                + hc_base[: self.mhc_mult]
            )
            + HC_PRE_EPS
        )
        post_ref = (
            torch.sigmoid(
                mixes[:, self.mhc_mult : 2 * self.mhc_mult] * hc_scale[1]
                + hc_base[self.mhc_mult : 2 * self.mhc_mult]
            )
            * HC_POST_MULT_VALUE
        ).unsqueeze(-1)

        comb_ref = (
            mixes[:, 2 * self.mhc_mult :] * hc_scale[2]
            + hc_base[2 * self.mhc_mult :]
        ).view(-1, self.mhc_mult, self.mhc_mult)
        comb_ref = torch.softmax(comb_ref, dim=-1) + HC_SINKHORN_EPS
        comb_ref = comb_ref / (
            comb_ref.sum(dim=-2, keepdim=True) + HC_SINKHORN_EPS
        )
        for _ in range(SINKHORN_REPEAT - 1):
            comb_ref = comb_ref / (
                comb_ref.sum(dim=-1, keepdim=True) + HC_SINKHORN_EPS
            )
            comb_ref = comb_ref / (
                comb_ref.sum(dim=-2, keepdim=True) + HC_SINKHORN_EPS
            )

        layer_ref = (
            residual.float() * pre_mix.unsqueeze(-1)
        ).sum(dim=1).bfloat16()

        out_all = torch.cat(
            [
                post_mix.flatten().float(),
                comb_mix.flatten().float(),
                layer_input.flatten().float(),
            ]
        )
        ref_all = torch.cat(
            [
                post_ref.flatten().float(),
                comb_ref.flatten().float(),
                layer_ref.flatten().float(),
            ]
        )
        return self.check_diff(out_all, ref_all, threshold=0.9999)
