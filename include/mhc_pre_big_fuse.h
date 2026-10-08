#pragma once

#include <torch/extension.h>

void mhc_pre_big_fuse_out(
    torch::Tensor gemm_out_mul, torch::Tensor gemm_out_sqrsum,
    torch::Tensor hc_scale, torch::Tensor hc_base, torch::Tensor residual,
    torch::Tensor post_mix, torch::Tensor comb_mix, torch::Tensor layer_input,
    double rms_eps, double hc_pre_eps, double hc_sinkhorn_eps,
    double hc_post_mult_value, int64_t sinkhorn_repeat, int64_t n_splits);
