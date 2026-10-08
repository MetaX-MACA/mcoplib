#pragma once

#include <ATen/ATen.h>

// Applies per-head RMSNorm to the packed Q and K regions of token_data in
// place. The packed V region is left unchanged. The returned tensor aliases
// token_data.
torch::Tensor qk_rms_norm_inplace_cuda(
    torch::Tensor token_data,
    torch::Tensor const& q_norm_weight,
    torch::Tensor const& k_norm_weight,
    int64_t q_head_num,
    int64_t kv_head_num,
    int64_t qk_head_dim,
    double eps);
