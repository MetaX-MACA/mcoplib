#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> indexer_norm_rope(
    torch::Tensor const& index_q, torch::Tensor const& index_k,
    torch::Tensor const& index_z, torch::Tensor const& q_norm_weight,
    torch::Tensor const& k_norm_weight, torch::Tensor const& k_norm_bias,
    torch::Tensor const& cos, torch::Tensor const& sin,
    torch::Tensor const& positions, int64_t head_dim, int64_t rotary_dim,
    double eps, double q_norm_weight_bias);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> indexer_norm_rope_out(
    torch::Tensor const& index_q, torch::Tensor const& index_k,
    torch::Tensor const& index_z, torch::Tensor q_out, torch::Tensor k_out,
    torch::Tensor const& q_norm_weight, torch::Tensor const& k_norm_weight,
    torch::Tensor const& k_norm_bias, torch::Tensor const& cos,
    torch::Tensor const& sin, torch::Tensor const& positions,
    int64_t head_dim, int64_t rotary_dim, double eps,
    double q_norm_weight_bias);

torch::Tensor indexer_norm_rope_packed_(
    torch::Tensor index_qkz, torch::Tensor const& q_norm_weight,
    torch::Tensor const& k_norm_weight, torch::Tensor const& k_norm_bias,
    torch::Tensor const& cos, torch::Tensor const& sin,
    torch::Tensor const& positions, int64_t num_q_heads,
    int64_t num_k_heads, int64_t head_dim, int64_t rotary_dim, double eps,
    double q_norm_weight_bias);
