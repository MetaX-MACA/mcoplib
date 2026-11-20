#pragma once

#include <torch/extension.h>

#include <optional>

// Native CUDA implementation of the GLM5.3-flash KDA target-verify contract.
// The out entry mirrors the captured Triton kernel arguments.  Unsupported
// combinations fail explicitly instead of silently changing semantics.
void fused_sigmoid_gating_delta_rule_update_native_out(
    torch::Tensor const& A_log, torch::Tensor const& a,
    torch::Tensor const& dt_bias, double softplus_beta,
    double softplus_threshold, double lower_bound, torch::Tensor const& q,
    torch::Tensor const& k, torch::Tensor const& v, torch::Tensor const& b,
    torch::Tensor o, torch::Tensor const& h0_source,
    torch::Tensor const& h0_indices, int64_t stride_h0_source,
    torch::Tensor const& cu_seqlens, torch::Tensor intermediate_states_buffer,
    torch::Tensor const& intermediate_state_indices, int64_t cache_steps,
    std::optional<torch::Tensor> const& retrieve_parent_token_ptr,
    int64_t stride_retrieve_parent_token_seq,
    int64_t stride_retrieve_parent_token_token, double scale, int64_t T,
    int64_t stride_a, int64_t stride_q, int64_t stride_k, int64_t stride_v,
    int64_t stride_b, int64_t NP2_T, int64_t B, int64_t H, int64_t HV,
    int64_t K, int64_t V, int64_t BK, int64_t BV,
    bool USE_INITIAL_STATE, bool USE_QK_L2NORM_IN_KERNEL, bool IS_VARLEN,
    bool IS_KDA, bool USE_LOWER_BOUND, bool DISABLE_STATE_UPDATE,
    bool CACHE_INTERMEDIATE_STATES, bool HAS_EAGLE_TREE_CUSTOM_ATTN_MASK,
    std::optional<torch::Tensor> const& replayssm_rawv,
    std::optional<torch::Tensor> const& replayssm_rawk,
    std::optional<torch::Tensor> const& replayssm_g,
    std::optional<torch::Tensor> const& replayssm_beta,
    int64_t stride_rawv_slot, int64_t stride_rawk_slot,
    int64_t stride_g_slot, int64_t stride_beta_slot, int64_t MAX_CACHE_LEN,
    bool CACHE_RING, bool SPLIT_N_HV_GRID,
    std::optional<torch::Tensor> const& workspace = std::nullopt,
    int64_t value_block = 0);

// Allocating entry with the same user-facing argument order as the Triton
// wrapper for the supported target-verify cases.
torch::Tensor fused_sigmoid_gating_delta_rule_update_native(
    torch::Tensor const& A_log, torch::Tensor const& a,
    torch::Tensor const& dt_bias, double softplus_beta,
    double softplus_threshold, torch::Tensor const& q, torch::Tensor const& k,
    torch::Tensor const& v, torch::Tensor const& b,
    torch::Tensor const& initial_state_source,
    torch::Tensor const& initial_state_indices,
    std::optional<double> scale = std::nullopt,
    bool use_qk_l2norm_in_kernel = false,
    std::optional<torch::Tensor> const& cu_seqlens = std::nullopt,
    bool is_kda = false, std::optional<double> lower_bound = std::nullopt,
    bool disable_state_update = false,
    std::optional<torch::Tensor> const& intermediate_states_buffer =
        std::nullopt,
    std::optional<torch::Tensor> const& intermediate_state_indices =
        std::nullopt,
    std::optional<int64_t> cache_steps = std::nullopt,
    std::optional<torch::Tensor> const& retrieve_parent_token = std::nullopt,
    bool cache_ring = false,
    std::optional<torch::Tensor> const& replayssm_rawv = std::nullopt,
    std::optional<torch::Tensor> const& replayssm_rawk = std::nullopt,
    std::optional<torch::Tensor> const& replayssm_g = std::nullopt,
    std::optional<torch::Tensor> const& replayssm_beta = std::nullopt);

// Diagnostic entries used to time the two production launches separately and
// to test the alternative value-block tiling.  They reuse the production
// kernels, validation rules and workspace layout; production dispatch keeps
// using the entries above.
void fused_sigmoid_gating_delta_rule_update_native_precompute(
    torch::Tensor const& A_log, torch::Tensor const& a,
    torch::Tensor const& dt_bias, torch::Tensor const& b,
    torch::Tensor const& q, torch::Tensor const& k,
    torch::Tensor const& workspace, double lower_bound, int64_t stride_a,
    int64_t stride_b, int64_t stride_q, int64_t stride_k, int64_t T);

void fused_sigmoid_gating_delta_rule_update_native_recurrent(
    torch::Tensor const& workspace, torch::Tensor const& q,
    torch::Tensor const& k, torch::Tensor const& v, torch::Tensor o,
    torch::Tensor const& h0_source, torch::Tensor const& h0_indices,
    torch::Tensor const& cu_seqlens,
    torch::Tensor intermediate_states_buffer,
    torch::Tensor const& intermediate_state_indices, int64_t stride_h0_source,
    int64_t stride_q, int64_t stride_k, int64_t stride_v, double scale,
    int64_t T, int64_t value_block, int64_t variant = 0);
