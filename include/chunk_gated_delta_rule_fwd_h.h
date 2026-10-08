#pragma once

#include <torch/extension.h>

#include <optional>
#include <string>
#include <tuple>

// Retained CUDA implementation of chunk_gated_delta_rule_fwd_h.
//
// Public allocating entry. Its Python-visible signature mirrors the Triton
// entry; unsupported contracts raise instead of silently falling back.
std::tuple<torch::Tensor, std::optional<torch::Tensor>>
chunk_gated_delta_rule_fwd_h(
    torch::Tensor const& k, torch::Tensor const& w, torch::Tensor const& u,
    std::optional<torch::Tensor> const& g,
    std::optional<torch::Tensor> const& gk,
    std::optional<torch::Tensor> const& initial_state,
    std::optional<torch::Tensor> const& initial_state_indices,
    bool save_new_value,
    std::optional<torch::Tensor> const& cu_seqlens,
    std::optional<torch::Tensor> const& chunk_indices,
    bool use_exp2, std::optional<int64_t> block_v,
    std::optional<int64_t> num_warps, int64_t num_stages);

// Preallocated-output native entry. The optimized path is BF16 K=V=128;
// a generic CUDA path covers K<=256, arbitrary V, dense B>1, fp16/bf16/fp32
// tensors, whole chunks or ragged tails, optional scalar g and/or gk, GQA,
// pooled state
// with negative sentinels, optional v_new write-back, and use_exp2.
void chunk_gated_delta_rule_fwd_h_native_out(
    torch::Tensor const& k, torch::Tensor const& w, torch::Tensor const& u,
    std::optional<torch::Tensor> const& gk,
    torch::Tensor const& initial_state,
    torch::Tensor const& initial_state_indices,
    torch::Tensor const& cu_seqlens, torch::Tensor const& chunk_offsets,
    torch::Tensor h, std::optional<torch::Tensor> const& v_new,
    int64_t sequences,
    int64_t chunks_per_sequence, bool use_exp2, bool ragged,
    bool state_bf16, std::optional<torch::Tensor> const& g = std::nullopt);

// Mirrors the production Triton host-side dispatch. Returns
// "<branch> <cuda_serves> <cuda_executes>" for the given contract, i.e. the same
// three fields the probe's `--dispatch` oracle reports.
std::string chunk_gated_delta_rule_fwd_h_coverage(
    int64_t B, int64_t T, int64_t Hg, int64_t K, int64_t H, int64_t V,
    int64_t block_v, int64_t num_warps, int64_t num_stages, int64_t N,
    int64_t NT, bool has_g, bool has_gk, bool has_v_new, bool has_cu_seqlens,
    bool tail_free);
