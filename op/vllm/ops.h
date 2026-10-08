#pragma once

#include <optional>
#include <torch/library.h>

#include "core/scalar_type.hpp"

#include <vector>
#include <torch/csrc/stable/ops.h>
torch::Tensor weak_ref_tensor(torch::Tensor& tensor) {
  // Ensure tensor is on CUDA
  if (!tensor.is_cuda()) {
    throw std::runtime_error("Tensor must be on CUDA device");
  }

  // Get the raw data pointer
  void* data_ptr = tensor.data_ptr();

  // Get tensor sizes and strides
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();

  // Get tensor options (dtype, device)
  auto options = tensor.options();

  // Create a new tensor from the raw data pointer
  auto new_tensor = torch::from_blob(data_ptr, sizes, strides, options);

  return new_tensor;
}

void paged_attention_v1(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void paged_attention_v2(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void merge_attn_states(
    torch::Tensor& output, std::optional<torch::Tensor> output_lse,
    const torch::Tensor& prefix_output, const torch::Tensor& prefix_lse,
    const torch::Tensor& suffix_output, const torch::Tensor& suffix_lse,
    const std::optional<int64_t> prefill_tokens_with_context,
    const std::optional<torch::Tensor>& output_scale = std::nullopt);

void rms_norm(torch::Tensor& out, torch::Tensor& input,
              std::optional<torch::Tensor> weight, double epsilon);

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        std::optional<torch::Tensor> weight,
                        double epsilon);

//Todo：fused_qk_norm_rope算子新增参数，部分依赖（async_util.cuh），先保持原样
// void fused_qk_norm_rope(torch::Tensor& qkv, int64_t num_heads_q,
//                         int64_t num_heads_k, int64_t num_heads_v,
//                         int64_t head_dim, double eps, torch::Tensor& q_weight,
//                         torch::Tensor& k_weight, torch::Tensor& cos_sin_cache,
//                         bool is_neox, torch::Tensor& position_ids,
//                         int64_t forced_token_heads_per_warp);

void fused_qk_norm_rope(torch::Tensor& qkv, int64_t num_heads_q,
                        int64_t num_heads_k, int64_t num_heads_v,
                        int64_t head_dim, double eps, torch::Tensor& q_weight,
                        torch::Tensor& k_weight, torch::Tensor& cos_sin_cache,
                        bool is_neox, torch::Tensor& position_ids);
//Todo：该算子后续替换成int8量化
torch::Tensor fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    torch::Tensor const& q_in, torch::Tensor const& kv, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping, torch::Tensor const& position_ids,
    torch::Tensor const& cos_sin_cache, int64_t q_head_padded, double eps,
    int64_t cache_block_size);

void fused_deepseek_v4_qnorm_rope_kv_rope_insert(
    torch::Tensor& q, torch::Tensor const& kv, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping, torch::Tensor const& position_ids,
    torch::Tensor const& cos_sin_cache, double eps, int64_t cache_block_size);

void fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
    torch::Tensor& q, torch::Tensor const& kv,
    torch::Tensor& k_cache, torch::Tensor const& slot_mapping,
    torch::Tensor const& position_ids,
    torch::Tensor const& cos_sin_cache, double eps,
    int64_t cache_block_size);

void fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
    torch::Tensor const& q, torch::Tensor const& kv,
    torch::Tensor& q_fp8, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping,
    torch::Tensor const& position_ids,
    torch::Tensor const& cos_sin_cache,
    torch::Tensor const& fp8_scale,
    torch::Tensor const& q_fp8_scale_inv, double eps,
    int64_t cache_block_size);
    
void apply_repetition_penalties_(torch::Tensor& logits,
                                 const torch::Tensor& prompt_mask,
                                 const torch::Tensor& output_mask,
                                 const torch::Tensor& repetition_penalties);

void top_k_per_row_prefill(const torch::Tensor& logits,
                           const torch::Tensor& rowStarts,
                           const torch::Tensor& rowEnds, torch::Tensor& indices,
                           int64_t numRows, int64_t stride0, int64_t stride1,
                           int64_t topK);

void top_k_per_row_decode(const torch::Tensor& logits, int64_t next_n,
                          const torch::Tensor& seqLens, torch::Tensor& indices,
                          int64_t numRows, int64_t stride0, int64_t stride1,
                          int64_t topK);


void persistent_topk(const torch::Tensor& logits, const torch::Tensor& lengths,
                     torch::Tensor& output, torch::Tensor& workspace, int64_t k,
                     int64_t max_seq_len);

void fused_unpack(const torch::Tensor& packed,
                  int64_t topk, int64_t n,
                    torch::Tensor& topk_weights,
                    torch::Tensor& topk_ids,
                    torch::Tensor& scale);

void rms_norm_static_fp8_quant(torch::Tensor& out, torch::Tensor& input,
                               torch::Tensor& weight, torch::Tensor& scale,
                               double epsilon);

void fused_add_rms_norm_static_fp8_quant(torch::Tensor& out,
                                         torch::Tensor& input,
                                         torch::Tensor& residual,
                                         torch::Tensor& weight,
                                         torch::Tensor& scale, double epsilon);

void rms_norm_dynamic_per_token_quant(torch::Tensor& out,
                                      torch::Tensor const& input,
                                      torch::Tensor const& weight,
                                      torch::Tensor& scales,
                                      double const epsilon,
                                      std::optional<torch::Tensor> scale_ub,
                                      std::optional<torch::Tensor> residual);

void rms_norm_per_block_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& weight,
                              torch::Tensor& scales, double const epsilon,
                              std::optional<torch::Tensor> scale_ub,
                              std::optional<torch::Tensor> residual,
                              int64_t group_size, bool is_scale_transposed);

void silu_and_mul_per_block_quant(torch::Tensor& out,
                                  torch::Tensor const& input,
                                  torch::Tensor& scales, int64_t group_size,
                                  std::optional<torch::Tensor> scale_ub,
                                  bool is_scale_transposed);

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse);

void batched_rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                              std::optional<torch::Tensor> key,
                              int64_t head_size, torch::Tensor& cos_sin_cache,
                              bool is_neox, int64_t rot_dim,
                              torch::Tensor& cos_sin_cache_offsets);

void persistent_masked_m_silu_mul_quant(
    const torch::Tensor& input,              // (E, T, 2*H)
    const torch::Tensor& tokens_per_expert,  // (E)
    torch::Tensor& y_q,                      // (E, T, H) [OUT]
    torch::Tensor& y_s,  // (E, T, H//group_size) [OUT]
    bool use_ue8m0);

// SwiGLU-step variant: Gate_act = min(SiLU(gate), limit);
// Up_act = clip(up, -limit, limit); Result = Gate_act * Up_act.
// limit is a scalar tensor (bf16 or fp32).
void persistent_masked_m_swiglu_mul_quant(
    const torch::Tensor& input,              // (E, T, 2*H)
    const torch::Tensor& tokens_per_expert,  // (E)
    torch::Tensor& y_q,                      // (E, T, H) [OUT]
    torch::Tensor& y_s,  // (E, T, H//group_size) [OUT]
    const torch::Tensor& limit,              // scalar tensor (bf16 or fp32)
    bool use_ue8m0);
    

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void silu_and_mul_clamp(torch::Tensor& out, torch::Tensor& input, double limit,
                        double alpha = 1.0, double beta = 0.0);

void silu_and_mul_quant(torch::Tensor& out, torch::Tensor& input,
                        torch::Tensor& scale);

void mul_and_silu(torch::Tensor& out, torch::Tensor& input);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void fatrelu_and_mul(torch::Tensor& out, torch::Tensor& input,
                     double threshold);

void swigluoai_and_mul(torch::Tensor& out, torch::Tensor& input,
                       double alpha = 1.702, double limit = 7.0);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

void gelu_quick(torch::Tensor& out, torch::Tensor& input);

torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor);

torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int64_t split_k_iters, torch::Tensor _temp_space,
                       bool dtype_bf16);

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy);

torch::Tensor awq_to_gptq_4bit(torch::Tensor qweight);

torch::Tensor permute_cols(torch::Tensor const& A, torch::Tensor const& perm);

torch::Tensor ggml_dequantize(torch::Tensor W, int64_t type, int64_t m,
                              int64_t n,
                              std::optional<at::ScalarType> const& dtype);

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W, torch::Tensor X,
                                  int64_t type, int64_t row);

torch::Tensor ggml_mul_mat_a8(torch::Tensor W, torch::Tensor X, int64_t type,
                              int64_t row);

torch::Tensor ggml_moe_a8(torch::Tensor X, torch::Tensor W,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t type,
                          int64_t row, int64_t top_k, int64_t tokens);

torch::Tensor ggml_moe_a8_vec(torch::Tensor X, torch::Tensor W,
                              torch::Tensor topk_ids, int64_t top_k,
                              int64_t type, int64_t row, int64_t tokens);

int64_t ggml_moe_get_block_size(int64_t type);

// void scaled_fp4_quant(torch::Tensor& output, torch::Tensor const& input,
//                       torch::Tensor& output_scale,
//                       torch::Tensor const& input_scale);

void scaled_fp4_experts_quant(
    torch::Tensor& output, torch::Tensor& output_scale,
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts);

void static_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& scale,
                              std::optional<torch::Tensor> const& azp);

void dynamic_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                               torch::Tensor& scales,
                               std::optional<torch::Tensor> const& azp);

torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, int64_t bit, int64_t group_size,
                        torch::Tensor perm_space, torch::Tensor temp_space,
                        bool dtype_bf16);

void gptq_shuffle(torch::Tensor q_weight, torch::Tensor q_perm, int64_t bit);

void static_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor const& scale,
    std::optional<std::tuple<int64_t, int64_t>> group_shape = std::nullopt);

void dynamic_scaled_fp8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor& scale);

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale,
    std::optional<torch::Tensor> const& scale_ub);

void per_token_group_quant_fp8(
    torch::Tensor const& input, torch::Tensor& output_q,
    torch::Tensor& output_s, int64_t group_size, double eps, double fp8_min,
    double fp8_max, bool scale_ue8m0,
    bool dummy_is_scale_transposed = false,
    bool dummy_is_tma_aligned = false);

// Fused activation quantisation + DeepGEMM-compatible UE8M0-packed scales.
void per_token_group_quant_8bit_packed(const torch::Tensor& input,
                                       torch::Tensor& output_q,
                                       torch::Tensor& output_s_packed,
                                       int64_t group_size, double eps,
                                       double min_8bit, double max_8bit);

void per_token_group_quant_int8(const torch::Tensor& input,
                                torch::Tensor& output_q,
                                torch::Tensor& output_s,
                                int64_t group_size, double eps, double int8_min,
                                double int8_max);   

void selective_scan_fwd(const torch::Tensor& u, const torch::Tensor& delta,
                        const torch::Tensor& A, const torch::Tensor& B,
                        const torch::Tensor& C,
                        const std::optional<torch::Tensor>& D_,
                        const std::optional<torch::Tensor>& z_,
                        const std::optional<torch::Tensor>& delta_bias_,
                        bool delta_softplus,
                        const std::optional<torch::Tensor>& query_start_loc,
                        const std::optional<torch::Tensor>& cache_indices,
                        const std::optional<torch::Tensor>& has_initial_state,
                        const torch::Tensor& ssm_states, int64_t pad_slot_id);

void dsv3_fused_a_gemm(torch::Tensor& output, torch::Tensor const& mat_a,
                       torch::Tensor const& mat_b);

void fp32_router_gemm(torch::Tensor& output, torch::Tensor const& mat_a,
                      torch::Tensor const& mat_b);

// Todo:PTX2CPP，minimax_reduce_rms_kernel中有两个device函数依赖PTX
torch::Tensor minimax_allreduce_rms(torch::Tensor const& input,
                                    torch::Tensor const& norm_weight,
                                    torch::Tensor workspace, int64_t const rank,
                                    int64_t const nranks, double const eps);
std::tuple<torch::Tensor, torch::Tensor> minimax_allreduce_rms_qk(
    torch::Tensor qkv, torch::Tensor const& norm_weight_q,
    torch::Tensor const& norm_weight_k, torch::Tensor workspace,
    int64_t const q_size, int64_t const kv_size, int64_t const rank,
    int64_t const nranks, double const eps);

// Horizontally-fused MiniMax-M3 QK-norm + partial NeoX RoPE (+ optional KV /
// index-cache insert). Dense layer: norm+RoPE only; sparse layer: also packs
// the index branch and scatters k/v/index_k into their paged caches.
void fused_minimax_m3_qknorm_rope_kv_insert(
    torch::stable::Tensor& qkv, torch::stable::Tensor const& q_norm_weight,
    torch::stable::Tensor const& k_norm_weight,
    torch::stable::Tensor const& cos_sin_cache,
    torch::stable::Tensor const& positions, int64_t num_heads,
    int64_t num_kv_heads, int64_t rotary_dim, double eps,
    std::optional<torch::stable::Tensor> index_q_norm_weight,
    std::optional<torch::stable::Tensor> index_k_norm_weight,
    int64_t num_index_heads, std::optional<torch::stable::Tensor> slot_mapping,
    std::optional<torch::stable::Tensor> index_slot_mapping,
    std::optional<torch::stable::Tensor> kv_cache,
    std::optional<torch::stable::Tensor> index_cache, int64_t block_size,
    std::optional<torch::stable::Tensor> q_out,
    std::optional<torch::stable::Tensor> index_q_out,
    const std::string& kv_cache_dtype);

