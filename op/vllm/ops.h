#pragma once

#include <optional>
#include <tuple>
#include <torch/library.h>

#include "core/scalar_type.hpp"

#include <vector>
#include <torch/csrc/stable/ops.h>
inline torch::Tensor weak_ref_tensor(torch::Tensor& tensor) {
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
              std::optional<torch::Tensor> weight, double epsilon,
              bool zero_centered = false);
void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        std::optional<torch::Tensor> weight,
                        double epsilon, bool zero_centered = false);
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
step5_fused_qk_norm_rope(
    const torch::Tensor& qkv, const torch::Tensor& q_weight,
    const torch::Tensor& k_weight, const torch::Tensor& cos,
    const torch::Tensor& sin, const torch::Tensor& positions,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t rotary_pairs, double eps, double norm_weight_bias,
    const std::optional<torch::Tensor>& q_out,
    const std::optional<torch::Tensor>& k_out,
    const std::optional<torch::Tensor>& v_out);
                        
torch::stable::Tensor fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    torch::stable::Tensor const& q_in, torch::stable::Tensor const& kv,
    torch::stable::Tensor& k_cache, torch::stable::Tensor const& slot_mapping,
    torch::stable::Tensor const& position_ids,
    torch::stable::Tensor const& cos_sin_cache, int64_t q_head_padded,
    double eps, int64_t cache_block_size, bool apply_q_norm);

void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out(
    torch::Tensor const& q_in, torch::Tensor const& kv,
    torch::Tensor& q_out, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping,
    torch::Tensor const& position_ids,
    torch::Tensor const& cos_sin_cache, int64_t q_head_padded,
    double eps, int64_t cache_block_size);

// Metax-added: same as above, but the KV NoPE plane is quantized with plain
// symmetric int8 (absmax/127, fp32 per-tile scale) instead of UE8M0 FP8, for
// the SWA int8 cache path.
torch::Tensor fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert(
    torch::Tensor const& q_in, torch::Tensor const& kv, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping, torch::Tensor const& position_ids,
    torch::Tensor const& cos_sin_cache, int64_t q_head_padded, double eps,
    int64_t cache_block_size);

torch::Tensor fused_deepseek_v41_qnorm_rope_kv_rope_int8_insert(
    const torch::Tensor& q_in, const torch::Tensor& kv, torch::Tensor& k_cache,
    const torch::Tensor& slot_mapping, const torch::Tensor& position_ids,
    const torch::Tensor& cos_sin_cache, int64_t q_head_padded, double eps,
    int64_t cache_block_size,  bool apply_q_norm);

void fused_deepseek_v4_qnorm_rope_kv_rope_insert(
    torch::Tensor& q, torch::Tensor const& kv, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping, torch::Tensor const& position_ids,
    torch::Tensor const& cos_sin_cache, double eps, int64_t cache_block_size);

void fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
    torch::stable::Tensor& q, torch::stable::Tensor const& kv,
    torch::stable::Tensor& k_cache, torch::stable::Tensor const& slot_mapping,
    torch::stable::Tensor const& position_ids,
    torch::stable::Tensor const& cos_sin_cache, double eps,
    int64_t cache_block_size, bool apply_q_norm);

void fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
    torch::stable::Tensor const& q, torch::stable::Tensor const& kv,
    torch::stable::Tensor& q_fp8, torch::stable::Tensor& k_cache,
    torch::stable::Tensor const& slot_mapping,
    torch::stable::Tensor const& position_ids,
    torch::stable::Tensor const& cos_sin_cache,
    torch::stable::Tensor const& fp8_scale,
    torch::stable::Tensor const& q_fp8_scale_inv, double eps,
    int64_t cache_block_size, bool apply_q_norm);
    
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

torch::Tensor region_topk_ids(const torch::Tensor& logits,
                              const torch::Tensor& lengths, int64_t topk);

void stable_topk_gathered(torch::Tensor gathered, torch::Tensor out,
                          int64_t topk);

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
// SwiGLU-step + per-group quant:
//   result = min(SiLU_alpha(gate), limit) * (clamp(up, -limit, limit) + beta)
// with SiLU_alpha(x) = x * sigmoid(alpha * x). Supports FP8-e4m3fn and INT8
// output; group_size 64/128; scales [M, ceil(H/gs)] or transposed [G, M].
void swiglu_step_and_mul_per_block_quant(
    torch::Tensor& out, torch::Tensor const& input, double limit,
    torch::Tensor& scales, int64_t group_size,
    std::optional<torch::Tensor> scale_ub, bool is_scale_transposed,
    double alpha = 1.0, double beta = 0.0);
    

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
                        double alpha = 1.0, double beta = 0.0,
                        bool step4 = false);

void silu_and_mul_quant(torch::Tensor& out, torch::Tensor& input,
                        torch::Tensor& scale);

void mul_and_silu(torch::Tensor& out, torch::Tensor& input);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void fatrelu_and_mul(torch::Tensor& out, torch::Tensor& input,
                     double threshold);

void swigluoai_and_mul(torch::Tensor& out, torch::Tensor& input,
                       double alpha = 1.702, double limit = 7.0);

void situ_and_mul(torch::Tensor& out, torch::Tensor& input,
                  double beta = 1.0, double linear_beta = -1.0);
void masked_situ_and_mul(torch::Tensor& out,
                         torch::Tensor& input,
                         const torch::Tensor& expert_num_tokens,
                         double beta = 1.0, double linear_beta = -1.0);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

void gelu_quick(torch::Tensor& out, torch::Tensor& input);

void relu_squared(torch::Tensor& out, torch::Tensor& input);

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

void per_token_group_quant_fp8(const torch::stable::Tensor& input,
                               torch::stable::Tensor& output_q,
                               torch::stable::Tensor& output_s,
                               int64_t group_size, double eps, double fp8_min,
                               double fp8_max, bool scale_ue8m0,
                               bool dummy_is_scale_transposed,
                               bool dummy_is_tma_aligned);

// Fused activation quantisation + DeepGEMM-compatible UE8M0-packed scales.
void per_token_group_quant_8bit_packed(const torch::stable::Tensor& input,
                                       torch::stable::Tensor& output_q,
                                       torch::stable::Tensor& output_s_packed,
                                       int64_t group_size, double eps,
                                       double min_8bit, double max_8bit);
void per_token_group_quant_int8(const torch::stable::Tensor& input,
                                torch::stable::Tensor& output_q,
                                torch::stable::Tensor& output_s,
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
                       torch::Tensor const& mat_b, bool enable_pdl);

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

void concat_and_cache_mla_grouped(torch::Tensor& kv_c,
                                  torch::Tensor& k_pe,
                                  torch::Tensor& kv_cache_ptrs,
                                  torch::Tensor& slot_mapping,
                                  int64_t block_size, int64_t block_stride,
                                  int64_t entry_stride);

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
    const std::string& kv_cache_dtype, bool skip_index_branch,
    std::optional<torch::stable::Tensor> q_fp8_out, double q_fp8_scale);

void fused_kimi_k3_mla_key_concat_kv_cache_insert(
    torch::Tensor& q, torch::Tensor const& k_nope,
    torch::Tensor const& k_pe, torch::Tensor const& kv_c_normed,
    torch::Tensor& k_out, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping, int64_t cache_block_size,
    std::optional<torch::Tensor> position_ids,
    std::optional<torch::Tensor> cos_sin_cache);

void fused_kimi_k3_mla_key_concat_ds_mla_insert(
    torch::Tensor& q, torch::Tensor const& k_nope,
    torch::Tensor const& k_pe, torch::Tensor const& kv_c_normed,
    torch::Tensor& k_out, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping, int64_t cache_block_size,
    std::optional<torch::Tensor> position_ids,
    std::optional<torch::Tensor> cos_sin_cache);

void fused_kimi_k3_mla_kv_concat(torch::Tensor const& k_nope,
                                 torch::Tensor const& k_pe,
                                 torch::Tensor& k_out);

void fused_kimi_k3_mla_kv_concat_quant_fp8(torch::Tensor const& k_nope,
                                           torch::Tensor const& k_pe,
                                           torch::Tensor const& v,
                                           torch::Tensor& k_fp8,
                                           torch::Tensor& v_fp8);

void fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert(
    torch::Tensor const& q, torch::Tensor const& k_nope,
    torch::Tensor const& k_pe, torch::Tensor const& kv_c_normed,
    torch::Tensor const& v, torch::Tensor& q_fp8,
    torch::Tensor& k_fp8, torch::Tensor& v_fp8,
    torch::Tensor& k_cache, torch::Tensor const& slot_mapping,
    torch::Tensor const& q_scale_inv,
    torch::Tensor const& k_scale_inv,
    torch::Tensor const& v_scale_inv,
    torch::Tensor const& cache_scale_inv, int64_t cache_block_size,
    std::optional<torch::Tensor> position_ids,
    std::optional<torch::Tensor> cos_sin_cache);

void fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(
    torch::Tensor const& ql_nope, torch::Tensor const& q_pe,
    torch::Tensor const& kv_c_normed, torch::Tensor const& k_pe,
    torch::Tensor& mqa_q, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping, int64_t cache_block_size,
    std::optional<torch::Tensor> position_ids,
    std::optional<torch::Tensor> cos_sin_cache);

void fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert(
    torch::Tensor const& ql_nope, torch::Tensor const& q_pe,
    torch::Tensor const& kv_c_normed, torch::Tensor const& k_pe,
    torch::Tensor& mqa_q, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping,
    torch::Tensor const& q_scale_inv,
    torch::Tensor const& cache_scale_inv, int64_t cache_block_size,
    std::optional<torch::Tensor> position_ids,
    std::optional<torch::Tensor> cos_sin_cache);

void fused_kimi_k3_mla_decode_q_concat_ds_mla_insert(
    torch::Tensor const& ql_nope, torch::Tensor const& q_pe,
    torch::Tensor const& kv_c_normed, torch::Tensor const& k_pe,
    torch::Tensor& mqa_q, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping, int64_t cache_block_size,
    std::optional<torch::Tensor> position_ids,
    std::optional<torch::Tensor> cos_sin_cache);

#ifdef VLLM_ENABLE_FUSED_KDA_DECODE
void fused_kda_decode(
    torch::Tensor const& x, torch::Tensor const& weight,
    std::optional<torch::Tensor> bias,
    torch::Tensor& conv_state, torch::Tensor const& raw_g,
    torch::Tensor const& raw_beta, torch::Tensor const& a_log,
    torch::Tensor const& dt_bias,
    torch::Tensor const& state_indices, torch::Tensor& state,
    torch::Tensor& out, std::optional<double> lower_bound,
    std::optional<torch::Tensor> output_gate,
    std::optional<torch::Tensor> norm_weight, double norm_eps);

void fused_gdn_decode_post_conv_mtp(
    torch::Tensor const& mixed_qkv, torch::Tensor const& a,
    torch::Tensor const& b, torch::Tensor const& a_log,
    torch::Tensor const& dt_bias,
    torch::Tensor const& state_indices,
    torch::Tensor const& cu_seqlens,
    torch::Tensor const& num_accepted_tokens,
    torch::Tensor& state, torch::Tensor const& output_gate,
    torch::Tensor const& norm_weight, torch::Tensor& out,
    double scale, double norm_eps);

#endif

#ifdef VLLM_ENABLE_KIMI_K3_ATTN_RES
void kimi_k3_attn_res(torch::Tensor& prefix,
                      torch::Tensor const& delta,
                      torch::Tensor const& blocks,
                      torch::Tensor const& norm_weight,
                      torch::Tensor const& qk_weight,
                      torch::Tensor const& output_norm_weight,
                      torch::Tensor& output, int64_t num_blocks,
                      double eps, double output_norm_eps);
#endif

using fptr_t = int64_t;
fptr_t init_custom_ar(const std::vector<int64_t>& fake_ipc_ptrs,
                      torch::stable::Tensor& rank_data, int64_t rank,
                      bool fully_connected);
void all_reduce(fptr_t _fa, torch::stable::Tensor& inp,
                torch::stable::Tensor& out, fptr_t reg_buffer,
                int64_t reg_buffer_sz_bytes);

void custom_all_gather(fptr_t _fa, torch::Tensor& inp,
                       torch::Tensor& out, fptr_t reg_buffer,
                       int64_t reg_buffer_sz_bytes);
void mnnvl_lamport_all_gather(fptr_t _fa, torch::Tensor& inp,
                              torch::Tensor& out, fptr_t local_buffer,
                              fptr_t multicast_buffer, fptr_t epoch_buffer,
                              int64_t stage_sz_bytes);
void custom_reduce_scatter(fptr_t _fa, torch::Tensor& inp,
                           torch::Tensor& out, fptr_t reg_buffer,
                           int64_t reg_buffer_sz_bytes);
void mnnvl_lamport_reduce_scatter(fptr_t _fa, torch::Tensor& inp,
                                  torch::Tensor& out,
                                  fptr_t local_buffer, fptr_t epoch_buffer,
                                  int64_t stage_sz_bytes);
void dispose(fptr_t _fa);
int64_t meta_size();
void register_buffer(fptr_t _fa, const std::vector<int64_t>& fake_ipc_ptrs);
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa);
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);
std::tuple<int64_t, torch::stable::Tensor> allocate_shared_buffer_and_handle(
    int64_t size);
int64_t open_mem_handle(torch::stable::Tensor& mem_handle);
void free_shared_buffer(int64_t buffer);

// LongCat n-gram embedding index kernel (see ngram_embedding_kernels.cu).
void ngram_compute_n_gram_ids(
    int64_t ne_n, int64_t ne_k, torch::Tensor& ne_weights,
    torch::Tensor& ne_mods,
    torch::Tensor& exclusive_ne_embedder_size_sums,
    torch::Tensor& exclusive_req_len_sums,
    torch::Tensor& ne_token_table, torch::Tensor& row_indices,
    torch::Tensor& column_starts, torch::Tensor& n_gram_ids);