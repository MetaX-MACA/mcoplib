#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include "core/registration.h"

#include <torch/library.h>
#include <torch/version.h>
#include <torch/csrc/stable/library.h>
// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

#if ENABLE_OP_PROFILING
#warning "ENABLE_OP_PROFILING is ENABLED"
#else
#warning "ENABLE_OP_PROFILING is DISABLED"
#endif


TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // vLLM custom ops
  //

  // The default behavior in PyTorch 2.6 was changed to "requires_contiguous",
  // so we need
  // to override this for many GEMMs with the following tag. Otherwise,
  // torch.compile will force all input tensors to be contiguous(), which
  // will break many custom ops that require column-major weight matrices.
  // This was a bug and PyTorch 2.7 has since fixed this.
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 6
  #define stride_tag at::Tag::needs_fixed_stride_order
#else
  #define stride_tag
#endif

  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
  ops.impl("weak_ref_tensor", torch::kCUDA, &weak_ref_tensor);

  ops.def("get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor");
  ops.impl("get_cuda_view_from_cpu_tensor", torch::kCPU,
           &get_cuda_view_from_cpu_tensor);

  // ┌------------------------  No Used Backend for Metax
  // -------------------------┐ Attention ops Compute the attention between an
  // input query and the cached keys/values using PagedAttention.
  ops.def(
      "paged_attention_v1("
      "    Tensor! out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);

  // PagedAttention V2.
  ops.def(
      "paged_attention_v2("
      "    Tensor! out, Tensor! exp_sums, Tensor! max_logits,"
      "    Tensor! tmp_out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v2", torch::kCUDA, &paged_attention_v2);

  // Merge attn states
  // Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
  // can be used to combine partial attention results (in the split-KV case)
  ops.def(
      "merge_attn_states("
      "    Tensor! output,"
      "    Tensor!? output_lse,"
      "    Tensor prefix_output,"
      "    Tensor prefix_lse,"
      "    Tensor suffix_output,"
      "    Tensor suffix_lse,"
      "    int!? prefill_tokens_with_context,"
      "    Tensor? output_scale=None) -> ()");
  ops.impl("merge_attn_states", torch::kCUDA, &merge_attn_states);


  // Activation ops
  // Activation function used in SwiGLU.
  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  ops.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  // SwiGLU activation with input clamping.
  ops.def(
      "silu_and_mul_with_clamp(Tensor! result, Tensor input, float limit, "
      "float alpha=1.0, float beta=0.0, bool step4=False) -> ()");
  ops.impl("silu_and_mul_with_clamp", torch::kCUDA, &silu_and_mul_clamp);

  ops.def(
      "persistent_masked_m_silu_mul_quant(Tensor input, Tensor counts, Tensor! "
      "y_q, Tensor! y_s, bool use_ue8m0) -> ()");
  ops.impl("persistent_masked_m_silu_mul_quant",torch::kCUDA, &persistent_masked_m_silu_mul_quant);

  // SwiGLU-step variant: SiLU(gate) clamped to [_, limit], up clamped to
  // [-limit, limit], then per-group FP8 quant.
  ops.def(
      "persistent_masked_m_swiglu_mul_quant(Tensor input, Tensor counts, "
      "Tensor! y_q, Tensor! y_s, Tensor limit, bool use_ue8m0) -> ()");
  ops.impl("persistent_masked_m_swiglu_mul_quant", torch::kCUDA,
           &persistent_masked_m_swiglu_mul_quant);

  ops.def(
      "silu_and_mul_quant(Tensor! result, Tensor input, Tensor scale) -> ()");
  ops.impl("silu_and_mul_quant", torch::kCUDA, &silu_and_mul_quant);

  // Fused SiLU+Mul + per-block quantization
  ops.def(
      "silu_and_mul_per_block_quant("
      "Tensor! out, "
      "Tensor input, "
      "Tensor! scales, "
      "int group_size, "
      "Tensor? scale_ub=None, "
      "bool is_scale_transposed=False) -> ()");
  ops.impl("silu_and_mul_per_block_quant", torch::kCUDA,
           &silu_and_mul_per_block_quant);

  // SwiGLU-step + per-block quantization:
  //   min(SiLU(gate), limit) * (clamp(up, -limit, limit) + beta)
  ops.def(
      "swiglu_step_and_mul_per_block_quant("
      "Tensor! out, "
      "Tensor input, "
      "float limit, "
      "Tensor! scales, "
      "int group_size, "
      "Tensor? scale_ub=None, "
      "bool is_scale_transposed=False, "
      "float alpha=1.0, "
      "float beta=0.0) -> ()");
  ops.impl("swiglu_step_and_mul_per_block_quant", torch::kCUDA,
           &swiglu_step_and_mul_per_block_quant);
           
  ops.def("mul_and_silu(Tensor! out, Tensor input) -> ()");
  ops.impl("mul_and_silu", torch::kCUDA, &mul_and_silu);

  // Activation function used in GeGLU with `none` approximation.
  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  // Activation function used in GeGLU with `tanh` approximation.
  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  // FATReLU implementation.
  ops.def("fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()");
  ops.impl("fatrelu_and_mul", torch::kCUDA, &fatrelu_and_mul);

  ops.def(
      "swigluoai_and_mul(Tensor! out, Tensor input, float alpha=1.702, float "
      "limit=7.0) "
      "-> ()");
  ops.impl("swigluoai_and_mul", torch::kCUDA, &swigluoai_and_mul);

  // SituGLU implementation used in Kimi models.
  ops.def(
      "situ_and_mul(Tensor! out, Tensor input, float beta=1.0, float "
      "linear_beta=-1.0) -> ()");
  ops.impl("situ_and_mul", torch::kCUDA, &situ_and_mul);
  ops.def(
      "masked_situ_and_mul(Tensor! out, Tensor input, Tensor "
      "expert_num_tokens, float beta=1.0, float linear_beta=-1.0) -> ()");
  ops.impl("masked_situ_and_mul", torch::kCUDA, &masked_situ_and_mul);

  // GELU implementation used in GPT-2.
  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_new", torch::kCUDA, &gelu_new);

  // Approximate GELU implementation.
  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_fast", torch::kCUDA, &gelu_fast);

  // Quick GELU implementation.
  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_quick", torch::kCUDA, &gelu_quick);

  // relu(x)^2 activation from https://arxiv.org/abs/2109.08668v2
  ops.def("relu_squared(Tensor! out, Tensor input) -> ()");
  ops.impl("relu_squared", torch::kCUDA, &relu_squared);

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon, "
      "bool zero_centered=False) -> ()");
  ops.impl("rms_norm", torch::kCUDA, &rms_norm);
  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, "
      "float epsilon, bool zero_centered=False) -> ()");
  ops.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);
  // Grouped concat_and_cache_mla across all layers (bf16 only). Each
  // layer's cache base pointer is read from kv_cache_ptrs.
  ops.def(
      "concat_and_cache_mla_grouped(Tensor kv_c, Tensor k_pe,"
      "                             Tensor kv_cache_ptrs,"
      "                             Tensor slot_mapping,"
      "                             int block_size, int block_stride,"
      "                             int entry_stride) -> ()");
  ops.impl("concat_and_cache_mla_grouped", torch::kCUDA, &concat_and_cache_mla_grouped);

//   // Function for fused QK Norm and RoPE
//   ops.def(
//       "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, "
//       "int num_heads_k, int num_heads_v, int head_dim, float eps, "
//       "Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache, "
//       "bool is_neox, Tensor position_ids, "
//       "int forced_token_heads_per_warp=-1) -> ()");
//   ops.impl("fused_qk_norm_rope", torch::kCUDA, &fused_qk_norm_rope);

  // Function for fused QK Norm and RoPE
  ops.def(
      "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, "
      "int num_heads_k, int num_heads_v, int head_dim, float eps, "
      "Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache, "
      "bool is_neox, Tensor position_ids) -> ()");
  ops.impl("fused_qk_norm_rope", torch::kCUDA, &fused_qk_norm_rope);

  ops.def(
      "step5_fused_qk_norm_rope("
      "Tensor qkv, Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin, "
      "Tensor positions, int num_q_heads, int num_kv_heads, int head_dim, "
      "int rotary_pairs, float eps, float norm_weight_bias, "
      "Tensor? q_out=None, Tensor? k_out=None, Tensor? v_out=None) -> "
      "(Tensor, Tensor, Tensor)");
  ops.impl("step5_fused_qk_norm_rope", torch::kCUDA,
           &step5_fused_qk_norm_rope);
  // Horizontally-fused DeepseekV4-MLA: per-head RMSNorm + GPT-J RoPE for Q, and
  // GPT-J RoPE + UE8M0 FP8 quant + paged cache insert for KV, all in one
  // kernel launch.
  ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert("
      "Tensor q_in, Tensor kv, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor position_ids, Tensor cos_sin_cache, "
      "int q_head_padded, float eps, int cache_block_size) -> Tensor");
  ops.impl("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert", torch::kCUDA,
           &fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert);

  ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out("
      "Tensor q_in, Tensor kv, Tensor! q_out, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor position_ids, Tensor cos_sin_cache, "
      "int q_head_padded, float eps, int cache_block_size) -> ()");
  ops.impl("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out", torch::kCUDA,
           &fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out);

  ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_insert("
      "Tensor! q, Tensor kv, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor position_ids, Tensor cos_sin_cache, "
      "float eps, int cache_block_size) -> ()");
  ops.impl("fused_deepseek_v4_qnorm_rope_kv_rope_insert", torch::kCUDA,
           &fused_deepseek_v4_qnorm_rope_kv_rope_insert);

  ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert("
      "Tensor q, Tensor kv, Tensor! q_fp8, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor position_ids, Tensor cos_sin_cache, "
      "Tensor fp8_scale, Tensor q_fp8_scale_inv, float eps, "
      "int cache_block_size) -> ()");
  ops.impl("fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert", torch::kCUDA, &fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert);
    
  ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert("
      "Tensor! q, Tensor kv, Tensor! k_cache, Tensor slot_mapping, "
      "Tensor position_ids, Tensor cos_sin_cache, float eps, "
      "int cache_block_size) -> ()");
  ops.impl("fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert", torch::kCUDA,
        &fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert);
           
  // Apply repetition penalties to logits in-place
  ops.def(
      "apply_repetition_penalties_(Tensor! logits, Tensor prompt_mask, "
      "Tensor output_mask, Tensor repetition_penalties) -> ()");
  ops.impl("apply_repetition_penalties_", torch::kCUDA,
           &apply_repetition_penalties_);

  // Optimized top-k per row operation
  ops.def(
      "top_k_per_row_prefill(Tensor logits, Tensor rowStarts, Tensor rowEnds, "
      "Tensor! indices, int numRows, int stride0, "
      "int stride1, int topK) -> ()");
  ops.impl("top_k_per_row_prefill", torch::kCUDA, &top_k_per_row_prefill);

  ops.def(
      "top_k_per_row_decode(Tensor logits, int next_n, "
      "Tensor seq_lens, Tensor! indices, "
      "int numRows, int stride0, int stride1, int topK) -> ()");
  ops.impl("top_k_per_row_decode", torch::kCUDA, &top_k_per_row_decode);
   
  ops.def(
      "persistent_topk(Tensor logits, Tensor lengths, Tensor! output, "
      "Tensor workspace, int k, int max_seq_len) -> ()");
  ops.impl("persistent_topk", torch::kCUDA, &persistent_topk);

  ops.def(
      "region_topk_ids(Tensor logits, Tensor lengths, int topk) -> Tensor");
  ops.impl("region_topk_ids", torch::kCUDA, &region_topk_ids);
  
  ops.def(
      "stable_topk_gathered(Tensor gathered, Tensor! out, int topk) -> ()");
  ops.impl("stable_topk_gathered", torch::kCUDA, &stable_topk_gathered);

  ops.def(
     "fused_unpack(Tensor packed, int topk, int n, "
     "Tensor(a!) topk_weights, Tensor(b!) topk_ids, Tensor(c!) scale) -> ()");
  ops.impl("fused_unpack", torch::kCUDA, &fused_unpack);

  // ┌------------------------  Not supported for Metax
  // ------------------------┐ Layernorm-quant Apply Root Mean Square (RMS)
  // -------------------------┐ Layernorm-quant Apply Root Mean Square (RMS)
  // Normalization to the input tensor.
  ops.def(
      "rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor weight, "
      "Tensor scale, float epsilon) -> "
      "()");
  ops.impl("rms_norm_static_fp8_quant", torch::kCUDA,
           &rms_norm_static_fp8_quant);
  // └------------------------- Not supported for Metax
  // -------------------------┘

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! residual, Tensor weight, "
      "Tensor scale, float epsilon) -> ()");
  ops.impl("fused_add_rms_norm_static_fp8_quant", torch::kCUDA,
           &fused_add_rms_norm_static_fp8_quant);

  // Fused Layernorm + Quant kernels
  ops.def(
      "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual) -> ()");
  ops.impl("rms_norm_dynamic_per_token_quant", torch::kCUDA,
           &rms_norm_dynamic_per_token_quant);

  // Fused Layernorm + Block quant kernels
  ops.def(
      "rms_norm_per_block_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual, int group_size, "
      "bool is_scale_transposed) -> ()");
  ops.impl("rms_norm_per_block_quant", torch::kCUDA, &rms_norm_per_block_quant);

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox, int "
      "rope_dim_offset=0, bool inverse=False) -> ()");
  ops.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);

  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key
  // (supports multiple loras).
  ops.def(
      "batched_rotary_embedding(Tensor positions, Tensor! query,"
      "                         Tensor!? key, int head_size,"
      "                         Tensor cos_sin_cache, bool is_neox,"
      "                         int rot_dim,"
      "                         Tensor cos_sin_cache_offsets) -> ()");
  ops.impl("batched_rotary_embedding", torch::kCUDA, &batched_rotary_embedding);

  // Quantization ops
  // DeepSeek V3 fused A GEMM (SM 9.0+, bf16 only, 1-16 tokens).
  ops.def(
      "dsv3_fused_a_gemm(Tensor! output, Tensor mat_a, Tensor mat_b, "
      "bool enable_pdl=False) -> ()");
  // conditionally compiled so impl registration is in source file

  // BF16/FP32 activation x FP32 weight -> FP32 router GEMM.
  ops.def("fp32_router_gemm(Tensor! output, Tensor mat_a, Tensor mat_b) -> ()");
  ops.impl("fp32_router_gemm", torch::kCUDA, &fp32_router_gemm);

  // Quantized GEMM for AWQ.
  ops.def(
      "awq_gemm(Tensor _in_feats, Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, SymInt split_k_iters, Tensor _temp_space, bool "
      "dtype_bf16) -> Tensor");
  ops.impl("awq_gemm", torch::kCUDA, &awq_gemm);

  // Dequantization for AWQ.
  ops.def(
      "awq_dequantize(Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, SymInt split_k_iters, int thx, int thy) -> Tensor");
  ops.impl("awq_dequantize", torch::kCUDA, &awq_dequantize);

  // Convert AWQ to GPTQ
  ops.def("awq_to_gptq_4bit(Tensor qweight) -> Tensor");
  ops.impl("awq_to_gptq_4bit", torch::kCUDA, &awq_to_gptq_4bit);

  // Quantized GEMM for GPTQ.
  // Note: even though the C++ inferred schema is correct for this op, it seems
  // to prevent the meta function registry.
  ops.def(
      "gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, "
      "Tensor b_gptq_scales, Tensor b_g_idx, bool use_exllama, int bit, int "
      "group_size, Tensor perm_space, "
      "Tensor temp_space, bool dtype_bf16)-> Tensor");
  ops.impl("gptq_gemm", torch::kCUDA, &gptq_gemm);

  // Post processing for GPTQ.
  ops.def("gptq_shuffle(Tensor! q_weight, Tensor q_perm, int bit) -> ()");
  ops.impl("gptq_shuffle", torch::kCUDA, &gptq_shuffle);

  // Dequantization for GGML.
  ops.def(
      "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
      "dtype) -> Tensor");
  ops.impl("ggml_dequantize", torch::kCUDA, &ggml_dequantize);

  // mmvq kernel for GGML.
  ops.def(
      "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.impl("ggml_mul_mat_vec_a8", torch::kCUDA, &ggml_mul_mat_vec_a8);

  // mmq kernel for GGML.
  ops.def("ggml_mul_mat_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.impl("ggml_mul_mat_a8", torch::kCUDA, &ggml_mul_mat_a8);

  // mmq kernel for GGML (MoE).
  ops.def(
      "ggml_moe_a8(Tensor X, Tensor W, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, int type, "
      "SymInt row, SymInt top_k, SymInt tokens) -> Tensor");
  ops.impl("ggml_moe_a8", torch::kCUDA, &ggml_moe_a8);

  // mmvq kernel for GGML (MoE).
  ops.def(
      "ggml_moe_a8_vec(Tensor X, Tensor W, "
      "Tensor topk_ids, int top_k, "
      "int type, SymInt row, SymInt tokens) -> Tensor");
  ops.impl("ggml_moe_a8_vec", torch::kCUDA, &ggml_moe_a8_vec);

  ops.def("ggml_moe_get_block_size(int type) -> int");
  ops.impl("ggml_moe_get_block_size", &ggml_moe_get_block_size);

  // ┌------------------------  Not supported for Metax
  // -------------------------┐ Compute FP8 quantized tensor for given scaling
  // factor.
  ops.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale, "
      "(int, int)? group_shape=None) -> ()");
  ops.impl("static_scaled_fp8_quant", torch::kCUDA, &static_scaled_fp8_quant);

  // Compute dynamic-per-tensor FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale) "
      "-> "
      "()");
  ops.impl("dynamic_scaled_fp8_quant", torch::kCUDA, &dynamic_scaled_fp8_quant);

  // Compute dynamic-per-token FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> "
      "()");
  ops.impl("dynamic_per_token_scaled_fp8_quant", torch::kCUDA,
           &dynamic_per_token_scaled_fp8_quant);

  // Compute per-token-group FP8 quantized tensor and scaling factor.
  // The trailing bool args are kept for vLLM call-site compatibility.
  ops.def(
      "per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! "
      "output_s, int group_size, float eps, float fp8_min, float fp8_max, "
      "bool scale_ue8m0, bool dummy_is_scale_transposed, "
      "bool dummy_is_tma_aligned) -> ()");

  // Compute per-token-group 8-bit quantized tensor and UE8M0-packed,
  // TMA-aligned scales for DeepGEMM.
  ops.def(
      "per_token_group_fp8_quant_packed(Tensor input, Tensor! output_q, "
      "Tensor! output_s_packed, int group_size, float eps, float fp8_min, "
      "float fp8_max) -> ()");
  // Compute per-token-group INT8 quantized tensor and scaling factor.
  ops.def(
      "per_token_group_quant_int8(Tensor input, Tensor! output_q, Tensor! "
      "output_s, int group_size, float eps, float int8_min, float int8_max) -> "
      "()");

  ops.def("permute_cols(Tensor A, Tensor perm) -> Tensor");
  ops.impl("permute_cols", torch::kCUDA, &permute_cols);

  // └------------------------- Not supported for Metax
  // -------------------------┘

  // Compute int8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_int8_quant(Tensor! result, Tensor input, Tensor scale,"
      "Tensor? azp) -> ()");
  ops.impl("static_scaled_int8_quant", torch::kCUDA, &static_scaled_int8_quant);

  // Compute int8 quantized tensor and scaling factor
  ops.def(
      "dynamic_scaled_int8_quant(Tensor! result, Tensor input, Tensor! scale, "
      "Tensor!? azp) -> ()");
  ops.impl("dynamic_scaled_int8_quant", torch::kCUDA,
           &dynamic_scaled_int8_quant);

  // Mamba selective scan kernel
  ops.def(
      "selective_scan_fwd(Tensor! u, Tensor! delta,"
      "Tensor! A, Tensor! B, Tensor! C,"
      "Tensor? D_, Tensor!? z_, Tensor? delta_bias_,"
      "bool delta_softplus,"
      "Tensor? query_start_loc,"
      "Tensor? cache_indices,"
      "Tensor? has_initial_state,"
      "Tensor! ssm_states,"
      "int pad_slot_id) -> ()");
  ops.impl("selective_scan_fwd", torch::kCUDA, &selective_scan_fwd);

  // LongCat n-gram embedding index kernel. All tensor args are marked mutable
  // to match the (non-const) stable-Tensor& C++ signature; only ne_token_table
  // and n_gram_ids are actually written in place.
  ops.def(
      "ngram_compute_n_gram_ids(int ne_n, int ne_k, Tensor(a!) ne_weights, "
      "Tensor(b!) ne_mods, Tensor(c!) exclusive_ne_embedder_size_sums, "
      "Tensor(d!) exclusive_req_len_sums, Tensor(e!) ne_token_table, "
      "Tensor(f!) row_indices, Tensor(g!) column_starts, "
      "Tensor(h!) n_gram_ids) -> ()");
  // LongCat n-gram embedding index kernel.
  ops.impl("ngram_compute_n_gram_ids", torch::kCUDA, &ngram_compute_n_gram_ids);

  ops.def(
      "minimax_allreduce_rms("
      "Tensor input,"
      "Tensor norm_weight,"
      "Tensor workspace,"
      "int rank,"
      "int nranks,"
      "float eps) -> Tensor");
  ops.impl("minimax_allreduce_rms", torch::kCUDA, &minimax_allreduce_rms);
  ops.def(
      "minimax_allreduce_rms_qk("
      "Tensor qkv,"
      "Tensor norm_weight_q,"
      "Tensor norm_weight_k,"
      "Tensor workspace,"
      "int q_size,"
      "int kv_size,"
      "int rank,"
      "int nranks,"
      "float eps) -> (Tensor, Tensor)");
  ops.impl("minimax_allreduce_rms_qk", torch::kCUDA, &minimax_allreduce_rms_qk);
  // Kimi-K3 MLA epilogues: optional RoPE followed by concat/cache insertion.
  ops.def(
      "fused_kimi_k3_mla_key_concat_kv_cache_insert("
      "Tensor! q, Tensor k_nope, Tensor k_pe, Tensor kv_c_normed, "
      "Tensor! k_out, Tensor! k_cache, Tensor slot_mapping, "
      "int cache_block_size, Tensor? position_ids=None, "
      "Tensor? cos_sin_cache=None) -> ()");
  ops.impl("fused_kimi_k3_mla_key_concat_kv_cache_insert", torch::kCUDA, &fused_kimi_k3_mla_key_concat_kv_cache_insert);
  ops.def(
      "fused_kimi_k3_mla_key_concat_ds_mla_insert("
      "Tensor! q, Tensor k_nope, Tensor k_pe, Tensor kv_c_normed, "
      "Tensor! k_out, Tensor! k_cache, Tensor slot_mapping, "
      "int cache_block_size, Tensor? position_ids=None, "
      "Tensor? cos_sin_cache=None) -> ()");
  ops.impl("fused_kimi_k3_mla_key_concat_ds_mla_insert", torch::kCUDA, &fused_kimi_k3_mla_key_concat_ds_mla_insert);
  ops.def(
      "fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert("
      "Tensor q, Tensor k_nope, Tensor k_pe, Tensor kv_c_normed, Tensor v, "
      "Tensor! q_fp8, Tensor! k_fp8, Tensor! v_fp8, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor q_scale_inv, Tensor k_scale_inv, "
      "Tensor v_scale_inv, Tensor cache_scale_inv, int cache_block_size, "
      "Tensor? position_ids=None, Tensor? cos_sin_cache=None) -> ()");
  ops.impl("fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert", torch::kCUDA, &fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert);
  ops.def(
      "fused_kimi_k3_mla_decode_q_concat_kv_cache_insert("
      "Tensor ql_nope, Tensor q_pe, Tensor kv_c_normed, Tensor k_pe, "
      "Tensor! mqa_q, Tensor! k_cache, Tensor slot_mapping, "
      "int cache_block_size, Tensor? position_ids=None, "
      "Tensor? cos_sin_cache=None) -> ()");
  ops.impl("fused_kimi_k3_mla_decode_q_concat_kv_cache_insert", torch::kCUDA, &fused_kimi_k3_mla_decode_q_concat_kv_cache_insert);
  ops.def(
      "fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert("
      "Tensor ql_nope, Tensor q_pe, Tensor kv_c_normed, Tensor k_pe, "
      "Tensor! mqa_q, Tensor! k_cache, Tensor slot_mapping, "
      "Tensor q_scale_inv, Tensor cache_scale_inv, int cache_block_size, "
      "Tensor? position_ids=None, Tensor? cos_sin_cache=None) -> ()");
  ops.impl("fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert", torch::kCUDA, &fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert);
  ops.def(
      "fused_kimi_k3_mla_decode_q_concat_ds_mla_insert("
      "Tensor ql_nope, Tensor q_pe, Tensor kv_c_normed, Tensor k_pe, "
      "Tensor! mqa_q, Tensor! k_cache, Tensor slot_mapping, "
      "int cache_block_size, Tensor? position_ids=None, "
      "Tensor? cos_sin_cache=None) -> ()");
  ops.impl("fused_kimi_k3_mla_decode_q_concat_ds_mla_insert", torch::kCUDA, &fused_kimi_k3_mla_decode_q_concat_ds_mla_insert);
#ifdef VLLM_ENABLE_FUSED_KDA_DECODE
  ops.def(
      "fused_kda_decode("
      "Tensor x, Tensor weight, Tensor? bias, Tensor! conv_state, "
      "Tensor raw_g, Tensor raw_beta, Tensor A_log, Tensor dt_bias, "
      "Tensor state_indices, Tensor! state, Tensor! out, "
      "float? lower_bound=None, Tensor? output_gate=None, "
      "Tensor? norm_weight=None, float norm_eps=1e-5) -> ()");
  ops.impl("fused_kda_decode", torch::kCUDA, &fused_kda_decode);
#endif
#ifdef VLLM_ENABLE_KIMI_K3_ATTN_RES
  ops.def(
      "kimi_k3_attn_res("
      "Tensor! prefix, Tensor delta, Tensor blocks, Tensor norm_weight, "
      "Tensor qk_weight, Tensor output_norm_weight, Tensor! output, "
      "int num_blocks, float eps, float output_norm_eps) -> ()");
  ops.impl("kimi_k3_attn_res", torch::kCUDA, &kimi_k3_attn_res);
#endif
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C_custom_ar, custom_ar) {
  custom_ar.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool fully_connected) -> int");
  custom_ar.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  custom_ar.def("dispose(int fa) -> ()");
  custom_ar.def("meta_size() -> int");
  custom_ar.def("register_buffer(int fa, int[] ipc_tensors) -> ()");
  custom_ar.def("get_graph_buffer_ipc_meta(int fa) -> (int[], int[])");
  custom_ar.def(
      "register_graph_buffers(int fa, int[][] handles, int[][] offsets) -> ()");
  custom_ar.def("allocate_shared_buffer_and_handle(int size) -> (int, Tensor)");
  custom_ar.def("open_mem_handle(Tensor mem_handle) -> int");
  custom_ar.def("free_shared_buffer(int ptr) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CUDA, custom_ar) {
  custom_ar.impl("init_custom_ar", TORCH_BOX(&init_custom_ar));
  custom_ar.impl("all_reduce", TORCH_BOX(&all_reduce));
}

STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CPU, custom_ar) {
  custom_ar.impl("open_mem_handle", TORCH_BOX(&open_mem_handle));
}

STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CompositeExplicitAutograd, custom_ar) {
  custom_ar.impl("dispose", TORCH_BOX(&dispose));
  custom_ar.impl("meta_size", TORCH_BOX(&meta_size));
  custom_ar.impl("register_buffer", TORCH_BOX(&register_buffer));
  custom_ar.impl("get_graph_buffer_ipc_meta",
                 TORCH_BOX(&get_graph_buffer_ipc_meta));
  custom_ar.impl("register_graph_buffers", TORCH_BOX(&register_graph_buffers));
  custom_ar.impl("allocate_shared_buffer_and_handle",
                 TORCH_BOX(&allocate_shared_buffer_and_handle));
  custom_ar.impl("free_shared_buffer", TORCH_BOX(&free_shared_buffer));
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), cache_ops) {
  // Cache ops
  // Swap in (out) the cache blocks from src to dst.
  cache_ops.def(
      "swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()");
  cache_ops.impl("swap_blocks", torch::kCUDA, &swap_blocks);

  // Reshape the key and value tensors and cache them.
  cache_ops.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  Tensor k_scale, Tensor v_scale) -> ()");
  cache_ops.impl("reshape_and_cache", torch::kCUDA, &reshape_and_cache);

  // Batch swap: submit all block copies in a single driver call.
  cache_ops.def(
      "swap_blocks_batch(Tensor src_ptrs, Tensor dst_ptrs,"
      "                  Tensor sizes,"
      "                  bool is_src_access_order_any=False) -> ()");
  cache_ops.impl("swap_blocks_batch", torch::kCPU, &swap_blocks_batch);

  // Reshape the key and value tensors and cache them.
  cache_ops.def(
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "                        Tensor! key_cache,"
      "                        Tensor! value_cache,"
      "                        Tensor slot_mapping,"
      "                        str kv_cache_dtype,"
      "                        Tensor k_scale, Tensor v_scale) -> ()");
  cache_ops.impl("reshape_and_cache_flash", torch::kCUDA,
                 &reshape_and_cache_flash);

  // Concat kv_c and k_pe and cache them.
  cache_ops.def(
      "concat_and_cache_mla(Tensor kv_c, Tensor k_pe,"
      "                     Tensor! kv_cache,"
      "                     Tensor slot_mapping,"
      "                     str kv_cache_dtype,"
      "                     Tensor scale) -> ()");
  cache_ops.impl("concat_and_cache_mla", torch::kCUDA, &concat_and_cache_mla);

  // Rotate Q and K, then write to kv cache for MLA
  cache_ops.def(
      "concat_and_cache_mla_rope_fused("
      "                     Tensor positions,"
      "                     Tensor! q_pe,"
      "                     Tensor! k_pe,"
      "                     Tensor kv_c,"
      "                     Tensor cos_sin_cache,"
      "                     bool is_neox,"
      "                     Tensor slot_mapping,"
      "                     Tensor! kv_cache,"
      "                     str kv_cache_dtype,"
      "                     Tensor kv_cache_scale) -> ()");
  cache_ops.impl("concat_and_cache_mla_rope_fused", torch::kCUDA,
                 &concat_and_cache_mla_rope_fused);

  // Convert the key and value cache to fp8 data type.
  cache_ops.def(
      "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
      "str kv_cache_dtype) -> ()");
  cache_ops.impl("convert_fp8", torch::kCUDA, &convert_fp8);

  // Gather cache blocks from src_cache to dst, dequantizing from
  // src_cache's dtype to dst's dtype if necessary.
  cache_ops.def(
      "gather_and_maybe_dequant_cache(Tensor src_cache, Tensor! dst, "
      "                               Tensor block_table, Tensor cu_seq_lens, "
      "                               Tensor token_to_seq, "
      "                               int num_tokens, "
      "                               str kv_cache_dtype, "
      "                               Tensor scale, Tensor? seq_starts) -> ()");
  cache_ops.impl("gather_and_maybe_dequant_cache", torch::kCUDA,
                 &gather_and_maybe_dequant_cache);

  cache_ops.def(
      "cp_gather_cache(Tensor src_cache, Tensor! dst, Tensor block_table, "
      "Tensor cu_seq_lens, int batch_size, Tensor? seq_starts) -> ()");
  cache_ops.impl("cp_gather_cache", torch::kCUDA, &cp_gather_cache);

  cache_ops.def(
      "cp_gather_and_upconvert_fp8_kv_cache(Tensor src_cache, Tensor! dst, "
      "Tensor block_table, Tensor workspace_starts, int batch_size, Tensor? "
      "seq_starts) -> ()");
  cache_ops.impl("cp_gather_and_upconvert_fp8_kv_cache", torch::kCUDA,
                 &cp_gather_and_upconvert_fp8_kv_cache);

  cache_ops.def(
      "indexer_k_quant_and_cache(Tensor k, Tensor! kv_cache, Tensor "
      "slot_mapping, "
      "int quant_block_size, str kv_cache_dtype) -> ()");
  cache_ops.impl("indexer_k_quant_and_cache", torch::kCUDA,
                 &indexer_k_quant_and_cache);

  cache_ops.def(
      "indexer_k_cache(Tensor k, Tensor! kv_cache, Tensor "
      "slot_mapping) -> ()");
  cache_ops.impl("indexer_k_cache", torch::kCUDA, &indexer_k_cache);

   cache_ops.def(
      "cp_gather_indexer_k_cache(Tensor kv_cache, Tensor! dst_k, "
      "Tensor block_table, Tensor cu_seq_lens) -> ()");
  cache_ops.impl("cp_gather_indexer_k_cache", torch::kCUDA,
                 &cp_gather_indexer_k_cache);                

  cache_ops.def(
      "cp_gather_indexer_k_quant_cache(Tensor kv_cache, Tensor! dst_k, Tensor! "
      "dst_scale, Tensor block_table, Tensor cu_seq_lens) -> ()");
  cache_ops.impl("cp_gather_indexer_k_quant_cache", torch::kCUDA,
                 &cp_gather_indexer_k_quant_cache);

  cache_ops.def(
      "concat_mla_q(Tensor ql_nope, Tensor q_pe, Tensor! q_out) -> ()");
  cache_ops.impl("concat_mla_q", torch::kCUDA, &concat_mla_q);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cuda_utils), cuda_utils) {
  // Cuda utils

  // Gets the specified device attribute.
  cuda_utils.def("get_device_attribute(int attribute, int device_id) -> int");
  cuda_utils.impl("get_device_attribute", &get_device_attribute);

  // Gets the maximum shared memory per block device attribute.
  cuda_utils.def(
      "get_max_shared_memory_per_block_device_attribute(int device_id) -> int");
  cuda_utils.impl("get_max_shared_memory_per_block_device_attribute",
                  &get_max_shared_memory_per_block_device_attribute);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
    //在 PyTorch 的 Schema 定义语言中（这与 Python 和 C++ 的函数传参规则完全一致）：一旦某个参数被赋予了默认值，那么它后面的所有参数都必须拥有默认值。不能出现“带默认值的参数”后面紧跟着“不带默认值的参数”
    ops.def(
      "fused_minimax_m3_qknorm_rope_kv_insert("
      "Tensor! qkv, Tensor q_norm_weight, Tensor k_norm_weight, "
      "Tensor cos_sin_cache, Tensor positions, int num_heads, "
      "int num_kv_heads, int rotary_dim, float eps, "
      "Tensor? index_q_norm_weight, Tensor? index_k_norm_weight, "
      "int num_index_heads, "
      "Tensor? slot_mapping, Tensor? index_slot_mapping, "
      "Tensor!? kv_cache, Tensor!? index_cache, "
      "int block_size, Tensor!? q_out, Tensor!? index_q_out, "
      "str kv_cache_dtype, bool skip_index_branch=False, "
      "Tensor!? q_fp8_out=None, float q_fp8_scale=1.0) -> ()");                                                
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
    // torch::stable::Tensor kernels are not classic unboxed at::Tensor fns.
    ops.impl("per_token_group_fp8_quant",
             TORCH_BOX(&per_token_group_quant_fp8));
    ops.impl("per_token_group_fp8_quant_packed",
             TORCH_BOX(&per_token_group_quant_8bit_packed));
    ops.impl("per_token_group_quant_int8",
             TORCH_BOX(&per_token_group_quant_int8));
    ops.impl("fused_minimax_m3_qknorm_rope_kv_insert",
           TORCH_BOX(&fused_minimax_m3_qknorm_rope_kv_insert));
}
REGISTER_EXTENSION(_C_stable_libtorch)