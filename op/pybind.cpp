#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "../include/fused_bias_dropout.h"
#include "../include/fused_rope.h"
#include "../include/fused_bias_swiglu.h"
#include "../include/fused_repeat_kv.h"
#include "../include/fused_gelu.h"
#include "../include/fused_rms_norm_dq.h"
#include "../include/fused_softplus_sqrt.h"
#include "../include/moe_swiglu_dq.h"
#include "../include/moe_softmax_topk.h"
#include "../include/moe_init_routing_v4.h"
#include "../include/all_reduce.h"
#include "../include/moe_gather.h"
#include "../include/rotary_embedding.h"
#include "../include/store_kv.h"
#include "../include/moe_scatter_dynamic_quant.h"
#include "../include/scale_dynamic_quant.h"
#include "../include/rope_train.h"
#include "../include/recv_from_attention_node_post_process.h"
#include "../include/send_to_attention_node_pre_process.h"
#include "../include/int8_quant_kernel.h"
#include "../include/fp8_quant_kernel.h"
#include "../include/fused_add_layernorm_per_token_quant_padding_output.h"
#include "../include/fused_add_gemma_rmsnorm_per_token_quant_padding_output.h"
#include "../include/rms_norm_dynamic_per_token_quant.h"
#include "../include/fused_moe_gate_deepseek.h"
#include "../include/glm_attention_prepare.h"
#include "../include/moe_step4_weighted_topk_gather.h"
#include "../include/router_bias_topk.h"
#include "../include/silu_mul_clamp_dsv4.h"

#ifdef ENABLE_BUILD_GPTQ_MARLIN_OP
    #include "gptq_marlin.h"
#endif
#include "fused_moe_gate_opt.h"
#include "../include/fused_deepseekv4_qkv_rms_norm_rope.h"
#include "../include/fused_split_gemma_rmsnorm_rope.h"
#include "../include/fused_split_gemma_rmsnorm_rope_no_pack.h"
#include "../include/qk_rms_norm.h"
#include "../include/fused_rmsnorm_rope_quant_reshape_and_cache.h"
#include "../include/mhc_pre_big_fuse.h"
#include "../include/mhc_post_cuda.h"
#include "../include/chunk_gated_delta_rule_fwd_h.h"
#include "../include/fused_sigmoid_gating_delta_rule_update.h"
#include "../include/indexer_norm_rope.h"
#include "../include/moe_finalize_routing.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mhc_post_cuda", &mhc_post_cuda,
          py::arg("x"), py::arg("residual"),
          py::arg("post_layer_mix"), py::arg("comb_res_mix"));
    m.def(
        "mhc_pre_big_fuse_out", &mhc_pre_big_fuse_out,
        py::arg("gemm_out_mul"), py::arg("gemm_out_sqrsum"),
        py::arg("hc_scale"), py::arg("hc_base"), py::arg("residual"),
        py::arg("post_mix"), py::arg("comb_mix"), py::arg("layer_input"),
        py::arg("rms_eps"), py::arg("hc_pre_eps"),
        py::arg("hc_sinkhorn_eps"), py::arg("hc_post_mult_value"),
        py::arg("sinkhorn_repeat"), py::arg("n_splits"),
        py::arg("norm_weight") = py::none(),
        py::arg("pre_mix_in") = py::none(),
        py::arg("pre_mix_out") = py::none(),
        py::arg("aux_out") = py::none(), py::arg("norm_eps") = 1e-6,
        py::arg("rms_numel") = -1,
        py::arg("residual_out") = py::none(),
        py::arg("quant_scales") = py::none(),
        py::arg("output_rms") = py::none(),
        py::arg("output_quant_int8") = py::none());
    m.def("fused_bias_dropout", &fused_bias_dropout);
    m.def("fused_rope_fwd", &fused_rope_fwd);
    m.def("fused_rope_bwd", &fused_rope_bwd);
    m.def("fused_bias_swiglu_fwd", &fused_bias_swiglu_fwd);
    m.def("fused_bias_swiglu_bwd", &fused_bias_swiglu_bwd);
    m.def("fused_repeat_kv_fwd", &fused_repeat_kv_fwd);
    m.def("fused_repeat_kv_bwd", &fused_repeat_kv_bwd);
    m.def("fused_gelu_fwd", &fused_gelu_fwd);
    m.def("fused_gelu_bwd", &fused_gelu_bwd);
    m.def("moe_swiglu_dynamic_quantize", &moe_swiglu_dynamic_quantize);
    m.def("moe_softmax_topk", &moe_softmax_topk);
    m.def("moe_init_routing_v4_workspace_size", &moe_init_routing_v4_workspace_size,
          py::arg("x"), py::arg("expert_idx"), py::arg("expert_num"),
          py::arg("quant_mode") = -1);
    m.def("moe_init_routing_v4", &moe_init_routing_v4,
          py::arg("x"), py::arg("expert_idx"), py::arg("expert_num"),
          py::arg("quant_mode") = -1, py::arg("row_idx_type") = 0,
          py::arg("expert_tokens_num_type") = 1,
          py::arg("expert_tokens_num_flag") = true,
          py::arg("active_num") = 0,
          py::arg("active_expert_range") = std::vector<int64_t>{},
          py::arg("scale") = py::none(), py::arg("topk_weight") = py::none(),
          py::arg("active_num_optional") = py::none(),
          py::arg("active_info") = py::none());
    m.def("moe_init_routing_v4_out", &moe_init_routing_v4_out,
          py::arg("x"), py::arg("expert_idx"), py::arg("expanded_x"),
          py::arg("expanded_row_idx"), py::arg("expert_tokens"),
          py::arg("expanded_scale"), py::arg("expanded_topk_weight"),
          py::arg("workspace"), py::arg("expert_num"),
          py::arg("quant_mode") = -1, py::arg("row_idx_type") = 0,
          py::arg("expert_tokens_num_type") = 1,
          py::arg("expert_tokens_num_flag") = true,
          py::arg("active_num") = 0,
          py::arg("active_expert_range") = std::vector<int64_t>{},
          py::arg("scale") = py::none(), py::arg("topk_weight") = py::none(),
          py::arg("active_num_optional") = py::none(),
          py::arg("active_info") = py::none());
    m.def("rms_norm_dynamic_per_token_quant", &rms_norm_dynamic_per_token_quant);
    m.def("head_rms_norm", &head_rms_norm);
    m.def("rms_norm", &rms_norm);
    m.def("qk_rms_norm_inplace_cuda",
          &qk_rms_norm_inplace_cuda,
          py::arg("token_data"),
          py::arg("q_norm_weight"),
          py::arg("k_norm_weight"),
          py::arg("q_head_num"),
          py::arg("kv_head_num"),
          py::arg("qk_head_dim"),
          py::arg("eps"));
    m.def("indexer_norm_rope", &indexer_norm_rope,
          "DeepSeek-V4 Stage-0 indexer Q/K norm + partial NeoX RoPE",
          py::arg("index_q"), py::arg("index_k"), py::arg("index_z"),
          py::arg("q_norm_weight"), py::arg("k_norm_weight"),
          py::arg("k_norm_bias"), py::arg("cos"), py::arg("sin"),
          py::arg("positions"), py::arg("head_dim") = 256,
          py::arg("rotary_dim") = 16, py::arg("eps") = 1e-6,
          py::arg("q_norm_weight_bias") = 1.0);
    m.def("indexer_norm_rope_out", &indexer_norm_rope_out,
          "Preallocated-output DeepSeek-V4 Stage-0 indexer preprocessing",
          py::arg("index_q"), py::arg("index_k"), py::arg("index_z"),
          py::arg("q_out"), py::arg("k_out"),
          py::arg("q_norm_weight"), py::arg("k_norm_weight"),
          py::arg("k_norm_bias"), py::arg("cos"), py::arg("sin"),
          py::arg("positions"), py::arg("head_dim") = 256,
          py::arg("rotary_dim") = 16, py::arg("eps") = 1e-6,
          py::arg("q_norm_weight_bias") = 1.0);
    m.def("indexer_norm_rope_packed_", &indexer_norm_rope_packed_,
          "In-place packed DeepSeek-V4 Stage-0 indexer preprocessing",
          py::arg("index_qkz"), py::arg("q_norm_weight"),
          py::arg("k_norm_weight"), py::arg("k_norm_bias"),
          py::arg("cos"), py::arg("sin"), py::arg("positions"),
          py::arg("num_q_heads"), py::arg("num_k_heads") = 1,
          py::arg("head_dim") = 256, py::arg("rotary_dim") = 16,
          py::arg("eps") = 1e-6, py::arg("q_norm_weight_bias") = 1.0);
    m.def("all_reduce_max", &all_reduce_max);
    m.def("all_reduce_sum", &all_reduce_sum);
    m.def("moe_gather", &moe_gather,
          py::arg("scatter_tokens"), py::arg("scatter_token_id"),
          py::arg("scatter_tokens_weight"), py::arg("convergent_tokens"),
          py::arg("residual_tokens") = py::none(), py::arg("res_scale") = 1.0,
          py::arg("res_token_start") = 0,
          py::arg("scatter_tokens_scale") = py::none());
    m.def("rotary_embedding", &rotary_embedding);
    m.def("store_kv_cache_cuda_interface", &store_kv_cache_cuda_interface);
    m.def("moe_scatter_dynamic_quant", &moe_scatter_dynamic_quant);
    m.def("scale_dynamic_quant", &scale_dynamic_quant);
    m.def("rotary_pos_emb_forward", &rotary_pos_emb_forward);
    m.def("rotary_pos_emb_backward", &rotary_pos_emb_backward);
    m.def("FusedAttentionPrepare", &FusedAttentionPrepare);
    m.def("fused_add_rms_norm_dynamic_per_token_quant_padding_output", &add_rms_norm_dynamic_per_token_quant_padding_output);
    m.def("add_gemma_rms_norm_dynamic_per_token_quant_padding_output", &add_gemma_rms_norm_dynamic_per_token_quant_padding_output);
    m.def("rms_norm_dynamic_per_token_quant_custom", &rms_norm_dynamic_per_token_quant_custom);
    m.def("softplus_sqrt_f16", &softplus_sqrt_cuda, "Fused softplus + sqrt");
    m.def("recv_from_attention_node_post_process", &recv_from_attention_node_post_process);
    m.def("send_to_attention_node_pre_process", &send_to_attention_node_pre_process);
    m.def("step4_weighted_topk_gather",&step4_weighted_topk_gather);
    m.def(
        "fused_silu_mul_dq_mask_quant",
        &fused_silu_mul_dq_mask_quant_pack,
        py::arg("out"),
        py::arg("input"),
        py::arg("mask"),
        py::arg("swiglu_limit") = 0.0f,
        py::arg("weight") = py::none(),
        py::arg("gemm1_alpha") = 1.0f,
        py::arg("gemm1_limit") = 0.0f
        );
    m.def(
        "fused_situ_mul_dq_mask_quant_pack",
        &fused_situ_mul_dq_mask_quant_pack,
        py::arg("out"),
        py::arg("input"),
        py::arg("mask"),
        py::arg("beta") = 1.0f,
        py::arg("linear_beta") = 1.0f,
        py::arg("has_linear_beta") = 0
    );
    m.def("per_token_quant_int8_pack", 
        &per_token_quant_int8_pack,
        py::arg("out"),
        py::arg("input")
    );
    
    m.def(
        "silu_mul_mask",
        &silu_mul_mask_interface,
        py::arg("out"),
        py::arg("input"),
        py::arg("mask"),
        py::arg("swiglu_limit") = 0.0f
    );
    m.def("fused_silu_mul_dq_mask_fp8_quant", &fused_silu_mul_dq_mask_quant_fp8_pack);

    m.def("fused_silu_mul_dq_reorder_quant", &fused_silu_mul_dq_quant_reordered_topk_interface);
    m.def("fused_silu_mul_dq_quant", &fused_silu_mul_dq_quant_interface);    
    m.def("fused_silu_mul_dq_quant", &fused_silu_mul_dq_quant_smooth_interface,
          py::arg("out"), py::arg("scale"), py::arg("input"),
          py::arg("per_token_scale"));
    
    m.def(
        "fused_silu_mul_dq_mask_quant_fp8_nopack",
        &fused_silu_mul_dq_mask_quant_fp8_nopack,
        py::arg("output"),
        py::arg("output_scale"),
        py::arg("input"),
        py::arg("mask"),
        py::arg("quant_group"),
        py::arg("swiglu_limit"),
        py::arg("isTranspose") = py::none()
    );

    m.def(
        "router_bias_topk",
        &router_bias_topk,
        py::arg("gating_output"),
        py::arg("router_bias"),
        py::arg("topk_weights"),
        py::arg("topk_ids"),
        py::arg("topk"),
        py::arg("renormalize"),
        py::arg("check_nan"),
        py::arg("routed_scaling_factor"),
        py::arg("nan_row_i_out")
    );
    m.def(
        "silu_and_mul_clamp",
        &silu_and_mul_clamp,
        py::arg("input"),
        py::arg("output"),
        py::arg("swiglu_limit")
    );
    py::object torch_bfloat16 = py::module::import("torch").attr("bfloat16");

#ifdef ENABLE_BUILD_GPTQ_MARLIN_OP
    m.def("gptq_marlin_gemm_legacy", &gptq_marlin_gemm_legacy,
          "Function to perform GEMM using Marlin quantization.", py::arg("a"),
          py::arg("b_q_weight"), py::arg("b_scales"), py::arg("g_idx"),
          py::arg("perm"), py::arg("workspace"), py::arg("num_bits"),
          py::arg("size_m"), py::arg("size_n"), py::arg("size_k"),
          py::arg("is_k_full"), py::arg("dtype") = torch_bfloat16,
          py::arg("use_atomic_cache") = true);
    m.def("gptq_marlin_gemm", &gptq_marlin_gemm, "Function to perform GEMM using Marlin quantization.", 
        py::arg("a"), py::arg("b_q_weight"), py::arg("b_scales"), py::arg("g_idx"),
        py::arg("perm"), py::arg("workspace"), py::arg("num_bits"), py::arg("size_m_tensor"),
        py::arg("size_m"), py::arg("size_n"), py::arg("size_k"),
        py::arg("sms"), py::arg("is_k_full"), py::arg("dtype") = torch_bfloat16,
        py::arg("use_atomic_cache") = true);
#endif 
    m.def("fused_moe_gate_deepseek", &fused_moe_gate_deepseek, "Fused moe gate topk selection",
        py::arg("gating_outputs"),
        py::arg("correction_bias"),
        py::arg("out_routing_weights"),
        py::arg("out_selected_experts"),
        py::arg("topk"),
        py::arg("renormalize"),
        py::arg("num_expert_group"),
        py::arg("topk_group"),
        py::arg("num_fused_shared_experts"),
        py::arg("scale_factor"),
        py::arg("moegate_type").none(true)
    );

    m.def("fused_moe_gate_opt", &fused_moe_gate_opt, "Fused MoE Gate optimized kernel",
        py::arg("gating_outputs"),
        py::arg("correction_bias"),
        py::arg("out_routing_weights"),
        py::arg("out_selected_experts"),
        py::arg("topk"),
        py::arg("renormalize"),
        py::arg("num_expert_group"),
        py::arg("topk_group"),
        py::arg("num_fused_shared_experts") = py::none(),  // 设置默认值为 None
        py::arg("routed_scaling_factor") = py::none()      // 设置默认值为 None
    );

    m.def("fused_rms_norm_rope", &fused_rms_norm_rope, "Fused RMS Norm + RoPE for DeepSeekV4 (in-place)",
        py::arg("q"),
        py::arg("kv"),
        py::arg("positions"),
        py::arg("freqs_cis"),
        py::arg("qk_rope_head_dim") = 64,
        py::arg("eps") = 1e-6,
        py::arg("weight_q") = py::none(),
        py::arg("weight_kv") = py::none()
    );
	
    m.def("gemma_fused_rmsnorm_rope", &gemma_fused_rmsnorm_rope, "Gemma Fused RMSNorm and Neox RoPE Kernel");

    m.def("gemma_fused_rmsnorm_rope_no_pack", &gemma_fused_rmsnorm_rope_no_pack, "Gemma Fused RMSNorm and Neox RoPE Kernel_no_pack",
        py::arg("qkv"),
        py::arg("q_weight"),
        py::arg("k_weight"),
        py::arg("positions"),
        py::arg("q_size"),
        py::arg("kv_size"),
        py::arg("head_dim"),
        py::arg("eps"),
        py::arg("cos_sin_cache")
    );

    m.def("fused_rmsnorm_rope_quant_reshape_and_cache",
      &fused_rmsnorm_rope_quant_reshape_and_cache,
      "Fused RMSNorm + RoPE + Quant + Reshape&Cache (in-place: packed_qkv / k_cache / v_cache)");

    m.def("chunk_gated_delta_rule_fwd_h",
      &chunk_gated_delta_rule_fwd_h,
      "CUDA chunk_gated_delta_rule_fwd_h; same public signature as the Triton entry",
      py::arg("k"), py::arg("w"), py::arg("u"),
      py::arg("g") = py::none(), py::arg("gk") = py::none(),
      py::arg("initial_state") = py::none(),
      py::arg("initial_state_indices") = py::none(),
      py::arg("save_new_value") = true,
      py::arg("cu_seqlens") = py::none(),
      py::arg("chunk_indices") = py::none(),
      py::arg("use_exp2") = false,
      py::kw_only(),
      py::arg("_block_v") = py::none(),
      py::arg("_num_warps") = py::none(),
      py::arg("_num_stages") = 1);

    m.def("chunk_gated_delta_rule_fwd_h_native_out",
      &chunk_gated_delta_rule_fwd_h_native_out,
      "Retained CUDA chunk_gated_delta_rule_fwd_h (optimized K=V=128 plus "
      "generic K<=256 path, BT=64, preallocated h and optional v_new; "
      "peer of the Triton entry in mcoplib.triton_sglang_chunk_delta_h)",
      py::arg("k"), py::arg("w"), py::arg("u"), py::arg("gk"),
      py::arg("initial_state"), py::arg("initial_state_indices"),
      py::arg("cu_seqlens"), py::arg("chunk_offsets"),
      py::arg("h"), py::arg("v_new"), py::arg("sequences"),
      py::arg("chunks_per_sequence"), py::arg("use_exp2") = false,
      py::arg("ragged") = false, py::arg("state_bf16") = false,
      py::arg("g") = py::none());

    m.def("chunk_gated_delta_rule_fwd_h_coverage",
      &chunk_gated_delta_rule_fwd_h_coverage,
      "Mirror of the production Triton host-side dispatch: returns "
      "'<branch> <cuda_serves> <cuda_executes>'",
      py::arg("B"), py::arg("T"), py::arg("Hg"), py::arg("K"), py::arg("H"),
      py::arg("V"), py::arg("block_v"), py::arg("num_warps"),
      py::arg("num_stages"), py::arg("N"), py::arg("NT"), py::arg("has_g"),
      py::arg("has_gk"), py::arg("has_v_new"), py::arg("has_cu_seqlens"),
      py::arg("tail_free"));

    m.def(
      "fused_sigmoid_gating_delta_rule_update_native_out",
      &fused_sigmoid_gating_delta_rule_update_native_out,
      "Preallocated CUDA native GLM5.3-flash KDA target-verify kernel",
      py::arg("A_log"), py::arg("a"), py::arg("dt_bias"),
      py::arg("softplus_beta"), py::arg("softplus_threshold"),
      py::arg("lower_bound"), py::arg("q"), py::arg("k"), py::arg("v"),
      py::arg("b"), py::arg("o"), py::arg("h0_source"),
      py::arg("h0_indices"), py::arg("stride_h0_source"),
      py::arg("cu_seqlens"), py::arg("intermediate_states_buffer"),
      py::arg("intermediate_state_indices"), py::arg("cache_steps"),
      py::arg("retrieve_parent_token_ptr"),
      py::arg("stride_retrieve_parent_token_seq"),
      py::arg("stride_retrieve_parent_token_token"), py::arg("scale"),
      py::arg("T"), py::arg("stride_a"), py::arg("stride_q"),
      py::arg("stride_k"), py::arg("stride_v"), py::arg("stride_b"),
      py::arg("NP2_T"), py::arg("B"), py::arg("H"), py::arg("HV"),
      py::arg("K"), py::arg("V"), py::arg("BK"), py::arg("BV"),
      py::arg("USE_INITIAL_STATE"), py::arg("USE_QK_L2NORM_IN_KERNEL"),
      py::arg("IS_VARLEN"), py::arg("IS_KDA"),
      py::arg("USE_LOWER_BOUND"), py::arg("DISABLE_STATE_UPDATE"),
      py::arg("CACHE_INTERMEDIATE_STATES"),
      py::arg("HAS_EAGLE_TREE_CUSTOM_ATTN_MASK"),
      py::arg("replayssm_rawv"), py::arg("replayssm_rawk"),
      py::arg("replayssm_g"), py::arg("replayssm_beta"),
      py::arg("stride_rawv_slot"), py::arg("stride_rawk_slot"),
      py::arg("stride_g_slot"), py::arg("stride_beta_slot"),
      py::arg("MAX_CACHE_LEN"), py::arg("CACHE_RING"),
      py::arg("SPLIT_N_HV_GRID") = true, py::arg("workspace") = py::none(),
      py::arg("value_block") = 0);

    m.def(
      "fused_sigmoid_gating_delta_rule_update_native_precompute",
      &fused_sigmoid_gating_delta_rule_update_native_precompute,
      "Diagnostic: materialize only the native precompute workspace",
      py::arg("A_log"), py::arg("a"), py::arg("dt_bias"), py::arg("b"),
      py::arg("q"), py::arg("k"), py::arg("workspace"),
      py::arg("lower_bound"), py::arg("stride_a"), py::arg("stride_b"),
      py::arg("stride_q"), py::arg("stride_k"), py::arg("T"));

    m.def(
      "fused_sigmoid_gating_delta_rule_update_native_recurrent",
      &fused_sigmoid_gating_delta_rule_update_native_recurrent,
      "Diagnostic: run only the native recurrent launch on a ready workspace",
      py::arg("workspace"), py::arg("q"), py::arg("k"), py::arg("v"),
      py::arg("o"), py::arg("h0_source"), py::arg("h0_indices"),
      py::arg("cu_seqlens"), py::arg("intermediate_states_buffer"),
      py::arg("intermediate_state_indices"), py::arg("stride_h0_source"),
      py::arg("stride_q"), py::arg("stride_k"), py::arg("stride_v"),
      py::arg("scale"), py::arg("T"), py::arg("value_block"),
      py::arg("variant") = 0);

    m.def(
      "fused_sigmoid_gating_delta_rule_update",
      &fused_sigmoid_gating_delta_rule_update_native,
      "Allocating CUDA native peer of the Triton fused sigmoid gating update",
      py::arg("A_log"), py::arg("a"), py::arg("dt_bias"),
      py::arg("softplus_beta"), py::arg("softplus_threshold"),
      py::arg("q"), py::arg("k"), py::arg("v"), py::arg("b"),
      py::arg("initial_state_source"), py::arg("initial_state_indices"),
      py::arg("scale") = py::none(),
      py::arg("use_qk_l2norm_in_kernel") = false,
      py::arg("cu_seqlens") = py::none(), py::arg("is_kda") = false,
      py::arg("lower_bound") = py::none(),
      py::arg("disable_state_update") = false,
      py::arg("intermediate_states_buffer") = py::none(),
      py::arg("intermediate_state_indices") = py::none(),
      py::arg("cache_steps") = py::none(),
      py::arg("retrieve_parent_token") = py::none(),
      py::arg("cache_ring") = false, py::arg("replayssm_rawv") = py::none(),
      py::arg("replayssm_rawk") = py::none(),
      py::arg("replayssm_g") = py::none(),
      py::arg("replayssm_beta") = py::none());

     m.def(
        "moe_finalize_routing",
        &moe_finalize_routing,
        py::arg("out"),
        py::arg("expanded_x"),
        py::arg("expanded_row_idx"),
        py::arg("x1") = py::none(),
        py::arg("x2") = py::none(),
        py::arg("bias") = py::none(),
        py::arg("scales") = py::none(),
        py::arg("expert_idx") = py::none(),
        py::arg("drop_pad_mode") = 0,
        py::arg("k") = 1);
}
