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
#ifdef ENABLE_BUILD_GPTQ_MARLIN_OP
    #include "gptq_marlin.h"
#endif
#include "fused_moe_gate_opt.h"
#include "../include/fused_deepseekv4_qkv_rms_norm_rope.h"
#include "../include/fused_split_gemma_rmsnorm_rope.h"
#include "../include/fused_split_gemma_rmsnorm_rope_no_pack.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
    m.def("rms_norm_dynamic_per_token_quant", &rms_norm_dynamic_per_token_quant);
    m.def("head_rms_norm", &head_rms_norm);
    m.def("rms_norm", &rms_norm);
    m.def("all_reduce_max", &all_reduce_max);
    m.def("all_reduce_sum", &all_reduce_sum);
    m.def("moe_gather", &moe_gather);
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
    m.def(
        "fused_silu_mul_dq_mask_quant",
        &fused_silu_mul_dq_mask_quant_pack,
        py::arg("out"),
        py::arg("input"),
        py::arg("mask"),
        py::arg("swiglu_limit") = 0.0f,
        py::arg("weight") = py::none()
        );
    m.def("fused_silu_mul_dq_mask_fp8_quant", &fused_silu_mul_dq_mask_quant_fp8_pack);
    m.def("fused_silu_mul_dq_reorder_quant", &fused_silu_mul_dq_quant_reordered_topk_interface);
    m.def("fused_silu_mul_dq_quant", &fused_silu_mul_dq_quant_interface);
    
    m.def("fused_silu_mul_dq_mask_quant_fp8_nopack", &fused_silu_mul_dq_mask_quant_fp8_nopack);

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
}
