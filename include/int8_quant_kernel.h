#include <ATen/ATen.h>
void fused_silu_mul_dq_mask_quant_pack(
    at::Tensor& out,          
    at::Tensor const& input, 
    at::Tensor const &mask);

void fused_silu_mul_dq_mask_quant_fp8_pack(
    torch::Tensor& out,          
    torch::Tensor const& input, 
    torch::Tensor const &mask);

void fused_silu_mul_dq_quant_reordered_topk_interface(
    at::Tensor& out,
    at::Tensor& scale,   
    at::Tensor const& input,
    at::Tensor const& reorder_topk_ids,
    at::Tensor const& w2_scale,
    int64_t start_expert_id,
    int64_t end_expert_id);