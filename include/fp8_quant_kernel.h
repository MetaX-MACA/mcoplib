#include <ATen/ATen.h>
void fused_silu_mul_dq_mask_quant_fp8_nopack(
    torch::Tensor& output,
    torch::Tensor& output_scale,
    torch::Tensor const& input,
    torch::Tensor const& mask,
    int quant_group,
    std::optional<float> swiglu_limit,
    std::optional<int> isTranspose
);
