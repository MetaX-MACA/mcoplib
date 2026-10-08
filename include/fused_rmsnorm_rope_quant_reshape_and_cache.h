#include <ATen/ATen.h>

void fused_rmsnorm_rope_quant_reshape_and_cache(
    torch::Tensor& packed_qkv,
    torch::Tensor const& q_norm_weight,
    torch::Tensor const& k_norm_weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    torch::Tensor const& q_lens,
    torch::Tensor const& cache_lens,
    torch::Tensor const& accum_q_lens,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor const& slot_mapping,
    std::optional<at::Tensor> k_scale,
    std::optional<at::Tensor> v_scale,
    const std::string& kv_cache_dtype,
    int64_t rope_offset, int64_t block_size, double eps);