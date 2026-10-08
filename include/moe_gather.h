#include <ATen/ATen.h>

void moe_gather(at::Tensor scatter_tokens,
                at::Tensor scatter_tokens_offset,
                at::Tensor scatter_tokens_weight,
                at::Tensor convergent_tokens,
                c10::optional<at::Tensor> residual_tokens = c10::nullopt,
                double res_scale = 1.0,
                int64_t res_token_start = 0);
