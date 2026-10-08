#pragma once

#include <torch/extension.h>

// x: BF16 [N, H], residual: BF16 [N, HC, H].
// post_layer_mix: FP32 [N, HC] or [N, HC, 1].
// comb_res_mix: FP32 [N, HC, HC], indexed [token, input, output].
// All tensors must be contiguous and on the same CUDA/MACA device.
torch::Tensor mhc_post_cuda(torch::Tensor x, torch::Tensor residual,
    torch::Tensor post_layer_mix, torch::Tensor comb_res_mix);
