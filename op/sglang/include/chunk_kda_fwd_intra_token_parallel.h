#pragma once
#include <torch/extension.h>

void chunk_kda_fwd_intra_token_parallel(torch::Tensor q, torch::Tensor k, torch::Tensor g,
                     torch::Tensor beta, torch::Tensor aqk,
                     torch::Tensor akk, double scale);
