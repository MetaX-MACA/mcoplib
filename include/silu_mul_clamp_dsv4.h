#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>

void silu_and_mul_clamp(const torch::Tensor& input,
                        torch::Tensor& output, float swiglu_limit);