#include <ATen/ATen.h>
#include <c10/util/Optional.h>

void router_bias_topk( 
    at::Tensor gating_output,
    at::Tensor router_bias,
    at::Tensor topk_weights,
    at::Tensor topk_ids,
    const int topk,
    const bool renormalize,
    const bool check_nan,
    const float routed_scaling_factor,
    const int nan_row_i_out
    );