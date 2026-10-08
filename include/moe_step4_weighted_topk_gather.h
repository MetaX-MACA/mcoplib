#include <ATen/ATen.h>

void step4_weighted_topk_gather(
    at::Tensor input,               // [T, K, H]
    at::Tensor router_weight,       // [T, K]
    at::Tensor output               // [T, H]
);