#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <tuple>
#include <vector>

int64_t fused_moe_gate_opt(
    at::Tensor& gating_outputs, 
    at::Tensor& correction_bias, 
    at::Tensor& out_routing_weights, 
    at::Tensor& out_selected_experts, 
    int64_t topk, 
    bool renormalize, 
    int64_t num_expert_groupm, 
    int64_t topk_group, 
    std::optional<int64_t> num_fused_shared_experts, 
    std::optional<double> routed_scaling_factor);