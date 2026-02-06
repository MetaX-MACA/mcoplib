// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "fused_moe_gate.cuh"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

int64_t fused_moe_gate_opt(
    torch::Tensor& gating_outputs, //[bs, num_experts], dtype=bf16
    torch::Tensor& correction_bias, //[num_experts], dtype=bf16
    torch::Tensor& out_routing_weights, //[bs, num_selected_experts], dtype=float
    torch::Tensor& out_selected_experts, //[bs, num_selected_experts], dtype=int32
    int64_t topk,
    bool renormalize,
    int64_t num_expert_group,
    int64_t topk_group,
    std::optional<int64_t> num_fused_shared_experts,
    std::optional<double>  routed_scaling_factor
) {
  DEBUG_TRACE_PARAMS(gating_outputs, correction_bias, out_routing_weights, out_selected_experts, topk, renormalize, num_expert_group, topk_group, num_fused_shared_experts, routed_scaling_factor);
  DEBUG_DUMP_PARAMS(gating_outputs, correction_bias, out_routing_weights, out_selected_experts, topk, renormalize, num_expert_group, topk_group, num_fused_shared_experts, routed_scaling_factor);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_outputs));
    TORCH_CHECK(((topk == 8) || (topk == 9)), "Expected topk = 8, but get topk = ", topk);
    int dev = gating_outputs.get_device();
    int grid = gating_outputs.size(0);
    int num_experts = gating_outputs.size(1);
    double scale_factor = 1.0f;
    if (routed_scaling_factor.has_value())
        scale_factor = 1.0f / (*routed_scaling_factor);
    int *d_shared_experts_ids = nullptr;
    int num_shared_experts = (num_fused_shared_experts.has_value() ? *num_fused_shared_experts : 0);
    if(num_shared_experts > 0)
        TORCH_CHECK(((topk >= 9)), "Expected topk >= 9 when fused_num_shared_experts > 0, but get topk = ", topk);
    if(num_shared_experts == 0)
        TORCH_CHECK(((topk == 8)), "Expected topk == 8 when fused_num_shared_experts = 0 or None, but get topk = ", topk);
    if (num_shared_experts > 1)
    {
        cudaMalloc((void**)&d_shared_experts_ids, 512 * sizeof(int));
        int shared_experts_ids[512];
        for(int i = 0; i < 512; i++){
            float tmp = ((rand() & 0xff) + 0.0f) / 256.0f;
            shared_experts_ids[i] = num_experts + num_shared_experts * tmp;
        }
        cudaMemcpy(d_shared_experts_ids, shared_experts_ids, 512 * sizeof(int), cudaMemcpyHostToDevice);
    }

#define LAUNCH_MOE_GATE(NUM_SHARED_EXPERTS, NUM_EXPERTS, NUM_EXPERT_GROUP, TOPK_GROUP, TOPK) \
    else if (num_shared_experts == NUM_SHARED_EXPERTS && num_expert_group == NUM_EXPERT_GROUP && topk_group == TOPK_GROUP && num_experts == NUM_EXPERTS) { \
        int block = NUM_EXPERTS;                                                                                                            \
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, gating_outputs.scalar_type(), "moe_gate fused_topk", [&]{                 \
            fused_mla::fused_topk<scalar_t, NUM_SHARED_EXPERTS, NUM_EXPERTS, NUM_EXPERT_GROUP, TOPK_GROUP, TOPK><<<grid, block, 0, at::cuda::getCurrentCUDAStream(dev)>>>(\
                (const scalar_t*)gating_outputs.data_ptr<scalar_t>(),   \
                (const scalar_t*)correction_bias.data_ptr<scalar_t>(),  \
                (float*)out_routing_weights.data_ptr(),                 \
                (int32_t*)out_selected_experts.data_ptr(),              \
                renormalize,                                            \
                scale_factor,                                           \
                d_shared_experts_ids);                                  \
        });                                                             \
    }

    if (false) {
    }
    LAUNCH_MOE_GATE(0, 256, 8, 4, 8)
    LAUNCH_MOE_GATE(0, 320, 1, 1, 8)
    LAUNCH_MOE_GATE(0, 384, 1, 1, 8)
    LAUNCH_MOE_GATE(0, 448, 1, 1, 8)
    LAUNCH_MOE_GATE(1, 256, 8, 4, 9)
    LAUNCH_MOE_GATE(1, 320, 1, 1, 9)
    LAUNCH_MOE_GATE(1, 384, 1, 1, 9)
    LAUNCH_MOE_GATE(1, 448, 1, 1, 9)
    LAUNCH_MOE_GATE(2, 256, 8, 4, 9)
    LAUNCH_MOE_GATE(2, 320, 1, 1, 9)
    LAUNCH_MOE_GATE(2, 384, 1, 1, 9)
    LAUNCH_MOE_GATE(2, 448, 1, 1, 9)
    else {
        if(d_shared_experts_ids != nullptr)
            cudaFree(d_shared_experts_ids);
        TORCH_CHECK(false, "Invalid arguments with TOPK = ", topk, ", NUM_EXPERT_GROUP = ", num_expert_group, ", TOPK_GROUP = ", topk_group, ", NUM_EXPERTS = ", num_experts);
        return 1;
    }
    if(d_shared_experts_ids != nullptr)
        cudaFree(d_shared_experts_ids);

    return 0;
}
