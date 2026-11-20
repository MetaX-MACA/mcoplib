#pragma once

#include <torch/extension.h>
#include <optional>
#include <vector>

// Outputs: expanded_x, expanded_row_idx, expert_counts_or_cumsum,
// expanded_scale, expanded_topk_weight. See experiments/moe_init_routing_v4/API.md.
int64_t moe_init_routing_v4_workspace_size(
    const at::Tensor& x, const at::Tensor& expert_idx,
    int64_t expert_num, int64_t quant_mode);

std::vector<at::Tensor> moe_init_routing_v4(
    const at::Tensor& x, const at::Tensor& expert_idx,
    int64_t expert_num, int64_t quant_mode, int64_t row_idx_type,
    int64_t expert_tokens_num_type, bool expert_tokens_num_flag,
    int64_t active_num, const std::vector<int64_t>& active_expert_range,
    const std::optional<at::Tensor>& scale,
    const std::optional<at::Tensor>& topk_weight,
    const std::optional<at::Tensor>& active_num_optional = std::nullopt,
    const std::optional<at::Tensor>& active_info = std::nullopt);

void moe_init_routing_v4_out(
    const at::Tensor& x, const at::Tensor& expert_idx,
    at::Tensor expanded_x, at::Tensor expanded_row_idx,
    at::Tensor expert_tokens, at::Tensor expanded_scale,
    at::Tensor expanded_topk_weight, at::Tensor workspace,
    int64_t expert_num, int64_t quant_mode, int64_t row_idx_type,
    int64_t expert_tokens_num_type, bool expert_tokens_num_flag,
    int64_t active_num, const std::vector<int64_t>& active_expert_range,
    const std::optional<at::Tensor>& scale,
    const std::optional<at::Tensor>& topk_weight,
    const std::optional<at::Tensor>& active_num_optional = std::nullopt,
    const std::optional<at::Tensor>& active_info = std::nullopt);

// active_num_optional: device INT64 scalar or [1]. The allocating API reads it
// outside capture to determine the exact output shape. The out API reads it on
// device on every replay; expanded_x.size(0) is fixed storage capacity.
// Tensor out calls require device INT64 active_info[4]:
// [logical_rows, effective_rows, available_rows, status].
// status: 0=OK, 1=negative activeNum, 2=logical_rows exceeds storage capacity.
// On error, effective_rows=0 and data/scale/weight storage is untouched.
