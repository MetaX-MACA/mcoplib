#include <ATen/ATen.h>
#include <c10/util/Optional.h>
void moe_finalize_routing(
    at::Tensor& out,
    const at::Tensor& expanded_x,
    const at::Tensor& expanded_row_idx,
    const c10::optional<at::Tensor>& x1 = c10::nullopt,
    const c10::optional<at::Tensor>& x2 = c10::nullopt,
    const c10::optional<at::Tensor>& bias = c10::nullopt,
    const c10::optional<at::Tensor>& scales = c10::nullopt,
    const c10::optional<at::Tensor>& expert_idx = c10::nullopt,
    int64_t drop_pad_mode = 0,
    int64_t k = 1);