#pragma once

#include <c10/util/BFloat16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Internal host entry point: keep the new device kernel out of the retained
// routing translation unit. The caller establishes the full-tile dispatch.
namespace mcoplib::moe_routing_detail {
void launch_mx4_quantize_scatter(
    const c10::BFloat16* x, void* data, uint8_t* scales, const int* inverse,
    int64_t tokens, int topk, int hidden, int capacity, cudaStream_t stream);
}  // namespace mcoplib::moe_routing_detail
