// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include "utils.h"
#include "mctlass/mctlass_ex.h"
#include "mctlass/half.h"
#include "mctlass/layout/matrix.h"
#include "mctlass/epilogue/thread/scale_type.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

int64_t mctlass_moe_w4a16_gemm_kernel_mnk(int64_t num_valid_tokens, int64_t N, int64_t K, int64_t group)
{
    DEBUG_TRACE_PARAMS(num_valid_tokens, N, K, group);
    DEBUG_DUMP_PARAMS(num_valid_tokens, N, K, group);
    using ElementA = maca_bfloat16;
    using ElementB = uint8_t;
    using ElementC = maca_bfloat16;
    using ElementCompute = float;
    using LayoutA = mctlass::layout::RowMajor;
    using LayoutB = mctlass::layout::ColumnMajor;
    using LayoutC = mctlass::layout::RowMajor;

    using ArchTag = mctlass::arch::Sm80;

    using mctlassMoeGemmOp = mctlassMoeGemm<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementCompute,
        ArchTag
    >;
    mctlassMoeGemmOp mctlass_op;
    mctlass::gemm::BatchedGemmCoord problem_size(num_valid_tokens, N, K, group);
    mctlass::gemm::GemmCoord kernel_size;
    int pack_factor = 2;
    int group_size = 64;
    mctlass::Status status = mctlass_op.gemm_kernel_mnk(problem_size, kernel_size, pack_factor, group_size);
    if (status != mctlass::Status::kSuccess) {
        printf("Error: Not find supported kernel!\n");
    }
    return static_cast<int64_t>(kernel_size.m());
}

void mctlass_fused_moe_kernel_w4a16(
    torch::Tensor const& a, torch::Tensor const& b, torch::Tensor& c, torch::Tensor const& b_scales, torch::Tensor const& zp_b,
    torch::Tensor const& moe_weight, torch::Tensor const& token_ids, torch::Tensor const& expert_ids, 
    torch::Tensor const& num_tokens_post_padded, int64_t N, int64_t K, int64_t EM, int64_t num_valid_tokens, int64_t topk, bool mul_routed_weight)
{
    DEBUG_TRACE_PARAMS(a, b, c, b_scales, zp_b, moe_weight, token_ids, expert_ids, num_tokens_post_padded, N, K, EM, num_valid_tokens, topk, mul_routed_weight);
    DEBUG_DUMP_PARAMS(a, b, c, b_scales, zp_b, moe_weight, token_ids, expert_ids, num_tokens_post_padded, N, K, EM, num_valid_tokens, topk, mul_routed_weight);
    using ElementA = maca_bfloat16;
    using ElementB = uint8_t;
    using ElementC = maca_bfloat16;
    using ElementCompute = float;
    using LayoutA = mctlass::layout::RowMajor;
    using LayoutB = mctlass::layout::ColumnMajor;
    using LayoutC = mctlass::layout::RowMajor;

    using ArchTag = mctlass::arch::Sm80;

    using mctlassMoeGemmOp = mctlassMoeGemm<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementCompute,
        ArchTag
    >;
    int64_t num_experts = b.size(0);

    auto a_ptr = static_cast<ElementA const*>(a.data_ptr());
    auto b_ptr = static_cast<ElementB const*>(b.data_ptr());
    auto c_ptr = static_cast<ElementC*>(c.data_ptr());
    auto scale_b_ptr = static_cast<ElementA*>(b_scales.data_ptr());
    auto zp_b_ptr = static_cast<ElementB*>(zp_b.data_ptr());
    auto moe_weight_ptr = moe_weight.data_ptr<float>();
    auto token_ids_ptr = token_ids.data_ptr<int32_t>();
    auto expert_ids_ptr = expert_ids.data_ptr<int32_t>();
    auto num_tokens_post_padded_ptr = num_tokens_post_padded.data_ptr<int32_t>();

    mctlassMoeGemmOp mctlass_op;
    mctlass::gemm::BatchedGemmCoord problem_size(num_valid_tokens, N, K, num_experts);

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    int pack_factor = 2;
    int group_size = 64;
    // mctlass
    typename mctlassMoeGemmOp::Arguments arguments{
        mctlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {moe_weight_ptr},
        a_ptr,
        b_ptr,
        c_ptr,
        {token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        static_cast<int32_t>(EM),
        static_cast<int32_t>(topk),
        mul_routed_weight,
        pack_factor,
        group_size,
        scale_b_ptr,
        zp_b_ptr}};
    
    mctlass::Status is_success = mctlass_op(arguments, NULL, stream);
    assert(mctlass::Status::kSuccess == is_success);
}