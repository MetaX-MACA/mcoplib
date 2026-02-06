// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include "../kernel/moe_softmax_topk.cuh"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

// Constructs some constants needed to partition the work across threads at compile time.
template <typename scalar_t, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(scalar_t);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");
    static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};

template <typename scalar_t, int EXPERTS, int WARPS_PER_TB>
void topkGatingSoftmaxLauncherHelper(const scalar_t* input, const bool* finished, scalar_t* output, int* indices, 
    const int num_rows, const int k, const int start_expert, const int end_expert, cudaStream_t stream)
{
    static constexpr int MAX_BYTES_PER_LDG = 8;

    static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(scalar_t) * EXPERTS);
    using Constants = TopkConstants<scalar_t, EXPERTS, BYTES_PER_LDG>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
    mc_moe_softmax_topk::topkGatingSoftmax<scalar_t, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<num_blocks, block_dim, sizeof(scalar_t) * k * WARPS_PER_TB, stream>>>(
        input, finished, output, num_rows, indices, k, start_expert, end_expert);
}

template <typename scalar_t, int EXPERTS, int WARPS_PER_TB>
void topkDecodeGatingSoftmaxLauncherHelper(const scalar_t* input, const bool* finished, scalar_t* output, int* indices,
    const int num_rows, const int k, const int start_expert, const int end_expert, cudaStream_t stream) {
    if (k == 8 and num_rows < 1024){
        constexpr int WAVE_SIZE = 64;
        constexpr int WAVES_PER_ROW = 2;
        dim3 block_dim(WAVE_SIZE * WAVES_PER_ROW, WARPS_PER_TB / WAVES_PER_ROW);
        mc_moe_softmax_topk::topkGatingSoftmaxDecodeOpttt<scalar_t, int64_t, EXPERTS, WARPS_PER_TB, 8, WAVE_SIZE, WAVES_PER_ROW><<<num_rows, block_dim, 0, stream>>>(input, output, num_rows, indices);
        
    } else {
        static constexpr std::size_t MAX_BYTES_PER_LDG = 16;

        static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(float) * EXPERTS);
        using Constants = TopkConstants<scalar_t, EXPERTS, BYTES_PER_LDG>;
        static constexpr int VPT = Constants::VPT;
        static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
        const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
        const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

        dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
        mc_moe_softmax_topk::topkGatingSoftmax<scalar_t, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<num_blocks, block_dim, 0, stream>>>(
            input, finished, output, num_rows, indices, k, start_expert, end_expert);
    }
}

#define LAUNCH_SOFTMAX(NUM_EXPERTS, WARPS_PER_TB)                       \
    topkGatingSoftmaxLauncherHelper<scalar_t, NUM_EXPERTS, WARPS_PER_TB>(         \
        gating_output, nullptr, topk_weights, topk_indicies,            \
        num_tokens, topk, 0, num_experts,         \
        stream);

#define LAUNCH_SOFTMAX_OPT(NUM_EXPERTS, WARPS_PER_TB)                   \
    topkDecodeGatingSoftmaxLauncherHelper<scalar_t, NUM_EXPERTS, WARPS_PER_TB>(  \
        gating_output, nullptr, topk_weights, topk_indicies,            \
        num_tokens, topk, 0, num_experts,         \
        stream);

#define LAUNCH_SELECT_TOPK_SOFTMAX(power2SortSize)      \
    mc_moe_softmax_topk::selectTopKSoftmax<scalar_t, int, BLOCK_SIZE, power2SortSize><<<num_tokens, BLOCK_SIZE, smem_size, stream>>>(    \
                gating_output, topk_weights, topk_indicies,         \
                topk, num_experts, num_experts, topk, topk);        

template<typename scalar_t>
void topkGatingSoftmaxKernelLauncher(
    const scalar_t* gating_output,
    scalar_t* topk_weights,
    int* topk_indicies,
    scalar_t* softmax_workspace,
    const int num_tokens,
    const int num_experts,
    const int topk,
    const bool pre_softmax,
    cudaStream_t stream) {

    if (num_experts >= 1024) {
        const int sortBlockSize = getSortSize(topk);
        static constexpr int BLOCK_SIZE = 512;
        const int smem_size = (sizeof(scalar_t) + sizeof(int)) * topk;

        switch (sortBlockSize)
        {
            case 1:{
                LAUNCH_SELECT_TOPK_SOFTMAX(1);
                break;
            }
            case 4:{
                LAUNCH_SELECT_TOPK_SOFTMAX(4);
                break;
            }
            case 64:{
                LAUNCH_SELECT_TOPK_SOFTMAX(64);
                break;
            }
            case 128:{
                LAUNCH_SELECT_TOPK_SOFTMAX(128);
                break;
            }
            case 256:{
                LAUNCH_SELECT_TOPK_SOFTMAX(256);
                break;
            }
            case 512:{
                LAUNCH_SELECT_TOPK_SOFTMAX(512);
                break;
            }
            case 1024:{
                LAUNCH_SELECT_TOPK_SOFTMAX(1024);
                break;
            }
            default: {
                TORCH_CHECK(softmax_workspace != nullptr,
                    "softmax_workspace must be provided for num_experts that are not a power of 2.");
                static constexpr int TPB = 256;
                mc_moe_softmax_topk::moeSoftmax<scalar_t, TPB><<<num_tokens, TPB, 0, stream>>>(
                    gating_output, softmax_workspace, num_experts);
                mc_moe_softmax_topk::moeTopK<scalar_t, TPB><<<num_tokens, TPB, sizeof(scalar_t) * topk, stream>>>(
                    softmax_workspace, topk_weights, topk_indicies,
                    num_experts, topk, 0, num_experts);
            }
        }
        return;
    }
    // if (!pre_softmax) {
    //     static constexpr int TPB = 256;
    //     mc_moe_softmax_topk::moeTopKSoftmax<scalar_t, TPB><<<num_tokens, TPB, sizeof(scalar_t) * topk, stream>>>(
    //         gating_output, topk_weights, topk_indicies,
    //         num_experts, topk, 0, num_experts);
    //     return;
    // }

    static constexpr int WARPS_PER_TB = 4;
    switch (num_experts) {
        case 1:
            LAUNCH_SOFTMAX(1, WARPS_PER_TB);
            break;
        case 2:
            LAUNCH_SOFTMAX(2, WARPS_PER_TB);
            break;
        case 4:
            LAUNCH_SOFTMAX(4, WARPS_PER_TB);
            break;
        case 8:
            LAUNCH_SOFTMAX(8, WARPS_PER_TB);
            break;
        case 16:
            LAUNCH_SOFTMAX(16, WARPS_PER_TB);
            break;
        case 32:
            LAUNCH_SOFTMAX(32, WARPS_PER_TB);
            break;
        case 64:
            LAUNCH_SOFTMAX(64, WARPS_PER_TB);
            break;
        case 128:
            LAUNCH_SOFTMAX_OPT(128, 8);
            // LAUNCH_SOFTMAX(128, WARPS_PER_TB);
            break;
        case 256:
            LAUNCH_SOFTMAX(256, WARPS_PER_TB);
            break;
        default: {
            TORCH_CHECK(softmax_workspace != nullptr,
                "softmax_workspace must be provided for num_experts that are not a power of 2.");
            static constexpr int TPB = 256;
            mc_moe_softmax_topk::moeSoftmax<scalar_t, TPB><<<num_tokens, TPB, 0, stream>>>(
                gating_output, softmax_workspace, num_experts);
            mc_moe_softmax_topk::moeTopK<scalar_t, TPB><<<num_tokens, TPB, sizeof(scalar_t) * topk, stream>>>(
                softmax_workspace, topk_weights, topk_indicies,
                num_experts, topk, 0, num_experts);
        }
    }
}

template <typename scalar_in_t>
void TopkSoftmaxByteDanceDispatch (at::Tensor input,
                                    at::Tensor out,
                                    at::Tensor indices,
                                    at::Tensor workspace,
                                    int num_tokens,
                                    int num_experts,
                                    int topk,
                                    bool pre_softmax) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    topkGatingSoftmaxKernelLauncher<scalar_in_t>(
        (input.data_ptr<float>()),
        (out.data_ptr<float>()),
        indices.data_ptr<int>(),
        (workspace.data_ptr<float>()),
        num_tokens,
        num_experts,
        topk,
        pre_softmax,
        stream);
}

void moe_softmax_topk(
    at::Tensor topk_weights,                // [num_tokens, topk]
    at::Tensor topk_indices,                // [num_tokens, topk]
    at::Tensor gating_output,
    const bool pre_softmax)               // [num_tokens, num_experts]
{
	DEBUG_TRACE_PARAMS(topk_weights, topk_indices, gating_output, pre_softmax);
	DEBUG_DUMP_PARAMS(topk_weights, topk_indices, gating_output, pre_softmax);
    const int num_experts = gating_output.size(-1);
    const int num_tokens = gating_output.numel() / num_experts;
    const int topk = topk_weights.size(-1);

    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    const bool needs_workspace = !is_pow_2 || num_experts > 256;
    const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;

    torch::Tensor softmax_workspace = torch::empty({workspace_size}, gating_output.options());
    CHECK_DTYPE(gating_output, at::ScalarType::Float);
    TopkSoftmaxByteDanceDispatch<float>(gating_output, topk_weights, topk_indices, softmax_workspace, num_tokens, num_experts, topk, pre_softmax);
}
