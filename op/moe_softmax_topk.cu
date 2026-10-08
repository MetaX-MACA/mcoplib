#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include <cstring>
#include <cstdlib>
#include "../kernel/moe_softmax_topk.cuh"

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

// Generalized fused path: one 64-thread wave per row, read-once, no workspace.
// Handles arbitrary num_experts and compile-time topk in {4, 8, 16}.
template<typename scalar_t>
bool launchFusedSoftmaxTopk(
    const scalar_t* gating_output, scalar_t* topk_weights, int* topk_indicies,
    const int num_tokens, const int num_experts, const int topk, cudaStream_t stream) {

    if (topk > 16) return false;
    const int vpt = (num_experts + 63) / 64;      // elements per lane
    if (vpt > 14) return false;                    // covers up to 896 experts

    // Diagnostic override (MOE_STK_MODE); unset -> auto (production) dispatch.
    //   orig  -> vendor workspace path (baseline)   probe -> read-ceiling kernel
    //   sort  -> single bitonic pass                sub16 -> force 16-lane subgroup
    static const int stk_mode = [] {
        const char* e = getenv("MOE_STK_MODE");
        if (!e) return 0;
        if (strcmp(e, "orig")  == 0) return 2;
        if (strcmp(e, "probe") == 0) return 1;
        if (strcmp(e, "sort")  == 0) return 3;
        if (strcmp(e, "sub16") == 0) return 4;
        if (strcmp(e, "serial")== 0) return 5;
        if (strcmp(e, "wave64")== 0) return 6;
        if (strcmp(e, "smem")  == 0) return 7;
        if (strcmp(e, "plain") == 0) return 8;
        if (strcmp(e, "batch") == 0) return 9;
        if (strcmp(e, "rprobe")== 0) return 10;
        if (strcmp(e, "bitonic")==0) return 11;
        if (strcmp(e, "w8bitonic")==0) return 12;
        if (strcmp(e, "f32bitonic")==0) return 13;
        return 0;
    }();
    if (stk_mode == 2) return false;               // vendor original path

    // AUTO / sub16: one row per 16-lane subgroup keeps every argmax shuffle
    // intra-16 (C600U fast-shuffle domain) and runs 4 rows per 64-wide warp.
    // Needs VPT16=ceil(E/16) regs/lane; for E>128 that vals[] array spills to
    // local memory -> bandwidth cliff, so auto-select it only for E<=128 where
    // it is a measured ~2.4x win over the 64-lane serial kernel.
    const int vpt16 = (num_experts + 15) / 16;

    // SUBW=8 bitonic (MOE_STK_MODE=w8bitonic): each row owned by an 8-lane
    // subgroup (16 experts/lane), so top-8 needs only 3 butterfly steps = 24
    // shuffles vs 32, and a 64-wide warp runs 8 rows. E=128 fp32 only.
    // AUTO redirect: topk==8 @ E=128 fp32 -> the w8bitonic top-8 kernel is a
    // measured ~1.44x win over the sub16 tournament (192->277 GB/s @ T=65536),
    // numerically identical (cos=1.0 all shapes). topk 4/16 stay on their paths
    // (k=4 tournament is faster; the 3-step bitonic is only exact for top-8).
    const bool auto_w8 = (stk_mode == 0 && topk == 8
                          && sizeof(scalar_t) == sizeof(float));
    if ((stk_mode == 12 || auto_w8) && num_experts == 128) {
        static const int wpc_w8 = [] {
            const char* e = getenv("MOE_STK_WPC");
            return e ? atoi(e) : 0;
        }();
        const int WPC = wpc_w8 > 0 ? wpc_w8 : 4;
        dim3 b8(64, WPC);
#define LAUNCH_W8(WPCv, MAX_K) \
        do { dim3 g8((num_tokens + (size_t)WPCv * 8 - 1) / ((size_t)WPCv * 8)); \
        mc_moe_softmax_topk::fusedSoftmaxTopk8Bitonic<scalar_t, WPCv, MAX_K> \
            <<<g8, b8, 0, stream>>>(gating_output, topk_weights, topk_indicies, \
                                    num_tokens, num_experts); } while (0)
#define DISPATCH_W8_WPC(MAX_K) do { switch (WPC) { \
        case 2: LAUNCH_W8(2, MAX_K); break;  case 6: LAUNCH_W8(6, MAX_K); break; \
        case 8: LAUNCH_W8(8, MAX_K); break;  case 12: LAUNCH_W8(12, MAX_K); break; \
        case 16: LAUNCH_W8(16, MAX_K); break; default: LAUNCH_W8(4, MAX_K); break; \
        } } while (0)
#define LAUNCH_W4(WPCv, PRED) \
        do { dim3 g4((num_tokens + (size_t)WPCv * 16 - 1) / ((size_t)WPCv * 16)); \
        mc_moe_softmax_topk::fusedSoftmaxTopk4Bitonic<scalar_t, WPCv, 8, PRED> \
            <<<g4, b8, 0, stream>>>(gating_output, topk_weights, topk_indicies, \
                                    num_tokens, num_experts); } while (0)
#define DISPATCH_W4_WPC(PRED) do { switch (WPC) { \
        case 2: LAUNCH_W4(2,PRED); break;  case 6: LAUNCH_W4(6,PRED); break; \
        case 8: LAUNCH_W4(8,PRED); break;  case 12: LAUNCH_W4(12,PRED); break; \
        case 16: LAUNCH_W4(16,PRED); break; default: LAUNCH_W4(4,PRED); break; \
        } } while (0)
        switch (topk) {
            case 4:  DISPATCH_W8_WPC(4); break;
            case 8:
                if constexpr (sizeof(scalar_t) == sizeof(float)) {
                    if (num_tokens >= 4096) {
                        if (num_tokens >= 20480) { DISPATCH_W4_WPC(2); }
                        else { DISPATCH_W4_WPC(false); }
                        break;
                    }
                }
                DISPATCH_W8_WPC(8); break;
            case 16: DISPATCH_W8_WPC(16); break;
            default: return false;
        }
#undef DISPATCH_W4_WPC
#undef LAUNCH_W4
#undef DISPATCH_W8_WPC
#undef LAUNCH_W8
        return true;
    }

    // sub16 family: auto (0) for E<=128, force (4), and the experimental modes
    // 7/8/9/10 (smem / plain / batch / rprobe) all dispatch inside this block.
    const bool use_sub16 = (stk_mode == 4) || (stk_mode >= 7)
        || (stk_mode == 0 && num_experts <= 128);
    if (stk_mode == 6) { /* wave64: skip sub16, fall through to contiguous 64-lane kernel */ }
    else if (use_sub16 && vpt16 <= 14) {
        // Occupancy sweep hook: MOE_STK_WPC overrides waves-per-CTA (default 4).
        // The tournament kernel issues only 2 float4 loads then a 32-shuffle
        // serial argmax chain, so more resident waves hide that latency.
        static const int wpc_env = [] {
            const char* e = getenv("MOE_STK_WPC");
            return e ? atoi(e) : 0;
        }();
        const int wpc_sel = wpc_env > 0 ? wpc_env : 4;
        // Row-batched path (MOE_STK_MODE=batch): each subgroup owns RPS rows,
        // firing all loads up front to raise memory-level parallelism. RPS is
        // env-tunable via MOE_STK_RPS (default 4). Only for E=128, topk in {4,8,16}.
        if (stk_mode == 9 && vpt16 == 8) {
            static const int rps_env = [] {
                const char* e = getenv("MOE_STK_RPS");
                return e ? atoi(e) : 0;
            }();
            const int rps = rps_env > 0 ? rps_env : 4;
            const int WPC = 4;
            dim3 b16(64, WPC);
#define LAUNCH_BATCH(RPS, MAX_K) \
            do { dim3 gB((num_tokens + (size_t)WPC * 4 * RPS - 1) / ((size_t)WPC * 4 * RPS)); \
            mc_moe_softmax_topk::fusedSoftmaxTopk16Vpt8BatchTournament<scalar_t, WPC, MAX_K, RPS> \
                <<<gB, b16, 0, stream>>>(gating_output, topk_weights, topk_indicies, \
                                         num_tokens, num_experts); } while (0)
#define DISPATCH_BATCH(MAX_K) do { switch (rps) { \
            case 2: LAUNCH_BATCH(2, MAX_K); break;  case 3: LAUNCH_BATCH(3, MAX_K); break; \
            case 4: LAUNCH_BATCH(4, MAX_K); break;  case 6: LAUNCH_BATCH(6, MAX_K); break; \
            case 8: LAUNCH_BATCH(8, MAX_K); break;  default: LAUNCH_BATCH(4, MAX_K); break; \
            } } while (0)
            switch (topk) {
                case 4:  DISPATCH_BATCH(4);  break;
                case 8:  DISPATCH_BATCH(8);  break;
                case 16: DISPATCH_BATCH(16); break;
                default: return false;
            }
#undef DISPATCH_BATCH
#undef LAUNCH_BATCH
            return true;
        }
#define LAUNCH_SUB16_W(WPC, V, MAX_K) \
        do { dim3 b16(64, WPC); dim3 g16((num_tokens + WPC * 4 - 1) / (WPC * 4)); \
        mc_moe_softmax_topk::fusedSoftmaxTopk16<scalar_t, WPC, V, MAX_K> \
            <<<g16, b16, 0, stream>>>(gating_output, topk_weights, topk_indicies, \
                                      num_tokens, num_experts); } while (0)
#define LAUNCH_SUB16_VPT8_W(WPC, MAX_K) \
        do { dim3 b16(64, WPC); dim3 g16((num_tokens + WPC * 4 - 1) / (WPC * 4)); \
        if (stk_mode == 7) \
            mc_moe_softmax_topk::fusedSoftmaxTopk16Vpt8SmemTournament<scalar_t, WPC, MAX_K> \
                <<<g16, b16, 0, stream>>>(gating_output, topk_weights, topk_indicies, \
                                          num_tokens, num_experts); \
        else if (stk_mode == 8) \
            mc_moe_softmax_topk::fusedSoftmaxTopk16Vpt8PlainTournament<scalar_t, WPC, MAX_K> \
                <<<g16, b16, 0, stream>>>(gating_output, topk_weights, topk_indicies, \
                                          num_tokens, num_experts); \
        else if (stk_mode == 10) \
            mc_moe_softmax_topk::fusedSoftmaxTopk16Vpt8ReadProbe<scalar_t, WPC, MAX_K> \
                <<<g16, b16, 0, stream>>>(gating_output, topk_weights, topk_indicies, \
                                          num_tokens, num_experts); \
        else if (stk_mode == 11) \
            mc_moe_softmax_topk::fusedSoftmaxTopk16Vpt8Bitonic<scalar_t, WPC, MAX_K> \
                <<<g16, b16, 0, stream>>>(gating_output, topk_weights, topk_indicies, \
                                          num_tokens, num_experts); \
        else if (stk_mode == 13) \
            mc_moe_softmax_topk::fusedSoftmaxTopk16Vpt8BitonicF32<scalar_t, WPC, MAX_K> \
                <<<g16, b16, 0, stream>>>(gating_output, topk_weights, topk_indicies, \
                                          num_tokens, num_experts); \
        else \
            mc_moe_softmax_topk::fusedSoftmaxTopk16Vpt8Tournament<scalar_t, WPC, MAX_K> \
                <<<g16, b16, 0, stream>>>(gating_output, topk_weights, topk_indicies, \
                                          num_tokens, num_experts); } while (0)
#define LAUNCH_SUB16(V, MAX_K) do { switch (wpc_sel) { \
        case 2:  LAUNCH_SUB16_W(2, V, MAX_K);  break;  case 6:  LAUNCH_SUB16_W(6, V, MAX_K);  break; \
        case 8:  LAUNCH_SUB16_W(8, V, MAX_K);  break;  case 12: LAUNCH_SUB16_W(12, V, MAX_K); break; \
        case 16: LAUNCH_SUB16_W(16, V, MAX_K); break;  default: LAUNCH_SUB16_W(4, V, MAX_K);  break; \
        } } while (0)
#define LAUNCH_SUB16_VPT8(MAX_K) do { switch (wpc_sel) { \
        case 2:  LAUNCH_SUB16_VPT8_W(2, MAX_K);  break;  case 6:  LAUNCH_SUB16_VPT8_W(6, MAX_K);  break; \
        case 8:  LAUNCH_SUB16_VPT8_W(8, MAX_K);  break;  case 12: LAUNCH_SUB16_VPT8_W(12, MAX_K); break; \
        case 16: LAUNCH_SUB16_VPT8_W(16, MAX_K); break;  default: LAUNCH_SUB16_VPT8_W(4, MAX_K);  break; \
        } } while (0)
#define DISPATCH_SUB16(MAX_K) do { \
        switch (vpt16) { \
            case 1: LAUNCH_SUB16(1, MAX_K); break;   case 2: LAUNCH_SUB16(2, MAX_K); break; \
            case 3: LAUNCH_SUB16(3, MAX_K); break;   case 4: LAUNCH_SUB16(4, MAX_K); break; \
            case 5: LAUNCH_SUB16(5, MAX_K); break;   case 6: LAUNCH_SUB16(6, MAX_K); break; \
            case 7: LAUNCH_SUB16(7, MAX_K); break;   case 8: LAUNCH_SUB16_VPT8(MAX_K); break; \
            case 9: LAUNCH_SUB16(9, MAX_K); break;   case 10: LAUNCH_SUB16(10, MAX_K); break; \
            case 11: LAUNCH_SUB16(11, MAX_K); break; case 12: LAUNCH_SUB16(12, MAX_K); break; \
            case 13: LAUNCH_SUB16(13, MAX_K); break; case 14: LAUNCH_SUB16(14, MAX_K); break; \
            default: return false; \
        } \
    } while (0)
        switch (topk) {
            case 4:  DISPATCH_SUB16(4);  break;
            case 8:  DISPATCH_SUB16(8);  break;
            case 16: DISPATCH_SUB16(16); break;
            default: return false;  // fall back to the original runtime-k path
        }
#undef DISPATCH_SUB16
#undef LAUNCH_SUB16_VPT8
#undef LAUNCH_SUB16
#undef LAUNCH_SUB16_VPT8_W
#undef LAUNCH_SUB16_W
        return true;
    }

    // Sort mode: one bitonic pass emits all top-k (no per-round shuffle chain).
    if (stk_mode == 3) {
        const int warps_per_row  = (num_experts + 63) / 64;
        if (warps_per_row * topk > 64) return false;   // two-level merge constraint
        const int threads_per_row = warps_per_row * 64;
        const int rows_per_cta = MAX(1, 512 / threads_per_row);
        const int block_threads = rows_per_cta * threads_per_row;
        const int num_blocks = (num_tokens + rows_per_cta - 1) / rows_per_cta;
        const int smem = rows_per_cta * warps_per_row * topk * sizeof(int64_t);
        mc_moe_softmax_topk::fusedSoftmaxTopkSort<scalar_t>
            <<<num_blocks, block_threads, smem, stream>>>(
                gating_output, topk_weights, topk_indicies,
                num_tokens, num_experts, topk, rows_per_cta);
        return true;
    }

    constexpr int WAVES_PER_CTA = 8;               // 512 threads/block
    dim3 block(64, WAVES_PER_CTA);
    dim3 grid((num_tokens + WAVES_PER_CTA - 1) / WAVES_PER_CTA);

#define LAUNCH_FUSED(VPT, MAX_K)                                                   \
    do { if (stk_mode == 1)                                                        \
        mc_moe_softmax_topk::fusedSoftmaxTopkProbe<scalar_t, WAVES_PER_CTA, VPT>    \
            <<<grid, block, 0, stream>>>(gating_output, topk_weights,              \
                                         topk_indicies, num_tokens, num_experts, topk); \
    else                                                                           \
        mc_moe_softmax_topk::fusedSoftmaxTopk<scalar_t, WAVES_PER_CTA, VPT, MAX_K>  \
            <<<grid, block, 0, stream>>>(gating_output, topk_weights, topk_indicies,\
                                         num_tokens, num_experts); } while (0)
#define DISPATCH_FUSED(MAX_K) do { \
        switch (vpt) { \
            case 1:  LAUNCH_FUSED(1, MAX_K);  break; \
            case 2:  LAUNCH_FUSED(2, MAX_K);  break; \
            case 3:  LAUNCH_FUSED(3, MAX_K);  break; \
            case 4:  LAUNCH_FUSED(4, MAX_K);  break; \
            case 5:  LAUNCH_FUSED(5, MAX_K);  break; \
            case 6:  LAUNCH_FUSED(6, MAX_K);  break; \
            case 7:  LAUNCH_FUSED(7, MAX_K);  break; \
            case 8:  LAUNCH_FUSED(8, MAX_K);  break; \
            case 9:  LAUNCH_FUSED(9, MAX_K);  break; \
            case 10: LAUNCH_FUSED(10, MAX_K); break; \
            case 11: LAUNCH_FUSED(11, MAX_K); break; \
            case 12: LAUNCH_FUSED(12, MAX_K); break; \
            case 13: LAUNCH_FUSED(13, MAX_K); break; \
            case 14: LAUNCH_FUSED(14, MAX_K); break; \
            default: return false; \
        } \
    } while (0)
    switch (topk) {
        case 4:  DISPATCH_FUSED(4);  break;
        case 8:  DISPATCH_FUSED(8);  break;
        case 16: DISPATCH_FUSED(16); break;
        default: return false;  // fall back to the original runtime-k path
    }
#undef DISPATCH_FUSED
#undef LAUNCH_FUSED
    return true;
}

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

    // Fast fused path for all supported (num_experts, topk) combinations.
    if (launchFusedSoftmaxTopk<scalar_t>(gating_output, topk_weights, topk_indicies,
                                         num_tokens, num_experts, topk, stream)) {
        return;
    }

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