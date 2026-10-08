#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include "../kernel/utils.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

__host__ __device__ __forceinline__
int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

template<typename scalar_t, int VPT, int SRC_PER_BLOCK>
__global__ void moe_gather_bijective_kernel(
    const scalar_t* __restrict__ input,
    const int*      __restrict__ token_offset,
    const float*    __restrict__ weight,
    const scalar_t* __restrict__ residual,
    scalar_t*       __restrict__ output,
    const float     res_scale,
    const int       res_token_start,
    const int       res_token_end,
    const int       res_row_stride,
    const int       hidden_size,
    const int       D)
{
    using VecType = AlignedArrayI4<scalar_t, VPT>;

    const int src_base = blockIdx.x * SRC_PER_BLOCK;
    const int col      = threadIdx.x * VPT;

    // Number of valid source rows this block owns (last block may be partial).
    int nvalid = D - src_base;
    if (nvalid <= 0) return;
    if (nvalid > SRC_PER_BLOCK) nvalid = SRC_PER_BLOCK;

    // Hoist ALL source-row vector loads up front so the SRC_PER_BLOCK global
    // reads have no mutual address/data dependency and can all be in flight at
    // once (memory-level parallelism == SRC_PER_BLOCK). Then compute + store.
    VecType v[SRC_PER_BLOCK];
    float   w[SRC_PER_BLOCK];
    int     dst[SRC_PER_BLOCK];
    #pragma unroll
    for (int s = 0; s < SRC_PER_BLOCK; ++s) {
        if (s < nvalid) {
            const int src_row = src_base + s;
            w[s]   = weight[src_row];
            dst[s] = token_offset[src_row];
            v[s]   = *reinterpret_cast<const VecType*>(
                         input + src_row * hidden_size + col);
        }
    }

    #pragma unroll
    for (int s = 0; s < SRC_PER_BLOCK; ++s) {
        if (s >= nvalid) break;

        __maca_bfloat162* v2 = reinterpret_cast<__maca_bfloat162*>(&v[s]);
        const float ws = w[s];

        float acc[VPT];
        #pragma unroll
        for (int i = 0; i < VPT/2; ++i) {
            acc[2*i    ] = __bfloat162float(v2[i].x) * ws;
            acc[2*i + 1] = __bfloat162float(v2[i].y) * ws;
        }

        const int d = dst[s];
        if (residual != nullptr && d >= res_token_start && d < res_token_end) {
            const int r_row = (d - res_token_start) * res_row_stride;
            VecType rv = *reinterpret_cast<const VecType*>(
                             residual + r_row * hidden_size + col);
            __maca_bfloat162* rv2 = reinterpret_cast<__maca_bfloat162*>(&rv);
            #pragma unroll
            for (int i = 0; i < VPT/2; ++i) {
                acc[2*i    ] += __bfloat162float(rv2[i].x) * res_scale;
                acc[2*i + 1] += __bfloat162float(rv2[i].y) * res_scale;
            }
        }

        VecType out_v;
        __maca_bfloat162* out2 = reinterpret_cast<__maca_bfloat162*>(&out_v);
        #pragma unroll
        for (int i = 0; i < VPT/2; ++i) {
            out2[i].x = __float2bfloat16(acc[2*i    ]);
            out2[i].y = __float2bfloat16(acc[2*i + 1]);
        }
        *reinterpret_cast<VecType*>(output + d * hidden_size + col) = out_v;
    }
}

__global__ void build_dst_info_kernel(
    const int*   __restrict__ scatter_token_id,
    const float* __restrict__ src_weight,
    int2*        __restrict__ dst_info,
    int*         __restrict__ count,
    const int D, const int topk)
{
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;
    for (int d = tid; d < D; d += step) {
        const int   dst  = scatter_token_id[d];
        const float w    = src_weight[d];
        const int   slot = atomicAdd(count + dst, 1);
        int2 pack = { d, __float_as_int(w) };
        dst_info[dst * topk + slot] = pack;
    }
}

template<typename scalar_t, int VPT, int TOPK, bool IS_SWIZZLE>
__global__ void moe_gather_reverse_kernel(
    const scalar_t* __restrict__ input,
    const int2*     __restrict__ dst_info,
    const scalar_t* __restrict__ residual,
    scalar_t*       __restrict__ output,
    const float     res_scale,
    const int       res_token_start,
    const int       res_token_end,
    const int       res_row_stride,
    const int64_t   hidden_size)
{
    using VecType = AlignedArrayI4<scalar_t, VPT>;
    int dst_row = blockIdx.x;
    if constexpr (IS_SWIZZLE) {
        const int TILE = 2;
        const int ntile = (gridDim.x + TILE - 1) / TILE;
        const int tile  = blockIdx.x / TILE;
        const int lane  = blockIdx.x % TILE;
        dst_row = ((tile * 2654435761u) % ntile) * TILE + lane;
        if (dst_row >= gridDim.x) dst_row = blockIdx.x;
    }

    const int col     = threadIdx.x * VPT;

    float acc[VPT];
    #pragma unroll
    for (int i = 0; i < VPT; ++i) acc[i] = 0.f;

    const int2* my_info = dst_info + dst_row * TOPK;
    #pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        const int2  info    = my_info[k];
        const int   src_row = info.x;
        const float w       = __int_as_float(info.y);

        VecType v = *reinterpret_cast<const VecType*>(input + src_row * hidden_size + col);
        __maca_bfloat162* v2 = reinterpret_cast<__maca_bfloat162*>(&v);
        #pragma unroll
        for (int i = 0; i < VPT/2; ++i) {
            acc[2*i    ] += __bfloat162float(v2[i].x) * w;
            acc[2*i + 1] += __bfloat162float(v2[i].y) * w;
        }
    }

    if (residual != nullptr && dst_row >= res_token_start && dst_row < res_token_end) {
        const int r_row = (dst_row - res_token_start) * res_row_stride;
        VecType v = *reinterpret_cast<const VecType*>(residual + r_row * hidden_size + col);
        __maca_bfloat162* v2 = reinterpret_cast<__maca_bfloat162*>(&v);
        #pragma unroll
        for (int i = 0; i < VPT/2; ++i) {
            acc[2*i    ] += __bfloat162float(v2[i].x) * res_scale;
            acc[2*i + 1] += __bfloat162float(v2[i].y) * res_scale;
        }
    }

    VecType out_v;
    __maca_bfloat162* out2 = reinterpret_cast<__maca_bfloat162*>(&out_v);
    #pragma unroll
    for (int i = 0; i < VPT/2; ++i) {
        out2[i].x = __float2bfloat16(acc[2*i    ]);
        out2[i].y = __float2bfloat16(acc[2*i + 1]);
    }
    *reinterpret_cast<VecType*>(output + dst_row * hidden_size + col) = out_v;
}

template<typename scalar_t>
__global__ void moe_gather_reverse_kernel_generic(
    const scalar_t* __restrict__ input,
    const int2*     __restrict__ dst_info,
    const scalar_t* __restrict__ residual,
    scalar_t*       __restrict__ output,
    const float     res_scale,
    const int       res_token_start,
    const int       res_token_end,
    const int       res_row_stride,
    const int       hidden_size,
    const int       topk)
{
    const int dst_row = blockIdx.x;
    const int2* my_info = dst_info + dst_row * topk;

    const bool has_res = (residual != nullptr) &&
                         (dst_row >= res_token_start) && (dst_row < res_token_end);
    const int  r_row   = has_res ? (dst_row - res_token_start) * res_row_stride : 0;

    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float acc = 0.f;

        for (int k = 0; k < topk; ++k) {
            const int2  info    = my_info[k];
            const int   src_row = info.x;
            const float w       = __int_as_float(info.y);
            acc += __bfloat162float(input[src_row * hidden_size + idx]) * w;
        }

        if (has_res) {
            acc += __bfloat162float(residual[r_row * hidden_size + idx]) * res_scale;
        }

        output[dst_row * hidden_size + idx] = __float2bfloat16(acc);
    }
}

template<typename scalar_t>
void launch_moe_gather(
    const scalar_t* input,
    const int*      scatter_token_id,
    const float*    src_weight,
    const scalar_t* residual,
    scalar_t*       output,
    const float     res_scale,
    const int       res_token_start,
    const int       res_len,
    const int       num_tokens,
    const int       hidden_size,
    const int       D,
    const cudaStream_t& stream,
    at::TensorOptions int_opts)
{
    constexpr int VPT = 8;
    // Residual broadcast semantics: a single residual row (res_len==1) is
    // broadcast to every destination token (matches the sp==1 reference where
    // residual[0] is added to all num_tokens rows); otherwise it is added
    // per-row over [res_token_start, res_token_start+res_len).
    int res_end, res_row_stride;
    if (res_len == 1) {
        res_end        = num_tokens;
        res_row_stride = 0;
    } else {
        res_end        = res_token_start + res_len;
        res_row_stride = 1;
    }
    const bool vec_ok = (hidden_size % VPT == 0);

    const int mpc  = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    if (D <= num_tokens && vec_ok) {
        // VPT=8 (16 B/thread) gives block = H/8 threads. For H=1536 that is 192
        // threads = 3 full 64-wide warps; VPT=16 was measured ~2x SLOWER because
        // block=H/16=96 is only 1.5 warps (half-idle third warp + low occupancy).
        const int block = hidden_size / VPT;
        // Adaptive rows-per-block: the block size is fixed by VPT, so the only
        // knob for grid size is how many source rows each block processes.
        // C600-U keeps several of these blocks resident per AP; pick the LARGEST
        // SRC_PER_BLOCK whose resulting grid still covers >=~3 waves of the whole
        // machine (mpc*32 blocks). Large D keeps SPB=8 (max memory-level
        // parallelism); mid/small D drops SPB to 1-2 to fill all APs and
        // amortize the launch/tail wave. Unrolled per-row loads give MLP == SPB.
        const int target_grid = mpc * 32;
        int spb = 1;
        if (ceil_div(D, 8) >= target_grid)      spb = 8;
        else if (ceil_div(D, 4) >= target_grid) spb = 4;
        else if (ceil_div(D, 2) >= target_grid) spb = 2;
        else                                    spb = 1;

        const int grid = ceil_div(D, spb);
        #define LAUNCH_BIJ(SPB) \
            moe_gather_bijective_kernel<scalar_t, VPT, SPB><<<grid, block, 0, stream>>>( \
                input, scatter_token_id, src_weight, residual, output, \
                res_scale, res_token_start, res_end, res_row_stride, hidden_size, D)
        switch (spb) {
            case 8: LAUNCH_BIJ(8); break;
            case 4: LAUNCH_BIJ(4); break;
            case 2: LAUNCH_BIJ(2); break;
            default: LAUNCH_BIJ(1); break;
        }
        #undef LAUNCH_BIJ
        return;
    }

    const int topk = (D == num_tokens) ? 1 : (D / num_tokens);
    TORCH_CHECK(D == num_tokens * topk, "D must be a multiple of num_tokens");

    auto dst_info = at::zeros({num_tokens, topk, 2}, int_opts);
    auto count    = at::zeros({num_tokens},          int_opts);
    {
        const int b = 256;
        const int g = min(ceil_div(D, b), mpc * 8);
        build_dst_info_kernel<<<g, b, 0, stream>>>(
            scatter_token_id, src_weight,
            reinterpret_cast<int2*>(dst_info.data_ptr<int>()),
            count.data_ptr<int>(), D, topk);
    }
    const int2* info_ptr = reinterpret_cast<const int2*>(dst_info.data_ptr<int>());

    if (vec_ok) {
        const int block = hidden_size / VPT;
        #define LAUNCH(TK) \
            if (num_tokens % 2048 == 0) { \
                moe_gather_reverse_kernel<scalar_t, VPT, TK, true><<<num_tokens, block, 0, stream>>>( \
                    input, info_ptr, residual, output, \
                    res_scale, res_token_start, res_end, res_row_stride, hidden_size); \
            } else { \
                moe_gather_reverse_kernel<scalar_t, VPT, TK, false><<<num_tokens, block, 0, stream>>>( \
                    input, info_ptr, residual, output, \
                    res_scale, res_token_start, res_end, res_row_stride, hidden_size); \
            }

        switch (topk) {
            case 2:  LAUNCH(2);  return;
            case 4:  LAUNCH(4);  return;
            case 6:  LAUNCH(6);  return;
            case 8:  LAUNCH(8);  return;
            case 16: LAUNCH(16); return;
            default: break;
        }
        #undef LAUNCH
    }

    int block = hidden_size < 256 ? hidden_size : 256;
    block = ((block + 31) / 32) * 32;
    if (block == 0) block = 32;

    moe_gather_reverse_kernel_generic<scalar_t>
        <<<num_tokens, block, 0, stream>>>(
            input, info_ptr, residual, output,
            res_scale, res_token_start, res_end, res_row_stride,
            hidden_size, topk);
}

void moe_gather(at::Tensor scatter_tokens,
                at::Tensor scatter_token_id,
                at::Tensor scatter_tokens_weight,
                at::Tensor convergent_tokens,
                c10::optional<at::Tensor> residual_tokens = c10::nullopt,
                double res_scale = 1.0,
                int64_t res_token_start = 0)
{
    DEBUG_TRACE_PARAMS(scatter_tokens, scatter_token_id, scatter_tokens_weight, convergent_tokens, residual_tokens, res_scale, res_token_start);
    DEBUG_DUMP_PARAMS(scatter_tokens, scatter_token_id, scatter_tokens_weight, convergent_tokens, residual_tokens, res_scale, res_token_start);

    const int hidden_size = scatter_tokens.size(-1);
    const int D           = scatter_tokens.numel() / hidden_size;
    const int num_tokens  = convergent_tokens.size(0);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (scatter_tokens.dtype() != at::ScalarType::BFloat16) {
        TORCH_CHECK(false, "Only float16, bfloat16 are supported");
    }

    const __maca_bfloat16* res_ptr = nullptr;
    int res_len = 0;
    if (residual_tokens.has_value()) {
        res_ptr = reinterpret_cast<__maca_bfloat16*>(residual_tokens->data_ptr<at::BFloat16>());
        res_len = residual_tokens->size(0);
    }

    auto int_opts = at::TensorOptions().dtype(at::kInt).device(scatter_tokens.device());

    launch_moe_gather<__maca_bfloat16>(
        reinterpret_cast<__maca_bfloat16*>(scatter_tokens.data_ptr<at::BFloat16>()),
        scatter_token_id.data_ptr<int>(),
        scatter_tokens_weight.data_ptr<float>(),
        res_ptr,
        reinterpret_cast<__maca_bfloat16*>(convergent_tokens.data_ptr<at::BFloat16>()),
        static_cast<float>(res_scale),
        static_cast<int>(res_token_start),
        res_len,
        num_tokens, hidden_size, D, stream, int_opts);
}
