#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include <maca_fp8.h>
#include "../kernel/utils.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

using v4f32 = float __attribute__((ext_vector_type(4)));

template<typename T>
float __device__ __forceinline__ convert(T value) {
    return static_cast<float>(value);
}

template<>
float __device__ __forceinline__ convert(bfloat16 value) {
    return __bfloat162float(value);
}

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ __forceinline__ void copy_data(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;
    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

template<>
__device__ __forceinline__ void copy_data<32>(const void* local, void* data)
{
    const int8_t* in = static_cast<const int8_t*>(local);
    int8_t* out = static_cast<int8_t*>(data);
    *(float4*)out = *(float4*)in;
    *(float4*)(out + 16) = *(float4*)(in + 16);
}

// ==========================================================================
// Token-centric fast path for C600U (MetaX).
//
//   - One 64-lane warp processes ONE routed token row (H = hidden_size).
//   - Each lane vector-loads VPT float4 chunks (N=8 bf16 each) of gate and up,
//     computes silu(gate)*up*smooth_scale, keeps the fp32 result in registers
//     (reg_g[VPT][N]), reduces |absmax| with a pure 64-lane warp shuffle
//     (intra-16 fast steps + cross-group), then quantizes to int8 OR fp8-e4m3
//     straight from registers without shared-memory reduction.
//   - Grid = ceil(num_routed / (WPB*RPW)) blocks -> floods all APs for large T,
//     unlike the old (experts, sm/experts) = 32-block launch.
//   - Row -> expert mapping: block caches the (<=128) expert start offsets in
//     shared memory once, then each warp uses binary search to find the correct
//     smooth_scale row. (ep_size=1 -> 128 local experts.)
//
//   Requires H % (64*N) == 0 (H=1536, N=8 -> VPT=3). Both int8 and fp8 write
//   1 byte/element, so their logical input/output byte counts are identical.
// ==========================================================================

#define WARP 64
// max local experts across all supported ep_size values (ep_size=1 -> 128)
#define MAX_LOCAL_EXPERTS 128

// 64-lane full-warp absmax reduction: fast intra-16 steps then merge 4 groups
// via broadcast reads (xor across 16-groups is the slow path but only 2 steps).
__device__ __forceinline__ float warp_absmax(float v) {
    v = fmaxf(v, __shfl_down_sync_16(0xffffffffffffffffULL, v, 8));
    v = fmaxf(v, __shfl_down_sync_16(0xffffffffffffffffULL, v, 4));
    v = fmaxf(v, __shfl_down_sync_16(0xffffffffffffffffULL, v, 2));
    v = fmaxf(v, __shfl_down_sync_16(0xffffffffffffffffULL, v, 1));
    // now lane 0 of each 16-group holds that group's max; broadcast within group
    v = __shfl_sync(0xffffffffffffffffULL, v, (threadIdx.x & 48), 64);
    // merge the 4 group leaders (lanes 0,16,32,48) via xor(16),xor(32)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffffffffffULL, v, 16, 64));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffffffffffULL, v, 32, 64));
    return v;
}

// OutT     : int8_t (symmetric int8) or __maca_fp8_e4m3 (fp8 e4m3)
// IS_FP8   : selects quant range/pack path. QMAX = 448 (fp8) or 127 (int8).
template<typename T, typename VT, int VPT, int WPB, int RPW,
         typename OutT, bool IS_FP8>
__global__ void __launch_bounds__(WARP * WPB, 4) silu_and_mul_quant_tokencentric(
    const T* __restrict__ input, const float* __restrict__ smooth_scale,
    OutT* __restrict__ output, float* __restrict__ scale,
    const int32_t* __restrict__ expert_token_start,
    int total_experts_num, int64_t hidden_size, int num_routed)
{
    constexpr int N = sizeof(VT) / sizeof(T);      // elems per float4 load (8)
    constexpr float QMAX     = IS_FP8 ? 448.0f : 127.0f;
    constexpr float INV_QMAX = IS_FP8 ? (1.0f / 448.0f) : 0.0078740157f; // 1/127
    const int warp_id = threadIdx.x >> 6;          // 0..WPB-1
    const int lane = threadIdx.x & (WARP - 1);     // 0..63
    const int64_t row0 = ((int64_t)(blockIdx.x) * WPB + warp_id) * RPW;

    // cache expert start offsets in shared mem (<=128 experts)
    __shared__ int32_t s_start[MAX_LOCAL_EXPERTS];
    for (int i = threadIdx.x; i < total_experts_num; i += blockDim.x)
        s_start[i] = expert_token_start[i];
    __syncthreads();

    if (row0 >= num_routed) return;

    const int64_t hidden_size2 = hidden_size << 1;

    // Find the last expert e with s_start[e] <= row0. Starts are monotonic, so
    // binary search keeps the per-warp lookup O(log E).
    int lo = 0, hi = total_experts_num;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (s_start[mid] <= row0) lo = mid; else hi = mid;
    }
    int e = lo;
    // next-expert boundary (row index at which we must reload smooth)
    int next_bnd = (e + 1 < total_experts_num) ? s_start[e + 1] : num_routed;

    // cache this expert's smooth row in registers (reused across RPW rows)
    float sm[VPT][N];
    #pragma unroll
    for (int v = 0; v < VPT; ++v) {
        const int idx = (v * WARP + lane) * N;
        copy_data<sizeof(float) * N>((void*)(smooth_scale + (int64_t)e * hidden_size + idx),
                                     (void*)sm[v]);
    }

    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int64_t row = row0 + r;
        if (row >= num_routed) return;

        // crossed into next expert? advance forward (rows are monotonic, and
        // an expert may own 0 rows so this is a while, not an if).
        if (row >= next_bnd) {
            do {
                ++e;
                next_bnd = (e + 1 < total_experts_num) ? s_start[e + 1] : num_routed;
            } while (row >= next_bnd);
            #pragma unroll
            for (int v = 0; v < VPT; ++v) {
                const int idx = (v * WARP + lane) * N;
                copy_data<sizeof(float) * N>((void*)(smooth_scale + (int64_t)e * hidden_size + idx),
                                             (void*)sm[v]);
            }
        }

        const T* ptr_g = input + row * hidden_size2;
        const T* ptr_u = ptr_g + hidden_size;
        OutT* ptr_out = output + row * hidden_size;

        float reg_g[VPT][N];
        float absmax = 0.0f;
        #pragma unroll
        for (int v = 0; v < VPT; ++v) {
            const int idx = (v * WARP + lane) * N;
            VT vg = *(const VT*)(ptr_g + idx);
            VT vu = *(const VT*)(ptr_u + idx);
            const T* pg = (const T*)&vg;
            const T* pu = (const T*)&vu;
            #pragma unroll
            for (int k = 0; k < N; ++k) {
                float g = convert(pg[k]);
                float u = convert(pu[k]);
                float sig = g * __builtin_mxc_rcpf(1.0f + __builtin_expf(-g));
                float go = u * sig * sm[v][k];
                reg_g[v][k] = go;
                absmax = fmaxf(absmax, fabsf(go));
            }
        }

        absmax = warp_absmax(absmax);
        if (lane == 0)
            scale[row] = absmax * INV_QMAX;
        // guard all-zero row: tmp_scale=0 -> exact zeros (no nan/inf)
        const float tmp_scale = (absmax > 0.0f)
                                  ? (QMAX * __builtin_mxc_rcpf(absmax))
                                  : 0.0f;

        #pragma unroll
        for (int v = 0; v < VPT; ++v) {
            const int idx = (v * WARP + lane) * N;
            if constexpr (IS_FP8) {
                // Convert/pack N=8 fp8 values, then issue one 8B store.
                uint32_t packed[N / 4];
                #pragma unroll
                for (int j = 0; j < N; j += 4) {
                    v4f32 tmp;
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        float rq = reg_g[v][j + t] * tmp_scale;
                        rq = fminf(fmaxf(rq, -448.0f), 448.0f);
                        tmp[t] = rq;
                    }
                    packed[j / 4] = __builtin_mxc_cvt_pk4_f32tof8(tmp);
                }
                *(uint2*)(ptr_out + idx) = *(uint2*)packed;
            } else {
                int8_t vq[N];
                #pragma unroll
                for (int k = 0; k < N; ++k)
                    vq[k] = float_to_int8_rn(reg_g[v][k] * tmp_scale);
                *(float2*)(ptr_out + idx) = *(float2*)vq;
            }
        }
    }
}

// ---- original expert-centric kernel (kept as int8 fallback) ---------------

template<typename T, typename VT, typename VT1, int NUM_VT, int NUM_THREADS>
__global__ void silu_and_mul_quant(const T* input, const float* smooth_scale, int8_t* output, float* scale,
    const int32_t* expert_token_start, const int32_t* expert_token_count, int64_t hidden_size)
{
    constexpr int N = sizeof(VT) / sizeof(T);
    int const tid = threadIdx.x;
    int stride = NUM_THREADS * N;
    int const expert_id = blockIdx.x;
    int64_t hidden_size2 = hidden_size << 1;
    int gridDim_y = gridDim.y;
    int block_count = expert_token_count[expert_id];
    int block_start = expert_token_start[expert_id];
    const float * ptr_smooth_scale = smooth_scale + expert_id * hidden_size;
    using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;
    __shared__ typename BlockReduce::TempStorage reduceStorage;
    __shared__ float block_absmax_val;

    for(int bk = blockIdx.y; bk < block_count; bk += gridDim_y) {
        const T* ptr_input0 = input + (block_start + bk) * hidden_size2;
        const T* ptr_input1 = ptr_input0 + hidden_size;
        int8_t * ptr_output = output + (block_start + bk) * hidden_size;
        float absmax_val = 0.0f;
        float reg_i[NUM_VT][N];
        for(int i = tid*N, j = 0; i < hidden_size; i += stride, j++) {
            VT vsrc0, vsrc1;
            vsrc0 = *(VT*)(ptr_input0 + i);
            vsrc1 = *(VT*)(ptr_input1 + i);
            T* ptr_local0 = (T*)&vsrc0;
            T* ptr_local1 = (T*)&vsrc1;
            float reg_smooth_scale[N];
            copy_data<sizeof(float)*N>((void*)(ptr_smooth_scale + i), (void*)reg_smooth_scale);
            #pragma unroll N
            for(int k = 0; k < N; k++) {
                float val0 = convert(ptr_local0[k]);
                float val1 = convert(ptr_local1[k]);
                float sigmoid = val0 * __builtin_mxc_rcpf(1.0f + __builtin_expf(-val0));
                float gate_up = val1 * sigmoid * reg_smooth_scale[k];
                reg_i[j][k] = gate_up;
                absmax_val = max(absmax_val, abs(gate_up));
            }
        }
        __syncthreads();
        float const block_absmax_val_maybe =
            BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, NUM_THREADS);

        if (tid == 0) {
            block_absmax_val = block_absmax_val_maybe;
            scale[(block_start + bk)] = block_absmax_val * 0.0078740157;
        }
        __syncthreads();
        float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
        for (int i = tid*N, k = 0; i < hidden_size; i += stride, k++) {
            VT1 vdst;
            int8_t* ptr_dst = (int8_t*)&vdst;
            #pragma unroll N
            for(int j = 0; j < N; ++j) {
                ptr_dst[j] = float_to_int8_rn(reg_i[k][j] * tmp_scale);
            }
            *(VT1*)(ptr_output + i) = vdst;
        }
    }
}

// dispatch: OutT/IS_FP8 select the quant dtype for the token-centric fast path.
template<typename T, typename OutT, bool IS_FP8>
void launch_silu_mul_quant_no_mask(const T* input,const float* smooth_scale, OutT* output, float* scale, const int32_t* expert_token_start , const int32_t* expert_token_count,  int total_experts_num, int64_t hidden_size, int num_routed, cudaStream_t stream) {
    int64_t inner_hidden_size = hidden_size / 2;
    constexpr int N = sizeof(float4) / sizeof(T);      // 8 for bf16/half

    // Token-centric fast path: H divisible by 64*N, one warp per row.
    // VPT = H / (64*N). Covers H up to 64*N*VPTMAX.
    if ((inner_hidden_size % (WARP * N)) == 0) {
        const int vpt = inner_hidden_size / (WARP * N);
        // Adaptive rows-per-warp policy selected by the C600U sweep for
        // EP=1/8, H=1536 and both int8/fp8 outputs.  RPW=1 keeps enough blocks
        // for small shapes, RPW=2 improves the transition region, and RPW=4
        // maximizes smooth-scale reuse after the grid reaches steady state.
        const int RPW = (num_routed < 1024) ? 1
                      : (num_routed < 4096) ? 2
                                            : 4;

        bool launched = true;
        #define LAUNCH_TC_R(V, R) do { \
            dim3 block(WARP * 4); \
            const int rows_per_block = 4 * (R); \
            dim3 grid((num_routed + rows_per_block - 1) / rows_per_block); \
            silu_and_mul_quant_tokencentric<T, float4, V, 4, R, OutT, IS_FP8> \
                <<<grid, block, 0, stream>>>(input, smooth_scale, output, scale, \
                    expert_token_start, total_experts_num, inner_hidden_size, num_routed); \
        } while(0)
        #define LAUNCH_TC(V) do { switch (RPW) { \
            case 1: LAUNCH_TC_R(V,1); break; \
            case 2: LAUNCH_TC_R(V,2); break; \
            default: LAUNCH_TC_R(V,4); break; } } while(0)
        switch (vpt) {
            case 1: LAUNCH_TC(1); break;
            case 2: LAUNCH_TC(2); break;
            case 3: LAUNCH_TC(3); break;
            case 4: LAUNCH_TC(4); break;
            case 5: LAUNCH_TC(5); break;
            case 6: LAUNCH_TC(6); break;
            case 8: LAUNCH_TC(8); break;
            default: launched = false; break;
        }
        #undef LAUNCH_TC
        #undef LAUNCH_TC_R
        if (launched) return;
    }

    // Fallback: original expert-centric kernel (int8 only).
    if constexpr (IS_FP8) {
        TORCH_CHECK(false, "moe_swiglu fp8 path requires hidden_size % (64*N) == 0 (e.g. 1536)");
    } else {
        int8_t* output_i8 = reinterpret_cast<int8_t*>(output);
        int dev = 0;
        cudaGetDevice(&dev);
        int sm_count = 0;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        constexpr int blocksize = 512;
        dim3 grid(total_experts_num, (sm_count + total_experts_num - 1) / total_experts_num, 1);
        if(N == 8&&(inner_hidden_size & (N - 1)) == 0) {
            int base = blocksize * N;
            if(inner_hidden_size <= base) {
                silu_and_mul_quant<T, float4, float2, 1, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output_i8, scale, expert_token_start, expert_token_count, inner_hidden_size);
            } else if(inner_hidden_size <= base*2) {
                silu_and_mul_quant<T, float4, float2, 2, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output_i8, scale, expert_token_start, expert_token_count, inner_hidden_size);
            } else if(inner_hidden_size <= base * 3) {
                silu_and_mul_quant<T, float4, float2, 3, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output_i8, scale, expert_token_start, expert_token_count, inner_hidden_size);
            } else if(inner_hidden_size <= base * 4) {
                silu_and_mul_quant<T, float4, float2, 4, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output_i8, scale, expert_token_start, expert_token_count, inner_hidden_size);
            } else {
                TORCH_CHECK(false, "silu_and_mul_quant not support this hidden_size\n");
            }
        } else if(N == 4 && (inner_hidden_size & (N - 1)) == 0) {
            int base = blocksize * N;
            if(inner_hidden_size <= base) {
                silu_and_mul_quant<T, float4, float, 1, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output_i8, scale, expert_token_start, expert_token_count, inner_hidden_size);
            } else if(inner_hidden_size <= base*2) {
                silu_and_mul_quant<T, float4, float, 2, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output_i8, scale, expert_token_start, expert_token_count, inner_hidden_size);
            } else if(inner_hidden_size <= base * 3) {
                silu_and_mul_quant<T, float4, float, 3, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output_i8, scale, expert_token_start, expert_token_count, inner_hidden_size);
            } else if(inner_hidden_size <= base * 4) {
                silu_and_mul_quant<T, float4, float, 4, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output_i8, scale, expert_token_start, expert_token_count, inner_hidden_size);
            } else if(inner_hidden_size <= base * 8) {
                silu_and_mul_quant<T, float4, float, 8, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output_i8, scale, expert_token_start, expert_token_count, inner_hidden_size);
            } else {
                TORCH_CHECK(false, "silu_and_mul_quant not support this hidden_size\n");
            }
        } else {
            TORCH_CHECK(false, "silu_and_mul_quant does not support an unaligned hidden_size\n");
        }
    }
}

void moe_swiglu_dynamic_quantize(at::Tensor scatter_tokens, at::Tensor smooth_scale, at::Tensor experts_tokens_start, at::Tensor experts_tokens_count,
    at::Tensor& y, at::Tensor& per_tokens_scale, int total_experts_num)
{
      DEBUG_TRACE_PARAMS(scatter_tokens, smooth_scale, experts_tokens_start, experts_tokens_count, y, per_tokens_scale, total_experts_num);
  DEBUG_DUMP_PARAMS(scatter_tokens, smooth_scale, experts_tokens_start, experts_tokens_count, y, per_tokens_scale, total_experts_num);
    CHECK_DEVICE(scatter_tokens);
    CHECK_DEVICE(smooth_scale);
    CHECK_DEVICE(experts_tokens_start);
    CHECK_DEVICE(experts_tokens_count);
    CHECK_DEVICE(y);
    CHECK_DEVICE(per_tokens_scale);
    TORCH_CHECK(total_experts_num <= MAX_LOCAL_EXPERTS,
        "moe_swiglu_dynamic_quantize supports at most 128 local experts");

    int64_t const hidden_size = scatter_tokens.size(-1);
    int const num_routed = scatter_tokens.size(0);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const bool out_is_fp8 = (y.dtype() == at::ScalarType::Float8_e4m3fn);
    const bool out_is_int8 = (y.dtype() == at::ScalarType::Char);
    TORCH_CHECK(out_is_fp8 || out_is_int8,
        "moe_swiglu_dynamic_quantize: y must be int8 or float8_e4m3fn");

    if(scatter_tokens.dtype() == at::ScalarType::Half) {
        const half* in = reinterpret_cast<const half*>(scatter_tokens.data_ptr<at::Half>());
        const float* sm = reinterpret_cast<const float*>(smooth_scale.data_ptr<float>());
        float* ps = per_tokens_scale.data_ptr<float>();
        const int32_t* es = (const int32_t*)experts_tokens_start.data_ptr<int32_t>();
        const int32_t* ec = (const int32_t*)experts_tokens_count.data_ptr<int32_t>();
        if (out_is_fp8) {
            launch_silu_mul_quant_no_mask<half, __maca_fp8_e4m3, true>(in, sm,
                reinterpret_cast<__maca_fp8_e4m3*>(y.data_ptr<at::Float8_e4m3fn>()), ps, es, ec, total_experts_num, hidden_size, num_routed, stream);
        } else {
            launch_silu_mul_quant_no_mask<half, int8_t, false>(in, sm,
                y.data_ptr<int8_t>(), ps, es, ec, total_experts_num, hidden_size, num_routed, stream);
        }
    } else if(scatter_tokens.dtype() == at::ScalarType::BFloat16) {
        const bfloat16* in = reinterpret_cast<const bfloat16*>(scatter_tokens.data_ptr<at::BFloat16>());
        const float* sm = reinterpret_cast<const float*>(smooth_scale.data_ptr<float>());
        float* ps = per_tokens_scale.data_ptr<float>();
        const int32_t* es = (const int32_t*)experts_tokens_start.data_ptr<int32_t>();
        const int32_t* ec = (const int32_t*)experts_tokens_count.data_ptr<int32_t>();
        if (out_is_fp8) {
            launch_silu_mul_quant_no_mask<bfloat16, __maca_fp8_e4m3, true>(in, sm,
                reinterpret_cast<__maca_fp8_e4m3*>(y.data_ptr<at::Float8_e4m3fn>()), ps, es, ec, total_experts_num, hidden_size, num_routed, stream);
        } else {
            launch_silu_mul_quant_no_mask<bfloat16, int8_t, false>(in, sm,
                y.data_ptr<int8_t>(), ps, es, ec, total_experts_num, hidden_size, num_routed, stream);
        }
    }else {
        TORCH_CHECK(false, "Only float16, bfloat16 are supported");
    }
}
