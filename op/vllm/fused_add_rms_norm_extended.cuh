// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstring>

// Separate storage dtypes while retaining the existing FP32 dispatch and DAG.
// These helpers are included after fused_norm_load/store in layernorm_kernels.cu.
struct FusedNormExtendedParams {
    const void* input;
    void* residual;
    const void* weight;
    void* output;
    void* input_output;
    int input_type, residual_type, weight_type, output_type;
    uint32_t hidden;
    float epsilon;
};

static int fused_norm_storage_type(at::ScalarType type) {
    switch (type) {
        case at::ScalarType::Float: return 0;
        case at::ScalarType::Half: return 1;
        case at::ScalarType::BFloat16: return 2;
        default: TORCH_CHECK(false, "fused_add_rms_norm requires float32, float16 or bfloat16 storage");
    }
    return 0;
}

template <int N>
__device__ __forceinline__ vllm::vec_n_t<float, N>
fused_norm_load_fp32(const void* ptr, int type, int64_t offset) {
    if (type == 0)
        return fused_norm_load<float, N>(static_cast<const float*>(ptr) + offset);
    vllm::vec_n_t<float, N> result;
    if (type == 1) {
        const auto v = fused_norm_load<half, N>(static_cast<const half*>(ptr) + offset);
#pragma unroll
        for (int j = 0; j < N; ++j) result.val[j] = static_cast<float>(v.val[j]);
    } else {
        const auto v = fused_norm_load<maca_bfloat16, N>(static_cast<const maca_bfloat16*>(ptr) + offset);
#pragma unroll
        for (int j = 0; j < N; ++j) result.val[j] = static_cast<float>(v.val[j]);
    }
    return result;
}

template <int N>
__device__ __forceinline__ void fused_norm_store_fp32(
    void* ptr, int type, int64_t offset, const vllm::vec_n_t<float, N>& value) {
    if (type == 0) {
        fused_norm_store<float, N>(static_cast<float*>(ptr) + offset, value);
    } else if (type == 1) {
        vllm::vec_n_t<half, N> result;
#pragma unroll
        for (int j = 0; j < N; ++j) result.val[j] = float_to_dstT<half>(value.val[j]);
        fused_norm_store<half, N>(static_cast<half*>(ptr) + offset, result);
    } else {
        vllm::vec_n_t<maca_bfloat16, N> result;
#pragma unroll
        for (int j = 0; j < N; ++j) result.val[j] = float_to_dstT<maca_bfloat16>(value.val[j]);
        fused_norm_store<maca_bfloat16, N>(static_cast<maca_bfloat16*>(ptr) + offset, result);
    }
}

// The extra output keeps its existing conversion path. The optional second
// destination controls input writeback independently of the output buffer.
template <int N>
__device__ __forceinline__ void fused_norm_store_results(
    const FusedNormExtendedParams& p, int64_t offset,
    const vllm::vec_n_t<float, N>& value) {
    if (p.output) fused_norm_store_fp32<N>(p.output, p.output_type, offset, value);
    if (p.input_output)
        fused_norm_store_fp32<N>(p.input_output, p.input_type, offset, value);
}

// Match FusedAddRMSNormKernelOpt's ordered subgroup/shared reduction.
template <int NUM_THREADS>
__device__ __forceinline__ float fused_norm_extended_opt_rms(float ss, uint32_t d, float eps) {
    constexpr int sm_size = NUM_THREADS >> 4;
    constexpr int sm_size2 = sm_size / 2;

    __shared__ float sm_sum[sm_size];

    if constexpr (sm_size == 32) {

        for (int i = 8; i > 0; i >>= 1) {
            ss += __shfl_down_sync_16(0xffffffffffffffff, ss, i);
        }

        int lane_id = threadIdx.x & 15;
        int group_id = threadIdx.x >> 4;

        if (lane_id == 0) {
            sm_sum[group_id] = ss;
        }

        __syncthreads();

        __shared__ float sm_sum2[sm_size >> 4];

        if (threadIdx.x < sm_size) {

            float data = sm_sum[threadIdx.x];

            for (int i = 8; i >= 1; i >>= 1) {
                data += __shfl_down_sync_16(0xffffffffffffffff, data, i);
            }

            if (lane_id == 0) {
                sm_sum2[group_id] = data;
            }
        }

        __syncthreads();

        ss = sm_sum2[0] + sm_sum2[1];

    } else if constexpr (sm_size == 16) {

        for (int i = 8; i > 0; i >>= 1) {
            ss += __shfl_down_sync_16(0xffffffffffffffff, ss, i);
        }

        int lane_id = threadIdx.x & 15;
        int group_id = threadIdx.x >> 4;

        if (lane_id == 0) {
            sm_sum[group_id] = ss;
        }

        __syncthreads();

        if (threadIdx.x < sm_size) {

            float data = sm_sum[threadIdx.x];

            for (int i = 8; i >= 1; i >>= 1) {
                data += __shfl_down_sync_16(0xffffffffffffffff, data, i);
            }

            if (threadIdx.x == 0) {
                sm_sum[0] = data;
            }
        }

        __syncthreads();

        ss = sm_sum[0];

    } else if constexpr (sm_size == 8) {

        for (int i = 8; i > 0; i >>= 1) {
            ss += __shfl_down_sync_16(0xffffffffffffffff, ss, i);
        }

        int lane_id = threadIdx.x & 15;
        int group_id = threadIdx.x >> 4;

        if (lane_id == 0) {
            sm_sum[group_id] = ss;
        }

        __syncthreads();

        if (threadIdx.x < sm_size) {

            float data = sm_sum[threadIdx.x];

            for (int i = 4; i >= 1; i >>= 1) {
                data += __shfl_down_sync_16(0xffffffffffffffff, data, i);
            }

            if (threadIdx.x == 0) {
                sm_sum[0] = data;
            }
        }

        __syncthreads();

        ss = sm_sum[0];

    } else if constexpr (sm_size == 4) {

        for (int i = 8; i > 0; i >>= 1) {
            ss += __shfl_down_sync_16(0xffffffffffffffff, ss, i);
        }

        int lane_id = threadIdx.x & 15;
        int group_id = threadIdx.x >> 4;

        if (lane_id == 0) {
            sm_sum[group_id] = ss;
        }

        __syncthreads();

        if (threadIdx.x < sm_size) {

            float data = sm_sum[threadIdx.x];

            for (int i = 2; i >= 1; i >>= 1) {
                data += __shfl_down_sync_16(0xffffffffffffffff, data, i);
            }

            if (threadIdx.x == 0) {
                sm_sum[0] = data;
            }
        }

        __syncthreads();

        ss = sm_sum[0];
    }

    __shared__ float s_rms;

    if (threadIdx.x == 0) {
        s_rms = rsqrtf(ss / (float)d + eps);
    }

    __syncthreads();

    return s_rms;

}

template <int NUM_THREADS, int NUM_REG, bool MULTIROW_DAG, bool ZERO_CENTERED>
__global__ __launch_bounds__(NUM_THREADS) void FusedAddRMSNormExtendedVector(
    FusedNormExtendedParams p) {
    using FloatVec = vllm::vec_n_t<float, 4>;
    const int tid = threadIdx.x;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * p.hidden;
    FloatVec values[NUM_REG];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < NUM_REG; ++k) {
        const int index = (tid + k * NUM_THREADS) * 4;
        if (MULTIROW_DAG || index < p.hidden) {
            const auto x = fused_norm_load_fp32<4>(p.input, p.input_type, row + index);
            const auto r = fused_norm_load_fp32<4>(p.residual, p.residual_type, row + index);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float z = x.val[j] + r.val[j];
                values[k].val[j] = z;
                ss += z * z;
            }
        }
    }
    float rms;
    if constexpr (MULTIROW_DAG) {
        constexpr int NW = NUM_THREADS / 64;
#pragma unroll
        for (int off = 32; off > 0; off >>= 1)
            ss += __shfl_xor_sync(0xffffffffffffffffULL, ss, off);
        if constexpr (NW > 1) {
            __shared__ float red[NW];
            if ((tid & 63) == 0) red[tid / 64] = ss;
            __syncthreads();
            ss = 0.0f;
#pragma unroll
            for (int k = 0; k < NW; ++k) ss += red[k];
        }
        const float inv_d = 1.0f / static_cast<float>(p.hidden);
        rms = __builtin_mxc_rcpf(sqrtf(ss * inv_d + p.epsilon));
    } else {
        rms = fused_norm_extended_opt_rms<NUM_THREADS>(ss, p.hidden, p.epsilon);
    }
#pragma unroll
    for (int k = 0; k < NUM_REG; ++k) {
        const int index = (tid + k * NUM_THREADS) * 4;
        if (MULTIROW_DAG || index < p.hidden) {
            FloatVec w, result;
            if (p.weight) w = fused_norm_load_fp32<4>(p.weight, p.weight_type, index);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float value;
                if constexpr (MULTIROW_DAG) {
                    // The FP32 MultiRow oracle lowers both multiplies with +0.
                    value = __builtin_fmaf(values[k].val[j], rms, 0.0f);
                    if (p.weight)
                        value = __builtin_fmaf(value, vllm::rms_weight<ZERO_CENTERED, float>(w.val[j]), 0.0f);
                } else {
                    value = values[k].val[j] * rms;
                    if (p.weight) value *= vllm::rms_weight<ZERO_CENTERED, float>(w.val[j]);
                }
                result.val[j] = value;
            }
            fused_norm_store_fp32<4>(p.residual, p.residual_type, row + index, values[k]);
            fused_norm_store_results<4>(p, row + index, result);
        }
    }
}

// Aligned Step5 storage modes. Keep the original 256 logical threads, local
// accumulation order and four-warp reduction; only storage choices are static.
template <typename InputT, typename OutputT, bool INPUT_INPLACE, bool HAS_OUT,
          bool ZERO_CENTERED>
__global__ __launch_bounds__(256) void FusedAddRMSNormStep5Typed(
    FusedNormExtendedParams p) {
    using InputVec = vllm::vec_n_t<InputT, 4>;
    using OutputVec = vllm::vec_n_t<OutputT, 4>;
    using FloatVec = vllm::vec_n_t<float, 4>;
    constexpr int NT = 256, NR = 4, ROW_VECTORS = 4096 / 4;
    const int tid = threadIdx.x;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * ROW_VECTORS;
    auto* input = reinterpret_cast<InputVec*>(const_cast<void*>(p.input));
    auto* residual = reinterpret_cast<FloatVec*>(p.residual);
    const auto* weight = reinterpret_cast<const FloatVec*>(p.weight);
    auto* output = reinterpret_cast<OutputVec*>(p.output);
    FloatVec values[NR];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < NR; ++k) {
        const int index = tid + k * NT;
        const InputVec x = input[row + index];
        const FloatVec r = residual[row + index];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float z = static_cast<float>(x.val[j]) + r.val[j];
            values[k].val[j] = z;
            ss += z * z;
        }
    }
#pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        ss += __shfl_xor_sync(0xffffffffffffffffULL, ss, off);
    __shared__ float red[4];
    if ((tid & 63) == 0) red[tid / 64] = ss;
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) ss += red[k];
    // Retain runtime 1/H and the FP32 oracle's sqrt/reciprocal DAG.
    const float inv_d = 1.0f / static_cast<float>(p.hidden);
    const float rms = __builtin_mxc_rcpf(sqrtf(ss * inv_d + p.epsilon));
#pragma unroll
    for (int k = 0; k < NR; ++k) {
        const int index = tid + k * NT;
        const FloatVec w = weight[index];
        InputVec input_result;
        OutputVec output_result;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float value = __builtin_fmaf(values[k].val[j], rms, 0.0f);
            value = __builtin_fmaf(value,
                vllm::rms_weight<ZERO_CENTERED, float>(w.val[j]), 0.0f);
            if constexpr (INPUT_INPLACE) input_result.val[j] = float_to_dstT<InputT>(value);
            if constexpr (HAS_OUT) output_result.val[j] = float_to_dstT<OutputT>(value);
        }
        residual[row + index] = values[k];
        if constexpr (HAS_OUT) output[row + index] = output_result;
        if constexpr (INPUT_INPLACE) input[row + index] = input_result;
    }
}

template <bool ZERO_CENTERED>
static bool launch_fused_norm_step5_typed(
    const FusedNormExtendedParams& p, int rows, bool has_out,
    bool input_inplace, cudaStream_t stream) {
    if (p.input_type == 2 && input_inplace && !has_out) {
        FusedAddRMSNormStep5Typed<maca_bfloat16, maca_bfloat16, true, false, ZERO_CENTERED>
            <<<rows, 256, 0, stream>>>(p);
        return true;
    }
    if (p.input_type == 0 && !input_inplace && has_out && p.output_type == 2) {
        FusedAddRMSNormStep5Typed<float, maca_bfloat16, false, true, ZERO_CENTERED>
            <<<rows, 256, 0, stream>>>(p);
        return true;
    }
    if (p.input_type == 2 && input_inplace && has_out && p.output_type == 0) {
        FusedAddRMSNormStep5Typed<maca_bfloat16, float, true, true, ZERO_CENTERED>
            <<<rows, 256, 0, stream>>>(p);
        return true;
    }
    return false;
}

// Tail path: mirror the scalar FP32 oracle's local sum and CUB reduction.
// Recompute the unrounded sum after reduction, so low-precision residual storage
// never becomes the input to normalization. No full-tensor scratch is allocated.
template <bool ZERO_CENTERED>
__global__ void FusedAddRMSNormExtendedScalar(FusedNormExtendedParams p) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * p.hidden;
    float variance = 0.0f;
    for (int idx = threadIdx.x; idx < p.hidden; idx += blockDim.x) {
        const float z = fused_norm_load_fp32<1>(p.input, p.input_type, row + idx).val[0]
                      + fused_norm_load_fp32<1>(p.residual, p.residual_type, row + idx).val[0];
        variance += z * z;
    }
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduce_store;
    __shared__ float rms;
    variance = BlockReduce(reduce_store).Reduce(variance, CubAddOp{}, blockDim.x);
    if (threadIdx.x == 0) rms = rsqrtf(variance / p.hidden + p.epsilon);
    __syncthreads();
    for (int idx = threadIdx.x; idx < p.hidden; idx += blockDim.x) {
        vllm::vec_n_t<float, 1> sum, output;
        sum.val[0] = fused_norm_load_fp32<1>(p.input, p.input_type, row + idx).val[0]
                   + fused_norm_load_fp32<1>(p.residual, p.residual_type, row + idx).val[0];
        float value = sum.val[0] * rms;
        if (p.weight) {
            const float w = fused_norm_load_fp32<1>(p.weight, p.weight_type, idx).val[0];
            value *= vllm::rms_weight<ZERO_CENTERED, float>(w);
        }
        output.val[0] = value;
        fused_norm_store_fp32<1>(p.residual, p.residual_type, row + idx, sum);
        fused_norm_store_results<1>(p, row + idx, output);
    }
}

// The FP32 fallback groups eight elements per lane. Its device code rounds the
// odd square, then fuses the even square into that pair, and adds pairs in order.
// Explicit RN operations keep the mixed-storage loads from changing that FMA
// contraction through compiler vectorization (observable with FP32 output).
template <bool ZERO_CENTERED>
__global__ void FusedAddRMSNormExtendedPacked(FusedNormExtendedParams p) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * p.hidden;
    float variance = 0.0f;
    for (int index = threadIdx.x; index < p.hidden / 8; index += blockDim.x) {
        vllm::_f16Vec<float, 8> sum;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const auto x = fused_norm_load_fp32<4>(p.input, p.input_type, row + index * 8 + k * 4);
            const auto r = fused_norm_load_fp32<4>(p.residual, p.residual_type, row + index * 8 + k * 4);
#pragma unroll
            for (int j = 0; j < 4; ++j) sum.data[k * 4 + j] = x.val[j] + r.val[j];
        }
        float group = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const float pair = __fmaf_rn(sum.data[j], sum.data[j],
                                         __fmul_rn(sum.data[j + 1], sum.data[j + 1]));
            group = __fadd_rn(group, pair);
        }
        variance = __fadd_rn(variance, group);
    }
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduce_store;
    __shared__ float rms;
    variance = BlockReduce(reduce_store).Reduce(variance, CubAddOp{}, blockDim.x);
    if (threadIdx.x == 0) rms = rsqrtf(variance / p.hidden + p.epsilon);
    __syncthreads();
    for (int index = threadIdx.x; index < p.hidden / 8; index += blockDim.x) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int offset = index * 8 + k * 4;
            const auto x = fused_norm_load_fp32<4>(p.input, p.input_type, row + offset);
            const auto r = fused_norm_load_fp32<4>(p.residual, p.residual_type, row + offset);
            vllm::vec_n_t<float, 4> sum, result, w;
            if (p.weight) w = fused_norm_load_fp32<4>(p.weight, p.weight_type, offset);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sum.val[j] = x.val[j] + r.val[j];
                float value = sum.val[j] * rms;
                if (p.weight) value *= vllm::rms_weight<ZERO_CENTERED, float>(w.val[j]);
                result.val[j] = value;
            }
            fused_norm_store_fp32<4>(p.residual, p.residual_type, row + offset, sum);
            fused_norm_store_results<4>(p, row + offset, result);
        }
    }
}

template <bool ZERO_CENTERED>
static void launch_fused_norm_extended(
    const FusedNormExtendedParams& p, int rows, bool multirow, bool fp32_vector_aligned,
    bool fp32_packed_aligned, bool batch_invariant, cudaStream_t stream) {
    // Select exactly the tile of the existing all-FP32 reference dispatcher.
    if (multirow && fp32_vector_aligned && p.hidden % 4 == 0) {
        const int count = p.hidden / 4;
        int threads = 0, registers = 0;
        for (int n : {64, 128, 256, 512}) {
            if (count % n == 0 && count / n <= 4) {
                threads = n; registers = count / n; break;
            }
        }
#define EXT_MULTI(NT, NR) \
        FusedAddRMSNormExtendedVector<NT, NR, true, ZERO_CENTERED><<<rows, NT, 0, stream>>>(p); return
#define EXT_THREADS(NT) \
        case NT: switch (registers) { \
            case 1: EXT_MULTI(NT, 1); \
            case 2: EXT_MULTI(NT, 2); \
            case 3: EXT_MULTI(NT, 3); \
            case 4: EXT_MULTI(NT, 4); \
        } break
        switch (threads) {
            EXT_THREADS(64); EXT_THREADS(128); EXT_THREADS(256); EXT_THREADS(512);
        }
#undef EXT_THREADS
#undef EXT_MULTI
    } else if (!multirow && p.hidden % 4 == 0) {
#define EXT_OPT(NT, NR) \
        FusedAddRMSNormExtendedVector<NT, NR, false, ZERO_CENTERED><<<rows, NT, 0, stream>>>(p); return
        if (p.hidden <= 256) { EXT_OPT(64, 1); }
        if (p.hidden <= 512) { EXT_OPT(128, 1); }
        if (p.hidden <= 1024) { EXT_OPT(256, 1); }
        if (p.hidden <= 2048) { EXT_OPT(512, 1); }
        if (p.hidden <= 8192) { EXT_OPT(512, 2); }
#undef EXT_OPT
    }
    const int limit = batch_invariant ? 1024 : (rows < 256 ? 1024 : 256);
    const int threads = std::min<int>(p.hidden, limit);
    if (fp32_packed_aligned && p.hidden % 8 == 0 && !batch_invariant)
        FusedAddRMSNormExtendedPacked<ZERO_CENTERED><<<rows, threads, 0, stream>>>(p);
    else
        FusedAddRMSNormExtendedScalar<ZERO_CENTERED><<<rows, threads, 0, stream>>>(p);
}

static void fused_add_rms_norm_extended(
    const torch::Tensor& input, torch::Tensor& residual,
    const std::optional<torch::Tensor>& weight, double epsilon, bool zero_centered,
    const std::optional<torch::Tensor>& out, bool input_inplace = true) {
    const int input_type = fused_norm_storage_type(input.scalar_type());
    const int residual_type = fused_norm_storage_type(residual.scalar_type());
    TORCH_CHECK(residual.scalar_type() == input.scalar_type() || residual_type == 0,
                "fused_add_rms_norm residual dtype must match input or be float32");
    TORCH_CHECK(input.is_cuda() && residual.is_cuda() && input.device() == residual.device(),
                "fused_add_rms_norm requires tensors on the same CUDA device");
    TORCH_CHECK(input.dim() == 2 && residual.sizes() == input.sizes(),
                "fused_add_rms_norm requires matching [T, H] input and residual");
    TORCH_CHECK(input.is_contiguous() && residual.is_contiguous(),
                "fused_add_rms_norm requires contiguous input and residual");
    TORCH_CHECK(input.size(1) > 0 && input.size(1) <= INT_MAX && input.size(0) <= INT_MAX &&
                input.numel() <= UINT_MAX, "fused_add_rms_norm dimensions exceed supported range");
    TORCH_CHECK(!zero_centered || weight.has_value(),
                "fused_add_rms_norm zero_centered requires weight");
    int weight_type = 0;
    if (weight) {
        TORCH_CHECK(weight->is_cuda() && weight->device() == input.device(),
                    "fused_add_rms_norm weight must be on the input CUDA device");
        TORCH_CHECK(weight->scalar_type() == at::ScalarType::Float || weight->scalar_type() == input.scalar_type(),
                    "fused_add_rms_norm weight dtype must match input or be float32");
        TORCH_CHECK(weight->dim() == 1 && weight->numel() == input.size(1) && weight->is_contiguous(),
                    "fused_add_rms_norm requires contiguous weight[H]");
        weight_type = fused_norm_storage_type(weight->scalar_type());
    }
    const torch::Tensor& output = out ? *out : input;
    const int output_type = fused_norm_storage_type(output.scalar_type());
    if (out) {
        TORCH_CHECK(output.is_cuda() && output.device() == input.device(),
                    "fused_add_rms_norm out must be on the input CUDA device");
        TORCH_CHECK(output.sizes() == input.sizes() && output.is_contiguous(),
                    "fused_add_rms_norm out must be contiguous with the input shape");
    }
    const auto overlaps = [](const torch::Tensor& a, const torch::Tensor& b) {
        const auto ab = reinterpret_cast<std::uintptr_t>(a.data_ptr());
        const auto bb = reinterpret_cast<std::uintptr_t>(b.data_ptr());
        const auto an = a.numel() * a.element_size(), bn = b.numel() * b.element_size();
        return an && bn && ab < bb + bn && bb < ab + an;
    };
    TORCH_CHECK(!overlaps(input, residual) && (!weight ||
                (!overlaps(input, *weight) && !overlaps(residual, *weight))),
                "fused_add_rms_norm requires non-overlapping input, residual and weight");
    if (out) {
        TORCH_CHECK(!overlaps(output, input) && !overlaps(output, residual) &&
                    (!weight || !overlaps(output, *weight)),
                    "fused_add_rms_norm out must not overlap input, residual or weight");
    }
    if (input.size(0) == 0) return;
    const at::cuda::OptionalCUDAGuard guard(device_of(input));
    const auto stream = at::cuda::getCurrentCUDAStream();
    FusedNormExtendedParams p{input.data_ptr(), residual.data_ptr(), weight ? weight->data_ptr() : nullptr,
        out || input_inplace ? output.data_ptr() : nullptr,
        out && input_inplace ? input.data_ptr() : nullptr,
        input_type, residual_type, weight_type, output_type,
        static_cast<uint32_t>(input.size(1)), static_cast<float>(epsilon)};
    // Low-precision inputs are promoted into aligned FP32 storage by the oracle;
    // tensors already stored as FP32 retain their actual pointer alignment.
    const auto aligned_ref = [](const torch::Tensor& t) {
        return t.scalar_type() != at::ScalarType::Float ||
               reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16 == 0;
    };
    const bool aligned = aligned_ref(input) && aligned_ref(residual) && (!weight || aligned_ref(*weight));
    const bool packed_aligned = aligned_ref(input) && aligned_ref(residual) &&
        (!weight || weight->scalar_type() != at::ScalarType::Float ||
         reinterpret_cast<std::uintptr_t>(weight->data_ptr()) % 32 == 0);
    const bool multirow = GetDeviceProp().sharedMemPerBlock > 65536;
    const bool invariant = vllm::vllm_is_batch_invariant();
    const auto aligned16 = [](const torch::Tensor& t) {
        return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16 == 0;
    };
    if (multirow && p.hidden == 4096 && residual_type == 0 && weight && weight_type == 0 &&
        aligned16(input) && aligned16(residual) && aligned16(*weight) &&
        (!out || aligned16(*out)) &&
        std::strcmp(at::cuda::getDeviceProperties(input.get_device())->name, "MetaX C600-U") == 0) {
        const bool launched = zero_centered
            ? launch_fused_norm_step5_typed<true>(p, input.size(0), out.has_value(), input_inplace, stream)
            : launch_fused_norm_step5_typed<false>(p, input.size(0), out.has_value(), input_inplace, stream);
        if (launched) return;
    }
    if (zero_centered) launch_fused_norm_extended<true>(p, input.size(0), multirow, aligned, packed_aligned, invariant, stream);
    else launch_fused_norm_extended<false>(p, input.size(0), multirow, aligned, packed_aligned, invariant, stream);
}
