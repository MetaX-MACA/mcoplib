// 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.
// All Rights Reserved.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Float8_e4m3fn.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kGroupSize = 128;
constexpr int kElementsPerThread = 8;
constexpr int kThreadsPerGroup = kGroupSize / kElementsPerThread;
constexpr int kMaxExperts = 256;
constexpr float kFp8Max = 448.0f;
constexpr unsigned long long kFullMask = 0xffffffffffffffffULL;

template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedArray {
  T data[N];
};

struct SiluMulQuantVarlenParams {
  const at::BFloat16* __restrict__ input;
  c10::Float8_e4m3fn* __restrict__ output;
  void* __restrict__ output_scale;
  const int32_t* __restrict__ masked_m;
  float swiglu_limit;
  int64_t hidden_dim;
  int32_t num_tokens;
  int32_t num_experts;
};

struct alignas(8) WorkItem {
  int32_t expert_id;
  int32_t expert_token_id;
};

static_assert(sizeof(WorkItem) == 8);
constexpr int64_t kWorkTableHeaderInt32 = 2;

__device__ __forceinline__ float bf16_to_float(at::BFloat16 value) {
  return static_cast<float>(value);
}

__device__ __forceinline__ at::BFloat16 float_to_bf16(float value) {
  return static_cast<at::BFloat16>(value);
}

__device__ __forceinline__ uint8_t cast_to_ue8m0(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t exponent = (bits >> 23) & 0xffu;
  const uint32_t mantissa = bits & 0x7fffffu;
  return static_cast<uint8_t>(exponent + (mantissa != 0));
}

__device__ __forceinline__ float ue8m0_to_float(uint8_t exponent) {
  return __uint_as_float(static_cast<uint32_t>(exponent) << 23);
}

template <bool kApplySwigluLimit>
__device__ __forceinline__ float silu_and_mul(
    at::BFloat16 gate,
    at::BFloat16 up,
    float limit) {
  if constexpr (kApplySwigluLimit) {
    // Match the upstream implementation exactly: clamp the BF16 operands
    // before converting to FP32 and evaluating SiLU(gate) * up.
    const float bf16_limit = bf16_to_float(float_to_bf16(limit));
    gate = float_to_bf16(fminf(bf16_to_float(gate), bf16_limit));
    up = float_to_bf16(
        fminf(fmaxf(bf16_to_float(up), -bf16_limit), bf16_limit));
  }

  const float gate_f = bf16_to_float(gate);
  const float up_f = bf16_to_float(up);
  const float silu = gate_f / (1.0f + __builtin_expf(-gate_f));
  return silu * up_f;
}

__device__ __forceinline__ bool get_work(
    const SiluMulQuantVarlenParams& params,
    int32_t& expert_id,
    int32_t& expert_token_id) {
  __shared__ int32_t work_expert;
  __shared__ int32_t work_token;
  __shared__ bool work_valid;

  if (threadIdx.x == 0) {
    const int32_t work_id = static_cast<int32_t>(blockIdx.x);
    int32_t prefix = 0;
    work_valid = false;
    for (int32_t expert = 0; expert < params.num_experts; ++expert) {
      const int32_t count = params.masked_m[expert];
      if (work_id >= prefix && work_id < prefix + count) {
        work_expert = expert;
        work_token = work_id - prefix;
        work_valid = true;
        break;
      }
      prefix += count;
    }
  }
  __syncthreads();

  expert_id = work_expert;
  expert_token_id = work_token;
  return work_valid;
}

template <bool kScaleUE8M0, bool kTransposed, bool kSwizzle,
          bool kApplySwigluLimit>
__device__ __forceinline__ void silu_mul_quant_varlen_body(
    const SiluMulQuantVarlenParams& params,
    int32_t expert_id,
    int32_t token_id) {
  static_assert(!(kTransposed && !kScaleUE8M0));

  const int64_t token_offset =
      static_cast<int64_t>(expert_id) * params.num_tokens + token_id;
  const at::BFloat16* input =
      params.input + token_offset * params.hidden_dim * 2;
  c10::Float8_e4m3fn* output =
      params.output + token_offset * params.hidden_dim;

  const int32_t group_id = threadIdx.x / kThreadsPerGroup;
  const int32_t lane_id = threadIdx.x % kThreadsPerGroup;
  const int64_t column =
      static_cast<int64_t>(group_id) * kGroupSize +
      lane_id * kElementsPerThread;

  using InputVec = AlignedArray<at::BFloat16, kElementsPerThread>;
  InputVec gate_vec;
  InputVec up_vec;

  if constexpr (kSwizzle) {
    const int64_t chunk = column / kElementsPerThread;
    gate_vec = *reinterpret_cast<const InputVec*>(
        input + chunk * kElementsPerThread * 2);
    up_vec = *reinterpret_cast<const InputVec*>(
        input + (chunk * 2 + 1) * kElementsPerThread);
  } else {
    gate_vec = *reinterpret_cast<const InputVec*>(input + column);
    up_vec = *reinterpret_cast<const InputVec*>(
        input + params.hidden_dim + column);
  }

  float values[kElementsPerThread];
  float local_absmax = 0.0f;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const float value = silu_and_mul<kApplySwigluLimit>(
        gate_vec.data[i], up_vec.data[i], params.swiglu_limit);
    values[i] = value;
    local_absmax = fmaxf(local_absmax, fabsf(value));
  }

#pragma unroll
  for (int offset = kThreadsPerGroup / 2; offset > 0; offset >>= 1) {
    const float other =
        __shfl_xor_sync(kFullMask, local_absmax, offset, kThreadsPerGroup);
    local_absmax = fmaxf(local_absmax, other);
  }

  const float absmax = fmaxf(local_absmax, 1e-10f);
  float scale;
  uint8_t ue8m0_exp = 0;
  if constexpr (kScaleUE8M0) {
    ue8m0_exp = cast_to_ue8m0(absmax / kFp8Max);
    scale = ue8m0_to_float(ue8m0_exp);
  } else {
    scale = absmax / kFp8Max;
  }
  const float inv_scale = 1.0f / scale;

  using OutputVec = AlignedArray<c10::Float8_e4m3fn, kElementsPerThread>;
  OutputVec out_vec;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const float quantized =
        fminf(fmaxf(values[i] * inv_scale, -kFp8Max), kFp8Max);
    out_vec.data[i] = static_cast<c10::Float8_e4m3fn>(quantized);
  }
  *reinterpret_cast<OutputVec*>(output + column) = out_vec;

  if (lane_id == 0) {
    if constexpr (kTransposed) {
      // Physical int32 layout: [E, G / 4, T]. Each int32 packs four
      // consecutive group exponent bytes for a token.
      auto* scale_bytes = static_cast<uint8_t*>(params.output_scale);
      const int64_t num_groups = params.hidden_dim / kGroupSize;
      const int64_t byte_offset =
          static_cast<int64_t>(expert_id) * num_groups * params.num_tokens +
          static_cast<int64_t>(group_id / 4) * params.num_tokens * 4 +
          static_cast<int64_t>(token_id) * 4 + group_id % 4;
      scale_bytes[byte_offset] = ue8m0_exp;
    } else {
      auto* scales = static_cast<float*>(params.output_scale);
      const int64_t num_groups = params.hidden_dim / kGroupSize;
      scales[token_offset * num_groups + group_id] = scale;
    }
  }
}

template <bool kScaleUE8M0, bool kTransposed, bool kSwizzle,
          bool kApplySwigluLimit>
__global__ void silu_mul_quant_varlen_kernel(
    SiluMulQuantVarlenParams params) {
  int32_t expert_id;
  int32_t token_id;
  if (!get_work(params, expert_id, token_id)) {
    return;
  }

  silu_mul_quant_varlen_body<
      kScaleUE8M0, kTransposed, kSwizzle, kApplySwigluLimit>(
      params, expert_id, token_id);
}

__global__ void generate_silu_mul_quant_work_table_kernel(
    const int32_t* masked_m,
    WorkItem* work_table,
    uint32_t* active_count,
    uint32_t* error_flag,
    int32_t num_experts,
    int32_t num_tokens,
    uint32_t max_work) {
  __shared__ uint32_t expert_offsets[kMaxExperts + 1];
  const uint32_t tid = threadIdx.x;

  if (tid == 0) {
    uint64_t offset = 0;
    bool error = false;
    for (int32_t expert = 0; expert < num_experts; ++expert) {
      expert_offsets[expert] =
          error ? 0u : static_cast<uint32_t>(offset);
      const int32_t count = masked_m[expert];
      if (count < 0 || count > num_tokens) {
        error = true;
      } else if (!error) {
        offset += static_cast<uint32_t>(count);
        if (offset > max_work) {
          error = true;
        }
      }
    }

    const uint32_t active =
        error ? 0u : static_cast<uint32_t>(offset);
    expert_offsets[num_experts] = active;
    *active_count = active;
    *error_flag = error ? 1u : 0u;
  }
  __syncthreads();

  const uint32_t active = expert_offsets[num_experts];
  for (uint32_t index = tid; index < active; index += blockDim.x) {
    int32_t low = 0;
    int32_t high = num_experts;
    while (low + 1 < high) {
      const int32_t mid = (low + high) / 2;
      if (expert_offsets[mid] <= index) {
        low = mid;
      } else {
        high = mid;
      }
    }
    work_table[index] = WorkItem{
        low, static_cast<int32_t>(index - expert_offsets[low])};
  }
}

template <bool kScaleUE8M0, bool kTransposed, bool kSwizzle,
          bool kApplySwigluLimit>
__global__ void silu_mul_quant_varlen_persistent_kernel(
    SiluMulQuantVarlenParams params,
    const WorkItem* work_table,
    const uint32_t* active_count) {
  const uint32_t active = *active_count;
  for (uint32_t index = blockIdx.x; index < active; index += gridDim.x) {
    const WorkItem work = work_table[index];
    silu_mul_quant_varlen_body<
        kScaleUE8M0, kTransposed, kSwizzle, kApplySwigluLimit>(
        params, work.expert_id, work.expert_token_id);
  }
}

template <bool kScaleUE8M0, bool kTransposed, bool kSwizzle,
          bool kApplySwigluLimit>
void launch_silu_mul_quant_varlen(
    const SiluMulQuantVarlenParams& params,
    int64_t grid_size,
    int64_t num_threads,
    cudaStream_t stream) {
  silu_mul_quant_varlen_kernel<
      kScaleUE8M0, kTransposed, kSwizzle, kApplySwigluLimit>
      <<<grid_size, num_threads, 0, stream>>>(params);
}

template <bool kScaleUE8M0, bool kTransposed, bool kSwizzle,
          bool kApplySwigluLimit>
void launch_silu_mul_quant_varlen_persistent(
    const SiluMulQuantVarlenParams& params,
    int32_t* workspace,
    int64_t persistent_grid,
    uint32_t max_work,
    int64_t num_threads,
    cudaStream_t stream) {
  auto* active_count = reinterpret_cast<uint32_t*>(workspace);
  auto* error_flag = reinterpret_cast<uint32_t*>(workspace + 1);
  auto* work_table =
      reinterpret_cast<WorkItem*>(workspace + kWorkTableHeaderInt32);

  generate_silu_mul_quant_work_table_kernel<<<1, 256, 0, stream>>>(
      params.masked_m,
      work_table,
      active_count,
      error_flag,
      params.num_experts,
      params.num_tokens,
      max_work);
  silu_mul_quant_varlen_persistent_kernel<
      kScaleUE8M0, kTransposed, kSwizzle, kApplySwigluLimit>
      <<<persistent_grid, num_threads, 0, stream>>>(
          params, work_table, active_count);
}

template <bool kScaleUE8M0, bool kTransposed, bool kSwizzle>
void dispatch_persistent_limit(
    const SiluMulQuantVarlenParams& params,
    int32_t* workspace,
    int64_t persistent_grid,
    uint32_t max_work,
    int64_t num_threads,
    cudaStream_t stream,
    bool use_limit) {
  if (use_limit) {
    launch_silu_mul_quant_varlen_persistent<
        kScaleUE8M0, kTransposed, kSwizzle, true>(
        params,
        workspace,
        persistent_grid,
        max_work,
        num_threads,
        stream);
  } else {
    launch_silu_mul_quant_varlen_persistent<
        kScaleUE8M0, kTransposed, kSwizzle, false>(
        params,
        workspace,
        persistent_grid,
        max_work,
        num_threads,
        stream);
  }
}

template <bool kScaleUE8M0, bool kTransposed, bool kSwizzle>
void dispatch_limit(
    const SiluMulQuantVarlenParams& params,
    int64_t grid_size,
    int64_t num_threads,
    cudaStream_t stream,
    bool use_limit) {
  if (use_limit) {
    launch_silu_mul_quant_varlen<
        kScaleUE8M0, kTransposed, kSwizzle, true>(
        params, grid_size, num_threads, stream);
  } else {
    launch_silu_mul_quant_varlen<
        kScaleUE8M0, kTransposed, kSwizzle, false>(
        params, grid_size, num_threads, stream);
  }
}

template <bool kScaleUE8M0, bool kTransposed, bool kSwizzle>
void dispatch_selected_path(
    const SiluMulQuantVarlenParams& params,
    int64_t grid_size,
    int64_t num_threads,
    cudaStream_t stream,
    bool use_limit,
    int32_t* workspace,
    int64_t persistent_grid) {
  if (workspace != nullptr) {
    dispatch_persistent_limit<
        kScaleUE8M0, kTransposed, kSwizzle>(
        params,
        workspace,
        persistent_grid,
        static_cast<uint32_t>(grid_size),
        num_threads,
        stream,
        use_limit);
  } else {
    dispatch_limit<kScaleUE8M0, kTransposed, kSwizzle>(
        params, grid_size, num_threads, stream, use_limit);
  }
}

}  // namespace

void silu_mul_quant_varlen(
    const torch::Tensor& input,
    torch::Tensor& output,
    torch::Tensor& output_scale,
    const torch::Tensor& masked_m,
    int64_t topk,
    bool scale_ue8m0,
    bool transposed,
    bool swizzle,
    c10::optional<double> swiglu_limit,
    bool enable_pdl,
    const c10::optional<torch::Tensor>& workspace,
    int64_t persistent_grid) {
  TORCH_CHECK(!enable_pdl,
              "silu_mul_quant_varlen does not support enable_pdl=true on "
              "the current MetaX backend");

  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(output_scale.is_cuda(), "output_scale must be a CUDA tensor");
  TORCH_CHECK(masked_m.is_cuda(), "masked_m must be a CUDA tensor");
  TORCH_CHECK(input.device() == output.device() &&
                  input.device() == output_scale.device() &&
                  input.device() == masked_m.device(),
              "all tensors must be on the same CUDA device");

  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
              "input dtype must be bfloat16");
  TORCH_CHECK(output.scalar_type() == torch::kFloat8_e4m3fn,
              "output dtype must be float8_e4m3fn");
  TORCH_CHECK(masked_m.scalar_type() == torch::kInt32,
              "masked_m dtype must be int32");

  TORCH_CHECK(input.dim() == 3, "input must have shape [E, T, 2H]");
  TORCH_CHECK(output.dim() == 3, "output must have shape [E, T, H]");
  TORCH_CHECK(masked_m.dim() == 1, "masked_m must have shape [E]");

  const int64_t num_experts = input.size(0);
  const int64_t num_tokens = input.size(1);
  const int64_t input_hidden = input.size(2);
  TORCH_CHECK(input_hidden % 2 == 0,
              "input last dimension must equal 2 * hidden_dim");
  const int64_t hidden_dim = input_hidden / 2;
  const int64_t num_groups = hidden_dim / kGroupSize;

  TORCH_CHECK(output.size(0) == num_experts &&
                  output.size(1) == num_tokens &&
                  output.size(2) == hidden_dim,
              "output must have shape [E, T, H]");
  TORCH_CHECK(masked_m.size(0) == num_experts,
              "masked_m length must equal E");
  TORCH_CHECK(num_experts > 0 && num_experts <= kMaxExperts,
              "num_experts must be in [1, 256]");
  TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
  TORCH_CHECK(hidden_dim > 0 && hidden_dim % kGroupSize == 0,
              "hidden_dim must be positive and divisible by 128");
  TORCH_CHECK(hidden_dim % (kElementsPerThread * 32) == 0,
              "hidden_dim must be divisible by 256");

  const int64_t num_threads = hidden_dim / kElementsPerThread;
  TORCH_CHECK(num_threads >= num_experts,
              "hidden_dim / 8 must be at least num_experts");
  TORCH_CHECK(num_threads <= 1024,
              "hidden_dim is too large: hidden_dim / 8 must be <= 1024");
  TORCH_CHECK(topk > 0, "topk must be positive");
  TORCH_CHECK(num_tokens <= INT32_MAX && num_experts <= INT32_MAX,
              "tensor dimensions exceed the kernel's int32 range");
  TORCH_CHECK(topk <= INT64_MAX / num_tokens,
              "num_tokens * topk overflows int64");
  const int64_t grid_size = num_tokens * topk;
  TORCH_CHECK(grid_size <= INT32_MAX,
              "num_tokens * topk exceeds the kernel's int32 work-id range");

  int32_t* work_table_workspace = nullptr;
  int64_t work_table_grid = 0;
  if (persistent_grid > 0 && workspace.has_value() &&
      workspace->defined()) {
    const torch::Tensor& scratch = *workspace;
    const int64_t required_workspace_int32 =
        kWorkTableHeaderInt32 +
        grid_size * (sizeof(WorkItem) / sizeof(int32_t));
    const bool workspace_is_suitable =
        scratch.is_cuda() &&
        scratch.device() == input.device() &&
        scratch.scalar_type() == torch::kInt32 &&
        scratch.is_contiguous() &&
        scratch.dim() == 1 &&
        scratch.numel() >= required_workspace_int32;
    if (workspace_is_suitable) {
      work_table_workspace = scratch.data_ptr<int32_t>();
      work_table_grid =
          persistent_grid < grid_size ? persistent_grid : grid_size;
    }
  }

  if (transposed) {
    TORCH_CHECK(scale_ue8m0,
                "transposed layout requires scale_ue8m0=true");
    TORCH_CHECK(num_groups % 4 == 0,
                "transposed layout requires hidden_dim / 128 divisible by 4");
    TORCH_CHECK(output_scale.is_contiguous(),
                "transposed output_scale must be contiguous");
    TORCH_CHECK(output_scale.scalar_type() == torch::kInt32,
                "transposed output_scale dtype must be int32");
    TORCH_CHECK(output_scale.dim() == 3 &&
                    output_scale.size(0) == num_experts &&
                    output_scale.size(1) == num_groups / 4 &&
                    output_scale.size(2) == num_tokens,
                "transposed output_scale must have shape [E, G / 4, T]");
  } else {
    TORCH_CHECK(output_scale.is_contiguous(),
                "output_scale must be contiguous");
    TORCH_CHECK(output_scale.scalar_type() == torch::kFloat32,
                "non-transposed output_scale dtype must be float32");
    TORCH_CHECK(output_scale.dim() == 3 &&
                    output_scale.size(0) == num_experts &&
                    output_scale.size(1) == num_tokens &&
                    output_scale.size(2) == num_groups,
                "non-transposed output_scale must have shape [E, T, G]");
  }

  float limit = 0.0f;
  bool use_limit = false;
  if (swiglu_limit.has_value()) {
    const double value = *swiglu_limit;
    TORCH_CHECK(std::isfinite(value), "swiglu_limit must be finite");
    TORCH_CHECK(value > 0.0,
                "swiglu_limit must be strictly positive");
    limit = static_cast<float>(value);
    use_limit = true;
  }

  const SiluMulQuantVarlenParams params{
      input.data_ptr<at::BFloat16>(),
      output.data_ptr<c10::Float8_e4m3fn>(),
      output_scale.data_ptr(),
      masked_m.data_ptr<int32_t>(),
      limit,
      hidden_dim,
      static_cast<int32_t>(num_tokens),
      static_cast<int32_t>(num_experts),
  };

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (transposed) {
    if (swizzle) {
      dispatch_selected_path<true, true, true>(
          params, grid_size, num_threads, stream, use_limit,
          work_table_workspace, work_table_grid);
    } else {
      dispatch_selected_path<true, true, false>(
          params, grid_size, num_threads, stream, use_limit,
          work_table_workspace, work_table_grid);
    }
  } else if (scale_ue8m0) {
    if (swizzle) {
      dispatch_selected_path<true, false, true>(
          params, grid_size, num_threads, stream, use_limit,
          work_table_workspace, work_table_grid);
    } else {
      dispatch_selected_path<true, false, false>(
          params, grid_size, num_threads, stream, use_limit,
          work_table_workspace, work_table_grid);
    }
  } else {
    if (swizzle) {
      dispatch_selected_path<false, false, true>(
          params, grid_size, num_threads, stream, use_limit,
          work_table_workspace, work_table_grid);
    } else {
      dispatch_selected_path<false, false, false>(
          params, grid_size, num_threads, stream, use_limit,
          work_table_workspace, work_table_grid);
    }
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
