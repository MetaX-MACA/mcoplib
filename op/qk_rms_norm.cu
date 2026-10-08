#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

#include "../include/qk_rms_norm.h"
#include "../kernel/utils.h"

namespace {


constexpr int kQkHeadDim = 128;
constexpr int kVecSize = 2;
constexpr int kWarpSize = 64;
constexpr int kThreadsPerHead = kQkHeadDim / kVecSize;
constexpr int kQkRegVecCount =
    (kQkHeadDim / kVecSize + kThreadsPerHead - 1) / kThreadsPerHead;
constexpr int kSmallTokenCount = 16;
constexpr int kSmallTokensHeadsPerBlock = 4;
constexpr int kDefaultHeadsPerBlock = 2;
constexpr int64_t kMaxGridDimY = 65535;

template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T data[Size];
};

template <typename scalar_t, int VecSize, int RegVecCount, int VecCount>
__device__ __forceinline__ void load_input_vector(
    AlignedVector<scalar_t, VecSize> const* head_data,
    int vec_idx,
    int reg_idx,
    float (&reg_input)[RegVecCount][VecSize],
    float& sum_squares) {
  if (vec_idx >= VecCount) {
    return;
  }

  AlignedVector<scalar_t, VecSize> const packed_input = head_data[vec_idx];
#pragma unroll
  for (int element = 0; element < VecSize; ++element) {
    float const input = target_to_float<scalar_t>(packed_input.data[element]);
    reg_input[reg_idx][element] = input;
    sum_squares += input * input;
  }
}

template <int RegIndex, typename scalar_t, int VecSize, int RegVecCount,
          int VecCount, int ThreadsPerHead>
__device__ __forceinline__ void load_input_vectors_unrolled(
    AlignedVector<scalar_t, VecSize> const* head_data,
    int lane_id,
    float (&reg_input)[RegVecCount][VecSize],
    float& sum_squares) {
  if constexpr (RegIndex < RegVecCount) {
    int const vec_idx = lane_id + RegIndex * ThreadsPerHead;
    load_input_vector<scalar_t, VecSize, RegVecCount, VecCount>(
        head_data, vec_idx, RegIndex, reg_input, sum_squares);
    load_input_vectors_unrolled<RegIndex + 1, scalar_t, VecSize, RegVecCount,
                                VecCount, ThreadsPerHead>(
        head_data, lane_id, reg_input, sum_squares);
  }
}

template <typename scalar_t, int VecSize, int RegVecCount, int VecCount,
          int ThreadsPerHead>
__device__ __forceinline__ void load_input_vectors(
    AlignedVector<scalar_t, VecSize> const* head_data,
    int lane_id,
    float (&reg_input)[RegVecCount][VecSize],
    float& sum_squares) {
  if constexpr (RegVecCount <= 4) {
    load_input_vectors_unrolled<0, scalar_t, VecSize, RegVecCount, VecCount,
                                ThreadsPerHead>(
        head_data, lane_id, reg_input, sum_squares);
  } else {
#pragma unroll 1
    for (int reg_idx = 0, vec_idx = lane_id;
         reg_idx < RegVecCount && vec_idx < VecCount;
         ++reg_idx, vec_idx += ThreadsPerHead) {
      load_input_vector<scalar_t, VecSize, RegVecCount, VecCount>(
          head_data, vec_idx, reg_idx, reg_input, sum_squares);
    }
  }
}

template <int WarpSize>
__device__ __forceinline__ float warp_all_reduce_sum(float value) {
  static_assert(WarpSize == 32 || WarpSize == 64,
                "only 32-lane and 64-lane shuffle reductions are supported");
  constexpr uint64_t kFullMask =
      WarpSize == 64 ? 0xffffffffffffffffULL : 0xffffffffULL;
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFullMask, value, offset, WarpSize);
  }
  return value;
}

// Targeted implementation for the initial xpu-perf workload:
//   token_data: BF16 [num_tokens, (q_heads + 2 * kv_heads) * 128]
//   weights:    FP32 [128]
// Grid.x selects a group of Q/K heads and grid.y selects a token. Each
// hardware warp independently processes one head.
template <typename scalar_t, typename weight_t, int HeadDim, int VecSize,
          int RegVecCount, int ThreadsPerHead, int HeadsPerBlock, int WarpSize>
__global__ void qk_rms_norm_inplace_kernel(
    scalar_t* token_data,
    weight_t const* q_norm_weight,
    weight_t const* k_norm_weight,
    int64_t token_stride,
    int64_t norm_head_num,
    int64_t q_head_num,
    float eps) {
  static_assert(HeadDim % VecSize == 0,
                "head dimension must be divisible by vector width");
  static_assert(ThreadsPerHead == WarpSize,
                "the current reduction requires one warp per head");
  static_assert(HeadsPerBlock * ThreadsPerHead <= 1024,
                "block size exceeds the CUDA limit");
  constexpr int kVecCount = HeadDim / VecSize;
  static_assert(RegVecCount > 0, "RegVecCount must be positive");
  static_assert(RegVecCount ==
                    (kVecCount + ThreadsPerHead - 1) / ThreadsPerHead,
                "RegVecCount does not match the kernel configuration");
  using InputVector = AlignedVector<scalar_t, VecSize>;
  using WeightVector = AlignedVector<weight_t, VecSize>;

  int const local_head = threadIdx.x / ThreadsPerHead;
  int const lane_id = threadIdx.x % ThreadsPerHead;
  int64_t const head_idx =
      static_cast<int64_t>(blockIdx.x) * HeadsPerBlock + local_head;
  if (head_idx >= norm_head_num) {
    return;
  }

  int64_t const token_idx = static_cast<int64_t>(blockIdx.y);
  int64_t const head_offset = token_idx * token_stride + head_idx * HeadDim;
  InputVector* const head_data =
      reinterpret_cast<InputVector*>(token_data + head_offset);

  float reg_input[RegVecCount][VecSize];
  float sum_squares = 0.0f;
  load_input_vectors<scalar_t, VecSize, RegVecCount, kVecCount,
                     ThreadsPerHead>(
      head_data, lane_id, reg_input, sum_squares);

  sum_squares = warp_all_reduce_sum<WarpSize>(sum_squares);
  float const inv_rms =
      rsqrtf(sum_squares / static_cast<float>(HeadDim) + eps);

  weight_t const* const weight =
      head_idx < q_head_num ? q_norm_weight : k_norm_weight;
  WeightVector const* const packed_weight =
      reinterpret_cast<WeightVector const*>(weight);
  int reg_idx = 0;
  for (int vec_idx = lane_id; vec_idx < kVecCount;
       vec_idx += ThreadsPerHead) {
    WeightVector const weight_vec = packed_weight[vec_idx];
    InputVector output_vec;
#pragma unroll
    for (int element = 0; element < VecSize; ++element) {
      float const output =
          reg_input[reg_idx][element] * inv_rms *
          target_to_float<weight_t>(weight_vec.data[element]);
      output_vec.data[element] = convert_fp32_to_fp16_rn<scalar_t>(output);
    }
    head_data[vec_idx] = output_vec;
    ++reg_idx;
  }
}

template <typename scalar_t, typename weight_t, int HeadDim, int VecSize,
          int RegVecCount, int ThreadsPerHead, int HeadsPerBlock, int WarpSize>
void launch_qk_rms_norm(
    scalar_t* token_data,
    weight_t const* q_norm_weight,
    weight_t const* k_norm_weight,
    int64_t token_stride,
    int64_t num_tokens,
    int64_t norm_head_num,
    int64_t q_head_num,
    float eps,
    cudaStream_t stream) {
  constexpr int kBlockSize = ThreadsPerHead * HeadsPerBlock;
  dim3 const grid(static_cast<uint32_t>(
                      (norm_head_num + HeadsPerBlock - 1) / HeadsPerBlock),
                  static_cast<uint32_t>(num_tokens));
  qk_rms_norm_inplace_kernel<scalar_t, weight_t, HeadDim, VecSize, RegVecCount,
                             ThreadsPerHead, HeadsPerBlock, WarpSize>
      <<<grid, kBlockSize, 0, stream>>>(
          token_data, q_norm_weight, k_norm_weight, token_stride,
          norm_head_num, q_head_num, eps);
}

template <typename scalar_t, typename weight_t>
void dispatch_qk_rms_norm(
    scalar_t* token_data,
    weight_t const* q_norm_weight,
    weight_t const* k_norm_weight,
    int64_t token_stride,
    int64_t num_tokens,
    int64_t norm_head_num,
    int64_t q_head_num,
    int64_t qk_head_dim,
    float eps,
    cudaStream_t stream) {
  switch (qk_head_dim) {
    case kQkHeadDim:
      if (num_tokens == kSmallTokenCount) {
        launch_qk_rms_norm<scalar_t, weight_t, kQkHeadDim, kVecSize,
                           kQkRegVecCount, kThreadsPerHead,
                           kSmallTokensHeadsPerBlock, kWarpSize>(
            token_data, q_norm_weight, k_norm_weight, token_stride, num_tokens,
            norm_head_num, q_head_num, eps, stream);
      } else {
        launch_qk_rms_norm<scalar_t, weight_t, kQkHeadDim, kVecSize,
                           kQkRegVecCount, kThreadsPerHead,
                           kDefaultHeadsPerBlock, kWarpSize>(
            token_data, q_norm_weight, k_norm_weight, token_stride, num_tokens,
            norm_head_num, q_head_num, eps, stream);
      }
      break;
    default:
      TORCH_CHECK(false, "unsupported qk_head_dim: ", qk_head_dim);
  }
}

}  // namespace

torch::Tensor qk_rms_norm_inplace_cuda(
    torch::Tensor token_data,
    torch::Tensor const& q_norm_weight,
    torch::Tensor const& k_norm_weight,
    int64_t q_head_num,
    int64_t kv_head_num,
    int64_t qk_head_dim,
    double eps) {
  TORCH_CHECK(token_data.is_cuda(), "token_data must be a CUDA tensor");
  TORCH_CHECK(q_norm_weight.is_cuda(),
              "q_norm_weight must be a CUDA tensor");
  TORCH_CHECK(k_norm_weight.is_cuda(),
              "k_norm_weight must be a CUDA tensor");
  TORCH_CHECK(token_data.is_contiguous(), "token_data must be contiguous");
  TORCH_CHECK(q_norm_weight.is_contiguous(),
              "q_norm_weight must be contiguous");
  TORCH_CHECK(k_norm_weight.is_contiguous(),
              "k_norm_weight must be contiguous");
  TORCH_CHECK(token_data.dim() == 2,
              "token_data must be 2D [num_tokens, total_dim]");
  TORCH_CHECK(q_norm_weight.scalar_type() == at::ScalarType::Float,
              "q_norm_weight must be float32");
  TORCH_CHECK(k_norm_weight.scalar_type() == at::ScalarType::Float,
              "k_norm_weight must be float32");
  TORCH_CHECK(token_data.device() == q_norm_weight.device() &&
                  token_data.device() == k_norm_weight.device(),
              "token_data and norm weights must be on the same device");
  TORCH_CHECK(q_head_num > 0, "q_head_num must be positive");
  TORCH_CHECK(kv_head_num > 0, "kv_head_num must be positive");
  TORCH_CHECK(qk_head_dim > 0, "qk_head_dim must be positive");
  TORCH_CHECK(eps >= 0.0, "eps must be non-negative");
  TORCH_CHECK(q_norm_weight.dim() == 1 &&
                  q_norm_weight.numel() == qk_head_dim,
              "q_norm_weight must have shape [qk_head_dim]");
  TORCH_CHECK(k_norm_weight.dim() == 1 &&
                  k_norm_weight.numel() == qk_head_dim,
              "k_norm_weight must have shape [qk_head_dim]");

  // The targeted layout has q_head_dim == v_head_dim == 128:
  // [Q(q_head_num), K(kv_head_num), V(kv_head_num)].
  int64_t const expected_total_dim =
      (q_head_num + 2 * kv_head_num) * qk_head_dim;
  TORCH_CHECK(
      token_data.size(1) == expected_total_dim,
      "token_data.size(1) must equal (q_head_num + 2 * kv_head_num) * "
      "qk_head_dim (expected ",
      expected_total_dim, ", got ", token_data.size(1), ")");

  int64_t const num_tokens = token_data.size(0);
  if (num_tokens == 0) {
    return token_data;
  }

  int64_t const norm_head_num = q_head_num + kv_head_num;
  TORCH_CHECK(norm_head_num <= std::numeric_limits<int32_t>::max(),
              "qk_rms_norm grid.x is too large");
  TORCH_CHECK(num_tokens <= kMaxGridDimY,
              "qk_rms_norm grid.y supports at most ", kMaxGridDimY,
              " tokens, got ", num_tokens);

  at::cuda::OptionalCUDAGuard const device_guard(device_of(token_data));
  cudaStream_t const stream =
      at::cuda::getCurrentCUDAStream(token_data.get_device());

  switch (token_data.scalar_type()) {
    case at::ScalarType::BFloat16:
      dispatch_qk_rms_norm<bfloat16, float>(
          reinterpret_cast<bfloat16*>(token_data.data_ptr<at::BFloat16>()),
          q_norm_weight.data_ptr<float>(), k_norm_weight.data_ptr<float>(),
          token_data.size(1), num_tokens, norm_head_num, q_head_num,
          qk_head_dim, static_cast<float>(eps), stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported token_data dtype: ",
                  token_data.scalar_type());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return token_data;
}
