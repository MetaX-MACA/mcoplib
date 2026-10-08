#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "../include/qk_rms_norm.h"
#include "../kernel/utils.h"

#ifndef __shfl_xor_sync_16
#define __shfl_xor_sync_16(mask, val, offset) \
  __shfl_xor_sync(mask, val, offset, 16)
#endif

namespace {

constexpr int kQkHeadDim = 128;
// 128-bit vectorized access: a bf16 float4 holds 8 elements (16 bytes).
constexpr int kVecSize = 8;
constexpr int kVecsPerHead = kQkHeadDim / kVecSize;  // 16

// Four lanes process one head. A native C600-U 16-lane shuffle subgroup
// therefore contains four aligned heads, allowing their reductions to share
// the native shuffle path without mixing values. Each lane owns four 16-byte
// vectors: 64 bytes read and 64 bytes written.
constexpr int kThreadsPerHead = 4;
constexpr int kHeadsPerBlock = 32;                 // block = 128 threads = 2 warps
constexpr int kScanThreadsPerHead = 8;
constexpr int kScanHeadsPerBlock = 8;               // block = 64 threads = 1 warp
constexpr int kSmallThreadsPerHead = 16;
constexpr int kSmallHeadsPerBlock = 4;              // block = 64 threads = 1 warp
constexpr int kMinBlocksPerAP = 8;

// 0 selects the automatic policy. A forced value is cached because reading an
// environment variable on every operator invocation would add host overhead.
int qk_rms_norm_threads_per_head_override() {
  static int const value = [] {
    char const* const env =
        std::getenv("MCOPLIB_QK_RMS_NORM_THREADS_PER_HEAD");
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "auto") == 0) {
      return 0;
    }
    if (std::strcmp(env, "4") == 0) {
      return 4;
    }
    if (std::strcmp(env, "8") == 0) {
      return 8;
    }
    if (std::strcmp(env, "16") == 0) {
      return 16;
    }
    TORCH_CHECK(
        false,
        "MCOPLIB_QK_RMS_NORM_THREADS_PER_HEAD must be auto, 4, 8, or 16; got ",
        env);
    return 0;
  }();
  return value;
}

template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T data[Size];
};

// A native 16-lane subgroup contains an integer number of aligned heads for
// all supported ThreadsPerHead values. XOR offsets below ThreadsPerHead stay
// inside each head, so their reductions execute independently on the native
// shuffle path.
template <int ThreadsPerHead>
__device__ __forceinline__ float subgroup_all_reduce_sum(float value) {
  static_assert(ThreadsPerHead <= 16 && (ThreadsPerHead & (ThreadsPerHead - 1)) == 0,
                "subgroup reduction requires a power-of-two size <= 16");
  static_assert(16 % ThreadsPerHead == 0,
                "heads must align inside a native 16-lane subgroup");
#pragma unroll
  for (int offset = ThreadsPerHead / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync_16(0xffffffffffffffffULL, value, offset);
  }
  return value;
}

// token_data: BF16 [num_tokens, (q_heads + 2*kv_heads) * 128]
// weights:    FP32 [128]
// A flat 1D grid enumerates every (token, normed-head) pair, so there is no
// per-token tail-column waste and the grid floods all APs.
template <typename scalar_t, typename weight_t, int HeadDim, int VecSize,
          int ThreadsPerHead, int HeadsPerBlock>
__global__ void __launch_bounds__(ThreadsPerHead* HeadsPerBlock, kMinBlocksPerAP)
    qk_rms_norm_inplace_kernel(
        scalar_t* __restrict__ token_data,
        weight_t const* __restrict__ q_norm_weight,
        weight_t const* __restrict__ k_norm_weight,
        int64_t token_stride,
        int64_t norm_head_num,
        int64_t q_head_num,
        int64_t total_heads,
        float eps) {
  constexpr int kVecCount = HeadDim / VecSize;          // 16
  constexpr int kRegVec = kVecCount / ThreadsPerHead;   // vectors per lane
  static_assert(kRegVec * ThreadsPerHead == kVecCount,
                "ThreadsPerHead must divide the vectors per head");
  using InputVector = AlignedVector<scalar_t, VecSize>;
  using WeightVector = AlignedVector<weight_t, VecSize>;

  int const local_head = threadIdx.x / ThreadsPerHead;
  int const lane_id = threadIdx.x % ThreadsPerHead;
  int64_t const global_head =
      static_cast<int64_t>(blockIdx.x) * HeadsPerBlock + local_head;
  if (global_head >= total_heads) {
    return;
  }

  // Map the flat head index back to (token, head-within-token).
  int64_t const token_idx = global_head / norm_head_num;
  int const head_in_token =
      static_cast<int>(global_head - token_idx * norm_head_num);
  int64_t const head_offset =
      token_idx * token_stride + static_cast<int64_t>(head_in_token) * HeadDim;
  InputVector* const head_data =
      reinterpret_cast<InputVector*>(token_data + head_offset);

  // Keep this lane's slice in registers as PACKED bf16 (half the register
  // footprint of expanded fp32 -> higher occupancy), converting to float only
  // transiently to accumulate the sum of squares and, later, to normalize.
  InputVector reg_input[kRegVec];
  float sum_squares = 0.0f;
#pragma unroll
  for (int r = 0; r < kRegVec; ++r) {
    reg_input[r] = head_data[lane_id + r * ThreadsPerHead];
#pragma unroll
    for (int e = 0; e < VecSize; ++e) {
      float const v = target_to_float<scalar_t>(reg_input[r].data[e]);
      sum_squares += v * v;
    }
  }

  sum_squares = subgroup_all_reduce_sum<ThreadsPerHead>(sum_squares);
  float const inv_rms =
      __builtin_mxc_rcpf(sqrtf(sum_squares / static_cast<float>(HeadDim) + eps));

  weight_t const* const weight =
      head_in_token < q_head_num ? q_norm_weight : k_norm_weight;
  WeightVector const* const packed_weight =
      reinterpret_cast<WeightVector const*>(weight);
#pragma unroll
  for (int r = 0; r < kRegVec; ++r) {
    int const vec_idx = lane_id + r * ThreadsPerHead;
    WeightVector const weight_vec = packed_weight[vec_idx];
    InputVector output_vec;
#pragma unroll
    for (int e = 0; e < VecSize; ++e) {
      float const out = target_to_float<scalar_t>(reg_input[r].data[e]) *
                        inv_rms * target_to_float<weight_t>(weight_vec.data[e]);
      output_vec.data[e] = convert_fp32_to_fp16_rn<scalar_t>(out);
    }
    head_data[vec_idx] = output_vec;
  }
}

template <typename scalar_t, typename weight_t, int ThreadsPerHead,
          int HeadsPerBlock>
void launch_qk_rms_norm(
    scalar_t* token_data,
    weight_t const* q_norm_weight,
    weight_t const* k_norm_weight,
    int64_t token_stride,
    int64_t norm_head_num,
    int64_t q_head_num,
    int64_t total_heads,
    float eps,
    cudaStream_t stream) {
  int64_t const num_blocks =
      (total_heads + HeadsPerBlock - 1) / HeadsPerBlock;
  TORCH_CHECK(num_blocks <= std::numeric_limits<int32_t>::max(),
              "qk_rms_norm grid is too large");
  dim3 const grid(static_cast<uint32_t>(num_blocks));
  constexpr int kLaunchBlockSize = ThreadsPerHead * HeadsPerBlock;
  qk_rms_norm_inplace_kernel<scalar_t, weight_t, kQkHeadDim, kVecSize,
                             ThreadsPerHead, HeadsPerBlock>
      <<<grid, kLaunchBlockSize, 0, stream>>>(
          token_data, q_norm_weight, k_norm_weight, token_stride,
          norm_head_num, q_head_num, total_heads, eps);
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
  TORCH_CHECK(qk_head_dim == kQkHeadDim, "unsupported qk_head_dim: ", qk_head_dim);

  int64_t const total_heads = num_tokens * norm_head_num;
  int const threads_per_head = qk_rms_norm_threads_per_head_override();
  if (threads_per_head == 4) {
    launch_qk_rms_norm<scalar_t, weight_t, kThreadsPerHead, kHeadsPerBlock>(
        token_data, q_norm_weight, k_norm_weight, token_stride, norm_head_num,
        q_head_num, total_heads, eps, stream);
    return;
  }
  if (threads_per_head == 8) {
    launch_qk_rms_norm<scalar_t, weight_t, kScanThreadsPerHead,
                       kScanHeadsPerBlock>(
        token_data, q_norm_weight, k_norm_weight, token_stride, norm_head_num,
        q_head_num, total_heads, eps, stream);
    return;
  }
  if (threads_per_head == 16) {
    launch_qk_rms_norm<scalar_t, weight_t, kSmallThreadsPerHead,
                       kSmallHeadsPerBlock>(
        token_data, q_norm_weight, k_norm_weight, token_stride, norm_head_num,
        q_head_num, total_heads, eps, stream);
    return;
  }

  if (24576 <= total_heads && total_heads < 98304) {
    launch_qk_rms_norm<scalar_t, weight_t, kScanThreadsPerHead,
                       kScanHeadsPerBlock>(
        token_data, q_norm_weight, k_norm_weight, token_stride, norm_head_num,
        q_head_num, total_heads, eps, stream);
  } else if (norm_head_num >= 32 && total_heads < 4096) {
    launch_qk_rms_norm<scalar_t, weight_t, kScanThreadsPerHead,
                       kScanHeadsPerBlock>(
        token_data, q_norm_weight, k_norm_weight, token_stride, norm_head_num,
        q_head_num, total_heads, eps, stream);
  } else if (norm_head_num >= 32 && total_heads < 8192) {
    launch_qk_rms_norm<scalar_t, weight_t, kSmallThreadsPerHead,
                       kSmallHeadsPerBlock>(
        token_data, q_norm_weight, k_norm_weight, token_stride, norm_head_num,
        q_head_num, total_heads, eps, stream);
  } else if (norm_head_num < 32 && total_heads < 256) {
    launch_qk_rms_norm<scalar_t, weight_t, kSmallThreadsPerHead,
                       kSmallHeadsPerBlock>(
        token_data, q_norm_weight, k_norm_weight, token_stride, norm_head_num,
        q_head_num, total_heads, eps, stream);
  } else {
    launch_qk_rms_norm<scalar_t, weight_t, kThreadsPerHead, kHeadsPerBlock>(
        token_data, q_norm_weight, k_norm_weight, token_stride, norm_head_num,
        q_head_num, total_heads, eps, stream);
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
  TORCH_CHECK(q_norm_weight.is_cuda(), "q_norm_weight must be a CUDA tensor");
  TORCH_CHECK(k_norm_weight.is_cuda(), "k_norm_weight must be a CUDA tensor");
  TORCH_CHECK(token_data.is_contiguous(), "token_data must be contiguous");
  TORCH_CHECK(q_norm_weight.is_contiguous(), "q_norm_weight must be contiguous");
  TORCH_CHECK(k_norm_weight.is_contiguous(), "k_norm_weight must be contiguous");
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
  TORCH_CHECK(q_norm_weight.dim() == 1 && q_norm_weight.numel() == qk_head_dim,
              "q_norm_weight must have shape [qk_head_dim]");
  TORCH_CHECK(k_norm_weight.dim() == 1 && k_norm_weight.numel() == qk_head_dim,
              "k_norm_weight must have shape [qk_head_dim]");

  // Layout: [Q(q_head_num), K(kv_head_num), V(kv_head_num)], all head_dim 128.
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
