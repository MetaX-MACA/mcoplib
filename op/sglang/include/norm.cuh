/*
 * Ported from flashinfer norm.cuh (norm::FusedAddRMSNorm) so that mcoplib's
 * sglang subproject no longer depends on the third-party flashinfer headers.
 *
 * flashinfer-specific helpers have been replaced with MACA-native equivalents:
 *   math::shfl_xor_sync(x, off) -> __shfl_xor_sync(uint64_t(-1), x, off)
 *   math::rsqrt(x)              -> rsqrtf(x)
 *   FLASHINFER_CUDA_CALL        -> MCOPLIB_CUDA_CALL
 *   ceil_div / DISPATCH_ALIGNED_VEC_SIZE / vec_t come from mcoplib headers.
 *
 * This is the fallback path used by fused_add_rms_norm_kernel.cu for shapes not
 * handled by the repo's own vectorized FusedAddRMSNormKernelOpt.
 */
#ifndef MCOPLIB_SGL_NORM_CUH_
#define MCOPLIB_SGL_NORM_CUH_

#include <numeric>
#include <algorithm>

#include "mcoplib_sgl_common.cuh"

namespace mcoplib {
namespace norm {

using mcoplib::ceil_div;
using mcoplib::vec_t;

#define MCOPLIB_NORM_DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...) \
  switch (aligned_vec_size) {                                                           \
    case 16: {                                                                          \
      constexpr size_t ALIGNED_VEC_SIZE = 16;                                           \
      __VA_ARGS__                                                                       \
      break;                                                                            \
    }                                                                                   \
    case 8: {                                                                           \
      constexpr size_t ALIGNED_VEC_SIZE = 8;                                            \
      __VA_ARGS__                                                                       \
      break;                                                                            \
    }                                                                                   \
    case 4: {                                                                           \
      constexpr size_t ALIGNED_VEC_SIZE = 4;                                            \
      __VA_ARGS__                                                                       \
      break;                                                                            \
    }                                                                                   \
    case 2: {                                                                           \
      constexpr size_t ALIGNED_VEC_SIZE = 2;                                            \
      __VA_ARGS__                                                                       \
      break;                                                                            \
    }                                                                                   \
    case 1: {                                                                           \
      constexpr size_t ALIGNED_VEC_SIZE = 1;                                            \
      __VA_ARGS__                                                                       \
      break;                                                                            \
    }                                                                                   \
    default: {                                                                          \
      throw std::invalid_argument("Unsupported aligned_vec_size");                      \
    }                                                                                   \
  }

template <uint32_t VEC_SIZE, typename T>
__global__ void FusedAddRMSNormKernel(T* __restrict__ input, T* __restrict__ residual,
                                      T* __restrict__ weight, const uint32_t d,
                                      const uint32_t stride_input, const uint32_t stride_residual,
                                      float weight_bias, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + ceil_div(num_warps, 4) * 4;

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    vec_t<T, VEC_SIZE> residual_vec;
    residual_vec.fill(0.f);
    vec_t<float, VEC_SIZE> x_vec;
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      residual_vec.load(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                        thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = float(input_vec[j]);
      x += float(residual_vec[j]);
      sum_sq += x * x;
      residual_vec[j] = (T)x;
      x_vec[j] = x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.store(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                         thread_id * VEC_SIZE);
      x_vec.store(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += __shfl_xor_sync(uint64_t(-1), sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += __shfl_xor_sync(uint64_t(-1), sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = rsqrtf(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    vec_t<T, VEC_SIZE> weight_vec;
    vec_t<float, VEC_SIZE> x_vec;
    input_vec.fill(0.f);
    weight_vec.fill(0.f);
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      x_vec.load(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      input_vec[j] = x_vec[j] * rms_rcp * (weight_bias + float(weight_vec[j]));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.store(input + bx * stride_input + i * num_threads * VEC_SIZE +
                      thread_id * VEC_SIZE);
    }
  }
}

template <typename T>
cudaError_t FusedAddRMSNorm(T* input, T* residual, T* weight, uint32_t batch_size, uint32_t d,
                            uint32_t stride_input, uint32_t stride_residual, float eps = 1e-5,
                            bool enable_pdl = false, cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, 32);
  dim3 nblks(batch_size);
  dim3 nthrs(32, num_warps);
  const uint32_t smem_size = (ceil_div(num_warps, 4) * 4 + d) * sizeof(float);
  float weight_bias = 0.f;
  void* args[] = {&input,        &residual,        &weight,      &d,
                  &stride_input, &stride_residual, &weight_bias, &eps};

  MCOPLIB_NORM_DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = FusedAddRMSNormKernel<VEC_SIZE, T>;
    MCOPLIB_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    MCOPLIB_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });

  return cudaSuccess;
}

}  // namespace norm
}  // namespace mcoplib

#endif  // MCOPLIB_SGL_NORM_CUH_
