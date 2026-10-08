#ifndef __MCOPLIB_SGL_COMMON_CUH__
#define __MCOPLIB_SGL_COMMON_CUH__

#include <cuda_bf16.h>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>
#include "vec_dtypes.cuh"
#define WARP_SIZE 32

#define VLLM_LDG(arg) __ldg(arg)

#ifndef NDEBUG
#define MCOPLIB_CUDA_CALL(func, ...)                                                     \
  {                                                                                         \
    cudaError_t e = (func);                                                                 \
    if (e != cudaSuccess) {                                                                 \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e << ") " << __FILE__ \
                << ": line " << __LINE__ << " at function " << STR(func) << std::endl;      \
      return e;                                                                             \
    }                                                                                       \
  }
#else
#define MCOPLIB_CUDA_CALL(func, ...) \
  {                                     \
    cudaError_t e = (func);             \
    if (e != cudaSuccess) {             \
      return e;                         \
    }                                   \
  }
#endif

// ---------------------------------------------------------------------------
// Small helpers ported from flashinfer (utils.cuh::ceil_div,
// layout.cuh::get_elem_offset_impl) so mcoplib's sglang subproject no longer
// depends on the third-party flashinfer headers. Kept in namespace mcoplib and
// pulled in wherever mcoplib_sgl_common.cuh is included.
// ---------------------------------------------------------------------------
namespace mcoplib {

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y) {
  return (x + y - 1) / y;
}

__host__ __device__ __forceinline__ size_t get_elem_offset_impl(size_t elem_idx, size_t head_idx,
                                                                size_t feat_idx, size_t stride_n,
                                                                size_t stride_h) {
  return elem_idx * stride_n + head_idx * stride_h + feat_idx;
}

}  // namespace mcoplib

#endif//__MCOPLIB_SGL_COMMON_CUH__