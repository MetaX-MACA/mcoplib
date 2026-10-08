// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "../../torch_utils.h"

namespace vllm::direct_dcp {

constexpr uint64_t kSpinLimit = 100000000;

// Advance the invocation ID; its low bit selects one of two staging slots.
static __global__ void increment_epoch_kernel(int64_t* epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    epoch[0] += 1;
  }
}

template <typename T>
__device__ __forceinline__ T* get_peer_ptr(const int64_t* peer_ptrs,
                                           int64_t peer) {
  return reinterpret_cast<T*>(static_cast<uintptr_t>(peer_ptrs[peer]));
}

// Replicate one 16-byte payload to every symmetric-buffer replica.
__device__ __forceinline__ void multimem_store_16(uint4* mc_ptr, uint4 value) {
  *mc_ptr = value;
}

// Publish prior system-scope writes and signal every replica.
__device__ __forceinline__ void multimem_store_release_system(uint32_t* mc_ptr,
                                                              uint32_t value) {
  *mc_ptr = value;
}

__device__ __forceinline__ void store_release_system(uint32_t* ptr,
                                                     uint32_t value) {
  *ptr = value;
}

__device__ __forceinline__ uint32_t load_acquire_system(const uint32_t* ptr) {
  uint32_t value = *ptr;
  __threadfence_system();
  return value;
}

__device__ __forceinline__ bool wait_for_epoch(const uint32_t* ptr,
                                               uint32_t epoch) {
  for (uint64_t spins = 0; spins < kSpinLimit; ++spins) {
    if (load_acquire_system(ptr) == epoch) {
      return true;
    }
  }
  return false;
}

inline void check_cuda_launch(const char* operation) {
  cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
                  std::string(operation) +
                      " kernel launch failed: " + cudaGetErrorString(error));
}

}  // namespace vllm::direct_dcp
