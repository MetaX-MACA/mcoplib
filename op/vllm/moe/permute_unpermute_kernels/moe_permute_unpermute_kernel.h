#pragma once
// reference from tensorrt_llm moe kernel implementation archive in
// https://github.com/BBuf/tensorrt-llm-moe/tree/master

#include <ATen/core/Tensor.h>
#include <torch/types.h>

namespace mcoplib_moe_detail {

template <typename T, int N>
struct alignas(16) AlignedArray {
  using Element = T;
  static constexpr int kElements = N;
  T data[N];

  __host__ __device__ T& operator[](int index) { return data[index]; }
  __host__ __device__ const T& operator[](int index) const {
    return data[index];
  }
};

}  // namespace mcoplib_moe_detail

#include "moe/permute_unpermute_kernels/dispatch.h"

template <typename T>
inline T* get_ptr(torch::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
}

template <typename T>
inline const T* get_ptr(const torch::Tensor& t) {
  return reinterpret_cast<const T*>(t.data_ptr());
}

class MacaKeyValueSorter {
 public:
  MacaKeyValueSorter();

  explicit MacaKeyValueSorter(int num_experts);

  void updateNumExperts(int num_experts);

  size_t getWorkspaceSize(size_t num_key_value_pairs,
                          int num_experts);

  void run(void* workspace,
           size_t workspace_size,
           const int* keys_in,
           int* keys_out,
           const int* values_in,
           int* values_out,
           size_t num_key_value_pairs,
           cudaStream_t stream);

 private:
  static int expertsToBits(int num_experts);

  int num_experts_;
  int num_bits_;
};

void computeExpertFirstTokenOffset(
    const int* sorted_indices,
    int total_indices,
    int num_experts,
    int64_t* expert_first_token_offset,
    cudaStream_t stream);

template <typename Sorter>
void sortAndScanExpert(
    const int* expert_for_source_row,
    const int* source_rows,
    int* permuted_experts,
    int* permuted_rows,
    int64_t* expert_first_token_offset,
    int num_rows,
    int num_experts,
    int num_experts_per_node,
    int k,
    Sorter& sorter,
    void* workspace,
    cudaStream_t stream);

template <typename T>
void expandInputRowsKernelLauncher(
    const T* unpermuted_input,
    T* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row,
    int* permuted_idx,
    const int64_t* expert_first_token_offset,
    int64_t num_rows,
    const int64_t* num_valid_tokens_ptr,
    int64_t cols,
    int k,
    int num_local_experts,
    cudaStream_t stream);

template <class T, class OutputType>
void finalizeMoeRoutingKernelLauncher(
    const T* expanded_permuted_rows,
    OutputType* reduced_unpermuted_output,
    const float* scales,
    const int* expanded_source_row_to_expanded_dest_row,
    int64_t num_rows,
    int64_t cols,
    int64_t k,
    const int64_t* num_valid_ptr,
    cudaStream_t stream);

void preprocessTopkIdLauncher(
    int* topk_id_ptr,
    int size,
    const int* expert_map_ptr,
    int num_experts,
    cudaStream_t stream);

#include "moe_permute_unpermute_kernel.inl"
