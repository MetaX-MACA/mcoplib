// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include "../kernel/utils.h"
#include "../kernel/all_reduce_kernel.cuh"
#include "../include/moe_scatter_dynamic_quant.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

__device__ __forceinline__ int32_t ScanWarp2(int32_t val, int32_t mask = 8) {
  int32_t lane = threadIdx.x & 31;
  int32_t tmp = __shfl_up_sync(0xffffffff, val, 1, mask);
  if (lane >= 1) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 2, mask);
  if (lane >= 2) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 4, mask);
  if (lane >= 4) {
    val += tmp;
  }
  return val;
}

__device__ __forceinline__ void sortInPlace(int start_idx, int end_idx, int* scatter_tokens_offset) {
  const int length = end_idx - start_idx;
  auto start_ptr = scatter_tokens_offset + start_idx;
  for (int ii = 0; ii < length; ii++) {
    int first = start_ptr[ii];
    int now_min = first;
    int fir_idx = ii;
    for (int jj = ii + 1; jj < length; jj++) {
      int tmp = start_ptr[jj];
      if (tmp < now_min) {
        now_min = tmp;
        fir_idx = jj;
      }
    }
    start_ptr[ii] = now_min;
    start_ptr[fir_idx] = first;
  }
}

template<int NUM_EXPERTS>
__global__ void moe_align_token_offset(const int* selected_experts, int* scatter_tokens_offset, int* experts_token_count, 
                                        int* experts_token_start, int topk, int num_tokens, int num_experts_per_rank, 
                                        const int shared_tokens_per_sp, 
                                        const int num_shared_experts_per_rank) {
	static constexpr int32_t experts_per_warp = 8;
	__shared__ int shared_counts[32][8];
	__shared__ int32_t blocksum[32];
	__shared__ int32_t cumsum[NUM_EXPERTS + 1];
    // __shared__ int32_t block_offset[NUM_EXPERTS];

	if(threadIdx.x < num_shared_experts_per_rank) {
        experts_token_count[threadIdx.x] = num_tokens;
        experts_token_start[threadIdx.x] = threadIdx.x * shared_tokens_per_sp;
    }
    for (int idx = threadIdx.x; idx < shared_tokens_per_sp * num_shared_experts_per_rank; idx += blockDim.x) {
        scatter_tokens_offset[idx] = idx;
    }

	if (threadIdx.x < NUM_EXPERTS) {
		shared_counts[threadIdx.x / experts_per_warp][threadIdx.x % experts_per_warp] = 0; 
		cumsum[0] = 0;
	}
	__syncthreads();

	#pragma unroll
	for (int idx = threadIdx.x; idx < topk * num_tokens; idx += blockDim.x) {
		int target_expert_id = selected_experts[idx];
		if (target_expert_id < num_experts_per_rank) {
			int idx_in_expert = atomicAdd(&shared_counts[target_expert_id / experts_per_warp][target_expert_id % experts_per_warp], 1);
		}
	}
	__syncthreads();

	int val = 0;
	if (threadIdx.x < 256) {
		int row = threadIdx.x / experts_per_warp;
		int line = threadIdx.x % experts_per_warp;
		val = shared_counts[row][line];
		if (threadIdx.x < num_experts_per_rank) {
			experts_token_count[threadIdx.x + shared_tokens_per_sp * num_shared_experts_per_rank] = val;
		}
	}
	__syncthreads();

	if (threadIdx.x < 256) {
		val = ScanWarp(val);
		if(threadIdx.x % 32 == 31) {
			blocksum[threadIdx.x / 32] = val;
		}
	}
	__syncthreads();

	if(threadIdx.x < 8) {
		int res = blocksum[threadIdx.x];
		blocksum[threadIdx.x] = ScanWarp2(res);
	}
	__syncthreads();

	if(threadIdx.x < 256 && threadIdx.x / 32 > 0){
		val += blocksum[threadIdx.x / 32 - 1];
	}
	__syncthreads();

	if(threadIdx.x < 256){
		cumsum[threadIdx.x + 1] = val;
	}
	__syncthreads();

	if (threadIdx.x < num_experts_per_rank) {
		experts_token_start[threadIdx.x + shared_tokens_per_sp * num_shared_experts_per_rank] = cumsum[threadIdx.x];
	}
    // if (threadIdx.x < NUM_EXPERTS) {
	// 	block_offset[threadIdx.x] = 0; 
	// }
//     __syncthreads();

//   for (int idx = threadIdx.x; idx < topk * num_tokens; idx += blockDim.x) {
// 		int target_expert_id = selected_experts[idx];
// 		int token_id = idx / topk;
// 		if (target_expert_id < num_experts_per_rank) {
// 			int target_expert_start = cumsum[target_expert_id];
// 			int idx_in_expert = atomicAdd(&block_offset[target_expert_id], 1);
// 			scatter_tokens_offset[target_expert_start + idx_in_expert + shared_tokens_per_sp * num_shared_experts_per_rank] = token_id; 
// 		}
// 	}
// 	__syncthreads();
// 	if (threadIdx.x < num_experts_per_rank) {
// 		int start_idx = cumsum[threadIdx.x] + shared_tokens_per_sp * num_shared_experts_per_rank;
// 		int end_idx = cumsum[threadIdx.x + 1] + shared_tokens_per_sp * num_shared_experts_per_rank;
// 		sortInPlace(start_idx, end_idx, scatter_tokens_offset);
// 	}
}

__device__ __forceinline__ int32_t get_expert_id(int32_t output_idx, const int* experts_token_start, const int num_shared_total_tokens, const int num_experts_per_rank) {
    for (int i = 0; i < num_experts_per_rank - 1; ++i) {
        if (output_idx < experts_token_start[i + 1 + num_shared_total_tokens]) {
          return i;
        }
    }
    return num_experts_per_rank - 1;
}

template<int VPT, int BLOCK_SIZE>
__global__ void sort_offset(int* scatter_tokens_offset, const int* experts_token_count, const int* experts_token_start, 
							const int* selected_experts, const int num_tokens, const int topk, const int shared_tokens_per_sp, 
                            const int num_shared_experts_per_rank) {
    __shared__ int count;
    __shared__ int sorted_size;
	int target_expert_id = blockIdx.x;
	int write_start = experts_token_start[target_expert_id];
	int write_size = experts_token_count[target_expert_id];
	int writen_idx = 0;
    const int max_unsorted_size = blockDim.x * VPT;

    typedef cub::BlockRadixSort<int, BLOCK_SIZE, VPT> BlockRadixSort;
    typedef cub::BlockLoad<int, BLOCK_SIZE, VPT> BlockLoad;
    typedef cub::BlockStore<int, BLOCK_SIZE, VPT> BlockStore;

    __shared__ union {
        typename BlockRadixSort::TempStorage sort;
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
    } temp_storage;

    int row_chunk[VPT];

    for (int i = 0; i < VPT; ++i) {
        row_chunk[i] = 999999;
    }

    if (threadIdx.x == 0) {
        count = 0;
        sorted_size = 0;
    }
    int unsorted_size;
    __syncthreads();

	for (int idx = threadIdx.x; idx < num_tokens * topk; idx += blockDim.x) {
		int tmp_expert_id = selected_experts[idx];
        if (tmp_expert_id == target_expert_id) {
            int token_id = idx / topk;
            int idx_in_expert = atomicAdd(&count, 1);
            scatter_tokens_offset[write_start + idx_in_expert + shared_tokens_per_sp * num_shared_experts_per_rank] = token_id;
        }
        __syncthreads();
        unsorted_size = count - sorted_size;
        if (unsorted_size >= max_unsorted_size) {
 
            BlockLoad(temp_storage.load).Load(scatter_tokens_offset + write_start + sorted_size, row_chunk, max_unsorted_size, 999999);
            __syncthreads();

            BlockRadixSort(temp_storage.sort).Sort(row_chunk);
            __syncthreads();

            BlockStore(temp_storage.store).Store(scatter_tokens_offset + write_start + sorted_size, row_chunk, max_unsorted_size);
            if (threadIdx.x == 0) {
                sorted_size += max_unsorted_size;
            }
            __syncthreads();
        }
        if (count == write_size) break;
	}
    __syncthreads();
    if (count == write_size) {
        if (sorted_size < write_size) {
 
            BlockLoad(temp_storage.load).Load(scatter_tokens_offset + write_start + sorted_size, row_chunk, unsorted_size, 999999);
            __syncthreads();

            BlockRadixSort(temp_storage.sort).Sort(row_chunk);
            __syncthreads();

            BlockStore(temp_storage.store).Store(scatter_tokens_offset + write_start + sorted_size, row_chunk, unsorted_size);
        }
    }
}

template<typename scalar_t>
__global__ void moe_scatter_dynamic_quant_kernel(const scalar_t* hidden_states, const int* selected_experts, const float* moe_weights, const float* smooth_scale, 
                                                    int8_t* scatter_tokens, float* scatter_per_token_scale, const int* scatter_tokens_offset, const int* experts_token_start,
                                                    const int topk, const int hidden_size, const int num_experts_per_rank, const int num_shared_total_tokens, const int num_tokens) {
    const int output_idx = blockIdx.x + num_shared_total_tokens;
    const int input_token_idx = scatter_tokens_offset[output_idx];
    if (input_token_idx < 0 || input_token_idx >= topk * num_tokens) return;

    auto input_ptr = hidden_states + input_token_idx * hidden_size;
    auto output_ptr = scatter_tokens + output_idx * hidden_size;
    const int expert_idx = get_expert_id(output_idx, experts_token_start, num_shared_total_tokens, num_experts_per_rank);
    auto smooth_ptr = smooth_scale + expert_idx * hidden_size;

    int token_k_idx = -1;
    for (int i = 0; i < topk; i++) {
        if (selected_experts[input_token_idx * topk + i] == expert_idx){
            token_k_idx = i;
            break;
        }
    }
    const float weights = moe_weights[input_token_idx * topk + token_k_idx];
    float max_val = -99999.f;
    
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float val = abs(__bfloat162float(input_ptr[idx]) * weights * smooth_ptr[idx]);
        max_val = max_val > val ? max_val : val;
    }

    max_val = BlockReduceMax<float>(max_val);
    __shared__ float s_scale;
    if (threadIdx.x == 0) {
      scatter_per_token_scale[output_idx] = max_val * 0.0078740157;
      s_scale = 127.0f * __builtin_mxc_rcpf(max_val);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float val = __bfloat162float(input_ptr[idx]) * weights * smooth_ptr[idx] * s_scale;
        int8_t res = float_to_int8_rn(val);
        output_ptr[idx] = res;
    }
}

#define LAUNCH_SORT(VPT, BLOCK_SIZE)    \
    sort_offset<VPT, BLOCK_SIZE><<<num_experts_per_rank, BLOCK_SIZE, 0, stream>>>(scatter_tokens_offset, experts_token_count, \
                                                                                    experts_token_start, selected_experts, num_tokens, topk, shared_tokens_per_sp,  \
                                                                                    num_shared_experts_per_rank);

template<typename scalar_t>
void launch_moe_scatter_dynamic_quant_kernel(const scalar_t* hidden_status, const int* selected_experts, const float* moe_weights, const float* smooth_scale, 
                                            int8_t* scatter_tokens, float* scatter_per_token_scale, int* scatter_tokens_offset, int* experts_token_count, int* experts_token_start,
                                            const int num_shared_experts_per_rank, const int hidden_size, const int num_tokens, const int num_experts_per_rank,
                                            const int topk, const int shared_tokens_per_sp, const cudaStream_t& stream) {
    constexpr int max_experts = 256;
 	const int total_num = num_tokens * topk;
    const int num_shared_total_tokens = num_shared_experts_per_rank * shared_tokens_per_sp;
    TORCH_CHECK(num_experts_per_rank <= max_experts, "Unsupported, num_experts_per_rank should less than or equal to 256")
    moe_align_token_offset<max_experts><<<1, 512, 0, stream>>>(selected_experts, scatter_tokens_offset, experts_token_count, 
                                                                                    experts_token_start, topk, num_tokens, num_experts_per_rank, 
                                                                                    shared_tokens_per_sp, num_shared_experts_per_rank);

    if (total_num < 16) {
        LAUNCH_SORT(16, 1);
    } else if (total_num < 32) {
        LAUNCH_SORT(1, 16);
    } else if (total_num < 128) {
        LAUNCH_SORT(1, 32);
    } else if (total_num < 256) {
        LAUNCH_SORT(1, 32);
    } else if (total_num < 512) {
        LAUNCH_SORT(2, 64);
    } else if (total_num < 2048) {
        LAUNCH_SORT(4, 128);
    } else {
        LAUNCH_SORT(4, 256);
    }

    moe_scatter_dynamic_quant_kernel<<<total_num, 512, 0, stream>>>(hidden_status, selected_experts, moe_weights, smooth_scale, 
																			scatter_tokens, scatter_per_token_scale, scatter_tokens_offset, experts_token_start,
																			topk, hidden_size, num_experts_per_rank, num_shared_total_tokens, num_tokens);
}

#undef LAUNCH_SORT


void moe_scatter_dynamic_quant(at::Tensor hidden_status, at::Tensor selected_experts, at::Tensor moe_weights, at::Tensor smooth_scale,
                                at::Tensor scatter_tokens, at::Tensor scatter_per_token_scale, at::Tensor scatter_tokens_offset, at::Tensor experts_token_count, at::Tensor experts_token_start,
                                const int experts_per_rank, const int shared_experts_per_rank, const int shared_tokens_per_sp)
{
	DEBUG_TRACE_PARAMS(hidden_status, selected_experts, moe_weights, smooth_scale, scatter_tokens, scatter_per_token_scale, scatter_tokens_offset, experts_token_count, experts_token_start, experts_per_rank, shared_experts_per_rank, shared_tokens_per_sp);
	DEBUG_DUMP_PARAMS(hidden_status, selected_experts, moe_weights, smooth_scale, scatter_tokens, scatter_per_token_scale, scatter_tokens_offset, experts_token_count, experts_token_start, experts_per_rank, shared_experts_per_rank, shared_tokens_per_sp);
	CHECK_DEVICE(hidden_status);
	CHECK_DEVICE(selected_experts);
	CHECK_DEVICE(smooth_scale);

	CHECK_CONTIGUOUS(hidden_status);
	CHECK_CONTIGUOUS(selected_experts);
	CHECK_CONTIGUOUS(smooth_scale);
	const int total_shared_tokens = shared_experts_per_rank * shared_tokens_per_sp;
	
	const int hidden_size = hidden_status.size(-1);
	const int num_tokens = hidden_status.numel() / hidden_size;
	const int topk = selected_experts.size(-1);

	const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	
	if (hidden_status.dtype() == at::ScalarType::BFloat16) {
		launch_moe_scatter_dynamic_quant_kernel<bfloat16>(reinterpret_cast<bfloat16*>(hidden_status.data_ptr<at::BFloat16>()), selected_experts.data_ptr<int>(), moe_weights.data_ptr<float>(), smooth_scale.data_ptr<float>(), 
															scatter_tokens.data_ptr<int8_t>(), scatter_per_token_scale.data_ptr<float>(), scatter_tokens_offset.data_ptr<int>(), experts_token_count.data_ptr<int>(),
															experts_token_start.data_ptr<int>(), 
															shared_experts_per_rank, hidden_size, num_tokens, experts_per_rank, topk, shared_tokens_per_sp, stream);
	} else {
		TORCH_CHECK(false, "Only float16, bfloat16 are supported");
	}
}