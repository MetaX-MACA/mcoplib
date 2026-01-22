// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace fused_mla {

#ifdef __HPCC_ARCH__

#define __builtin_rcpf(x) __builtin_htc_rcpf(x)

#define __builtin_mbcnt_lo(mask, initial_value) __builtin_htc_mbcnt_lo(mask, initial_value)

#elif defined(__MACA_ARCH__)

#define __builtin_rcpf(x) __builtin_mxc_rcpf(x)

#define __builtin_mbcnt_lo(mask, initial_value) __builtin_mxc_mbcnt_lo(mask, initial_value)

#endif

template <typename scalar_t>
__device__ __forceinline__ float scalar2float(scalar_t val) {
    if constexpr (std::is_same_v<scalar_t, half> || std::is_same_v<scalar_t, at::Half>) {
        return __half2float(val);
    }
    else if constexpr (std::is_same_v<scalar_t, nv_bfloat16> || std::is_same_v<scalar_t, at::BFloat16>) {
        return static_cast<float>(val);
    }
    else {
        if((threadIdx.x == 0) && (blockIdx.x == 0)){
            printf("unsupported scalar type. %s %d\n", __func__, __LINE__);
        } 
    }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t float2scalar(float val) {
    if constexpr (std::is_same_v<scalar_t, half> || std::is_same_v<scalar_t, at::Half>) {
        return __float2half(val);
    } else if constexpr (std::is_same_v<scalar_t, nv_bfloat16> || std::is_same_v<scalar_t, at::BFloat16>) {
        return __float2bfloat16_rn(val);
    } else {
        if((threadIdx.x == 0) && (blockIdx.x == 0)){
            printf("unsupported scalar type. %s %d\n", __func__, __LINE__);
        }
    }
}

template<class scalar_t>
    __device__ __forceinline__ scalar_t get_weight(const int32_t& v) {
        const scalar_t* idx_and_weight = (const scalar_t*)&v;
        return idx_and_weight[0];
    }

    template<class scalar_t, int WARP_SIZE=32, uint64_t MASK=0xffffffff>
    __device__ __forceinline__ void warpSortDescending(scalar_t (&idx_and_weight)[2], int tid) {
        int32_t key, val, other_temp_val;
        scalar_t weight;
        if constexpr (std::is_same_v<scalar_t, float>) {
            key = *(int32_t*)&idx_and_weight[1];
            weight = idx_and_weight[0];
        } else {
            val = *(int32_t*)idx_and_weight;
        }

        for (int width = 2; width <= WARP_SIZE; width <<=1) {
            for (int step = width >> 1; step > 0; step >>=1) {
                const bool is_not_final_phase = (width != WARP_SIZE);
                const uint32_t bitmask = (tid & width);
                const bool direction = is_not_final_phase & (bitmask == 0);
                scalar_t current_weight_bits, other_weight_bits;
                int current_index, other_index;
                if constexpr (std::is_same_v<scalar_t, float>) {
                    current_weight_bits = weight;
                    other_index = __shfl_xor_sync(MASK, key, step);
                    other_weight_bits = __shfl_xor_sync(MASK, weight, step);
                    current_index = key;
                } else {
                    other_temp_val = __shfl_xor_sync(MASK, val, step);
                    current_weight_bits = get_weight<scalar_t>(val);
                    other_weight_bits = get_weight<scalar_t>(other_temp_val);
                    current_index = val >> 16;
                    other_index = other_temp_val >> 16;
                }

                bool weight_gt = false;
                bool weight_eq = false;
                if constexpr (std::is_same_v<scalar_t, __half> || std::is_same_v<scalar_t, at::Half>)
                {
                    float val_other = __half2float((__half)other_weight_bits);
                    float val_curr  = __half2float((__half)current_weight_bits);
                        
                    weight_gt = val_other > val_curr;
                    weight_eq = val_other == val_curr;
                } else if constexpr (std::is_same_v<scalar_t, __nv_bfloat16> || std::is_same_v<scalar_t, at::BFloat16>)
                {
                    weight_gt = other_weight_bits > current_weight_bits;
                    weight_eq = other_weight_bits == current_weight_bits;
                } else {
                    weight_gt = other_weight_bits > current_weight_bits;
                    weight_eq = other_weight_bits == current_weight_bits;
                }
                int other_tid = tid ^ step;
                bool index_lt = other_index < current_index;
                bool cond = (tid < other_tid) ^ direction;
                bool swap = false;
                if constexpr (std::is_same_v<scalar_t, __half> || std::is_same_v<scalar_t, at::Half>) {
                    // 提前转换变量，代码更清晰
                    float val_other = __half2float((__half)other_weight_bits);
                    float val_curr  = __half2float((__half)current_weight_bits);
                    swap = (cond & (weight_gt | (weight_eq & index_lt))) |
                            (!cond & ((val_other < val_curr) | (weight_eq & (other_index > current_index))));
            
                    val = swap ? other_temp_val : val;
                    //swap = (cond & (weight_gt | (weight_eq & index_lt))) |
                    //        (!cond & (((__half)other_weight_bits < (__half)current_weight_bits) | (weight_eq & (other_index > current_index))));
                    //val = swap ? other_temp_val : val;
                } else if constexpr (std::is_same_v<scalar_t, __nv_bfloat16> || std::is_same_v<scalar_t, at::BFloat16>) {
                    swap = (cond & (weight_gt | (weight_eq & index_lt))) |
                            (!cond & ((other_weight_bits < current_weight_bits) | (weight_eq & (other_index > current_index))));
                    val = swap ? other_temp_val : val;
                } else {
                    swap = (cond & (weight_gt | (weight_eq & index_lt))) |
                            (!cond & ((other_weight_bits < current_weight_bits) | (weight_eq & (other_index > current_index))));
                    weight = swap ? other_weight_bits : weight;
                    key = swap ? other_index : key;
                }
            }
        }
        if constexpr (std::is_same_v<scalar_t, float>) {
            idx_and_weight[0] = weight;
            *((int32_t*)&idx_and_weight[1]) = key;
        } else {
            *(int32_t*)idx_and_weight = val;
        }
    }

    template<class scalar_t, int NUM_GROUPS = 8, int TOPK_GROUP=2, int TOPK=8>
    __device__ __forceinline__ int moe_topk_block_phase1(const scalar_t* w, const scalar_t* bias, scalar_t* global_score, int tid) {
        scalar_t idx_and_weight[2]; //Stores weight and index for further top1
        scalar_t idx_and_weight_2[2]; //used for top8 calculation

        float fw = scalar2float<scalar_t>(w[tid]);
        scalar_t fw_sim = float2scalar<scalar_t>(__builtin_rcpf((1.0f + __expf(-fw))));
        scalar_t fw_bf16 =  fw_sim + bias[tid];
        global_score[tid] = fw_sim;
        idx_and_weight[0] = fw_bf16;
        if constexpr (std::is_same_v<scalar_t, at::BFloat16> || std::is_same_v<scalar_t, at::Half>)
        {
            idx_and_weight[1] = float2scalar<scalar_t>(0.0f);
            *(int32_t*)idx_and_weight |= (tid << 16);
        }
        else { // for float
            *((int32_t*)&idx_and_weight[1]) = tid;
        }

        warpSortDescending<scalar_t, 32, 0xffffffff>(idx_and_weight, tid);

        scalar_t group_weight[2];

        //Eight groups
        __shared__ scalar_t shared_group_weights[8][2];
        //group weight are distributed in different groups, we need a shared memory to broadcast to all threads
        //save group_weight to shared memory
        scalar_t second_val = __shfl_sync(0x00000003, idx_and_weight[0], 1);

        if (tid % 32 == 0) {
            int group_idx = tid / 32;
            if constexpr (std::is_same_v<scalar_t, float>) {
                shared_group_weights[group_idx][0] = second_val + idx_and_weight[0];
                *((int32_t*)&shared_group_weights[group_idx][1]) = group_idx;
            } else {
                group_weight[0] = second_val + idx_and_weight[0];
                group_weight[1] = float2scalar<scalar_t>(0.0f); //reset the high 16 bits
                *(int32_t*)group_weight |= group_idx << 16;
                *((int32_t*)(shared_group_weights) + group_idx) = *(int32_t*)group_weight;
            }
        }
        __syncthreads();

        //get top 4
        // ~ 200 cycles
        scalar_t group_weight_for_sort[2] = {float2scalar<scalar_t>(0.0f), float2scalar<scalar_t>(0.0f)};
        //Move all the group weights into one warp so that we can do top4
        if (tid < 8) {
            if constexpr (std::is_same_v<scalar_t, float>) {
                group_weight_for_sort[0] = shared_group_weights[tid][0];
                *((int32_t*)&group_weight_for_sort[1]) = *((int32_t*)&shared_group_weights[tid][1]);

            } else {
                *(int32_t*)group_weight_for_sort = *((int32_t*)(shared_group_weights) + tid);
            }
        }
        warpSortDescending<scalar_t, 8 , 0x000000ff>(group_weight_for_sort, tid);

        // ~ 60 cycles
        __shared__ int shared_group_idex[8];
        uint32_t mask_group=0;

        if (tid < 4) {
            if constexpr (std::is_same_v<scalar_t, float>)
                mask_group |= (1 << (*((int32_t*)&group_weight_for_sort[1])));
            else
                mask_group |= (1 << (*(int32_t*)group_weight_for_sort >> 16));
        }

        if (tid < 8) {
            for (int i = 4; i > 0; i>>=1) {
                mask_group |= __shfl_xor_sync(0x0000000f, mask_group, i);
            }

            bool disabled = (mask_group & (1 << tid)) == 0;
            int store_pos0 = __builtin_mbcnt_lo(mask_group, 1);

            shared_group_idex[tid] = disabled ? 0 : store_pos0;
        }
        __syncthreads();
        int group_idx = shared_group_idex[tid / 32];

        __shared__ scalar_t shared_experts[32][2];

        if (tid % 32 < 8) {
            if (group_idx !=0) {
                if constexpr (std::is_same_v<scalar_t, float>) {
                    int new_idx = (group_idx-1)*8 + tid % 32;
                    shared_experts[new_idx][0] = idx_and_weight[0];
                    *((int32_t*)&shared_experts[new_idx][1]) = *((int32_t*)&idx_and_weight[1]);
                } else
                    *((int32_t*)(shared_experts) + (group_idx-1)*8 + tid % 32) = *(int32_t*)idx_and_weight;
            }
        }
        __syncthreads();

        if (tid < 32) {
            if constexpr (std::is_same_v<scalar_t, float>) {
                idx_and_weight_2[0] = shared_experts[tid][0];
                *((int32_t*)&idx_and_weight_2[1]) = *((int32_t*)&shared_experts[tid][1]);
            } else
                *(int32_t*)idx_and_weight_2 = *((int32_t*)shared_experts + tid);
        }

        warpSortDescending<scalar_t, 32 , 0xffffffff>(idx_and_weight_2, tid);

        //Now all values are stored into shared_max_experts
        int top_k_idx = 0;
        if constexpr (std::is_same_v<scalar_t, float>)
            top_k_idx = *((int32_t*)&idx_and_weight_2[1]);
        else
            top_k_idx = *((int32_t*)idx_and_weight_2) >> 16;
        return top_k_idx;
    }

    template<class scalar_t, int NUM_EXPERTS=384, int NUM_GROUPS = 1, int TOPK=8>
    __device__ __forceinline__ int moe_topk_1group_block_phase1(const scalar_t* w, const scalar_t* bias, scalar_t* global_score, int tid) {
        constexpr int WAVE_SIZE = 64;
        int wave_lane = tid % WAVE_SIZE;
        int wave_idx = tid / WAVE_SIZE;
        scalar_t idx_and_weight[2]; //Stores weight and index for further top1

        float fw = scalar2float<scalar_t>(w[tid]);
        scalar_t fw_sim = float2scalar<scalar_t>(__builtin_rcpf((1.0f + __expf(-fw))));
        scalar_t fw_bf16 = fw_sim + bias[tid];
        idx_and_weight[0] = fw_bf16;
        global_score[tid] = fw_sim;
        if constexpr (std::is_same_v<scalar_t, at::BFloat16> || std::is_same_v<scalar_t, at::Half>)
        {
            idx_and_weight[1] = float2scalar<scalar_t>(0.0f);
            *(int32_t*)idx_and_weight |= (tid << 16);
        }
        else { // for float
            *((int32_t*)&idx_and_weight[1]) = tid;
        }

        //Divide NUM_EXPERTS into NUM_EXPERTS/WAVE_SIZE groups
        //And do descending sort
        warpSortDescending<scalar_t, 64, 0xffffffffffffffff>(idx_and_weight, tid);

        __shared__ scalar_t max_cache[64][2];
        int offset = wave_lane + wave_idx * TOPK;
        if (wave_lane < TOPK) {
            if constexpr (std::is_same_v<scalar_t, float>) {
                max_cache[offset][0] = idx_and_weight[0];
                *((uint32_t*)&max_cache[offset][1]) = *((uint32_t*)&idx_and_weight[1]);
            }
            else
                ((float*)(max_cache))[offset] = *(float*)idx_and_weight;
        }
        // 使用天花板除法计算实际wave数量
        // 对于160专家: (160+64-1)/64 = 3, 3*8 = 24
        constexpr int num_waves = (NUM_EXPERTS + WAVE_SIZE - 1) / WAVE_SIZE;
        constexpr int topks_in_block = num_waves * TOPK;
        // must use small value to fill max_cache[topks_in_block~63]
        if (wave_idx == 0 && wave_lane >= topks_in_block) {
            if constexpr (std::is_same_v<scalar_t, float>) {
                max_cache[wave_lane][0] = 0.f;
                *((uint32_t*)&max_cache[wave_lane][1]) = tid;
            }
            else
                ((float*)(max_cache))[wave_lane] = *(float*)idx_and_weight;
        }
        __syncthreads();

        //We get NUM_EXPERTS/WAVE_SIZE*TOPK experts&weights
        //Sort NUM_EXPERTS/WAVE_SIZE*TOPK elements in 1 wave
        int top_k_idx = 0;
        if (wave_idx == 0) {
            if constexpr (std::is_same_v<scalar_t, float>) {
                idx_and_weight[0] = max_cache[wave_lane][0];
                *((uint32_t*)&idx_and_weight[1]) = *((uint32_t*)&max_cache[wave_lane][1]);
            }
            else
                *(float*)idx_and_weight = ((float*)(max_cache))[wave_lane];
            __syncthreads();
            warpSortDescending<scalar_t, 64, 0xffffffffffffffff>(idx_and_weight, tid);
            //Now all values are stored into shared_max_experts
            if constexpr (std::is_same_v<scalar_t, float>)
                top_k_idx = *((int32_t*)&idx_and_weight[1]);
            else
                top_k_idx = ((int32_t*)idx_and_weight)[tid] >> 16;
        }
        return top_k_idx;
    }

    template<class scalar_t, int TOPK=8>
    __device__ __forceinline__ void deepseek_topk_phase2(int top_k_idx, bool renormalize, scalar_t *global_score, int* topk_indices, float* topk_w, float scale_factor = 1.0f)
    {
        int tid = threadIdx.x;
        if (tid < TOPK) {
            topk_indices[tid] = top_k_idx;
            scalar_t top_k_sigmoid = global_score[top_k_idx];
            if (renormalize) {
                scalar_t top_k_sum = top_k_sigmoid;
                for (int offset = 4; offset > 0; offset >>= 1) {
                    top_k_sum += __shfl_xor_sync(0x000000ff, top_k_sum, offset);
                }
                top_k_sigmoid /= top_k_sum;
            }
            topk_w[tid] = scalar2float<scalar_t>(top_k_sigmoid) * scale_factor;
        }
    }

    template<class scalar_t, int NUM_SHARED_EXPERTS=0, int NUM_EXPERTS=256, int TOPK=8>
    __device__ __forceinline__ void sglang_topk_phase2(int top_k_idx, bool renormalize, scalar_t *global_score, int* topk_indices, float* topk_w, double scale_factor = 1.0f, int *shared_expert_ids=nullptr)
    {
        int tid = threadIdx.x;
        if(tid >= 32) return;
        int last_pos = TOPK;
        float top_k_sigmoid = scalar2float<scalar_t>(global_score[top_k_idx]);
        if(tid == (TOPK - 1)){
            if constexpr (NUM_SHARED_EXPERTS == 1)
                top_k_idx = NUM_EXPERTS;
            else if constexpr (NUM_SHARED_EXPERTS > 1)
                top_k_idx = shared_expert_ids[blockIdx.x];
        }
        if(tid < TOPK)
            topk_indices[tid] = top_k_idx;
        if constexpr (NUM_SHARED_EXPERTS > 0)
            last_pos = TOPK - 1;
        if(tid >= last_pos) // set w[TOPK-1~..]=0 when SHARE_EXPERTS>0, else set w[TOPK~..]=0
            top_k_sigmoid = 0.f;
        float top_k_sum = top_k_sigmoid;
        for (int offset = 8; offset > 0; offset >>= 1) {
            top_k_sum += __shfl_xor_sync(0x0000ffff, top_k_sum, offset);
        }

        if (tid < TOPK){
            if constexpr (NUM_SHARED_EXPERTS > 0) {
                if(tid == (TOPK - 1)) {
                    // 共享专家的权重 = (sum of routed weights) / routed_scaling_factor
                    // 修复：不要在这里对共享专家进行任何操作，直接设为0
                    // 后面会重新计算并设置正确的值
                    top_k_sigmoid = 0.f;
                } else {
                    // 路由专家：先归一化到sum=1.0
                    if (renormalize)
                        top_k_sigmoid = scalar2float<scalar_t>(float2scalar<scalar_t>(top_k_sigmoid) / float2scalar<scalar_t>(top_k_sum));
                }
            } else {
                // 无共享专家：直接归一化
                if (renormalize)
                    top_k_sigmoid = scalar2float<scalar_t>(float2scalar<scalar_t>(top_k_sigmoid) / float2scalar<scalar_t>(top_k_sum));
            }
            topk_w[tid] = top_k_sigmoid;
        }

        // 共享专家的权重计算和归一化逻辑
        //   1. 归一化路由专家（不包括共享专家）
        //   2. 共享专家权重 = sum(routed) / routed_scaling_factor
        //   3. 不再进行第二次归一化！
        //
        //   - 归一化基数只包含路由专家，不包含共享专家
        //   - 共享专家的权重 = normalized_sum / routed_scaling_factor
        //   - 最终权重和 = 1.0 (routed) + shared_weight（可能>1.0）
        if constexpr (NUM_SHARED_EXPERTS > 0) {
            // 第一步：重新计算归一化后的路由专家和（应该等于1.0）
            if (tid < TOPK) {
                top_k_sum = topk_w[tid];
            } else {
                top_k_sum = 0.f;
            }
            for (int offset = 8; offset > 0; offset >>= 1) {
                top_k_sum += __shfl_xor_sync(0x0000ffff, top_k_sum, offset);
            }

            // 第二步：设置共享专家的权重 = normalized_sum / routed_scaling_factor
            // 注意：这里不再进行第二次归一化！
            if (tid == (TOPK - 1) && renormalize) {
                topk_w[tid] = top_k_sum * scale_factor;  // scale_factor = 1.0 / routed_scaling_factor
            }
        }
    }

    template<typename scalar_t, int NUM_EXPERTS=256, int NUM_GROUPS=8, int TOPK_GROUP=2, int TOPK=8>
    __global__ void fused_deepseek_topk(const scalar_t* w, const scalar_t* bias, float* topk_w, int32_t* topk_indices, bool renormalize, float scale_factor = 1.0f) {
        int tid = threadIdx.x;
        int bdx = blockIdx.x;
        if constexpr (std::is_same_v<scalar_t, double>)
        {
            if((tid == 0)&&(bdx == 0))
                printf("%s unsupported double type.\n", __func__);
            return;
        }
        __shared__ scalar_t global_score[NUM_EXPERTS];
        int top_k_idx = 0;
        if constexpr (NUM_GROUPS == 1) {
            top_k_idx = moe_topk_1group_block_phase1<scalar_t, NUM_EXPERTS, 1, TOPK>(w + bdx * NUM_EXPERTS, bias, global_score, tid);
        } else {
            top_k_idx = moe_topk_block_phase1<scalar_t>(w + bdx * NUM_EXPERTS, bias, global_score, tid);
        }
        deepseek_topk_phase2(top_k_idx, renormalize, global_score, topk_indices + bdx * TOPK, topk_w + bdx * TOPK, scale_factor);
    }

    template<typename scalar_t, int NUM_SHARED_EXPERTS=0, int NUM_EXPERTS=256, int NUM_GROUPS=8, int TOPK_GROUP=4, int TOPK=8>
    __global__ void fused_topk(const scalar_t* w, const scalar_t* bias, float* topk_w, int32_t* topk_indices, bool renormalize, double scale_factor = 1.0f, int *shared_expert_ids = nullptr) {
        int tid = threadIdx.x;
        int bdx = blockIdx.x;
        if constexpr (std::is_same_v<scalar_t, double>)
        {
            if((tid == 0)&&(bdx == 0))
                printf("%s unsupported double type.\n", __func__);
            return;
        }
        __shared__ scalar_t global_score[NUM_EXPERTS];
        int top_k_idx = 0;
        if constexpr (NUM_GROUPS == 1) {
            top_k_idx = moe_topk_1group_block_phase1<scalar_t, NUM_EXPERTS, 1, TOPK>(w + bdx * NUM_EXPERTS, bias, global_score, tid);
        } else {
            top_k_idx = moe_topk_block_phase1<scalar_t>(w + bdx * NUM_EXPERTS, bias, global_score, tid);
        }
        sglang_topk_phase2<scalar_t, NUM_SHARED_EXPERTS, NUM_EXPERTS, TOPK>(top_k_idx, renormalize, global_score, topk_indices + bdx * TOPK, topk_w + bdx * TOPK, scale_factor, shared_expert_ids);

    }

}
