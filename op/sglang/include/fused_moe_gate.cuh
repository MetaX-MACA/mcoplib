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
    }else if constexpr (std::is_same_v<scalar_t, float> ) {
        return val;
    } else {
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
    } else if constexpr (std::is_same_v<scalar_t, float> ) {
        return val;
    } 
    else {
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
    __device__ __forceinline__ int moe_topk_block_phase1(const scalar_t* w, const scalar_t* bias, float* global_score, int tid) {
        // FP32-precise grouped (num_expert_group>1) TopK: the sigmoid gate, the
        // (score+bias) sort key, the group score (top1+top2 sum), all three warp
        // sorts (intra-group / top-group / final) and the stored gate weight are
        // ALL computed and compared in fp32, independent of the input dtype.
        // scalar_t is only used to load w/bias. Doing this in bf16 rounded the
        // sort keys / group scores and flipped expert (and group) selection on
        // near-ties — the same failure class as the single-group GLM path.
        // Layout: idx_and_weight[0] = weight (fp32), idx_and_weight[1] = index (int32).
        float idx_and_weight[2];   //Stores weight and index for further top1
        float idx_and_weight_2[2]; //used for top8 calculation

        float fw = scalar2float<scalar_t>(w[tid]);
        float fw_sim = __builtin_rcpf((1.0f + __expf(-fw)));           // sigmoid in fp32
        float key = fw_sim + scalar2float<scalar_t>(bias[tid]);        // (score + bias) in fp32
        global_score[tid] = fw_sim;   // store fp32 gate weight for phase2
        idx_and_weight[0] = key;
        *((int32_t*)&idx_and_weight[1]) = tid;

        warpSortDescending<float, 32, 0xffffffff>(idx_and_weight, tid);

        float group_weight[2];

        //Eight groups
        __shared__ float shared_group_weights[8][2];
        //group weight are distributed in different groups, we need a shared memory to broadcast to all threads
        //save group_weight to shared memory
        float second_val = __shfl_sync(0x00000003, idx_and_weight[0], 1); // top-2 weight (fp32)

        if (tid % 32 == 0) {
            int group_idx = tid / 32;
            shared_group_weights[group_idx][0] = second_val + idx_and_weight[0]; // top1+top2 in fp32
            *((int32_t*)&shared_group_weights[group_idx][1]) = group_idx;
        }
        __syncthreads();

        //get top 4
        // ~ 200 cycles
        float group_weight_for_sort[2] = {-INFINITY, 0.0f};
        //Move all the group weights into one warp so that we can do top4
        if (tid < 8) {
            group_weight_for_sort[0] = shared_group_weights[tid][0];
            *((int32_t*)&group_weight_for_sort[1]) = *((int32_t*)&shared_group_weights[tid][1]);
        }
        warpSortDescending<float, 8 , 0x000000ff>(group_weight_for_sort, tid);

        // ~ 60 cycles
        __shared__ int shared_group_idex[8];
        uint32_t mask_group=0;

        if (tid < 4) {
            mask_group |= (1 << (*((int32_t*)&group_weight_for_sort[1])));
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

        __shared__ float shared_experts[32][2];

        if (tid % 32 < 8) {
            if (group_idx !=0) {
                int new_idx = (group_idx-1)*8 + tid % 32;
                shared_experts[new_idx][0] = idx_and_weight[0];
                *((int32_t*)&shared_experts[new_idx][1]) = *((int32_t*)&idx_and_weight[1]);
            }
        }
        __syncthreads();

        if (tid < 32) {
            idx_and_weight_2[0] = shared_experts[tid][0];
            *((int32_t*)&idx_and_weight_2[1]) = *((int32_t*)&shared_experts[tid][1]);
        }

        warpSortDescending<float, 32 , 0xffffffff>(idx_and_weight_2, tid);

        //Now all values are stored into shared_max_experts
        int top_k_idx = *((int32_t*)&idx_and_weight_2[1]);
        return top_k_idx;
    }

    template<class scalar_t, int NUM_EXPERTS=384, int NUM_GROUPS = 1, int TOPK=8>
    __device__ __forceinline__ int moe_topk_1group_block_phase1(const scalar_t* w, const scalar_t* bias, float* global_score, int tid) {
        // FP32-precise TopK: the sigmoid gate, the (score + bias) sort key, the
        // warp sort comparison and the stored gate weight are ALL computed and
        // compared in fp32, regardless of the input dtype (bf16/half/float).
        // Selecting experts on bf16-rounded keys flipped the top-k boundary
        // (8th vs 9th expert on near-ties) and caused a ~3% accuracy drop on
        // GLM/ceval; keeping the whole gate in fp32 removes that flip and
        // matches the upstream sglang single-group fp32 kernel.
        constexpr int WAVE_SIZE = 64;
        int wave_lane = tid % WAVE_SIZE;
        int wave_idx = tid / WAVE_SIZE;
        // Always sort in fp32: idx_and_weight[0] = weight (fp32),
        //                      idx_and_weight[1] = index reinterpreted as int32.
        float idx_and_weight[2];

        if (tid < NUM_EXPERTS) {
            float fw = scalar2float<scalar_t>(w[tid]);
            float fw_sim = __builtin_rcpf((1.0f + __expf(-fw)));   // sigmoid in fp32, kept in fp32
            float key = fw_sim + scalar2float<scalar_t>(bias[tid]); // (score + bias) in fp32
            idx_and_weight[0] = key;
            global_score[tid] = fw_sim; // store the fp32 gate weight (no bias)
        } else {
            // padding threads sort to the end
            idx_and_weight[0] = -INFINITY;
        }
        *((int32_t*)&idx_and_weight[1]) = tid;

        //Divide NUM_EXPERTS into NUM_EXPERTS/WAVE_SIZE groups
        //And do descending sort (fp32)
        warpSortDescending<float, 64, 0xffffffffffffffff>(idx_and_weight, tid);

        // candidate buffer: [weight_f32, index_i32] pairs
        __shared__ float max_cache[64][2];
        int offset = wave_lane + wave_idx * TOPK;
        if (wave_lane < TOPK ) {
            max_cache[offset][0] = idx_and_weight[0];
            *((int32_t*)&max_cache[offset][1]) = *((int32_t*)&idx_and_weight[1]);
        }

        // 对于160专家: (160+64-1)/64 = 3, 3*8 = 24
        constexpr int num_waves = (NUM_EXPERTS + WAVE_SIZE - 1) / WAVE_SIZE;
        constexpr int topks_in_block = num_waves * TOPK;
        // must use small value to fill max_cache[topks_in_block~63]
        if (wave_idx == 0 && wave_lane >= topks_in_block) {
            max_cache[wave_lane][0] = -INFINITY;
            *((int32_t*)&max_cache[wave_lane][1]) = wave_lane;
        }
        __syncthreads();

        //We get NUM_EXPERTS/WAVE_SIZE*TOPK experts&weights
        //Sort NUM_EXPERTS/WAVE_SIZE*TOPK elements in 1 wave
        int top_k_idx = -1;
        if (wave_idx == 0) {
            idx_and_weight[0] = max_cache[wave_lane][0];
            *((int32_t*)&idx_and_weight[1]) = *((int32_t*)&max_cache[wave_lane][1]);
            __syncthreads();
            warpSortDescending<float, 64, 0xffffffffffffffff>(idx_and_weight, tid);
            top_k_idx = *((int32_t*)&idx_and_weight[1]);
        }
        return top_k_idx;
    }

    template<class scalar_t, int TOPK=8>
    __device__ __forceinline__ void deepseek_topk_phase2(int top_k_idx, bool renormalize, float *global_score, int* topk_indices, float* topk_w, float scale_factor = 1.0f)
    {
        int tid = threadIdx.x;
        if (tid < TOPK) {
            topk_indices[tid] = top_k_idx;
            float top_k_sigmoid = global_score[top_k_idx];   // fp32 gate weight
            if (renormalize) {
                float top_k_sum = top_k_sigmoid;
                for (int offset = 4; offset > 0; offset >>= 1) {
                    top_k_sum += __shfl_xor_sync(0x000000ff, top_k_sum, offset);
                }
                top_k_sigmoid /= top_k_sum;                  // fp32 renorm
            }
            topk_w[tid] = top_k_sigmoid * scale_factor;
        }
    }

    template<class scalar_t, int NUM_SHARED_EXPERTS=0, int NUM_EXPERTS=256, int TOPK=8>
    __device__ __forceinline__ void sglang_topk_phase2(int top_k_idx, bool renormalize, float *global_score, int* topk_indices, float* topk_w, double scale_factor = 1.0f, int *shared_expert_ids=nullptr)
    {
        int tid = threadIdx.x;
        if(tid >= 32) return;
        int last_pos = TOPK;
        float top_k_sigmoid = global_score[top_k_idx];   // fp32 gate weight
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
                if(tid == (TOPK - 1))
                    top_k_sigmoid = top_k_sum * (float)scale_factor;   // fp32
            }
            if (renormalize)
                top_k_sigmoid = top_k_sigmoid / top_k_sum;             // fp32 renorm
            topk_w[tid] = top_k_sigmoid;
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
        __shared__ float global_score[NUM_EXPERTS];
        int top_k_idx = 0;
        if constexpr (NUM_GROUPS == 1) {
            top_k_idx = moe_topk_1group_block_phase1<scalar_t, NUM_EXPERTS, 1, TOPK>(w + bdx * NUM_EXPERTS, bias, global_score, tid);

        } else {
            top_k_idx = moe_topk_block_phase1<scalar_t>(w + bdx * NUM_EXPERTS, bias, global_score, tid);
        }
        deepseek_topk_phase2(top_k_idx, renormalize, global_score, topk_indices + bdx * TOPK, topk_w + bdx * TOPK, scale_factor);
    }

    // Constexpr next-power-of-two (>= n). Used to size the candidate buffer
    // in moe_topk_1group_block_phase1_generic. A simple constexpr loop avoids
    // __clz/__builtin_clz, which are not constexpr on MACA.
    constexpr int next_pow2(int n) {
        int p = 1;
        while (p < n) p <<= 1;
        return p;
    }

    // Generic single-group TopK phase1 for arbitrary NUM_EXPERTS (up to ~1024)
    // and TOPK in [1, 16]. Used when num_waves * TOPK exceeds the legacy
    // max_cache[64][2] capacity (e.g. NUM_EXPERTS=896, TOPK=16 -> 240 candidates).
    //
    // Algorithm (3 stages), all in shared memory, no atomics:
    //
    //   Stage A — per-wave local sort:
    //     Each 64-thread wave sorts its 64 (or fewer for the last wave)
    //     elements via warpSortDescending<64>. Each wave stores its top TOPK
    //     candidates (packed weight+index) into its private slice of cand_buf.
    //     Non-existent waves (when num_waves is not a power of two) leave their
    //     slices as -inf, set by a single thread before the sync.
    //
    //   Stage B — sequential merge reduction:
    //     Run 0's TOPK slice is the running "accumulator". For each subsequent
    //     run r in 1..num_waves-1, wave 0 loads the 2*TOPK elements (acc + run r)
    //     into per-lane registers, sorts them with warpSortDescending<2*TOPK>
    //     (lanes 0..2*TOPK-1 carry data, the rest carry -inf), and stores the
    //     top TOPK back to the accumulator slice. After all merges, acc holds
    //     the global top TOPK. O(num_waves) merges, each O(TOPK) work.
    //
    //   Stage C — readout: wave 0's lanes 0..TOPK-1 hold the global top TOPK
    //     after the final sort; return the per-lane top index so phase2 can
    //     scatter per-thread results.
    //
    // Shared memory: cand_buf[MAX_RUNS * TOPK] int32 (one packed slot per
    // candidate for bf16/half; two int32 for float). MAX_RUNS is the next power
    // of two >= num_waves to keep slicing simple; non-existent runs are -inf.
    // For NUM_EXPERTS<=1024, TOPK<=16: MAX_RUNS<=32 -> 32*16*8B = 4KB max.
    template<class scalar_t, int NUM_EXPERTS=896, int TOPK=16>
    __device__ __forceinline__ int moe_topk_1group_block_phase1_generic(const scalar_t* w, const scalar_t* bias, float* global_score, int tid) {
        constexpr int WAVE_SIZE = 64;
        constexpr int num_waves = (NUM_EXPERTS + WAVE_SIZE - 1) / WAVE_SIZE;
        // Next power of two >= num_waves. Simplifies slicing (each wave's slice
        // is a clean TOPK run; extra waves stay -inf). Uses constexpr helper
        // since __clz is not constexpr on MACA.
        constexpr int MAX_RUNS = next_pow2(num_waves);
        // FP32-precise: the gate, the (score+bias) sort key, the warp sort
        // comparison and the stored gate weight are ALL fp32, independent of the
        // input dtype. The candidate buffer holds [weight_f32, index_i32] pairs
        // (2 int32 slots per candidate). scalar_t is only used to load w/bias.
        constexpr int kSlotsPerCand = 2;

        __shared__ int32_t cand_buf[MAX_RUNS * TOPK * kSlotsPerCand];

        int wave_lane = tid % WAVE_SIZE;
        int wave_idx = tid / WAVE_SIZE;

        // ---- Stage A: per-wave local sort + store top TOPK ----
        float idx_and_weight[2];
        if (tid < NUM_EXPERTS) {
            float fw = scalar2float<scalar_t>(w[tid]);
            float fw_sim = __builtin_rcpf((1.0f + __expf(-fw)));   // sigmoid in fp32
            idx_and_weight[0] = fw_sim + scalar2float<scalar_t>(bias[tid]); // key in fp32
            global_score[tid] = fw_sim;                            // fp32 gate weight
        } else {
            idx_and_weight[0] = -INFINITY;
        }
        *((int32_t*)&idx_and_weight[1]) = tid;

        warpSortDescending<float, 64, 0xffffffffffffffff>(idx_and_weight, tid);

        // Each wave stores its top TOPK (lanes 0..TOPK-1) into its slice.
        if (wave_idx < num_waves && wave_lane < TOPK) {
            int pos = (wave_idx * TOPK + wave_lane) * kSlotsPerCand;
            cand_buf[pos + 0] = *(int32_t*)&idx_and_weight[0];
            cand_buf[pos + 1] = *(int32_t*)&idx_and_weight[1];
        }
        // Pad non-existent runs (num_waves..MAX_RUNS) with -inf so the merge
        // step sees full TOPK runs everywhere. Done by a few threads.
        if (tid == 0) {
            float neg_inf = -INFINITY;
            int32_t packed_neg_inf = *(int32_t*)&neg_inf;
            for (int run = num_waves; run < MAX_RUNS; ++run) {
                for (int j = 0; j < TOPK; ++j) {
                    int pos = (run * TOPK + j) * kSlotsPerCand;
                    cand_buf[pos + 0] = packed_neg_inf;
                    cand_buf[pos + 1] = 0;
                }
            }
        }
        __syncthreads();

        // ---- Stage B: sequential merge of runs 1..num_waves-1 into run 0 ----
        // Wave 0 does all merges (sequential). Each merge reads 2*TOPK elements
        // (run 0 + run r), sorts, writes top TOPK back to run 0.
        if (wave_idx == 0) {
            float iw[2];
            for (int r = 1; r < MAX_RUNS; ++r) {
                // Load 2*TOPK elements into per-lane registers. Lanes 0..2*TOPK-1
                // carry data; the rest are -inf so they sort to the end.
                if (wave_lane < 2 * TOPK) {
                    int pos;
                    bool use_a = (wave_lane < TOPK);
                    int slot = use_a ? wave_lane : (wave_lane - TOPK);
                    int run = use_a ? 0 : r;
                    pos = (run * TOPK + slot) * kSlotsPerCand;
                    iw[0] = *(float*)&cand_buf[pos + 0];
                    *((int32_t*)&iw[1]) = cand_buf[pos + 1];
                } else {
                    iw[0] = -INFINITY;
                    *((int32_t*)&iw[1]) = wave_lane;
                }
                // Sort 2*TOPK lanes (the rest are -inf, harmlessly sorted to end).
                // Use a mask covering 2*TOPK lanes; if 2*TOPK>=64 use full mask.
                constexpr uint64_t kMergeMask = (2 * TOPK >= 64) ? 0xffffffffffffffffULL
                                                                 : ((1ULL << (2 * TOPK)) - 1ULL);
                warpSortDescending<float, 2 * TOPK, kMergeMask>(iw, wave_lane);
                // Lanes 0..TOPK-1 hold the merged top TOPK. Store back to run 0.
                if (wave_lane < TOPK) {
                    int pos = wave_lane * kSlotsPerCand;
                    cand_buf[pos + 0] = *(int32_t*)&iw[0];
                    cand_buf[pos + 1] = *(int32_t*)&iw[1];
                }
                // No sync needed: only wave 0 touches run 0, and the next merge
                // iteration also runs on wave 0 — same threads, no inter-warp hazard.
            }

            // ---- Stage C: readout per-lane top index for phase2 ----
            // phase2 reads global_score[top_k_idx] for ALL lanes 0..31 (even
            // lanes >= TOPK, which it later zeros before the reduction). So we
            // MUST return a valid in-range index for every lane in wave 0 —
            // returning -1 would cause global_score[-1] -> OOB. For lanes
            // >= TOPK we wrap around to a valid stored candidate (its value is
            // irrelevant since phase2 zeros these lanes).
            int valid_lane = wave_lane % TOPK;
            int pos = valid_lane * kSlotsPerCand;
            int32_t v = cand_buf[pos + 1];
            return v;
        }
        return 0;  // other waves: harmless (phase2 returns early for tid>=32)
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
        __shared__ float global_score[NUM_EXPERTS];
        int top_k_idx = 0;
        if constexpr (NUM_GROUPS == 1) {
            // Use the generic phase1 when the legacy max_cache[64][2] (64-slot)
            // would overflow: num_waves * TOPK > 64. Otherwise keep the legacy
            // path for parity with existing supported configs.
            constexpr int WAVE_SIZE = 64;
            constexpr int num_waves = (NUM_EXPERTS + WAVE_SIZE - 1) / WAVE_SIZE;
            if constexpr (num_waves * TOPK > 64) {
                top_k_idx = moe_topk_1group_block_phase1_generic<scalar_t, NUM_EXPERTS, TOPK>(w + bdx * NUM_EXPERTS, bias, global_score, tid);
            } else {
                top_k_idx = moe_topk_1group_block_phase1<scalar_t, NUM_EXPERTS, 1, TOPK>(w + bdx * NUM_EXPERTS, bias, global_score, tid);
            }
            // if(bdx == 0) {
            //     printf("fused_topk Info: tid=%d, top_k_idx =%d, \n", tid, top_k_idx);
            // }
        } else {
            top_k_idx = moe_topk_block_phase1<scalar_t>(w + bdx * NUM_EXPERTS, bias, global_score, tid);
        }
        sglang_topk_phase2<scalar_t, NUM_SHARED_EXPERTS, NUM_EXPERTS, TOPK>(top_k_idx, renormalize, global_score, topk_indices + bdx * TOPK, topk_w + bdx * TOPK, scale_factor, shared_expert_ids);

    }

}
