#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include "../kernel/utils.h"
#include "../include/router_bias_topk.h"

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

template<typename Type>
__device__ Type getmin() {
    return (Type)(-3.40e38);
}

template<>
__device__ half getmin() {
    return half(-65504);
}

template<>
__device__ maca_bfloat16 getmin() {
    return maca_bfloat16(-3.40e38);
}

__device__ __forceinline__ float get_weight(const int64_t& v) {
    const float* idx_and_weight = (const float*)&v;
    return idx_and_weight[0];
}

template<uint64_t MASK=0xffffffffffffffff>
__device__ __forceinline__ void warpSortDescendingUpdate(float (&idx_and_weight)[2], int tid) {

    int64_t val = *(int64_t*)idx_and_weight;
    for (int width = 2; width < 64; width <<=1 ) {
        for (int step = width >> 1; step > 0; step >>=1) {
            const bool direction = ((tid & width) == 0);
            int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
            int other_tid = tid ^ step;

            float current_weight_bits = get_weight(val);
            float other_weight_bits = get_weight(other_temp_val);
            int current_index = val >> 32;
            int other_index = other_temp_val >> 32;
            bool weight_gt, weight_eq, index_lt;
            weight_gt = other_weight_bits > current_weight_bits;
            weight_eq = other_weight_bits == current_weight_bits;
            index_lt = other_index < current_index;

            bool other_is_big = weight_gt | (weight_eq & index_lt);
            bool swap = (tid < other_tid) ^ (other_is_big) ^ (direction);
            val = swap ? other_temp_val : val;
        }
    }
    for (int step = 32; step > 0; step >>= 1) {
        int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
        int other_tid = tid ^ step;

        float current_weight_bits = get_weight(val);
        float other_weight_bits = get_weight(other_temp_val);
        int current_index = val >> 32;
        int other_index = other_temp_val >> 32;
        bool weight_gt, weight_eq, index_lt;
        
        weight_gt = other_weight_bits > current_weight_bits;
        weight_eq = other_weight_bits == current_weight_bits;
        index_lt = other_index < current_index;

        bool other_is_big = weight_gt | (weight_eq & index_lt);
        bool swap = (tid < other_tid) ^ (!other_is_big);
        val = swap ? other_temp_val : val;
    }
    *(int64_t*)idx_and_weight = val;
}

__device__ __forceinline__ void SortDescendingUpdate(int64_t * sm, int length, int tid) {
    __syncthreads();
    for(int width = 2; width <= length; width <<= 1) {
        for(int step = width >> 1; step > 0; step >>= 1) {
            const bool direction = ((tid & width) == 0);
            int other_tid = tid ^ step;
            if(tid < other_tid) {
                int64_t val = sm[tid];
                int64_t other_temp_val = sm[other_tid];
                float a_weight_bits = get_weight(val);
                int a_index = val >> 32;
                float b_weight_bits = get_weight(other_temp_val);
                int b_index = other_temp_val >> 32;
                bool eq_weight = a_weight_bits == b_weight_bits;
                bool less_weight;
                less_weight = a_weight_bits < b_weight_bits;
                
                bool large_index = a_index > b_index;
                bool swap = (((eq_weight&large_index) | less_weight) & direction) | ((!direction) && ((eq_weight&&(!large_index)) | ((!less_weight) && (!eq_weight))));
                // bool swap = direction ? isLess<dir>(val, other_temp_val) : isGreater<dir>(val, other_temp_val);
                if(swap) {
                    sm[tid] = other_temp_val;
                    sm[other_tid] = val;
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ bool __devicecheck_nan(float x)        { return isnan(x); }
__device__ __forceinline__ bool __devicecheck_nan(__half x)       { return __hisnan(x); }
__device__ __forceinline__ bool __devicecheck_nan(__nv_bfloat16 x){ return __hisnan(x); }

constexpr uint64_t MASK64 = 0xffffffffffffffffull;   // MACA 64 路 wave 掩码

template <typename T>
struct alignas(16) Vec16 {
    T v[16 / sizeof(T)];   // 元素个数 = 16B / 单个元素大小，编译期确定
};

// 键序：(weight 降序, index 升序) —— 与原版 tie-break 一致
__device__ __forceinline__ bool key_greater(int64_t a, int64_t b) {
    const float wa = get_weight(a);
    const float wb = get_weight(b);
    return (wa != wb) ? (wa > wb) : ((a >> 32) < (b >> 32));
}

__device__ __forceinline__ int64_t pack_key(float w, int idx) {
    float kv[2] = {w, 0.0f};
    return *(int64_t*)kv | ((int64_t)idx << 32);
}

// 8 元素寄存器内 bitonic 降序排序（全部静态索引，不产生 local memory）
__device__ __forceinline__ void bitonic_sort8_desc(int64_t v[8]) {
    #pragma unroll
    for (int width = 2; width <= 8; width <<= 1) {
        #pragma unroll
        for (int step = width >> 1; step > 0; step >>= 1) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                if ((i & step) == 0) {            // 每对由低位 i 执行
                    const int j = i ^ step;
                    const bool up = ((i & width) == 0);   // 降序网络：大者放低位
                    if (key_greater(v[j], v[i]) == up) {
                        const int64_t t = v[i]; v[i] = v[j]; v[j] = t;
                    }
                }
            }
        }
    }
}

// 与距离 dist 的伙伴 lane 归并：小号 lane 保留 top-8 of 16，大号 lane 保留 bottom-8（不再使用）
__device__ __forceinline__ void merge_top8(int64_t v[8], int lane, int dist) {
    int64_t other[8];                              // 必须先整体快照伙伴列表，边改边读会读到已更新的值
    #pragma unroll
    for (int e = 0; e < 8; ++e) other[e] = __shfl_xor_sync(MASK64, v[e], dist);

    const bool keep_top = ((lane & dist) == 0);    // 本对中较小的 lane
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int64_t b = other[7 - i];            // 伙伴列表反序配对：max(A[i], B[7-i])
        if (key_greater(b, v[i]) == keep_top) v[i] = b;
    }
    // 此时 v 是 bitonic 序列，3 步寄存器 bitonic 归并恢复降序
    #pragma unroll
    for (int s = 4; s > 0; s >>= 1) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((i & s) == 0) {
                const int j = i ^ s;
                if (key_greater(v[j], v[i])) { const int64_t t = v[i]; v[i] = v[j]; v[j] = t; }
            }
        }
    }
}

// float 位型 → 无符号保序（负数也正确；只处理非负权重时可简化为 f | 0x80000000）
__device__ __forceinline__ uint32_t f32_to_ord(uint32_t f) {
    return f ^ (((int32_t)f >> 31) | 0x80000000u);
}
__device__ __forceinline__ uint32_t ord_to_f32(uint32_t g) {   // 逆变换（出口用一次）
    return g ^ ((g & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu);
}

// (weight 低32, index 高32) → 单 key：index 取反折叠，使"u64 大" = "权重高/同权重 index 小"
__device__ __forceinline__ uint64_t fold_key(int64_t v) {
    return ((uint64_t)f32_to_ord((uint32_t)v) << 32)
         | (uint32_t)~(uint32_t)(((uint64_t)v) >> 32);
}
__device__ __forceinline__ int64_t unfold_key(uint64_t k) {
    return (int64_t)ord_to_f32((uint32_t)(k >> 32)) | ((int64_t)(uint32_t)~(uint32_t)k << 32);
}

template <uint64_t MASK = 0xffffffffffffffff>
__device__ __forceinline__ void warpSortDescending(int64_t& idx_and_weight, int tid) {
    uint64_t key = fold_key(idx_and_weight);              // tid 必须是 warp 内 lane 号(0-63)

#pragma unroll
    for (int width = 2; width <= 64; width <<= 1) {
        const bool down = (tid & width) == 0;             // 内层不变量，外提
#pragma unroll
        for (int step = width >> 1; step > 0; step >>= 1) {
            const uint64_t o = __shfl_xor_sync(MASK, key, step);
            const bool low = (tid & step) == 0;           // 等价 tid < tid^step，省一条
            if ((o > key) == (low == down)) key = o;      // 赢家留下，一次 u64 比较
        }
    }
    idx_and_weight = unfold_key(key);
}

__device__ __forceinline__ void GroupSelectTop8Keyed(int64_t& key) {
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    const int lane = threadIdx.x & 15;
  /* mask/lane 同上 */
  auto greater = [](int64_t a, int64_t b) {
    const float wa = __int_as_float((int32_t)a), wb = __int_as_float((int32_t)b);
    return wa != wb ? wa > wb : (a >> 32) < (b >> 32);   // 同值 index 小者优先
  };
  #pragma unroll
  for (int w = 2; w <= 8; w <<= 1)
    #pragma unroll
    for (int s = w >> 1; s > 0; s >>= 1) {
      const int64_t o = __shfl_xor_sync(mask, key, s, 16);
      const bool low = ((lane & s) == 0), down = ((lane & w) == 0);
      key = greater(key, o) == (low == down) ? key : o;
    }
  const int64_t o = __shfl_xor_sync(mask, key, 8, 16);
  key = greater(key, o) == ((lane & 8) == 0) ? key : o;
}

// topk 特化为 8；容量 64 lane × 8 slot = 512，要求 col_num <= 512（host 已检查）
template <typename IndType, typename DataType, bool Renormalize, bool VEC_IO>
__global__ void router_bias_warp_top8_kernel(const DataType* __restrict__ gating_output,
        const DataType* __restrict__ router_bias, DataType* __restrict__ output, const int num_rows,
        IndType* __restrict__ indices, const int col_num, const bool check_nan,
        const int nan_row_i_out, const float routed_scaling_factor)
{
    constexpr int TOPK = 8;
    const int lane = threadIdx.x;                    // 64 路 wave 内 lane
    const int row  = blockIdx.x * TOPK + threadIdx.y;// 每 warp 一行，每 block 8 行
    if (row >= num_rows) return;                     // warp 级一致退出（kernel 内没有任何 barrier，安全）

    const int col0 = lane * TOPK;
    int64_t v[TOPK];
    bool has_nan = false;

    if (VEC_IO) {                                    // col_num % 8 == 0 且指针 16B 对齐
        if (col0 < col_num) {                        // 8 个元素要么全有效要么全越界
            float w[TOPK];
            if constexpr (sizeof(DataType) == 4) {   // float：两条 float4
                const float4 g0 = *reinterpret_cast<const float4*>(gating_output + (size_t)row * col_num + col0);
                const float4 g1 = *reinterpret_cast<const float4*>(gating_output + (size_t)row * col_num + col0 + 4);
                const float4 b0 = *reinterpret_cast<const float4*>(router_bias + col0);
                const float4 b1 = *reinterpret_cast<const float4*>(router_bias + col0 + 4);
                const float gx[TOPK] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
                const float gb[TOPK] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
                #pragma unroll
                for (int e = 0; e < TOPK; ++e) {
                    w[e] = __builtin_mxc_rcpf(1.0f + __builtin_expf(-gx[e])) + gb[e];
                    if (check_nan && __devicecheck_nan(gx[e])) has_nan = true;
                }
            } else {                                 // half/bf16：一条 16B 向量
                const Vec16<DataType> g = *reinterpret_cast<const Vec16<DataType>*>(gating_output + (size_t)row * col_num + col0);
                const Vec16<DataType> b = *reinterpret_cast<const Vec16<DataType>*>(router_bias + col0);
                #pragma unroll
                for (int e = 0; e < TOPK; ++e) {
                    const float x = (float)g.v[e];
                    w[e] = __builtin_mxc_rcpf(1.0f + __builtin_expf(-x)) + (float)b.v[e];
                    if (check_nan && __devicecheck_nan(x)) has_nan = true;
                }
            }
            #pragma unroll
            for (int e = 0; e < TOPK; ++e) v[e] = pack_key(w[e], col0 + e);
        } else {
            const float pmin = (float)getmin<DataType>();
            #pragma unroll
            for (int e = 0; e < TOPK; ++e) v[e] = pack_key(pmin, col0 + e);
        }
    } else {                                         // 标量回退：任意 col_num / 未对齐
        const float pmin = (float)getmin<DataType>();
        #pragma unroll
        for (int e = 0; e < TOPK; ++e) {
            const int col = col0 + e;
            if (col < col_num) {
                const float x = (float)gating_output[(size_t)row * col_num + col];
                v[e] = pack_key(__builtin_mxc_rcpf(1.0f + __builtin_expf(-x)) + (float)router_bias[col], col);
                if (check_nan && __devicecheck_nan(x)) has_nan = true;
            } else {
                v[e] = pack_key(pmin, col);
            }
        }
    }

    if (check_nan && __any_sync(MASK64, has_nan)) {  // 64 位掩码不可用时换 __ballot_sync(MASK64,has_nan)!=0
        if (lane < TOPK) {
            output[TOPK * row + lane] = DataType(0);
            indices[TOPK * row + lane] = nan_row_i_out;
        }
        return;
    }

    GroupSelectTop8Keyed(v[0]);
    GroupSelectTop8Keyed(v[1]);
    GroupSelectTop8Keyed(v[2]);
    GroupSelectTop8Keyed(v[3]);
    GroupSelectTop8Keyed(v[4]);
    GroupSelectTop8Keyed(v[5]);
    GroupSelectTop8Keyed(v[6]);
    GroupSelectTop8Keyed(v[7]);

    __shared__ int64_t sm_buffer[TOPK][256];
    int group_lane = threadIdx.x & 15;
    int group_id = threadIdx.x >> 4;
    if(group_lane < TOPK) {
        int64_t * ptr_sm_buffer = sm_buffer[threadIdx.y];
        #pragma unroll TOPK
        for(int i = 0; i < TOPK; i++) {
            ptr_sm_buffer[(group_id * TOPK + group_lane)*TOPK + i] = v[i];
        }
    }
    __syncthreads();
    int64_t* ptr_sm_buffer = sm_buffer[threadIdx.y];
    #pragma unroll 4
    for(int l = 0; l < 4; l++) {
        int64_t *ptr_buffer = ptr_sm_buffer + l * 64;
        int64_t local_v = ptr_buffer[threadIdx.x];
        GroupSelectTop8Keyed(local_v);
        __syncthreads();
        if(group_lane < TOPK) {
            ptr_sm_buffer[l * 32 + group_id * TOPK + group_lane] = local_v;
        }
    }
    __syncthreads();

    #pragma unroll 2
    for(int l = 0; l < 2; l++) {
        int64_t *ptr_buffer = ptr_sm_buffer + l * 64;
        int64_t local_v = ptr_buffer[threadIdx.x];
        GroupSelectTop8Keyed(local_v);
        __syncthreads();
        if(group_lane < TOPK) {
            ptr_sm_buffer[l * 32 + group_id * TOPK + group_lane] = local_v;
        }
    }

    __syncthreads();

    int64_t local_v = ptr_sm_buffer[threadIdx.x];
    warpSortDescending(local_v, threadIdx.x);
    float sum = 0;
    float w = 0;
    if(lane < TOPK) {
        const int expert = (int)(local_v >> 32);
        w = get_weight(local_v) - (float)router_bias[expert];   // bias 在 L1，8 个标量读 
        indices[TOPK * row + lane] = (IndType)expert;
        sum = w;
        if constexpr(!Renormalize) {
            output[TOPK * row + lane] = (DataType)(w * routed_scaling_factor);
        }
    }

    if constexpr(Renormalize) {        
        for(int step = 4; step > 0; step = step >> 1) {
            sum += __shfl_down_sync_16(0xffffffffffffffff, sum, step);
        }
        sum = __shfl_sync(uint64_t(-1), sum, 0);
        if(lane < TOPK) {
            output[TOPK * row + lane] = (DataType)(w * routed_scaling_factor / sum);
        }
    } 
}

template <typename IndType,
          typename DataType, bool Renormalize>
__global__ void router_bias_small_topk_kernel(const DataType* gating_output, const DataType* router_bias, DataType* output, const int num_rows, IndType* indices,
        const int col_num, const int topk, const int length_power2, const bool check_nan, const int nan_row_i_out, const float routed_scaling_factor)
{
    const int NUM_THREADS = blockDim.x;
    int warps_per_row = (col_num + 63) / 64;
    const int thread_row = blockIdx.x;
    if(thread_row >= num_rows) {
        return;
    }
    const int64_t tid_in_row = threadIdx.x;
    int col_index = tid_in_row  & 63;
    const int warp_id = tid_in_row >> 6;
    DataType row_val;
    float prob;
    row_val = getmin<DataType>();
    float idx_and_value_2[2];
    idx_and_value_2[0] = row_val;
    idx_and_value_2[1] = 0.0f;
    prob = (float)row_val;
    __shared__ bool sm_check_nan;
    extern __shared__ int8_t sm_buffer[];
    float* ptr_sm_bias = (float*)sm_buffer;
    int8_t* shared_memory = (int8_t*)(ptr_sm_bias + col_num);
    if(threadIdx.x < col_num) {
        ptr_sm_bias[threadIdx.x] = (float)router_bias[threadIdx.x];
    }
    if(threadIdx.x == 0) sm_check_nan = false;
    __syncthreads();
    *((int64_t*)idx_and_value_2) |= (tid_in_row << 32);
    if(tid_in_row < col_num) {
        row_val = gating_output[thread_row * col_num + tid_in_row];
        prob =  __builtin_mxc_rcpf((1.0f + __builtin_expf(-row_val)));
        prob = prob + ptr_sm_bias[tid_in_row];
        if(check_nan && __devicecheck_nan(row_val)) {
            if(threadIdx.x == 0) {
                sm_check_nan = true;
            }
        }
        __syncthreads();
    }
    if(sm_check_nan == true) {
        if(threadIdx.x < topk) {
            const int idx = topk * thread_row + threadIdx.x;
            output[idx] = DataType(0);
            indices[idx] = nan_row_i_out;
        }
        return;
    }

    float idx_and_value[2];
    idx_and_value[0] = prob;
    idx_and_value[1] = 0.0f;
    *((int64_t*)idx_and_value) |= (tid_in_row << 32);
    warpSortDescendingUpdate<0xffffffffffffffff>(idx_and_value, col_index);

    if (col_index < topk) {
        int64_t *ptr_sm = (int64_t*)(shared_memory);
        *(ptr_sm + warp_id * topk + col_index) = *(int64_t*)idx_and_value;
    }

    int length = warps_per_row * topk;
    for(int id = length + threadIdx.x; id < length_power2; id += NUM_THREADS )
    {
        *((int64_t*)(shared_memory) + id) = *(int64_t*)idx_and_value_2;
    }
    __syncthreads();

    if(threadIdx.x < length_power2) {
        SortDescendingUpdate((int64_t*)shared_memory, length_power2, threadIdx.x);
    }
    if constexpr(Renormalize) {
        float sum = 0;
        float data = 0;
        if(threadIdx.x < topk) {
            int64_t res = *((int64_t*)shared_memory + threadIdx.x);
            const int idx = topk * thread_row + threadIdx.x;
            data  = get_weight(res) - ptr_sm_bias[res>>32];
            sum = data;
            indices[idx] = res>>32;
        }

        for (int offset = 8; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync_16(0xffffffffffffffff, sum, offset);
        }
        __shared__ float sm_sum;
        if(threadIdx.x == 0) {
            sm_sum = sum;
        }
        __syncthreads();
        if(threadIdx.x < topk) {
            const int idx = topk * thread_row + threadIdx.x;
            output[idx] = data * routed_scaling_factor / sm_sum;
        }
    } else {
        if(threadIdx.x < topk) {
            int64_t res = *((int64_t*)shared_memory + threadIdx.x);
            const int idx = topk * thread_row + threadIdx.x;
            output[idx]  = (get_weight(res) - ptr_sm_bias[res>>32]) * routed_scaling_factor;
            indices[idx] = res>>32;
        }
    }
}

uint32_t next_pow2(uint32_t n)
{
    if(n <= 1) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

template <typename IndType,
          typename DataType>
void launch_kernel(const cudaStream_t &stream, const DataType* gating_output, const DataType* router_bias, DataType* output, const int num_rows, IndType* indices,
        const int col_num, const int topk, const bool check_nan, const int nan_row_i_out, const float routed_scaling_factor, bool Renormalize)
{
    if (topk == 8 && num_rows > 0 && num_rows >= 64) {
        const dim3 grid((num_rows + 7) / 8, 1, 1);
        const dim3 block(64, 8, 1);                  // 8 个 64 路 warp，x 为 wave 内 lane
        // 行偏移 = row*col_num*itemsize，col_num%8==0 时每行行首及各 lane 向量都 16B 对齐
        const bool vec_ok = (col_num % 8 == 0)
                         && (reinterpret_cast<size_t>(gating_output) % 16 == 0)
                         && (reinterpret_cast<size_t>(router_bias) % 16 == 0);
#define LAUNCH_TOP8(REN, VEC) router_bias_warp_top8_kernel<IndType, DataType, REN, VEC> \
        <<<grid, block, 0, stream>>>(gating_output, router_bias, output, num_rows, indices, \
                                     col_num, check_nan, nan_row_i_out, routed_scaling_factor)
        if (Renormalize) { if (vec_ok) LAUNCH_TOP8(true, true); else LAUNCH_TOP8(true, false); }
        else             { if (vec_ok) LAUNCH_TOP8(false, true); else LAUNCH_TOP8(false, false); }
#undef LAUNCH_TOP8
        return;
    }
    dim3 GridSize(num_rows, 1,1);
    dim3 blockSize(512, 1, 1);
    uint32_t warps_per_row = (col_num + 63) / 64;
    uint32_t threads_per_row = warps_per_row * 64;
    uint32_t length_power2 = next_pow2(warps_per_row * topk);
    int share_memory_size = col_num * sizeof(float) + length_power2 * sizeof(int64_t);
    if(Renormalize) {
        router_bias_small_topk_kernel<IndType, DataType, true><<<GridSize, blockSize, share_memory_size, stream>>>(
          gating_output, router_bias, output, num_rows, indices, col_num, topk, length_power2, check_nan, nan_row_i_out, routed_scaling_factor
        );
    } else {
        router_bias_small_topk_kernel<IndType, DataType, false><<<GridSize, blockSize, share_memory_size, stream>>>(
          gating_output, router_bias, output, num_rows, indices, col_num, topk, length_power2, check_nan, nan_row_i_out, routed_scaling_factor
        );
    }
}

void router_bias_topk( 
    at::Tensor gating_output,
    at::Tensor router_bias,
    at::Tensor topk_weights,
    at::Tensor topk_ids,
    const int topk,
    const bool renormalize,
    const bool check_nan,
    const float routed_scaling_factor,
    const int nan_row_i_out
    )
{
    TORCH_CHECK(gating_output.device().is_cuda(), "input must be on CUDA");
    TORCH_CHECK(router_bias.device().is_cuda(), "sin must be on CUDA");
    TORCH_CHECK(topk_weights.device().is_cuda(), "cos must be on CUDA");
    TORCH_CHECK(topk_ids.device().is_cuda(), "cumsum_len must be on CUDA");
    TORCH_CHECK(gating_output.dim() == 2, "gating_output should be 2 dims");
    TORCH_CHECK(router_bias.dim() == 1, "router bias should 1 dims");
    TORCH_CHECK(topk <= 16 , "Current topk should be less than 16");
    TORCH_CHECK(
      topk_ids.dtype() == torch::kInt32 || topk_ids.dtype() == torch::kInt64,
      "router_bias_topk topk_ids only supports int32 or int64 for mask dtype, got ",
      c10::toString(topk_ids.dtype())
    );

    int T = gating_output.size(0);
    int num_experts = gating_output.size(1);

    TORCH_CHECK(num_experts <= 512, "Current num_experts should be less than 512");

    if(T == 0) return;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if(gating_output.dtype() == at::ScalarType::BFloat16) {
        if(topk_ids.dtype() == torch::kInt32) {
            launch_kernel<int, maca_bfloat16>(stream,  
                reinterpret_cast<const maca_bfloat16*>(gating_output.data_ptr<at::BFloat16>()),
                reinterpret_cast<const maca_bfloat16*>(router_bias.data_ptr<at::BFloat16>()),
                reinterpret_cast<maca_bfloat16*>(topk_weights.data_ptr<at::BFloat16>()),
                T, 
                reinterpret_cast<int*>(topk_ids.data_ptr<int>()),
                num_experts,
                topk,
                check_nan,
                nan_row_i_out,
                routed_scaling_factor,
                renormalize
            );
        } else {
            launch_kernel<int64_t, maca_bfloat16>(stream,  
                reinterpret_cast<const maca_bfloat16*>(gating_output.data_ptr<at::BFloat16>()),
                reinterpret_cast<const maca_bfloat16*>(router_bias.data_ptr<at::BFloat16>()),
                reinterpret_cast<maca_bfloat16*>(topk_weights.data_ptr<at::BFloat16>()),
                T, 
                reinterpret_cast<int64_t*>(topk_ids.data_ptr<int64_t>()),
                num_experts,
                topk,
                check_nan,
                nan_row_i_out,
                routed_scaling_factor,
                renormalize
            );
        }
    } else if(gating_output.dtype() == at::ScalarType::Half) { 
        if(topk_ids.dtype() == torch::kInt32) {
            launch_kernel<int, half>(stream,  
                reinterpret_cast<const half*>(gating_output.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(router_bias.data_ptr<at::Half>()),
                reinterpret_cast<half*>(topk_weights.data_ptr<at::Half>()),
                T, 
                reinterpret_cast<int*>(topk_ids.data_ptr<int>()),
                num_experts,
                topk,
                check_nan,
                nan_row_i_out,
                routed_scaling_factor,
                renormalize
            );
        } else {
            launch_kernel<int64_t, half>(stream,  
                reinterpret_cast<const half*>(gating_output.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(router_bias.data_ptr<at::Half>()),
                reinterpret_cast<half*>(topk_weights.data_ptr<at::Half>()),
                T, 
                reinterpret_cast<int64_t*>(topk_ids.data_ptr<int64_t>()),
                num_experts,
                topk,
                check_nan,
                nan_row_i_out,
                routed_scaling_factor,
                renormalize
            );
        }
    } else if(gating_output.dtype() == at::ScalarType::Float) {
        if(topk_ids.dtype() == torch::kInt32) {
            launch_kernel<int, float>(stream,  
                reinterpret_cast<const float*>(gating_output.data_ptr<float>()),
                reinterpret_cast<const float*>(router_bias.data_ptr<float>()),
                reinterpret_cast<float*>(topk_weights.data_ptr<float>()),
                T, 
                reinterpret_cast<int*>(topk_ids.data_ptr<int>()),
                num_experts,
                topk,
                check_nan,
                nan_row_i_out,
                routed_scaling_factor,
                renormalize
            );
        } else {
            launch_kernel<int64_t, float>(stream,  
                reinterpret_cast<const float*>(gating_output.data_ptr<float>()),
                reinterpret_cast<const float*>(router_bias.data_ptr<float>()),
                reinterpret_cast<float*>(topk_weights.data_ptr<float>()),
                T, 
                reinterpret_cast<int64_t*>(topk_ids.data_ptr<int64_t>()),
                num_experts,
                topk,
                check_nan,
                nan_row_i_out,
                routed_scaling_factor,
                renormalize
            );
        }
    } else {
        TORCH_CHECK(0, "rope forward not support this type");
    }
}