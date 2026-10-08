#pragma once

#include "utils.cuh"
#include "utils.h"
#include "all_reduce_kernel.cuh"
#include <cub/cub.cuh>

#define WARP_SIZE 32

#define SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/// Aligned array type
template <
    typename T,
    /// Number of elements in the array
    int N,
    /// Alignment requirement in bytes
    int Alignment = sizeof(T) * N
>
class alignas(Alignment) AlignedArray {
public:
    T data[N];
};

template<typename scalar_t>
__device__ __forceinline__ scalar_t get_weight(const int32_t& v) {
    const scalar_t* idx_and_weight = (const scalar_t*)&v;
    return idx_and_weight[0];
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t get_weight(const int64_t& v) {
    const scalar_t* idx_and_weight = (const scalar_t*)&v;
    return idx_and_weight[0];
}

template<typename scalar_t, typename mask_t>
struct WarpSortProcessor {
    static __device__ __forceinline__ void warpSortDescendingOpt(scalar_t (&idx_and_weight)[2], int tid) ;
};

template<typename scalar_t>
struct WarpSortProcessor<scalar_t, uint16_t> {
    static __device__ __forceinline__ void warpSortDescendingOpt(scalar_t (&idx_and_weight)[2], int tid) {
        const uint32_t MASK = 0xffffffff;
        int64_t val = *(int64_t*)idx_and_weight;
        for (int width = 2; width < 16; width <<=1 ) {
            for (int step = width >> 1; step > 0; step >>=1) {
                const bool direction = ((tid & width) == 0);
                int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
                int other_tid = tid ^ step;

                float current_weight_bits = get_weight<scalar_t>(val);
                float other_weight_bits = get_weight<scalar_t>(other_temp_val);
                int current_index = val >> 32;
                int other_index = other_temp_val >> 32;

                bool weight_gt = other_weight_bits > current_weight_bits;
                bool weight_eq = other_weight_bits == current_weight_bits;
                bool index_lt = other_index < current_index;

                bool other_is_big = weight_gt | (weight_eq & index_lt);
                bool swap = (tid < other_tid) ^ (other_is_big) ^ (direction);

                val = swap ? other_temp_val : val;
            }
        }
        for (int step = 8; step > 0; step >>= 1) {
            int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
            int other_tid = tid ^ step;

            float current_weight_bits = get_weight<scalar_t>(val);
            float other_weight_bits = get_weight<scalar_t>(other_temp_val);
            int current_index = val >> 32;
            int other_index = other_temp_val >> 32;

            bool weight_gt = other_weight_bits > current_weight_bits;
            bool weight_eq = other_weight_bits == current_weight_bits;
            bool index_lt = other_index < current_index;

            bool other_is_big = weight_gt | (weight_eq & index_lt);
            bool swap = (tid < other_tid) ^ (!other_is_big);
            val = swap ? other_temp_val : val;
        }
        *(int64_t*)idx_and_weight = val;
    }
};

template<typename scalar_t>
struct WarpSortProcessor<scalar_t, uint32_t> {
    static __device__ __forceinline__ void warpSortDescendingOpt(scalar_t (&idx_and_weight)[2], int tid) {
        uint32_t MASK = 0xffffffff;
        int64_t val = *(int64_t*)idx_and_weight;
        for (int width = 2; width <= WARP_SIZE; width <<=1) {
            for (int step = width >> 1; step > 0; step >>=1) {
                const bool is_not_final_phase = (width != WARP_SIZE);
                const uint32_t bitmask = (tid & width);
                const bool direction = is_not_final_phase & (bitmask == 0);
                int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
                int other_tid = tid ^ step;

                scalar_t current_weight_bits = get_weight<scalar_t>(val);
                scalar_t other_weight_bits = get_weight<scalar_t>(other_temp_val);

                int current_index = val >> 32;
                int other_index = other_temp_val >> 32;

                bool weight_gt = other_weight_bits > current_weight_bits;
                bool weight_eq = other_weight_bits == current_weight_bits;
                bool index_lt = other_index < current_index;
                bool cond = (tid < other_tid) ^ direction;

                bool swap = (cond & (weight_gt | (weight_eq & index_lt))) |
                            (!cond & ((other_weight_bits < current_weight_bits) | (weight_eq & (other_index > current_index))));

                val = swap ? other_temp_val : val;
            }
        }
        *(int64_t*)idx_and_weight = val;
    }
};

template<typename scalar_t>
struct WarpSortProcessor<scalar_t, uint64_t> {
    static __device__ __forceinline__ void warpSortDescendingOpt(scalar_t (&idx_and_weight)[2], int tid) {
        uint64_t MASK=0xffffffffffffffff;
        int64_t val = *(int64_t*)idx_and_weight;
        for (int width = 2; width < 64; width <<=1 ) {
            for (int step = width >> 1; step > 0; step >>=1) {
                const bool direction = ((tid & width) == 0);
                int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
                int other_tid = tid ^ step;

                float current_weight_bits = get_weight<scalar_t>(val);
                float other_weight_bits = get_weight<scalar_t>(other_temp_val);
                int current_index = val >> 32;
                int other_index = other_temp_val >> 32;

                bool weight_gt = other_weight_bits > current_weight_bits;
                bool weight_eq = other_weight_bits == current_weight_bits;
                bool index_lt = other_index < current_index;

                bool other_is_big = weight_gt | (weight_eq & index_lt);
                bool swap = (tid < other_tid) ^ (other_is_big) ^ (direction);

                val = swap ? other_temp_val : val;
            }
        }
        for (int step = 32; step > 0; step >>= 1) {
            int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
            int other_tid = tid ^ step;

            float current_weight_bits = get_weight<scalar_t>(val);
            float other_weight_bits = get_weight<scalar_t>(other_temp_val);
            int current_index = val >> 32;
            int other_index = other_temp_val >> 32;

            bool weight_gt = other_weight_bits > current_weight_bits;
            bool weight_eq = other_weight_bits == current_weight_bits;
            bool index_lt = other_index < current_index;

            bool other_is_big = weight_gt | (weight_eq & index_lt);
            bool swap = (tid < other_tid) ^ (!other_is_big);
            val = swap ? other_temp_val : val;
        }
        *(int64_t*)idx_and_weight = val;
    }
};

template<typename scalar_t, typename mask_t>
__device__ __forceinline__ void SortElement(scalar_t (&idx_and_weight)[2], int tid) {
    WarpSortProcessor<scalar_t, mask_t>::warpSortDescendingOpt(idx_and_weight, tid);
}

template <typename T, typename RT, typename AT>
struct Math {
    static inline __device__ bool lt(T lhs, T rhs)
    {
        return lhs < rhs;
    }
    static inline __device__ bool gt(T lhs, T rhs)
    {
        return lhs > rhs;
    }
    static inline __device__ bool eq(T lhs, T rhs)
    {
        return lhs == rhs;
    }
};

template <>
struct Math<half, half, half> {
    static inline __device__ bool lt(half lhs, half rhs)
    {
        return __hlt(lhs, rhs);
    }

    static inline __device__ bool gt(half lhs, half rhs)
    {
        return __hgt(lhs, rhs);
    }
    
    static inline __device__ bool eq(half lhs, half rhs)
    {
        return __heq(lhs, rhs);
    }
};

template <typename T>
__device__ unsigned int convert2u(T value);

template <>
__device__ unsigned int convert2u(__half value)
{
    // must use short, for reverse convert
    unsigned short int x    = __half_as_ushort(value);
    unsigned short int mask = (x & 0x8000) ? 0xffff : 0x8000;
    unsigned int res        = x ^ mask;
    return res;
}
template <>
__device__ unsigned int convert2u(float value)
{
    unsigned int x    = __float_as_uint(value);
    unsigned int mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
    unsigned int res  = x ^ mask;
    return res;
}

template <typename T>
__device__ T convertu2(unsigned int value);

template <>
__device__ __half convertu2(unsigned int value)
{
    unsigned short int sht  = (unsigned short int)value;
    unsigned short int mask = (sht & 0x8000) ? 0x8000 : 0xffff;
    unsigned short int x    = sht ^ mask;
    return __ushort_as_half(x);
}
template <>
__device__ float convertu2(unsigned int value)
{
    unsigned int mask = (value & 0x80000000) ? 0x80000000 : 0xffffffff;
    unsigned int x    = value ^ mask;
    return __uint_as_float(x);
}

__device__ __inline__ void bfi(unsigned int &ret, unsigned int y, unsigned int bit_start, unsigned int num_bits)
{
    y = y << bit_start;
    unsigned int MASK_X = ((1 << num_bits) - 1) << bit_start;
    unsigned int MASK_Y = ~MASK_X;
    ret = (ret & MASK_Y) | (y & MASK_X);
}

__device__ __inline__  unsigned int bfe(unsigned int source, unsigned int bit_start, unsigned int num_bits)
{
    return (static_cast<unsigned int>(source) << (32 - bit_start - num_bits)) >> (32 - num_bits);
}

template <typename T>
__device__ unsigned int find_desired(
    unsigned int *smem,
    int lane,
    const unsigned int mask,
    const unsigned int desired,
    const int inputSliceStride,
    const T *inputSlice,
    const int sliceSize)
{
    if (threadIdx.x == 0) {
        smem[0] = 0;
    }
    __syncthreads();

    for (int off = threadIdx.x; off - lane < sliceSize; off += blockDim.x) {
        bool inRange          = off < sliceSize;
        T value               = inRange ? inputSlice[off * inputSliceStride] : (T)0;
        unsigned int intValue = convert2u<T>(value);
        bool flag             = inRange && ((intValue & mask) == desired);
        if (flag) {
            smem[0] = 1;
            smem[1] = intValue;
        }
        __syncthreads();

        unsigned int isFound = smem[0];
        intValue             = smem[1];

        if (isFound) {
        return intValue;
        }
    }
    return 0;
}

template <typename T>
__device__ T scanInWarp(T value, int lane)
{
    T lanePrefix = value;
    for (int i = 1; i < 64; i <<= 1) {
        value = __shfl_up_sync(0xffffffffffffffff, lanePrefix, i, 64);
        if (lane >= i) {
        lanePrefix += value;
        }
    }
    return lanePrefix;
}

__device__ void prefix_scan(
    int *smem,
    const uint64_t active,
    const int activeWarps,
    const bool flag,
    int &index,
    int &blkTotal)
{
    if (threadIdx.x < blockDim.x / 32 + 2) {
        smem[threadIdx.x] = 0;
    }
    __syncthreads();

    uint64_t ballot   = __ballot_sync(active, flag);
    int lane              = threadIdx.x & 63;
    uint64_t laneMask = ~(0xffffffffffffffff << lane);
    laneMask              = active & laneMask;
    int warpId            = threadIdx.x >> 6;
    unsigned int leader   = __ffsll(active) - 1;
    int total             = __popcll(ballot);
    int prefix            = __popcll(laneMask & ballot);


    if (lane == leader) {
        smem[warpId] = total;
    }
    __syncthreads();

    int warpOff = 0;
    if (threadIdx.x < blockDim.x / 32 + 2) {
        int value         = smem[threadIdx.x];
        int warpPrefix    = scanInWarp<int>(value, lane);
        smem[threadIdx.x] = warpPrefix;
    }
    __syncthreads();

    if (warpId >= 1)
        warpOff = smem[warpId - 1];
    blkTotal = smem[activeWarps - 1];

    if (flag) {
        index = warpOff + prefix;
    }
    // write-after-read dependency
    __syncthreads();
}

template <typename T, bool dir>
__device__ T find_kth_value(
    int* smem,
    int K,
    const int sliceSize,
    const T* __restrict__ inputSlice,
    const int inputSliceStride)
{
    static constexpr int RADIX_SIZE = 16;
    static constexpr int RADIX_BITS = 4;
    static constexpr int RADIX_MASK = RADIX_SIZE - 1;
    int count[RADIX_SIZE];
    // use fixed higher bits to filter data
    unsigned int mask    = 0; // fixed high bit
    unsigned int desired = 0; // current radix bits to fix
    int *radix_hist      = smem;
    unsigned int kthValue;
    for (int pos = 8 * sizeof(int) - RADIX_BITS; pos >= 0; pos -= RADIX_BITS) {
        // reinit radix_hist to 0 every loop
        for (int i = 0; i < RADIX_SIZE; i++) {
            count[i] = 0;
        }
        if (threadIdx.x < RADIX_SIZE) {
            radix_hist[threadIdx.x] = 0;
        }
        __syncthreads();

        const int lane = threadIdx.x & 63;
        for (int off = threadIdx.x; off - lane < sliceSize; off += blockDim.x) {
            bool inRange          = off < sliceSize;
            T value               = inRange ? inputSlice[off * inputSliceStride] : (T)0;
            uint64_t active   = __ballot_sync(0xffffffffffffffff, inRange);
            unsigned int intValue = convert2u<T>(value);

            // filter with desired
            bool inRadix = inRange && ((intValue & mask) == desired);
            int valueRadix = 0;
            valueRadix = bfe(intValue, pos, RADIX_BITS);
    #pragma unroll
            for (int i = 0; i < RADIX_SIZE; i++) {
            bool flag           = inRadix && (valueRadix == i);
            uint64_t ballot = __ballot_sync(active, flag);
            count[i] += __popcll(ballot);
            }
        }
        if ((threadIdx.x & 63) == 0) {
            for (int i = 0; i < RADIX_SIZE; i++) {
            atomicAdd(radix_hist + i, count[i]);
            }
        }
        __syncthreads();

        // all threads in blk are the same
        for (int i = 0; i < RADIX_SIZE; i++) {
        count[i] = radix_hist[i];
        }
        __syncthreads();

        // search K count
        if (dir == 1) { // topK largest
        for (int i = RADIX_SIZE - 1; i >= 0; --i) {
            if (K == count[i] && K == 1) {
            bfi(desired, i, pos, RADIX_BITS);
            bfi(mask, RADIX_MASK, pos, RADIX_BITS);
            kthValue      = find_desired<T>((unsigned int *)smem, threadIdx.x, mask, desired, inputSliceStride, inputSlice, sliceSize);
            T fp_kthValue = convertu2<T>(kthValue);
            return fp_kthValue;
            } else if (K <= count[i]) { // narrow radix unitl K == count[i] == 1
            bfi(desired, i, pos, RADIX_BITS);
            bfi(mask, RADIX_MASK, pos, RADIX_BITS);
            break;
            }
            K -= count[i];
        }
        } else {
        for (int i = 0; i < RADIX_SIZE; ++i) {
            if (K == count[i] && K == 1) {
            bfi(desired, i, pos, RADIX_BITS);
            bfi(mask, RADIX_MASK, pos, RADIX_BITS);
            kthValue      = find_desired<T>((unsigned int *)smem, threadIdx.x, mask, desired, inputSliceStride, inputSlice, sliceSize);
            T fp_kthValue = convertu2<T>(kthValue);
            return fp_kthValue;
            } else if (K <= count[i]) { // narrow radix unitl K == count[i] == 1
            bfi(desired, i, pos, RADIX_BITS);
            bfi(mask, RADIX_MASK, pos, RADIX_BITS);
            break;
            }
            K -= count[i];
        }
        }
    }
    kthValue      = desired;
    T fp_kthValue = convertu2<T>(kthValue);
    return fp_kthValue;
}

__device__ __inline__ int64_t Align(int64_t x, int64_t y) {
    return (x + y - 1) / y * y;
}

int getSortSize(int size) {
    if (size == 1) {
        return 1;
    } else if (size <= 4) {
        return 4;
    } else if (size <= 64) {
        return 64;
    } else if (size <= 128) {
        return 128;
    } else if (size <= 256) {
        return 256;
    } else if (size <= 512) {
        return 512;
    } else if (size <= 1024) {
        return 1024;
    } else {
        return -1;
    }
}

template <typename KEY, typename VALUE, bool largest>
__device__ inline void swap(
    const bool isOdd,
    bool &valid1,
    KEY &value1,
    VALUE &index1,
    bool &valid2,
    KEY &value2,
    VALUE &index2)
{
    bool isLarge = (largest ^ Math<KEY, KEY, KEY>::lt(value1, value2) && valid1) || !valid2;
    bool isEqual = Math<KEY, KEY, KEY>::eq(value1, value2);

    bool indexLarge = (Math<KEY, KEY, KEY>::gt(index1, index2) && valid2) || !valid1;
    bool if_exchange = ((isLarge == isOdd) && !isEqual) || (isEqual && indexLarge);

    if (if_exchange) {
        KEY tmpValue   = value1;
        VALUE tmpIndex = index1;
        bool tmpValid  = valid1;
        value1         = value2;
        index1         = index2;
        valid1         = valid2;
        value2         = tmpValue;
        index2         = tmpIndex;
        valid2         = tmpValid;
    }
}

template <typename KEY, typename VALUE, bool dir, int power2SortSize>
__device__ void bitonicSort(
    KEY *Key,
    VALUE *Value,
    const int sliceSize)
{
    __shared__ KEY smemTopk[power2SortSize];
    __shared__ VALUE smemIndices[power2SortSize];
    __shared__ bool smemValid[power2SortSize];

    KEY *topKSlice      = Key;
    VALUE *indicesSlice = Value;

    int tid       = threadIdx.x;
    int off1      = threadIdx.x;
    int off2      = threadIdx.x + power2SortSize / 2;
    bool inRange1 = off1 < sliceSize;
    bool inRange2 = off2 < sliceSize;
    KEY value1    = inRange1 ? topKSlice[off1] : (KEY)0;
    VALUE index1  = inRange1 ? indicesSlice[off1] : (VALUE)0;
    KEY value2    = inRange2 ? topKSlice[off2] : (KEY)0;
    VALUE index2  = inRange2 ? indicesSlice[off2] : (VALUE)0;

    smemTopk[off1]    = value1;
    smemIndices[off1] = index1;
    smemValid[off1]   = inRange1;
    smemTopk[off2]    = value2;
    smemIndices[off2] = index2;
    smemValid[off2]   = inRange2;
    __syncthreads();

    #pragma unroll
    for (int size = 2; size < power2SortSize; size *= 2) {
        int oddSeg = (tid & (size / 2)) != 0;
    #pragma unroll
        // sort each size
        for (int sub_size = size; sub_size > 1; sub_size /= 2) {
            int stride = sub_size / 2;
            int off    = (tid / stride) * sub_size + (tid & (stride - 1));

            bool inRange1 = smemValid[off];
            KEY value1    = smemTopk[off];
            VALUE index1  = smemIndices[off];
            bool inRange2 = smemValid[off + stride];
            KEY value2    = smemTopk[off + stride];
            VALUE index2  = smemIndices[off + stride];

            swap<KEY, VALUE, dir>(oddSeg,
                                    inRange1,
                                    value1,
                                    index1,
                                    inRange2,
                                    value2,
                                    index2);

            smemTopk[off]             = value1;
            smemIndices[off]          = index1;
            smemValid[off]            = inRange1;
            smemTopk[off + stride]    = value2;
            smemIndices[off + stride] = index2;
            smemValid[off + stride]   = inRange2;

            __syncthreads();
        }
    }

    // sort the whole power2SortSize
    for (int sub_size = power2SortSize; sub_size > 1; sub_size /= 2) {
        int stride = sub_size / 2;
        int off    = (tid / stride) * sub_size + (tid & (stride - 1));

        bool inRange1 = smemValid[off];
        KEY value1    = smemTopk[off];
        VALUE index1  = smemIndices[off];
        bool inRange2 = smemValid[off + stride];
        KEY value2    = smemTopk[off + stride];
        VALUE index2  = smemIndices[off + stride];

        swap<KEY, VALUE, dir>(false,
                                inRange1,
                                value1,
                                index1,
                                inRange2,
                                value2,
                                index2);

        smemTopk[off]             = value1;
        smemIndices[off]          = index1;
        smemValid[off]            = inRange1;
        smemTopk[off + stride]    = value2;
        smemIndices[off + stride] = index2;
        smemValid[off + stride]   = inRange2;

        __syncthreads();
    }

    inRange1 = smemValid[off1];
    value1   = smemTopk[off1];
    index1   = smemIndices[off1];
    inRange2 = smemValid[off2];
    value2   = smemTopk[off2];
    index2   = smemIndices[off2];
    if (inRange1) {
        topKSlice[off1]    = value1;
        indicesSlice[off1] = index1;
    }
    if (inRange2) {
        topKSlice[off2]    = value2;
        indicesSlice[off2] = index2;
    }
    __syncthreads();

    if(tid * 2 < sliceSize) {
        if (tid > 0 && (topKSlice[tid * 2] == topKSlice[tid * 2 - 1]) && (indicesSlice[tid * 2] < indicesSlice[tid * 2 - 1])){
            VALUE tmp;
            tmp = indicesSlice[tid * 2];
            indicesSlice[tid * 2] = indicesSlice[tid * 2 - 1];
            indicesSlice[tid * 2 - 1] = tmp;
        }
    __syncthreads();
        if ((topKSlice[tid * 2] == topKSlice[tid * 2 + 1]) && (indicesSlice[tid * 2] > indicesSlice[tid * 2 + 1])){
            VALUE tmp;
            tmp = indicesSlice[tid * 2];
            indicesSlice[tid * 2] = indicesSlice[tid * 2 + 1];
            indicesSlice[tid * 2 + 1] = tmp;
        }
    }
}

namespace mc_moe_softmax_topk
{

template <typename scalar_t, int TPB>
__launch_bounds__(TPB) __global__
    void moeSoftmax(const scalar_t* input, scalar_t* output, const int num_cols)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizing_factor;
    __shared__ float float_max;

    const int thread_row_offset = blockIdx.x * num_cols;

    cub::Sum sum;
    float threadData(-FLT_MAX);

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        threadData = max(input[idx], threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (threadIdx.x == 0)
    {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        threadData += exp(((input[idx]) - float_max));
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0)
    {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = exp(((input[idx]) - float_max)) * normalizing_factor;
        output[idx] = (val);
    }
}

template <typename scalar_t, int TPB>
__launch_bounds__(TPB) __global__ void moeTopK(const scalar_t* inputs_after_softmax, scalar_t* output,
    int* indices, const int num_experts, const int k, const int start_expert, const int end_expert)
{

    using cub_kvp = cub::KeyValuePair<int, scalar_t>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const int num_rows = gridDim.x;
    const int block_row = blockIdx.x;
    extern __shared__ float topk_value[];
    float byte_sum = 0;

    const int thread_read_offset = blockIdx.x * num_experts;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        thread_kvp.key = 0;
        thread_kvp.value = (-1.f); // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB)
        {
            const int idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            inp_kvp.value = inputs_after_softmax[idx];

            for (int prior_k = 0; prior_k < k_idx; ++prior_k)
            {
                const int prior_winning_expert = indices[k * block_row + prior_k];

                if (prior_winning_expert == expert)
                {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0)
        {
            // Ignore experts the node isn't responsible for with expert parallelism
            const int expert = result_kvp.key;
            const int idx = k * block_row + k_idx;
            // output[idx] = result_kvp.value;
            topk_value[k_idx] = result_kvp.value;
            byte_sum += result_kvp.value;
            indices[idx] = expert;
            assert(indices[idx] >= 0);
            // source_rows[idx] = k_idx * num_rows + block_row;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int idx = k * block_row + k_idx;
            output[idx] = topk_value[k_idx] / byte_sum;
        }
    }
}

template <typename scalar_t, int TPB>
__launch_bounds__(TPB) __global__ void moeTopKSoftmax(const scalar_t* inputs_after_softmax, scalar_t* output,
    int* indices, const int num_experts, const int k, const int start_expert, const int end_expert)
{

    using cub_kvp = cub::KeyValuePair<int, scalar_t>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const int num_rows = gridDim.x;
    const int block_row = blockIdx.x;

    extern __shared__ float topk_value[];
    float byte_sum = 0;
    float byte_max = -99999.f;

    const int thread_read_offset = blockIdx.x * num_experts;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        thread_kvp.key = 0;
        thread_kvp.value = (-1.f); // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB)
        {
            const int idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            inp_kvp.value = inputs_after_softmax[idx];

            for (int prior_k = 0; prior_k < k_idx; ++prior_k)
            {
                const int prior_winning_expert = indices[k * block_row + prior_k];

                if (prior_winning_expert == expert)
                {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0)
        {
            // Ignore experts the node isn't responsible for with expert parallelism
            const int expert = result_kvp.key;

            const int idx = k * block_row + k_idx;
            // output[idx] = result_kvp.value;
            topk_value[k_idx] = result_kvp.value;
            indices[idx] = expert;
            assert(indices[idx] >= 0);
            // source_rows[idx] = k_idx * num_rows + block_row;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            byte_max = max(topk_value[k_idx], byte_max);
        }
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            topk_value[k_idx] = __builtin_expf(topk_value[k_idx] - byte_max);
            byte_sum += topk_value[k_idx];
        }
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int idx = k * block_row + k_idx;
            output[idx] = topk_value[k_idx] / byte_sum;
        }
    }
}

template <typename scalar_t, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topkGatingSoftmax(const scalar_t* input, const bool* finished, scalar_t* output, const int num_rows, int* indices,
        const int k, const int start_expert, const int end_expert)
{
    // We begin by enforcing compile time assertions and setting up compile time constants.
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(scalar_t);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    // Restrictions based on previous section.
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and a block contains WARPS_PER_CTA warps.
    // This, each block processes a chunk of rows. We start by computing the start row for each block.
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the base row per warp.
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    extern __shared__ float topk_value[];
    float byte_sum = 0;

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows)
    {
        return;
    }
    // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
    // row it will read.
    const scalar_t* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the first column to start loads.
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const scalar_t* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    // Determine the pointer type to use to read in the data depending on the BYTES_PER_LDG template param. In theory,
    // this can support all powers of 2 up to 16.
    // NOTE(woosuk): The original implementation uses CUTLASS aligned array here.
    // We defined our own aligned array and use it here to avoid the dependency on CUTLASS.
    using AccessType = AlignedArray<scalar_t, ELTS_PER_LDG>;

    // Finally, we pull in the data from global mem
    scalar_t row_chunk[VPT];
    AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    // First, we perform a max reduce within the thread. We can do the max in fp16 safely (I think) and just
    // convert to float afterwards for the exp + sum reduction.
    float thread_max = (row_chunk[0]);
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii)
    {
        thread_max = max(thread_max, (row_chunk[ii]));
    }

// Now, we find the max within the thread group and distribute among the threads. We use a butterfly reduce.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        thread_max = max(thread_max, SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
    }

    // From this point, thread max in all the threads have the max within the row.
    // Now, we subtract the max from each element in the thread and take the exp. We also compute the thread local sum.
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        float tmp = __builtin_expf((row_chunk[ii]) - thread_max);
        row_chunk[ii] = (tmp);
        row_sum += tmp;
    }

// Now, we perform the sum reduce within each thread group. Similar to the max reduce, we use a bufferfly pattern.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        row_sum += SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
    }

    // From this point, all threads have the max and the sum for their rows in the thread_max and thread_sum variables
    // respectively. Finally, we can scale the rows for the softmax. Technically, for top-k gating we don't need to
    // compute the entire softmax row. We can likely look at the maxes and only compute for the top-k values in the row.
    // However, this kernel will likely not be a bottle neck and it seems better to closer match torch and find the
    // argmax after computing the softmax.
    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = ((row_chunk[ii]) * reciprocal_row_sum);
    }

    // Now, softmax_res contains the softmax of the row chunk. Now, I want to find the topk elements in each row, along
    // with the max index.
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        // First, each thread does the local argmax
        scalar_t max_val = row_chunk[0];
        int expert = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                scalar_t val = row_chunk[ldg * ELTS_PER_LDG + ii];

                // No check on the experts here since columns with the smallest index are processed first and only
                // updated if > (not >=)
                if (val > max_val)
                {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads reach consensus about the max.
// This will be useful for K > 1 so that the threads can agree on "who" had the max value. That thread can
// then blank out their max with -inf and the warp can run more iterations...
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            scalar_t other_max = SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
            int other_expert = SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);

            // We want lower indices to "win" in every thread so we break ties this way
            if (other_max > max_val || (other_max == max_val && other_expert < expert))
            {
                max_val = other_max;
                expert = other_expert;
            }
        }

        // Write the max for this k iteration to global memory.
        if (thread_group_idx == 0)
        {
            // The lead thread from each sub-group will write out the final results to global memory. (This will be a
            // single) thread per row of the input/output matrices.
            const int idx = k * thread_row + k_idx;
            // output[idx] = max_val;
            topk_value[threadIdx.y * k + k_idx] = max_val;
            byte_sum += max_val;
            indices[idx] = expert;
        }

        // Finally, we clear the value in the thread with the current max if there is another iteration to run.
        if (k_idx + 1 < k)
        {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            // Only the thread in the group which produced the max will reset the "winning" value to -inf.
            if (thread_group_idx == thread_to_clear_in_group)
            {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                // Safe to set to any negative value since row_chunk values must be between 0 and 1.
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = (-999);
            }
        }
    }

    __syncthreads();
    if (thread_group_idx == 0) {
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int idx = k * thread_row + k_idx;
            output[idx] = topk_value[threadIdx.y * k + k_idx] / byte_sum;
        }
    }
}

template <typename scalar_t, typename bitwise_t, int NUM_EXPERTS = 128, int WARPS_PER_CTA = 8, int TOPK = 8, int WAVE_SIZE = 64, int WAVES_PER_ROW = 2>
__launch_bounds__(WARPS_PER_CTA * WAVE_SIZE) __global__
    void topkGatingSoftmaxDecodeOpttt(const scalar_t* input, scalar_t* output, const int num_rows, int* indices)
{
    const int thread_row = blockIdx.x * WARPS_PER_CTA / WAVES_PER_ROW + threadIdx.y;
    const int wave_id_in_row = threadIdx.x / WAVE_SIZE;
    constexpr int bit_offset = sizeof(bitwise_t) * 4;
    if (thread_row >= num_rows) {
        return;
    }
    __shared__ float s_max_v[WARPS_PER_CTA];
    __shared__ float s_sum_v[WARPS_PER_CTA];

    scalar_t row_chunk = input[thread_row * NUM_EXPERTS + threadIdx.x];
    float max_val = (row_chunk);
    float sum_val = __builtin_expf(max_val);

    max_val = fmaxf(max_val, __shfl_down_sync_16(0xffffffffffffffff, max_val, 1));
    max_val = fmaxf(max_val, __shfl_down_sync_16(0xffffffffffffffff, max_val, 2));
    max_val = fmaxf(max_val, __shfl_down_sync_16(0xffffffffffffffff, max_val, 4));
    max_val = fmaxf(max_val, __shfl_down_sync_16(0xffffffffffffffff, max_val, 8));
    max_val = fmaxf(max_val, __shfl_down_sync(0xffffffffffffffff, max_val, 16, WAVE_SIZE));
    max_val = fmaxf(max_val, __shfl_down_sync(0xffffffffffffffff, max_val, 32, WAVE_SIZE));

    sum_val += __shfl_down_sync_16(0xffffffffffffffff, sum_val, 1);
    sum_val += __shfl_down_sync_16(0xffffffffffffffff, sum_val, 2);
    sum_val += __shfl_down_sync_16(0xffffffffffffffff, sum_val, 4);
    sum_val += __shfl_down_sync_16(0xffffffffffffffff, sum_val, 8);
    sum_val += __shfl_down_sync(0xffffffffffffffff, sum_val, 16, WAVE_SIZE);
    sum_val += __shfl_down_sync(0xffffffffffffffff, sum_val, 32, WAVE_SIZE);

    bitwise_t tid = (bitwise_t)threadIdx.x;

    if(threadIdx.x % WAVE_SIZE == 0) {
        s_max_v[threadIdx.y * WAVES_PER_ROW + wave_id_in_row] = max_val;
        s_sum_v[threadIdx.y * WAVES_PER_ROW + wave_id_in_row] = sum_val;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        s_max_v[threadIdx.y * WAVES_PER_ROW] = max(s_max_v[threadIdx.y * WAVES_PER_ROW], s_max_v[threadIdx.y * WAVES_PER_ROW + 1]);
        s_sum_v[threadIdx.y * WAVES_PER_ROW] = s_sum_v[threadIdx.y * WAVES_PER_ROW] + s_sum_v[threadIdx.y * WAVES_PER_ROW + 1];
    }
    __syncthreads();
    
    float rep = 1.0f / (s_sum_v[threadIdx.y * WAVES_PER_ROW] * __builtin_expf(-s_max_v[threadIdx.y * WAVES_PER_ROW]));
    row_chunk = (__builtin_expf((row_chunk) - s_max_v[threadIdx.y * WAVES_PER_ROW]) * rep);

    __shared__ scalar_t shared_experts[WARPS_PER_CTA * TOPK * NUM_EXPERTS / (WAVE_SIZE * WAVES_PER_ROW)][2];
    scalar_t idx_and_weight[2];
    idx_and_weight[0] = row_chunk;
    idx_and_weight[1] = (0.0f);
    *((bitwise_t*)idx_and_weight) |= (tid << bit_offset);
    SortElement<scalar_t, uint64_t>(idx_and_weight, threadIdx.x);
    if (threadIdx.x % WAVE_SIZE < TOPK) {
        *(((bitwise_t*)(shared_experts)) + threadIdx.y * (TOPK * NUM_EXPERTS / WAVE_SIZE) + (threadIdx.x % WAVE_SIZE) + wave_id_in_row * TOPK) = *(bitwise_t*)idx_and_weight;
    }
    __syncthreads();

    *(bitwise_t*)idx_and_weight = threadIdx.x < TOPK * WAVES_PER_ROW ? *((bitwise_t*)shared_experts + threadIdx.y * (TOPK * NUM_EXPERTS / WAVE_SIZE) + threadIdx.x) : 0;
    SortElement<scalar_t, uint16_t>(idx_and_weight, threadIdx.x);

    if (threadIdx.x < TOPK) {
        bitwise_t res = *(bitwise_t*)idx_and_weight;
        int expert_id_ordered = (res >> bit_offset);
        scalar_t max_val_ordered = get_weight<scalar_t>(res);
        scalar_t sum_ = WarpAllReduceSum<scalar_t>(max_val_ordered, TOPK);
        output[thread_row * TOPK + threadIdx.x] = max_val_ordered / sum_;
        indices[thread_row * TOPK + threadIdx.x] = expert_id_ordered;
    }
}

// template <typename scalar_t, int TPB, int BLOCK_SIZE>
// __launch_bounds__(TPB) __global__ void moeTopKSoftmaxVectorlized(const scalar_t* input, scalar_t* output,
//     int* indices, const int num_experts, const int k, const int start_expert, const int end_expert)
// {
//     typedef cub::BlockRadixSort<scalar_t, BLOCK_SIZE, TPB> BlockRadixSort;
//     typedef cub::BlockLoad<scalar_t, BLOCK_SIZE, TPB, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
//     typedef cub::BlockStore<scalar_t, BLOCK_SIZE, TPB, cub::BLOCK_STORE_TRANSPOSE> BlockStore;

//     __shared__ union {
//         typename BlockRadixSort::TempStorage sort;
//         typename BlockLoad::TempStorage load;
//         typename BlockStore::TempStorage store;
//     } temp_storage;

//     float row_chunk[TPB];
//     int block_offset = blockIdx.x * TPB * BLOCK_SIZE;
//     BlockLoad(temp_storage.load).Load(input + block_offset, row_chunk);
//     __syncthreads();

//     BlockRadixSort(temp_storage.sort).Sort(row_chunk);
//     __syncthreads();

//     BlockStore(temp_storage.store).Store(output + block_offset, row_chunk);
// }

template <typename scalar_t, typename index_t, int64_t BLOCK_SIZE, int sortBlockSize = 512>
__global__ void selectTopKSoftmax(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ topK,
    index_t* __restrict__ indices,
    const int K,
    const int64_t sliceSize,
    const int inputStride,
    const int topKStride, 
    const int indicesStride
) {
    extern __shared__ scalar_t topk_value[];
    scalar_t* key_smem = &topk_value[0];
    index_t* val_smem = reinterpret_cast<index_t*>(key_smem + K);

    const int inputSliceStride = 1;
    const int topKSliceStride = 1;
    const int indicesSliceStride = 1;

    const scalar_t* inputSlice = input + blockIdx.x * inputStride;
    scalar_t* topKSliceOut = topK + blockIdx.x * K;
    index_t* indicesSliceOut = indices + blockIdx.x * K;
    scalar_t* topKSlice = key_smem;
    index_t* indicesSlice = val_smem;

    __shared__ int radix_hist[2 + BLOCK_SIZE / 32];
    int *smem = radix_hist;

    scalar_t fp_kthValue = find_kth_value<scalar_t, true>(smem, K, sliceSize, inputSlice, inputSliceStride);

    int writeStart  = 0;
    int activeWarps = 0;
    int64_t tmpSize     = sliceSize;
    for (int64_t off = threadIdx.x; off < Align(sliceSize, BLOCK_SIZE); off += BLOCK_SIZE) {
        int curSize         = tmpSize >= BLOCK_SIZE ? BLOCK_SIZE : tmpSize;
        activeWarps         = (curSize + 63) >> 6;
        bool inRange        = off < sliceSize;
        scalar_t value             = inRange ? inputSlice[off * inputSliceStride] : (scalar_t)0;
        uint64_t active = __ballot_sync(0xffffffffffffffff, inRange);

        bool flag;
        flag = inRange && Math<scalar_t, scalar_t, scalar_t>::gt(value, fp_kthValue);
        int index, blkTotal;
        prefix_scan(smem, active, activeWarps, flag, index, blkTotal);

        if (flag) {
            int topKOffset            = writeStart + index;
            int indexOffset           = writeStart + index;
            topKSlice[topKOffset]     = value;
            indicesSlice[indexOffset] = off;
        }
        writeStart += blkTotal;
        tmpSize -= BLOCK_SIZE;
    }
    __syncthreads();

    int topKRemaining = K - writeStart;
    tmpSize           = sliceSize;
    for (int64_t off = threadIdx.x; off < Align(sliceSize, BLOCK_SIZE); off += BLOCK_SIZE) {
        int curSize         = tmpSize >= BLOCK_SIZE ? BLOCK_SIZE : tmpSize;
        activeWarps         = (curSize + 63) >> 6;
        bool inRange        = off < sliceSize;
        scalar_t value             = inRange ? inputSlice[off * inputSliceStride] : (scalar_t)0;
        uint64_t active = __ballot_sync(0xffffffffffffffff, inRange);

        bool flag;
        flag = inRange && Math<scalar_t, scalar_t, scalar_t>::eq(value, fp_kthValue);
        int index, blkTotal;
        prefix_scan(smem, active, activeWarps, flag, index, blkTotal);

        if (flag) {
            int outputIndex = writeStart + index;
            if (outputIndex < K) {
                int topKOffset            = outputIndex;
                int indexOffset           = outputIndex;
                topKSlice[topKOffset]     = value;
                indicesSlice[indexOffset] = off;
            }
        }
        if (topKRemaining < blkTotal) {
            break;
        }
        topKRemaining -= blkTotal;
        writeStart += blkTotal;
        tmpSize -= BLOCK_SIZE;
    }
    __syncthreads();

    if (threadIdx.x < sortBlockSize / 2) {
        bitonicSort<scalar_t, index_t, true, sortBlockSize>(topKSlice, indicesSlice, K);
    }
    __syncthreads();
    scalar_t max_val = -9999;
    scalar_t sum_val = 0.0f;

    if (threadIdx.x >= 64) return;

    for (int idx = threadIdx.x; idx < K; idx += 64) {
        scalar_t weights_ = topKSlice[idx];
        index_t indices_ = indicesSlice[idx];
        indicesSliceOut[idx] = indices_;
        max_val = max(max_val, weights_);
        sum_val += __builtin_expf(weights_);
    }
    for (int stride = 32; stride > 0; stride >>= 1) {
        max_val = max(SHFL_XOR_SYNC_WIDTH(max_val, stride, 64), max_val);
        sum_val += __shfl_xor_sync(0xffffffffffffffff, sum_val, stride);
    }
    
    for (int idx = threadIdx.x; idx < K; idx += 64) {
        scalar_t res_weight = topKSlice[idx];
        res_weight = __builtin_expf(res_weight - max_val) / (sum_val * __builtin_expf(-max_val));
        topKSliceOut[idx] = res_weight;
    }
}

// ============================================================================
// Generalized fused softmax + top-k for ARBITRARY num_experts (incl. non-pow2).
// One 64-thread wave processes one row (token). The row is read ONCE into
// registers. Key math: the renormalized top-k weight is
//     w_i = softmax_i / sum_{j in topk} softmax_j = exp(x_i-m) / sum_topk exp(x_j-m),
// so the global softmax normalizer Z cancels -> we only need the row max, NOT the
// full-row sum. Top-k is extracted by k rounds of wave-argmax (ties -> smaller
// expert index), masking the winner each round. No global workspace, no atomics.
//
// NOTE: all wave-wide shuffles MUST use the 64-bit mask (warp = 64 lanes on
// C600U); a 32-bit mask leaves lanes 32..63 out of sync -> garbage reductions.
#define SHFL_XOR_64(var, m) __shfl_xor_sync(0xffffffffffffffffULL, (var), (m), 64)
// ============================================================================
template <typename scalar_t, int WAVES_PER_CTA, int VPT, int MAX_K>
__launch_bounds__(WAVES_PER_CTA * 64) __global__
void fusedSoftmaxTopk(const scalar_t* __restrict__ input,
                      scalar_t* __restrict__ output,
                      int* __restrict__ indices,
                      const int num_rows, const int num_experts)
{
    static_assert(MAX_K > 0 && MAX_K <= 16, "MAX_K must be in [1, 16]");
    constexpr int WAVE  = 64;
    constexpr int VEC   = 4;
    constexpr bool VECTORIZED_LAYOUT = (VPT % VEC) == 0;
    const int lane = threadIdx.x;            // 0..63
    const int wid  = threadIdx.y;            // wave index within the block
    const int row  = blockIdx.x * WAVES_PER_CTA + wid;
    if (row >= num_rows) return;

    const scalar_t* __restrict__ row_ptr = input + (size_t)row * num_experts;

    // A vector-layout lane owns four adjacent experts per 256-expert chunk.
    // Full, 16-byte-aligned rows use float4 loads; tails use the same ownership
    // mapping with guarded scalar loads. Other VPT values keep the striped layout.
    float vals[VPT];
    const bool use_vector_load = VECTORIZED_LAYOUT
        && sizeof(scalar_t) == sizeof(float)
        && num_experts == VPT * WAVE
        && ((reinterpret_cast<size_t>(row_ptr) & (alignof(float4) - 1)) == 0);
    if (use_vector_load) {
        const float4* __restrict__ vec_ptr =
            reinterpret_cast<const float4*>(row_ptr);
#pragma unroll
        for (int g = 0; g < VPT / VEC; ++g) {
            const float4 v = vec_ptr[g * WAVE + lane];
            vals[g * VEC] = v.x; vals[g * VEC + 1] = v.y;
            vals[g * VEC + 2] = v.z; vals[g * VEC + 3] = v.w;
        }
    } else {
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            const int e = VECTORIZED_LAYOUT
                ? (i / VEC) * (WAVE * VEC) + lane * VEC + (i % VEC)
                : lane + i * WAVE;
            vals[i] = (e < num_experts) ? (float)row_ptr[e] : -FLT_MAX;
        }
    }

    __shared__ float s_lg[WAVES_PER_CTA][MAX_K];   // selected raw logits
    __shared__ int   s_i [WAVES_PER_CTA][MAX_K];   // selected expert ids

#pragma unroll
    for (int kk = 0; kk < MAX_K; ++kk) {
        // Local argmax over this lane's VPT elements (ties -> smaller expert index).
        float best = -FLT_MAX;
        int   best_e = num_experts;               // large sentinel loses ties
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            const int e = VECTORIZED_LAYOUT
                ? (i / VEC) * (WAVE * VEC) + lane * VEC + (i % VEC)
                : lane + i * WAVE;
            if (vals[i] > best || (vals[i] == best && e < best_e)) { best = vals[i]; best_e = e; }
        }
        // Wave-wide argmax with two cheap 32-bit shuffles (faster than one 64-bit).
#pragma unroll
        for (int m = 32; m > 0; m >>= 1) {
            const float ov = SHFL_XOR_64(best, m);
            const int   oe = SHFL_XOR_64(best_e, m);
            if (ov > best || (ov == best && oe < best_e)) { best = ov; best_e = oe; }
        }
        if (lane == 0) { s_lg[wid][kk] = best; s_i[wid][kk] = best_e; }
        // Mask the winner. Use a STATIC-indexed compare (never vals[dynamic]):
        // a dynamic index forces vals[] to local memory -> ~10x bandwidth cliff.
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            const int e = VECTORIZED_LAYOUT
                ? (i / VEC) * (WAVE * VEC) + lane * VEC + (i % VEC)
                : lane + i * WAVE;
            if (e == best_e) vals[i] = -FLT_MAX;
        }
    }

    // Renormalize over the k winners (rank-0 logit is the row max) and emit.
    if (lane == 0) {
        const float mx = s_lg[wid][0];
        float sum = 0.f;
#pragma unroll
        for (int kk = 0; kk < MAX_K; ++kk) { const float e = __builtin_expf(s_lg[wid][kk] - mx); s_lg[wid][kk] = e; sum += e; }
        const float inv = __builtin_mxc_rcpf(sum);
#pragma unroll
        for (int kk = 0; kk < MAX_K; ++kk) {
            output[row * MAX_K + kk]  = (scalar_t)(s_lg[wid][kk] * inv);
            indices[row * MAX_K + kk] = s_i[wid][kk];
        }
    }
}

// ============================================================================
// DIAGNOSTIC: read-ceiling probe. Identical grid/block/VPT access pattern as
// fusedSoftmaxTopk, but the whole k-round selection is replaced by ONE wave-max
// (6 shuffle steps, no index tracking, no masking, no per-round dependency).
// Writes k garbage outputs so read+write traffic matches. If this is far faster
// than the serial kernel, the bottleneck is the selection compute, not memory.
template <typename scalar_t, int WAVES_PER_CTA, int VPT>
__launch_bounds__(WAVES_PER_CTA * 64) __global__
void fusedSoftmaxTopkProbe(const scalar_t* __restrict__ input,
                           scalar_t* __restrict__ output,
                           int* __restrict__ indices,
                           const int num_rows, const int num_experts, const int k)
{
    constexpr int WAVE = 64;
    const int lane = threadIdx.x;
    const int wid  = threadIdx.y;
    const int row  = blockIdx.x * WAVES_PER_CTA + wid;
    if (row >= num_rows) return;

    const scalar_t* __restrict__ row_ptr = input + (size_t)row * num_experts;
    float best = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int e = lane + i * WAVE;
        const float v = (e < num_experts) ? (float)row_ptr[e] : -FLT_MAX;
        best = v > best ? v : best;
    }
#pragma unroll
    for (int m = 32; m > 0; m >>= 1) {
        const float o = SHFL_XOR_64(best, m);
        best = o > best ? o : best;
    }
    if (lane == 0) {
#pragma unroll 1
        for (int kk = 0; kk < k; ++kk) {
            output[row * k + kk]  = (scalar_t)best;
            indices[row * k + kk] = kk;
        }
    }
}

struct PackedTopKFields
{
    float value;
    int32_t expert;
};

union alignas(8) PackedTopK
{
    PackedTopKFields fields;
    int64_t raw;
};

static_assert(sizeof(PackedTopK) == sizeof(int64_t), "PackedTopK must be 64-bit");
static_assert(alignof(PackedTopK) == alignof(int64_t), "PackedTopK must be 8-byte aligned");

__device__ __forceinline__ PackedTopK makePackedTopK(float value, int expert)
{
    PackedTopK result;
    result.fields.value = value;
    result.fields.expert = expert;
    return result;
}

__device__ __forceinline__ PackedTopK maxPackedTopK(PackedTopK lhs, PackedTopK rhs)
{
    if (rhs.fields.value > lhs.fields.value
        || (rhs.fields.value == lhs.fields.value
            && rhs.fields.expert < lhs.fields.expert)) {
        return rhs;
    }
    return lhs;
}

__device__ __forceinline__ void compareExchangePackedDesc(
    PackedTopK& lhs, PackedTopK& rhs)
{
    const bool rhs_wins = rhs.fields.value > lhs.fields.value
        || (rhs.fields.value == lhs.fields.value
            && rhs.fields.expert < lhs.fields.expert);
    const PackedTopK old_lhs = lhs;
    const PackedTopK old_rhs = rhs;
    lhs = rhs_wins ? old_rhs : old_lhs;
    rhs = rhs_wins ? old_lhs : old_rhs;
}

__device__ __forceinline__ PackedTopK argmax4(float4 values, int base_expert)
{
    const PackedTopK p0 = makePackedTopK(values.x, base_expert);
    const PackedTopK p1 = makePackedTopK(values.y, base_expert + 1);
    const PackedTopK p2 = makePackedTopK(values.z, base_expert + 2);
    const PackedTopK p3 = makePackedTopK(values.w, base_expert + 3);
    return maxPackedTopK(
        maxPackedTopK(p0, p1),
        maxPackedTopK(p2, p3));
}

// ============================================================================
// Iteration 4: one row per 16-lane SUBGROUP (C600U native fast-shuffle domain).
// Skill rule #1: intra-16 shuffles (__shfl_xor width 16) are the FAST path;
// steps that cross the 16-lane group (32,16 on a 64-wide warp) are SLOW and sit
// on the serial critical path of every argmax round. By giving each row its own
// 16-lane subgroup, ALL butterfly steps stay intra-16 (fast) with no cross-group
// merge, and a 64-lane warp runs 4 independent rows -> more latency hiding.
// Softmax monotonic -> select on raw logits; rank-0 is the row max; exp only k.
//   SUBW = 16, ROWS_PER_WARP = 4. Each lane holds VPT = ceil(E/16) experts.
#define SHFL_XOR_16(var, m) __shfl_xor_sync(0xffffffffffffffffULL, (var), (m), 16)

__device__ __forceinline__ PackedTopK subgroupArgmax16(PackedTopK local)
{
    PackedTopK winner = local;
#pragma unroll
    for (int m = 8; m > 0; m >>= 1) {
        PackedTopK other;
        other.raw = SHFL_XOR_16(winner.raw, m);
        winner = maxPackedTopK(winner, other);    }
    return winner;
}

// ---- Bitonic top-8 selection within a 16-lane subgroup (Iteration 6) --------
// Each lane sorts its 8 owned experts descending, then a 4-step butterfly merge
// (XOR distances 1,2,4,8) keeps the 8 largest at every step. After 4 steps ALL
// 16 lanes hold the identical global top-8, sorted descending. Every merge step
// issues 8 INDEPENDENT shuffles -> critical path is 4 shuffle latencies, versus
// the 8-round argmax chain's 32 serial shuffles. maxPackedTopK breaks value ties
// by smaller expert; owned expert sets are disjoint so ranks are deterministic.
__device__ __forceinline__ void sort8_desc(PackedTopK a[8])
{
    // Sorting network for 8 elements (Batcher odd-even merge, 19 compare-exchange).
#define CE(i, j) do { \
        const PackedTopK hi = maxPackedTopK(a[i], a[j]); \
        const PackedTopK lo = (hi.raw == a[i].raw) ? a[j] : a[i]; \
        a[i] = hi; a[j] = lo; } while (0)
    CE(0,1); CE(2,3); CE(4,5); CE(6,7);
    CE(0,2); CE(1,3); CE(4,6); CE(5,7);
    CE(1,2); CE(5,6); CE(0,4); CE(1,5); CE(2,6); CE(3,7);
    CE(2,4); CE(3,5);
    CE(1,2); CE(3,4); CE(5,6);
#undef CE
}

// Merge this lane's descending top-8 (a[]) with the partner's descending top-8
// received over the subgroup shuffle, keeping the 8 largest (descending).
__device__ __forceinline__ void mergeKeepTop8(PackedTopK a[8], int m)
{
    PackedTopK b[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) b[i].raw = SHFL_XOR_16(a[i].raw, m);
    // a desc + b desc: c[i] = max(a[i], b[7-i]) forms a bitonic sequence, then a
    // standard bitonic halver (distances 4,2,1) leaves the top-8 sorted desc.
    PackedTopK c[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = maxPackedTopK(a[i], b[7 - i]);
#define HALVE(i, j) do { \
        const PackedTopK hi = maxPackedTopK(c[i], c[j]); \
        const PackedTopK lo = (hi.raw == c[i].raw) ? c[j] : c[i]; \
        c[i] = hi; c[j] = lo; } while (0)
    HALVE(0,4); HALVE(1,5); HALVE(2,6); HALVE(3,7);
    HALVE(0,2); HALVE(1,3); HALVE(4,6); HALVE(5,7);
    HALVE(0,1); HALVE(2,3); HALVE(4,5); HALVE(6,7);
#undef HALVE
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = c[i];
}


template <typename scalar_t, int WAVES_PER_CTA, int VPT, int MAX_K>
__launch_bounds__(WAVES_PER_CTA * 64) __global__
void fusedSoftmaxTopk16(const scalar_t* __restrict__ input,
                        scalar_t* __restrict__ output,
                        int* __restrict__ indices,
                        const int num_rows, const int num_experts)
{
    static_assert(MAX_K > 0 && MAX_K <= 16, "MAX_K must be in [1, 16]");
    constexpr int SUBW = 16;
    constexpr int VEC  = 4;
    constexpr bool VECTORIZED_LAYOUT = (VPT % VEC) == 0;
    const int lane  = threadIdx.x & (SUBW - 1);        // 0..15 within subgroup
    const int sub   = (threadIdx.x >> 4) & 3;          // 0..3 subgroup within warp
    const int warp  = threadIdx.y;                     // 0..WAVES_PER_CTA-1
    const int row   = (blockIdx.x * WAVES_PER_CTA + warp) * 4 + sub;
    if (row >= num_rows) return;

    const scalar_t* __restrict__ row_ptr = input + (size_t)row * num_experts;

    // For E=128/VPT=8 each lane owns two adjacent float4 vectors: experts
    // [lane*4, lane*4+3] and [64+lane*4, 64+lane*4+3]. Full aligned rows use
    // two float4 loads; partial rows use guarded scalar loads with the same map.
    float vals[VPT];
    const bool use_vector_load = VECTORIZED_LAYOUT
        && sizeof(scalar_t) == sizeof(float)
        && num_experts == VPT * SUBW
        && ((reinterpret_cast<size_t>(row_ptr) & (alignof(float4) - 1)) == 0);
    if (use_vector_load) {
        const float4* __restrict__ vec_ptr =
            reinterpret_cast<const float4*>(row_ptr);
#pragma unroll
        for (int g = 0; g < VPT / VEC; ++g) {
            const float4 v = vec_ptr[g * SUBW + lane];
            vals[g * VEC] = v.x; vals[g * VEC + 1] = v.y;
            vals[g * VEC + 2] = v.z; vals[g * VEC + 3] = v.w;
        }
    } else {
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            const int e = VECTORIZED_LAYOUT
                ? (i / VEC) * (SUBW * VEC) + lane * VEC + (i % VEC)
                : lane + i * SUBW;
            vals[i] = (e < num_experts) ? (float)row_ptr[e] : -FLT_MAX;
        }
    }

    float sel_lg[MAX_K];
    int   sel_id[MAX_K];
#pragma unroll
    for (int kk = 0; kk < MAX_K; ++kk) {
        float best = -FLT_MAX;
        int   best_e = num_experts;
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            const int e = VECTORIZED_LAYOUT
                ? (i / VEC) * (SUBW * VEC) + lane * VEC + (i % VEC)
                : lane + i * SUBW;
            if (vals[i] > best || (vals[i] == best && e < best_e)) { best = vals[i]; best_e = e; }
        }
        // Overlay value/index with one int64 so shuffle needs no explicit
        // shift/or packing and the pair has guaranteed 8-byte alignment.
        PackedTopK packed;
        packed.fields.value = best;
        packed.fields.expert = best_e;
        int64_t p = packed.raw;
#pragma unroll
        for (int m = 8; m > 0; m >>= 1) {
            PackedTopK other;
            other.raw = SHFL_XOR_16(p, m);
            const float ov = other.fields.value;
            const int oe = other.fields.expert;
            if (ov > best || (ov == best && oe < best_e)) {
                best = ov; best_e = oe; p = other.raw;
            }
        }
        sel_lg[kk] = best; sel_id[kk] = best_e;        // replicated across the 16 lanes
        // Mask the winner with a STATIC-indexed compare (never vals[dynamic]):
        // a dynamic index forces vals[] to local memory -> bandwidth cliff.
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            const int e = VECTORIZED_LAYOUT
                ? (i / VEC) * (SUBW * VEC) + lane * VEC + (i % VEC)
                : lane + i * SUBW;
            if (e == best_e) vals[i] = -FLT_MAX;
        }
    }

    // Renormalize over the k winners (rank-0 is the row max). Parallelize across
    // the k lanes: each winner lane does its own exp; sum via fast intra-16 xor
    // reduction; each lane emits its own weight+index (no serial lane-0 tail).
    const float mx = sel_lg[0];
    float myval = (lane < MAX_K) ? __builtin_expf(sel_lg[lane] - mx) : 0.f;
    float sum = myval;
#pragma unroll
    for (int m = 8; m > 0; m >>= 1) sum += SHFL_XOR_16(sum, m);
    const float inv = __builtin_mxc_rcpf(sum);
    if (lane < MAX_K) {
        output[row * MAX_K + lane]  = (scalar_t)(myval * inv);
        indices[row * MAX_K + lane] = sel_id[lane];
    }
}

// Iteration 7 (SUBW=8 bitonic): halve the cross-lane shuffle COUNT, which is the
// real ceiling (read-probe with 6 shuffles = 329 GB/s; both 32-shuffle kernels =
// 171). Here each row is owned by an 8-lane subgroup (16 experts/lane, 4 float4
// loads), so the top-8 all-reduce is only 3 butterfly steps = 24 shuffles (vs 32
// at SUBW=16), and a 64-wide warp runs 8 rows. #16-lane fast-shuffle domain still
// holds (m=1,2,4 all intra-8). Bit-identical top-k semantics.
#define SHFL_XOR_8(var, m) __shfl_xor_sync(0xffffffffffffffffULL, (var), (m), 8)

__device__ __forceinline__ void sort8_desc_packed(PackedTopK a[8])
{
#define CE(i, j) do { \
        const PackedTopK hi = maxPackedTopK(a[i], a[j]); \
        const PackedTopK lo = (hi.raw == a[i].raw) ? a[j] : a[i]; \
        a[i] = hi; a[j] = lo; } while (0)
    // Batcher odd-even mergesort network for 8 elements (19 compare-exchange).
    CE(0,1);CE(2,3);CE(4,5);CE(6,7);
    CE(0,2);CE(1,3);CE(4,6);CE(5,7);
    CE(1,2);CE(5,6);CE(0,4);CE(1,5);CE(2,6);CE(3,7);
    CE(2,4);CE(3,5);CE(1,2);CE(3,4);CE(5,6);
#undef CE
}

// Merge two descending local top-8 lists, keeping the descending top-8 union.
// This is the same bitonic half-merge used after a subgroup exchange, but it
// also lets the first local sort overlap the second pair of async loads.
__device__ __forceinline__ void mergeKeepTop8Local_w8(
    PackedTopK a[8], const PackedTopK b[8])
{
    PackedTopK c[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = maxPackedTopK(a[i], b[7 - i]);
#define HALVE(i, j) do { \
        const PackedTopK hi = maxPackedTopK(c[i], c[j]); \
        const PackedTopK lo = (hi.raw == c[i].raw) ? c[j] : c[i]; \
        c[i] = hi; c[j] = lo; } while (0)
    HALVE(0,4); HALVE(1,5); HALVE(2,6); HALVE(3,7);
    HALVE(0,2); HALVE(1,3); HALVE(4,6); HALVE(5,7);
    HALVE(0,1); HALVE(2,3); HALVE(4,5); HALVE(6,7);
#undef HALVE
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = c[i];
}

__device__ __forceinline__ void sort8_desc_packed_pred(PackedTopK a[8])
{
#define CEP(i, j) compareExchangePackedDesc(a[i],a[j])
    CEP(0,1);CEP(2,3);CEP(4,5);CEP(6,7);
    CEP(0,2);CEP(1,3);CEP(4,6);CEP(5,7);
    CEP(1,2);CEP(5,6);CEP(0,4);CEP(1,5);CEP(2,6);CEP(3,7);
    CEP(2,4);CEP(3,5);CEP(1,2);CEP(3,4);CEP(5,6);
#undef CEP
}

__device__ __forceinline__ void mergeKeepTop8Local_w8_pred(
    PackedTopK a[8], const PackedTopK b[8])
{
    PackedTopK c[8];
#pragma unroll
    for (int i=0;i<8;++i) c[i]=maxPackedTopK(a[i],b[7-i]);
#define HALVEP(i, j) compareExchangePackedDesc(c[i],c[j])
    HALVEP(0,4);HALVEP(1,5);HALVEP(2,6);HALVEP(3,7);
    HALVEP(0,2);HALVEP(1,3);HALVEP(4,6);HALVEP(5,7);
    HALVEP(0,1);HALVEP(2,3);HALVEP(4,5);HALVEP(6,7);
#undef HALVEP
#pragma unroll
    for (int i=0;i<8;++i) a[i]=c[i];
}

// Merge this lane's descending top-8 with the partner's descending top-8 (over an
// 8-lane subgroup shuffle), keeping the 8 largest descending.
__device__ __forceinline__ void mergeKeepTop8_w8(PackedTopK a[8], int m)
{
    PackedTopK b[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) b[i].raw = SHFL_XOR_8(a[i].raw, m);
    PackedTopK c[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = maxPackedTopK(a[i], b[7 - i]);
#define HALVE(i, j) do { \
        const PackedTopK hi = maxPackedTopK(c[i], c[j]); \
        const PackedTopK lo = (hi.raw == c[i].raw) ? c[j] : c[i]; \
        c[i] = hi; c[j] = lo; } while (0)
    HALVE(0,4); HALVE(1,5); HALVE(2,6); HALVE(3,7);
    HALVE(0,2); HALVE(1,3); HALVE(4,6); HALVE(5,7);
    HALVE(0,1); HALVE(2,3); HALVE(4,5); HALVE(6,7);
#undef HALVE
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = c[i];
}

#define SHFL_XOR_4(var, m) __shfl_xor_sync(0xffffffffffffffffULL, (var), (m), 4)

__device__ __forceinline__ void mergeKeepTop8_w4(PackedTopK a[8], int m)
{
    PackedTopK b[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) b[i].raw = SHFL_XOR_4(a[i].raw, m);
    PackedTopK c[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = maxPackedTopK(a[i], b[7 - i]);
#define HALVE4(i, j) do { \
        const PackedTopK hi = maxPackedTopK(c[i], c[j]); \
        const PackedTopK lo = (hi.raw == c[i].raw) ? c[j] : c[i]; \
        c[i] = hi; c[j] = lo; } while (0)
    HALVE4(0,4); HALVE4(1,5); HALVE4(2,6); HALVE4(3,7);
    HALVE4(0,2); HALVE4(1,3); HALVE4(4,6); HALVE4(5,7);
    HALVE4(0,1); HALVE4(2,3); HALVE4(4,5); HALVE4(6,7);
#undef HALVE4
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = c[i];
}

template <typename scalar_t, int WAVES_PER_CTA, int MAX_K, int PRED_MODE=0>
__launch_bounds__(WAVES_PER_CTA * 64) __global__
void fusedSoftmaxTopk4Bitonic(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int* __restrict__ indices,
    const int num_rows, const int num_experts)
{
    static_assert(sizeof(scalar_t) == sizeof(float), "w4 bitonic is FP32-only");
    static_assert(MAX_K == 8, "w4 bitonic is exact only for top-8");
    constexpr int SUBW = 4;
    const int lane = threadIdx.x & 3;
    const int sub = (threadIdx.x >> 2) & 15;
    const int warp = threadIdx.y;
    const int row = (blockIdx.x * WAVES_PER_CTA + warp) * 16 + sub;
    if (row >= num_rows) return;

    const scalar_t* __restrict__ row_ptr = input + (size_t)row * num_experts;
    const int b0=lane*4, b1=16+b0, b2=32+b0, b3=48+b0;
    const int b4=64+b0, b5=80+b0, b6=96+b0, b7=112+b0;
    float4 v0,v1;
    PackedTopK a[8], b[8];
#define FILL8(dst, va, bx, vb, by) do { \
    dst[0]=makePackedTopK(va.x,bx); dst[1]=makePackedTopK(va.y,bx+1); \
    dst[2]=makePackedTopK(va.z,bx+2); dst[3]=makePackedTopK(va.w,bx+3); \
    dst[4]=makePackedTopK(vb.x,by); dst[5]=makePackedTopK(vb.y,by+1); \
    dst[6]=makePackedTopK(vb.z,by+2); dst[7]=makePackedTopK(vb.w,by+3); \
} while (0)
#define SORT8_LOCAL(x) do { if constexpr (PRED_MODE==1 || PRED_MODE==2) sort8_desc_packed_pred(x); \
                            else sort8_desc_packed(x); } while (0)
#define MERGE8_LOCAL(x,y,stage) do { if constexpr (PRED_MODE==1 || PRED_MODE==3 || \
        (PRED_MODE==4 && stage>=1)) mergeKeepTop8Local_w8_pred(x,y); \
                               else mergeKeepTop8Local_w8(x,y); } while (0)
    ldg_b128_reg_async(cast_b128(&v0)[0], const_cast<scalar_t*>(row_ptr+b0), true, true);
    ldg_b128_reg_async(cast_b128(&v1)[0], const_cast<scalar_t*>(row_ptr+b1), true, true);
    arrive_gvmcnt(0);
    FILL8(a,v0,b0,v1,b1);

    ldg_b128_reg_async(cast_b128(&v0)[0], const_cast<scalar_t*>(row_ptr+b2), true, true);
    ldg_b128_reg_async(cast_b128(&v1)[0], const_cast<scalar_t*>(row_ptr+b3), true, true);
    SORT8_LOCAL(a);
    arrive_gvmcnt(0);
    FILL8(b,v0,b2,v1,b3);

    ldg_b128_reg_async(cast_b128(&v0)[0], const_cast<scalar_t*>(row_ptr+b4), true, true);
    ldg_b128_reg_async(cast_b128(&v1)[0], const_cast<scalar_t*>(row_ptr+b5), true, true);
    SORT8_LOCAL(b); MERGE8_LOCAL(a,b,0);
    arrive_gvmcnt(0);
    FILL8(b,v0,b4,v1,b5);

    ldg_b128_reg_async(cast_b128(&v0)[0], const_cast<scalar_t*>(row_ptr+b6), true, true);
    ldg_b128_reg_async(cast_b128(&v1)[0], const_cast<scalar_t*>(row_ptr+b7), true, true);
    SORT8_LOCAL(b); MERGE8_LOCAL(a,b,1);
    arrive_gvmcnt(0);
    FILL8(b,v0,b6,v1,b7);
    SORT8_LOCAL(b); MERGE8_LOCAL(a,b,2);
#undef FILL8
#undef SORT8_LOCAL
#undef MERGE8_LOCAL

#pragma unroll
    for (int m=1; m<SUBW; m<<=1) mergeKeepTop8_w4(a,m);

    const float row_max=a[0].fields.value;
    const float e0=__builtin_expf(a[lane].fields.value-row_max);
    const float e1=__builtin_expf(a[lane+4].fields.value-row_max);
    float sum=e0+e1;
#pragma unroll
    for (int m=1; m<SUBW; m<<=1) sum+=SHFL_XOR_4(sum,m);
    const float inv=__builtin_mxc_rcpf(sum);
    output[row*8+lane]=(scalar_t)(e0*inv);
    indices[row*8+lane]=a[lane].fields.expert;
    output[row*8+lane+4]=(scalar_t)(e1*inv);
    indices[row*8+lane+4]=a[lane+4].fields.expert;
}

template <typename scalar_t, int WAVES_PER_CTA, int MAX_K>
__launch_bounds__(WAVES_PER_CTA * 64) __global__
void fusedSoftmaxTopk8Bitonic(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int* __restrict__ indices,
    const int num_rows, const int num_experts)
{
    static_assert(MAX_K == 4 || MAX_K == 8 || MAX_K == 16,
                  "MAX_K must be one of 4, 8, 16");
    constexpr int SUBW = 8;
    const int lane = threadIdx.x & (SUBW - 1);       // 0..7
    const int sub  = (threadIdx.x >> 3) & 7;         // 0..7  (8 rows/warp)
    const int warp = threadIdx.y;
    const int row  = (blockIdx.x * WAVES_PER_CTA + warp) * 8 + sub;
    if (row >= num_rows) return;

    const scalar_t* __restrict__ row_ptr = input + (size_t)row * num_experts;
    // Lane owns four float4 chunks, lane-interleaved for coalescing:
    // experts [lane*4..+3], [32+lane*4..], [64+lane*4..], [96+lane*4..].
    const int b0 = lane * 4, b1 = 32 + b0, b2 = 64 + b0, b3 = 96 + b0;
    float4 v0, v1, v2, v3;
    PackedTopK a[8], b[8];
    const bool use_vector_load = sizeof(scalar_t) == sizeof(float)
        && num_experts == 128
        && ((reinterpret_cast<size_t>(row_ptr) & (alignof(float4) - 1)) == 0);
    if (use_vector_load) {
        ldg_b128_reg_async(cast_b128(&v0)[0], const_cast<scalar_t*>(row_ptr + b0), true, true);
        ldg_b128_reg_async(cast_b128(&v1)[0], const_cast<scalar_t*>(row_ptr + b1), true, true);
        ldg_b128_reg_async(cast_b128(&v2)[0], const_cast<scalar_t*>(row_ptr + b2), true, true);
        ldg_b128_reg_async(cast_b128(&v3)[0], const_cast<scalar_t*>(row_ptr + b3), true, true);
        // The first two requests are now consumable while v2/v3 remain in
        // flight.  Hide their remaining latency behind the first local sort.
        arrive_gvmcnt(2);
        a[0]=makePackedTopK(v0.x,b0); a[1]=makePackedTopK(v0.y,b0+1);
        a[2]=makePackedTopK(v0.z,b0+2); a[3]=makePackedTopK(v0.w,b0+3);
        a[4]=makePackedTopK(v1.x,b1); a[5]=makePackedTopK(v1.y,b1+1);
        a[6]=makePackedTopK(v1.z,b1+2); a[7]=makePackedTopK(v1.w,b1+3);
        sort8_desc_packed(a);
        arrive_gvmcnt(0);
        b[0]=makePackedTopK(v2.x,b2); b[1]=makePackedTopK(v2.y,b2+1);
        b[2]=makePackedTopK(v2.z,b2+2); b[3]=makePackedTopK(v2.w,b2+3);
        b[4]=makePackedTopK(v3.x,b3); b[5]=makePackedTopK(v3.y,b3+1);
        b[6]=makePackedTopK(v3.z,b3+2); b[7]=makePackedTopK(v3.w,b3+3);
        sort8_desc_packed(b);
    } else {
        const float nf = -FLT_MAX;
        v0.x=(b0<num_experts)?(float)row_ptr[b0]:nf;   v0.y=(b0+1<num_experts)?(float)row_ptr[b0+1]:nf;
        v0.z=(b0+2<num_experts)?(float)row_ptr[b0+2]:nf; v0.w=(b0+3<num_experts)?(float)row_ptr[b0+3]:nf;
        v1.x=(b1<num_experts)?(float)row_ptr[b1]:nf;   v1.y=(b1+1<num_experts)?(float)row_ptr[b1+1]:nf;
        v1.z=(b1+2<num_experts)?(float)row_ptr[b1+2]:nf; v1.w=(b1+3<num_experts)?(float)row_ptr[b1+3]:nf;
        v2.x=(b2<num_experts)?(float)row_ptr[b2]:nf;   v2.y=(b2+1<num_experts)?(float)row_ptr[b2+1]:nf;
        v2.z=(b2+2<num_experts)?(float)row_ptr[b2+2]:nf; v2.w=(b2+3<num_experts)?(float)row_ptr[b2+3]:nf;
        v3.x=(b3<num_experts)?(float)row_ptr[b3]:nf;   v3.y=(b3+1<num_experts)?(float)row_ptr[b3+1]:nf;
        v3.z=(b3+2<num_experts)?(float)row_ptr[b3+2]:nf; v3.w=(b3+3<num_experts)?(float)row_ptr[b3+3]:nf;
        a[0]=makePackedTopK(v0.x,b0); a[1]=makePackedTopK(v0.y,b0+1);
        a[2]=makePackedTopK(v0.z,b0+2); a[3]=makePackedTopK(v0.w,b0+3);
        a[4]=makePackedTopK(v1.x,b1); a[5]=makePackedTopK(v1.y,b1+1);
        a[6]=makePackedTopK(v1.z,b1+2); a[7]=makePackedTopK(v1.w,b1+3);
        b[0]=makePackedTopK(v2.x,b2); b[1]=makePackedTopK(v2.y,b2+1);
        b[2]=makePackedTopK(v2.z,b2+2); b[3]=makePackedTopK(v2.w,b2+3);
        b[4]=makePackedTopK(v3.x,b3); b[5]=makePackedTopK(v3.y,b3+1);
        b[6]=makePackedTopK(v3.z,b3+2); b[7]=makePackedTopK(v3.w,b3+3);
        sort8_desc_packed(a);
        sort8_desc_packed(b);
    }
    mergeKeepTop8Local_w8(a, b);

    // Retained disabled max-only probe: after local top-8 construction, a[0] is
    // this lane's maximum; reduce that value over the subgroup and intentionally
    // write the same diagnostic maximum to every output slot.
    // Diagnostic toggle disabled: getenv + function-local static guards are
    // host-only and cannot appear in device code (breaks the device link).
    // Production always took the default (false) path.
    constexpr bool max_only = false;
    if (max_only) {
        float mx = a[0].fields.value;
#pragma unroll
        for (int m = 1; m < SUBW; m <<= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffffffffffULL, mx, m, 8));
        if (lane < MAX_K) {
            output[row * MAX_K + lane]  = (scalar_t)mx;
            indices[row * MAX_K + lane] = lane;
        }
        return;
    }

    // 3-step butterfly all-reduce keeping top-8 -> every lane holds global top-8.
#pragma unroll
    for (int m = 1; m < SUBW; m <<= 1) mergeKeepTop8_w8(a, m);

    const float row_max = a[0].fields.value;
    // Softmax-tail cost isolation. The full top-8 select is free (24 vs 32
    // shuffles, 6 vs 123 compares all pin at 171 GB/s); the only thing the
    // 329 GB/s read-probe skips is this expf renorm. Two levers here:
    //   MOE_STK_NOEXP=1 -> skip expf entirely (diagnostic ceiling).
    //   default         -> each active lane computes ONLY its own expf, then a
    //                      3-step butterfly sum. 8 expf/row instead of 64 (every
    //                      lane no longer recomputes all k terms for the sum).
    constexpr bool no_exp = false;  // diagnostic getenv toggle disabled (see above)
    if (no_exp) {
        if (lane < MAX_K) {
            output[row * MAX_K + lane]  = (scalar_t)(a[lane].fields.value - row_max);
            indices[row * MAX_K + lane] = a[lane].fields.expert;
        }
        return;
    }
    const float my_e = (lane < MAX_K)
        ? __builtin_expf(a[lane].fields.value - row_max) : 0.f;
    float sum = my_e;
#pragma unroll
    for (int m = 1; m < SUBW; m <<= 1)
        sum += __shfl_xor_sync(0xffffffffffffffffULL, sum, m, 8);
    const float inv = __builtin_mxc_rcpf(sum);
    if (lane < MAX_K) {
        output[row * MAX_K + lane]  = (scalar_t)(my_e * inv);
        indices[row * MAX_K + lane] = a[lane].fields.expert;
    }
}

// Iteration 8 (f32-only bitonic): the kernel is shuffle-THROUGHPUT bound (16-lane
// read-probe = 842 GB/s, full top-8 = 171, and bitonic's shorter critical path
// didn't help -> it's shuffle count*width, not latency). The prior bitonic moved
// 64-bit PackedTopK; a 64-bit __shfl is ~2x a 32-bit one. Here the butterfly
// carries ONLY the 32-bit float value. Indices are recovered afterward with a
// shuffle-free local scatter: random fp32 logits never tie, so each global winner
// value has exactly one owning lane, which writes its (expert,weight) to that
// rank slot. Halves cross-lane shuffle width in the throughput-bound region.
__device__ __forceinline__ void sort8_desc_f32(float a[8])
{
#define CEF(i, j) do { const float hi = fmaxf(a[i], a[j]); \
        const float lo = fminf(a[i], a[j]); a[i] = hi; a[j] = lo; } while (0)
    CEF(0,1);CEF(2,3);CEF(4,5);CEF(6,7);
    CEF(0,2);CEF(1,3);CEF(4,6);CEF(5,7);
    CEF(1,2);CEF(5,6);CEF(0,4);CEF(1,5);CEF(2,6);CEF(3,7);
    CEF(2,4);CEF(3,5);CEF(1,2);CEF(3,4);CEF(5,6);
#undef CEF
}

__device__ __forceinline__ void mergeKeepTop8_f32(float a[8], int m)
{
    float b[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        b[i] = __shfl_xor_sync(0xffffffffffffffffULL, a[i], m, 16);
    float c[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = fmaxf(a[i], b[7 - i]);
#define HALVEF(i, j) do { const float hi = fmaxf(c[i], c[j]); \
        const float lo = fminf(c[i], c[j]); c[i] = hi; c[j] = lo; } while (0)
    HALVEF(0,4);HALVEF(1,5);HALVEF(2,6);HALVEF(3,7);
    HALVEF(0,2);HALVEF(1,3);HALVEF(4,6);HALVEF(5,7);
    HALVEF(0,1);HALVEF(2,3);HALVEF(4,5);HALVEF(6,7);
#undef HALVEF
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = c[i];
}

template <typename scalar_t, int WAVES_PER_CTA, int MAX_K>
__launch_bounds__(WAVES_PER_CTA * 64) __global__
void fusedSoftmaxTopk16Vpt8BitonicF32(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int* __restrict__ indices,
    const int num_rows, const int num_experts)
{
    static_assert(MAX_K == 4 || MAX_K == 8 || MAX_K == 16,
                  "MAX_K must be one of 4, 8, 16");
    constexpr int SUBW = 16;
    constexpr int UPPER_BASE = 64;
    const int lane = threadIdx.x & (SUBW - 1);
    const int sub  = (threadIdx.x >> 4) & 3;
    const int warp = threadIdx.y;
    const int row  = (blockIdx.x * WAVES_PER_CTA + warp) * 4 + sub;
    if (row >= num_rows) return;

    const scalar_t* __restrict__ row_ptr = input + (size_t)row * num_experts;
    const int base0 = lane * 4;
    const int base1 = UPPER_BASE + base0;

    float4 v0, v1;
    const bool use_vector_load = sizeof(scalar_t) == sizeof(float)
        && num_experts == 128
        && ((reinterpret_cast<size_t>(row_ptr) & (alignof(float4) - 1)) == 0);
    if (use_vector_load) {
        ldg_b128_reg_async(cast_b128(&v0)[0], const_cast<scalar_t*>(row_ptr + base0), true, true);
        ldg_b128_reg_async(cast_b128(&v1)[0], const_cast<scalar_t*>(row_ptr + base1), true, true);
        arrive_gvmcnt(0);
    } else {
        v0.x = (base0 < num_experts) ? (float)row_ptr[base0] : -FLT_MAX;
        v0.y = (base0 + 1 < num_experts) ? (float)row_ptr[base0 + 1] : -FLT_MAX;
        v0.z = (base0 + 2 < num_experts) ? (float)row_ptr[base0 + 2] : -FLT_MAX;
        v0.w = (base0 + 3 < num_experts) ? (float)row_ptr[base0 + 3] : -FLT_MAX;
        v1.x = (base1 < num_experts) ? (float)row_ptr[base1] : -FLT_MAX;
        v1.y = (base1 + 1 < num_experts) ? (float)row_ptr[base1 + 1] : -FLT_MAX;
        v1.z = (base1 + 2 < num_experts) ? (float)row_ptr[base1 + 2] : -FLT_MAX;
        v1.w = (base1 + 3 < num_experts) ? (float)row_ptr[base1 + 3] : -FLT_MAX;
    }

    // Keep local (value, expert) pairs for index recovery; reduce values only.
    const float lv[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };
    const int   le[8] = { base0, base0+1, base0+2, base0+3, base1, base1+1, base1+2, base1+3 };
    float a[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };
    sort8_desc_f32(a);
#pragma unroll
    for (int m = 1; m < SUBW; m <<= 1) mergeKeepTop8_f32(a, m);
    // a[0..7] now hold the global top-8 values (descending) in every lane.

    const float row_max = a[0];
    float sum = 0.f;
#pragma unroll
    for (int kk = 0; kk < MAX_K; ++kk) sum += __builtin_expf(a[kk] - row_max);
    const float inv = __builtin_mxc_rcpf(sum);

    // Shuffle-free index scatter: each lane owns the winners it originally held.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int r = 0; r < MAX_K; ++r) {
            if (lv[i] == a[r]) {
                output[row * MAX_K + r]  = (scalar_t)(__builtin_expf(a[r] - row_max) * inv);
                indices[row * MAX_K + r] = le[i];
            }
        }
    }
}

// Iteration 6 (bitonic top-8): replaces the 8-round serial argmax-and-mask chain
// with a butterfly all-reduce whose state is a per-lane descending top-8 list.
// The pure-read async ceiling is ~329 GB/s while the serial kernel runs at 171,
// so this path is compute-bound; the merge issues 8 INDEPENDENT shuffles per
// step (4 steps) instead of 32 serial ones, and renorm needs NO shuffle because
// every lane ends holding the full global top-8. Bit-identical top-k semantics:
// softmax monotonic -> select on raw logits; rank-0 is the row max; exp over k.
template <typename scalar_t, int WAVES_PER_CTA, int MAX_K>
__launch_bounds__(WAVES_PER_CTA * 64) __global__
void fusedSoftmaxTopk16Vpt8Bitonic(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int* __restrict__ indices,
    const int num_rows, const int num_experts)
{
    static_assert(MAX_K == 4 || MAX_K == 8 || MAX_K == 16,
                  "MAX_K must be one of 4, 8, 16");
    constexpr int SUBW = 16;
    constexpr int UPPER_BASE = 64;
    const int lane = threadIdx.x & (SUBW - 1);
    const int sub  = (threadIdx.x >> 4) & 3;
    const int warp = threadIdx.y;
    const int row  = (blockIdx.x * WAVES_PER_CTA + warp) * 4 + sub;
    if (row >= num_rows) return;

    const scalar_t* __restrict__ row_ptr = input + (size_t)row * num_experts;
    const int base0 = lane * 4;
    const int base1 = UPPER_BASE + base0;

    float4 v0, v1;
    const bool use_vector_load = sizeof(scalar_t) == sizeof(float)
        && num_experts == 128
        && ((reinterpret_cast<size_t>(row_ptr) & (alignof(float4) - 1)) == 0);
    if (use_vector_load) {
        ldg_b128_reg_async(cast_b128(&v0)[0],
                           const_cast<scalar_t*>(row_ptr + base0), true, true);
        ldg_b128_reg_async(cast_b128(&v1)[0],
                           const_cast<scalar_t*>(row_ptr + base1), true, true);
        arrive_gvmcnt(1);
        (void)0;
        arrive_gvmcnt(0);
    } else {
        v0.x = (base0 < num_experts) ? (float)row_ptr[base0] : -FLT_MAX;
        v0.y = (base0 + 1 < num_experts) ? (float)row_ptr[base0 + 1] : -FLT_MAX;
        v0.z = (base0 + 2 < num_experts) ? (float)row_ptr[base0 + 2] : -FLT_MAX;
        v0.w = (base0 + 3 < num_experts) ? (float)row_ptr[base0 + 3] : -FLT_MAX;
        v1.x = (base1 < num_experts) ? (float)row_ptr[base1] : -FLT_MAX;
        v1.y = (base1 + 1 < num_experts) ? (float)row_ptr[base1 + 1] : -FLT_MAX;
        v1.z = (base1 + 2 < num_experts) ? (float)row_ptr[base1 + 2] : -FLT_MAX;
        v1.w = (base1 + 3 < num_experts) ? (float)row_ptr[base1 + 3] : -FLT_MAX;
    }

    // This lane's 8 owned experts, sorted descending.
    PackedTopK a[8];
    a[0] = makePackedTopK(v0.x, base0);
    a[1] = makePackedTopK(v0.y, base0 + 1);
    a[2] = makePackedTopK(v0.z, base0 + 2);
    a[3] = makePackedTopK(v0.w, base0 + 3);
    a[4] = makePackedTopK(v1.x, base1);
    a[5] = makePackedTopK(v1.y, base1 + 1);
    a[6] = makePackedTopK(v1.z, base1 + 2);
    a[7] = makePackedTopK(v1.w, base1 + 3);
    sort8_desc(a);

    // Butterfly all-reduce with "keep top-8" (commutative + associative) leaves
    // every lane holding the identical global descending top-8.
#pragma unroll
    for (int m = 1; m < SUBW; m <<= 1) mergeKeepTop8(a, m);

    // Renorm over the k winners; rank-0 is the row max. No cross-lane comm.
    const float row_max = a[0].fields.value;
    float sum = 0.f;
#pragma unroll
    for (int kk = 0; kk < MAX_K; ++kk) sum += __builtin_expf(a[kk].fields.value - row_max);
    const float inv = __builtin_mxc_rcpf(sum);
    if (lane < MAX_K) {
        const float w = __builtin_expf(a[lane].fields.value - row_max) * inv;
        output[row * MAX_K + lane]  = (scalar_t)w;
        indices[row * MAX_K + lane] = a[lane].fields.expert;
    }
}

// Read-ceiling probe for the 16-lane tournament layout: uses the EXACT async
// ldg_b128 load path as the production kernel, but performs ONE argmax (not
// MAX_K masked rounds). Isolates this layout's async-read + minimal-compute
// ceiling from the 8-round serial argmax cost. Output is intentionally wrong.
template <typename scalar_t, int WAVES_PER_CTA, int MAX_K>
__launch_bounds__(WAVES_PER_CTA * 64) __global__
void fusedSoftmaxTopk16Vpt8ReadProbe(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int* __restrict__ indices,
    const int num_rows, const int num_experts)
{
    constexpr int SUBW = 16;
    constexpr int UPPER_BASE = 64;
    const int lane = threadIdx.x & (SUBW - 1);
    const int sub  = (threadIdx.x >> 4) & 3;
    const int warp = threadIdx.y;
    const int row  = (blockIdx.x * WAVES_PER_CTA + warp) * 4 + sub;
    if (row >= num_rows) return;
    const scalar_t* __restrict__ row_ptr = input + (size_t)row * num_experts;
    const int base0 = lane * 4;
    const int base1 = UPPER_BASE + base0;
    float4 v0, v1;
    ldg_b128_reg_async(cast_b128(&v0)[0],
                       const_cast<scalar_t*>(row_ptr + base0), true, true);
    ldg_b128_reg_async(cast_b128(&v1)[0],
                       const_cast<scalar_t*>(row_ptr + base1), true, true);
    arrive_gvmcnt(1);
    PackedTopK head0 = argmax4(v0, base0);
    arrive_gvmcnt(0);
    PackedTopK head1 = argmax4(v1, base1);
    PackedTopK w = subgroupArgmax16(maxPackedTopK(head0, head1));
    if (lane < MAX_K) {
        output[row * MAX_K + lane] = (scalar_t)w.fields.value;
        indices[row * MAX_K + lane] = w.fields.expert;
    }
}

// Iteration 5 (row-batched): each 16-lane subgroup owns RPS rows and issues ALL
// RPS*2 plain float4 loads up front, then runs RPS independent tournaments. The
// baseline kernel reads only 32 B/thread (two float4) before a 32-shuffle serial
// chain -> too few loads in flight to saturate DRAM (pure-read probe ~227 GB/s
// while a contiguous read of the same [T,128] tensor sustains ~372). Batching
// rows raises memory-level parallelism without changing per-row math; ownership
// map and results are bit-identical to the baseline tournament.
template <typename scalar_t, int WAVES_PER_CTA, int MAX_K, int RPS>
__launch_bounds__(WAVES_PER_CTA * 64) __global__
void fusedSoftmaxTopk16Vpt8BatchTournament(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int* __restrict__ indices,
    const int num_rows, const int num_experts)
{
    static_assert(MAX_K == 4 || MAX_K == 8 || MAX_K == 16,
                  "MAX_K must be one of 4, 8, 16");
    constexpr int SUBW = 16;
    constexpr int UPPER_BASE = 64;

    const int lane = threadIdx.x & (SUBW - 1);
    const int sub  = (threadIdx.x >> 4) & 3;
    const int warp = threadIdx.y;
    const int g    = (blockIdx.x * WAVES_PER_CTA + warp) * 4 + sub;  // subgroup id
    const int row0 = g * RPS;
    if (row0 >= num_rows) return;

    const int base0 = lane * 4;
    const int base1 = UPPER_BASE + base0;
    const bool aligned = sizeof(scalar_t) == sizeof(float) && num_experts == 128
        && ((reinterpret_cast<size_t>(input) & (alignof(float4) - 1)) == 0);

    float4 v0[RPS], v1[RPS];
    bool active[RPS];
    // Phase 1: fire all loads up front (independent -> high MLP).
#pragma unroll
    for (int r = 0; r < RPS; ++r) {
        const int row = row0 + r;
        active[r] = (row < num_rows);
        const scalar_t* __restrict__ row_ptr = input + (size_t)row * num_experts;
        if (active[r] && aligned) {
            const float4* __restrict__ p = reinterpret_cast<const float4*>(row_ptr);
            v0[r] = p[lane];
            v1[r] = p[16 + lane];
        } else if (active[r]) {
            v0[r].x = (base0 < num_experts) ? (float)row_ptr[base0] : -FLT_MAX;
            v0[r].y = (base0 + 1 < num_experts) ? (float)row_ptr[base0 + 1] : -FLT_MAX;
            v0[r].z = (base0 + 2 < num_experts) ? (float)row_ptr[base0 + 2] : -FLT_MAX;
            v0[r].w = (base0 + 3 < num_experts) ? (float)row_ptr[base0 + 3] : -FLT_MAX;
            v1[r].x = (base1 < num_experts) ? (float)row_ptr[base1] : -FLT_MAX;
            v1[r].y = (base1 + 1 < num_experts) ? (float)row_ptr[base1 + 1] : -FLT_MAX;
            v1[r].z = (base1 + 2 < num_experts) ? (float)row_ptr[base1 + 2] : -FLT_MAX;
            v1[r].w = (base1 + 3 < num_experts) ? (float)row_ptr[base1 + 3] : -FLT_MAX;
        }
    }

    // Phase 2: independent tournament per row.
#pragma unroll 1
    for (int r = 0; r < RPS; ++r) {
        if (!active[r]) continue;
        const int row = row0 + r;
        PackedTopK head0 = argmax4(v0[r], base0);
        PackedTopK head1 = argmax4(v1[r], base1);
        float selected_value = -FLT_MAX;
        int selected_id = num_experts;
        float row_max = -FLT_MAX;
#pragma unroll
        for (int kk = 0; kk < MAX_K; ++kk) {
            const PackedTopK local = maxPackedTopK(head0, head1);
            const PackedTopK winner = subgroupArgmax16(local);
            const int winner_id = winner.fields.expert;
            if (kk == 0) row_max = winner.fields.value;
            if (lane == kk) { selected_value = winner.fields.value; selected_id = winner_id; }
            const int winner_lane = (winner_id & (UPPER_BASE - 1)) >> 2;
            const int winner_half = winner_id >> 6;
            const int winner_slot = winner_id & 3;
            if (lane == winner_lane) {
                if (winner_half == 0) {
                    if (winner_slot == 0) v0[r].x = -FLT_MAX;
                    if (winner_slot == 1) v0[r].y = -FLT_MAX;
                    if (winner_slot == 2) v0[r].z = -FLT_MAX;
                    if (winner_slot == 3) v0[r].w = -FLT_MAX;
                    head0 = argmax4(v0[r], base0);
                } else {
                    if (winner_slot == 0) v1[r].x = -FLT_MAX;
                    if (winner_slot == 1) v1[r].y = -FLT_MAX;
                    if (winner_slot == 2) v1[r].z = -FLT_MAX;
                    if (winner_slot == 3) v1[r].w = -FLT_MAX;
                    head1 = argmax4(v1[r], base1);
                }
            }
        }
        const float myval = (lane < MAX_K)
            ? __builtin_expf(selected_value - row_max) : 0.f;
        float sum = myval;
#pragma unroll
        for (int m = 8; m > 0; m >>= 1) sum += SHFL_XOR_16(sum, m);
        const float inv = __builtin_mxc_rcpf(sum);
        if (lane < MAX_K) {
            output[row * MAX_K + lane] = (scalar_t)(myval * inv);
            indices[row * MAX_K + lane] = selected_id;
        }
    }
}

// Specialized E=113..128 path (VPT=8): cache one argmax per float4 half and
// only recompute the half containing the winner. Each rank is retained by its
// corresponding lane, avoiding per-thread sel_lg[MAX_K]/sel_id[MAX_K] arrays.
template <typename scalar_t, int WAVES_PER_CTA, int MAX_K>
__launch_bounds__(WAVES_PER_CTA * 64) __global__
void fusedSoftmaxTopk16Vpt8Tournament(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int* __restrict__ indices,
    const int num_rows, const int num_experts)
{
    static_assert(MAX_K == 4 || MAX_K == 8 || MAX_K == 16,
                  "MAX_K must be one of 4, 8, 16");
    constexpr int SUBW = 16;
    constexpr int UPPER_BASE = 64;

    const int lane = threadIdx.x & (SUBW - 1);
    const int sub = (threadIdx.x >> 4) & 3;
    const int warp = threadIdx.y;
    const int row = (blockIdx.x * WAVES_PER_CTA + warp) * 4 + sub;
    if (row >= num_rows) return;

    const scalar_t* __restrict__ row_ptr = input + (size_t)row * num_experts;
    const int base0 = lane * 4;
    const int base1 = UPPER_BASE + base0;

    float4 v0;
    float4 v1;
    PackedTopK head0;
    PackedTopK head1;
    const bool use_vector_load = sizeof(scalar_t) == sizeof(float)
        && num_experts == 128
        && ((reinterpret_cast<size_t>(row_ptr) & (alignof(float4) - 1)) == 0);
    if (use_vector_load) {
        ldg_b128_reg_async(cast_b128(&v0)[0],
                           const_cast<scalar_t*>(row_ptr + base0), true, true);
        ldg_b128_reg_async(cast_b128(&v1)[0],
                           const_cast<scalar_t*>(row_ptr + base1), true, true);
        arrive_gvmcnt(1);
        head0 = argmax4(v0, base0);
        arrive_gvmcnt(0);
        head1 = argmax4(v1, base1);
    } else {
        v0.x = (base0 < num_experts) ? (float)row_ptr[base0] : -FLT_MAX;
        v0.y = (base0 + 1 < num_experts) ? (float)row_ptr[base0 + 1] : -FLT_MAX;
        v0.z = (base0 + 2 < num_experts) ? (float)row_ptr[base0 + 2] : -FLT_MAX;
        v0.w = (base0 + 3 < num_experts) ? (float)row_ptr[base0 + 3] : -FLT_MAX;
        v1.x = (base1 < num_experts) ? (float)row_ptr[base1] : -FLT_MAX;
        v1.y = (base1 + 1 < num_experts) ? (float)row_ptr[base1 + 1] : -FLT_MAX;
        v1.z = (base1 + 2 < num_experts) ? (float)row_ptr[base1 + 2] : -FLT_MAX;
        v1.w = (base1 + 3 < num_experts) ? (float)row_ptr[base1 + 3] : -FLT_MAX;
        head0 = argmax4(v0, base0);
        head1 = argmax4(v1, base1);
    }

    float selected_value = -FLT_MAX;
    int selected_id = num_experts;
    float row_max = -FLT_MAX;

#pragma unroll
    for (int kk = 0; kk < MAX_K; ++kk) {
        const PackedTopK local = maxPackedTopK(head0, head1);
        const PackedTopK winner = subgroupArgmax16(local);
        const int winner_id = winner.fields.expert;

        if (kk == 0) row_max = winner.fields.value;
        if (lane == kk) {
            selected_value = winner.fields.value;
            selected_id = winner_id;
        }

        // E=113..128 uses a fixed ownership map:
        // lane=(expert%64)/4, half=expert/64, slot=expert%4.
        const int winner_lane = (winner_id & (UPPER_BASE - 1)) >> 2;
        const int winner_half = winner_id >> 6;
        const int winner_slot = winner_id & 3;
        if (lane == winner_lane) {
            if (winner_half == 0) {
                if (winner_slot == 0) v0.x = -FLT_MAX;
                if (winner_slot == 1) v0.y = -FLT_MAX;
                if (winner_slot == 2) v0.z = -FLT_MAX;
                if (winner_slot == 3) v0.w = -FLT_MAX;
                head0 = argmax4(v0, base0);
            } else {
                if (winner_slot == 0) v1.x = -FLT_MAX;
                if (winner_slot == 1) v1.y = -FLT_MAX;
                if (winner_slot == 2) v1.z = -FLT_MAX;
                if (winner_slot == 3) v1.w = -FLT_MAX;
                head1 = argmax4(v1, base1);
            }
        }
    }

    const float myval = (lane < MAX_K)
        ? __builtin_expf(selected_value - row_max) : 0.f;
    float sum = myval;
#pragma unroll
    for (int m = 8; m > 0; m >>= 1) sum += SHFL_XOR_16(sum, m);
    const float inv = __builtin_mxc_rcpf(sum);
    if (lane < MAX_K) {
        output[row * MAX_K + lane] = (scalar_t)(myval * inv);
        indices[row * MAX_K + lane] = selected_id;
    }
}

// Iteration 4 (plain-load): identical math to the Vpt8 tournament but replaces
// ldg_b128_reg_async + arrive_gvmcnt with a plain synchronous float4 load pair
// (p[lane] -> experts [lane*4..+3] = base0, p[16+lane] -> base1). Isolates
// whether the async-load path (bounded outstanding requests / gvmcnt barriers)
// is what caps the pure-read probe at ~227 GB/s vs the ~372 GB/s a contiguous
// read of the same [T,128] tensor sustains.
template <typename scalar_t, int WAVES_PER_CTA, int MAX_K>
__launch_bounds__(WAVES_PER_CTA * 64) __global__
void fusedSoftmaxTopk16Vpt8PlainTournament(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int* __restrict__ indices,
    const int num_rows, const int num_experts)
{
    static_assert(MAX_K == 4 || MAX_K == 8 || MAX_K == 16,
                  "MAX_K must be one of 4, 8, 16");
    constexpr int SUBW = 16;
    constexpr int UPPER_BASE = 64;

    const int lane = threadIdx.x & (SUBW - 1);
    const int sub  = (threadIdx.x >> 4) & 3;
    const int warp = threadIdx.y;
    const int row  = (blockIdx.x * WAVES_PER_CTA + warp) * 4 + sub;
    if (row >= num_rows) return;

    const scalar_t* __restrict__ row_ptr = input + (size_t)row * num_experts;
    const int base0 = lane * 4;
    const int base1 = UPPER_BASE + base0;

    float4 v0, v1;
    const bool use_vector_load = sizeof(scalar_t) == sizeof(float)
        && num_experts == 128
        && ((reinterpret_cast<size_t>(row_ptr) & (alignof(float4) - 1)) == 0);
    if (use_vector_load) {
        const float4* __restrict__ p = reinterpret_cast<const float4*>(row_ptr);
        v0 = p[lane];          // experts [lane*4 .. +3]      == base0
        v1 = p[16 + lane];     // experts [64+lane*4 .. +3]   == base1
    } else {
        v0.x = (base0 < num_experts) ? (float)row_ptr[base0] : -FLT_MAX;
        v0.y = (base0 + 1 < num_experts) ? (float)row_ptr[base0 + 1] : -FLT_MAX;
        v0.z = (base0 + 2 < num_experts) ? (float)row_ptr[base0 + 2] : -FLT_MAX;
        v0.w = (base0 + 3 < num_experts) ? (float)row_ptr[base0 + 3] : -FLT_MAX;
        v1.x = (base1 < num_experts) ? (float)row_ptr[base1] : -FLT_MAX;
        v1.y = (base1 + 1 < num_experts) ? (float)row_ptr[base1 + 1] : -FLT_MAX;
        v1.z = (base1 + 2 < num_experts) ? (float)row_ptr[base1 + 2] : -FLT_MAX;
        v1.w = (base1 + 3 < num_experts) ? (float)row_ptr[base1 + 3] : -FLT_MAX;
    }
    PackedTopK head0 = argmax4(v0, base0);
    PackedTopK head1 = argmax4(v1, base1);

    float selected_value = -FLT_MAX;
    int selected_id = num_experts;
    float row_max = -FLT_MAX;

#pragma unroll
    for (int kk = 0; kk < MAX_K; ++kk) {
        const PackedTopK local = maxPackedTopK(head0, head1);
        const PackedTopK winner = subgroupArgmax16(local);
        const int winner_id = winner.fields.expert;

        if (kk == 0) row_max = winner.fields.value;
        if (lane == kk) {
            selected_value = winner.fields.value;
            selected_id = winner_id;
        }

        const int winner_lane = (winner_id & (UPPER_BASE - 1)) >> 2;
        const int winner_half = winner_id >> 6;
        const int winner_slot = winner_id & 3;
        if (lane == winner_lane) {
            if (winner_half == 0) {
                if (winner_slot == 0) v0.x = -FLT_MAX;
                if (winner_slot == 1) v0.y = -FLT_MAX;
                if (winner_slot == 2) v0.z = -FLT_MAX;
                if (winner_slot == 3) v0.w = -FLT_MAX;
                head0 = argmax4(v0, base0);
            } else {
                if (winner_slot == 0) v1.x = -FLT_MAX;
                if (winner_slot == 1) v1.y = -FLT_MAX;
                if (winner_slot == 2) v1.z = -FLT_MAX;
                if (winner_slot == 3) v1.w = -FLT_MAX;
                head1 = argmax4(v1, base1);
            }
        }
    }

    const float myval = (lane < MAX_K)
        ? __builtin_expf(selected_value - row_max) : 0.f;
    float sum = myval;
#pragma unroll
    for (int m = 8; m > 0; m >>= 1) sum += SHFL_XOR_16(sum, m);
    const float inv = __builtin_mxc_rcpf(sum);
    if (lane < MAX_K) {
        output[row * MAX_K + lane] = (scalar_t)(myval * inv);
        indices[row * MAX_K + lane] = selected_id;
    }
}

// Iteration 3 (smem-staged): the Vpt8Tournament read pattern issues two STRIDED
// float4 loads per warp (v0 -> f4 {0-15,32-47,64-79,96-111}, v1 -> complement),
// each spanning a 2 KB window at 50%% transaction utilization -> the pure-read
// probe caps at ~227 GB/s while a contiguous read of the same [T,128] tensor
// sustains ~372 GB/s. Here the full 64-lane warp first stages its 4 rows to
// shared memory with a FULLY-COALESCED contiguous load (consecutive lanes ->
// consecutive float4), then each 16-lane subgroup runs the identical cheap
// tournament out of shared. Read DRAM traffic is unchanged but coalescing is
// perfect; correctness is bit-identical to the register path.
template <typename scalar_t, int WAVES_PER_CTA, int MAX_K>
__launch_bounds__(WAVES_PER_CTA * 64) __global__
void fusedSoftmaxTopk16Vpt8SmemTournament(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int* __restrict__ indices,
    const int num_rows, const int num_experts)
{
    static_assert(MAX_K == 4 || MAX_K == 8 || MAX_K == 16,
                  "MAX_K must be one of 4, 8, 16");
    constexpr int SUBW = 16;
    constexpr int UPPER_BASE = 64;
    constexpr int ROWF4 = 32;                 // float4 per 128-expert row

    const int lane = threadIdx.x & (SUBW - 1);
    const int sub  = (threadIdx.x >> 4) & 3;
    const int warp = threadIdx.y;
    const int tid  = threadIdx.x;             // 0..63
    const int rowblock = (blockIdx.x * WAVES_PER_CTA + warp) * 4;

    __shared__ float4 smem[WAVES_PER_CTA][4 * ROWF4];   // 4 rows x 32 f4 = 8 KB/warp

    const bool aligned = (num_experts == 128)
        && ((reinterpret_cast<size_t>(input) & (alignof(float4) - 1)) == 0);

    if (aligned) {
        // Coalesced stage: 64 lanes load 128 float4 (4 contiguous rows) as two
        // contiguous 1 KB bursts. Lane t -> smem slot t and t+64.
        const float4* __restrict__ in4 = reinterpret_cast<const float4*>(input);
        const size_t gbase = (size_t)rowblock * ROWF4;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int sidx  = tid + j * 64;             // 0..127
            const int g_row = rowblock + (sidx >> 5);   // 32 f4 per row
            float4 val;
            if (g_row < num_rows) {
                val = in4[gbase + sidx];
            } else {
                val.x = val.y = val.z = val.w = -FLT_MAX;
            }
            smem[warp][sidx] = val;
        }
    }
    __syncthreads();

    const int row = rowblock + sub;
    if (row >= num_rows) return;

    const int base0 = lane * 4;
    const int base1 = UPPER_BASE + base0;

    // Read this subgroup's row back from shared (v0 = experts [lane*4..+3],
    // v1 = experts [64+lane*4..+3]); identical ownership map to the reg path.
    float4 v0 = smem[warp][sub * ROWF4 + lane];
    float4 v1 = smem[warp][sub * ROWF4 + 16 + lane];
    PackedTopK head0 = argmax4(v0, base0);
    PackedTopK head1 = argmax4(v1, base1);

    float selected_value = -FLT_MAX;
    int selected_id = num_experts;
    float row_max = -FLT_MAX;

#pragma unroll
    for (int kk = 0; kk < MAX_K; ++kk) {
        const PackedTopK local = maxPackedTopK(head0, head1);
        const PackedTopK winner = subgroupArgmax16(local);
        const int winner_id = winner.fields.expert;

        if (kk == 0) row_max = winner.fields.value;
        if (lane == kk) {
            selected_value = winner.fields.value;
            selected_id = winner_id;
        }

        const int winner_lane = (winner_id & (UPPER_BASE - 1)) >> 2;
        const int winner_half = winner_id >> 6;
        const int winner_slot = winner_id & 3;
        if (lane == winner_lane) {
            if (winner_half == 0) {
                if (winner_slot == 0) v0.x = -FLT_MAX;
                if (winner_slot == 1) v0.y = -FLT_MAX;
                if (winner_slot == 2) v0.z = -FLT_MAX;
                if (winner_slot == 3) v0.w = -FLT_MAX;
                head0 = argmax4(v0, base0);
            } else {
                if (winner_slot == 0) v1.x = -FLT_MAX;
                if (winner_slot == 1) v1.y = -FLT_MAX;
                if (winner_slot == 2) v1.z = -FLT_MAX;
                if (winner_slot == 3) v1.w = -FLT_MAX;
                head1 = argmax4(v1, base1);
            }
        }
    }

    const float myval = (lane < MAX_K)
        ? __builtin_expf(selected_value - row_max) : 0.f;
    float sum = myval;
#pragma unroll
    for (int m = 8; m > 0; m >>= 1) sum += SHFL_XOR_16(sum, m);
    const float inv = __builtin_mxc_rcpf(sum);
    if (lane < MAX_K) {
        output[row * MAX_K + lane] = (scalar_t)(myval * inv);
        indices[row * MAX_K + lane] = selected_id;
    }
}

// ============================================================================
// One expert per thread; each 64-lane warp sorts its slice descending in a single
// parallel bitonic pass (SortElement<float,uint64_t>), then warp 0 merges the
// per-warp top-k candidates (warps_per_row*topk <= 64) in a second sort.
// Softmax is monotonic in the logit -> select by RAW logit (no full-row exp).
// The rank-0 logit IS the row max, so exp+renorm run over just the k winners.
//   Requires: warps_per_row * topk <= 64  (checked at launch).
#define SHFL_IDX_64(var, src) __shfl_sync(0xffffffffffffffffULL, (var), (src), 64)
template <typename scalar_t>
__global__ void fusedSoftmaxTopkSort(const scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ output,
                                     int* __restrict__ indices,
                                     const int num_rows, const int num_experts,
                                     const int topk, const int rows_per_cta)
{
    const int warps_per_row  = (num_experts + 63) / 64;
    const int threads_per_row = warps_per_row * 64;
    const int row_in_block = threadIdx.x / threads_per_row;
    const int thread_row   = blockIdx.x * rows_per_cta + row_in_block;
    if (thread_row >= num_rows) return;

    const int tid_in_row  = threadIdx.x % threads_per_row;
    const int warp_in_row = tid_in_row / 64;
    const int lane_id     = threadIdx.x % 64;

    // One expert per thread (coalesced: adjacent lanes -> adjacent experts).
    float logit = -3.4e38f;
    if (tid_in_row < num_experts)
        logit = (float)input[(size_t)thread_row * num_experts + tid_in_row];

    float iw[2];
    iw[0] = logit;
    iw[1] = 0.0f;
    *((int64_t*)iw) |= ((int64_t)tid_in_row << 32);

    // Full 64-lane descending bitonic sort (ties -> smaller expert index first).
    SortElement<float, uint64_t>(iw, lane_id);

    // Stash each warp's local top-k into shared memory.
    extern __shared__ int64_t s_cand[];        // [rows_per_cta * warps_per_row * topk]
    const int cand_base = (row_in_block * warps_per_row + warp_in_row) * topk;
    if (lane_id < topk)
        s_cand[cand_base + lane_id] = *(int64_t*)iw;
    __syncthreads();

    // Warp 0 of each row merges warps_per_row*topk (<=64) candidates.
    if (warp_in_row == 0) {
        float m2[2];
        m2[0] = -3.4e38f;
        m2[1] = 0.0f;
        *((int64_t*)m2) |= ((int64_t)(num_experts + lane_id) << 32);   // unique sentinel
        const int ncand = warps_per_row * topk;
        if (lane_id < ncand)
            *(int64_t*)m2 = s_cand[row_in_block * warps_per_row * topk + lane_id];

        SortElement<float, uint64_t>(m2, lane_id);   // lanes 0..topk-1 = final top-k desc

        const int64_t res = *(int64_t*)m2;
        const float sel_logit  = get_weight<float>(res);
        const int   sel_expert = (int)(res >> 32);

        const float max_logit = SHFL_IDX_64(sel_logit, 0);            // rank-0 = row max
        const float e = (lane_id < topk) ? __builtin_expf(sel_logit - max_logit) : 0.0f;
        float sum = e;
#pragma unroll
        for (int m = 32; m > 0; m >>= 1) sum += SHFL_XOR_64(sum, m);  // sum over k winners
        const float inv = __builtin_mxc_rcpf(sum);

        if (lane_id < topk) {
            output[(size_t)topk * thread_row + lane_id]  = (scalar_t)(e * inv);
            indices[(size_t)topk * thread_row + lane_id] = sel_expert;
        }
    }
}

}   // namespace moe_softmax_topk

#undef SHFL_XOR_SYNC_WIDTH
