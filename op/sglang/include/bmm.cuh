#include "mma_common.cuh"

namespace fused_mla {
    template<typename scalar_t, int BLOCKS_M=1, int BLOCKS_K=8, int BLOCKS_N=1, int NUM_LOCAL_HEADS=128, int KV_LORA_RANK=512, int QK_NOPE_HEAD_DIM=128, int QK_ROPE_HEAD_DIM=64>
    __device__ __forceinline__ void do_mma_column_major(
        const int q_len,
        const scalar_t* shared_q, //[q_len, QK_NOPE_HEAD_DIM]
        const scalar_t* w_kc, //[QK_NOPE_HEAD_DIM, KV_LORA_RANK]
        const uint32_t wave_idx,
        const uint32_t wave_group_idx,
        const uint32_t wave_group_lane,
        const uint32_t local_head_idx,
        const uint32_t mdx,
        const uint32_t ndx,
        scalar_t* q_out
    ) {
        //BLOCKS_N = 4, As there is only few registers for mma, we should finish one col of B before loading other cols
        //So A will be load 4 times
        float C[2][4];
        scalar_t b_cache[2][4];
        scalar_t a_cache[4];
        scalar_t out[2];
        //#pragma unroll
        //for (int ndx = 0; ndx < BLOCKS_N; ndx++) {
            //Initialize C
            #pragma unroll
            for (int x = 0; x < 2; x++) {
                #pragma unroll
                for (int y = 0; y < 4; y++) {
                    C[x][y] = 0;
                }
            }
            #pragma unroll
            for (int kdx = 0; kdx < BLOCKS_K; kdx++) {
                uint32_t b_offset = kdx * 16                    //K offset, every mma will process 16 k lines
                    + wave_idx * 2 * 16 * QK_NOPE_HEAD_DIM      //wave_index in n axis, each wave load 32 values in n axis
                    + wave_group_idx * 4                        //K offset, every mma group process 4 k lines
                    + wave_group_lane * 2 * QK_NOPE_HEAD_DIM;   //each lane in a wave group will load 2 values along n axis and 4 values along k axis
                    //+ ndx * 2 * 16 * 4;                         //each thread loads 2, each wave group load 32, 4 waves load 128
                //Load 4 times to fill the b_cache

                *((PackTypeInt2*)b_cache) = *(PackTypeInt2*)(w_kc + b_offset);
                *((PackTypeInt2*)(b_cache) + 1) = *(PackTypeInt2*)(w_kc + b_offset + QK_NOPE_HEAD_DIM);
                // if (wave_idx == 0 && wave_group_idx == 0 && wave_group_lane == 0 && ndx == 0) {
                //     printf("kiter %d, Load B k = %d, %f,%f\n", kdx, k, __bfloat162float(load_kn[0]), __bfloat162float(load_kn[1]));
                // }

                //Load a
                //Assume that BLOCKS_M=1
                //TODO: if BLOCKS_M > 1, we should add a M loop here

                //In case if there is no padding for shared_q
                uint32_t a_offset = wave_group_lane * QK_NOPE_HEAD_DIM      //each thread in a wave_group calculates one M
                    + wave_group_idx * 4 + kdx * 4 * 4;
                *(PackTypeInt2*)a_cache = *(PackTypeInt2*)(shared_q + a_offset);

                //Do mma
                mma_16x16x16<scalar_t>(*(PackTypeInt2*)a_cache, *(PackTypeInt2*)b_cache[0], *(PackTypeInt4*)C[0]);
                mma_16x16x16<scalar_t>(*(PackTypeInt2*)a_cache, *(PackTypeInt2*)b_cache[1], *(PackTypeInt4*)C[1]);
                // if (wave_idx == 0 && wave_group_idx == 0 && wave_group_lane == 0 && ndx == 0) {
                //     printf("kiter = %d, Load A %f,%f,%f,%f\n", kdx, __bfloat162float(a_cache[0]), __bfloat162float(a_cache[1]), __bfloat162float(a_cache[2]), __bfloat162float(a_cache[3]));
                //     printf("kiter = %d, Out C %f,%f,%f,%f\n", kdx, C[0][0], C[0][1], C[0][2], C[0][3]);
                // }
            }

            //Save result to gvm
            //q_out is shape of [q_len, NUM_LOCAL_HEADS, KV_LORA_RANK + QK_ROPE_HEAD_DIM]
            //each thread saves 8 values

            for (int m = 0; m < 4; m++) {
                #ifdef __HPCC_ARCH__
                if constexpr(std::is_same_v<scalar_t, __hpcc_bfloat16>) {
                    out[0] = __float2bfloat16_rn(C[0][m]);
                    out[1] = __float2bfloat16_rn(C[1][m]);
                } else {
                    out[0] = __float2half(C[0][m]);
                    out[1] = __float2half(C[1][m]);
                }
                #elif defined(__MACA_ARCH__)
                if constexpr(std::is_same_v<scalar_t, __maca_bfloat16>) {
                    out[0] = __float2bfloat16_rn(C[0][m]);
                    out[1] = __float2bfloat16_rn(C[1][m]);
                } else {
                    out[0] = __float2half(C[0][m]);
                    out[1] = __float2half(C[1][m]);
                }
                #endif

                uint32_t out_dim0 = wave_group_idx * 4 + m + mdx*16;
                bool pred = out_dim0 < q_len;

                if (pred) {
                    uint32_t out_dim1 = local_head_idx;
                    uint32_t out_dim2 = ndx * 4 * 2 * 16 + wave_idx * 2 * 16 + wave_group_lane * 2;
                    uint32_t offset = out_dim0 * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * NUM_LOCAL_HEADS
                        + out_dim1 * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)
                        + out_dim2;
                    *(PackType*)(q_out + offset) = *(PackType*)out;
                    // if (wave_idx == 0 && wave_group_idx == 0 && wave_group_lane == 0 && ndx == 0) {
                    //     printf("save result m = %d, offset = %d, value = %f\n", m, offset, C[0][m]);
                    // }
                }
            }
        //}
    }
    //Calculate a QLEN x 128 @ 128 x 512 matrix
    //Do not worry about the edge
    template<typename scalar_t, int BLOCKS_M=1, int BLOCKS_K=8, int BLOCKS_N=1, int NUM_LOCAL_HEADS=128, int KV_LORA_RANK=512, int QK_NOPE_HEAD_DIM=128, int QK_ROPE_HEAD_DIM=64>
    __device__ __forceinline__ void do_mma(
        const int q_len,
        const scalar_t* shared_q, //[q_len, QK_NOPE_HEAD_DIM]
        const scalar_t* w_kc, //[QK_NOPE_HEAD_DIM, KV_LORA_RANK]
        const uint32_t wave_idx,
        const uint32_t wave_group_idx,
        const uint32_t wave_group_lane,
        const uint32_t local_head_idx,
        const uint32_t mdx,
        const uint32_t ndx,
        scalar_t* q_out
    ) {
        //BLOCKS_N = 4, As there is only few registers for mma, we should finish one col of B before loading other cols
        //So A will be load 4 times
        float C[4][4];
        scalar_t b_cache[4][4];
        scalar_t load_kn[4];
        scalar_t a_cache[4];
        scalar_t out[4];
        //#pragma unroll
        //for (int ndx = 0; ndx < BLOCKS_N; ndx++) {
            //Initialize C
            #pragma unroll
            for (int x = 0; x < 4; x++) {
                #pragma unroll
                for (int y = 0; y < 4; y++) {
                    C[x][y] = 0;
                }
            }
            #pragma unroll
            for (int kdx = 0; kdx < BLOCKS_K; kdx++) {
                uint32_t b_offset = kdx * 16 * KV_LORA_RANK     //K offset, every mma will process 16 k lines
                    + wave_idx * 4 * 16                         //wave_index in n axis, each wave load 32 values in n axis
                    + wave_group_idx * 4 * KV_LORA_RANK         //K offset, every mma group process 4 k lines
                    + wave_group_lane * 4;                       //each lane in a wave group will load 2 values along n axis and 4 values along k axis
                    //+ ndx * 2 * 16 * 4;                         //each thread loads 2, each wave group load 32, 4 waves load 128
                //Load 4 times to fill the b_cache
                for (int k = 0; k < 4; k++) {
                    *((PackTypeInt2*)load_kn) = *(PackTypeInt2*)(w_kc + b_offset + k * KV_LORA_RANK);
                    b_cache[0][k] = load_kn[0];
                    b_cache[1][k] = load_kn[1];
                    b_cache[2][k] = load_kn[2];
                    b_cache[3][k] = load_kn[3];
                    // if (wave_idx == 0 && wave_group_idx == 0 && wave_group_lane == 0 && ndx == 0) {
                    //     printf("kiter %d, Load B k = %d, %f,%f\n", kdx, k, __bfloat162float(load_kn[0]), __bfloat162float(load_kn[1]));
                    // }
                }

                //Load a
                //Assume that BLOCKS_M=1
                //TODO: if BLOCKS_M > 1, we should add a M loop here

                //In case if there is no padding for shared_q
                uint32_t a_offset = wave_group_lane * QK_NOPE_HEAD_DIM      //each thread in a wave_group calculates one M
                    + wave_group_idx * 4 + kdx * 4 * 4;
                *(PackTypeInt2*)a_cache = *(PackTypeInt2*)(shared_q + a_offset);

                //Do mma
                mma_16x16x16<scalar_t>(*(PackTypeInt2*)a_cache, *(PackTypeInt2*)b_cache[0], *(PackTypeInt4*)C[0]);
                mma_16x16x16<scalar_t>(*(PackTypeInt2*)a_cache, *(PackTypeInt2*)b_cache[1], *(PackTypeInt4*)C[1]);
                mma_16x16x16<scalar_t>(*(PackTypeInt2*)a_cache, *(PackTypeInt2*)b_cache[2], *(PackTypeInt4*)C[2]);
                mma_16x16x16<scalar_t>(*(PackTypeInt2*)a_cache, *(PackTypeInt2*)b_cache[3], *(PackTypeInt4*)C[3]);
                // if (wave_idx == 0 && wave_group_idx == 0 && wave_group_lane == 0 && ndx == 0) {
                //     printf("kiter = %d, Load A %f,%f,%f,%f\n", kdx, __bfloat162float(a_cache[0]), __bfloat162float(a_cache[1]), __bfloat162float(a_cache[2]), __bfloat162float(a_cache[3]));
                //     printf("kiter = %d, Out C %f,%f,%f,%f\n", kdx, C[0][0], C[0][1], C[0][2], C[0][3]);
                // }
            }

            //Save result to gvm
            //q_out is shape of [q_len, NUM_LOCAL_HEADS, KV_LORA_RANK + QK_ROPE_HEAD_DIM]
            //each thread saves 8 values

            for (int m = 0; m < 4; m++) {
                #ifdef __HPCC_ARCH__
                if constexpr(std::is_same_v<scalar_t, __hpcc_bfloat16>) {
                    out[0] = __float2bfloat16_rn(C[0][m]);
                    out[1] = __float2bfloat16_rn(C[1][m]);
                    out[2] = __float2bfloat16_rn(C[2][m]);
                    out[3] = __float2bfloat16_rn(C[3][m]);
                } else {
                    out[0] = __float2half(C[0][m]);
                    out[1] = __float2half(C[1][m]);
                    out[2] = __float2half(C[2][m]);
                    out[3] = __float2half(C[3][m]);
                }
                #elif defined(__MACA_ARCH__)
                if constexpr(std::is_same_v<scalar_t, __maca_bfloat16>) {
                    out[0] = __float2bfloat16_rn(C[0][m]);
                    out[1] = __float2bfloat16_rn(C[1][m]);
                    out[2] = __float2bfloat16_rn(C[2][m]);
                    out[3] = __float2bfloat16_rn(C[3][m]);
                } else {
                    out[0] = __float2half(C[0][m]);
                    out[1] = __float2half(C[1][m]);
                    out[2] = __float2half(C[2][m]);
                    out[3] = __float2half(C[3][m]);
                }
                #endif

                uint32_t out_dim0 = wave_group_idx * 4 + m + mdx*16;
                bool pred = out_dim0 < q_len;

                if (pred) {
                    uint32_t out_dim1 = local_head_idx;
                    uint32_t out_dim2 = ndx * 4 * 4 * 16 + wave_idx * 4 * 16 + wave_group_lane * 4;
                    uint32_t offset = out_dim0 * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * NUM_LOCAL_HEADS
                        + out_dim1 * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)
                        + out_dim2;
                    *(PackTypeInt2*)(q_out + offset) = *(PackTypeInt2*)out;
                    // if (wave_idx == 0 && wave_group_idx == 0 && wave_group_lane == 0 && ndx == 0) {
                    //     printf("save result m = %d, offset = %d, value = %f\n", m, offset, C[0][m]);
                    // }
                }
            }
        //}
    }


    template<typename scalar_t, int BLOCKS_M=1, int BLOCKS_K=8, int BLOCKS_N=4, int NUM_LOCAL_HEADS=128, int KV_LORA_RANK=512, int QK_NOPE_HEAD_DIM=128, int QK_ROPE_HEAD_DIM=64>
    __device__ __forceinline__ void do_bmm(
        const int q_len,
        const scalar_t* q, //[q_len, QK_NOPE_HEAD_DIM]
        const scalar_t* w_kc, //[QK_NOPE_HEAD_DIM, KV_LORA_RANK]
        scalar_t* q_out,
        int tid,
        int bid
    ) {
        __shared__ scalar_t shm[16*QK_NOPE_HEAD_DIM];
        //Step1: save transposed q into shared memory
        //TODO:
        uint32_t bnxbm_size = (q_len + 15)/16*2;
        uint32_t bn, bm;
        uint32_t remain = bid % bnxbm_size;
        // if (q_len > 16) {
        bn = (remain) % 2;
        bm = (remain) / 2;
        // } else {
        //     bn = bid % 4;
        //     bm = 0;
        // }
        uint32_t hdx = bid / bnxbm_size;
        uint32_t group_id = tid / 16;    // 0~15, kindex
        uint32_t lane_id = tid % 16;     // 0~15, j start
        uint32_t j_start = lane_id * 8;
        const uint32_t Q_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;
        const scalar_t* q_ptr = q + (group_id + bm * 16) * (NUM_LOCAL_HEADS * Q_DIM) + hdx * Q_DIM + j_start;
        scalar_t* shm_ptr = shm + group_id * QK_NOPE_HEAD_DIM + j_start;
        bool pred = (group_id + bm * 16) < q_len;
        ldg_b128_reg_noasync(*((PackTypeInt4*)shm_ptr), (PackTypeInt4*)q_ptr, pred, true);
        // *reinterpret_cast<float4*>(shm_ptr) = *reinterpret_cast<const float4*>(q_ptr);
        // if (q_len > 16) {
        //     pred = group_id < q_len - 16;
        //     ldg_b128_reg_noasync(*((PackTypeInt4*)(shm_ptr+16*QK_NOPE_HEAD_DIM)), (PackTypeInt4*)(q_ptr+16*(NUM_LOCAL_HEADS * Q_DIM)), pred, true);
        //     // *reinterpret_cast<float4*>(shm_ptr+16*QK_NOPE_HEAD_DIM) = *reinterpret_cast<const float4*>(q_ptr+16*(NUM_LOCAL_HEADS * Q_DIM));
        // }
        __syncthreads();

        uint32_t wave_idx = tid / 64;
        uint32_t wave_group_idx = (tid % 64) / 16;
        uint32_t wave_group_lane = (tid % 64) % 16;
        int w_kc_offset = hdx * (QK_NOPE_HEAD_DIM * KV_LORA_RANK) + bn * 256;
        // int shm_offset = bm * 16 * QK_NOPE_HEAD_DIM;
        do_mma<scalar_t, BLOCKS_M, BLOCKS_K, BLOCKS_N, NUM_LOCAL_HEADS, KV_LORA_RANK, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM>(q_len, shm, w_kc + w_kc_offset, wave_idx, wave_group_idx, wave_group_lane, hdx, bm,bn, q_out);
        //Step2: call do_mma
        //do_mma()
    }
}
    // 65536 1 128 column-major order
    // 65536 512 1  row-major order