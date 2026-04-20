#include "rms_norm.cuh"
#include "rotary_emb.cuh"
#include "bmm.cuh"
namespace fused_mla {

//Firstly we assume that each TBLOCK have 256 threads, and MLA parameters are default template values
template<typename scalar_t, int Q_LEN=16, int NUM_LOCAL_HEADS=128, int KV_LORA_RANK=512, int QK_ROPE_HEAD_DIM=64>
__global__ void fused_mla_absorb_rotary_emb(
    const scalar_t* q_nope_out, // [128, bs, 512], dtype=bf16
    const scalar_t* q_pe, // [bs, 128, 64]
    const scalar_t* latent_cache, // [bs, 576], dtype=bf16
    const float* cos_sin_cache,
    const int64_t* positions, //[bs]
     const scalar_t* norm_weight, // [512]
    scalar_t* q_input, //[bs, 128, 576], dtype=bf16
    scalar_t* k_input, //[bs, 1, 576], dtype=bf16
    scalar_t* v_input // [bs, 1, 512]
) {
    uint32_t bidx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (bidx < Q_LEN/4 * NUM_LOCAL_HEADS) {
        //do t1/t2
        bidx = 4*bidx;

        uint32_t m = bidx + tid/QK_ROPE_HEAD_DIM;
        uint32_t j = m / NUM_LOCAL_HEADS;
        uint32_t i = m % NUM_LOCAL_HEADS;

        uint32_t q_nope_offset = i * (Q_LEN * KV_LORA_RANK) + j * KV_LORA_RANK;
        uint32_t q_input_offset = j * (NUM_LOCAL_HEADS * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)) + i * (KV_LORA_RANK + QK_ROPE_HEAD_DIM);
        uint32_t q_pe_offset = j * (NUM_LOCAL_HEADS * QK_ROPE_HEAD_DIM) + i * QK_ROPE_HEAD_DIM;

        //Preload q_nope_out
        //Each TBLOCK have 512 scalar, and each thread read 2 scalar, packed into a float
        float r = ((float*)(q_nope_out + q_nope_offset))[tid];

        //Do rotary emb:
        uint32_t tidx = tid % QK_ROPE_HEAD_DIM;
        rotary_emb(
            tidx,
            j,
            q_pe + q_pe_offset,
            cos_sin_cache,
            positions,
            q_input + q_input_offset + KV_LORA_RANK
        );
        ((float*)(q_input + q_input_offset))[tid] = r;
    } else {
        bidx -= Q_LEN/4 * NUM_LOCAL_HEADS;
        bidx *= 4;

        uint32_t m = bidx + tid/QK_ROPE_HEAD_DIM;
        uint32_t k_input_offset = m * (KV_LORA_RANK + QK_ROPE_HEAD_DIM);
        uint32_t v_input_offset = m * KV_LORA_RANK;
        tid =  tid % QK_ROPE_HEAD_DIM;
        rms_norm(tid, norm_weight, latent_cache+k_input_offset, k_input+k_input_offset, v_input+v_input_offset);
        rotary_emb(
            tid,
            m,
            latent_cache + k_input_offset + KV_LORA_RANK,
            cos_sin_cache,
            positions,
            k_input+k_input_offset+KV_LORA_RANK
        );
    }
}

//The overall process of matrix absorption to rotational coding
template<typename scalar_t, int NUM_LOCAL_HEADS=128, int KV_LORA_RANK=512, int QK_NOPE_HEAD_DIM=128, int QK_ROPE_HEAD_DIM=64>
__global__ void fused_absorb_mla(
    const int Q_LEN,
    const scalar_t* q, // [bs, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM+QK_ROPE_HEAD_DIM]=>[16, 128, 192], dtype=bf16
    const scalar_t* w_kc, //[NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK]=>[128, 128, 512], dtype=bf16
    const scalar_t* latent_cache, // [bs, KV_LORA_RANK+QK_ROPE_HEAD_DIM]=>[16, 576], dtype=bf16
    const int latent_cache_stride,
    const float* cos_sin_cache,
    const int64_t* positions, //[bs]
    const scalar_t* norm_weight, // [512]
    scalar_t* q_input, //[bs, 128, 576], dtype=bf16
    scalar_t* k_input, //[bs, 1, 576], dtype=bf16
    scalar_t* v_input // [bs, 1, 512]
) {
    uint32_t bidx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (bidx < (Q_LEN + 15)/16*2*NUM_LOCAL_HEADS) {
        do_bmm<scalar_t, 1, 8, 4, NUM_LOCAL_HEADS, KV_LORA_RANK, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM>(Q_LEN, q, w_kc, q_input, tid, bidx);
    } else if (bidx < ((Q_LEN+3)/4 + (Q_LEN + 15)/16*2) * NUM_LOCAL_HEADS) {
        //do t1/t2
        bidx -= (Q_LEN + 15)/16*2*NUM_LOCAL_HEADS;
        bidx = 4*bidx;

        //#pragma unroll
        //for (int n = 0; n < 4; ++n) {
        uint32_t m = bidx + tid/QK_ROPE_HEAD_DIM;
        uint32_t j = m / NUM_LOCAL_HEADS;
        if (j < Q_LEN) {
            uint32_t i = m % NUM_LOCAL_HEADS;

            uint32_t q_input_offset = j * (NUM_LOCAL_HEADS * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)) + i * (KV_LORA_RANK + QK_ROPE_HEAD_DIM);
            uint32_t q_pe_offset = j * (NUM_LOCAL_HEADS * (QK_NOPE_HEAD_DIM+QK_ROPE_HEAD_DIM)) + i * (QK_NOPE_HEAD_DIM+QK_ROPE_HEAD_DIM)+ QK_NOPE_HEAD_DIM;

            tid = tid % QK_ROPE_HEAD_DIM;
            rotary_emb(
                tid,
                j,
                q + q_pe_offset,
                cos_sin_cache,
                positions,
                q_input + q_input_offset + KV_LORA_RANK
            );
        }
    } else {
        bidx -= ((Q_LEN+3)/4 + (Q_LEN + 15)/16*2) * NUM_LOCAL_HEADS;
        bidx *= 4;

        uint32_t m = bidx + tid/QK_ROPE_HEAD_DIM;
        if (m < Q_LEN) {
            uint32_t k_input_offset = m * (KV_LORA_RANK + QK_ROPE_HEAD_DIM);
            uint32_t latent_cache_offset = m * latent_cache_stride;
            uint32_t v_input_offset = m * KV_LORA_RANK;
            tid =  tid % QK_ROPE_HEAD_DIM;
            rms_norm(tid, norm_weight, latent_cache+latent_cache_offset, k_input+k_input_offset, v_input+v_input_offset);
            rotary_emb(
                tid,
                m,
                latent_cache + latent_cache_offset + KV_LORA_RANK,
                cos_sin_cache,
                positions,
                k_input+k_input_offset+KV_LORA_RANK
            );
        }
    }
}


template<typename scalar_t, int Q_LEN=16, int KV_LORA_RANK=512>
__global__ void rms_norm_test(const scalar_t* w, scalar_t* v_input) {
    uint32_t bidx = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t offset = bidx * KV_LORA_RANK;
    rms_norm(tid, w, v_input+offset, v_input+offset);
}

template<typename scalar_t, int Q_LEN=16, int NUM_LOCAL_HEADS=128, int QK_ROPE_HEAD_DIM=64>
__global__ void rotary_emb_test(scalar_t* q_pe, scalar_t* k_pe, const float* cos_sin_cache, const int64_t* positions) {
    uint32_t bidx = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (bidx < Q_LEN * NUM_LOCAL_HEADS) {
        uint32_t pid = bidx / NUM_LOCAL_HEADS;
        uint32_t pe_offset = QK_ROPE_HEAD_DIM*bidx;
        rotary_emb(tid, pid, q_pe + pe_offset, cos_sin_cache, positions, q_pe + pe_offset);
    } else {
        bidx -= Q_LEN * NUM_LOCAL_HEADS;
        uint32_t pe_offset = QK_ROPE_HEAD_DIM*bidx;
        rotary_emb(tid, bidx, k_pe + pe_offset, cos_sin_cache, positions, k_pe + pe_offset);
    }
}

template<typename scalar_t, int Q_LEN=16, int NUM_LOCAL_HEADS=128, int KV_LORA_RANK=512, int QK_NOPE_HEAD_DIM=128, int QK_ROPE_HEAD_DIM=64>
__global__ void bmm_test(const scalar_t* q, const scalar_t* w_kc, scalar_t* q_input) {
    //Input q: 128 x 16 x 128
    //Input w_kc: 128 x 128 x 512
    //Output q_input: 16 x 128 x (512+64), and q_input[:, :, 512:] will not be verified as we will not calculate them
    uint32_t tid = threadIdx.x;
    uint32_t wave_idx = tid / 64;
    uint32_t wave_group_idx = (tid % 64) / 16;
    uint32_t wave_group_lane = (tid % 64) % 16;
    uint32_t bidx = blockIdx.x;
    int q_offset = bidx * (Q_LEN * QK_NOPE_HEAD_DIM);
    int w_kc_offset = bidx * (QK_NOPE_HEAD_DIM * KV_LORA_RANK);
    do_mma(Q_LEN, q + q_offset, w_kc + w_kc_offset, wave_idx, wave_group_idx, wave_group_lane, bidx, 0,0, q_input);
}

template<typename scalar_t, int Q_LEN=16, int NUM_LOCAL_HEADS=128, int KV_LORA_RANK=512, int QK_NOPE_HEAD_DIM=128, int QK_ROPE_HEAD_DIM=64>
__global__ void gemm_test(const scalar_t* q, const scalar_t* w_kc, scalar_t* q_input) {
    //Only test 16 x 128 x 512 for simplicity
    //Input q: 16 x 128
    //Input w_kc: 128 x 512
    //Output q_input: 16 x (512+64), and q_input[:, 512:] will not be verified as we will not calculate them
    uint32_t tid = threadIdx.x;
    uint32_t wave_idx = tid / 64;
    uint32_t wave_group_idx = (tid % 64) / 16;
    uint32_t wave_group_lane = (tid % 64) % 16;
    do_mma<scalar_t,
        Q_LEN/16,
        QK_NOPE_HEAD_DIM/16,
        KV_LORA_RANK/128,
        1,
        KV_LORA_RANK,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM
        >(Q_LEN, q, w_kc, wave_idx, wave_group_idx, wave_group_lane, 0, 0,0, q_input);
}


template<typename scalar_t, int NUM_LOCAL_HEADS=128, int KV_LORA_RANK=512, int QK_NOPE_HEAD_DIM=128, int QK_ROPE_HEAD_DIM=64>
__global__ void fused_mla_RMS_rotary_emb(
    const int Q_LEN,
    scalar_t* q,
    scalar_t* latent_cache,
    const float* cos_sin_cache,
    const int64_t* positions,
    const scalar_t* norm_weight,
    scalar_t* kv_a
) {
    uint32_t bidx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (bidx < Q_LEN*NUM_LOCAL_HEADS/4) {
        bidx = bidx * 4;
        uint32_t m = bidx + tid/QK_ROPE_HEAD_DIM;
        uint32_t j = m / NUM_LOCAL_HEADS;
        uint32_t i = m % NUM_LOCAL_HEADS;

        uint32_t q_pe_offset = j * (NUM_LOCAL_HEADS * (QK_NOPE_HEAD_DIM+QK_ROPE_HEAD_DIM)) + i * (QK_NOPE_HEAD_DIM+QK_ROPE_HEAD_DIM)+ QK_NOPE_HEAD_DIM;

        tid = tid % QK_ROPE_HEAD_DIM;

        rotary_emb(
            tid,
            j,
            q + q_pe_offset,
            cos_sin_cache,
            positions,
            q + q_pe_offset
        );
    } else if (bidx < Q_LEN*NUM_LOCAL_HEADS/4 + (Q_LEN+3)/4) {
        bidx -= Q_LEN*NUM_LOCAL_HEADS/4;
        bidx *= 4;

        uint32_t m = bidx + tid/QK_ROPE_HEAD_DIM;

        if (m < Q_LEN) {
            uint32_t latent_cache_offset = m * (KV_LORA_RANK + QK_ROPE_HEAD_DIM);
            uint32_t kv_a_offset = m * KV_LORA_RANK;
            tid =  tid % QK_ROPE_HEAD_DIM;
            rms_norm(tid, norm_weight, latent_cache+latent_cache_offset, kv_a+kv_a_offset, latent_cache+latent_cache_offset);
            rotary_emb(
                tid,
                m,
                latent_cache+latent_cache_offset+KV_LORA_RANK,
                cos_sin_cache,
                positions,
                latent_cache+latent_cache_offset+KV_LORA_RANK
            );
        }
    }
}

template<typename scalar_t, int NUM_LOCAL_HEADS=128, int QK_NOPE_HEAD_DIM=128, int QK_ROPE_HEAD_DIM=64, int V_HEAD_DIM=128>
__global__ void fused_mla_normal_kv_element_wise(
    const int32_t K_LEN,
    const scalar_t* kv,
    const scalar_t* latent_cache,
    scalar_t* k,
    scalar_t* v
) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int k_front_total = K_LEN * NUM_LOCAL_HEADS * (QK_NOPE_HEAD_DIM / 8);
    if (idx < k_front_total) {
        const int k_front_group_size = QK_NOPE_HEAD_DIM / 8;
        int i = idx / (NUM_LOCAL_HEADS * k_front_group_size);      // 3201
        int j = (idx % (NUM_LOCAL_HEADS * k_front_group_size)) / k_front_group_size; // 128
        int k_group = (idx % k_front_group_size);      // 128/8=16

        int kv_offset = i *( NUM_LOCAL_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)/8) + j * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)/8 + k_group;

        // kv load
        float4 val = *((float4*) kv + kv_offset);//kv_vec[j * 64 + k_group]; // 256/4=64

        // k offset
        int k_offset = i * (NUM_LOCAL_HEADS * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)/8) + j * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)/8 + k_group; // 192/4=48
        // k_vec[k_offset] = val;
        *((float4*)k + k_offset) = val;
    }

    // // k_pe
    int k_back_start = k_front_total;
    int k_back_total = K_LEN * NUM_LOCAL_HEADS * (QK_ROPE_HEAD_DIM / 8); // 64col
    if (idx >= k_back_start && idx < (k_back_start + k_back_total)) {
        int local_idx = idx - k_back_start;
        const int k_back_group_size = QK_ROPE_HEAD_DIM / 8;
        int i = local_idx / (128 * k_back_group_size);      // 3201
        int j = (local_idx % (128 * k_back_group_size)) / k_back_group_size; // 128
        int k_group = (local_idx % k_back_group_size);      // 64/8=8

        // latent_cache load
        int lc_offset = i * (576 / 8) + (512 / 8) + k_group; // 576/8=72
        float4 val = *((float4*)latent_cache + lc_offset);

        // k offset
        int k_offset = i * (128 * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)/8) + j * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)/8 + QK_NOPE_HEAD_DIM/8 + k_group; // 128/8=16
        *((float4*)k + k_offset) = val;
    }

    int v_start = k_back_start + k_back_total;
    int v_total = (K_LEN * NUM_LOCAL_HEADS * V_HEAD_DIM) / 8; // 128col
    if (idx >= v_start && idx < (v_start + v_total)) {
        int local_idx = idx - v_start;
        const int v_group_size = V_HEAD_DIM / 8;
        int i = local_idx / (NUM_LOCAL_HEADS * v_group_size);       // 128
        int j = (local_idx % (NUM_LOCAL_HEADS * v_group_size)) / v_group_size;
        int v_group = local_idx % v_group_size; // 128/8=16

        int kv_offset = i * NUM_LOCAL_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)/8 + j * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)/8 + QK_NOPE_HEAD_DIM/8 + v_group;

        float4 val = *((float4*)kv + kv_offset); // 256/8=32

        int v_offset = i * NUM_LOCAL_HEADS * V_HEAD_DIM / 8 + j * V_HEAD_DIM / 8 + v_group;
        *((float4*)v + v_offset) = val;
    }
}

}
