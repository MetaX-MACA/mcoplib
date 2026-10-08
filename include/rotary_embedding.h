#include <ATen/ATen.h>

void rotary_embedding(at::Tensor packed_qkv,
                        at::Tensor q_len, at::Tensor accum_q_lens, at::Tensor cache_lens, at::Tensor cos,
                        at::Tensor sin, const int q_head_num, const int kv_head_num, const int rope_offset = 0);