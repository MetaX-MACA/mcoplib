import torch
import sys
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.op as op
except ImportError:
    op = None

class Rotary_embedding_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size_list = config.get("batch_size_list", [128, 64, 256, 32])
        self.q_head_num = config.get("q_head_num", 32)
        self.kv_head_num = config.get("kv_head_num", 8)
        self.head_size = config.get("head_size", 128)
        self.max_seq_len = config.get("max_seq_len", 2048)
        self.rope_offset = config.get("rope_offset", 0)
        self.num_seqs = len(self.batch_size_list)
        self.batch_size = sum(self.batch_size_list)
        self.total_head_num = self.q_head_num + 2 * self.kv_head_num

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        shape_str = f"(Tokens:{self.batch_size} Heads:{self.total_head_num} Dim:{self.head_size})"
        state.add_summary("Shape", shape_str)
        qkv_elems = self.batch_size * self.total_head_num * self.head_size
        state.add_element_count(qkv_elems)
        element_size = 2 if self.dtype == torch.float16 or self.dtype == torch.bfloat16 else 4
        rw_bytes = (qkv_elems * 2) * element_size
        cos_sin_elems = self.max_seq_len * self.head_size * 2
        rw_bytes += cos_sin_elems * 4
        state.add_global_memory_reads(qkv_elems * element_size + cos_sin_elems * 4)
        state.add_global_memory_writes(qkv_elems * element_size)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            packed_qkv = torch.randn(self.batch_size, self.total_head_num, self.head_size, dtype=self.dtype, device=dev)
            cos = torch.randn(self.max_seq_len, self.head_size, dtype=torch.float32, device=dev)
            sin = torch.randn(self.max_seq_len, self.head_size, dtype=torch.float32, device=dev)
            out = torch.empty_like(packed_qkv)
            q_len = torch.tensor(self.batch_size_list, dtype=torch.int32, device=dev)
            accum_q_lens = torch.tensor([0] + self.batch_size_list, dtype=torch.int32, device=dev).cumsum(0, dtype=torch.int32)
            cache_lens = torch.zeros(self.num_seqs, dtype=torch.int32, device=dev)
        return self.make_launcher(dev_id, op.rotary_embedding, 
                                  packed_qkv, 
                                  q_len, 
                                  accum_q_lens, 
                                  cache_lens, 
                                  cos, 
                                  sin, 
                                  out, 
                                  self.q_head_num, 
                                  self.kv_head_num, 
                                  self.rope_offset)

    def _torch_rope_impl(self, packed_qkv, cos, sin, seq_lens):
        out = torch.empty_like(packed_qkv)
        start_token_idx = 0
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        for i, seq_len in enumerate(seq_lens):
            end_token_idx = start_token_idx + seq_len
            current_tokens = packed_qkv[start_token_idx:end_token_idx]
            q = current_tokens[:, :self.q_head_num, :].float()
            k = current_tokens[:, self.q_head_num:self.q_head_num + self.kv_head_num, :].float()
            v = current_tokens[:, self.q_head_num + self.kv_head_num:, :].float()
            cur_cos = cos[:seq_len, :].unsqueeze(1)
            cur_sin = sin[:seq_len, :].unsqueeze(1)
            q_out = (q * cur_cos) + (rotate_half(q) * cur_sin)
            k_out = (k * cur_cos) + (rotate_half(k) * cur_sin)
            rotated_part = torch.cat([q_out, k_out, v], dim=1).to(packed_qkv.dtype)
            out[start_token_idx:end_token_idx] = rotated_part
            start_token_idx = end_token_idx
        return out

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        packed_qkv = torch.randn(self.batch_size, self.total_head_num, self.head_size, dtype=self.dtype, device=dev)
        cos = torch.randn(self.max_seq_len, self.head_size, dtype=torch.float32, device=dev)
        sin = torch.randn(self.max_seq_len, self.head_size, dtype=torch.float32, device=dev)
        out_op = torch.empty_like(packed_qkv)
        q_len = torch.tensor(self.batch_size_list, dtype=torch.int32, device=dev)
        accum_q_lens = torch.tensor([0] + self.batch_size_list, dtype=torch.int32, device=dev).cumsum(0, dtype=torch.int32)
        cache_lens = torch.zeros(self.num_seqs, dtype=torch.int32, device=dev)
        op.rotary_embedding(
            packed_qkv,
            q_len,
            accum_q_lens,
            cache_lens,
            cos,
            sin,
            out_op,
            self.q_head_num,
            self.kv_head_num,
            self.rope_offset
        )
        out_ref = self._torch_rope_impl(packed_qkv.clone(), cos, sin, self.batch_size_list)
        return self.check_diff(out_op, out_ref)