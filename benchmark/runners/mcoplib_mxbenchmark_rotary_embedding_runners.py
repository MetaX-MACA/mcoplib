import torch
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
        self.rope_dim = self.head_size // 2

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        shape_str = f"(Tokens:{self.batch_size} Heads:{self.total_head_num} Dim:{self.head_size})"
        state.add_summary("Shape", shape_str)
        qkv_elems = self.batch_size * self.total_head_num * self.head_size
        state.add_element_count(qkv_elems)
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        cos_sin_elems = self.max_seq_len * self.rope_dim * 2
        state.add_global_memory_reads(qkv_elems * element_size + cos_sin_elems * 4)
        state.add_global_memory_writes(qkv_elems * element_size)

    def _make_inputs(self, dev, seed=42):
        torch.manual_seed(seed)
        packed_qkv = torch.randn(
            self.batch_size, self.total_head_num, self.head_size,
            dtype=self.dtype, device=dev
        )
        cos = torch.randn(
            self.max_seq_len, self.rope_dim, dtype=torch.float32, device=dev
        )
        sin = torch.randn(
            self.max_seq_len, self.rope_dim, dtype=torch.float32, device=dev
        )
        q_len = torch.tensor(self.batch_size_list, dtype=torch.int32, device=dev)
        accum_q_lens = torch.tensor(
            [0] + self.batch_size_list, dtype=torch.int32, device=dev
        ).cumsum(0, dtype=torch.int32)
        cache_lens = torch.zeros(self.num_seqs, dtype=torch.int32, device=dev)
        return packed_qkv, cos, sin, q_len, accum_q_lens, cache_lens

    def _ref_impl(self, packed_qkv, cos, sin, q_len, accum_q_lens, cache_lens):
        # Op semantics (from op/rotary_embedding.cu):
        # - GPT-NeoX style: for each token in batch b at position t_in_batch,
        #   abs_pos = cache_lens[b] + t_in_batch; cos/sin indexed by abs_pos.
        # - Only Q heads [0, q_head_num) and K heads [q_head_num, q_head_num+kv_head_num)
        #   are rotated; V heads are NOT written by the op.
        # - ROPE_DIM = cos.size(-1); HALF = ROPE_DIM / 2.
        # - Rotation region per head: [rope_offset, rope_offset + ROPE_DIM).
        # - Pairs (i, i+HALF) for i in [0, HALF), using cos[i], sin[i]:
        #     out[i]           = x[i]*cos[i] - x[i+HALF]*sin[i]
        #     out[i+HALF]      = x[i+HALF]*cos[i] + x[i]*sin[i]
        out = packed_qkv.clone()
        qk_end = self.q_head_num + self.kv_head_num
        rope_dim = self.rope_dim
        half = rope_dim // 2
        offs = self.rope_offset
        for b in range(self.num_seqs):
            start = accum_q_lens[b].item()
            length = q_len[b].item()
            if length == 0:
                continue
            cache_start = cache_lens[b].item()
            for t in range(length):
                abs_pos = cache_start + t
                c = cos[abs_pos, :half]  # [half]
                s = sin[abs_pos, :half]  # [half]
                token_idx = start + t
                for head in range(qk_end):
                    x = packed_qkv[token_idx, head].float()
                    x1 = x[offs:offs + half]
                    x2 = x[offs + half:offs + 2 * half]
                    out[token_idx, head, offs:offs + half] = (x1 * c - x2 * s).to(packed_qkv.dtype)
                    out[token_idx, head, offs + half:offs + 2 * half] = (x2 * c + x1 * s).to(packed_qkv.dtype)
        return out

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            packed_qkv, cos, sin, q_len, accum_q_lens, cache_lens = self._make_inputs(dev)
        return self.make_launcher(
            dev_id, op.rotary_embedding,
            packed_qkv, q_len, accum_q_lens, cache_lens, cos, sin,
            self.q_head_num, self.kv_head_num, self.rope_offset
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        packed_qkv, cos, sin, q_len, accum_q_lens, cache_lens = self._make_inputs(dev)
        # rotary_embedding is in-place: result written back into packed_qkv.
        # Keep a clone of the original input for the reference implementation.
        packed_qkv_orig = packed_qkv.clone()
        op.rotary_embedding(
            packed_qkv, q_len, accum_q_lens, cache_lens, cos, sin,
            self.q_head_num, self.kv_head_num, self.rope_offset
        )
        torch.cuda.synchronize()
        out_ref = self._ref_impl(packed_qkv_orig, cos, sin, q_len, accum_q_lens, cache_lens)
        # Op only writes QK heads; compare only those.
        qk_end = self.q_head_num + self.kv_head_num
        return self.check_diff(packed_qkv[:, :qk_end], out_ref[:, :qk_end], threshold=0.99)
