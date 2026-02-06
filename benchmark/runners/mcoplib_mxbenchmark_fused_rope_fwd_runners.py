import torch
import sys
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.op as op
except ImportError:
    op = None

class Fused_rope_fwd_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 4096)
        self.qkv_num = config.get("qkv_num", 4) 
        self.num_head = config.get("num_head", 32)
        self.head_dim = config.get("head_dim", 128)
        self.head_dim_half = config.get("head_dim_half", 64)
        self.force_bf16 = config.get("force_bf16", False)
        self.use_indexes = config.get("use_indexes", True)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.batch_size} {self.qkv_num} {self.num_head} {self.head_dim})")
        rope_active_elems = self.batch_size * 2 * self.num_head * self.head_dim
        state.add_element_count(rope_active_elems)
        element_size = 2 if self.dtype == torch.float16 else 4
        tensor_rw_bytes = rope_active_elems * element_size
        aux_read_bytes = (self.batch_size * self.head_dim_half * element_size * 2) 
        if self.use_indexes:
            aux_read_bytes += self.batch_size * 8

        state.add_global_memory_reads(tensor_rw_bytes + aux_read_bytes) 
        state.add_global_memory_writes(tensor_rw_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            qkv = torch.randn(self.batch_size, self.qkv_num, self.num_head, self.head_dim, dtype=self.dtype, device=dev)
            cos = torch.randn(self.batch_size, self.head_dim_half, dtype=self.dtype, device=dev)
            sin = torch.randn(self.batch_size, self.head_dim_half, dtype=self.dtype, device=dev)
            idx = torch.randint(0, self.batch_size, (self.batch_size,), dtype=torch.long, device=dev) if self.use_indexes else None
        return self.make_launcher(dev_id, op.fused_rope_fwd, qkv, cos, sin, idx, self.force_bf16)

    def _torch_rope_impl(self, qkv, cos, sin):
        q = qkv[:, 0].float()
        k = qkv[:, 1].float()
        others = qkv[:, 2:].float() 
        cos = cos.float()
        sin = sin.float()   
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        c = cos.unsqueeze(1)
        s = sin.unsqueeze(1)
        c = torch.cat([c, c], dim=-1)
        s = torch.cat([s, s], dim=-1)
        q_out = (q * c) + (rotate_half(q) * s)
        k_out = (k * c) + (rotate_half(k) * s)
        rotated_part = torch.stack([q_out, k_out], dim=1)
        return torch.cat([rotated_part, others], dim=1)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        qkv = torch.randn(self.batch_size, self.qkv_num, self.num_head, self.head_dim, dtype=self.dtype, device=dev)
        cos = torch.randn(self.batch_size, self.head_dim_half, dtype=self.dtype, device=dev)
        sin = torch.randn(self.batch_size, self.head_dim_half, dtype=self.dtype, device=dev)
        idx = torch.randint(0, self.batch_size, (self.batch_size,), dtype=torch.long, device=dev) if self.use_indexes else None
        qkv_ref_in = qkv.clone()
        out_op = op.fused_rope_fwd(qkv, cos, sin, idx, self.force_bf16)
        if idx is not None:
            cos_ref = cos[idx]
            sin_ref = sin[idx]
        else:
            cos_ref = cos
            sin_ref = sin
        out_ref = self._torch_rope_impl(qkv_ref_in, cos_ref, sin_ref)
        out_ref = out_ref.to(self.dtype)
        return self.check_diff(out_op, out_ref)