import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass


def _ref_rmsnorm_self(x, eps):
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms).to(x.dtype)


def _ref_rope_interleaved(x, freqs_cis, positions, rope_dim):
    out = x.clone()
    B = x.size(0)
    head_dim = x.size(-1)
    nope_dim = head_dim - rope_dim
    for b in range(B):
        pos = positions[b].item()
        freq = freqs_cis[pos]
        rope_part = out[b, ..., nope_dim:].float()
        pairs = rope_part.reshape(*rope_part.shape[:-1], rope_dim // 2, 2)
        x_real = pairs[..., 0]
        x_imag = pairs[..., 1]
        freq_pairs = freq.reshape(rope_dim // 2, 2)
        f_real = freq_pairs[:, 0]
        f_imag = freq_pairs[:, 1]
        rot_real = x_real * f_real - x_imag * f_imag
        rot_imag = x_real * f_imag + x_imag * f_real
        result = torch.stack([rot_real, rot_imag], dim=-1).reshape(rope_part.shape)
        out[b, ..., nope_dim:] = result.to(x.dtype)
    return out


class Dsv4_fused_q_norm_rope_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 16)
        self.num_heads = config.get("num_heads", 8)
        self.head_dim = config.get("head_dim", 192)
        self.rope_dim = config.get("rope_dim", 64)
        self.max_pos = config.get("max_pos", 512)
        self.eps = config.get("eps", 1e-6)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        shape_str = f"({self.batch_size} {self.num_heads} {self.head_dim})"
        state.add_summary("Shape", shape_str)

        in_elems = self.batch_size * self.num_heads * self.head_dim
        out_elems = in_elems
        freqs_elems = self.max_pos * self.rope_dim
        state.add_element_count(in_elems + out_elems + freqs_elems)

        es = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        reads = in_elems * es + freqs_elems * 4 + self.batch_size * 4
        writes = out_elems * es
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            q_input = torch.randn(
                self.batch_size, self.num_heads, self.head_dim,
                dtype=self.dtype, device=dev
            )
            freqs_cis = torch.randn(
                self.max_pos, self.rope_dim, dtype=torch.float32, device=dev
            )
            positions = torch.randint(
                0, self.max_pos, (self.batch_size,),
                dtype=torch.int32, device=dev
            )
            q_output = torch.empty_like(q_input)
        return self.make_launcher(
            dev_id, torch.ops.sgl_kernel.dsv4_fused_q_norm_rope,
            q_input, q_output, freqs_cis, positions, self.eps
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        B, H, D = 4, 8, 192
        rope_dim = self.rope_dim
        max_pos = self.max_pos
        torch.manual_seed(42)
        q_input = torch.randn(B, H, D, dtype=self.dtype, device=dev)
        freqs_cis = torch.randn(max_pos, rope_dim, dtype=torch.float32, device=dev)
        positions = torch.randint(0, max_pos, (B,), dtype=torch.int32, device=dev)
        q_output = torch.empty_like(q_input)
        torch.ops.sgl_kernel.dsv4_fused_q_norm_rope(
            q_input, q_output, freqs_cis, positions, self.eps
        )
        normed = _ref_rmsnorm_self(q_input, self.eps)
        expected = _ref_rope_interleaved(normed, freqs_cis, positions, rope_dim)
        match = self.check_diff(q_output.float(), expected.float(), threshold=0.999)
        return match[0], 0.0 if match[0] else 1.0
