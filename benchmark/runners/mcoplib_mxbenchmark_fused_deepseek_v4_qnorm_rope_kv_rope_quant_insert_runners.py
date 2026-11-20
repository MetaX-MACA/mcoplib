import torch

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


def make_cos_sin_cache(max_pos: int, rope_dim: int, dtype, device):
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


class Fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_runner(OpBenchmarkBase):

    def __init__(self, name, config):
        super().__init__(name, config)

        self.device_id = config.get("device_id", 0)
        self.batch_size = config.get("batch_size", 1024)
        self.num_tokens = config.get("num_tokens", self.batch_size)
        self.num_heads = config.get("num_heads", 32)
        self.num_kv_heads = config.get("num_kv_heads", 1)
        self.seed = config.get("seed", None)
        self.q_head_padded = config.get("q_head_padded", self.num_heads)
        self.head_dim = config.get("head_dim", 512)
        self.rotary_dim = config.get("rotary_dim", 64)
        self.max_position = config.get("max_position", 4096)
        self.block_size = config.get("block_size", 64)
        self.eps = config.get("eps", 1e-6)
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        self.kv_cache_dtype = config.get("kv_cache_dtype", "uint8")

        self.apply_q_norm = config.get("apply_q_norm", True)
        self.kv_mxfp8 = config.get("kv_mxfp8", False)
        self.apply_q_rope = config.get("apply_q_rope", True)
        self.is_q_interleaved = config.get("is_q_interleaved", False)

        if self.q_head_padded != 0 and self.q_head_padded < self.num_heads:
            raise ValueError("q_head_padded must be 0 or >= num_heads")

        if self.num_kv_heads != 1:
            raise ValueError("num_kv_heads must be 1 for kv shape [N, 512]")

    def _make_inputs(self, dev):
        num_tokens = self.num_tokens

        q_in = torch.randn((num_tokens, self.num_heads, self.head_dim), dtype=self.dtype, device=dev).contiguous()
        kv = torch.randn((num_tokens, self.head_dim), dtype=self.dtype, device=dev).contiguous()

        cos_sin_cache = make_cos_sin_cache(self.max_position, self.rotary_dim, torch.float32, dev)

        positions = torch.randint(0, self.max_position, (num_tokens,), dtype=torch.int64, device=dev)

        num_blocks = ((num_tokens + self.block_size - 1) // self.block_size) + 1
        head_bytes = self.head_dim - self.rotary_dim + self.rotary_dim * 2 + 8

        k_cache = torch.zeros((num_blocks, self.block_size * head_bytes), dtype=torch.uint8, device=dev).contiguous()

        slot_mapping = torch.randperm(num_blocks * self.block_size, dtype=torch.int64, device=dev)[:num_tokens]

        return q_in, kv, k_cache, slot_mapping, positions, cos_sin_cache

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"{self.num_tokens}x{self.num_heads}x{self.q_head_padded}")
        state.add_summary("apply_q_norm", str(self.apply_q_norm))
        state.add_summary("kv_mxfp8", str(self.kv_mxfp8))
        state.add_summary("apply_q_rope", str(self.apply_q_rope))
        state.add_summary("is_q_interleaved", str(self.is_q_interleaved))

        q_elements = self.num_tokens * self.num_heads * self.head_dim
        kv_elements = self.num_tokens * self.num_kv_heads * self.head_dim
        total = q_elements + kv_elements

        state.add_element_count(total)

        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4

        state.add_global_memory_reads(total * element_size)

        q_out_elements = self.num_tokens * self.q_head_padded * self.head_dim
        state.add_global_memory_writes(q_out_elements * element_size)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        dev = f"cuda:{dev_id}"

        with torch.cuda.stream(tc_s):
            q_in, kv, k_cache, slot_mapping, positions, cos_sin_cache = self._make_inputs(dev)

        return self.make_launcher(dev_id, torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert, q_in, kv, k_cache, slot_mapping, positions, cos_sin_cache, self.q_head_padded, self.eps, self.block_size, self.apply_q_norm, self.kv_mxfp8, self.apply_q_rope, self.is_q_interleaved)

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        q_in, kv, k_cache, slot_mapping, positions, cos_sin_cache = self._make_inputs(dev)

        out = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(q_in, kv, k_cache, slot_mapping, positions, cos_sin_cache, self.q_head_padded, self.eps, self.block_size, self.apply_q_norm, self.kv_mxfp8, self.apply_q_rope, self.is_q_interleaved)

        torch.cuda.synchronize()

        return self.check_diff(out, out.clone())