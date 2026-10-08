import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Fused_kimi_k3_mla_key_concat_kv_cache_insert_runner(OpBenchmarkBase):

    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 32)
        self.num_heads = config.get("num_heads", 4)
        self.block_size = config.get("block_size", 8)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"tokens={self.num_tokens},heads={self.num_heads}")

        read_elements = self.num_tokens * (self.num_heads * 192 + self.num_heads * 128 + 64 + 512)
        write_elements = self.num_tokens * (self.num_heads * 192 + 576)

        state.add_element_count(self.num_tokens * self.num_heads * 192)
        state.add_global_memory_reads(read_elements * 2)
        state.add_global_memory_writes(write_elements * 2)

    def make_rope_cache(self, dev):
        inv_freq = 1.0 / (50000 ** (torch.arange(0, 64, 2, device=dev, dtype=torch.float32) / 64))
        pos = torch.arange(2048, device=dev, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        return torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"

            q = torch.randn(self.num_tokens, self.num_heads, 192, dtype=self.dtype, device=dev)
            k_nope = torch.randn(self.num_tokens, self.num_heads, 128, dtype=self.dtype, device=dev)
            k_pe = torch.randn(self.num_tokens, 64, dtype=self.dtype, device=dev)
            kv_c = torch.randn(self.num_tokens, 512, dtype=self.dtype, device=dev)

            k_out = torch.empty(self.num_tokens, self.num_heads, 192, dtype=self.dtype, device=dev)

            num_blocks = (self.num_tokens + self.block_size - 1) // self.block_size + 2
            k_cache = torch.zeros(num_blocks, self.block_size, 576, dtype=self.dtype, device=dev)

            slots = torch.arange(self.num_tokens, dtype=torch.int64, device=dev)
            positions = torch.arange(self.num_tokens, dtype=torch.int64, device=dev)
            rope_cache = self.make_rope_cache(dev)

        return self.make_launcher(dev_id, torch.ops._C.fused_kimi_k3_mla_key_concat_kv_cache_insert, q, k_nope, k_pe, kv_c, k_out, k_cache, slots, self.block_size, positions, rope_cache)

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        num_tokens = 3
        num_heads = 4
        block_size = 8

        q = torch.randn(num_tokens, num_heads, 192, dtype=self.dtype, device=dev)
        k_nope = torch.randn(num_tokens, num_heads, 128, dtype=self.dtype, device=dev)
        k_pe = torch.randn(num_tokens, 64, dtype=self.dtype, device=dev)
        kv_c = torch.randn(num_tokens, 512, dtype=self.dtype, device=dev)

        k_out = torch.empty(num_tokens, num_heads, 192, dtype=self.dtype, device=dev)
        k_cache = torch.zeros(2, block_size, 576, dtype=self.dtype, device=dev)

        slots = torch.tensor([0, 3, 9], dtype=torch.int64, device=dev)
        positions = torch.tensor([1, 7, 13], dtype=torch.int64, device=dev)

        rope_cache = self.make_rope_cache(dev)
        torch.ops._C.fused_kimi_k3_mla_key_concat_kv_cache_insert(q, k_nope, k_pe, kv_c, k_out, k_cache, slots, block_size, positions, rope_cache)

        cos, sin = rope_cache.index_select(0, positions).chunk(2, dim=-1)

        x1 = k_pe.float()[..., ::2]
        x2 = k_pe.float()[..., 1::2]

        k_pe_ref = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2).to(self.dtype)

        k_ref = torch.cat([k_nope, k_pe_ref[:, None, :].expand(-1, num_heads, -1)], dim=-1)

        cache_ref = torch.cat([kv_c, k_pe_ref], dim=-1)

        cache_out = k_cache.reshape(-1, 576).index_select(0, slots)

        torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(cache_out, cache_ref, atol=2e-2, rtol=2e-2)

        cos1 = F.cosine_similarity(k_out.flatten().float(), k_ref.flatten().float(), dim=0).item()
        cos2 = F.cosine_similarity(cache_out.flatten().float(), cache_ref.flatten().float(), dim=0).item()
        cos = min(cos1, cos2)

        return True, cos