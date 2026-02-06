import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
import mcoplib._C

class Swap_blocks_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_blocks = config.get("num_blocks", 1024)
        self.block_size = config.get("block_size", 16)
        self.num_kv_heads = config.get("num_kv_heads", 8)
        self.head_size = config.get("head_size", 128)
        self.num_pairs = config.get("num_pairs", 512)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"(Blocks={self.num_blocks} Heads={self.num_kv_heads} HeadSize={self.head_size})")
        element_size_bytes = 2 if self.dtype == torch.float16 or self.dtype == torch.bfloat16 else 4
        x = 16 // element_size_bytes
        block_elements = self.num_kv_heads * self.head_size * self.block_size
        total_elements = self.num_pairs * block_elements
        state.add_element_count(total_elements)
        state.add_global_memory_reads(total_elements * element_size_bytes)
        state.add_global_memory_writes(total_elements * element_size_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            element_size_bytes = 2 if self.dtype == torch.float16 or self.dtype == torch.bfloat16 else 4
            x = 16 // element_size_bytes
            cache_shape = (self.num_blocks, self.num_kv_heads, self.head_size // x, self.block_size, x)
            src_cache = torch.randn(cache_shape, dtype=self.dtype, device=dev)
            dst_cache = torch.zeros(cache_shape, dtype=self.dtype, device=dev)
            src_indices = torch.randint(0, self.num_blocks, (self.num_pairs,), device="cpu")
            dst_indices = torch.randint(0, self.num_blocks, (self.num_pairs,), device="cpu")
            mapping_data = torch.stack([src_indices, dst_indices], dim=1).to(torch.int64)
            def op_closure():
                torch.ops._C_cache_ops.swap_blocks(src_cache, dst_cache, mapping_data)
                return dst_cache
        return self.make_launcher(dev_id, op_closure)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        element_size_bytes = 2 if self.dtype == torch.float16 or self.dtype == torch.bfloat16 else 4
        x = 16 // element_size_bytes
        cache_shape = (self.num_blocks, self.num_kv_heads, self.head_size // x, self.block_size, x)
        src_cache = torch.randn(cache_shape, dtype=self.dtype, device=dev)
        dst_cache = torch.zeros(cache_shape, dtype=self.dtype, device=dev)
        src_idx = 0
        dst_idx = 5
        mapping_data = torch.tensor([[src_idx, dst_idx]], dtype=torch.int64, device="cpu")
        torch.ops._C_cache_ops.swap_blocks(src_cache, dst_cache, mapping_data)
        ref_block = src_cache[src_idx].clone()
        op_block = dst_cache[dst_idx]
        return self.check_diff(op_block, ref_block)