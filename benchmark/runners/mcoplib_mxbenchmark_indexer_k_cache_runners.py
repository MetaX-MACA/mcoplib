import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
try:
    import mcoplib._C
except ImportError:
    pass
class Indexer_k_cache_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 1024 * 1024)
        self.block_size = config.get("block_size", 16)
        self.head_dim = config.get("head_dim", 128)
        self.num_blocks = config.get("num_blocks", 65536)
        min_blocks = (self.num_tokens + self.block_size - 1) // self.block_size
        if self.num_blocks < min_blocks:
            self.num_blocks = min_blocks + 1024
    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"(Tokens={self.num_tokens} H={self.head_dim})")
        total_elements = self.num_tokens * self.head_dim
        state.add_element_count(total_elements)
        element_size = 2 if self.dtype == torch.float16 else 4
        idx_bytes = self.num_tokens * 4
        data_bytes = total_elements * element_size
        rw_bytes = (data_bytes * 2) + idx_bytes
        state.add_global_memory_reads(data_bytes + idx_bytes)
        state.add_global_memory_writes(data_bytes)
    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            k = torch.randn((self.num_tokens, self.head_dim), dtype=self.dtype, device=dev)
            cache_shape = (self.num_blocks, self.block_size, self.head_dim)
            kv_cache = torch.zeros(cache_shape, dtype=self.dtype, device=dev)
            max_slots = self.num_blocks * self.block_size
            slot_mapping = torch.randperm(max_slots, device=dev, dtype=torch.int64)[:self.num_tokens]
        return self.make_launcher(dev_id, torch.ops._C_cache_ops.indexer_k_cache, 
                                  k, kv_cache, slot_mapping)
    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        k = torch.randn((self.num_tokens, self.head_dim), dtype=self.dtype, device=dev)
        cache_shape = (self.num_blocks, self.block_size, self.head_dim)
        kv_cache = torch.zeros(cache_shape, dtype=self.dtype, device=dev)
        max_slots = self.num_blocks * self.block_size
        slot_mapping = torch.randperm(max_slots, device=dev, dtype=torch.int64)[:self.num_tokens]
        torch.ops._C_cache_ops.indexer_k_cache(k, kv_cache, slot_mapping)
        block_indices = slot_mapping // self.block_size
        block_offsets = slot_mapping % self.block_size
        k_recovered = kv_cache[block_indices, block_offsets, :]
        return self.check_diff(k_recovered, k)