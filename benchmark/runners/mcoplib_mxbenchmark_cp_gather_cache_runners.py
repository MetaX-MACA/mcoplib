import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
try:
    import mcoplib._C
except ImportError:
    pass
class Cp_gather_cache_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 32)
        self.block_size = config.get("block_size", 16)
        self.num_heads = config.get("num_heads", 32)
        self.head_size = config.get("head_size", 128)
        self.seq_len = config.get("seq_len", 128)
        self.total_tokens = self.batch_size * self.seq_len
        self.blocks_per_seq = (self.seq_len + self.block_size - 1) // self.block_size
        self.num_blocks = self.batch_size * self.blocks_per_seq
    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"(B={self.batch_size} L={self.seq_len} H={self.num_heads} D={self.head_size})")
        total_elements = self.total_tokens * self.num_heads * self.head_size
        state.add_element_count(total_elements)
        element_size = 2 if self.dtype == torch.float16 else 4
        if self.dtype == torch.float32: element_size = 4
        elif self.dtype == torch.int8: element_size = 1
        rw_bytes = total_elements * element_size * 2
        state.add_global_memory_reads(rw_bytes)
        state.add_global_memory_writes(rw_bytes)
    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            src_shape = (self.num_blocks, self.block_size, self.num_heads, self.head_size)
            src_cache = torch.randn(src_shape, dtype=self.dtype, device=dev)
            dst_shape = (self.total_tokens, self.num_heads, self.head_size)
            dst = torch.zeros(dst_shape, dtype=self.dtype, device=dev)
            block_table = torch.arange(self.num_blocks, dtype=torch.int32, device=dev).view(self.batch_size, self.blocks_per_seq)
            cu_seq_lens = torch.arange(0, (self.batch_size + 1) * self.seq_len, step=self.seq_len, dtype=torch.int32, device=dev)
        return self.make_launcher(dev_id, torch.ops._C_cache_ops.cp_gather_cache, 
                                  src_cache, dst, block_table, cu_seq_lens, 
                                  self.batch_size, None)
    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        src_shape = (self.num_blocks, self.block_size, self.num_heads, self.head_size)
        src_cache = torch.randn(src_shape, dtype=self.dtype, device=dev)
        dst_shape = (self.total_tokens, self.num_heads, self.head_size)
        dst = torch.zeros(dst_shape, dtype=self.dtype, device=dev)
        block_table = torch.arange(self.num_blocks, dtype=torch.int32, device=dev).view(self.batch_size, self.blocks_per_seq)
        cu_seq_lens = torch.arange(0, (self.batch_size + 1) * self.seq_len, step=self.seq_len, dtype=torch.int32, device=dev)
        torch.ops._C_cache_ops.cp_gather_cache(
            src_cache, dst, block_table, cu_seq_lens, 
            self.batch_size, None
        )
        src_cpu = src_cache.cpu()
        dst_ref = torch.zeros_like(dst, device='cpu')
        table_cpu = block_table.cpu()
        cu_lens_cpu = cu_seq_lens.cpu()
        for b in range(self.batch_size):
            seq_start = cu_lens_cpu[b].item()
            seq_end = cu_lens_cpu[b+1].item()
            cur_len = seq_end - seq_start
            for t in range(cur_len):
                block_idx = t // self.block_size
                block_off = t % self.block_size
                phys_block = table_cpu[b, block_idx].item()
                dst_ref[seq_start + t] = src_cpu[phys_block, block_off]
        dst_ref = dst_ref.to(dev)
        return self.check_diff(dst, dst_ref)