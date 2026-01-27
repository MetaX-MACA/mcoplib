import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Gather_and_maybe_dequant_cache_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 32)
        self.block_size = config.get("block_size", 16)
        
        # [Fix] 该算子是 MLA 专用，固定处理 576 元素/Token
        # 因此 num_kv_heads 必须为 1，head_size 必须为 576
        self.num_kv_heads = 1  
        self.head_size = 576
        
        self.seq_len = config.get("seq_len", 128)
        self.kv_cache_dtype = config.get("kv_cache_dtype", "auto")

        self.total_tokens = self.batch_size * self.seq_len
        self.blocks_per_seq = (self.seq_len + self.block_size - 1) // self.block_size
        self.num_blocks = self.batch_size * self.blocks_per_seq

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"(B={self.batch_size} L={self.seq_len} H={self.head_size})")
        
        total_elements = self.total_tokens * self.num_kv_heads * self.head_size
        state.add_element_count(total_elements)
        
        element_size = 2 if self.dtype == torch.float16 else 4
        rw_bytes = total_elements * element_size * 2
        state.add_global_memory_reads(rw_bytes) 
        state.add_global_memory_writes(rw_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            # src shape: [blocks, block_size, 1, 576]
            src_shape = (self.num_blocks, self.block_size, self.num_kv_heads, self.head_size)
            src_cache = torch.randn(src_shape, dtype=self.dtype, device=dev)
            
            # dst shape: [total_tokens, 1, 576]
            dst_shape = (self.total_tokens, self.num_kv_heads, self.head_size)
            dst = torch.zeros(dst_shape, dtype=self.dtype, device=dev)
            
            block_table = torch.arange(self.num_blocks, dtype=torch.int32, device=dev).view(self.batch_size, self.blocks_per_seq)
            cu_seq_lens = torch.arange(0, (self.batch_size + 1) * self.seq_len, step=self.seq_len, dtype=torch.int32, device=dev)
            
            token_to_seq = torch.arange(self.batch_size, device=dev, dtype=torch.int32).repeat_interleave(self.seq_len)
            scale = torch.tensor([1.0], dtype=torch.float32, device=dev)
            
        return self.make_launcher(dev_id, torch.ops._C_cache_ops.gather_and_maybe_dequant_cache, 
                                  src_cache, dst, block_table, cu_seq_lens, 
                                  token_to_seq,       
                                  self.total_tokens,  
                                  self.kv_cache_dtype, scale, None)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        
        src_shape = (self.num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        src_cache = torch.randn(src_shape, dtype=self.dtype, device=dev)
        
        dst_shape = (self.total_tokens, self.num_kv_heads, self.head_size)
        dst = torch.zeros(dst_shape, dtype=self.dtype, device=dev)
        
        block_table = torch.arange(self.num_blocks, dtype=torch.int32, device=dev).view(self.batch_size, self.blocks_per_seq)
        cu_seq_lens = torch.arange(0, (self.batch_size + 1) * self.seq_len, step=self.seq_len, dtype=torch.int32, device=dev)
        token_to_seq = torch.arange(self.batch_size, device=dev, dtype=torch.int32).repeat_interleave(self.seq_len)
        scale = torch.tensor([1.0], dtype=torch.float32, device=dev)
        
        torch.ops._C_cache_ops.gather_and_maybe_dequant_cache(
            src_cache, dst, block_table, cu_seq_lens, 
            token_to_seq,       
            self.total_tokens,  
            self.kv_cache_dtype, scale, None
        )
        
        # --- Python Validation ---
        src_cpu = src_cache.float().cpu()
        table_cpu = block_table.cpu()
        cu_lens_cpu = cu_seq_lens.cpu()
        dst_ref_cpu = torch.zeros(dst_shape, dtype=torch.float32)
        
        # 逻辑：对于每个 token，找到对应的 block 和 offset，复制所有通道 (1 * 576)
        for b in range(self.batch_size):
            seq_start = cu_lens_cpu[b].item()
            seq_end = cu_lens_cpu[b+1].item()
            current_seq_len = seq_end - seq_start
            
            for t in range(current_seq_len):
                block_idx = t // self.block_size
                block_off = t % self.block_size
                phys_block_id = table_cpu[b, block_idx].item()
                
                # 直接复制整个 [1, 576] 的数据块
                dst_ref_cpu[seq_start + t] = src_cpu[phys_block_id, block_off]
        
        dst_ref = dst_ref_cpu.to(dev).to(self.dtype)
        
        return self.check_diff(dst, dst_ref)