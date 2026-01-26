import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Convert_vertical_slash_indexes_mergehead_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 1)
        self.num_heads = config.get("num_heads", 4)
        self.context_size = config.get("context_size", 128)
        self.block_size_m = config.get("block_size_m", 64)
        self.block_size_n = config.get("block_size_n", 64)
        self.nnz_v = config.get("nnz_v", 8)
        self.nnz_s = config.get("nnz_s", 8)
        self.causal = config.get("causal", True)
        self.dtype = getattr(torch, config.get("dtype", "int32"))
        
        self.num_rows = (self.context_size + self.block_size_m - 1) // self.block_size_m

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"(B{self.batch_size}_H{self.num_heads}_CTX{self.context_size})")
        
        total_out = self.batch_size * self.num_heads * self.num_rows * (2 + self.nnz_s + self.nnz_v)
        state.add_element_count(total_out)
        
        element_size = 4
        # Inputs: seqlens(2*B) + indexes(B*H*(V+S)) + counts(2*H)
        size_in = (self.batch_size * 2) + \
                  (self.batch_size * self.num_heads * (self.nnz_v + self.nnz_s)) + \
                  (self.num_heads * 2)
        # Outputs: block/col counts + block/col offsets
        size_out = (self.batch_size * self.num_heads * self.num_rows) * (2 + self.nnz_s + self.nnz_v)
        
        state.add_global_memory_reads(size_in * element_size)
        state.add_global_memory_writes(size_out * element_size)

    def _prepare_args(self, device):
        q_seqlens = torch.full((self.batch_size,), self.context_size, dtype=self.dtype, device=device)
        kv_seqlens = torch.full((self.batch_size,), self.context_size, dtype=self.dtype, device=device)
        
        vertical_indexes = torch.randint(0, self.context_size, (self.batch_size, self.num_heads, self.nnz_v), dtype=self.dtype, device=device)
        slash_indexes = torch.randint(0, self.context_size, (self.batch_size, self.num_heads, self.nnz_s), dtype=self.dtype, device=device)
        
        vertical_indices_count = torch.randint(1, self.nnz_v + 1, (self.num_heads,), dtype=self.dtype, device=device)
        slash_indices_count = torch.randint(1, self.nnz_s + 1, (self.num_heads,), dtype=self.dtype, device=device)
        
        block_count = torch.zeros((self.batch_size, self.num_heads, self.num_rows), dtype=self.dtype, device=device)
        column_count = torch.zeros((self.batch_size, self.num_heads, self.num_rows), dtype=self.dtype, device=device)
        block_offset = torch.zeros((self.batch_size, self.num_heads, self.num_rows, self.nnz_s), dtype=self.dtype, device=device)
        column_index = torch.zeros((self.batch_size, self.num_heads, self.num_rows, self.nnz_v), dtype=self.dtype, device=device)
        
        return [block_count, block_offset, column_count, column_index, 
                q_seqlens, kv_seqlens, vertical_indexes, slash_indexes, 
                vertical_indices_count, slash_indices_count]

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            args = self._prepare_args(dev)
        return self.make_launcher(dev_id, torch.ops._C.convert_vertical_slash_indexes_mergehead, *args, 
                                  self.context_size, self.block_size_m, self.block_size_n, self.causal)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        args = self._prepare_args(dev)
        
        torch.ops._C.convert_vertical_slash_indexes_mergehead(
            *args,
            self.context_size,
            self.block_size_m,
            self.block_size_n,
            self.causal
        )
        
        block_count = args[0]
        column_count = args[2]
        
        # 验证逻辑：检查输出是否在有效范围内
        # 1. 检查是否有数据写入 (非全0)
        has_data = (block_count.sum() + column_count.sum()) > 0
        # 2. 检查数值是否越界 (count 不应超过 nnz 大小)
        valid_block = (block_count <= self.nnz_s).all()
        valid_col = (column_count <= self.nnz_v).all()
        
        if has_data and valid_block and valid_col:
            return True, block_count.float().mean().item()
        else:
            # 构造差异以报错
            ref = block_count.clone()
            if not valid_block: ref += 999
            return self.check_diff(block_count, ref)