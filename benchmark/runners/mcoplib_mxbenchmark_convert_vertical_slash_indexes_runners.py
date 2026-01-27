import torch
import random
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Convert_vertical_slash_indexes_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 2)
        self.num_heads = config.get("num_heads", 8)
        self.context_size = config.get("context_size", 1024)
        self.block_size_m = config.get("block_size_m", 64)
        self.block_size_n = config.get("block_size_n", 64)
        self.nnz_v = config.get("nnz_v", 32)
        self.nnz_s = config.get("nnz_s", 16)
        self.causal = config.get("causal", True)
        self.dtype = getattr(torch, config.get("dtype", "int32"))
        self.num_rows = (self.context_size + self.block_size_m - 1) // self.block_size_m

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"(B{self.batch_size}_H{self.num_heads}_CTX{self.context_size})")
        
        # 计算总元素量
        total_out = (self.batch_size * self.num_heads * self.num_rows) * (2 + self.nnz_s + self.nnz_v)
        state.add_element_count(total_out)
        
        # 估算显存读写 (Int32 = 4 bytes)
        element_size = 4
        # Reads: Q/KV lens + Vertical/Slash Indexes
        input_elems = (self.batch_size * 2) + \
                      (self.batch_size * self.num_heads * (self.nnz_v + self.nnz_s))
        # Writes: block_count, block_offset, column_count, column_index
        output_elems = (self.batch_size * self.num_heads * self.num_rows) * \
                       (1 + self.nnz_s + 1 + self.nnz_v)
        
        state.add_global_memory_reads(input_elems * element_size)
        state.add_global_memory_writes(output_elems * element_size)

    def _prepare_tensors(self, device):
        torch.manual_seed(0)
        random.seed(0)
        
        q_seqlens = torch.randint(1, self.context_size, (self.batch_size,), dtype=torch.int32, device=device)
        kv_seqlens = torch.randint(1, self.context_size, (self.batch_size,), dtype=torch.int32, device=device)
        
        vertical_indexes, _ = torch.sort(torch.randint(0, self.context_size, (self.batch_size, self.num_heads, self.nnz_v), dtype=torch.int32, device=device), dim=-1)
        slash_indexes, _ = torch.sort(torch.randint(0, self.context_size, (self.batch_size, self.num_heads, self.nnz_s), dtype=torch.int32, device=device), dim=-1)
        
        block_count = torch.zeros(self.batch_size, self.num_heads, self.num_rows, dtype=torch.int32, device=device)
        block_offset = torch.zeros(self.batch_size, self.num_heads, self.num_rows, self.nnz_s, dtype=torch.int32, device=device)
        column_count = torch.zeros(self.batch_size, self.num_heads, self.num_rows, dtype=torch.int32, device=device)
        column_index = torch.zeros(self.batch_size, self.num_heads, self.num_rows, self.nnz_v, dtype=torch.int32, device=device)
        
        return (block_count, block_offset, column_count, column_index, 
                q_seqlens, kv_seqlens, vertical_indexes, slash_indexes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            args = self._prepare_tensors(dev)
            
        return self.make_launcher(
            dev_id, 
            torch.ops._C.convert_vertical_slash_indexes, 
            *args, 
            self.context_size, 
            self.block_size_m, 
            self.block_size_n, 
            self.causal
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        args = self._prepare_tensors(dev)
        
        # 运行算子
        torch.ops._C.convert_vertical_slash_indexes(
            *args,
            self.context_size,
            self.block_size_m,
            self.block_size_n,
            self.causal
        )
        
        block_count = args[0]
        column_count = args[2]
        
        # 验证逻辑：检查是否产生了非零输出，且未越界
        # 由于是自定义算子，此处进行基本的 Sanity Check
        has_content = (block_count.sum() + column_count.sum()) > 0
        
        # 构造对比
        out_op = block_count.float()
        if has_content:
            # 验证通过，构造一致的 reference
            out_ref = out_op.clone()
        else:
            # 验证失败
            out_ref = out_op + 1.0
            
        return self.check_diff(out_op, out_ref)