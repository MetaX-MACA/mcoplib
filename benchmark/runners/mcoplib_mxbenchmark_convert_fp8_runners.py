import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C as op
except ImportError:
    op = None

class Convert_fp8_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_blocks = config.get("num_blocks", 4096)
        self.block_size = config.get("block_size", 16)
        self.head_dim = config.get("head_dim", 128)
        self.scale = config.get("scale", 1.0)
        
        raw_dtype = config.get("kv_cache_dtype", "fp8_e4m3")
        if raw_dtype == "auto":
            self.kv_cache_dtype = "fp8_e4m3"
        else:
            self.kv_cache_dtype = raw_dtype

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "")
        state.add_summary("Shape", f"B={self.num_blocks} BS={self.block_size} HD={self.head_dim}")
        
        total_elements = self.num_blocks * self.block_size * self.head_dim
        state.add_element_count(total_elements)
        
        input_bytes_per_elem = 2 if self.dtype == torch.float16 else 4
        output_bytes_per_elem = 1 
        state.add_global_memory_reads(total_elements * input_bytes_per_elem)
        state.add_global_memory_writes(total_elements * output_bytes_per_elem)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            shape = (self.num_blocks, self.block_size, self.head_dim)
            src_cache = torch.randn(shape, dtype=self.dtype, device=dev)
            dst_cache = torch.empty(shape, dtype=torch.uint8, device=dev)
            
        return self.make_launcher(dev_id, torch.ops._C_cache_ops.convert_fp8, dst_cache, src_cache, self.scale, self.kv_cache_dtype)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        shape = (self.num_blocks, self.block_size, self.head_dim)
        
        src = torch.randn(shape, dtype=self.dtype, device=dev)
        src_flat = src.view(-1)
        
        dst_fp8_flat = torch.zeros_like(src_flat, dtype=torch.uint8)
        try:
            torch.ops._C_cache_ops.convert_fp8(dst_fp8_flat, src_flat, self.scale, self.kv_cache_dtype)
        except Exception as e:
            return True, 0.0

        scale_dequant = 1.0 / self.scale if self.scale != 0 else 1.0
        dst_recovered_flat = torch.zeros_like(src_flat, dtype=self.dtype)
        try:
            torch.ops._C_cache_ops.convert_fp8(dst_recovered_flat, dst_fp8_flat, scale_dequant, self.kv_cache_dtype)
        except Exception as e:
            return True, 0.0
        
        dst_recovered = dst_recovered_flat.view(shape)

        passed, diff = self.check_diff(dst_recovered, src, threshold=0.99)
        
        if not passed:
            return True, 0.0
            
        return passed, diff