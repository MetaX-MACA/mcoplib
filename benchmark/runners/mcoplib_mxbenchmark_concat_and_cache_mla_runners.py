import torch
import sys
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Concat_and_cache_mla_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 8192)
        self.kv_lora_rank = config.get("kv_lora_rank", 512)
        self.pe_dim = config.get("pe_dim", 64)
        self.block_size = config.get("block_size", 16)
        self.num_blocks = config.get("num_blocks", 1024)
        
        self.total_dim = self.kv_lora_rank + self.pe_dim
        self.kv_cache_dtype = config.get("kv_cache_dtype", "auto")

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        
        shape_str = f"(Tokens:{self.batch_size} Rank:{self.kv_lora_rank} PE:{self.pe_dim})"
        state.add_summary("Shape", shape_str)
        
        input_elems_c = self.batch_size * self.kv_lora_rank
        input_elems_pe = self.batch_size * self.pe_dim
        output_elems = self.batch_size * self.total_dim
        
        state.add_element_count(input_elems_c + input_elems_pe + output_elems)
        
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        
        r_bytes = (input_elems_c + input_elems_pe) * element_size + self.batch_size * 8
        w_bytes = output_elems * element_size
        
        state.add_global_memory_reads(r_bytes) 
        state.add_global_memory_writes(w_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            kv_c = torch.randn((self.batch_size, self.kv_lora_rank), dtype=self.dtype, device=dev)
            k_pe = torch.randn((self.batch_size, self.pe_dim), dtype=self.dtype, device=dev)
            
            kv_cache = torch.zeros((self.num_blocks, self.block_size, self.total_dim), dtype=self.dtype, device=dev)
            slot_mapping = torch.arange(self.batch_size, dtype=torch.int64, device=dev)
            
            scale = torch.tensor([1.0], dtype=torch.float32, device=dev)
            
        return self.make_launcher(dev_id, torch.ops._C_cache_ops.concat_and_cache_mla, 
                                  kv_c,
                                  k_pe,
                                  kv_cache,
                                  slot_mapping,
                                  self.kv_cache_dtype,
                                  scale)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        
        kv_c = torch.randn((self.batch_size, self.kv_lora_rank), dtype=self.dtype, device=dev)
        k_pe = torch.randn((self.batch_size, self.pe_dim), dtype=self.dtype, device=dev)
        kv_cache = torch.zeros((self.num_blocks, self.block_size, self.total_dim), dtype=self.dtype, device=dev)
        slot_mapping = torch.arange(self.batch_size, dtype=torch.int64, device=dev)
        scale = torch.tensor([1.0], dtype=torch.float32, device=dev)
        
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            self.kv_cache_dtype,
            scale
        )
        
        out_ref = torch.cat([kv_c, k_pe], dim=-1)
        
        flat_cache = kv_cache.view(-1, self.total_dim)
        out_op = flat_cache[slot_mapping]
        
        return self.check_diff(out_op, out_ref)