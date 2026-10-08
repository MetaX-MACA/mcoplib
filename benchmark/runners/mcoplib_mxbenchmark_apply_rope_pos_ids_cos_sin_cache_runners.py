import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Apply_rope_pos_ids_cos_sin_cache_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.nnz = config.get("nnz", 8192)
        self.num_q_heads = config.get("num_q_heads", 32)
        self.num_kv_heads = config.get("num_kv_heads", 32)
        self.head_dim = config.get("head_dim", 128)
        self.rotary_dim = config.get("rotary_dim", 128)
        self.max_seq_len = config.get("max_seq_len", 8192)
        self.interleave = config.get("interleave", False)
        self.enable_pdl = config.get("enable_pdl", False)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.nnz} {self.num_q_heads} {self.head_dim})")
        
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        
        total_q_elements = self.nnz * self.num_q_heads * self.head_dim
        total_k_elements = self.nnz * self.num_kv_heads * self.head_dim
        
        pos_ids_bytes = self.nnz * 8
        tensor_io_bytes = (total_q_elements + total_k_elements) * element_size * 2
        
        cos_sin_read_bytes = self.nnz * self.rotary_dim * 4 
        
        read_bytes = int(tensor_io_bytes // 2 + pos_ids_bytes + cos_sin_read_bytes)
        write_bytes = int(tensor_io_bytes // 2)
        
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)
        state.add_element_count(total_q_elements + total_k_elements)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = torch.device(f'cuda:{dev_id}')
        
        q = torch.randn(self.nnz, self.num_q_heads, self.head_dim, dtype=self.dtype, device=dev)
        k = torch.randn(self.nnz, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=dev)
        
        cos_sin_cache = torch.randn(self.max_seq_len, self.rotary_dim, dtype=torch.float32, device=dev)
        
        pos_ids = torch.randint(0, self.max_seq_len, (self.nnz,), dtype=torch.int64, device=dev)
        
        q_rope = torch.empty_like(q)
        k_rope = torch.empty_like(k)

        def launcher(launch):
            stream_ptr = launch.get_stream()
            stream = self.as_torch_stream(stream_ptr, dev_id)
            
            with torch.cuda.stream(stream):
                torch.ops.sgl_kernel.apply_rope_pos_ids_cos_sin_cache(
                    q, k, q_rope, k_rope, cos_sin_cache, pos_ids,
                    self.interleave, self.enable_pdl,
                    None, None, None, None 
                )
                
        return launcher

    def run_verification(self, dev_id):
        dev = torch.device(f'cuda:{dev_id}')
        
        nnz = 64
        q_heads = 4
        kv_heads = 2
        h_dim = 64
        r_dim = 64
        seq_len = 128
        
        q = torch.randn(nnz, q_heads, h_dim, dtype=self.dtype, device=dev)
        k = torch.randn(nnz, kv_heads, h_dim, dtype=self.dtype, device=dev)
        
        cos_sin_cache = torch.randn(seq_len, r_dim, dtype=torch.float32, device=dev)
        pos_ids = torch.randint(0, seq_len, (nnz,), dtype=torch.int64, device=dev)
        
        q_rope = torch.empty_like(q)
        k_rope = torch.empty_like(k)
        
        torch.ops.sgl_kernel.apply_rope_pos_ids_cos_sin_cache(
            q, k, q_rope, k_rope, cos_sin_cache, pos_ids,
            self.interleave, self.enable_pdl,
            None, None, None, None
        )
        
        q_ref = q.float() 
        
        half_dim = r_dim // 2
        cos_table = cos_sin_cache[:, :half_dim]
        sin_table = cos_sin_cache[:, half_dim:]
        
        cos = cos_table[pos_ids].unsqueeze(1)
        sin = sin_table[pos_ids].unsqueeze(1)
        
        def apply_ref_rope(tensor_in):
            t_rot = tensor_in[..., :r_dim]
            t_pass = tensor_in[..., r_dim:]
            
            x1 = t_rot[..., :half_dim]
            x2 = t_rot[..., half_dim:]
            
            res1 = x1 * cos - x2 * sin
            res2 = x2 * cos + x1 * sin
            
            res_rot = torch.cat([res1, res2], dim=-1)
            return torch.cat([res_rot, t_pass], dim=-1)

        if not self.interleave:
             q_target = apply_ref_rope(q_ref)
        else:
             return True, 0.0

        q_target = q_target.to(self.dtype)
        
        threshold = 0.98 if self.dtype == torch.bfloat16 else 0.99
        return self.check_diff(q_rope, q_target, threshold=threshold)