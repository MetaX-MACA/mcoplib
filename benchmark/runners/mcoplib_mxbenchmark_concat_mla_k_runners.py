import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Concat_mla_k_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 4096)
        
        self.num_heads = 128
        self.nope_dim = 128
        self.rope_dim = 64
        self.total_dim = self.nope_dim + self.rope_dim

        if self.dtype != torch.bfloat16:
            print(f"[Warning] {name} only supports bfloat16 in kernel. Forcing dtype to bfloat16.")
            self.dtype = torch.bfloat16

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "bfloat16")
        state.add_summary("Shape", f"({self.num_tokens} {self.num_heads} {self.total_dim})")
        
        total_out_elements = self.num_tokens * self.num_heads * self.total_dim
        state.add_element_count(total_out_elements)
        
        element_size = 2

        read_nope = self.num_tokens * self.num_heads * self.nope_dim
        read_rope = self.num_tokens * 1 * self.rope_dim
        
        total_reads = (read_nope + read_rope) * element_size
        total_writes = total_out_elements * element_size

        state.add_global_memory_reads(total_reads)
        state.add_global_memory_writes(total_writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(torch.ops.sgl_kernel, "concat_mla_k"):
             raise RuntimeError("Operator 'torch.ops.sgl_kernel.concat_mla_k' not found. Please ensure the extension is loaded.")

        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            k_nope = torch.randn((self.num_tokens, self.num_heads, self.nope_dim), dtype=self.dtype, device=dev)
            k_rope = torch.randn((self.num_tokens, 1, self.rope_dim), dtype=self.dtype, device=dev)
            
            k_out = torch.empty((self.num_tokens, self.num_heads, self.total_dim), dtype=self.dtype, device=dev)

        return self.make_launcher(dev_id, torch.ops.sgl_kernel.concat_mla_k, k_out, k_nope, k_rope)

    def run_verification(self, dev_id):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(torch.ops.sgl_kernel, "concat_mla_k"):
             print("Error: torch.ops.sgl_kernel.concat_mla_k not available.")
             return False, 1.0

        dev = f'cuda:{dev_id}'
        
        k_nope = torch.randn((self.num_tokens, self.num_heads, self.nope_dim), dtype=self.dtype, device=dev)
        k_rope = torch.randn((self.num_tokens, 1, self.rope_dim), dtype=self.dtype, device=dev)
        k_out_op = torch.empty((self.num_tokens, self.num_heads, self.total_dim), dtype=self.dtype, device=dev)
        
        torch.ops.sgl_kernel.concat_mla_k(k_out_op, k_nope, k_rope)
        
        k_rope_expanded = k_rope.expand(self.num_tokens, self.num_heads, self.rope_dim)
        k_out_ref = torch.cat([k_nope, k_rope_expanded], dim=-1)
        
        return self.check_diff(k_out_op, k_out_ref)