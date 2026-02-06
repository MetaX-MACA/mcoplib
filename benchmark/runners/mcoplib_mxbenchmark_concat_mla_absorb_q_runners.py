import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Concat_mla_absorb_q_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 64)
        self.num_heads = config.get("num_heads", 128)
        
        self.dim_nope = 512
        self.dim_rope = 64
        self.dim_out = self.dim_nope + self.dim_rope

        if self.dtype != torch.bfloat16:
            print(f"[Warning] {name} only supports bfloat16 in kernel. Forcing dtype to bfloat16.")
            self.dtype = torch.bfloat16

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "bfloat16")
        state.add_summary("Shape", f"({self.batch_size} {self.num_heads} {self.dim_out})")
        
        total_elements = self.batch_size * self.num_heads * self.dim_out
        state.add_element_count(total_elements)
        
        element_size = 2

        read_nope = self.batch_size * self.num_heads * self.dim_nope
        read_rope = self.batch_size * self.num_heads * self.dim_rope
        total_reads = (read_nope + read_rope) * element_size
        
        total_writes = total_elements * element_size

        state.add_global_memory_reads(total_reads)
        state.add_global_memory_writes(total_writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(torch.ops.sgl_kernel, "concat_mla_absorb_q"):
             raise RuntimeError("Operator 'torch.ops.sgl_kernel.concat_mla_absorb_q' not found. Please ensure the extension is loaded.")

        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            q_nope = torch.randn((self.batch_size, self.num_heads, self.dim_nope), dtype=self.dtype, device=dev)
            q_rope = torch.randn((self.batch_size, self.num_heads, self.dim_rope), dtype=self.dtype, device=dev)
            
            out = torch.empty((self.batch_size, self.num_heads, self.dim_out), dtype=self.dtype, device=dev)

        return self.make_launcher(dev_id, torch.ops.sgl_kernel.concat_mla_absorb_q, q_nope, q_rope, out)

    def run_verification(self, dev_id):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(torch.ops.sgl_kernel, "concat_mla_absorb_q"):
             print("Error: torch.ops.sgl_kernel.concat_mla_absorb_q not available.")
             return False, 1.0

        dev = f'cuda:{dev_id}'
        
        q_nope = torch.randn((self.batch_size, self.num_heads, self.dim_nope), dtype=self.dtype, device=dev)
        q_rope = torch.randn((self.batch_size, self.num_heads, self.dim_rope), dtype=self.dtype, device=dev)
        out_op = torch.empty((self.batch_size, self.num_heads, self.dim_out), dtype=self.dtype, device=dev)
        
        torch.ops.sgl_kernel.concat_mla_absorb_q(q_nope, q_rope, out_op)
        
        q_nope_ref = q_nope.float()
        q_rope_ref = q_rope.float()
        out_ref = torch.cat([q_nope_ref, q_rope_ref], dim=-1)
        
        return self.check_diff(out_op, out_ref)