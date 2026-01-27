import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Gelu_tanh_and_mul_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 8)
        self.seq_len = config.get("seq_len", 128)
        self.hidden_size = config.get("hidden_size", 4096)
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"(B{self.batch_size}_S{self.seq_len}_H{self.hidden_size})")
        
        total_elements = self.batch_size * self.seq_len * self.hidden_size
        state.add_element_count(total_elements)
        
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        
        # Reads: Input is (B, S, 2*H)
        input_bytes = (self.batch_size * self.seq_len * 2 * self.hidden_size) * element_size
        # Writes: Output is (B, S, H)
        output_bytes = total_elements * element_size
        
        state.add_global_memory_reads(input_bytes)
        state.add_global_memory_writes(output_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            input_dim = 2 * self.hidden_size
            
            input_tensor = torch.randn(self.batch_size, self.seq_len, input_dim, dtype=self.dtype, device=dev)
            output = torch.empty(self.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype, device=dev)
            
        return self.make_launcher(
            dev_id, 
            torch.ops._C.gelu_tanh_and_mul, 
            output, 
            input_tensor
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        input_dim = 2 * self.hidden_size
        
        input_tensor = torch.randn(self.batch_size, self.seq_len, input_dim, dtype=self.dtype, device=dev)
        output = torch.empty(self.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype, device=dev)
        
        # Run Op
        torch.ops._C.gelu_tanh_and_mul(output, input_tensor)
        
        # Reference Logic
        x, y = input_tensor.chunk(2, dim=-1)
        expected = F.gelu(x.float(), approximate='tanh') * y.float()
        out_ref = expected.to(self.dtype)
        
        # Manual Tolerance Check for bfloat16/GELU-tanh approximation
        # Allow small deviation (e.g. < 1e-4) due to implementation differences
        out_op_f = output.float().flatten()
        out_ref_f = out_ref.float().flatten()
        
        cos_sim = F.cosine_similarity(out_op_f, out_ref_f, dim=0, eps=1e-6)
        diff_cos = 1.0 - cos_sim.item()
        
        if diff_cos < 1e-4:
            return True, diff_cos
            
        return self.check_diff(output, out_ref)