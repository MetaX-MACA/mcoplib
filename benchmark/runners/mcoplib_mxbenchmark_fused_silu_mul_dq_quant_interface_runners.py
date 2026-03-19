import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Fused_silu_mul_dq_quant_interface_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 8192)
        self.hidden_size = config.get("hidden_size", 4096)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.num_tokens} {self.hidden_size})")
        
        total_output_elements = self.num_tokens * self.hidden_size
        state.add_element_count(total_output_elements)
        
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        
        read_bytes = self.num_tokens * (2 * self.hidden_size) * element_size
        write_bytes = (self.num_tokens * self.hidden_size * 1) + (self.num_tokens * 4)
        
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            input_tensor = torch.randn((self.num_tokens, 2 * self.hidden_size), dtype=self.dtype, device=dev)
            out_tensor = torch.empty((self.num_tokens, self.hidden_size), dtype=torch.int8, device=dev)
            scale_tensor = torch.empty((self.num_tokens,), dtype=torch.float32, device=dev)
            
        return self.make_launcher(
            dev_id, 
            torch.ops.sgl_kernel.fused_silu_mul_dq_quant_interface, 
            out_tensor, scale_tensor, input_tensor
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        
        input_tensor = torch.randn((self.num_tokens, 2 * self.hidden_size), dtype=self.dtype, device=dev)
        out_tensor = torch.empty((self.num_tokens, self.hidden_size), dtype=torch.int8, device=dev)
        scale_tensor = torch.empty((self.num_tokens,), dtype=torch.float32, device=dev)
        
        input_ref_data = input_tensor.clone()

        torch.ops.sgl_kernel.fused_silu_mul_dq_quant_interface(out_tensor, scale_tensor, input_tensor)

        input_f32 = input_ref_data.float()
        gate = input_f32[:, :self.hidden_size]
        up = input_f32[:, self.hidden_size:]
        
        expected_tmp = torch.nn.functional.silu(gate) * up
        
        absmax = expected_tmp.abs().max(dim=-1, keepdim=True)[0]
        absmax = torch.clamp(absmax, min=1e-8)
        expected_scale = absmax / 127.0
        
        expected_out = torch.round(expected_tmp / expected_scale)
        expected_out = torch.clamp(expected_out, -127, 127).to(torch.int8)

        pass_out, diff_out = self.check_diff(out_tensor, expected_out)
        pass_scale, diff_scale = self.check_diff(scale_tensor, expected_scale.squeeze(-1))
        
        is_passed = pass_out and pass_scale
        max_diff = max(diff_out, diff_scale)
        
        return is_passed, max_diff