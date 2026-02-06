import torch
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

class Dynamic_per_token_scaled_fp8_quant_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 128)
        self.hidden_size = config.get("hidden_size", 4096)
        self.input_dtype = torch.float16
        self.output_dtype = torch.float8_e4m3fn

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "fp16->fp8")
        state.add_summary("Shape", f"({self.num_tokens} {self.hidden_size})")
        total_elements = self.num_tokens * self.hidden_size
        state.add_element_count(total_elements)
        read_bytes = total_elements * 2
        write_bytes = total_elements * 1 + (self.num_tokens * 4)
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            shape = (self.num_tokens, self.hidden_size)
            scale_shape = (self.num_tokens, 1)
            input_tensor = torch.randn(shape, dtype=self.input_dtype, device=dev)
            output_tensor = torch.empty(shape, dtype=self.output_dtype, device=dev)
            scales = torch.empty(scale_shape, dtype=torch.float32, device=dev)
            scale_ub = None
        return self.make_launcher(
            dev_id, 
            torch.ops._C.dynamic_per_token_scaled_fp8_quant, 
            output_tensor, 
            input_tensor, 
            scales,
            scale_ub
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        shape = (self.num_tokens, self.hidden_size)
        scale_shape = (self.num_tokens, 1)
        input_tensor = torch.randn(shape, dtype=self.input_dtype, device=dev)
        output_tensor = torch.empty(shape, dtype=self.output_dtype, device=dev)
        scales = torch.empty(scale_shape, dtype=torch.float32, device=dev)
        scale_ub = None
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(
            output_tensor, 
            input_tensor, 
            scales, 
            scale_ub
        )
        safe_scales = scales.clone()
        safe_scales[safe_scales == 0] = 1.0
        ref_tensor = (input_tensor.float() / safe_scales).to(self.output_dtype)
        return self.check_diff(output_tensor.float(), ref_tensor.float())