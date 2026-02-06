import torch
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

class Static_scaled_fp8_quant_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.rows = config.get("rows", 4096)
        self.cols = config.get("cols", 4096)
        self.input_dtype = torch.float16
        self.output_dtype = torch.float8_e4m3fn

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "fp16->fp8")
        state.add_summary("Shape", f"({self.rows} {self.cols})")
        total_elements = self.rows * self.cols
        state.add_element_count(total_elements)
        read_bytes = total_elements * 2
        write_bytes = total_elements * 1
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            shape = (self.rows, self.cols)
            input_tensor = torch.randn(shape, dtype=self.input_dtype, device=dev)
            scale = torch.tensor([1.0], dtype=torch.float32, device=dev)
            output_tensor = torch.empty(shape, dtype=self.output_dtype, device=dev)
        return self.make_launcher(
            dev_id,
            torch.ops._C.static_scaled_fp8_quant,
            output_tensor,
            input_tensor,
            scale
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        shape = (self.rows, self.cols)
        input_tensor = torch.randn(shape, dtype=self.input_dtype, device=dev)
        scale = torch.tensor([2.0], dtype=torch.float32, device=dev)
        output_tensor = torch.empty(shape, dtype=self.output_dtype, device=dev)
        torch.ops._C.static_scaled_fp8_quant(output_tensor, input_tensor, scale)
        ref_tensor = (input_tensor.float() / scale).to(self.output_dtype)
        return self.check_diff(output_tensor.float(), ref_tensor.float())