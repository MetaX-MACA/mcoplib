import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Moe_sum_reduce_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.token_num = config.get("token_num", 1024)
        self.topk_num = config.get("topk_num", 2)
        self.hidden_dim = config.get("hidden_dim", 4096)
        self.routed_scaling_factor = config.get("routed_scaling_factor", 1.0)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.token_num} {self.topk_num} {self.hidden_dim})")
        total_elements = self.token_num * self.topk_num * self.hidden_dim
        state.add_element_count(total_elements)
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        read_bytes = total_elements * element_size
        write_bytes = self.token_num * self.hidden_dim * element_size
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            shape_in = (self.token_num, self.topk_num, self.hidden_dim)
            shape_out = (self.token_num, self.hidden_dim)
            
            input_tensor = torch.randn(shape_in, dtype=self.dtype, device=dev)
            output_tensor = torch.empty(shape_out, dtype=self.dtype, device=dev)
            
        return self.make_launcher(
            dev_id, 
            torch.ops.sgl_kernel.moe_sum_reduce, 
            input_tensor, output_tensor, self.routed_scaling_factor
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        #token数量，每个token选中的专家数量，专家处理后的结果
        shape_in = (self.token_num, self.topk_num, self.hidden_dim)
        shape_out = (self.token_num, self.hidden_dim)
        input_tensor = torch.randn(shape_in, dtype=self.dtype, device=dev)
        output_tensor = torch.empty(shape_out, dtype=self.dtype, device=dev)
        ref_input_data = input_tensor.clone()
        torch.ops.sgl_kernel.moe_sum_reduce(
            input_tensor, output_tensor, self.routed_scaling_factor
        )

        
        input_f32 = ref_input_data.float()
        expected_out = torch.sum(input_f32, dim=1) * self.routed_scaling_factor
        is_passed, max_diff = self.check_diff(output_tensor, expected_out.to(self.dtype))
        
        return is_passed, max_diff