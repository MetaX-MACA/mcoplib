import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Mx_awq_dequantize_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.in_c = config.get("in_c", 4096)
        self.out_c = config.get("out_c", 4096)
        self.group_size = config.get("group_size", 128)
        self.pack_factor = 8

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.in_c} {self.out_c})")
        
        total_output_elements = self.in_c * self.out_c
        state.add_element_count(total_output_elements)
        
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        
        read_bytes = (self.in_c * (self.out_c // self.pack_factor) * 4) + \
                     ((self.in_c // self.group_size) * self.out_c * element_size) + \
                     ((self.in_c // self.group_size) * (self.out_c // self.pack_factor) * 4)
        write_bytes = total_output_elements * element_size
        
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            kernel = torch.randint(0, 2**31, (self.in_c, self.out_c // self.pack_factor), dtype=torch.int32, device=dev)
            scaling_factors = torch.randn((self.in_c // self.group_size, self.out_c), dtype=self.dtype, device=dev)
            zeros = torch.randint(0, 2**31, (self.in_c // self.group_size, self.out_c // self.pack_factor), dtype=torch.int32, device=dev)
            
        return self.make_launcher(
            dev_id, 
            torch.ops.sgl_kernel.mx_awq_dequantize, 
            kernel, scaling_factors, zeros, 1, 128, 8
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        
        kernel = torch.randint(0, 2**31, (self.in_c, self.out_c // self.pack_factor), dtype=torch.int32, device=dev)
        scaling_factors = torch.randn((self.in_c // self.group_size, self.out_c), dtype=self.dtype, device=dev)
        zeros = torch.randint(0, 2**31, (self.in_c // self.group_size, self.out_c // self.pack_factor), dtype=torch.int32, device=dev)
        
        output_tensor = torch.ops.sgl_kernel.mx_awq_dequantize(
            kernel, scaling_factors, zeros, 1, 128, 8
        )

        kernel_unpacked = torch.zeros((self.in_c, self.out_c), dtype=torch.int32, device=dev)
        zeros_unpacked = torch.zeros((self.in_c // self.group_size, self.out_c), dtype=torch.int32, device=dev)
        
        shift_map = [0, 16, 4, 20, 8, 24, 12, 28]
        
        for i in range(8):
            kernel_unpacked[:, i::8] = (kernel >> shift_map[i]) & 0xF
            zeros_unpacked[:, i::8] = (zeros >> shift_map[i]) & 0xF
            
        zeros_unpacked = zeros_unpacked.repeat_interleave(self.group_size, dim=0)
        scaling_factors_expanded = scaling_factors.repeat_interleave(self.group_size, dim=0)
        
        expected_out = (kernel_unpacked.to(self.dtype) - zeros_unpacked.to(self.dtype)) * scaling_factors_expanded
        
        pass_out, diff_out = self.check_diff(output_tensor, expected_out)
        
        return pass_out, diff_out