import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Fused_add_rmsnorm_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 1024)
        self.hidden_size = config.get("hidden_size", 512)
        self.eps = config.get("eps", 1e-6)
        self.enable_pdl = config.get("enable_pdl", False)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.batch_size} {self.hidden_size})")
        total_elements = self.batch_size * self.hidden_size
        state.add_element_count(total_elements)
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        read_bytes = (2 * total_elements * element_size) + (self.hidden_size * element_size)
        write_bytes = (2 * total_elements * element_size)
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            shape = (self.batch_size, self.hidden_size)
            inp = torch.randn(shape, dtype=self.dtype, device=dev)
            res = torch.randn(shape, dtype=self.dtype, device=dev)
            weight = torch.randn(self.hidden_size, dtype=self.dtype, device=dev)
        return self.make_launcher(
            dev_id, 
            torch.ops.sgl_kernel.fused_add_rmsnorm, 
            inp, res, weight, self.eps, self.enable_pdl
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        shape = (self.batch_size, self.hidden_size)
        inp_tensor = torch.randn(shape, dtype=self.dtype, device=dev)
        res_tensor = torch.randn(shape, dtype=self.dtype, device=dev)
        weight_tensor = torch.randn(self.hidden_size, dtype=self.dtype, device=dev)
        inp_ref_data = inp_tensor.clone()
        res_ref_data = res_tensor.clone()
        torch.ops.sgl_kernel.fused_add_rmsnorm(
            inp_tensor, res_tensor, weight_tensor, self.eps, self.enable_pdl
        )
        inp_f32 = inp_ref_data.float()
        res_f32 = res_ref_data.float()
        w_f32 = weight_tensor.float()
        expected_res = inp_f32 + res_f32
        rstd = torch.rsqrt(expected_res.pow(2).mean(-1, keepdim=True) + self.eps)
        expected_out = expected_res * rstd * w_f32
        pass_res, diff_res = self.check_diff(res_tensor, expected_res.to(self.dtype))
        pass_out, diff_out = self.check_diff(inp_tensor, expected_out.to(self.dtype))
        is_passed = pass_res and pass_out
        max_diff = max(diff_res, diff_out)
        return is_passed, max_diff