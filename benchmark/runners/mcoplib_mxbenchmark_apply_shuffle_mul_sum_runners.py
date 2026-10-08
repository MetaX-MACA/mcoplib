import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Apply_shuffle_mul_sum_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_elements = config.get("num_elements", 4096)
        self.num_reduced = config.get("num_reduced", 1024)
        self.hidden_dim = config.get("hidden_dim", 4096) 
        assert self.num_elements % self.num_reduced == 0, "num_elements 必须能被 num_reduced 整除"
        self.topk = self.num_elements // self.num_reduced

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.num_elements} -> {self.num_reduced} {self.hidden_dim})")
        read_elements = self.num_elements * self.hidden_dim
        write_elements = self.num_reduced * self.hidden_dim
        state.add_element_count(read_elements + write_elements)
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        read_bytes = (read_elements * element_size) + \
                     (self.num_elements * 4) + \
                     (self.num_elements * element_size)
        write_bytes = (write_elements * element_size)
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            inp = torch.randn((self.num_elements, self.hidden_dim), dtype=self.dtype, device=dev)
            out = torch.zeros((self.num_reduced, self.hidden_dim), dtype=self.dtype, device=dev)
            perm = torch.randint(0, self.num_elements, (self.num_elements,), dtype=torch.int32, device=dev)
            factors = torch.randn(self.num_elements, dtype=self.dtype, device=dev)
        return self.make_launcher(
            dev_id, 
            torch.ops.sgl_kernel.apply_shuffle_mul_sum, 
            inp, out, perm, factors
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        inp_tensor = torch.randn((self.num_elements, self.hidden_dim), dtype=self.dtype, device=dev)
        out_tensor = torch.zeros((self.num_reduced, self.hidden_dim), dtype=self.dtype, device=dev)
        perm_tensor = torch.randint(0, self.num_elements, (self.num_elements,), dtype=torch.int32, device=dev)
        factors_tensor = torch.randn(self.num_elements, dtype=self.dtype, device=dev)
        torch.ops.sgl_kernel.apply_shuffle_mul_sum(
            inp_tensor, out_tensor, perm_tensor, factors_tensor
        )
        inp_f32 = inp_tensor.float()
        factors_f32 = factors_tensor.float().unsqueeze(-1)
        gathered_input = inp_f32[perm_tensor.long()]
        scaled_input = gathered_input * factors_f32   
        m = self.num_reduced
        expected_out = scaled_input.view(m, self.topk, self.hidden_dim).sum(dim=1)
        pass_out, diff_out = self.check_diff(out_tensor, expected_out.to(self.dtype))
        
        return pass_out, diff_out