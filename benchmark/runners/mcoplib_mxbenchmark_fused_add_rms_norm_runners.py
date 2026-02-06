import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Fused_add_rms_norm_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 16384)
        self.hidden_size = config.get("hidden_size", 4096)
        self.epsilon = config.get("epsilon", 1e-5)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.batch_size} {self.hidden_size})")
        total = self.batch_size * self.hidden_size
        state.add_element_count(total)
        element_size = 2 if self.dtype == torch.float16 else 4
        rw_factor = 2
        state.add_global_memory_reads(rw_factor * total * element_size)
        state.add_global_memory_writes(rw_factor * total * element_size)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            shape = (self.batch_size, self.hidden_size)
            inp = torch.randn(shape, dtype=self.dtype, device=dev)
            res = torch.randn(shape, dtype=self.dtype, device=dev)
            weight = torch.ones(self.hidden_size, dtype=self.dtype, device=dev)
        return self.make_launcher(dev_id, torch.ops._C.fused_add_rms_norm, inp, res, weight, self.epsilon)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        shape = (self.batch_size, self.hidden_size)
        inp = torch.randn(shape, dtype=self.dtype, device=dev)
        res = torch.randn(shape, dtype=self.dtype, device=dev)
        weight = torch.randn(self.hidden_size, dtype=self.dtype, device=dev)
        inp_ref = inp.clone().float()
        res_ref = res.clone().float()
        w_ref = weight.clone().float()
        torch.ops._C.fused_add_rms_norm(inp, res, weight, self.epsilon)
        res_ref = inp_ref + res_ref
        rms = torch.sqrt(res_ref.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        normed_ref = (res_ref / rms) * w_ref
        out_ref = normed_ref.to(self.dtype)
        return self.check_diff(inp, out_ref)