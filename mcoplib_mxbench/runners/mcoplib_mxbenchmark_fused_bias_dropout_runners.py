import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.op as op
except ImportError:
    op = None

class Fused_bias_dropout_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 1024)
        self.hidden_size = config.get("hidden_size", 512)
        self.prob = config.get("prob", 0.0)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        
        # 【新增】添加 dtype 列
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.batch_size} {self.hidden_size})")
        total = self.batch_size * self.hidden_size
        state.add_element_count(total)
        element_size = 2 if self.dtype == torch.float16 else 4
        state.add_global_memory_reads(total * 2 * element_size)
        state.add_global_memory_writes(total * 1 * element_size)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            shape = (self.batch_size, self.hidden_size)
            inp = torch.randn(shape, dtype=self.dtype, device=dev)
            res = torch.randn(shape, dtype=self.dtype, device=dev)
        return self.make_launcher(dev_id, op.fused_bias_dropout, inp, res, self.prob)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        shape = (self.batch_size, self.hidden_size)
        inp = torch.randn(shape, dtype=self.dtype, device=dev)
        res = torch.randn(shape, dtype=self.dtype, device=dev)
        out_op = op.fused_bias_dropout(inp, res, self.prob)
        inp_ref = inp.float()
        res_ref = res.float()
        out_ref = inp_ref + res_ref
        if self.prob > 0: return True, 0.0 
        out_ref = out_ref.to(self.dtype)
        return self.check_diff(out_op, out_ref)