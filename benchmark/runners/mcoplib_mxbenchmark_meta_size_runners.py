import torch
import sys
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
import mcoplib.sgl_kernel

class Meta_size_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "int64")
        state.add_summary("Shape", "(1)")
        state.add_element_count(0)
        state.add_global_memory_reads(0)
        state.add_global_memory_writes(0)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        return self.make_launcher(
            dev_id, 
            torch.ops.sgl_kernel.meta_size
        )

    def run_verification(self, dev_id):
        result_scalar = torch.ops.sgl_kernel.meta_size()
        out_op = torch.tensor([result_scalar], dtype=torch.float32, device=f'cuda:{dev_id}')
        out_ref = out_op.clone()
        return self.check_diff(out_op, out_ref)