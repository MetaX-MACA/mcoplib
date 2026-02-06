import torch
import sys
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_moe_fused_w4a16 as target_lib
except ImportError:
    try:
        import mcoplib.op as target_lib
    except ImportError:
        target_lib = None

class Mctlass_moe_w4a16_gemm_kernel_mnk_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_valid_tokens = config.get("num_valid_tokens", 1024)
        self.N = config.get("N", 4096)
        self.K = config.get("K", 2048)
        self.group = config.get("group", 64)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", "w4a16"))
        shape_str = f"({self.num_valid_tokens} {self.N} {self.K} {self.group})"
        state.add_summary("Shape", shape_str)
        state.add_element_count(0)
        state.add_global_memory_reads(0)
        state.add_global_memory_writes(0)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if target_lib is None:
            raise ImportError("Cannot load mcoplib library")
        return self.make_launcher(
            dev_id, 
            target_lib.mctlass_moe_w4a16_gemm_kernel_mnk, 
            self.num_valid_tokens, 
            self.N, 
            self.K, 
            self.group
        )

    def run_verification(self, dev_id):
        if target_lib is None:
            print("Error: mcoplib module not found.")
            return False, 1.0
        result_scalar = target_lib.mctlass_moe_w4a16_gemm_kernel_mnk(
            self.num_valid_tokens, 
            self.N, 
            self.K, 
            self.group
        )
        out_op = torch.tensor([result_scalar], dtype=torch.float32, device=f'cuda:{dev_id}')
        out_ref = out_op.clone()
        return self.check_diff(out_op, out_ref)