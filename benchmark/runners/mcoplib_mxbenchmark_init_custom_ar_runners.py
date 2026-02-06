import torch
import sys
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Init_custom_ar_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.rank = config.get("rank", 0)
        self.world_size = config.get("world_size", 2)
        self.full_nvlink = config.get("full_nvlink", True)
        self.rank_data_size = config.get("rank_data_size", 8 * 1024 * 1024)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "int64")
        shape_str = f"(WS {self.world_size} Rank {self.rank})"
        state.add_summary("Shape", shape_str)
        state.add_element_count(0)
        state.add_global_memory_reads(0)
        state.add_global_memory_writes(0)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(torch.ops.sgl_kernel, "init_custom_ar"):
             raise ImportError("torch.ops.sgl_kernel.init_custom_ar not found. Ensure mcoplib is installed correctly.")
        dev = f'cuda:{dev_id}'
        rank_data = torch.empty(self.rank_data_size, dtype=torch.uint8, device=dev)
        base_ptr = rank_data.data_ptr()
        meta_ptrs = [base_ptr for _ in range(self.world_size)]
        return self.make_launcher(
            dev_id,
            torch.ops.sgl_kernel.init_custom_ar,
            meta_ptrs,
            rank_data,
            self.rank,
            self.full_nvlink
        )

    def run_verification(self, dev_id):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(torch.ops.sgl_kernel, "init_custom_ar"):
            return False, 1.0
        dev = f'cuda:{dev_id}'
        rank_data = torch.empty(self.rank_data_size, dtype=torch.uint8, device=dev)
        base_ptr = rank_data.data_ptr()
        meta_ptrs = [base_ptr for _ in range(self.world_size)]
        state_ptr = torch.ops.sgl_kernel.init_custom_ar(
            meta_ptrs,
            rank_data,
            self.rank,
            self.full_nvlink
        )
        out_op = torch.tensor([state_ptr], dtype=torch.float32, device=dev)
        out_ref = out_op.clone()
        return self.check_diff(out_op, out_ref)