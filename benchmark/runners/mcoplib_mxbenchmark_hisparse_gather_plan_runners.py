import torch
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


class Hisparse_gather_plan_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.row_bytes = config.get("row_bytes", 32)
        self.num_rows = config.get("num_rows", 4)
        self.top_k = config.get("top_k", 1)
        self.args = None

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "uint8/int32")
        state.add_summary("Shape", f"rows={self.num_rows} top_k={self.top_k} row_bytes={self.row_bytes}")
        state.add_element_count(self.num_rows * self.top_k)
        state.add_global_memory_reads(self.num_rows * self.row_bytes * 2)
        state.add_global_memory_writes(self.num_rows * self.row_bytes)

    def _build_inputs(self, dev_id):
        device = torch.device("cuda", dev_id)
        rows = self.num_rows
        top_k = self.top_k
        row_bytes = self.row_bytes

        if row_bytes <= 0 or row_bytes % 16 != 0:
            raise ValueError("row_bytes must be a positive multiple of 16")
        if rows < 0 or top_k < 0:
            raise ValueError("num_rows and top_k must be non-negative")

        host_cache = torch.arange(max(rows, 1) * row_bytes, dtype=torch.uint8).reshape(max(rows, 1), row_bytes).pin_memory()
        hot_cache = torch.full((max(rows, 1), row_bytes), 7, dtype=torch.uint8, device=device)
        global_indices = torch.zeros((rows, top_k), dtype=torch.int32, device=device)
        hot_indices = torch.zeros_like(global_indices)
        miss_mask = torch.ones_like(global_indices)

        if rows > 0 and top_k > 0:
            global_indices[:, 0] = torch.arange(rows, dtype=torch.int32, device=device)
            hot_indices[:, 0] = torch.arange(rows, dtype=torch.int32, device=device)

        return host_cache, hot_cache, global_indices, hot_indices, miss_mask, None, None, 0

    def prepare_and_get_launcher(self, dev_id, stream):
        self.args = self._build_inputs(dev_id)
        return lambda launch: torch.ops._C.hisparse_gather_plan(*self.args)

    def run_verification(self, dev_id):
        if not hasattr(torch.ops._C, "hisparse_gather_plan"):
            raise RuntimeError("hisparse_gather_plan is not registered")

        args = self._build_inputs(dev_id)
        host_cache, hot_cache, global_indices, hot_indices, miss_mask, request_state_indices, attention_indices, attention_block_stride = args

        try:
            torch.ops._C.hisparse_gather_plan(*args)
            torch.cuda.synchronize(dev_id)
        except Exception as exc:
            raise RuntimeError(
                f"hisparse_gather_plan failed: num_rows={self.num_rows}, "
                f"top_k={self.top_k}, row_bytes={self.row_bytes}: {exc}"
            ) from exc

        if self.num_rows > 0 and self.top_k > 0:
            expected = host_cache[:self.num_rows].to(device=hot_cache.device)
            torch.testing.assert_close(hot_cache, expected)

        return True, 0.0