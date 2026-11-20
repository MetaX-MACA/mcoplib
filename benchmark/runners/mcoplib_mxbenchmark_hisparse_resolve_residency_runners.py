
import math
import torch
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


class Hisparse_resolve_residency_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_rows = config.get("num_rows", 2)
        self.top_k = config.get("top_k", 2)
        self.hot_size = config.get("hot_size", 8)
        self.max_union_rows = config.get("max_union_rows", self.hot_size)
        self.block_size = config.get("block_size", 4)
        self.row_bytes = config.get("row_bytes", 16)
        self.args = None

        if self.num_rows < 0:
            raise ValueError("num_rows must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.hot_size < self.top_k:
            raise ValueError("hot_size must be >= top_k")
        if self.hot_size > 32768:
            raise ValueError("hot_size must fit int16 slots")
        if self.max_union_rows <= 0 or self.max_union_rows > self.hot_size:
            raise ValueError("max_union_rows must be in [1, hot_size]")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.row_bytes <= 0 or self.row_bytes % 16 != 0:
            raise ValueError("row_bytes must be a positive multiple of 16")

    def define_metrics(self, state):
        rows = self.num_rows
        top_k = self.top_k
        hot_size = self.hot_size
        max_union_rows = self.max_union_rows
        block_size = self.block_size
        row_bytes = self.row_bytes

        state.add_summary("Op", self.name)
        state.add_summary("dtype", "uint8/int32/int16")
        state.add_summary("Shape", f"rows={rows} top_k={top_k} hot_size={hot_size} max_union_rows={max_union_rows} block_size={block_size}")

        host_bytes = max(16, rows * hot_size) * row_bytes
        hot_bytes = 2 * block_size * row_bytes
        index_bytes = rows * top_k * 4
        state.add_element_count(rows * top_k)
        state.add_global_memory_reads(host_bytes + hot_bytes + index_bytes + rows * hot_size * 6)
        state.add_global_memory_writes(rows * hot_size * 6 + index_bytes)

    def _build_inputs(self, dev_id):
        device = torch.device("cuda", dev_id)
        rows = self.num_rows
        top_k = self.top_k
        hot_size = self.hot_size
        max_union_rows = self.max_union_rows
        block_size = self.block_size
        row_bytes = self.row_bytes

        host_rows = max(16, rows * hot_size)
        table_cols = max(1, math.ceil(hot_size / block_size))
        table_rows = max(rows, 1)

        host_cache = torch.zeros((host_rows, row_bytes), dtype=torch.uint8, device="cpu", pin_memory=True)
        hot_cache = torch.zeros((2, block_size, row_bytes), dtype=torch.uint8, device=device)
        hot_block_table = torch.zeros((table_rows, table_cols), dtype=torch.int32, device=device)
        global_indices = torch.zeros((rows, top_k), dtype=torch.int32, device=device)
        hot_indices = torch.zeros((rows, top_k), dtype=torch.int32, device=device)
        device_global_indices = torch.zeros((table_rows, hot_size), dtype=torch.int32, device=device)[:rows]
        lru_slots = torch.zeros((table_rows, hot_size), dtype=torch.int16, device=device)[:rows]

        
        return (
            host_cache,
            hot_cache,
            hot_block_table,
            global_indices,
            hot_indices,
            device_global_indices,
            lru_slots,
            None,               # request_state_indices
            hot_size,           # region_stride
            max_union_rows,     # max_union_rows
            None,               # miss_mask
            None,               # stats
            None,               # attention_indices
            0,                  # attention_block_stride
            None,               # request_ids
            None,               # source_block_table
            block_size,         # source_block_size
            None,               # resolved_global_indices
            None,               # valid_counts
            None,               # swap_host_physical_rows
            None,               # swap_device_physical_rows
            None,               # swap_counts
            None,               # resident_block_table
            block_size,         # resident_block_size
            0,                  # resident_null_block
        )


    def prepare_and_get_launcher(self, dev_id, tc_s):
        self.args = self._build_inputs(dev_id)
        return self.make_launcher(dev_id, torch.ops._C.hisparse_resolve_residency, *self.args)

    def run_verification(self, dev_id):
        if not hasattr(torch.ops._C, "hisparse_resolve_residency"):
            raise RuntimeError("hisparse_resolve_residency is not registered")

        args = self._build_inputs(dev_id)

        try:
            torch.ops._C.hisparse_resolve_residency(*args)
            torch.cuda.synchronize(dev_id)
        except Exception as exc:
            raise RuntimeError(
                f"hisparse_resolve_residency failed: num_rows={self.num_rows}, "
                f"top_k={self.top_k}, hot_size={self.hot_size}, "
                f"max_union_rows={self.max_union_rows}, "
                f"block_size={self.block_size}: {exc}"
            ) from exc

        return True, 0.0
