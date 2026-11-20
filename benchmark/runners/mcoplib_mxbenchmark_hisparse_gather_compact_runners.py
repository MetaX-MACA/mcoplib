import torch
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


class Hisparse_gather_compact_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.row_bytes = config.get("row_bytes", 32)
        self.num_rows = config.get("num_rows", 4)
        self.top_k = config.get("top_k", 4)
        self.num_host_rows = max(self.num_rows * self.top_k, 32)
        self.block_size = max(8, self.top_k)
        self.num_blocks = max(2, (self.num_rows * self.top_k + self.block_size - 1) // self.block_size)
        self.num_hot_rows = self.num_blocks * self.block_size
        self.args = None
        self.expected = None
        self._op_executed = False

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "uint8/int32")
        state.add_summary("Shape", f"rows={self.num_rows} top_k={self.top_k} row_bytes={self.row_bytes}")
        state.add_element_count(self.num_rows * self.top_k)
        state.add_global_memory_reads(self.num_rows * self.top_k * self.row_bytes + self.num_rows * self.top_k * 8 + self.num_rows * 4)
        state.add_global_memory_writes(self.num_rows * self.top_k * self.row_bytes)

    def _build_inputs(self, dev_id):
        rows = self.num_rows
        top_k = self.top_k
        row_bytes = self.row_bytes

        if row_bytes <= 0 or row_bytes % 16 != 0:
            raise ValueError("row_bytes must be a positive multiple of 16")
        if rows < 0 or top_k < 0:
            raise ValueError("num_rows and top_k must be non-negative")

        device = torch.device("cuda", dev_id)
        num_host_rows = max(rows * top_k, 32)
        block_size = max(8, top_k)
        num_blocks = max(2, (rows * top_k + block_size - 1) // block_size)
        num_hot_rows = num_blocks * block_size

        host_cache = torch.arange(num_host_rows * row_bytes, dtype=torch.int32).remainder(256).to(torch.uint8).reshape(num_host_rows, row_bytes).pin_memory()
        hot_cache = torch.full((num_blocks, block_size, row_bytes), 7, dtype=torch.uint8, device=device)
        miss_global_indices = torch.zeros((rows, top_k), dtype=torch.int32, device=device)
        miss_hot_indices = torch.zeros((rows, top_k), dtype=torch.int32, device=device)
        miss_counts = torch.full((rows,), top_k, dtype=torch.int32, device=device)

        if rows > 0 and top_k > 0:
            indices = torch.arange(rows * top_k, dtype=torch.int32, device=device).reshape(rows, top_k)
            miss_global_indices.copy_(indices.remainder(num_host_rows))
            miss_hot_indices.copy_(indices.remainder(num_hot_rows))

        return host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts

    def _build_expected(self, args):
        host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts = args
        expected = hot_cache.clone()
        expected_flat = expected.reshape(-1, self.row_bytes)
        global_cpu = miss_global_indices.cpu()
        hot_cpu = miss_hot_indices.cpu()
        counts_cpu = miss_counts.cpu()
        num_host_rows = host_cache.size(0)
        num_hot_rows = hot_cache.size(0) * hot_cache.size(1)

        for row in range(self.num_rows):
            count = min(max(int(counts_cpu[row].item()), 0), self.top_k)
            for col in range(count):
                src = int(global_cpu[row, col].item())
                dst = int(hot_cpu[row, col].item())

                if src < 0 or dst < 0 or dst >= num_hot_rows:
                    continue

                if src < num_host_rows:
                    expected_flat[dst].copy_(host_cache[src])
                else:
                    expected_flat[dst].zero_()

        return expected

    def prepare_and_get_launcher(self, dev_id, stream):
        self.args = self._build_inputs(dev_id)
        self.expected = self._build_expected(self.args)
        self._op_executed = False

        def launcher(launch):
            stream_obj = self.as_torch_stream(launch.get_stream(), dev_id)
            with torch.cuda.stream(stream_obj):
                torch.ops._C.hisparse_gather_compact(*self.args)
                self._op_executed = True

        return launcher

    def run_verification(self, dev_id):
        if not hasattr(torch.ops._C, "hisparse_gather_compact"):
            raise RuntimeError("hisparse_gather_compact is not registered")

        if self.args is None:
            self.args = self._build_inputs(dev_id)
            self.expected = self._build_expected(self.args)
            torch.ops._C.hisparse_gather_compact(*self.args)
            self._op_executed = True

        if self.expected is None:
            self.expected = self._build_expected(self.args)

        if not self._op_executed:
            raise RuntimeError("hisparse_gather_compact launcher was not executed on this runner instance")

        try:
            torch.cuda.synchronize(dev_id)
        except Exception as exc:
            raise RuntimeError(f"hisparse_gather_compact synchronization failed: {exc}") from exc

        hot_cache = self.args[1]
        actual_cpu = hot_cache.cpu()
        expected_cpu = self.expected.cpu()

        if not torch.equal(actual_cpu, expected_cpu):
            mismatch = torch.nonzero(actual_cpu != expected_cpu, as_tuple=False)
            pos = mismatch[0].tolist()
            actual_value = actual_cpu[tuple(pos)].item()
            expected_value = expected_cpu[tuple(pos)].item()
            raise AssertionError(f"hisparse_gather_compact verification failed: first_mismatch={pos}, actual={actual_value}, expected={expected_value}, row_bytes={self.row_bytes}, num_rows={self.num_rows}, top_k={self.top_k}")

        return True, 0.0