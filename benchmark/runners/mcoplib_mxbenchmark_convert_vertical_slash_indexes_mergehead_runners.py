import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass


class Convert_vertical_slash_indexes_mergehead_runner(OpBenchmarkBase):
    """Benchmark for sgl_kernel.convert_vertical_slash_indexes_mergehead.

    Like convert_vertical_slash_indexes but per-head index counts
    (vertical_indices_count, slash_indices_count). Reference:
    op/sglang/csrc/attention/vertical_slash_index.cu.

    Verification: determinism (two identical runs produce bit-exact outputs)
    plus range invariants (block_count <= nnz_s, column_count <= nnz_v).
    """

    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 1)
        self.num_heads = config.get("num_heads", 4)
        self.context_size = config.get("context_size", 128)
        self.block_size_m = config.get("block_size_m", 64)
        self.block_size_n = config.get("block_size_n", 64)
        self.nnz_v = config.get("nnz_v", 8)
        self.nnz_s = config.get("nnz_s", 8)
        self.causal = config.get("causal", True)
        self.dtype = getattr(torch, config.get("dtype", "int32"))
        self.num_rows = (self.context_size + self.block_size_m - 1) // self.block_size_m
        self.seed = config.get("seed", None)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"(B{self.batch_size}_H{self.num_heads}_CTX{self.context_size})")
        total_out = self.batch_size * self.num_heads * self.num_rows * (2 + self.nnz_s + self.nnz_v)
        state.add_element_count(total_out)
        element_size = 4
        size_in = (self.batch_size * 2) + \
                  (self.batch_size * self.num_heads * (self.nnz_v + self.nnz_s)) + \
                  (self.num_heads * 2)
        size_out = (self.batch_size * self.num_heads * self.num_rows) * (2 + self.nnz_s + self.nnz_v)
        state.add_global_memory_reads(size_in * element_size)
        state.add_global_memory_writes(size_out * element_size)

    def _prepare_args(self, device, seed=None):
        if seed is None:
            seed = self.seed if self.seed is not None else 42
        gen = torch.Generator(device=device).manual_seed(seed)
        q_seqlens = torch.full((self.batch_size,), self.context_size, dtype=self.dtype, device=device)
        kv_seqlens = torch.full((self.batch_size,), self.context_size, dtype=self.dtype, device=device)
        vertical_indexes, _ = torch.sort(torch.randint(0, self.context_size, (self.batch_size, self.num_heads, self.nnz_v), dtype=self.dtype, device=device, generator=gen), dim=-1)
        slash_indexes, _ = torch.sort(torch.randint(0, self.context_size, (self.batch_size, self.num_heads, self.nnz_s), dtype=self.dtype, device=device, generator=gen), dim=-1)
        vertical_indices_count = torch.randint(1, self.nnz_v + 1, (self.num_heads,), dtype=self.dtype, device=device, generator=gen)
        slash_indices_count = torch.randint(1, self.nnz_s + 1, (self.num_heads,), dtype=self.dtype, device=device, generator=gen)
        block_count = torch.zeros((self.batch_size, self.num_heads, self.num_rows), dtype=self.dtype, device=device)
        column_count = torch.zeros((self.batch_size, self.num_heads, self.num_rows), dtype=self.dtype, device=device)
        block_offset = torch.zeros((self.batch_size, self.num_heads, self.num_rows, self.nnz_s), dtype=self.dtype, device=device)
        column_index = torch.zeros((self.batch_size, self.num_heads, self.num_rows, self.nnz_v), dtype=self.dtype, device=device)
        return [block_count, block_offset, column_count, column_index,
                q_seqlens, kv_seqlens, vertical_indexes, slash_indexes,
                vertical_indices_count, slash_indices_count]

    def _call_op(self, args):
        torch.ops.sgl_kernel.convert_vertical_slash_indexes_mergehead(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
            args[8], args[9],
            self.context_size, self.block_size_m, self.block_size_n, self.causal,
        )

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            args = self._prepare_args(dev)
        return self.make_launcher(dev_id, torch.ops.sgl_kernel.convert_vertical_slash_indexes_mergehead,
                                  *args,
                                  self.context_size, self.block_size_m, self.block_size_n, self.causal)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        args1 = self._prepare_args(dev, seed=7)
        self._call_op(args1)
        args2 = self._prepare_args(dev, seed=7)
        self._call_op(args2)
        torch.cuda.synchronize()

        block_count, block_offset, column_count, column_index = args1[0], args1[1], args1[2], args1[3]
        bc2, bo2, cc2, ci2 = args2[0], args2[1], args2[2], args2[3]

        same = (torch.equal(block_count, bc2) and torch.equal(block_offset, bo2)
                and torch.equal(column_count, cc2) and torch.equal(column_index, ci2))
        valid = ((block_count >= 0).all() and (block_count <= self.nnz_s).all()
                 and (column_count >= 0).all() and (column_count <= self.nnz_v).all())
        passed = bool(same and valid)
        return passed, 0.0 if passed else 1.0
