import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass


class Dsv4_fused_q_indexer_rope_hadamard_quant_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 8)
        self.num_heads = config.get("num_heads", 4)
        self.head_dim = config.get("head_dim", 128)
        self.rope_dim = config.get("rope_dim", 64)
        self.max_pos = config.get("max_pos", 256)
        self.weight_scale = config.get("weight_scale", 0.5)
        self.seed = config.get("seed", None)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        shape_str = f"({self.batch_size} {self.num_heads} {self.head_dim})"
        state.add_summary("Shape", shape_str)

        q_elems = self.batch_size * self.num_heads * self.head_dim
        w_elems = self.batch_size * self.num_heads
        freqs_elems = self.max_pos * self.rope_dim
        state.add_element_count(q_elems * 2 + w_elems + freqs_elems)

        es = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        reads = q_elems * es + w_elems * es + freqs_elems * 4 + self.batch_size * 4
        writes = q_elems * 1 + w_elems * 4
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            q_input = torch.randn(
                self.batch_size, self.num_heads, self.head_dim,
                dtype=self.dtype, device=dev
            )
            q_fp8 = torch.empty(
                self.batch_size, self.num_heads, self.head_dim,
                dtype=torch.uint8, device=dev
            )
            weight = torch.randn(
                self.batch_size, self.num_heads,
                dtype=self.dtype, device=dev
            )
            weights_out = torch.empty(
                self.batch_size, self.num_heads, 1,
                dtype=torch.float32, device=dev
            )
            freqs_cis = torch.randn(
                self.max_pos, self.rope_dim, dtype=torch.float32, device=dev
            )
            positions = torch.randint(
                0, self.max_pos, (self.batch_size,),
                dtype=torch.int32, device=dev
            )
        return self.make_launcher(
            dev_id, torch.ops.sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant,
            q_input, q_fp8, weight, weights_out, self.weight_scale,
            freqs_cis, positions
        )

    def run_verification(self, dev_id):
        # Smoke test: op should produce finite weights_out and non-zero q_fp8
        # (matches unit_test/test_dsv4_norm_rope.py::test_fused_q_indexer_rope_hadamard_quant_runs)
        dev = f'cuda:{dev_id}'
        B, H, D = 8, 4, 128
        rope_dim = self.rope_dim
        max_pos = self.max_pos
        torch.manual_seed(42)
        q_input = torch.randn(B, H, D, dtype=self.dtype, device=dev)
        q_fp8 = torch.empty(B, H, D, dtype=torch.uint8, device=dev)
        weight = torch.randn(B, H, dtype=self.dtype, device=dev)
        weights_out = torch.empty(B, H, 1, dtype=torch.float32, device=dev)
        freqs_cis = torch.randn(max_pos, rope_dim, dtype=torch.float32, device=dev)
        positions = torch.randint(0, max_pos, (B,), dtype=torch.int32, device=dev)
        torch.ops.sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant(
            q_input, q_fp8, weight, weights_out, self.weight_scale,
            freqs_cis, positions
        )
        finite_ok = bool(torch.isfinite(weights_out).all())
        nonzero_ok = bool(q_fp8.any())
        passed = finite_ok and nonzero_ok
        return passed, 0.0 if passed else 1.0
