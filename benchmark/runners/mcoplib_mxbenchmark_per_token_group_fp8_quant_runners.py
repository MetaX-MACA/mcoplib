import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Per_token_group_fp8_quant_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.m = config.get("m", 32)
        self.n = config.get("n", 1024)
        self.group_size = config.get("group_size", 128)
        self.fp8_min = -448.0
        self.fp8_max = 448.0
        self.seed = config.get("seed", None)
        self.eps = 1e-10

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        shape_str = f"({self.m} {self.n})"
        state.add_summary("Shape", shape_str)

        in_elems = self.m * self.n
        out_elems = self.m * self.n
        scale_elems = self.m * (self.n // self.group_size)
        state.add_element_count(in_elems + out_elems + scale_elems)

        in_es = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        reads = in_elems * in_es
        writes = out_elems * 1 + scale_elems * 4
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            x = torch.randn(self.m, self.n, dtype=self.dtype, device=dev)
            out_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            out_s = torch.empty((self.m, self.n // self.group_size), device=dev, dtype=torch.float32)
        return self.make_launcher(
            dev_id, torch.ops._C.per_token_group_fp8_quant,
            x, out_q, out_s, self.group_size, self.eps,
            self.fp8_min, self.fp8_max, False, False, False
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        m, n, G = 32, 1024, 128
        x = torch.randn(m, n, dtype=self.dtype, device=dev)
        out_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        out_s = torch.empty((m, n // G), device=dev, dtype=torch.float32)
        torch.ops._C.per_token_group_fp8_quant(
            x, out_q, out_s, G, self.eps, self.fp8_min, self.fp8_max, False, False, False
        )
        x_view = x.view(m, n // G, G)
        scales = torch.clamp(x_view.abs().amax(dim=-1).float(), min=self.eps) / self.fp8_max
        q_ref = torch.clamp(x_view / scales.unsqueeze(-1), self.fp8_min, self.fp8_max).to(torch.float8_e4m3fn)
        q_ref = q_ref.view_as(x)
        return self.check_diff(out_q.float(), q_ref.float(), threshold=0.9999)
