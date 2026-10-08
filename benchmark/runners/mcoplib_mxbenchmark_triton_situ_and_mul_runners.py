import torch

from mcoplib.triton_situ_and_mul import situ_and_mul
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


class Triton_situ_and_mul_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.tokens = config.get("tokens", 2048)
        self.topk = config.get("topk", 16)
        self.input_width = config.get("input_width", 768)
        self.situ_beta = config.get("situ_beta", 4.0)
        self.situ_linear_beta = config.get("situ_linear_beta", 4.0)
        self.seed = config.get("seed", 1234)
        self.atol = config.get("atol", 3.0e-2)
        self.rtol = config.get("rtol", 3.0e-2)

        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if self.tokens <= 0:
            raise ValueError("tokens must be positive")
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        if self.input_width <= 0 or self.input_width % 2 != 0:
            raise ValueError("input_width must be positive and even")
        if self.situ_beta <= 0:
            raise ValueError("situ_beta must be greater than zero")
        if self.situ_linear_beta is not None and self.situ_linear_beta <= 0:
            raise ValueError("situ_linear_beta must be greater than zero")

        self.rows = self.tokens * self.topk
        self.hidden_size = self.input_width // 2

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary(
            "Shape",
            (
                f"tokens={self.tokens}, topk={self.topk}, "
                f"({self.rows} {self.input_width}) -> "
                f"({self.rows} {self.hidden_size})"
            ),
        )
        state.add_summary("situ_beta", str(self.situ_beta))
        state.add_summary("situ_linear_beta", str(self.situ_linear_beta))

        output_elements = self.rows * self.hidden_size
        element_size = torch.empty((), dtype=self.dtype).element_size()
        state.add_element_count(output_elements)
        state.add_global_memory_reads(
            self.rows * self.input_width * element_size
        )
        state.add_global_memory_writes(output_elements * element_size)

    def _prepare(self, dev_id, rows=None, seed=None):
        rows = self.rows if rows is None else rows
        seed = self.seed if seed is None else seed
        device = f"cuda:{dev_id}"
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        x = torch.randn(
            (rows, self.input_width),
            dtype=self.dtype,
            device=device,
        )
        out = torch.empty(
            (rows, self.hidden_size),
            dtype=self.dtype,
            device=device,
        )
        return x, out

    def _reference(self, x):
        gate, up = x.float().chunk(2, dim=-1)
        gate = (
            self.situ_beta
            * torch.tanh(gate / self.situ_beta)
            * torch.sigmoid(gate)
        )
        if self.situ_linear_beta is not None:
            up = self.situ_linear_beta * torch.tanh(
                up / self.situ_linear_beta
            )
        return (gate * up).to(x.dtype)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            x, out = self._prepare(dev_id)

        return self.make_launcher(
            dev_id,
            situ_and_mul,
            x,
            self.situ_beta,
            self.situ_linear_beta,
            out,
        )

    @torch.inference_mode()
    def run_verification(self, dev_id):
        # Use a non-tile-aligned row count to exercise the optimized kernel's
        # final masked program as well as its normal full tiles.
        verification_rows = min(self.rows, 4097)
        x, optimized = self._prepare(
            dev_id,
            rows=verification_rows,
            seed=self.seed + 1,
        )
        situ_and_mul(
            x,
            self.situ_beta,
            self.situ_linear_beta,
            out=optimized,
        )
        torch.cuda.synchronize()

        expected = self._reference(x)
        reference_match = torch.allclose(
            optimized,
            expected,
            atol=self.atol,
            rtol=self.rtol,
        )
        finite = bool(torch.isfinite(optimized).all().item())
        max_diff = (optimized.float() - expected.float()).abs().max().item()
        return bool(reference_match and finite), max_diff
