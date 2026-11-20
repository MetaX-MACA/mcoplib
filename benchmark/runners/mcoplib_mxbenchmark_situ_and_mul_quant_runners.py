import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Situ_and_mul_quant_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.num_tokens = config.get("num_tokens", 128)
        self.D = config.get("hidden_size", 3072)
        self.group_size = config.get("group_size", 128)
        self.beta = config.get("beta", 4.0)
        self.linear_beta = config.get("linear_beta", 25.0)
        self.num_valid_tokens = config.get("num_valid_tokens")
        self.topk = config.get("topk", 1)
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))

        if hasattr(torch, "float8_e4m3fn"):
            self.fp8_dtype = torch.float8_e4m3fn
        elif hasattr(torch, "float8_e4m3fnuz"):
            self.fp8_dtype = torch.float8_e4m3fnuz
        else:
            raise RuntimeError("PyTorch FP8 dtype is unavailable")

        if self.D % self.group_size != 0:
            raise ValueError("hidden_size must be divisible by group_size")

        if self.num_valid_tokens is not None:
            valid_rows = self.num_valid_tokens * self.topk
            if self.num_valid_tokens < 0 or valid_rows > self.num_tokens:
                raise ValueError("Invalid num_valid_tokens or topk")

        self.num_groups = self.D // self.group_size
        self.input_shape = (self.num_tokens, 2 * self.D)
        self.output_shape = (self.num_tokens, self.D)
        self.scale_shape = (self.num_tokens, self.num_groups)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary(
            "Shape",
            f"({self.num_tokens} {2 * self.D}) -> ({self.num_tokens} {self.D})"
        )

        input_elements = self.num_tokens * 2 * self.D
        output_elements = self.num_tokens * self.D
        scale_elements = self.num_tokens * self.num_groups

        state.add_element_count(input_elements)

        input_element_size = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        fp8_element_size = torch.empty((), dtype=self.fp8_dtype).element_size()

        state.add_global_memory_reads(input_elements * input_element_size)
        state.add_global_memory_writes(
            output_elements * fp8_element_size + scale_elements * 4
        )

    def _make_inputs(self, dev):
        inp = torch.randn(
            self.input_shape,
            dtype=torch.float32,
            device=dev
        )
        exps = torch.randint(
            1,
            13,
            (self.num_tokens, 1),
            device=dev
        ).float()
        inp = (inp * (2.0 ** exps)).to(self.dtype)

        out = torch.full(
            self.output_shape,
            17.0,
            dtype=self.fp8_dtype,
            device=dev
        )
        scale = torch.full(
            self.scale_shape,
            -1.0,
            dtype=torch.float32,
            device=dev
        )

        valid = None
        if self.num_valid_tokens is not None:
            valid = torch.tensor(
                self.num_valid_tokens,
                dtype=torch.int32,
                device=dev
            )

        return inp, out, scale, valid

    def _call_op(self, out, scale, inp, valid):
        torch.ops._C.situ_and_mul_quant(
            out,
            scale,
            inp,
            self.beta,
            self.linear_beta,
            self.group_size,
            valid,
            self.topk
        )

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = f"cuda:{dev_id}"

        with torch.cuda.stream(tc_s):
            inp, out, scale, valid = self._make_inputs(dev)

        return self.make_launcher(
            dev_id,
            torch.ops._C.situ_and_mul_quant,
            out,
            scale,
            inp,
            self.beta,
            self.linear_beta,
            self.group_size,
            valid,
            self.topk
        )

    def _reference(self, inp, rows):
        x = inp[:rows].double()
        gate, up = x.chunk(2, dim=-1)

        gate_out = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        up_out = (
            self.linear_beta * torch.tanh(up / self.linear_beta)
            if self.linear_beta > 0
            else up
        )

        act = (gate_out * up_out).to(self.dtype).float()
        act = act.reshape(rows, self.num_groups, self.group_size)

        fp8_max = torch.finfo(self.fp8_dtype).max
        ref_scale = act.abs().amax(dim=-1).clamp_min(1e-30) / fp8_max
        ref_out = (
            (act / ref_scale.unsqueeze(-1))
            .clamp(-fp8_max, fp8_max)
            .to(self.fp8_dtype)
        )

        return ref_out.reshape(rows, self.D), ref_scale

    def _fp8_code_dist(self, a, b):
        def ordered(x):
            bits = x.contiguous().view(torch.uint8).to(torch.int32)
            magnitude = bits & 0x7F
            return torch.where((bits & 0x80) != 0, -magnitude, magnitude)

        return (ordered(a) - ordered(b)).abs()

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        inp, out, scale, valid = self._make_inputs(dev)

        self._call_op(out, scale, inp, valid)

        rows = self.num_tokens if self.num_valid_tokens is None else self.num_valid_tokens * self.topk

        if rows < self.num_tokens:
            if not torch.equal(scale[rows:], torch.ones_like(scale[rows:])):
                return False, float("inf")

            sentinel = torch.full(out[rows:].shape, 17.0, dtype=self.fp8_dtype, device=dev)
            if not torch.equal(out[rows:].contiguous().view(torch.uint8), sentinel.contiguous().view(torch.uint8)):
                return False, float("inf")

        if rows == 0:
            return True, 0.0

        ref_out, ref_scale = self._reference(inp, rows)

        if not torch.isfinite(scale[:rows]).all() or not torch.isfinite(out[:rows].float()).all():
            return False, float("inf")

        scale_diff = (scale[:rows] - ref_scale).abs()
        scale_ok = torch.allclose(scale[:rows], ref_scale, rtol=3e-2, atol=1e-5)

        dist = self._fp8_code_dist(out[:rows], ref_out)
        max_dist = dist.max().item()
        ratio_le_one = (dist <= 1).float().mean().item()

        passed = scale_ok and max_dist <= 2 and ratio_le_one >= 0.99

        scale_rel_diff = (scale_diff / ref_scale.abs().clamp_min(1e-30)).max().item()
        diff_val = max(scale_rel_diff, float(max_dist))

        return passed, diff_val