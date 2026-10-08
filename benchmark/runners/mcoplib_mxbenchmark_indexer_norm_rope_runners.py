import torch

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

import mcoplib.op as ops


BF16_RELATIVE_ULP = 2.0**-7


class Indexer_norm_rope_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = int(config.get("num_tokens", 4096))
        self.q_heads = int(config.get("q_heads", 4))
        self.k_heads = int(config.get("k_heads", 1))
        self.head_dim = int(config.get("head_dim", 256))
        self.rotary_dim = int(config.get("rotary_dim", 16))
        self.max_position = int(config.get("max_position", 4096))
        self.eps = float(config.get("eps", 1e-6))
        self.q_weight_bias = float(config.get("q_weight_bias", 1.0))

    def _make_inputs(self, dev, seed=20260902):
        generator = torch.Generator(device=dev).manual_seed(
            seed + self.num_tokens + self.q_heads
        )

        def randn(*shape):
            return torch.randn(
                shape, generator=generator, device=dev, dtype=self.dtype
            ).contiguous()

        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(
                    self.rotary_dim, device=dev, dtype=torch.float32
                )
                / self.rotary_dim
            )
        )
        angles = torch.arange(
            self.max_position, device=dev, dtype=torch.float32
        )[:, None] * inv_freq[None, :]
        return {
            "q": randn(self.num_tokens, self.q_heads * self.head_dim),
            "k": randn(self.num_tokens, self.k_heads * self.head_dim),
            "z": randn(self.num_tokens, self.k_heads * self.head_dim),
            "qw": randn(self.head_dim),
            "kw": randn(self.head_dim),
            "kb": randn(self.head_dim),
            "cos": angles.cos().to(self.dtype).contiguous(),
            "sin": angles.sin().to(self.dtype).contiguous(),
            "positions": torch.randint(
                0,
                self.max_position,
                (self.num_tokens,),
                generator=generator,
                device=dev,
                dtype=torch.int32,
            ),
        }

    def _call(self, inputs):
        return ops.indexer_norm_rope(
            inputs["q"],
            inputs["k"],
            inputs["z"],
            inputs["qw"],
            inputs["kw"],
            inputs["kb"],
            inputs["cos"],
            inputs["sin"],
            inputs["positions"],
            self.head_dim,
            self.rotary_dim,
            self.eps,
            self.q_weight_bias,
        )

    def _reference(self, inputs):
        q = inputs["q"].view(
            self.num_tokens, self.q_heads, self.head_dim
        ).float()
        k = inputs["k"].view(
            self.num_tokens, self.k_heads, self.head_dim
        ).float()
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q = q * (inputs["qw"].float() + self.q_weight_bias)
        k_mean = k.mean(-1, keepdim=True)
        k_var = (k - k_mean).square().mean(-1, keepdim=True)
        k = (k - k_mean) * torch.rsqrt(k_var + self.eps)
        k = k * inputs["kw"].float() + inputs["kb"].float()
        cos = inputs["cos"][inputs["positions"].long()].float().unsqueeze(1)
        sin = inputs["sin"][inputs["positions"].long()].float().unsqueeze(1)

        def rotate(value):
            result = value.clone()
            real = value[..., : self.rotary_dim]
            imag = value[..., self.rotary_dim : 2 * self.rotary_dim]
            result[..., : self.rotary_dim] = real * cos - imag * sin
            result[..., self.rotary_dim : 2 * self.rotary_dim] = (
                real * sin + imag * cos
            )
            return result.to(self.dtype)

        return rotate(q).view_as(inputs["q"]), rotate(k).view_as(inputs["k"])

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary(
            "Shape",
            f"(T={self.num_tokens} QH={self.q_heads} KH={self.k_heads} "
            f"D={self.head_dim} R={self.rotary_dim} "
            f"qwb={self.q_weight_bias:g})",
        )

        heads = self.q_heads + self.k_heads
        qk_elements = self.num_tokens * heads * self.head_dim
        element_size = torch.empty((), dtype=self.dtype).element_size()
        qk_bytes = qk_elements * element_size
        state.add_element_count(qk_elements)
        state.add_global_memory_reads(qk_bytes)
        state.add_global_memory_writes(qk_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            inputs = self._make_inputs(f"cuda:{dev_id}")
        return self.make_launcher(dev_id, self._call, inputs)

    @torch.inference_mode()
    def run_verification(self, dev_id):
        inputs = self._make_inputs(f"cuda:{dev_id}", seed=20260812)
        q_ref, k_ref = self._reference(inputs)
        q_out, k_out, z_out = self._call(inputs)
        torch.cuda.synchronize()

        max_error = max(
            float((q_out.float() - q_ref.float()).abs().max()),
            float((k_out.float() - k_ref.float()).abs().max()),
        )
        peak = max(
            float(q_ref.float().abs().max()),
            float(k_ref.float().abs().max()),
        )
        bound = 2.0 * BF16_RELATIVE_ULP * peak
        passed = (
            max_error <= bound
            and torch.equal(z_out, inputs["z"])
            and z_out.data_ptr() == inputs["z"].data_ptr()
        )
        return passed, max_error
