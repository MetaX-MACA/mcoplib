import torch
import torch.nn.functional as F
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


class Masked_moe_activation_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.dtype = getattr(torch, config.get("dtype", "float16"))
        self.activation = config.get("activation", "silu")
        self.d = config.get("d", 64)
        self.mode = config.get("mode", "batched")
        self.num_tokens = config.get("num_tokens", 7)
        self.num_experts = config.get("num_experts", 4)
        self.max_num_tokens = config.get("max_num_tokens", 7)
        self.clamp_limit = config.get("clamp_limit", 1.0)
        self.alpha = config.get("alpha", 1.0)
        self.beta = config.get("beta", 0.0)
        self.situ_beta = config.get("situ_beta", 1.0)
        self.situ_linear_beta = config.get("situ_linear_beta", 0.0)
        self.device = "cuda"
        self.sentinel = 42.0
        self._prepare_inputs()

    def _prepare_inputs(self):
        torch.manual_seed(1234)
        gated_activations = ["silu", "gelu", "gelu_tanh", "silu_with_clamp", "swiglustep", "situ"]
        input_dim = 2 * self.d if self.activation in gated_activations else self.d

        if self.mode == "flat":
            self.input = torch.randn(self.num_tokens, input_dim, dtype=self.dtype, device=self.device)
            if self.activation in ["silu_with_clamp", "swiglustep"]:
                self.input.mul_(10)
            self.valid_counts = [self.num_tokens // 2]
            self.output = torch.full((self.num_tokens, self.d), self.sentinel, dtype=self.dtype, device=self.device)
        elif self.mode == "batched":
            self.input = torch.randn(self.num_experts, self.max_num_tokens, input_dim, dtype=self.dtype, device=self.device)
            if self.activation in ["silu_with_clamp", "swiglustep"]:
                self.input.mul_(10)
            default_counts = [0, 1, self.max_num_tokens // 2, self.max_num_tokens]
            self.valid_counts = default_counts[:self.num_experts]
            if self.num_experts > len(self.valid_counts):
                self.valid_counts.extend([self.max_num_tokens] * (self.num_experts - len(self.valid_counts)))
            self.output = torch.full((self.num_experts, self.max_num_tokens, self.d), self.sentinel, dtype=self.dtype, device=self.device)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        self.valid_token_counts = torch.tensor(self.valid_counts, dtype=torch.int32, device=self.device)

    def _get_op(self):
        if not hasattr(torch.ops._C, "masked_moe_activation"):
            raise RuntimeError("torch.ops._C.masked_moe_activation is not registered")
        return torch.ops._C.masked_moe_activation

    def define_metrics(self, state):
        pass

    def prepare_and_get_launcher(self, dev_id, stream):
        op = self._get_op()

        def launcher(_launch):
            with torch.cuda.stream(stream):
                op(self.output, self.input, self.valid_token_counts, self.activation, self.clamp_limit, self.alpha, self.beta, self.situ_beta, self.situ_linear_beta)

        return launcher

    def run_op(self):
        self._get_op()(self.output, self.input, self.valid_token_counts, self.activation, self.clamp_limit, self.alpha, self.beta, self.situ_beta, self.situ_linear_beta)
        torch.cuda.synchronize()

    def _reference(self, x):
        x = x.float()

        if self.activation == "silu_no_mul":
            return F.silu(x).to(self.dtype)
        if self.activation == "gelu_no_mul":
            return F.gelu(x, approximate="none").to(self.dtype)
        if self.activation == "gelu_tanh_no_mul":
            return F.gelu(x, approximate="tanh").to(self.dtype)
        if self.activation == "relu2_no_mul":
            return F.relu(x).square().to(self.dtype)

        gate, up = x.chunk(2, dim=-1)

        if self.activation == "silu":
            return (F.silu(gate) * up).to(self.dtype)
        if self.activation == "gelu":
            return (F.gelu(gate, approximate="none") * up).to(self.dtype)
        if self.activation == "gelu_tanh":
            return (F.gelu(gate, approximate="tanh") * up).to(self.dtype)

        if self.activation == "silu_with_clamp":
            gate = gate.clamp(max=self.clamp_limit)
            up = up.clamp(min=-self.clamp_limit, max=self.clamp_limit)
            activated = gate * torch.sigmoid(self.alpha * gate)
            return (activated * (up + self.beta)).to(self.dtype)

        if self.activation == "swiglustep":
            gate_silu = F.silu(gate).clamp(max=self.clamp_limit)
            up_clamped = up.clamp(min=-self.clamp_limit, max=self.clamp_limit)
            return (gate_silu * up_clamped).to(self.dtype)

        if self.activation == "situ":
            activated = self.situ_beta * torch.tanh(gate / self.situ_beta) * torch.sigmoid(gate)
            if self.situ_linear_beta > 0:
                up = self.situ_linear_beta * torch.tanh(up / self.situ_linear_beta)
            return (activated * up).to(self.dtype)

        raise ValueError(f"Unsupported activation: {self.activation}")

    def run_verification(self, dev_id):
        try:
            with torch.cuda.device(dev_id):
                self.output.fill_(self.sentinel)
                self.run_op()
                max_diff = 0.0

                if self.mode == "flat":
                    valid_count = self.valid_counts[0]
                    expected = self._reference(self.input[:valid_count])
                    actual = self.output[:valid_count]

                    if valid_count > 0:
                        diff = (actual.float() - expected.float()).abs()
                        max_diff = diff.max().item() if diff.numel() else 0.0
                        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

                    padding = self.output[valid_count:]
                    if padding.numel() > 0:
                        assert torch.all(padding == self.sentinel).item(), "Invalid rows were modified"

                else:
                    for expert, valid_count in enumerate(self.valid_counts):
                        expected = self._reference(self.input[expert, :valid_count])
                        actual = self.output[expert, :valid_count]

                        if valid_count > 0:
                            diff = (actual.float() - expected.float()).abs()
                            cur_diff = diff.max().item() if diff.numel() else 0.0
                            max_diff = max(max_diff, cur_diff)
                            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

                        padding = self.output[expert, valid_count:]
                        if padding.numel() > 0:
                            assert torch.all(padding == self.sentinel).item(), "Invalid rows were modified"

                return True, max_diff

        except (AssertionError, RuntimeError):
            return False, float("inf")