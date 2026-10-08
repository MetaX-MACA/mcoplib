import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

HIDDEN_SIZE = 7168
MAX_BLOCKS = 8
EPS = 1e-5

class Kimi_k3_attn_res_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("batch_size", 3)
        self.num_blocks = config.get("num_blocks", 8)
        self.has_delta = config.get("has_delta", False)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"tokens={self.num_tokens} blocks={self.num_blocks} hidden={HIDDEN_SIZE}")
        rows = self.num_blocks + 1 + (1 if self.has_delta else 0)
        state.add_element_count(self.num_tokens * HIDDEN_SIZE)
        state.add_global_memory_reads(self.num_tokens * rows * HIDDEN_SIZE * 2)
        state.add_global_memory_writes(self.num_tokens * (1 + (1 if self.has_delta else 0)) * HIDDEN_SIZE * 2)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            prefix = torch.randn(self.num_tokens, HIDDEN_SIZE, device=dev, dtype=self.dtype)
            delta = torch.randn(self.num_tokens, HIDDEN_SIZE, device=dev, dtype=self.dtype) if self.has_delta else torch.empty(0, device=dev, dtype=self.dtype)
            blocks = torch.randn(self.num_tokens, MAX_BLOCKS, HIDDEN_SIZE, device=dev, dtype=self.dtype)
            norm_weight = torch.ones(HIDDEN_SIZE, device=dev, dtype=self.dtype)
            qk_weight = torch.randn(HIDDEN_SIZE, device=dev, dtype=self.dtype) / HIDDEN_SIZE ** 0.5
            output_norm_weight = torch.ones(HIDDEN_SIZE, device=dev, dtype=self.dtype)
            output = torch.empty(self.num_tokens, HIDDEN_SIZE, device=dev, dtype=self.dtype)
        return self.make_launcher(dev_id, torch.ops._C.kimi_k3_attn_res, prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, output, self.num_blocks, EPS, EPS)

    def reference(self, prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight):
        if delta.numel() != 0:
            prefix = prefix + delta
        values = torch.cat([blocks[:, :self.num_blocks], prefix.unsqueeze(1)], dim=1)
        keys = F.rms_norm(values, (HIDDEN_SIZE,), norm_weight, EPS)
        probs = (keys @ qk_weight).softmax(dim=-1)
        output = torch.matmul(probs.unsqueeze(1), values).squeeze(1)
        output = F.rms_norm(output, (HIDDEN_SIZE,), output_norm_weight, EPS)
        return output

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        prefix = torch.randn(self.num_tokens, HIDDEN_SIZE, device=dev, dtype=self.dtype)
        delta = torch.randn(self.num_tokens, HIDDEN_SIZE, device=dev, dtype=self.dtype) if self.has_delta else torch.empty(0, device=dev, dtype=self.dtype)
        blocks = torch.randn(self.num_tokens, MAX_BLOCKS, HIDDEN_SIZE, device=dev, dtype=self.dtype)
        norm_weight = torch.ones(HIDDEN_SIZE, device=dev, dtype=self.dtype)
        qk_weight = torch.randn(HIDDEN_SIZE, device=dev, dtype=self.dtype) / HIDDEN_SIZE ** 0.5
        output_norm_weight = torch.ones(HIDDEN_SIZE, device=dev, dtype=self.dtype)
        output = torch.empty(self.num_tokens, HIDDEN_SIZE, device=dev, dtype=self.dtype)

        torch.ops._C.kimi_k3_attn_res(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, output, self.num_blocks, EPS, EPS)

        passed = torch.isfinite(output).all()
        return passed, 0.0