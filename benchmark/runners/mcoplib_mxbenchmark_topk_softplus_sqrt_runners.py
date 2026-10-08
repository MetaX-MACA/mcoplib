import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._moe_C
except ImportError:
    pass


TEST_SCENARIOS = [
    (0, 160, 1, 1, 8),
    (0, 256, 8, 4, 8),
    (0, 256, 1, 1, 8),
    (0, 320, 1, 1, 8),
    (0, 384, 1, 1, 8),
    (0, 448, 1, 1, 8),
    (0, 288, 1, 1, 8),
    (0, 896, 1, 1, 16),
    (1, 160, 1, 1, 9),
    (1, 256, 1, 1, 9),
    (1, 256, 8, 4, 9),
    (1, 320, 1, 1, 9),
    (1, 384, 1, 1, 9),
    (1, 448, 1, 1, 9),
    (2, 256, 8, 4, 9),
    (2, 320, 1, 1, 9),
    (2, 384, 1, 1, 9),
    (2, 448, 1, 1, 9),
]

SUPPORTED_EXPERTS = {
    1, 2, 4, 8, 16, 32, 64, 128,
    192, 256, 320, 384, 448, 512, 576
}


def topk_softplus_sqrt_reference(gating_output, topk, renormalize, routed_scaling_factor, correction_bias=None):
    scores = torch.sqrt(F.softplus(gating_output.float()))

    if correction_bias is not None:
        scores_for_choice = scores + correction_bias.float().unsqueeze(0)
    else:
        scores_for_choice = scores

    _, topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=True)
    topk_weights = torch.gather(scores, dim=-1, index=topk_indices)

    if renormalize:
        row_sum = topk_weights.sum(dim=-1, keepdim=True)
        row_sum = torch.where(row_sum > 0.0, row_sum, torch.ones_like(row_sum))
        topk_weights = topk_weights / row_sum

    topk_weights = topk_weights * float(routed_scaling_factor)

    return topk_weights.float(), topk_indices.int()


class Topk_softplus_sqrt_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.scenario_id = config.get("scenario_id", 5)
        self.num_tokens = config.get("num_tokens", 4096)
        self.renormalize = config.get("renormalize", True)
        self.routed_scaling_factor = config.get("routed_scaling_factor", 1.0)
        self.use_correction_bias = config.get("use_correction_bias", True)

        if self.scenario_id < 0 or self.scenario_id >= len(TEST_SCENARIOS):
            raise ValueError(f"Invalid scenario_id={self.scenario_id}, valid range is 0~{len(TEST_SCENARIOS) - 1}")

        self.shared_experts, self.num_experts, self.num_expert_group, self.group_topk, self.top_k = TEST_SCENARIOS[self.scenario_id]

        if self.num_experts not in SUPPORTED_EXPERTS:
            raise ValueError(f"Unsupported expert number: {self.num_experts}, supported: {sorted(SUPPORTED_EXPERTS)}")

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.num_tokens} {self.num_experts} {self.top_k})")
        state.add_summary("scenario_id", str(self.scenario_id))
        state.add_summary("shared_experts", str(self.shared_experts))
        state.add_summary("num_expert_group", str(self.num_expert_group))
        state.add_summary("group_topk", str(self.group_topk))
        state.add_summary("top_k", str(self.top_k))
        state.add_summary("renormalize", str(self.renormalize))
        state.add_summary("routed_scaling_factor", str(self.routed_scaling_factor))
        state.add_summary("use_correction_bias", str(self.use_correction_bias))

        element_size = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        gating_bytes = self.num_tokens * self.num_experts * element_size
        bias_bytes = self.num_experts * 4 if self.use_correction_bias else 0
        output_bytes = self.num_tokens * self.top_k * (4 + 4 + 4)

        state.add_element_count(self.num_tokens * self.num_experts)
        state.add_global_memory_reads(gating_bytes + bias_bytes)
        state.add_global_memory_writes(output_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            gating_output = torch.randn(self.num_tokens, self.num_experts, dtype=self.dtype, device=dev)
            topk_weights = torch.empty(self.num_tokens, self.top_k, dtype=torch.float32, device=dev)
            topk_indices = torch.empty(self.num_tokens, self.top_k, dtype=torch.int32, device=dev)
            token_expert_indices = torch.empty(self.num_tokens, self.top_k, dtype=torch.int32, device=dev)
            correction_bias = torch.randn(self.num_experts, dtype=torch.float32, device=dev) if self.use_correction_bias else None

        return self.make_launcher(dev_id, torch.ops._moe_C.topk_softplus_sqrt, topk_weights, topk_indices, token_expert_indices, gating_output, self.renormalize, self.routed_scaling_factor, correction_bias, None, None, None)

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        gating_output = torch.randn(self.num_tokens, self.num_experts, dtype=self.dtype, device=dev)
        topk_weights = torch.empty(self.num_tokens, self.top_k, dtype=torch.float32, device=dev)
        topk_indices = torch.empty(self.num_tokens, self.top_k, dtype=torch.int32, device=dev)
        token_expert_indices = torch.empty(self.num_tokens, self.top_k, dtype=torch.int32, device=dev)
        correction_bias = torch.randn(self.num_experts, dtype=torch.float32, device=dev) if self.use_correction_bias else None

        torch.ops._moe_C.topk_softplus_sqrt(topk_weights, topk_indices, token_expert_indices, gating_output, self.renormalize, self.routed_scaling_factor, correction_bias, None, None, None)

        ref_weights, ref_indices = topk_softplus_sqrt_reference(gating_output, self.top_k, self.renormalize, self.routed_scaling_factor, correction_bias)

        weights_match, diff = self.check_diff(topk_weights, ref_weights, threshold=0.9999)
        indices_match = torch.equal(torch.sort(topk_indices, dim=-1).values, torch.sort(ref_indices, dim=-1).values)

        passed = weights_match and indices_match
        return passed, diff if not passed else 0.0