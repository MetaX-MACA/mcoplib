
import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass


class Moe_fused_gate_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 8192)
        self.num_experts = config.get("num_experts", 160)
        self.num_expert_group = config.get("num_expert_group", 8)
        self.topk_group = config.get("topk_group", 2)
        self.topk = config.get("topk", 6)
        self.num_fused_shared_experts = config.get("num_fused_shared_experts", 0)
        self.routed_scaling_factor = config.get("routed_scaling_factor", 1.0)
        self.apply_routed_scaling_factor_on_output = config.get("apply_routed_scaling_factor_on_output", False)
        self.seed = config.get("seed", 42)
        self.dtype = torch.float32

    def _generate_gating_output(self, dev):
        generator = torch.Generator(device=dev)
        generator.manual_seed(self.seed)
        return torch.randn((self.num_tokens, self.num_experts), dtype=self.dtype, device=dev, generator=generator)

    def _generate_inputs(self, dev):
        gating_output = self._generate_gating_output(dev)
        correction_bias = torch.zeros((self.num_experts,), dtype=self.dtype, device=dev)
        return gating_output, correction_bias

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"({self.num_tokens} {self.num_experts})")
        total_elements = self.num_tokens * self.num_experts
        state.add_element_count(total_elements)
        element_size = 4
        read_bytes = (self.num_tokens * self.num_experts * element_size) + (self.num_experts * element_size)
        out_dim = self.topk + self.num_fused_shared_experts
        write_bytes = (self.num_tokens * out_dim * element_size) * 2
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = f"cuda:{dev_id}"
        with torch.cuda.stream(tc_s):
            gating_output, correction_bias = self._generate_inputs(dev)

        return self.make_launcher(
            dev_id,
            torch.ops.sgl_kernel.moe_fused_gate,
            gating_output,
            correction_bias,
            self.num_expert_group,
            self.topk_group,
            self.topk,
            self.num_fused_shared_experts,
            self.routed_scaling_factor,
            self.apply_routed_scaling_factor_on_output
        )

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        gating_output, correction_bias = self._generate_inputs(dev)

        op_weights, op_ids = torch.ops.sgl_kernel.moe_fused_gate(
            gating_output,
            correction_bias,
            self.num_expert_group,
            self.topk_group,
            self.topk,
            self.num_fused_shared_experts,
            self.routed_scaling_factor,
            self.apply_routed_scaling_factor_on_output
        )

        scores = torch.sigmoid(gating_output)
        experts_per_group = self.num_experts // self.num_expert_group
        group_scores_raw = scores.view(self.num_tokens, self.num_expert_group, experts_per_group)
        top2_in_group = torch.topk(group_scores_raw, 2, dim=-1)[0]
        group_scores = top2_in_group.sum(dim=-1)
        _, topk_group_idx = torch.topk(group_scores, self.topk_group, dim=-1)
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool, device=dev)
        group_mask.scatter_(1, topk_group_idx, True)
        mask = group_mask.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape(self.num_tokens, self.num_experts)
        masked_scores = scores.masked_fill(~mask, -float("inf"))
        ref_weights, ref_ids = torch.topk(masked_scores, self.topk, dim=-1)
        ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)

        if self.apply_routed_scaling_factor_on_output:
            ref_weights = ref_weights * self.routed_scaling_factor

        op_ids_sorted, op_sort_idx = torch.sort(op_ids, dim=-1)
        op_weights_sorted = torch.gather(op_weights, 1, op_sort_idx)
        ref_ids_sorted, ref_sort_idx = torch.sort(ref_ids, dim=-1)
        ref_weights_sorted = torch.gather(ref_weights, 1, ref_sort_idx)

        pass_weights, diff_weights = self.check_diff(op_weights_sorted, ref_weights_sorted)
        pass_ids, diff_ids = self.check_diff(op_ids_sorted.float(), ref_ids_sorted.float())
        is_passed = pass_weights and pass_ids
        max_diff = max(diff_weights, diff_ids)

        return is_passed, max_diff
