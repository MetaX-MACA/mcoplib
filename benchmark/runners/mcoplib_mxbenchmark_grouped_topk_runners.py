import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._moe_C
except ImportError:
    pass


def _ref_grouped_topk(scores, n_group, topk_group, topk, renormalize,
                      routed_scaling_factor, bias, scoring_func):
    scores_fp32 = scores.float()
    bias_fp32 = bias.float() if bias is not None else None
    if scoring_func == 0:
        act_scores = torch.softmax(scores_fp32, dim=-1)
    elif scoring_func == 1:
        act_scores = torch.sigmoid(scores_fp32)
    else:
        raise ValueError("Unsupported scoring_func")
    num_token = act_scores.size(0)
    num_experts = act_scores.size(-1)
    experts_per_group = num_experts // n_group
    original_scores = act_scores
    if bias_fp32 is not None:
        act_scores = act_scores + bias_fp32.unsqueeze(0)
        group_scores = act_scores.view(num_token, n_group, experts_per_group).topk(2, dim=-1)[0].sum(dim=-1)
    else:
        group_scores = act_scores.view(num_token, n_group, experts_per_group).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=True)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (group_mask.unsqueeze(-1).expand(num_token, n_group, experts_per_group)
                  .reshape(num_token, -1))
    tmp_scores = act_scores.masked_fill(~score_mask.bool(), float("-inf"))
    topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=True)[1]
    topk_weights = original_scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


class Grouped_topk_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 64)
        self.num_experts = config.get("num_experts", 256)
        self.n_group = config.get("n_group", 8)
        self.topk_group = config.get("topk_group", 4)
        self.seed = config.get("seed", None)
        self.topk = config.get("topk", 8)
        self.renormalize = config.get("renormalize", True)
        self.routed_scaling_factor = config.get("routed_scaling_factor", 2.0)
        self.scoring_func = config.get("scoring_func", 1)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        shape_str = f"({self.num_tokens} {self.num_experts} {self.topk})"
        state.add_summary("Shape", shape_str)

        in_elems = self.num_tokens * self.num_experts
        out_elems = self.num_tokens * self.topk
        state.add_element_count(in_elems + out_elems * 2)

        es = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        reads = in_elems * es + self.num_experts * 4
        writes = out_elems * 4 + out_elems * 4
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            scores = torch.randn(self.num_tokens, self.num_experts, dtype=self.dtype, device=dev)
            bias = torch.randn(self.num_experts, dtype=torch.float32, device=dev)
        return self.make_launcher(
            dev_id, torch.ops._moe_C.grouped_topk,
            scores, self.n_group, self.topk_group, self.topk,
            self.renormalize, self.routed_scaling_factor, bias, self.scoring_func
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        T, E = 64, 256
        scores = torch.randn(T, E, dtype=self.dtype, device=dev)
        bias = torch.randn(E, dtype=torch.float32, device=dev)
        op_w, op_i = torch.ops._moe_C.grouped_topk(
            scores, self.n_group, self.topk_group, self.topk,
            self.renormalize, self.routed_scaling_factor, bias, self.scoring_func
        )
        ref_w, ref_i = _ref_grouped_topk(
            scores, self.n_group, self.topk_group, self.topk,
            self.renormalize, self.routed_scaling_factor, bias, self.scoring_func
        )
        w_match = self.check_diff(op_w, ref_w, threshold=0.9999)
        i_match = torch.sort(op_i, dim=-1).values.equal(torch.sort(ref_i, dim=-1).values)
        passed = w_match[0] and bool(i_match)
        return passed, 0.0 if passed else 1.0
