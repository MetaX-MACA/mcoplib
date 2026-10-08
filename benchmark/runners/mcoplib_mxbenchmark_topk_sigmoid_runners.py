import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._moe_C
except ImportError:
    pass


def _ref_topk_sigmoid(gating_output, bias, topk, renormalize, routed_scaling_factor=1.0):
    gating_fp32 = gating_output.float()
    bias_fp32 = bias.float() if bias is not None else None
    sigmoid_scores = torch.sigmoid(gating_fp32)
    if bias_fp32 is not None:
        routing_scores = sigmoid_scores + bias_fp32
    else:
        routing_scores = sigmoid_scores
    _, topk_indices = torch.topk(routing_scores, k=topk, dim=-1)
    topk_weights = torch.gather(sigmoid_scores, dim=-1, index=topk_indices)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    if renormalize:
        row_sum = topk_weights.sum(dim=-1, keepdim=True)
        row_sum = torch.where(row_sum > 0.0, row_sum, torch.tensor(1.0, device=row_sum.device))
        topk_weights = topk_weights / row_sum
    return topk_weights, topk_indices.to(torch.int32)


class Topk_sigmoid_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 4096)
        self.num_experts = config.get("num_experts", 288)
        self.top_k = config.get("top_k", 8)
        self.renormalize = config.get("renormalize", True)
        self.routed_scaling_factor = config.get("routed_scaling_factor", 1.0)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.num_tokens} {self.num_experts} {self.top_k})")
        in_elems = self.num_tokens * self.num_experts
        out_elems = self.num_tokens * self.top_k
        state.add_element_count(in_elems + out_elems * 3)
        es = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        reads = in_elems * es + self.num_experts * 4
        writes = out_elems * 4 * 3
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            gating = torch.randn(self.num_tokens, self.num_experts, dtype=self.dtype, device=dev)
            bias = torch.randn(self.num_experts, dtype=torch.float32, device=dev)
            w = torch.empty(self.num_tokens, self.top_k, dtype=torch.float32, device=dev)
            idx = torch.empty(self.num_tokens, self.top_k, dtype=torch.int32, device=dev)
            tei = torch.empty(self.num_tokens, self.top_k, dtype=torch.int32, device=dev)
        return self.make_launcher(dev_id, torch.ops._moe_C.topk_sigmoid, w, idx, tei, gating, self.renormalize, bias, self.routed_scaling_factor, None)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        T, E = 256, 288
        gating = torch.randn(T, E, dtype=self.dtype, device=dev)
        bias = torch.randn(E, dtype=torch.float32, device=dev)
        w = torch.empty(T, self.top_k, dtype=torch.float32, device=dev)
        idx = torch.empty(T, self.top_k, dtype=torch.int32, device=dev)
        tei = torch.empty(T, self.top_k, dtype=torch.int32, device=dev)
        torch.ops._moe_C.topk_sigmoid(w, idx, tei, gating, self.renormalize, bias, self.routed_scaling_factor, None)
        ref_w, ref_i = _ref_topk_sigmoid(gating, bias, self.top_k, self.renormalize, self.routed_scaling_factor)
        w_match = self.check_diff(w, ref_w, threshold=0.9999)
        i_match = torch.sort(idx, dim=-1).values.equal(torch.sort(ref_i, dim=-1).values)
        passed = w_match[0] and bool(i_match)
        return passed, 0.0 if passed else 1.0