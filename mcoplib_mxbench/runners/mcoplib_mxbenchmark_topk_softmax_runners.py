import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._moe_C
except ImportError:
    pass

class Topk_softmax_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 131072)
        self.num_experts = config.get("num_experts", 64)
        self.top_k = config.get("top_k", 2)
        # 建议从 config 读取是否归一化，默认 True
        self.renormalize = config.get("renormalize", True) 

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.batch_size} {self.num_experts} {self.top_k})")
        
        state.add_element_count(self.batch_size * self.num_experts)
        
        element_size = 2 if self.dtype == torch.float16 else 4
        reads = self.batch_size * self.num_experts * element_size
        writes = (self.batch_size * self.top_k) * (4 + 4 + 4)
        
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            gating_output = torch.randn(self.batch_size, self.num_experts, dtype=self.dtype, device=dev)
            
            topk_weights = torch.empty(self.batch_size, self.top_k, dtype=torch.float32, device=dev)
            topk_indices = torch.empty(self.batch_size, self.top_k, dtype=torch.int32, device=dev)
            token_expert_indices = torch.empty(self.batch_size, self.top_k, dtype=torch.int32, device=dev)
            
        return self.make_launcher(dev_id, torch.ops._moe_C.topk_softmax, 
                                  topk_weights, topk_indices, token_expert_indices, gating_output, self.renormalize)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        N, E, K = 32, 16, 2
        gating = torch.randn(N, E, dtype=self.dtype, device=dev)
        
        out_weights = torch.empty(N, K, dtype=torch.float32, device=dev)
        out_indices = torch.empty(N, K, dtype=torch.int32, device=dev)
        out_expert_indices = torch.empty(N, K, dtype=torch.int32, device=dev)
        
        torch.ops._moe_C.topk_softmax(out_weights, out_indices, out_expert_indices, gating, True)
        
        # Reference
        ref_vals, ref_idxs = torch.topk(gating.float(), K, dim=-1)
        ref_weights = torch.softmax(ref_vals, dim=-1) # 注意：如果 renormalize=False，这里参考实现也要改
        
        indices_match = (out_indices.long() == ref_idxs).all().item()
        weights_match, diff = self.check_diff(out_weights, ref_weights)
        
        return indices_match and weights_match, diff