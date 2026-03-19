import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Prepare_moe_input_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 8192)
        self.num_experts = config.get("num_experts", 256)
        self.topk = config.get("topk", 6)
        self.intermediate_size = config.get("intermediate_size", 2048)
        self.hidden_size = config.get("hidden_size", 4096)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        # 补充 dtype 输出
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.num_tokens}, {self.topk})")
        
        total_topk_elements = self.num_tokens * self.topk
        state.add_element_count(total_topk_elements)
        
        element_size = 4 
        read_bytes = total_topk_elements * element_size
        write_bytes = (
            (self.num_experts + 1) * element_size + 
            (self.num_experts * 3) * element_size * 2 + 
            total_topk_elements * element_size * 2 
        )
        
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            topk_ids = torch.randint(0, self.num_experts, (self.num_tokens, self.topk), dtype=torch.int32, device=dev)
            
            total_tokens = self.num_tokens * self.topk
            src_idx = torch.empty((total_tokens,), dtype=torch.int32, device=dev)
            dst_idx = torch.empty((total_tokens,), dtype=torch.int32, device=dev)
            expert_offsets = torch.empty((self.num_experts + 1,), dtype=torch.int32, device=dev)
            problem_sizes1 = torch.zeros((self.num_experts, 3), dtype=torch.int32, device=dev)
            problem_sizes2 = torch.zeros((self.num_experts, 3), dtype=torch.int32, device=dev)
            
        return self.make_launcher(
            dev_id, 
            torch.ops.sgl_kernel.prepare_moe_input, 
            topk_ids, 
            expert_offsets, 
            None, 
            problem_sizes1, 
            problem_sizes2, 
            src_idx, 
            dst_idx, 
            self.num_experts, 
            self.intermediate_size, 
            self.hidden_size
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        topk_ids = torch.randint(0, self.num_experts, (self.num_tokens, self.topk), dtype=torch.int32, device=dev)
        
        total_tokens = self.num_tokens * self.topk
        src_idx = torch.empty((total_tokens,), dtype=torch.int32, device=dev)
        dst_idx = torch.empty((total_tokens,), dtype=torch.int32, device=dev)
        expert_offsets = torch.empty((self.num_experts + 1,), dtype=torch.int32, device=dev)
        problem_sizes1 = torch.zeros((self.num_experts, 3), dtype=torch.int32, device=dev)
        problem_sizes2 = torch.zeros((self.num_experts, 3), dtype=torch.int32, device=dev)

        torch.ops.sgl_kernel.prepare_moe_input(
            topk_ids, 
            expert_offsets, 
            None, 
            problem_sizes1, 
            problem_sizes2, 
            src_idx, 
            dst_idx, 
            self.num_experts, 
            self.intermediate_size, 
            self.hidden_size
        )
        
        topk_ids_flat = topk_ids.flatten()
        counts = torch.bincount(topk_ids_flat, minlength=self.num_experts)
        
        ref_expert_offsets = torch.zeros(self.num_experts + 1, dtype=torch.int32, device=dev)
        ref_expert_offsets[1:] = torch.cumsum(counts, dim=0)
        
        ref_problem_sizes1 = torch.zeros((self.num_experts, 3), dtype=torch.int32, device=dev)
        ref_problem_sizes1[:, 0] = counts
        ref_problem_sizes1[:, 1] = 2 * self.intermediate_size
        ref_problem_sizes1[:, 2] = self.hidden_size
        
        pass_offsets, diff_offsets = self.check_diff(expert_offsets.float(), ref_expert_offsets.float())
        pass_sizes, diff_sizes = self.check_diff(problem_sizes1.float(), ref_problem_sizes1.float())
        
        is_passed = pass_offsets and pass_sizes
        max_diff = max(diff_offsets, diff_sizes)
        
        return is_passed, max_diff