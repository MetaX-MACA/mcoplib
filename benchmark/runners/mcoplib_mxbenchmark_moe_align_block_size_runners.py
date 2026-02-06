import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._moe_C
except ImportError:
    pass

class Moe_align_block_size_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 2048)
        self.top_k = config.get("top_k", 2)
        self.num_experts = config.get("num_experts", 8)
        self.block_size = config.get("block_size", 64)
        self.dtype = torch.int32 

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "int32")
        shape_str = f"({self.num_tokens} {self.num_experts} {self.block_size})"
        state.add_summary("Shape", shape_str)
        total_ids = self.num_tokens * self.top_k
        state.add_element_count(total_ids)
        max_padded = total_ids + (self.num_experts * self.block_size)
        bytes_read = total_ids * 4
        bytes_write = (max_padded * 4) + (max_padded // self.block_size * 4) + 4
        state.add_global_memory_reads(bytes_read)
        state.add_global_memory_writes(bytes_write)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            topk_ids = torch.randint(0, self.num_experts, (self.num_tokens, self.top_k), 
                                     dtype=self.dtype, device=dev)
            total_elements = topk_ids.numel()
            max_num_tokens_padded = total_elements + self.num_experts * self.block_size
            sorted_token_ids = torch.full((max_num_tokens_padded,), total_elements, dtype=torch.int32, device=dev)
            experts_ids = torch.full((max_num_tokens_padded // self.block_size,), -1, dtype=torch.int32, device=dev)
            num_tokens_post_pad = torch.zeros(1, dtype=torch.int32, device=dev)
        return self.make_launcher(dev_id, torch.ops._moe_C.moe_align_block_size, 
                                 topk_ids, self.num_experts, self.block_size, 
                                 sorted_token_ids, experts_ids, num_tokens_post_pad, 
                                 None)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        topk_ids = torch.randint(0, self.num_experts, (self.num_tokens, self.top_k), 
                                 dtype=self.dtype, device=dev)
        total_elements = topk_ids.numel()
        max_num_tokens_padded = total_elements + self.num_experts * self.block_size
        sorted_token_ids = torch.full((max_num_tokens_padded,), total_elements, dtype=torch.int32, device=dev)
        experts_ids = torch.full((max_num_tokens_padded // self.block_size,), -1, dtype=torch.int32, device=dev)
        num_tokens_post_pad = torch.zeros(1, dtype=torch.int32, device=dev)
        torch.ops._moe_C.moe_align_block_size(topk_ids, self.num_experts, self.block_size, 
                                sorted_token_ids, experts_ids, num_tokens_post_pad, 
                                None)
        topk_cpu = topk_ids.cpu().flatten()
        num_el = topk_cpu.numel()
        expert_counts = torch.bincount(topk_cpu, minlength=self.num_experts)
        padded_counts = ((expert_counts + self.block_size - 1) // self.block_size) * self.block_size
        total_padded = padded_counts.sum().item()
        ref_sorted_token_ids = torch.full((max_num_tokens_padded,), num_el, dtype=torch.int32)
        ref_experts_ids = torch.full((max_num_tokens_padded // self.block_size,), -1, dtype=torch.int32)
        current_offset = 0
        for e_id in range(self.num_experts):
            indices = (topk_cpu == e_id).nonzero(as_tuple=True)[0]
            count = len(indices)
            if count > 0:
                ref_sorted_token_ids[current_offset : current_offset + count] = indices
            num_blocks = padded_counts[e_id].item() // self.block_size
            block_start = current_offset // self.block_size
            if num_blocks > 0:
                ref_experts_ids[block_start : block_start + num_blocks] = e_id
            current_offset += padded_counts[e_id].item()
        ref_num_tokens_post_pad = torch.tensor([total_padded], dtype=torch.int32)
        gpu_sorted_ids_cpu = sorted_token_ids.cpu()
        current_offset = 0
        for e_id in range(self.num_experts):
            length = padded_counts[e_id].item()
            if length > 0:
                segment = gpu_sorted_ids_cpu[current_offset : current_offset + length]
                sorted_segment, _ = torch.sort(segment)
                gpu_sorted_ids_cpu[current_offset : current_offset + length] = sorted_segment
            current_offset += length
        actual_res = torch.cat([
            gpu_sorted_ids_cpu.float(),
            experts_ids.cpu().float(),
            num_tokens_post_pad.cpu().float()
        ])
        expected_res = torch.cat([
            ref_sorted_token_ids.float(),
            ref_experts_ids.cpu().float(),
            ref_num_tokens_post_pad.float()
        ])
        return self.check_diff(actual_res, expected_res)