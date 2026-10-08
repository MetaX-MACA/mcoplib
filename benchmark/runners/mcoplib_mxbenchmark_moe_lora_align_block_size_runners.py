import torch
import math
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._moe_C
except ImportError:
    pass

class Moe_lora_align_block_size_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 16384)
        self.num_experts = config.get("num_experts", 128)
        self.block_size = config.get("block_size", 32)
        self.max_loras = config.get("max_loras", 4)
        self.top_k = config.get("top_k", 1)
        self.index_dtype = torch.int32

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "int32")
        shape_str = f"({self.num_tokens} {self.num_experts} {self.max_loras})"
        state.add_summary("Shape", shape_str)
        total_elements = self.num_tokens * self.top_k
        state.add_element_count(total_elements)
        capacity_per_lora = (total_elements + self.block_size - 1) // self.block_size * self.block_size
        max_blocks_per_lora = capacity_per_lora // self.block_size
        read_bytes = (total_elements * 4) + (self.num_tokens * 4) + (self.max_loras * 4 * 2)
        write_bytes = (self.max_loras * capacity_per_lora * 4) + \
                      (self.max_loras * max_blocks_per_lora * 4) + \
                      (self.max_loras * 4)
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            topk_ids = torch.randint(0, self.num_experts, (self.num_tokens, self.top_k), dtype=self.index_dtype, device=dev)
            token_lora_mapping = torch.randint(0, self.max_loras, (self.num_tokens,), dtype=self.index_dtype, device=dev)
            adapter_enabled = torch.ones(self.max_loras, dtype=self.index_dtype, device=dev)
            lora_ids = torch.arange(self.max_loras, dtype=self.index_dtype, device=dev)
            total_elements = self.num_tokens * self.top_k
            max_num_tokens_padded = (total_elements + self.block_size - 1) // self.block_size * self.block_size
            max_num_m_blocks = max_num_tokens_padded // self.block_size
            sorted_token_ids = torch.empty(self.max_loras * max_num_tokens_padded, dtype=self.index_dtype, device=dev)
            expert_ids = torch.empty(self.max_loras * max_num_m_blocks, dtype=self.index_dtype, device=dev)
            num_tokens_post_pad = torch.empty(self.max_loras, dtype=self.index_dtype, device=dev)
        return self.make_launcher(
            dev_id, torch.ops._moe_C.moe_lora_align_block_size,
            topk_ids, token_lora_mapping,
            self.num_experts, self.block_size, self.max_loras,
            max_num_tokens_padded, max_num_m_blocks,
            sorted_token_ids, expert_ids, num_tokens_post_pad,
            adapter_enabled, lora_ids, None
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        torch.manual_seed(42)
        topk_ids = torch.randint(0, self.num_experts, (self.num_tokens, self.top_k), dtype=self.index_dtype, device=dev)
        token_lora_mapping = torch.randint(0, self.max_loras, (self.num_tokens,), dtype=self.index_dtype, device=dev)
        adapter_enabled = torch.ones(self.max_loras, dtype=self.index_dtype, device=dev)
        lora_ids = torch.arange(self.max_loras, dtype=self.index_dtype, device=dev)
        total_elements = self.num_tokens * self.top_k
        max_num_tokens_padded = (total_elements + self.block_size - 1) // self.block_size * self.block_size
        max_num_m_blocks = max_num_tokens_padded // self.block_size
        sorted_token_ids = torch.full((self.max_loras * max_num_tokens_padded,), total_elements, dtype=self.index_dtype, device=dev)
        expert_ids = torch.full((self.max_loras * max_num_m_blocks,), -1, dtype=self.index_dtype, device=dev)
        num_tokens_post_pad = torch.zeros(self.max_loras, dtype=self.index_dtype, device=dev)
        torch.ops._moe_C.moe_lora_align_block_size(
            topk_ids, token_lora_mapping,
            self.num_experts, self.block_size, self.max_loras,
            max_num_tokens_padded, max_num_m_blocks,
            sorted_token_ids, expert_ids, num_tokens_post_pad,
            adapter_enabled, lora_ids, None
        )
        topk_flat = topk_ids.cpu().flatten()
        mapping_cpu = token_lora_mapping.cpu()
        mapping_expanded = mapping_cpu.repeat_interleave(self.top_k)
        ref_sorted = torch.full((self.max_loras * max_num_tokens_padded,), total_elements, dtype=self.index_dtype)
        ref_experts = torch.full((self.max_loras * max_num_m_blocks,), -1, dtype=self.index_dtype)
        ref_post_pad = torch.zeros(self.max_loras, dtype=self.index_dtype)
        bs = self.block_size
        for lora_idx in range(self.max_loras):
            lid = lora_ids[lora_idx].item()
            if adapter_enabled[lid] == 0: continue
            mask = (mapping_expanded == lid)
            relevant_indices = mask.nonzero(as_tuple=True)[0]
            if len(relevant_indices) == 0: continue
            sub_topk = topk_flat[relevant_indices]
            counts = torch.bincount(sub_topk, minlength=self.num_experts)
            curr_sorted_offset = lid * max_num_tokens_padded
            curr_expert_offset = lid * max_num_m_blocks
            lora_total_padded = 0
            for eid in range(self.num_experts):
                cnt = counts[eid].item()
                padded_cnt = math.ceil(cnt / bs) * bs
                num_blks = padded_cnt // bs
                if num_blks > 0:
                    start_blk = lora_total_padded // bs
                    ref_experts[curr_expert_offset + start_blk : curr_expert_offset + start_blk + num_blks] = eid
                if cnt > 0:
                    e_mask = (sub_topk == eid)
                    original_indices = relevant_indices[e_mask]
                    ref_sorted[curr_sorted_offset + lora_total_padded : curr_sorted_offset + lora_total_padded + cnt] = original_indices
                lora_total_padded += padded_cnt
            ref_post_pad[lid] = lora_total_padded
        passed_pad, diff_pad = self.check_diff(num_tokens_post_pad, ref_post_pad.to(dev))
        if not passed_pad:
            print(f"Post Pad mismatch: GPU={num_tokens_post_pad}, Ref={ref_post_pad}")
            return False, diff_pad
        passed_exp, diff_exp = self.check_diff(expert_ids, ref_experts.to(dev))
        if not passed_exp:
            print("Expert IDs mismatch")
            return False, diff_exp
        gpu_sorted = sorted_token_ids.cpu()
        for lora_idx in range(self.max_loras):
            lid = lora_ids[lora_idx].item()
            if adapter_enabled[lid] == 0: continue
            total_len = ref_post_pad[lid].item()
            if total_len == 0: continue
            base_offset = lid * max_num_tokens_padded
            mask = (mapping_expanded == lid)
            sub_topk = topk_flat[mask]
            counts = torch.bincount(sub_topk, minlength=self.num_experts)
            current_local_offset = 0
            for eid in range(self.num_experts):
                cnt = counts[eid].item()
                padded_cnt = math.ceil(cnt / bs) * bs
                if padded_cnt > 0:
                    start = base_offset + current_local_offset
                    end = start + padded_cnt
                    segment = gpu_sorted[start:end]
                    sorted_seg, _ = torch.sort(segment)
                    gpu_sorted[start:end] = sorted_seg
                current_local_offset += padded_cnt
        passed_sort, diff_sort = self.check_diff(gpu_sorted, ref_sorted)
        if not passed_sort:
            print("Sorted Token IDs mismatch")
        return passed_sort, diff_sort