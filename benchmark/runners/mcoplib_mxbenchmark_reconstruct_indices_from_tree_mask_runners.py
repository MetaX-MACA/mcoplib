import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Reconstruct_indices_from_tree_mask_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.bs = config.get("batch_size", 8192)
        self.draft_token_num = config.get("draft_token_num", 16)
        
    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", "bool"))
        state.add_summary("Shape", f"bs:{self.bs}_draft:{self.draft_token_num}")
        
        total_elements = self.bs * self.draft_token_num
        state.add_element_count(total_elements)
        
        bool_bytes = 1
        int64_bytes = 8
        
        read_bytes = (
            (self.bs * self.draft_token_num * self.draft_token_num * bool_bytes) + 
            (self.bs * int64_bytes)
        )
        write_bytes = (4 * self.bs * self.draft_token_num * int64_bytes)
        
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def _generate_inputs(self, dev):
        tree_mask = torch.randint(0, 2, (self.bs, self.draft_token_num, self.draft_token_num), dtype=torch.bool, device=dev)
        verified_seq_len = torch.randint(1, 100, (self.bs,), dtype=torch.int64, device=dev)
        
        positions = torch.zeros((self.bs, self.draft_token_num), dtype=torch.int64, device=dev)
        retrive_index = torch.zeros((self.bs, self.draft_token_num), dtype=torch.int64, device=dev)
        retrive_next_token = torch.zeros((self.bs, self.draft_token_num), dtype=torch.int64, device=dev)
        retrive_next_sibling = torch.zeros((self.bs, self.draft_token_num), dtype=torch.int64, device=dev)
        
        return (tree_mask, verified_seq_len, positions, retrive_index, 
                retrive_next_token, retrive_next_sibling, self.bs, self.draft_token_num)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            inputs = self._generate_inputs(dev)
            
        return self.make_launcher(
            dev_id, 
            torch.ops.sgl_kernel.reconstruct_indices_from_tree_mask, 
            *inputs
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        inputs = self._generate_inputs(dev)
        tree_mask = inputs[0]
        verified_seq_len = inputs[1]
        positions = inputs[2]
        retrive_index = inputs[3]
        retrive_next_token = inputs[4]
        retrive_next_sibling = inputs[5]
        torch.ops.sgl_kernel.reconstruct_indices_from_tree_mask(*inputs)
        tm_cpu = tree_mask.cpu().numpy()
        vsl_cpu = verified_seq_len.cpu().numpy()
        exp_positions = torch.zeros_like(positions.cpu())
        exp_retrive_index = torch.zeros_like(retrive_index.cpu())
        exp_retrive_next_token = torch.full_like(retrive_next_token.cpu(), -1)
        exp_retrive_next_sibling = torch.full_like(retrive_next_sibling.cpu(), -1)
        for bid in range(self.bs):
            for tid in range(self.draft_token_num):
                token_idx = bid * self.draft_token_num + tid
                depth = 0
                parent_idx = -1
                
                for i in range(tid - 1, -1, -1):
                    if tm_cpu[bid, tid, i]:
                        depth += 1
                        if parent_idx == -1:
                            parent_idx = i
                            
                exp_retrive_index[bid, tid] = token_idx
                exp_positions[bid, tid] = depth + vsl_cpu[bid]
                
                next_token_idx = -1
                for i in range(tid + 1, self.draft_token_num):
                    if tm_cpu[bid, i, tid]:
                        next_token_idx = i
                        break
                exp_retrive_next_token[bid, tid] = next_token_idx
                
                next_sibling_idx = -1
                if parent_idx != -1:
                    for i in range(tid + 1, self.draft_token_num):
                        if tm_cpu[bid, i, parent_idx]:
                            is_sibling = True
                            for j in range(parent_idx + 1, i):
                                if tm_cpu[bid, i, j]:
                                    is_sibling = False
                                    break
                            if is_sibling:
                                next_sibling_idx = i
                                break
                exp_retrive_next_sibling[bid, tid] = next_sibling_idx
        pass_pos, diff_pos = self.check_diff(positions, exp_positions.to(dev))
        pass_ri, diff_ri = self.check_diff(retrive_index, exp_retrive_index.to(dev))
        pass_rnt, diff_rnt = self.check_diff(retrive_next_token, exp_retrive_next_token.to(dev))
        pass_rns, diff_rns = self.check_diff(retrive_next_sibling, exp_retrive_next_sibling.to(dev))
        
        is_passed = pass_pos and pass_ri and pass_rnt and pass_rns
        max_diff = max(diff_pos, diff_ri, diff_rnt, diff_rns)
        
        return is_passed, max_diff