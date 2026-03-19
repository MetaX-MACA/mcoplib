import torch
import numpy as np
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Build_tree_kernel_efficient_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.bs = config.get("batch_size", 1024)
        self.topk = config.get("topk", 4)
        self.depth = config.get("depth", 5)
        self.draft_token_num = config.get("draft_token_num", 64)
        self.tree_mask_mode = config.get("tree_mask_mode", 1) 
        
    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"bs:{self.bs}_draft:{self.draft_token_num}")
        
        total_elements = self.bs * self.draft_token_num
        state.add_element_count(total_elements)
        
        int64_bytes = 8
        bool_bytes = 1
        
        parent_list_len = self.topk * (self.depth - 1) + 1
        
        read_bytes = (
            (self.bs * parent_list_len * int64_bytes) + 
            (self.bs * (self.draft_token_num - 1) * int64_bytes) + 
            (self.bs * int64_bytes)
        )
        write_bytes = (
            (self.bs * self.draft_token_num * self.draft_token_num * bool_bytes) + 
            (4 * self.bs * self.draft_token_num * int64_bytes)
        )
        
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def check_diff(self, output_op, output_ref, threshold=0.999999):
        if output_op.dtype in [torch.int32, torch.int64, torch.bool]:
            exact_match = torch.equal(output_op, output_ref)
            if exact_match:
                return True, 0.0
            else:
                return False, 1.0
        return super().check_diff(output_op, output_ref, threshold)

    def _generate_inputs(self, dev):
        parent_list_len = self.topk * (self.depth - 1) + 1
        
        parent_list = torch.zeros((self.bs, parent_list_len), dtype=torch.int64, device=dev)
        selected_index = torch.zeros((self.bs, self.draft_token_num - 1), dtype=torch.int64, device=dev)
        
        selected_index.fill_(self.topk * 1) 
        selected_index[:, 0] = 0
        parent_list[:, 1] = 0
        
        verified_seq_len = torch.randint(10, 50, (self.bs,), dtype=torch.int64, device=dev)
        
        tree_mask = torch.zeros((self.bs, self.draft_token_num, self.draft_token_num), dtype=torch.bool, device=dev)
        positions = torch.zeros((self.bs, self.draft_token_num), dtype=torch.int64, device=dev)
        retrive_index = torch.zeros((self.bs, self.draft_token_num), dtype=torch.int64, device=dev)
        
        retrive_next_token = torch.full((self.bs, self.draft_token_num), -1, dtype=torch.int64, device=dev)
        retrive_next_sibling = torch.full((self.bs, self.draft_token_num), -1, dtype=torch.int64, device=dev)
        
        return (parent_list, selected_index, verified_seq_len, tree_mask, 
                positions, retrive_index, retrive_next_token, retrive_next_sibling, 
                self.topk, self.depth, self.draft_token_num, self.tree_mask_mode)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = f'cuda:{dev_id}'
        inputs = self._generate_inputs(dev)
        
        tree_mask = inputs[3]
        retrive_next_token = inputs[6]
        retrive_next_sibling = inputs[7]
        
        def custom_launcher(launch):
            stream = self.as_torch_stream(launch.get_stream(), dev_id)
            with torch.cuda.stream(stream):
                tree_mask.zero_()
                retrive_next_token.fill_(-1)
                retrive_next_sibling.fill_(-1)
                torch.ops.sgl_kernel.build_tree_kernel_efficient(*inputs)
        return custom_launcher

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        inputs = self._generate_inputs(dev)
        
        parent_list, selected_index, verified_seq_len, tree_mask = inputs[0], inputs[1], inputs[2], inputs[3]
        positions, retrive_index, retrive_next_token, retrive_next_sibling = inputs[4], inputs[5], inputs[6], inputs[7]
        
        pl_cpu = parent_list.cpu().numpy()
        si_cpu = selected_index.cpu().numpy()
        vsl_cpu = verified_seq_len.cpu().numpy()
        
        exp_tm = np.zeros(tree_mask.shape, dtype=bool)
        exp_pos = np.zeros(positions.shape, dtype=np.int64)
        exp_ri = np.zeros(retrive_index.shape, dtype=np.int64)
        exp_rnt = np.full(retrive_next_token.shape, -1, dtype=np.int64)
        exp_rns = np.full(retrive_next_sibling.shape, -1, dtype=np.int64)
        
        for bid in range(self.bs):
            seq_len = vsl_cpu[bid]
            exp_pos[bid, 0] = seq_len
            ri_offset = bid * self.draft_token_num
            
            for tid in range(self.draft_token_num):
                exp_tm[bid, tid, 0] = True
                for i in range(self.draft_token_num - 1):
                    exp_tm[bid, tid, 1 + i] = False
                    
            for i in range(self.draft_token_num - 1, 0, -1):
                curr_token_idx = ri_offset + i
                exp_ri[bid, i] = curr_token_idx
                parent_tb_idx = si_cpu[bid, i - 1] // self.topk
                parent_pos = 0
                
                if parent_tb_idx > 0:
                    parent_token_idx = pl_cpu[bid, parent_tb_idx]
                    found = False
                    for p in range(self.draft_token_num - 1):
                        if si_cpu[bid, p] == parent_token_idx:
                            parent_pos = p + 1
                            found = True
                            break
                    if not found:
                        parent_pos = self.draft_token_num
                        
                if parent_pos == self.draft_token_num:
                    continue 
                    
                if exp_rnt[bid, parent_pos] == -1:
                    exp_rnt[bid, parent_pos] = i
                else:
                    origin_nt = exp_rnt[bid, parent_pos]
                    exp_rnt[bid, parent_pos] = i
                    exp_rns[bid, i] = origin_nt
                    
            exp_ri[bid, 0] = ri_offset
            
            for tid in range(1, self.draft_token_num):
                position = 0
                cur_pos = tid - 1
                while True:
                    position += 1
                    exp_tm[bid, tid, cur_pos + 1] = True
                    parent_tb_idx = si_cpu[bid, cur_pos] // self.topk
                    if parent_tb_idx == 0:
                        break
                    
                    token_idx = pl_cpu[bid, parent_tb_idx]
                    
                    found = False
                    for cp in range(self.draft_token_num - 1):
                        if si_cpu[bid, cp] == token_idx:
                            cur_pos = cp
                            found = True
                            break
                    if not found:
                        break
                exp_pos[bid, tid] = position + seq_len

        torch.ops.sgl_kernel.build_tree_kernel_efficient(*inputs)
        
        exp_tm_t = torch.from_numpy(exp_tm).to(dev)
        exp_pos_t = torch.from_numpy(exp_pos).to(dev)
        exp_ri_t = torch.from_numpy(exp_ri).to(dev)
        exp_rnt_t = torch.from_numpy(exp_rnt).to(dev)
        exp_rns_t = torch.from_numpy(exp_rns).to(dev)
        
        p_tm, d_tm = self.check_diff(tree_mask, exp_tm_t)
        p_pos, d_pos = self.check_diff(positions, exp_pos_t)
        p_ri, d_ri = self.check_diff(retrive_index, exp_ri_t)
        p_rnt, d_rnt = self.check_diff(retrive_next_token, exp_rnt_t)
        p_rns, d_rns = self.check_diff(retrive_next_sibling, exp_rns_t)
        
        is_passed = p_tm and p_pos and p_ri and p_rnt and p_rns
        max_diff = max(d_tm, d_pos, d_ri, d_rnt, d_rns)
        
        return is_passed, max_diff