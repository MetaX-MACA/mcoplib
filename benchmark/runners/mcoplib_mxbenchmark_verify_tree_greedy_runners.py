import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Verify_tree_greedy_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.bs = config.get("batch_size", 256)
        self.num_spec_step = config.get("num_spec_step", 8)
        self.num_draft_tokens = config.get("num_draft_tokens", 16)
        self.vocab_size = config.get("vocab_size", 32000)
        self.tot_num_draft_tokens = self.bs * self.num_draft_tokens

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"bs:{self.bs}_draft:{self.num_draft_tokens}_spec:{self.num_spec_step}")
        
        total_elements = self.tot_num_draft_tokens
        state.add_element_count(total_elements)
        
        int32_bytes = 4
        int64_bytes = 8
        
        read_bytes = (
            (self.tot_num_draft_tokens * int64_bytes * 5)
        )
        write_bytes = (
            (self.tot_num_draft_tokens * int32_bytes) + 
            (self.bs * self.num_spec_step * int32_bytes) + 
            (self.bs * int32_bytes)
        )
        
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def _generate_inputs(self, dev):
        predicts = torch.zeros(self.tot_num_draft_tokens, dtype=torch.int32, device=dev)
        accept_index = torch.zeros((self.bs, self.num_spec_step), dtype=torch.int32, device=dev)
        accept_token_num = torch.zeros(self.bs, dtype=torch.int32, device=dev)
        
        candidates = torch.randint(0, self.vocab_size, (self.bs, self.num_draft_tokens), dtype=torch.int64, device=dev)
        target_predict = torch.randint(0, self.vocab_size, (self.bs, self.num_draft_tokens), dtype=torch.int64, device=dev)
        
        retrive_index = torch.arange(self.tot_num_draft_tokens, dtype=torch.int64, device=dev).view(self.bs, self.num_draft_tokens)
        
        retrive_next_token = torch.full((self.bs, self.num_draft_tokens), -1, dtype=torch.int64, device=dev)
        retrive_next_sibling = torch.full((self.bs, self.num_draft_tokens), -1, dtype=torch.int64, device=dev)
        
        for bx in range(self.bs):
            for i in range(self.num_draft_tokens - 1):
                retrive_next_token[bx, i] = i + 1
        
        for bx in range(self.bs):
            if torch.rand(1).item() > 0.5:
                idx = retrive_index[bx, 0].item()
                target_predict.view(-1)[idx] = candidates[bx, 1]

        return (predicts, accept_index, accept_token_num, candidates, 
                retrive_index, retrive_next_token, retrive_next_sibling, target_predict)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = f'cuda:{dev_id}'
        inputs = self._generate_inputs(dev)
        
        predicts, accept_index, accept_token_num = inputs[0], inputs[1], inputs[2]
        
        def custom_launcher(launch):
            stream = self.as_torch_stream(launch.get_stream(), dev_id)
            with torch.cuda.stream(stream):
                predicts.zero_()
                accept_index.zero_()
                accept_token_num.zero_()
                
                torch.ops.sgl_kernel.verify_tree_greedy(*inputs)
        return custom_launcher

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        inputs = self._generate_inputs(dev)
        
        predicts, accept_index, accept_token_num = inputs[0], inputs[1], inputs[2]
        candidates, retrive_index, retrive_next_token, retrive_next_sibling, target_predict = inputs[3:]
        
        cand_cpu = candidates.cpu()
        ret_idx_cpu = retrive_index.cpu()
        ret_next_cpu = retrive_next_token.cpu()
        ret_sib_cpu = retrive_next_sibling.cpu()
        tgt_flat_cpu = target_predict.view(-1).cpu()
        
        exp_predicts = torch.zeros_like(predicts.cpu())
        exp_accept_index = torch.zeros_like(accept_index.cpu())
        exp_accept_token_num = torch.zeros_like(accept_token_num.cpu())
        
        for bx in range(self.bs):
            last_accepted_retrive_idx = ret_idx_cpu[bx, 0].item()
            exp_accept_index[bx, 0] = last_accepted_retrive_idx
            num_accepted_tokens = 0
            cur_index = 0
            
            for j in range(1, self.num_spec_step):
                cur_index = ret_next_cpu[bx, cur_index].item()
                while cur_index != -1:
                    draft_index = ret_idx_cpu[bx, cur_index].item()
                    draft_token_id = cand_cpu[bx, cur_index].item()
                    target_token_id = tgt_flat_cpu[last_accepted_retrive_idx].item()
                    
                    if draft_token_id == target_token_id:
                        exp_predicts[last_accepted_retrive_idx] = target_token_id
                        num_accepted_tokens += 1
                        exp_accept_index[bx, num_accepted_tokens] = draft_index
                        last_accepted_retrive_idx = draft_index
                        break
                    else:
                        cur_index = ret_sib_cpu[bx, cur_index].item()
                        
                if cur_index == -1:
                    break
                    
            exp_accept_token_num[bx] = num_accepted_tokens
            exp_predicts[last_accepted_retrive_idx] = tgt_flat_cpu[last_accepted_retrive_idx].item()
        
        torch.ops.sgl_kernel.verify_tree_greedy(*inputs)
        
        pass_p, diff_p = self.check_diff(predicts, exp_predicts.to(dev))
        pass_ai, diff_ai = self.check_diff(accept_index, exp_accept_index.to(dev))
        pass_an, diff_an = self.check_diff(accept_token_num, exp_accept_token_num.to(dev))
        
        is_passed = pass_p and pass_ai and pass_an
        max_diff = max(diff_p, diff_ai, diff_an)
        
        return is_passed, max_diff