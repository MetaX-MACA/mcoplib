import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Tree_speculative_sampling_target_only_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.bs = config.get("batch_size", 16)
        self.num_spec_step = config.get("num_spec_step", 5)
        self.num_draft_tokens = config.get("num_draft_tokens", 5)
        self.tot_num_draft_tokens = config.get("tot_num_draft_tokens", self.bs * self.num_draft_tokens)
        self.vocab_size = config.get("vocab_size", 32000)
        
        self.threshold_single = config.get("threshold_single", 0.5)
        self.threshold_acc = config.get("threshold_acc", 0.8)
        self.deterministic = config.get("deterministic", True)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"bs:{self.bs}_draft:{self.num_draft_tokens}_v:{self.vocab_size}")
        
        total_elements = self.bs * self.num_draft_tokens
        state.add_element_count(total_elements)
        
        float_bytes = 4
        int32_bytes = 4
        int64_bytes = 8
        
        prob_elements = self.bs * self.num_draft_tokens
        
        read_bytes = (
            (prob_elements * float_bytes * 2) + 
            (self.bs * self.num_draft_tokens * float_bytes * 2) + 
            (self.bs * self.num_draft_tokens * int64_bytes * 4) + 
            (self.tot_num_draft_tokens * int32_bytes)
        )
        
        write_bytes = (self.bs * self.num_spec_step * int32_bytes) + (self.bs * int32_bytes)
        
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def _generate_inputs(self, dev):
        predicts = torch.zeros(self.tot_num_draft_tokens, dtype=torch.int32, device=dev)
        accept_index = torch.zeros((self.bs, self.num_spec_step), dtype=torch.int32, device=dev)
        accept_token_num = torch.zeros(self.bs, dtype=torch.int32, device=dev)
        
        candidates = torch.randint(0, self.vocab_size, (self.bs, self.num_draft_tokens), dtype=torch.int64, device=dev)
        
        retrive_index = torch.zeros((self.bs, self.num_draft_tokens), dtype=torch.int64, device=dev)
        retrive_next_token = torch.full((self.bs, self.num_draft_tokens), -1, dtype=torch.int64, device=dev)
        retrive_next_sibling = torch.full((self.bs, self.num_draft_tokens), -1, dtype=torch.int64, device=dev)
        
        uniform_samples = torch.rand((self.bs, self.num_draft_tokens), dtype=torch.float32, device=dev)
        uniform_samples_final = torch.rand((self.bs, self.num_draft_tokens), dtype=torch.float32, device=dev)
        
        target_probs = torch.softmax(torch.randn((self.bs, self.num_draft_tokens, self.vocab_size), dtype=torch.float32, device=dev), dim=-1)
        draft_probs = torch.softmax(torch.randn((self.bs, self.num_draft_tokens, self.vocab_size), dtype=torch.float32, device=dev), dim=-1)
        
        return (predicts, accept_index, accept_token_num, candidates, retrive_index, 
                retrive_next_token, retrive_next_sibling, uniform_samples, 
                uniform_samples_final, target_probs, draft_probs)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = f'cuda:{dev_id}'
        inputs = self._generate_inputs(dev)
        
        predicts = inputs[0]
        accept_index = inputs[1]
        accept_token_num = inputs[2]
        
        def custom_launcher(launch):
            stream = self.as_torch_stream(launch.get_stream(), dev_id)
            with torch.cuda.stream(stream):
                predicts.zero_()
                accept_index.zero_()
                accept_token_num.zero_()
                
                torch.ops.sgl_kernel.tree_speculative_sampling_target_only(
                    *inputs, 
                    self.threshold_single, 
                    self.threshold_acc, 
                    self.deterministic
                )
        return custom_launcher

    def run_verification(self, dev_id):
        #在Token树中寻找最长的合法路径。如果某条分支的概率不达标，该分支及其后续子树都会被“剪掉”
        dev = f'cuda:{dev_id}'
        inputs = self._generate_inputs(dev)
        
        torch.ops.sgl_kernel.tree_speculative_sampling_target_only(
            *inputs, 
            self.threshold_single, 
            self.threshold_acc, 
            self.deterministic
        )
        
        accept_token_num = inputs[2]
        #检查是否有无效元素NaN
        is_passed = not torch.isnan(accept_token_num.float()).any().item()
        return is_passed, 0.0