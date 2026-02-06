import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Apply_repetition_penalties__runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 32)
        self.vocab_size = config.get("vocab_size", 32000)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.batch_size} {self.vocab_size})")
        
        total_elements = self.batch_size * self.vocab_size
        state.add_element_count(total_elements)
        
        element_size = 2 if self.dtype == torch.float16 else 4
        mask_size = 1
        
        total_read = (total_elements * element_size) + \
                     (total_elements * mask_size * 2) + \
                     (self.batch_size * element_size)
                     
        total_write = total_elements * element_size
        
        state.add_global_memory_reads(total_read)
        state.add_global_memory_writes(total_write)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            shape = (self.batch_size, self.vocab_size)
            
            logits = torch.randn(shape, dtype=self.dtype, device=dev)
            
            prompt_mask = (torch.rand(shape, device=dev) > 0.9).to(torch.bool)
            output_mask = (torch.rand(shape, device=dev) > 0.9).to(torch.bool)
            
            penalties = torch.rand(self.batch_size, dtype=self.dtype, device=dev) + 1.1

        return self.make_launcher(dev_id, torch.ops._C.apply_repetition_penalties_, 
                                  logits, prompt_mask, output_mask, penalties)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        shape = (self.batch_size, self.vocab_size)
        
        logits = torch.randn(shape, dtype=self.dtype, device=dev)
        prompt_mask = (torch.rand(shape, device=dev) > 0.9).to(torch.bool)
        output_mask = (torch.rand(shape, device=dev) > 0.9).to(torch.bool)
        penalties = (torch.rand(self.batch_size, dtype=self.dtype, device=dev) * 0.5) + 1.1
        
        logits_ref = logits.clone().float()
        penalties_ref = penalties.float().unsqueeze(1).expand_as(logits_ref)
        combined_mask = prompt_mask | output_mask
        
        pos_mask = (logits_ref > 0) & combined_mask
        logits_ref[pos_mask] = logits_ref[pos_mask] / penalties_ref[pos_mask]
        
        neg_mask = (logits_ref <= 0) & combined_mask
        logits_ref[neg_mask] = logits_ref[neg_mask] * penalties_ref[neg_mask]
        
        out_ref = logits_ref.to(self.dtype)
        
        torch.ops._C.apply_repetition_penalties_(logits, prompt_mask, output_mask, penalties)
        
        return self.check_diff(logits, out_ref)