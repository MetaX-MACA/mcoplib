import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.op as op
except ImportError:
    op = None

class Rms_norm_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.B = config.get("batch_size", 64)
        self.S = config.get("seq_len", 2048)
        self.H = config.get("hidden_size", 4096)
        self.var_epsilon = config.get("var_epsilon", 1e-6)
        self.rms_div = config.get("rms_div", False)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.B} {self.S} {self.H})")
        
        total_elements = self.B * self.S * self.H
        state.add_element_count(total_elements)
        
        # 估算显存读写 (Bytes)
        # float16 = 2 bytes
        element_size = 2 if self.dtype == torch.float16 else 4
        # Reads: input + weight + residual
        reads = (total_elements * 2 + self.H) * element_size
        # Writes: out + after_res
        writes = (total_elements * 2) * element_size
        
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            input_tensor = torch.randn(self.B, self.S, self.H, dtype=self.dtype, device=dev)
            weight = torch.randn(self.H, dtype=self.dtype, device=dev)
            residual = torch.randn(self.B, self.S, self.H, dtype=self.dtype, device=dev)
            
            out = torch.empty(self.B, self.S, self.H, dtype=self.dtype, device=dev)
            after_res = torch.empty(self.B, self.S, self.H, dtype=self.dtype, device=dev)
            
        return self.make_launcher(dev_id, op.rms_norm, 
                                  out, input_tensor, weight, self.var_epsilon, 
                                  after_res, residual, self.rms_div)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        # 使用小 Shape 验证
        B, S, H = 2, 16, 128
        input_tensor = torch.randn(B, S, H, dtype=self.dtype, device=dev)
        weight = torch.randn(H, dtype=self.dtype, device=dev)
        residual = torch.randn(B, S, H, dtype=self.dtype, device=dev)
        
        out_op = torch.empty(B, S, H, dtype=self.dtype, device=dev)
        after_res_op = torch.empty(B, S, H, dtype=self.dtype, device=dev)
        
        op.rms_norm(out_op, input_tensor, weight, self.var_epsilon, 
                    after_res_op, residual, self.rms_div)
        
        # Reference
        inp_f = input_tensor.float()
        res_f = residual.float()
        w_f = weight.float()
        
        # 1. Residual Add
        after_res_ref = inp_f + res_f
        
        # 2. RMS Calculation
        mean_sq = torch.mean(after_res_ref ** 2, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_sq + self.var_epsilon)
        
        # 3. Norm
        if self.rms_div:
            normed = after_res_ref / rms
        else:
            normed = after_res_ref * (1.0 / rms)
            
        out_ref = normed * w_f
        
        return self.check_diff(out_op, out_ref.to(self.dtype))