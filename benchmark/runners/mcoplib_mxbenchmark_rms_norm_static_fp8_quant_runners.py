import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Rms_norm_static_fp8_quant_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 16384)
        self.hidden_size = config.get("hidden_size", 4096)
        self.epsilon = config.get("epsilon", 1e-6)
        if hasattr(torch, 'float8_e4m3fn'):
            self.out_dtype = torch.float8_e4m3fn
        else:
            self.out_dtype = torch.uint8

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.batch_size} {self.hidden_size})")
        total_elements = self.batch_size * self.hidden_size
        state.add_element_count(total_elements)
        in_bytes = 2 if self.dtype == torch.float16 else 4
        out_bytes = 1
        read_vol = (total_elements * in_bytes) + (self.hidden_size * in_bytes) + 4
        write_vol = total_elements * out_bytes
        state.add_global_memory_reads(read_vol)
        state.add_global_memory_writes(write_vol)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            shape = (self.batch_size, self.hidden_size)
            inp = torch.randn(shape, dtype=self.dtype, device=dev)
            weight = torch.ones(self.hidden_size, dtype=self.dtype, device=dev)
            scale = torch.tensor([1.0], dtype=torch.float32, device=dev)
            out = torch.empty(shape, dtype=self.out_dtype, device=dev)
        return self.make_launcher(dev_id, torch.ops._C.rms_norm_static_fp8_quant, 
                                  out, inp, weight, scale, self.epsilon)

    def run_verification(self, dev_id):
        if not hasattr(torch, 'float8_e4m3fn'):
            return True, 0.0
        dev = f'cuda:{dev_id}'
        shape = (self.batch_size, self.hidden_size)
        inp = torch.randn(shape, dtype=self.dtype, device=dev)
        weight = torch.randn(self.hidden_size, dtype=self.dtype, device=dev)
        scale_val = 0.5
        scale = torch.tensor([scale_val], dtype=torch.float32, device=dev)
        out_op = torch.empty(shape, dtype=torch.float8_e4m3fn, device=dev)
        torch.ops._C.rms_norm_static_fp8_quant(out_op, inp, weight, scale, self.epsilon)
        inp_f = inp.float()
        w_f = weight.float()
        rms = torch.sqrt(inp_f.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        normed = (inp_f / rms) * w_f
        out_ref_fp8 = (normed / scale_val).to(torch.float8_e4m3fn)
        out_op_f = out_op.float()
        out_ref_f = out_ref_fp8.float()
        return self.check_diff(out_op_f, out_ref_f, threshold=0.95)