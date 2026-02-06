import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class StaticScaledInt8QuantRunner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 16)
        self.seq_len = config.get("seq_len", 128)
        self.hidden_size = config.get("hidden_size", 4096)
        self.has_azp = config.get("has_azp", False)
        self.out_dtype = torch.int8

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.batch_size} {self.seq_len} {self.hidden_size})")
        total_elements = self.batch_size * self.seq_len * self.hidden_size
        state.add_element_count(total_elements)
        if self.dtype == torch.float32:
            input_bytes_per_elem = 4
        elif self.dtype == torch.float16 or self.dtype == torch.bfloat16:
            input_bytes_per_elem = 2
        else:
            input_bytes_per_elem = 2
        output_bytes_per_elem = 1
        read_bytes = total_elements * input_bytes_per_elem
        write_bytes = total_elements * output_bytes_per_elem
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            shape = (self.batch_size, self.seq_len, self.hidden_size)
            inp = torch.randn(shape, dtype=self.dtype, device=dev)
            out = torch.empty(shape, dtype=self.out_dtype, device=dev)
            scale = torch.tensor([0.005], dtype=torch.float32, device=dev)
            azp = None
            if self.has_azp:
                azp = torch.tensor([10], dtype=torch.int32, device=dev)
        return self.make_launcher(dev_id, torch.ops._C.static_scaled_int8_quant, out, inp, scale, azp)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        shape = (self.batch_size, self.seq_len, self.hidden_size)
        inp = torch.randn(shape, dtype=self.dtype, device=dev)
        out_op = torch.empty(shape, dtype=self.out_dtype, device=dev)
        scale = torch.tensor([0.005], dtype=torch.float32, device=dev)
        azp = None
        azp_val = 0
        if self.has_azp:
            azp_val = 10
            azp = torch.tensor([azp_val], dtype=torch.int32, device=dev)
        torch.ops._C.static_scaled_int8_quant(out_op, inp, scale, azp)
        scale_val = scale.item()
        inp_float = inp.float()
        scaled = inp_float / scale_val
        if azp is not None:
            scaled = scaled + azp_val
        out_ref = torch.clamp(torch.round(scaled), -127, 127).to(torch.int8)
        return self.check_diff(out_op, out_ref, threshold=0.9999)