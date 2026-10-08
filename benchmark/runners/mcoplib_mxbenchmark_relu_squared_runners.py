import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Relu_squared_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.B = config.get("batch_size")
        self.S = config.get("seq_len")
        self.H = config.get("hidden_size")

        self.shape = (self.B, self.S, self.H)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", "(" + " ".join(map(str, self.shape)) + ")")

        total_elements = 1
        for x in self.shape:
            total_elements *= x

        state.add_element_count(total_elements)

        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4

        state.add_global_memory_reads(total_elements * element_size)
        state.add_global_memory_writes(total_elements * element_size)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            torch.manual_seed(42)

            dev = f"cuda:{dev_id}"

            input_tensor = torch.randn(*self.shape, dtype=self.dtype, device=dev)
            output = torch.empty_like(input_tensor)

        return self.make_launcher(dev_id, torch.ops._C.relu_squared, output, input_tensor)

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        torch.manual_seed(42)

        input_tensor = torch.randn(*self.shape, dtype=self.dtype, device=dev)
        output = torch.empty_like(input_tensor)

        torch.ops._C.relu_squared(output, input_tensor)

        torch.cuda.synchronize(device=dev)

        ref = torch.relu(input_tensor.float()) ** 2

        return self.check_diff(output, ref.to(self.dtype))