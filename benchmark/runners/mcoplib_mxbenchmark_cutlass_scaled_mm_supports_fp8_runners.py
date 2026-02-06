import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

func_name = "cutlass_scaled_mm_supports_fp8"
_kernel_func = None

try:
    import mcoplib._C
    if hasattr(mcoplib._C, func_name):
        _kernel_func = getattr(mcoplib._C, func_name)
except ImportError:
    pass

if _kernel_func is None:
    try:
        if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, func_name):
            _kernel_func = getattr(torch.ops._C, func_name)
    except:
        pass

class Cutlass_scaled_mm_supports_fp8_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.capability = config.get("cuda_device_capability", 90)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("Shape", f"Cap={self.capability}")
        state.add_summary("dtype", "")
        state.add_element_count(1)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if _kernel_func is None:
            raise RuntimeError(f"算子 {func_name} 未找到")
        with torch.cuda.stream(tc_s):
            pass
        return self.make_launcher(dev_id, _kernel_func, self.capability)

    def run_verification(self, dev_id):
        if _kernel_func is None:
            return False, 1.0
        ret = _kernel_func(self.capability)
        expected = False
        passed = (ret == expected)
        if passed:
            return True, 0.0
        else:
            print(f"Warning: Expected {expected} but got {ret}")
            return False, 1.0