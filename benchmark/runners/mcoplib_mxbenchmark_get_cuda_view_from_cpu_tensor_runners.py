import torch
import mcoplib._C  # 必须导入该模块以注册算子到 torch.ops
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

class Get_cuda_view_from_cpu_tensor_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.size = config.get("size", 1024 * 1024)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.size})")
        state.add_element_count(self.size)
        element_size = 2 if self.dtype == torch.float16 else 4
        state.add_global_memory_reads(self.size * element_size)
        state.add_global_memory_writes(0)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        # 输入必须是 CPU 上的 Pinned Memory
        inp = torch.randn(self.size, dtype=self.dtype, device="cpu").pin_memory()
        return self.make_launcher(dev_id, torch.ops._C.get_cuda_view_from_cpu_tensor, inp)

    def run_verification(self, dev_id):
        inp = torch.randn(self.size, dtype=self.dtype, device="cpu").pin_memory()
        
        # 执行算子获取 CUDA 视图
        cuda_view = torch.ops._C.get_cuda_view_from_cpu_tensor(inp)
        
        # 验证零拷贝：修改 GPU 视图，检查 CPU 原值
        check_value = 1234.5678
        cuda_view[0] = check_value
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        cpu_val = inp[0].item()
        diff = abs(cpu_val - check_value)
        
        return diff < 1e-4, diff