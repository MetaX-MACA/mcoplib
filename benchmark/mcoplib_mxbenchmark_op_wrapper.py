import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

# =============================================================================
#  [基类] OpBenchmarkBase
# =============================================================================
class OpBenchmarkBase(ABC):
    def __init__(self, name, config):
        self.name = name
        self.config = config
        
        if "dtype" in self.config:
            d_str = self.config["dtype"]
            if d_str == "float16": self.dtype = torch.float16
            elif d_str == "bfloat16": self.dtype = torch.bfloat16
            elif d_str == "float32": self.dtype = torch.float32
            elif hasattr(torch, d_str): self.dtype = getattr(torch, d_str)
            else: self.dtype = torch.float16 
        else:
            self.dtype = torch.float16

    @abstractmethod
    def define_metrics(self, state):
        pass

    @abstractmethod
    def prepare_and_get_launcher(self, dev_id, tc_s):
        pass

    @abstractmethod
    def run_verification(self, dev_id):
        pass

    def as_torch_stream(self, stream_ptr, dev_id):
        return torch.cuda.ExternalStream(
            stream_ptr=stream_ptr.addressof(), 
            device=torch.cuda.device(dev_id)
        )

    def make_launcher(self, dev_id, op_func, *args):
        def launcher(launch):
            stream = self.as_torch_stream(launch.get_stream(), dev_id)
            with torch.cuda.stream(stream):
                op_func(*args)
        return launcher

    def check_diff(self, output_op, output_ref, threshold=0.999999):
        out_op_flat = output_op.flatten().float()
        out_ref_flat = output_ref.flatten().float()
        cosine_sim = F.cosine_similarity(out_op_flat.unsqueeze(0), out_ref_flat.unsqueeze(0)).item()
        passed = cosine_sim >= threshold
        diff_val = 1.0 - cosine_sim
        return passed, diff_val