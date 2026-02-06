import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C as op
    if not hasattr(op, "awq_dequantize"):
        op = None
except ImportError:
    op = None

class Awq_dequantize_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.in_c = config.get("in_c", 4096)
        self.out_c = config.get("out_c", 4096)
        self.group_size = config.get("group_size", 128)
        self.split_k_iters = config.get("split_k_iters", 0)
        self.thx = config.get("thx", 0)
        self.thy = config.get("thy", 0)
        
        assert self.out_c % 8 == 0, "out_c must be divisible by 8"
        assert self.in_c % self.group_size == 0, "in_c must be divisible by group_size"
        
        self.qout_c = self.out_c // 8
        self.num_groups = self.in_c // self.group_size

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"({self.in_c}x{self.out_c} G={self.group_size})")
        
        total_elements = self.in_c * self.out_c
        state.add_element_count(total_elements)
        
        read_bytes = self.in_c * self.qout_c * 4
        read_bytes += self.num_groups * self.qout_c * 4
        read_bytes += self.num_groups * self.out_c * 2
        
        write_bytes = total_elements * 2
        
        state.add_global_memory_reads(int(read_bytes))
        state.add_global_memory_writes(int(write_bytes))

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            kernel = torch.randint(-2**31, 2**31-1, (self.in_c, self.qout_c), dtype=torch.int32, device=dev)
            zeros = torch.randint(-2**31, 2**31-1, (self.num_groups, self.qout_c), dtype=torch.int32, device=dev)
            scales = torch.randn(self.num_groups, self.out_c, dtype=self.dtype, device=dev)
            
            if op and hasattr(op, "awq_dequantize"):
                func = op.awq_dequantize
            else:
                func = torch.ops._C.awq_dequantize

        return self.make_launcher(dev_id, func, 
                                  kernel, scales, zeros, 
                                  self.split_k_iters, self.thx, self.thy)

    def unpack_tensor(self, packed, shifts, reorder_indices=None):
        unpacked_list = []
        for sh in shifts:
            val = (packed.to(torch.int64) >> sh) & 0xF
            unpacked_list.append(val)
        
        result = torch.stack(unpacked_list, dim=-1)
        
        if reorder_indices is not None:
            result = result[..., reorder_indices]
            
        return result.flatten(-2)

    def run_verification(self, dev_id):
        in_c, out_c = 1024, 256
        group_size = 128
        groups = in_c // group_size
        qout_c = out_c // 8
        
        dev = f'cuda:{dev_id}'
        dtype = self.dtype

        kernel = torch.randint(-2**31, 2**31-1, (in_c, qout_c), dtype=torch.int32, device=dev)
        zeros = torch.randint(-2**31, 2**31-1, (groups, qout_c), dtype=torch.int32, device=dev)
        scales = torch.randn(groups, out_c, dtype=dtype, device=dev)

        func = None
        if op and hasattr(op, "awq_dequantize"):
            func = op.awq_dequantize
        else:
            try:
                func = torch.ops._C.awq_dequantize
            except:
                print("[Verify] Op not found.")
                return False, 1.0

        try:
            op_out = func(kernel, scales, zeros, 0, 0, 0)
        except RuntimeError as e:
            print(f"[Verify] Op Error: {e}")
            return False, 1.0

        shifts = [0, 4, 8, 12, 16, 20, 24, 28]
        awq_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=dev)

        w_int = self.unpack_tensor(kernel, shifts, reorder_indices=awq_order).to(dtype)
        z_unpacked = self.unpack_tensor(zeros, shifts, reorder_indices=awq_order).to(dtype)
        
        s_expanded = scales.repeat_interleave(group_size, dim=0)
        z_expanded = z_unpacked.repeat_interleave(group_size, dim=0)
        
        ref_out = (w_int - z_expanded) * s_expanded
        
        return self.check_diff(op_out, ref_out, threshold=0.99)