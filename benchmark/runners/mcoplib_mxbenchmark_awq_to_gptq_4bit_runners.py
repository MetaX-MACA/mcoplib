import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C as op
    if not hasattr(op, "awq_gemm"):
        op = None
except ImportError:
    op = None

class Awq_to_gptq_4bit_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.in_c = config.get("in_c", 4096)
        self.out_c = config.get("out_c", 4096)
        self.pack_factor = 8
        
        assert self.out_c % self.pack_factor == 0, "out_c must be divisible by 8"
        
        self.qout_c = self.out_c // self.pack_factor
        self.compact_k = (self.in_c + self.pack_factor - 1) // self.pack_factor

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"IC={self.in_c} OC={self.out_c}")

        input_bytes = self.in_c * self.qout_c * 4
        output_bytes = self.out_c * self.compact_k * 4
        
        state.add_global_memory_reads(int(input_bytes))
        state.add_global_memory_writes(int(output_bytes))
        state.add_element_count(int(self.in_c * self.out_c))

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            qweight = torch.randint(
                -2**31, 2**31-1, 
                (self.in_c, self.qout_c), 
                dtype=torch.int32, 
                device=dev
            )
            
            if op and hasattr(op, "awq_to_gptq_4bit"):
                func = op.awq_to_gptq_4bit
            else:
                func = torch.ops._C.awq_to_gptq_4bit

        return self.make_launcher(dev_id, func, qweight)

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
        in_c, out_c = 128, 256
        qout_c = out_c // 8 
        
        dev = f'cuda:{dev_id}'
        dtype = torch.int32 

        qweight = torch.randint(-2**31, 2**31-1, (in_c, qout_c), dtype=dtype, device=dev)

        try:
            op_out = torch.ops._C.awq_to_gptq_4bit(qweight)
            
        except AttributeError:
            print("[Verify] Op not found in torch.ops._C."); return False, 1.0
        except RuntimeError as e:
            print(f"[Verify] Op Runtime Error: {e}"); return False, 1.0

        shifts = [0, 4, 8, 12, 16, 20, 24, 28]
        
        awq_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=dev)
        
        ref_unpacked = self.unpack_tensor(qweight, shifts, reorder_indices=awq_order)
        
        ref_transposed = ref_unpacked.t()

        op_out_unpacked = self.unpack_tensor(op_out, shifts, reorder_indices=awq_order)
        
        return self.check_diff(op_out_unpacked.float(), ref_transposed.float(), threshold=0.1)

    def unpack_tensor(self, packed_tensor, shifts, reorder_indices=None):
        rows, cols = packed_tensor.shape
        data = packed_tensor.unsqueeze(-1)
        
        unpacked = torch.stack([(data >> s) & 0xF for s in shifts], dim=-1).squeeze(-2)
        
        if reorder_indices is not None:
            unpacked = unpacked.index_select(-1, reorder_indices)
            
        return unpacked.view(rows, cols * 8)