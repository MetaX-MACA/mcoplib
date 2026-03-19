import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Segment_packbits_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 8192)
        self.bits_per_segment = config.get("bits_per_segment", 256)
        assert self.bits_per_segment % 8 == 0, "bits_per_segment must be a multiple of 8"

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "bool->uint8")
        state.add_summary("Shape", f"({self.batch_size}, {self.bits_per_segment})")
        
        total_bits = self.batch_size * self.bits_per_segment
        total_bytes_out = total_bits // 8
        state.add_element_count(total_bits)
        
        read_bytes = total_bits + (self.batch_size + 1) * 4 + (self.batch_size + 1) * 4
        write_bytes = total_bytes_out
        
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            total_bits = self.batch_size * self.bits_per_segment
            total_bytes_out = total_bits // 8
            
            x = torch.randint(0, 2, (total_bits,), dtype=torch.bool, device=dev)
            input_indptr = torch.arange(0, total_bits + 1, self.bits_per_segment, dtype=torch.int32, device=dev)
            output_indptr = torch.arange(0, total_bytes_out + 1, self.bits_per_segment // 8, dtype=torch.int32, device=dev)
            y = torch.empty((total_bytes_out,), dtype=torch.uint8, device=dev)
            
            stream_ptr = tc_s.cuda_stream
            
        return self.make_launcher(
            dev_id, 
            torch.ops.sgl_kernel.segment_packbits, 
            x, input_indptr, output_indptr, y, self.batch_size, stream_ptr
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        total_bits = self.batch_size * self.bits_per_segment
        total_bytes_out = total_bits // 8
        
        x = torch.randint(0, 2, (total_bits,), dtype=torch.bool, device=dev)
        input_indptr = torch.arange(0, total_bits + 1, self.bits_per_segment, dtype=torch.int32, device=dev)
        output_indptr = torch.arange(0, total_bytes_out + 1, self.bits_per_segment // 8, dtype=torch.int32, device=dev)
        y = torch.empty((total_bytes_out,), dtype=torch.uint8, device=dev)
        
        torch.ops.sgl_kernel.segment_packbits(
            x, input_indptr, output_indptr, y, self.batch_size, 0
        )
        
        x_reshaped = x.view(-1, 8).to(torch.uint8)
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=dev)
        expected_y = (x_reshaped * powers).sum(dim=1, dtype=torch.uint8)
        
        is_passed, max_diff = self.check_diff(y, expected_y)
        
        return is_passed, max_diff