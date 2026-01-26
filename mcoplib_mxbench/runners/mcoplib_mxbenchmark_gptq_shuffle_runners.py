import torch
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

class Gptq_shuffle_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.in_features = config.get("in_features", 1024)
        self.out_features = config.get("out_features", 4096)
        self.bits = config.get("bits", 4)
        self.pack_factor = 32 // self.bits
        self.packed_rows = self.in_features // self.pack_factor

    def define_metrics(self, state):
        # Use getattr to prevent AttributeError if __init__ fails in some environments
        in_features = getattr(self, "in_features", 1024)
        out_features = getattr(self, "out_features", 4096)
        bits = getattr(self, "bits", 4)
        packed_rows = getattr(self, "packed_rows", in_features // (32 // bits))

        state.add_summary("Op", self.name)
        state.add_summary("Bits", str(bits))
        state.add_summary("Shape", f"({in_features}, {out_features})")
        
        # Memory traffic analysis
        weight_elements = packed_rows * out_features
        perm_elements = in_features
        
        # 4 bytes per int32
        total_read = (weight_elements + perm_elements) * 4
        total_write = weight_elements * 4
        
        state.add_global_memory_reads(total_read)
        state.add_global_memory_writes(total_write)
        state.add_element_count(weight_elements)
        
        # Disable blocking kernel timeout to prevent "Deadlock detected" error
        state.set_blocking_kernel_timeout(-1)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        # Retrieve parameters
        in_features = getattr(self, "in_features", 1024)
        out_features = getattr(self, "out_features", 4096)
        bits = getattr(self, "bits", 4)
        packed_rows = getattr(self, "packed_rows", in_features // (32 // bits))

        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            q_weight = torch.randint(
                low=-2147483648, 
                high=2147483647, 
                size=(packed_rows, out_features), 
                dtype=torch.int32,
                device=dev
            )
            q_perm = torch.randperm(in_features, dtype=torch.int32, device=dev)
            
        return self.make_launcher(dev_id, torch.ops._C.gptq_shuffle, q_weight, q_perm, bits)

    def run_verification(self, dev_id):
        # Retrieve parameters locally for safety
        in_features = getattr(self, "in_features", 1024)
        out_features = getattr(self, "out_features", 4096)
        bits = getattr(self, "bits", 4)
        pack_factor = 32 // bits
        packed_rows = getattr(self, "packed_rows", in_features // pack_factor)
        
        dev = f'cuda:{dev_id}'
        
        # Generate random inputs
        q_weight = torch.randint(
            low=-2147483648, 
            high=2147483647, 
            size=(packed_rows, out_features), 
            dtype=torch.int32,
            device=dev
        )
        q_perm = torch.randperm(in_features, dtype=torch.int32, device=dev)
        
        # Run operator (In-place)
        q_weight_op = q_weight.clone()
        torch.ops._C.gptq_shuffle(q_weight_op, q_perm, bits)
        
        # Reference Implementation
        # 1. Unpack
        mask = (1 << bits) - 1
        unpacked = torch.empty((in_features, out_features), dtype=torch.int32, device=dev)
        
        for i in range(pack_factor):
            shift = i * bits
            # Extract bits for the i-th packed element
            unpacked[i::pack_factor, :] = (q_weight >> shift) & mask
            
        # 2. Shuffle (Gather)
        shuffled_unpacked = unpacked[q_perm.long(), :]
        
        # 3. Repack
        q_weight_ref = torch.zeros_like(q_weight)
        for i in range(pack_factor):
            shift = i * bits
            # Pack bits back
            q_weight_ref |= (shuffled_unpacked[i::pack_factor, :] << shift)
            
        return self.check_diff(q_weight_op, q_weight_ref)