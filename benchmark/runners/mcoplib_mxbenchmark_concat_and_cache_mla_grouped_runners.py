import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Concat_and_cache_mla_grouped_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_layers = config.get("num_layers", 8)
        self.num_tokens = config.get("num_tokens", 128)
        self.kv_lora_rank = config.get("kv_lora_rank", 512)
        self.pe_dim = config.get("pe_dim", 64)
        self.block_size = config.get("block_size", 16)
        self.dtype = torch.bfloat16

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"layers={self.num_layers}, tokens={self.num_tokens}, dim={self.kv_lora_rank+self.pe_dim}")

        elements = self.num_layers * self.num_tokens * (self.kv_lora_rank + self.pe_dim)
        bytes_size = elements * 2

        state.add_element_count(elements)
        state.add_global_memory_reads(bytes_size)
        state.add_global_memory_writes(bytes_size)


    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"

            num_blocks = max(64, self.num_tokens // self.block_size + 16)

            entry_stride = self.kv_lora_rank + self.pe_dim
            block_stride = self.block_size * entry_stride

            kv_c = torch.randn(self.num_layers, self.num_tokens, self.kv_lora_rank, dtype=self.dtype, device=dev)
            k_pe = torch.randn(self.num_layers, self.num_tokens, self.pe_dim, dtype=self.dtype, device=dev)

            slot_mapping = torch.full((self.num_layers, self.num_tokens), -1, dtype=torch.int64, device=dev)

            for i in range(self.num_layers):
                ids = torch.randperm(num_blocks * self.block_size, device=dev)[:self.num_tokens]
                mask = torch.rand(self.num_tokens, device=dev) > 0.1
                slot_mapping[i, mask] = ids[mask]

            caches = [torch.zeros(num_blocks, self.block_size, entry_stride, dtype=self.dtype, device=dev) for _ in range(self.num_layers)]

            cache_ptrs = torch.tensor([x.data_ptr() for x in caches], dtype=torch.int64, device=dev)

        return self.make_launcher(dev_id, torch.ops._C.concat_and_cache_mla_grouped, kv_c, k_pe, cache_ptrs, slot_mapping, self.block_size, block_stride, entry_stride)


    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        num_blocks = 32

        entry_stride = self.kv_lora_rank + self.pe_dim
        block_stride = self.block_size * entry_stride

        kv_c = torch.randn(self.num_layers, self.num_tokens, self.kv_lora_rank, dtype=self.dtype, device=dev)
        k_pe = torch.randn(self.num_layers, self.num_tokens, self.pe_dim, dtype=self.dtype, device=dev)

        slot_mapping = torch.arange(self.num_layers * self.num_tokens, device=dev, dtype=torch.int64)
        slot_mapping = slot_mapping.reshape(self.num_layers, self.num_tokens)
        slot_mapping %= num_blocks * self.block_size

        ref_cache = [torch.zeros(num_blocks, self.block_size, entry_stride, dtype=self.dtype, device=dev) for _ in range(self.num_layers)]
        out_cache = [torch.zeros_like(x) for x in ref_cache]

        for l in range(self.num_layers):
            for t in range(self.num_tokens):
                slot = int(slot_mapping[l,t])
                block = slot // self.block_size
                offset = slot % self.block_size
                ref_cache[l][block, offset, :self.kv_lora_rank] = kv_c[l,t]
                ref_cache[l][block, offset,self.kv_lora_rank:] = k_pe[l,t]


        ptrs = torch.tensor([x.data_ptr() for x in out_cache], dtype=torch.int64, device=dev)

        torch.ops._C.concat_and_cache_mla_grouped(kv_c,k_pe,ptrs,slot_mapping,self.block_size,block_stride,entry_stride)


        ref = torch.cat([x.flatten() for x in ref_cache])
        out = torch.cat([x.flatten() for x in out_cache])


        if torch.all(ref == 0) and torch.all(out == 0):
            cos = 1.0
        else:
            cos = F.cosine_similarity(ref.float(), out.float(), dim=0).item()


        passed = torch.equal(ref, out)

        return passed, 1.0 - cos