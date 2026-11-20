
import math
import torch
import mcoplib._C

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


class Fused_deepseek_v4_kv_rope_insert_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.n_tokens = config.get("n_tokens")
        self.block_size = config.get("block_size")
        self.cache_format = config.get("cache_format")
        self.max_pos = config.get("max_pos", 32)
        self.samples = config.get("samples", 100)

        self.kv_width = 512
        self.rope_dim = 64
        self.nope_dim = 448

        if self.cache_format == "bfloat16":
            self.cache_dtype = torch.bfloat16
            self.cache_width = 512
            self.kv_mxfp8 = False
            self.cache_element_size = 2
        elif self.cache_format == "uint8_mxfp8":
            self.cache_dtype = torch.uint8
            self.cache_width = 528
            self.kv_mxfp8 = True
            self.cache_element_size = 1
        elif self.cache_format == "uint8_non_mxfp8":
            self.cache_dtype = torch.uint8
            self.cache_width = 584
            self.kv_mxfp8 = False
            self.cache_element_size = 1
        elif self.cache_format == "float8_e4m3fn":
            if not hasattr(torch, "float8_e4m3fn"):
                raise RuntimeError("torch.float8_e4m3fn is unavailable")
            self.cache_dtype = torch.float8_e4m3fn
            self.cache_width = 512
            self.kv_mxfp8 = False
            self.cache_element_size = 1
        else:
            raise ValueError(f"Unsupported cache_format: {self.cache_format}")

        if self.block_size is None or self.block_size <= 0:
            raise ValueError(f"Invalid block_size: {self.block_size}")

        self.n_blocks = max(2, math.ceil(self.n_tokens / self.block_size))

        self.shape = (self.n_tokens, self.kv_width)
        self.cache_shape = (self.n_blocks, self.block_size, self.cache_width)
        self.total_slots = self.n_blocks * self.block_size

        self.shape_summary = (
            f"tokens={self.n_tokens} "
            f"block_size={self.block_size} "
            f"cache_format={self.cache_format} "
            f"cache_shape=({self.n_blocks}x{self.block_size}x{self.cache_width}) "
            f"max_pos={self.max_pos}"
        )

    def _make_inputs(self, dev):
        kv = torch.randn(self.shape, dtype=torch.bfloat16, device=dev)

        k_cache = torch.full(self.cache_shape, 7, dtype=self.cache_dtype, device=dev)

        slot_mapping = torch.arange(self.n_tokens, dtype=torch.int64, device=dev)

        position_ids = torch.arange(self.n_tokens, dtype=torch.int64, device=dev).remainder(self.max_pos)

        cos_sin_cache = torch.zeros((self.max_pos, self.rope_dim), dtype=torch.float32, device=dev)

        if self.cache_dtype == getattr(torch, "float8_e4m3fn", None):
            fp8_scale = torch.ones((), dtype=torch.float32, device=dev)
        else:
            fp8_scale = None

        return (kv, k_cache, slot_mapping, position_ids, cos_sin_cache, self.block_size, fp8_scale, self.kv_mxfp8)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", f"bfloat16/{self.cache_format}")
        state.add_summary("Shape", self.shape_summary)

        kv_bytes = self.n_tokens * self.kv_width * 2
        slot_bytes = self.n_tokens * 8
        position_bytes = self.n_tokens * 8
        cos_sin_bytes = self.max_pos * self.rope_dim * 4
        cache_bytes = self.n_tokens * self.cache_width * self.cache_element_size

        state.add_element_count(self.n_tokens * self.kv_width)
        state.add_global_memory_reads(kv_bytes + slot_bytes + position_bytes + cos_sin_bytes)
        state.add_global_memory_writes(cache_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            tensors = self._make_inputs(dev)

        return self.make_launcher(dev_id, torch.ops._C.fused_deepseek_v4_kv_rope_insert, *tensors)

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        tensors = self._make_inputs(dev)

        kv, k_cache, slot_mapping, position_ids, cos_sin_cache, block_size, fp8_scale, kv_mxfp8 = tensors

        cache_before = k_cache.clone()

        torch.ops._C.fused_deepseek_v4_kv_rope_insert(*tensors)
        torch.cuda.synchronize()

        for i in range(self.n_tokens):
            slot = int(slot_mapping[i].item())
            block, offset = divmod(slot, block_size)

            row = k_cache[block, offset]

            if self.cache_format == "bfloat16":
                torch.testing.assert_close(row[:self.nope_dim], kv[i, :self.nope_dim])
                torch.testing.assert_close(row[self.nope_dim:self.kv_width], torch.zeros(self.rope_dim, dtype=torch.bfloat16, device=dev))
            else:
                assert not torch.equal(row, cache_before[block, offset]), f"Cache row was not updated: token={i}, slot={slot}"

        touched_slots = set(slot_mapping.cpu().tolist())

        for slot in range(self.total_slots):
            if slot not in touched_slots:
                block, offset = divmod(slot, block_size)
                torch.testing.assert_close(k_cache[block, offset], cache_before[block, offset])

        return True, 0.0
