
# SPDX-License-Identifier: Apache-2.0

import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError as e:
    raise RuntimeError(f"Failed to import mcoplib._C: {e}") from e


OP_NAME = "cp_gather_and_upconvert_fp8_kv_cache"

SRC_CACHE_DIM = 656
FP8_DIM = 512
ROPE_SRC_OFFSET = 528
ROPE_DST_OFFSET = 512
DST_DIM = 576


class Cp_gather_and_upconvert_fp8_kv_cache_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.device_id = config.get("device_id", 0)
        self.batch_size = config.get("batch_size", 8)
        self.block_size = config.get("block_size", 16)
        self.seed = config.get("seed", 0)
        self.fp8_value = config.get("fp8_value", 0)
        self.use_seq_starts = config.get("use_seq_starts", False)
        self.use_host_rows = config.get("use_host_rows", False)
        self.reverse_block_table = config.get("reverse_block_table", False)

        if self.batch_size not in (4, 8):
            raise ValueError(f"batch_size must be 4 or 8, got {self.batch_size}")

        if self.block_size not in (16, 32):
            raise ValueError(f"block_size must be 16 or 32, got {self.block_size}")

        default_seq_lens = [17, 31, 33, 7, 16, 24, 48, 65]
        self.seq_lens = config.get("seq_lens", default_seq_lens[:self.batch_size])
        self.source_starts = config.get("source_starts", [0] * self.batch_size)

        if len(self.seq_lens) != self.batch_size:
            raise ValueError(f"seq_lens must contain {self.batch_size} values, got {len(self.seq_lens)}")

        if len(self.source_starts) != self.batch_size:
            raise ValueError(f"source_starts must contain {self.batch_size} values, got {len(self.source_starts)}")

        if any(seq_len <= 0 for seq_len in self.seq_lens):
            raise ValueError("All seq_lens values must be positive")

        if self.fp8_value < 0 or self.fp8_value > 126:
            raise ValueError(f"fp8_value must be in [0, 126], got {self.fp8_value}")

        if any(start < 0 for start in self.source_starts):
            raise ValueError("source_starts must be non-negative")

        self.total_tokens = sum(self.seq_lens)

        self.workspace_starts = [0]
        for seq_len in self.seq_lens[:-1]:
            self.workspace_starts.append(self.workspace_starts[-1] + seq_len)

        self.max_source_end = max(start + seq_len for start, seq_len in zip(self.source_starts, self.seq_lens))
        self.blocks_per_req = (self.max_source_end + self.block_size - 1) // self.block_size
        self.num_blocks = self.batch_size * self.blocks_per_req

        if self.blocks_per_req <= 0:
            raise RuntimeError(f"Invalid blocks_per_req={self.blocks_per_req}")

        if not hasattr(torch.ops._C_cache_ops, OP_NAME):
            raise RuntimeError(f"torch.ops._C_cache_ops.{OP_NAME} is not registered")

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "uint8->bfloat16")
        state.add_summary("Shape", f"({self.total_tokens} {DST_DIM})")

        src_bytes = self.num_blocks * self.block_size * SRC_CACHE_DIM
        block_table_bytes = self.batch_size * self.blocks_per_req * 4
        workspace_bytes = self.batch_size * 4
        dst_bytes = self.total_tokens * DST_DIM * 2

        state.add_element_count(self.total_tokens * DST_DIM)
        state.add_global_memory_reads(src_bytes + block_table_bytes + workspace_bytes)
        state.add_global_memory_writes(dst_bytes)

    def _make_src_cache(self, dev=None):
        if dev is None:
            dev = f"cuda:{self.device_id}"

        src_cache = torch.zeros((self.num_blocks, self.block_size, SRC_CACHE_DIM), dtype=torch.uint8, device=dev)

        fp8_values = torch.full((self.num_blocks, self.block_size, FP8_DIM), float(self.fp8_value), dtype=torch.float32, device=dev).to(torch.float8_e4m3fn)
        src_cache[:, :, :FP8_DIM] = fp8_values.view(torch.uint8)

        scales = torch.ones((self.num_blocks, self.block_size, 4), dtype=torch.float32, device=dev)
        src_cache[:, :, 512:528] = scales.view(torch.uint8)

        rope_values = torch.arange(32, dtype=torch.int32, device=dev).expand(self.num_blocks, self.block_size, 32).contiguous()
        src_cache[:, :, ROPE_SRC_OFFSET:SRC_CACHE_DIM] = rope_values.view(torch.uint8)

        return src_cache

    def _make_block_table(self, dev):
        physical_blocks = torch.arange(self.num_blocks, dtype=torch.int32, device=dev).view(self.batch_size, self.blocks_per_req)

        if self.reverse_block_table:
            physical_blocks = physical_blocks.flip(dims=[1]).contiguous()

        return physical_blocks

    def _make_host_rows(self, src_cache, dev):
        if not self.use_host_rows:
            return None, None, None

        host_cache = torch.empty((self.num_blocks * self.block_size, SRC_CACHE_DIM), dtype=torch.uint8, device="cpu", pin_memory=True)
        host_cache.copy_(src_cache.reshape(-1, SRC_CACHE_DIM), non_blocking=False)

        row_ids = torch.arange(self.num_blocks * self.block_size, dtype=torch.int32, device=dev)
        host_row_ids = row_ids.clone().contiguous()
        device_row_ids = row_ids.clone().contiguous()

        return host_cache, host_row_ids, device_row_ids

    def _make_inputs(self, dev):
        torch.manual_seed(self.seed)

        src_cache = self._make_src_cache(dev)
        block_table = self._make_block_table(dev)
        workspace_starts = torch.tensor(self.workspace_starts, dtype=torch.int32, device=dev)

        seq_starts = None
        if self.use_seq_starts:
            seq_starts = torch.tensor(self.source_starts, dtype=torch.int32, device=dev)

        dst = torch.zeros((self.total_tokens, DST_DIM), dtype=torch.bfloat16, device=dev)
        host_cache, host_row_ids, device_row_ids = self._make_host_rows(src_cache, dev)

        return src_cache, dst, block_table, workspace_starts, seq_starts, host_cache, host_row_ids, device_row_ids

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            inputs = self._make_inputs(dev)

        src_cache, dst, block_table, workspace_starts, seq_starts, host_cache, host_row_ids, device_row_ids = inputs

        return self.make_launcher(dev_id, torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache, src_cache, dst, block_table, workspace_starts, self.batch_size, seq_starts, host_cache, host_row_ids, device_row_ids)

    def _build_reference(self, src_cache, block_table, workspace_starts, seq_starts):
        dst_ref = torch.zeros((self.total_tokens, DST_DIM), dtype=torch.bfloat16, device=src_cache.device)

        fp8_src = src_cache[:, :, :FP8_DIM].contiguous().view(torch.float8_e4m3fn)
        fp8_bf16 = fp8_src.to(torch.bfloat16)

        for req_id in range(self.batch_size):
            output_begin = min(int(workspace_starts[req_id].item()), self.total_tokens)

            if req_id + 1 < self.batch_size:
                output_end = min(int(workspace_starts[req_id + 1].item()), self.total_tokens)
            else:
                output_end = self.total_tokens

            seq_len = max(output_end - output_begin, 0)
            source_begin = int(seq_starts[req_id].item()) if seq_starts is not None else 0

            for token in range(seq_len):
                source_pos = source_begin + token
                logical_block = source_pos // self.block_size
                block_offset = source_pos % self.block_size

                if logical_block >= block_table.size(1):
                    raise RuntimeError(f"Reference source block out of range: req_id={req_id}, source_pos={source_pos}, logical_block={logical_block}, blocks_per_req={block_table.size(1)}")

                physical_block = int(block_table[req_id, logical_block].item())

                dst_ref[output_begin + token, :FP8_DIM].copy_(fp8_bf16[physical_block, block_offset])

                src_token = src_cache[physical_block, block_offset]
                rope_src = src_token[ROPE_SRC_OFFSET:SRC_CACHE_DIM].view(torch.int32)
                rope_dst = dst_ref[output_begin + token, ROPE_DST_OFFSET:DST_DIM].view(torch.int32)
                rope_dst.copy_(rope_src)

        return dst_ref

    def _cosine_similarity(self, actual, expected):
        actual_f = actual.flatten().float()
        expected_f = expected.flatten().float()

        if not torch.isfinite(actual_f).all().item() or not torch.isfinite(expected_f).all().item():
            return float("nan")

        actual_norm = actual_f.norm()
        expected_norm = expected_f.norm()

        if actual_norm.item() == 0.0 and expected_norm.item() == 0.0:
            return 1.0

        denominator = actual_norm * expected_norm

        if denominator.item() == 0.0:
            return 0.0

        return float((actual_f.dot(expected_f) / denominator).item())

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        inputs = self._make_inputs(dev)
        src_cache, dst, block_table, workspace_starts, seq_starts, host_cache, host_row_ids, device_row_ids = inputs

        torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache(src_cache, dst, block_table, workspace_starts, self.batch_size, seq_starts, host_cache, host_row_ids, device_row_ids)

        torch.cuda.synchronize(dev)

        if dst.shape != (self.total_tokens, DST_DIM):
            raise RuntimeError(f"Unexpected dst shape: {dst.shape}, expected {(self.total_tokens, DST_DIM)}")

        if dst.dtype != torch.bfloat16:
            raise RuntimeError(f"Unexpected dst dtype: {dst.dtype}")

        if not dst.is_contiguous():
            raise RuntimeError("dst is not contiguous")

        actual_fp8 = dst[:, :FP8_DIM]

        if not torch.isfinite(actual_fp8.float()).all().item():
            bad_count = int((~torch.isfinite(actual_fp8.float())).sum().item())
            raise RuntimeError(f"FP8 output contains NaN or Inf: bad_count={bad_count}, fp8_value={self.fp8_value}")

        dst_ref = self._build_reference(src_cache, block_table, workspace_starts, seq_starts)
        expected_fp8 = dst_ref[:, :FP8_DIM]

        if not torch.equal(actual_fp8, expected_fp8):
            cosine = self._cosine_similarity(actual_fp8, expected_fp8)
            diff = float("nan") if not torch.isfinite(torch.tensor(cosine)).item() else max(0.0, 1.0 - cosine)
            return False, diff

        actual_rope_bytes = dst[:, ROPE_DST_OFFSET:DST_DIM].contiguous().view(torch.uint8)
        expected_rope_bytes = dst_ref[:, ROPE_DST_OFFSET:DST_DIM].contiguous().view(torch.uint8)

        if not torch.equal(actual_rope_bytes, expected_rope_bytes):
            cosine = self._cosine_similarity(actual_rope_bytes, expected_rope_bytes)
            diff = float("nan") if not torch.isfinite(torch.tensor(cosine)).item() else max(0.0, 1.0 - cosine)
            if diff == 0.0:
                diff = 1.0
            return False, diff

        return True, 0.0
