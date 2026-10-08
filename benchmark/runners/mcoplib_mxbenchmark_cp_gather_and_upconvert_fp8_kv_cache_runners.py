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
ROPE_BYTES = 128
DST_DIM = 576


class Cp_gather_and_upconvert_fp8_kv_cache_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.device_id = config.get("device_id", 0)
        self.batch_size = config.get("batch_size", 8)
        self.block_size = config.get("block_size", 16)
        self.seed = config.get("seed", 0)
        self.fp8_value = config.get("fp8_value", 0)

        self.seq_lens = [17, 31, 33, 7, 16, 24, 48, 65]
        self.source_starts = [0, 0, 0, 0, 0, 0, 0, 0]

        if self.batch_size != 8:
            raise ValueError(f"batch_size must be 8, got {self.batch_size}")

        if self.block_size != 16:
            raise ValueError(f"block_size must be 16, got {self.block_size}")

        if self.fp8_value < 0 or self.fp8_value > 255:
            raise ValueError(f"fp8_value must be in [0, 255], got {self.fp8_value}")

        self.total_tokens = sum(self.seq_lens)

        self.workspace_starts = [0]
        for seq_len in self.seq_lens[:-1]:
            self.workspace_starts.append(self.workspace_starts[-1] + seq_len)

        self.max_source_end = max(source_start + seq_len for source_start, seq_len in zip(self.seq_lens, self.source_starts))

        self.blocks_per_req = (self.max_source_end + self.block_size - 1) // self.block_size
        self.num_blocks = self.batch_size * self.blocks_per_req

        if self.total_tokens != 241:
            raise RuntimeError(f"Unexpected total_tokens={self.total_tokens}")

        if self.blocks_per_req != 5:
            raise RuntimeError(f"Unexpected blocks_per_req={self.blocks_per_req}")

        if self.num_blocks != 40:
            raise RuntimeError(f"Unexpected num_blocks={self.num_blocks}")

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

    def _make_src_cache(self, dev):
        src = torch.empty((self.num_blocks, self.block_size, SRC_CACHE_DIM), dtype=torch.uint8, device=dev)

        src[:, :, :512].fill_(self.fp8_value)

        scales = torch.ones((self.num_blocks, self.block_size, 4), dtype=torch.float32, device=dev)
        src[:, :, 512:528].copy_(scales.view(torch.uint8))

        rope = torch.arange(self.num_blocks * self.block_size * 32, dtype=torch.int32, device=dev).view(self.num_blocks, self.block_size, 32)
        src[:, :, 528:656].copy_(rope.view(torch.uint8))

        return src

    def _make_block_table(self, dev):
        num_blocks = self.batch_size * self.blocks_per_req

        physical_blocks = torch.arange(num_blocks, dtype=torch.int32, device=dev).view(self.batch_size, self.blocks_per_req)

        block_table = torch.empty_like(physical_blocks)

        for req_id in range(self.batch_size):
            block_table[req_id].copy_(physical_blocks[req_id])

        return block_table

    def _make_inputs(self, dev):
        torch.manual_seed(self.seed)

        src_cache = self._make_src_cache(dev)

        block_table = self._make_block_table(dev)

        workspace_starts = torch.tensor(self.workspace_starts, dtype=torch.int32, device=dev)

        dst = torch.zeros((self.total_tokens, DST_DIM), dtype=torch.bfloat16, device=dev)

        return src_cache, dst, block_table, workspace_starts

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"

            src_cache, dst, block_table, workspace_starts = self._make_inputs(dev)

        return self.make_launcher(dev_id, torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache, src_cache, dst, block_table, workspace_starts, self.batch_size, None)

    def _build_reference(self, src_cache, block_table, workspace_starts):
        dst_ref = torch.zeros((self.total_tokens, DST_DIM), dtype=torch.bfloat16, device=src_cache.device)

        for req_id in range(self.batch_size):
            output_begin = min(int(workspace_starts[req_id].item()), self.total_tokens)

            if req_id + 1 < self.batch_size:
                output_end = min(int(workspace_starts[req_id + 1].item()), self.total_tokens)
            else:
                output_end = self.total_tokens

            seq_len = max(output_end - output_begin, 0)

            source_begin = 0

            for token in range(seq_len):
                source_pos = source_begin + token

                logical_block = source_pos // self.block_size
                block_offset = source_pos % self.block_size

                physical_block = int(block_table[req_id, logical_block].item())

                src_token = src_cache[physical_block, block_offset]

                rope_src = src_token[ROPE_SRC_OFFSET:656].view(torch.int32)

                rope_dst = dst_ref[output_begin + token, ROPE_DST_OFFSET:DST_DIM].view(torch.int32)

                rope_dst.copy_(rope_src)

        return dst_ref

    def _cosine_similarity(self, actual, expected):
        actual_f = actual.flatten().float()
        expected_f = expected.flatten().float()

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

        src_cache, dst, block_table, workspace_starts = self._make_inputs(dev)

        torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache(src_cache, dst, block_table, workspace_starts, self.batch_size, None)

        torch.cuda.synchronize(dev)

        if dst.shape != (self.total_tokens, DST_DIM):
            return False, float("inf")

        if dst.dtype != torch.bfloat16:
            return False, float("inf")

        if not dst.is_contiguous():
            return False, float("inf")

        if not torch.isfinite(dst.float()).all():
            return False, float("inf")

        fp8_output = dst[:, :FP8_DIM]

        expected_fp8 = torch.zeros_like(fp8_output)

        if not torch.equal(fp8_output, expected_fp8):
            cosine = self._cosine_similarity(fp8_output, expected_fp8)
            return False, max(0.0, 1.0 - cosine)

        dst_ref = self._build_reference(src_cache, block_table, workspace_starts)

        actual_rope = dst[:, ROPE_DST_OFFSET:DST_DIM]
        expected_rope = dst_ref[:, ROPE_DST_OFFSET:DST_DIM]

        actual_rope_bytes = actual_rope.view(torch.uint8)
        expected_rope_bytes = expected_rope.view(torch.uint8)

        if not torch.equal(actual_rope_bytes, expected_rope_bytes):
            cosine = self._cosine_similarity(actual_rope_bytes, expected_rope_bytes)
            return False, max(0.0, 1.0 - cosine)

        return True, 0.0