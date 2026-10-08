# SPDX-License-Identifier: Apache-2.0

import math

import torch

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._moe_C
except ImportError as e:
    raise RuntimeError(f"Failed to import mcoplib._moe_C: {e}") from e


SENTINEL_EXPERT = -2
SENTINEL_TOKEN = -7
SENTINEL_NPAD = -13


def round_up(x, base):
    return ((x + base - 1) // base) * base


def ceil_div(x, y):
    return (x + y - 1) // y


class Moe_lora_align_block_size_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.device_id = config.get("device_id", 0)
        self.num_tokens = config.get("num_tokens", 262144)
        self.num_experts = config.get("num_experts", 128)
        self.block_size = config.get("block_size", 16)
        self.max_loras = config.get("max_loras", 64)
        self.top_k = config.get("top_k", 6)
        self.index_dtype = torch.int32
        self.seed = config.get("seed", 1)

        if self.num_tokens <= 0:
            raise ValueError(f"num_tokens must be > 0, got {self.num_tokens}")

        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {self.num_experts}")

        if self.max_loras <= 0:
            raise ValueError(f"max_loras must be > 0, got {self.max_loras}")

        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError(f"top_k must be in [1, {self.num_experts}], got {self.top_k}")

        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")

        if not hasattr(torch.ops._moe_C, "moe_lora_align_block_size"):
            raise RuntimeError("torch.ops._moe_C.moe_lora_align_block_size is not registered")

    def _get_output_shape(self):
        total_elements = self.num_tokens * self.top_k

        max_num_tokens_padded = total_elements + self.num_experts * (self.block_size - 1)
        max_num_tokens_padded = round_up(max_num_tokens_padded, self.block_size)

        if total_elements < self.num_experts:
            max_num_tokens_padded = total_elements * self.block_size

        max_num_m_blocks = ceil_div(max_num_tokens_padded, self.block_size)

        return total_elements, max_num_tokens_padded, max_num_m_blocks

    def _make_lora_ids(self, dev):
        lora_ids = torch.full((self.max_loras + 1,), -1, dtype=self.index_dtype, device=dev)
        lora_ids[1:self.max_loras + 1] = torch.arange(self.max_loras, dtype=self.index_dtype, device=dev)
        return lora_ids

    def _make_inputs(self, dev):
        torch.manual_seed(self.seed)

        total_elements, max_num_tokens_padded, max_num_m_blocks = self._get_output_shape()

        topk_ids = torch.randint(0, self.num_experts, (self.num_tokens, self.top_k), dtype=self.index_dtype, device=dev)

        token_lora_mapping = torch.randint(0, self.max_loras, (self.num_tokens,), dtype=self.index_dtype, device=dev)

        adapter_enabled = torch.ones((self.max_loras + 1,), dtype=self.index_dtype, device=dev)

        lora_ids = self._make_lora_ids(dev)

        sorted_token_ids = torch.full((self.max_loras * max_num_tokens_padded,), total_elements, dtype=self.index_dtype, device=dev)

        expert_ids = torch.full((self.max_loras * max_num_m_blocks,), -1, dtype=self.index_dtype, device=dev)

        num_tokens_post_pad = torch.zeros((self.max_loras,), dtype=self.index_dtype, device=dev)

        return topk_ids, token_lora_mapping, sorted_token_ids, expert_ids, num_tokens_post_pad, adapter_enabled, lora_ids, total_elements, max_num_tokens_padded, max_num_m_blocks

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "int32")
        state.add_summary("Shape", f"({self.num_tokens} {self.num_experts} {self.max_loras})")

        total_elements, max_num_tokens_padded, max_num_m_blocks = self._get_output_shape()

        state.add_element_count(total_elements)

        topk_ids_bytes = total_elements * 4
        token_lora_mapping_bytes = self.num_tokens * 4
        adapter_enabled_bytes = (self.max_loras + 1) * 4
        lora_ids_bytes = (self.max_loras + 1) * 4

        sorted_token_ids_bytes = self.max_loras * max_num_tokens_padded * 4
        expert_ids_bytes = self.max_loras * max_num_m_blocks * 4
        num_tokens_post_pad_bytes = self.max_loras * 4

        state.add_global_memory_reads(topk_ids_bytes + token_lora_mapping_bytes + adapter_enabled_bytes + lora_ids_bytes)
        state.add_global_memory_writes(sorted_token_ids_bytes + expert_ids_bytes + num_tokens_post_pad_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"

            topk_ids, token_lora_mapping, sorted_token_ids, expert_ids, num_tokens_post_pad, adapter_enabled, lora_ids, total_elements, max_num_tokens_padded, max_num_m_blocks = self._make_inputs(dev)

        return self.make_launcher(dev_id, torch.ops._moe_C.moe_lora_align_block_size, topk_ids, token_lora_mapping, self.num_experts, self.block_size, self.max_loras, max_num_tokens_padded, max_num_m_blocks, sorted_token_ids, expert_ids, num_tokens_post_pad, adapter_enabled, lora_ids, None)

    def _verify_output(self, topk_ids, token_lora_mapping, sorted_token_ids, expert_ids, num_tokens_post_pad, adapter_enabled, lora_ids, total_elements, max_num_tokens_padded, max_num_m_blocks):
        topk_cpu = topk_ids.cpu()
        mapping_cpu = token_lora_mapping.cpu()
        sorted_cpu = sorted_token_ids.cpu()
        expert_cpu = expert_ids.cpu()
        post_pad_cpu = num_tokens_post_pad.cpu()
        adapter_cpu = adapter_enabled.cpu()
        lora_ids_cpu = lora_ids.cpu()

        topk_flat = topk_cpu.flatten()
        mapping_expanded = mapping_cpu.repeat_interleave(self.top_k)

        for slot in range(self.max_loras):
            lid = int(lora_ids_cpu[slot + 1].item())

            if lid < 0 or lid >= self.max_loras:
                return False, float("inf")

            if int(adapter_cpu[lid + 1].item()) == 0:
                if int(post_pad_cpu[slot].item()) != 0:
                    return False, float("inf")

                if not torch.all(expert_cpu[slot] == SENTINEL_EXPERT):
                    return False, float("inf")

                if not torch.all(sorted_cpu[slot] == SENTINEL_TOKEN):
                    return False, float("inf")

                continue

            mapping_mask = mapping_cpu == lid
            expected_topk = topk_cpu[mapping_mask].flatten()

            expert_counts = torch.bincount(expected_topk, minlength=self.num_experts)
            expected_post_pad = int(round_up(expert_counts, self.block_size).sum().item())
            actual_post_pad = int(post_pad_cpu[slot].item())

            if actual_post_pad != expected_post_pad:
                return False, float("inf")

            if actual_post_pad < 0 or actual_post_pad > max_num_tokens_padded:
                return False, float("inf")

            if actual_post_pad % self.block_size != 0:
                return False, float("inf")

            num_blocks = actual_post_pad // self.block_size

            if expected_topk.numel() == 0:
                if actual_post_pad != 0:
                    return False, float("inf")

                if not torch.all(expert_cpu[slot] == -1):
                    return False, float("inf")

                if not torch.all(sorted_cpu[slot] == total_elements):
                    return False, float("inf")

                continue

            for block_idx in range(num_blocks):
                start = block_idx * self.block_size
                end = start + self.block_size

                expert_id = int(expert_cpu[slot, block_idx].item())

                if expert_id < 0 or expert_id >= self.num_experts:
                    return False, float("inf")

                block_tokens = sorted_cpu[slot, start:end]

                valid_tokens = block_tokens[block_tokens != total_elements]

                if valid_tokens.numel() == 0:
                    continue

                if torch.any(valid_tokens < 0):
                    return False, float("inf")

                if torch.any(valid_tokens >= total_elements):
                    return False, float("inf")

                if not torch.all(topk_flat[valid_tokens] == expert_id):
                    return False, float("inf")

            if num_blocks < max_num_m_blocks:
                if not torch.all(expert_cpu[slot, num_blocks:] == -1):
                    return False, float("inf")

            used_tokens = sorted_cpu[slot, :actual_post_pad]

            if not torch.all((used_tokens >= 0) & (used_tokens <= total_elements)):
                return False, float("inf")

            if actual_post_pad < max_num_tokens_padded:
                tail = sorted_cpu[slot, actual_post_pad:]

                if not torch.all(tail == total_elements):
                    return False, float("inf")

        return True, 0.0

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        torch.manual_seed(self.seed)

        topk_ids, token_lora_mapping, sorted_token_ids, expert_ids, num_tokens_post_pad, adapter_enabled, lora_ids, total_elements, max_num_tokens_padded, max_num_m_blocks = self._make_inputs(dev)

        torch.ops._moe_C.moe_lora_align_block_size(topk_ids, token_lora_mapping, self.num_experts, self.block_size, self.max_loras, max_num_tokens_padded, max_num_m_blocks, sorted_token_ids, expert_ids, num_tokens_post_pad, adapter_enabled, lora_ids, None)

        torch.cuda.synchronize(dev)

        passed, diff = self._verify_output(topk_ids, token_lora_mapping, sorted_token_ids.view(self.max_loras, max_num_tokens_padded), expert_ids.view(self.max_loras, max_num_m_blocks), num_tokens_post_pad, adapter_enabled, lora_ids, total_elements, max_num_tokens_padded, max_num_m_blocks)

        if passed:
            return True, 0.0

        return False, diff