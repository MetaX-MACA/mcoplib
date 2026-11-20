
import torch
import mcoplib._moe_C

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


class Moe_prepare_scatter_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.n_tokens = config.get("n_tokens")
        self.topk = config.get("topk")
        self.n_expert = config.get("n_expert")
        self.expert_map_mode = config.get("expert_map_mode")
        self.samples = config.get("samples", 100)

        if self.expert_map_mode == "none":
            self.n_local_expert = self.n_expert
        elif self.expert_map_mode == "half":
            self.n_local_expert = self.n_expert // 2
        else:
            raise ValueError(f"Unsupported expert_map_mode: {self.expert_map_mode}")

        self.shape = (self.n_tokens, self.topk)
        self.total_routes = self.n_tokens * self.topk

    def _make_inputs(self, dev):
        n_tokens = self.n_tokens
        topk = self.topk
        n_expert = self.n_expert
        n_local_expert = self.n_local_expert

        topk_ids = torch.arange(n_tokens * topk, dtype=torch.int32, device=dev).remainder(n_expert).reshape(n_tokens, topk)

        token_expert_indices = torch.zeros((n_tokens, topk), dtype=torch.int32, device=dev)

        if self.expert_map_mode == "none":
            expert_map = None
        else:
            expert_map = torch.full((n_expert,), -1, dtype=torch.int32, device=dev)
            expert_map[:n_local_expert] = torch.arange(n_local_expert, dtype=torch.int32, device=dev)

        expert_first_token_offset = torch.empty((n_local_expert + 1,), dtype=torch.int64, device=dev)
        scatter_idx = torch.empty((n_tokens, topk), dtype=torch.int32, device=dev)
        sort_workspace = torch.empty((n_tokens * topk * 4 * 4,), dtype=torch.int8, device=dev)
        sorted_experts = torch.empty((n_tokens, topk), dtype=torch.int32, device=dev)
        sorted_rows = torch.empty((n_tokens, topk), dtype=torch.int32, device=dev)

        # 与官方单测保持一致
        topk_ids_for_sort = topk_ids.clone()

        return (topk_ids, token_expert_indices, expert_map, n_expert, n_local_expert, expert_first_token_offset, scatter_idx, sort_workspace, sorted_experts, sorted_rows, topk_ids_for_sort)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "int32/int64")
        state.add_summary("Shape", f"tokens={self.n_tokens} top_k={self.topk} experts={self.n_expert} local_experts={self.n_local_expert}")

        total_routes = self.total_routes

        input_bytes = total_routes * 4 * 2
        if self.expert_map_mode != "none":
            input_bytes += self.n_expert * 4

        output_bytes = (self.n_local_expert + 1) * 8 + total_routes * 4 * 5
        workspace_bytes = total_routes * 4 * 4

        state.add_element_count(total_routes)
        state.add_global_memory_reads(input_bytes + workspace_bytes)
        state.add_global_memory_writes(output_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            tensors = self._make_inputs(dev)

        return self.make_launcher(dev_id, torch.ops._moe_C.moe_prepare_scatter, *tensors)

    def _reference_offsets(self, topk_ids, expert_map, dev):
        ids = topk_ids.reshape(-1).to(torch.int64)
        valid_global = (ids >= 0) & (ids < self.n_expert)
        mapped_ids = torch.full_like(ids, -1)

        if expert_map is None:
            mapped_ids[valid_global] = ids[valid_global]
        else:
            expert_map = expert_map.to(device=dev, dtype=torch.int64)
            mapped_ids[valid_global] = expert_map[ids[valid_global]]

        valid_local = (mapped_ids >= 0) & (mapped_ids < self.n_local_expert)
        valid_ids = mapped_ids[valid_local]

        counts = torch.zeros(self.n_local_expert, dtype=torch.int64, device=dev)

        if valid_ids.numel() > 0:
            counts.scatter_add_(0, valid_ids, torch.ones_like(valid_ids, dtype=torch.int64))

        expected_offsets = torch.zeros(self.n_local_expert + 1, dtype=torch.int64, device=dev)
        expected_offsets[1:] = torch.cumsum(counts, dim=0)

        return expected_offsets

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        tensors = self._make_inputs(dev)

        (topk_ids, token_expert_indices, expert_map, n_expert, n_local_expert, expert_first_token_offset, scatter_idx, sort_workspace, sorted_experts, sorted_rows, topk_ids_for_sort) = tensors

        reference_ids = topk_ids.clone()

        torch.ops._moe_C.moe_prepare_scatter(*tensors)
        torch.cuda.synchronize()

        expected_offsets = self._reference_offsets(reference_ids, expert_map, dev)

        torch.testing.assert_close(expert_first_token_offset, expected_offsets, rtol=0, atol=0)

        assert expert_first_token_offset.shape == (n_local_expert + 1,)
        assert expert_first_token_offset.dtype == torch.int64

        assert scatter_idx.shape == reference_ids.shape
        assert scatter_idx.dtype == torch.int32

        assert sorted_experts.shape == reference_ids.shape
        assert sorted_experts.dtype == torch.int32

        assert sorted_rows.shape == reference_ids.shape
        assert sorted_rows.dtype == torch.int32

        assert topk_ids_for_sort.shape == reference_ids.shape
        assert topk_ids_for_sort.dtype == torch.int32

        return self.check_diff(expert_first_token_offset, expected_offsets)
