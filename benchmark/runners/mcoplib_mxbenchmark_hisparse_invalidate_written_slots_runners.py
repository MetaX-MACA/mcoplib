import torch
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


class Hisparse_invalidate_written_slots_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_states = config.get("num_states", 8)
        self.region_stride = config.get("region_stride", 32)
        self.num_tokens = config.get("num_tokens", 4)
        self.num_requests = config.get("num_requests", 4)
        self.args = None

        if self.num_states < 0:
            raise ValueError("num_states must be non-negative")
        if self.region_stride < 0:
            raise ValueError("region_stride must be non-negative")
        if self.num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if self.num_requests < 0:
            raise ValueError("num_requests must be non-negative")

    def define_metrics(self, state):
        num_states = self.num_states
        region_stride = self.region_stride
        num_tokens = self.num_tokens
        num_requests = self.num_requests

        state.add_summary("Op", self.name)
        state.add_summary("dtype", "int32/int64")
        state.add_summary("Shape", f"states={num_states} stride={region_stride} tokens={num_tokens} requests={num_requests}")
        state.add_element_count(num_states * region_stride)
        state.add_global_memory_reads(num_states * region_stride * 4 + num_requests * 4 + num_tokens * 12)
        state.add_global_memory_writes(num_states * region_stride * 4)

    def _build_inputs(self, dev_id):
        device = torch.device("cuda", dev_id)
        num_states = self.num_states
        region_stride = self.region_stride
        num_tokens = self.num_tokens
        num_requests = self.num_requests

        device_global_indices = torch.arange(num_states * region_stride, dtype=torch.int32, device=device).reshape(num_states, region_stride).contiguous()
        request_state_indices = torch.arange(num_requests, dtype=torch.int32, device=device).contiguous()
        req_id_per_token = torch.arange(num_tokens, dtype=torch.int32, device=device).remainder(max(num_requests, 1)).contiguous()
        written_slots = torch.arange(num_tokens, dtype=torch.int64, device=device).remainder(max(region_stride, 1)).contiguous()

        return device_global_indices, request_state_indices, req_id_per_token, written_slots

    def prepare_and_get_launcher(self, dev_id, tc_s):
        self.args = self._build_inputs(dev_id)
        return self.make_launcher(dev_id, torch.ops._C.hisparse_invalidate_written_slots, *self.args)

    def run_verification(self, dev_id):
        if not hasattr(torch.ops._C, "hisparse_invalidate_written_slots"):
            raise RuntimeError("hisparse_invalidate_written_slots is not registered")

        args = self._build_inputs(dev_id)
        device_global_indices, request_state_indices, req_id_per_token, written_slots = args
        expected = device_global_indices.clone()

        indices_cpu = device_global_indices.cpu()
        request_states_cpu = request_state_indices.cpu()
        req_ids_cpu = req_id_per_token.cpu()
        slots_cpu = written_slots.cpu()

        num_states = indices_cpu.shape[0]
        region_stride = indices_cpu.shape[1]
        num_requests = request_states_cpu.numel()

        for token in range(req_ids_cpu.numel()):
            req_id = int(req_ids_cpu[token].item())
            slot = int(slots_cpu[token].item())

            if req_id < 0 or req_id >= num_requests:
                continue
            if slot < 0 or slot >= region_stride:
                continue

            state_idx = int(request_states_cpu[req_id].item())
            if state_idx < 0 or state_idx >= num_states:
                continue

            expected[state_idx][indices_cpu[state_idx] == slot] = -1

        try:
            torch.ops._C.hisparse_invalidate_written_slots(*args)
            torch.cuda.synchronize(dev_id)
        except Exception as exc:
            raise RuntimeError(f"hisparse_invalidate_written_slots failed: num_states={self.num_states}, region_stride={self.region_stride}, num_tokens={self.num_tokens}, num_requests={self.num_requests}: {exc}") from exc

        if not torch.equal(device_global_indices, expected):
            mismatch = torch.nonzero(device_global_indices != expected, as_tuple=False)
            pos = mismatch[0].tolist()
            actual_value = device_global_indices[tuple(pos)].item()
            expected_value = expected[tuple(pos)].item()
            raise AssertionError(f"hisparse_invalidate_written_slots verification failed: first_mismatch={pos}, actual={actual_value}, expected={expected_value}")

        return True, 0.0