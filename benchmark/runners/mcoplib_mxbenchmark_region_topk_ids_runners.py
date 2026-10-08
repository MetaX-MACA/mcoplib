import torch

import mcoplib._C  # noqa: F401 - registers torch.ops._C operators
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


def _reference(logits, lengths, topk):
    rows, width = logits.shape
    output = torch.full(
        (rows, topk), -1, dtype=torch.int32, device=logits.device
    )
    for row in range(rows):
        visible = max(0, min(int(lengths[row].item()), width))
        budget = min(topk, visible)
        if budget:
            output[row, :budget] = torch.argsort(
                -logits[row, :visible], stable=True
            )[:budget].to(torch.int32)
    return output


class Region_topk_ids_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.rows = int(config.get("rows", 512))
        self.seq_regions = int(config.get("seq_regions", 4096))
        self.topk = int(config.get("topk", 512))
        if self.dtype != torch.float32:
            raise ValueError("region_topk_ids only supports dtype=float32")

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary(
            "Shape", f"({self.rows}x{self.seq_regions} topk={self.topk})"
        )
        score_elements = self.rows * self.seq_regions
        output_elements = self.rows * self.topk
        state.add_element_count(score_elements + output_elements)
        state.add_global_memory_reads((score_elements + self.rows) * 4)
        state.add_global_memory_writes(output_elements * 4)

    def _prepare(self, dev_id, rows=None, width=None, topk=None):
        rows = self.rows if rows is None else rows
        width = self.seq_regions if width is None else width
        topk = self.topk if topk is None else topk
        device = f"cuda:{dev_id}"
        generator = torch.Generator(device=device).manual_seed(20260911)
        logits = torch.randn(
            (rows, width), device=device, dtype=torch.float32, generator=generator
        )
        lengths = torch.full((rows,), width, device=device, dtype=torch.int32)
        return logits, lengths, topk

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            logits, lengths, topk = self._prepare(dev_id)
        return self.make_launcher(
            dev_id, torch.ops._C.region_topk_ids, logits, lengths, topk
        )

    def run_verification(self, dev_id):
        logits, lengths, topk = self._prepare(
            dev_id, rows=33, width=513, topk=min(self.topk, 257)
        )
        # Exercise clamping and deterministic tie selection during every benchmark.
        logits[:, :64] = 0.5
        lengths[0] = -1
        lengths[1] = 0
        lengths[2] = 255
        lengths[3] = 513 + 7

        actual = torch.ops._C.region_topk_ids(logits, lengths, topk)
        expected = _reference(logits, lengths, topk)
        torch.cuda.synchronize(dev_id)
        passed = torch.equal(
            torch.sort(actual, dim=1).values,
            torch.sort(expected, dim=1).values,
        )
        return bool(passed), 0.0 if passed else 1.0
