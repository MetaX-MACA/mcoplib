import torch
import torch.nn.functional as F
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


class Vocab_parallel_embedding_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 257)
        self.vocab_size = config.get("vocab_size", 128256)
        self.org_vocab_size = config.get("org_vocab_size", 128256 if self.vocab_size == 128256 else 32000)
        self.hidden_size = config.get("hidden_size", 2048)
        self.tp_size = config.get("tp_size", 2)
        self.tp_rank = config.get("tp_rank", 0)
        self.id_dtype = getattr(torch, config.get("id_dtype", "int32"))
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        self.pad = 64
        self.padded_vocab_size = self._pad_vocab_size(self.vocab_size)
        self.padded_org_vocab_size = self._pad_vocab_size(self.org_vocab_size)
        if self.padded_vocab_size % self.tp_size != 0:
            raise ValueError(f"padded_vocab_size={self.padded_vocab_size} must be divisible by tp_size={self.tp_size}")
        self.rows_per_rank = self.padded_vocab_size // self.tp_size
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError(f"tp_rank={self.tp_rank} must be in [0, {self.tp_size})")
        self.indices = self._shard_indices()
        self.op = self._get_op()

    def _get_op(self):
        if hasattr(torch.ops._C, "vocab_parallel_embedding"):
            return torch.ops._C.vocab_parallel_embedding
        if hasattr(torch.ops.mcoplib, "vocab_parallel_embedding"):
            return torch.ops.mcoplib.vocab_parallel_embedding
        raise RuntimeError("vocab_parallel_embedding operator is not registered")

    def _pad_vocab_size(self, vocab_size):
        return (vocab_size + self.pad - 1) // self.pad * self.pad

    def _shard_indices(self):
        org_start, org_end, num_org_padding, added_start, added_end, shard_start, shard_end, rows_per_rank = self._calculate_shard_indices()
        return org_start, org_end, num_org_padding, added_start, added_end

    def _calculate_shard_indices(self):
        padded_vocab_size = self._pad_vocab_size(self.vocab_size)
        padded_org_vocab_size = self._pad_vocab_size(self.org_vocab_size)
        if padded_vocab_size % self.tp_size != 0:
            raise ValueError(f"padded_vocab_size={padded_vocab_size} must be divisible by tp_size={self.tp_size}")
        num_org_vocab_padding = padded_org_vocab_size - self.org_vocab_size
        rows_per_rank = padded_vocab_size // self.tp_size
        shard_start = self.tp_rank * rows_per_rank
        shard_end = shard_start + rows_per_rank
        org_vocab_start_index = min(shard_start, self.org_vocab_size)
        org_vocab_end_index = min(shard_end, self.org_vocab_size)
        added_vocab_start_index = self.org_vocab_size + max(0, shard_start - padded_org_vocab_size)
        added_vocab_end_index = self.org_vocab_size + max(0, shard_end - padded_org_vocab_size)
        added_vocab_start_index = min(added_vocab_start_index, self.vocab_size)
        added_vocab_end_index = min(added_vocab_end_index, self.vocab_size)
        return org_vocab_start_index, org_vocab_end_index, num_org_vocab_padding, added_vocab_start_index, added_vocab_end_index, shard_start, shard_end, rows_per_rank

    def _make_inputs(self, dev_id):
        device = torch.device(f"cuda:{dev_id}")
        input_ids = torch.randint(0, self.vocab_size, (self.num_tokens,), device=device, dtype=self.id_dtype)
        weight = torch.randn((self.rows_per_rank, self.hidden_size), device=device, dtype=self.dtype)
        output = torch.empty((self.num_tokens, self.hidden_size), device=device, dtype=self.dtype)
        return output, input_ids, weight

    def _reference(self, input_ids, weight):
        org_start, org_end, num_org_padding, added_start, added_end = self.indices
        is_org = (input_ids >= org_start) & (input_ids < org_end)
        is_added = (input_ids >= added_start) & (input_ids < added_end)
        input_mask = ~(is_org | is_added)
        masked_input = torch.zeros_like(input_ids, dtype=torch.long)
        masked_input = torch.where(is_org, input_ids.long() - org_start, masked_input)
        masked_input = torch.where(is_added, input_ids.long() - added_start + (org_end - org_start) + num_org_padding, masked_input)
        return F.embedding(masked_input, weight).masked_fill(input_mask.unsqueeze(-1), 0)

    def _call_op(self, output, input_ids, weight):
        self.op(output, input_ids, weight, *self.indices)
        return output

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("Dtype", str(self.dtype))
        state.add_summary("Shape", f"tokens={self.num_tokens}, hidden={self.hidden_size}")
        state.add_summary("Vocab_Size", str(self.vocab_size))
        state.add_summary("Org_Vocab_Size", str(self.org_vocab_size))
        state.add_summary("TP_Size", str(self.tp_size))
        state.add_summary("TP_Rank", str(self.tp_rank))
        state.add_summary("ID_Dtype", str(self.id_dtype))
        state.add_element_count(self.num_tokens * self.hidden_size)
        state.add_global_memory_reads(self.num_tokens * torch.empty((), dtype=self.id_dtype).element_size() + self.rows_per_rank * self.hidden_size * torch.empty((), dtype=self.dtype).element_size())
        state.add_global_memory_writes(self.num_tokens * self.hidden_size * torch.empty((), dtype=self.dtype).element_size())

    def prepare_and_get_launcher(self, dev_id, tc_s):
        output, input_ids, weight = self._make_inputs(dev_id)
        return self.make_launcher(dev_id, self.op, output, input_ids, weight, *self.indices)

    def run_verification(self, dev_id):
        try:
            output, input_ids, weight = self._make_inputs(dev_id)
            self._call_op(output, input_ids, weight)
            output_ref = self._reference(input_ids, weight)
            torch.cuda.synchronize(dev_id)
            if not torch.equal(output, output_ref):
                max_diff = (output.float() - output_ref.float()).abs().max().item()
                return False, max_diff
            return True, 0.0
        except Exception:
            return False, float("inf")
