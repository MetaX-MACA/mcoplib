import math
from typing import Dict, List, Tuple

import torch

from mcoplib.triton_sglang_chunk_delta_h import (
    _launch_chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fwd_h_baseline,
    prepare_chunk_offsets,
)
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


CHUNK_SIZE = 64
BLOCK_V = 64
NUM_WARPS = 4
NUM_STAGES = 1
COSINE_THRESHOLD = 0.9999
EXACT_8192_B4_CU_SEQLENS = [0, 824, 3896, 6971, 8192]


def _balanced_cu_seqlens(total_tokens: int, logical_batch: int) -> List[int]:
    """Return non-empty, nearly equal variable-length sequence offsets."""
    if logical_batch <= 0 or logical_batch > total_tokens:
        raise ValueError(
            f"logical_batch must be in [1, {total_tokens}], got {logical_batch}"
        )

    base, remainder = divmod(total_tokens, logical_batch)
    lengths = [base + (i < remainder) for i in range(logical_batch)]
    # Make neighboring sequence lengths non-uniform while preserving total T.
    for i in range(0, logical_batch - 1, 2):
        shift = min((i * 13 + 7) % CHUNK_SIZE, lengths[i + 1] - 1)
        lengths[i] += shift
        lengths[i + 1] -= shift

    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return offsets


def _case_cu_seqlens(total_tokens: int, logical_batch: int) -> List[int]:
    if total_tokens == 8192 and logical_batch == 4:
        return list(EXACT_8192_B4_CU_SEQLENS)
    return _balanced_cu_seqlens(total_tokens, logical_batch)


def _make_chunk_local_log_gates(
    total_tokens: int,
    num_heads: int,
    head_dim: int,
    cu_values: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Create production-like chunk-local cumulative non-positive log gates."""
    increments = -0.0125 * torch.rand(
        (1, total_tokens, num_heads, head_dim),
        device=device,
        dtype=torch.float32,
    )
    gates = torch.empty_like(increments)
    for sequence_start, sequence_end in zip(cu_values[:-1], cu_values[1:]):
        for chunk_start in range(sequence_start, sequence_end, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, sequence_end)
            gates[:, chunk_start:chunk_end] = torch.cumsum(
                increments[:, chunk_start:chunk_end], dim=1
            )
    return gates


class Triton_sglang_chunk_delta_h_runner(OpBenchmarkBase):
    """Benchmark the optimized preallocated chunk-delta hidden-state kernel."""

    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        shape = config.get("case", config)
        self.total_tokens = int(shape["total_tokens"])
        self.logical_batch = int(shape["logical_batch"])
        self.num_heads = int(shape["num_heads"])
        self.head_dim = int(config.get("head_dim", 128))
        self.key_head_ratio = int(config.get("key_head_ratio", 1))
        self.seed = int(config.get("seed", 20260910))

        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("dtype must be float16 or bfloat16")
        if self.head_dim != 128:
            raise ValueError(
                "This runner targets the optimized K=V=128 specialization"
            )
        if self.key_head_ratio <= 0 or self.num_heads % self.key_head_ratio:
            raise ValueError(
                "key_head_ratio must be positive and divide num_heads"
            )

        self.num_key_heads = self.num_heads // self.key_head_ratio
        self.cu_values = _case_cu_seqlens(
            self.total_tokens, self.logical_batch
        )
        self.total_chunks = sum(
            math.ceil((end - start) / CHUNK_SIZE)
            for start, end in zip(self.cu_values[:-1], self.cu_values[1:])
        )

    def _make_case(
        self,
        dev_id: int,
        total_tokens: int,
        logical_batch: int,
        num_heads: int,
        seed: int,
    ) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        device = torch.device(f"cuda:{dev_id}")
        cu_values = _case_cu_seqlens(total_tokens, logical_batch)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        scale = self.head_dim**-0.5
        key_heads = num_heads // self.key_head_ratio
        k = torch.randn(
            (1, total_tokens, key_heads, self.head_dim),
            device=device,
            dtype=self.dtype,
        ) * scale
        w = torch.randn(
            (1, total_tokens, num_heads, self.head_dim),
            device=device,
            dtype=self.dtype,
        ) * scale
        u = torch.randn(
            (1, total_tokens, num_heads, self.head_dim),
            device=device,
            dtype=self.dtype,
        )
        gk = _make_chunk_local_log_gates(
            total_tokens, num_heads, self.head_dim, cu_values, device
        )
        initial_state = torch.randn(
            (
                logical_batch,
                num_heads,
                self.head_dim,
                self.head_dim,
            ),
            device=device,
            dtype=self.dtype,
        ) * 0.01
        initial_state_indices = torch.arange(
            logical_batch, device=device, dtype=torch.int32
        )
        cu_seqlens = torch.tensor(
            cu_values, device=device, dtype=torch.int32
        )
        return (
            {
                "k": k,
                "w": w,
                "u": u,
                "gk": gk,
                "initial_state": initial_state,
                "initial_state_indices": initial_state_indices,
                "cu_seqlens": cu_seqlens,
            },
            cu_values,
        )

    def define_metrics(self, state):
        element_size = torch.empty((), dtype=self.dtype).element_size()
        gate_element_size = torch.empty(
            (), dtype=torch.float32
        ).element_size()
        value_tiles = math.ceil(self.head_dim / BLOCK_V)

        k_elements = self.total_tokens * self.num_key_heads * self.head_dim
        token_elements = self.total_tokens * self.num_heads * self.head_dim
        state_elements = (
            self.logical_batch
            * self.num_heads
            * self.head_dim
            * self.head_dim
        )
        chunk_state_elements = (
            self.total_chunks
            * self.num_heads
            * self.head_dim
            * self.head_dim
        )
        gate_endpoint_elements = (
            self.total_chunks * self.num_heads * self.head_dim
        )

        # CTA-issued tensor-byte model: K/W/GK are reread by every V tile.
        reads = (
            k_elements
            * self.key_head_ratio
            * value_tiles
            * element_size
            + token_elements * value_tiles * element_size
            + token_elements * element_size
            + gate_endpoint_elements * value_tiles * gate_element_size
            + state_elements * element_size
        )
        writes = (
            chunk_state_elements * element_size
            + token_elements * element_size
            + state_elements * element_size
        )

        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary(
            "Shape",
            f"B=1,T={self.total_tokens},N={self.logical_batch},"
            f"H={self.num_heads},Hg={self.num_key_heads},"
            f"K=V={self.head_dim},chunks={self.total_chunks}",
        )
        state.add_element_count(
            self.total_tokens * self.num_heads * self.head_dim
        )
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id: int, tc_s):
        with torch.cuda.stream(tc_s):
            case, _ = self._make_case(
                dev_id,
                self.total_tokens,
                self.logical_batch,
                self.num_heads,
                self.seed,
            )
            chunk_offsets = prepare_chunk_offsets(
                case["cu_seqlens"], CHUNK_SIZE
            )
            h = torch.empty(
                (
                    1,
                    self.total_chunks,
                    self.num_heads,
                    self.head_dim,
                    self.head_dim,
                ),
                device=case["k"].device,
                dtype=self.dtype,
            )
            v_new = torch.empty_like(case["u"])

        def launch_preallocated_kernel():
            # The state is intentionally recycled: reset/allocation is excluded
            # from the kernel-only latency reported by nvbench.
            _launch_chunk_gated_delta_rule_fwd_h(
                k=case["k"],
                w=case["w"],
                u=case["u"],
                h=h,
                v_new=v_new,
                gk=case["gk"],
                initial_state=case["initial_state"],
                initial_state_indices=case["initial_state_indices"],
                cu_seqlens=case["cu_seqlens"],
                chunk_offsets=chunk_offsets,
                logical_batch_size=self.logical_batch,
                total_chunks=self.total_chunks,
                block_v=BLOCK_V,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )

        return self.make_launcher(dev_id, launch_preallocated_kernel)

    @torch.inference_mode()
    def run_verification(self, dev_id: int):
        # Ragged lengths [76, 61] cover a full chunk and both tail paths.
        case, _ = self._make_case(
            dev_id,
            total_tokens=137,
            logical_batch=2,
            num_heads=self.num_heads,
            seed=self.seed + 1,
        )
        optimized_state = case["initial_state"].clone()
        baseline_state = case["initial_state"].clone()

        optimized_h, optimized_v = chunk_gated_delta_rule_fwd_h(
            k=case["k"],
            w=case["w"],
            u=case["u"],
            gk=case["gk"],
            initial_state=optimized_state,
            initial_state_indices=case["initial_state_indices"],
            cu_seqlens=case["cu_seqlens"],
        )
        baseline_h, baseline_v = chunk_gated_delta_rule_fwd_h_baseline(
            k=case["k"],
            w=case["w"],
            u=case["u"],
            gk=case["gk"],
            initial_state=baseline_state,
            initial_state_indices=case["initial_state_indices"],
            cu_seqlens=case["cu_seqlens"],
        )
        torch.cuda.synchronize(dev_id)

        comparisons = (
            self.check_diff(
                optimized_h, baseline_h, threshold=COSINE_THRESHOLD
            ),
            self.check_diff(
                optimized_v, baseline_v, threshold=COSINE_THRESHOLD
            ),
            self.check_diff(
                optimized_state,
                baseline_state,
                threshold=COSINE_THRESHOLD,
            ),
        )
        finite = all(
            bool(torch.isfinite(tensor).all().item())
            for tensor in (optimized_h, optimized_v, optimized_state)
        )
        passed = finite and all(result[0] for result in comparisons)
        max_cosine_distance = max(result[1] for result in comparisons)
        return passed, max_cosine_distance
