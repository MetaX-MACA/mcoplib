import torch

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
from mcoplib.triton_silu_mul_fp8_quant_deep_gemm import (
    persistent_masked_m_swiglu_mul_quant,
)
from runners.mcoplib_mxbenchmark_persistent_masked_m_silu_mul_quant_runners import (
    GROUP_SIZE,
    _collect_valid,
    _cosine_similarity,
    _make_input,
    _quantize_reference,
    _silu_bf16,
)


def _reference_valid(
    y,
    tokens_per_expert,
    limit,
    fp8_dtype,
    group_size,
):
    hidden = y.shape[-1] // 2
    q_parts = []
    s_parts = []

    limit_bf16 = torch.tensor(
        [limit],
        dtype=torch.bfloat16,
        device=y.device,
    )
    neg_limit_bf16 = torch.tensor(
        [-limit],
        dtype=torch.bfloat16,
        device=y.device,
    )

    for expert_id in range(y.shape[0]):
        valid_tokens = int(tokens_per_expert[expert_id].item())
        if valid_tokens == 0:
            continue

        gate = y[expert_id, :valid_tokens, :hidden]
        up = y[expert_id, :valid_tokens, hidden:]

        gate_activation = torch.minimum(
            _silu_bf16(gate),
            limit_bf16,
        )
        up_activation = torch.clamp(
            up,
            min=neg_limit_bf16,
            max=limit_bf16,
        )

        activation = (
            gate_activation * up_activation
        ).to(torch.bfloat16)

        q_ref, s_ref = _quantize_reference(
            activation,
            fp8_dtype,
            group_size,
        )
        q_parts.append(q_ref)
        s_parts.append(s_ref)

    return torch.cat(q_parts, dim=0), torch.cat(s_parts, dim=0)


class Persistent_masked_m_swiglu_mul_quant_runner(OpBenchmarkBase):
    _force_sync = True
    def __init__(self, name, config):
        super().__init__(name, config)

        self.num_experts = config.get("num_experts", 8)
        self.max_tokens_per_expert = config.get(
            "max_tokens_per_expert", 16
        )
        self.hidden_size = config.get("hidden_size", 1024)
        self.group_size = config.get("group_size", GROUP_SIZE)
        self.num_parallel_tokens = config.get(
            "num_parallel_tokens", 16
        )
        self.quant_scale_fmt = config.get(
            "quant_scale_fmt", "float32"
        ).lower()
        self.limit = float(config.get("limit", 7.0))
        self.seed = config.get("seed", 42)
        self.cos_threshold = config.get("cos_threshold", 0.999)
        self.scale_threshold = config.get("scale_threshold", 0.999)

        if self.dtype != torch.bfloat16:
            raise ValueError("dtype must be bfloat16")

        if self.group_size != GROUP_SIZE:
            raise ValueError("group_size must be 128")

        if self.hidden_size % self.group_size != 0:
            raise ValueError(
                "hidden_size must be divisible by group_size"
            )

        if self.quant_scale_fmt not in ("float32", "fp32"):
            raise ValueError(
                'only quant_scale_fmt="float32" is supported'
            )

        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")

        if self.max_tokens_per_expert <= 0:
            raise ValueError(
                "max_tokens_per_expert must be positive"
            )

        if self.limit <= 0:
            raise ValueError("limit must be positive")

        self.groups = self.hidden_size // self.group_size

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "bfloat16->fp8")
        state.add_summary(
            "Shape",
            (
                f"({self.num_experts} "
                f"{self.max_tokens_per_expert} "
                f"{self.hidden_size * 2}) -> "
                f"({self.num_experts} "
                f"{self.max_tokens_per_expert} "
                f"{self.hidden_size})"
            ),
        )

        output_elements = (
            self.num_experts
            * self.max_tokens_per_expert
            * self.hidden_size
        )

        state.add_element_count(output_elements)

        # 输入 BF16 E*T*2H。
        input_bytes = output_elements * 2 * 2

        # tokens_per_expert int32。
        count_bytes = self.num_experts * 4

        # 1 元素 BF16 limit。
        limit_bytes = 2

        # FP8 输出。
        output_q_bytes = output_elements

        # FP32 scales。
        output_s_bytes = (
            self.num_experts
            * self.max_tokens_per_expert
            * self.groups
            * 4
        )

        state.add_global_memory_reads(
            input_bytes + count_bytes + limit_bytes
        )
        state.add_global_memory_writes(
            output_q_bytes + output_s_bytes
        )

    def _prepare(self, dev_id, verification=False):
        device = f"cuda:{dev_id}"

        if verification:
            experts = min(self.num_experts, 4)
            tokens = min(self.max_tokens_per_expert, 16)
            hidden = min(self.hidden_size, 512)
            hidden = max(hidden, self.group_size)
            hidden = (
                hidden // self.group_size
            ) * self.group_size
            wide_range = True
            seed = self.seed + 1
        else:
            experts = self.num_experts
            tokens = self.max_tokens_per_expert
            hidden = self.hidden_size
            wide_range = False
            seed = self.seed

        y = _make_input(
            (experts, tokens, hidden * 2),
            self.dtype,
            device,
            seed,
            wide_range=wide_range,
        )

        if verification:
            pattern = [
                tokens,
                max(tokens - 1, 0),
                tokens // 2,
                0,
            ]
            counts = [
                pattern[i % len(pattern)] for i in range(experts)
            ]
            tokens_per_expert = torch.tensor(
                counts,
                dtype=torch.int32,
                device=device,
            )
        else:
            tokens_per_expert = torch.full(
                (experts,),
                tokens,
                dtype=torch.int32,
                device=device,
            )

        limit_tensor = torch.tensor(
            [self.limit],
            dtype=self.dtype,
            device=device,
        )

        return y, tokens_per_expert, limit_tensor

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            y, tokens_per_expert, limit_tensor = self._prepare(
                dev_id
            )

        # quant_scale_fmt 使用函数默认值 FLOAT32。
        return self.make_launcher(
            dev_id,
            persistent_masked_m_swiglu_mul_quant,
            y,
            tokens_per_expert,
            limit_tensor,
            self.num_parallel_tokens,
            self.group_size,
        )

    @torch.inference_mode()
    def run_verification(self, dev_id):
        y, tokens_per_expert, limit_tensor = self._prepare(
            dev_id,
            verification=True,
        )

        output_q, output_s = (
            persistent_masked_m_swiglu_mul_quant(
                y,
                tokens_per_expert,
                limit_tensor,
                self.num_parallel_tokens,
                self.group_size,
            )
        )
        torch.cuda.synchronize()

        q_actual, s_actual = _collect_valid(
            output_q,
            output_s,
            tokens_per_expert,
        )

        q_ref, s_ref = _reference_valid(
            y,
            tokens_per_expert,
            self.limit,
            output_q.dtype,
            self.group_size,
        )

        q_cos = _cosine_similarity(q_actual, q_ref)
        s_cos = _cosine_similarity(s_actual, s_ref)

        passed = (
            q_cos >= self.cos_threshold
            and s_cos >= self.scale_threshold
        )
        diff = max(1.0 - q_cos, 1.0 - s_cos)

        return passed, diff
