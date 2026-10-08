import torch

from mcoplib.op import fused_silu_mul_dq_mask_quant
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


INT8_SCALE_FACTOR = 0.0078740157


def _get_dtype(dtype_name):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    try:
        return dtype_map[dtype_name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {dtype_name}") from exc


class Fused_silu_mul_dq_mask_quant_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.case_config = config.get("case_config")
        if not isinstance(self.case_config, dict):
            raise ValueError("case_config must be an expanded case dictionary")

        self.case_name = self.case_config["name"]
        self.num_experts = self.case_config["num_experts"]
        self.hidden_size = self.case_config["hidden_size"]
        self.mask_values = self.case_config["mask_values"]
        self.dtype = _get_dtype(self.case_config["input_dtype"])
        self.mask_dtype = _get_dtype(self.case_config["mask_dtype"])
        self.with_weight = self.case_config["with_weight"]

        self.token_capacity = config.get("token_capacity", 12)
        self.swiglu_limit = config.get("swiglu_limit", 0.0)
        self.gemm1_alpha = config.get("gemm1_alpha", 1.702)
        self.gemm1_limit = config.get("gemm1_limit", 7.0)

        if len(self.mask_values) != self.num_experts:
            raise ValueError("mask_values length must equal num_experts")
        if any(
            valid_tokens < 0 or valid_tokens > self.token_capacity
            for valid_tokens in self.mask_values
        ):
            raise ValueError("each mask value must be within token_capacity")
        if self.hidden_size % 8 != 0:
            raise ValueError("hidden_size must be divisible by 8")
        if self.gemm1_limit <= 0.0:
            raise ValueError("GPT-OSS benchmark requires gemm1_limit > 0")
        if self.swiglu_limit > 0.0:
            raise ValueError(
                "swiglu_limit and gemm1_limit cannot both be enabled"
            )

        self.valid_tokens = sum(self.mask_values)
        self.input_hidden_size = self.hidden_size * 2
        self.out_stride = (
            self.input_hidden_size // 4 + 257
        ) // 256 * 256

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary(
            "dtype",
            (
                f"{str(self.dtype).removeprefix('torch.')}/"
                f"{str(self.mask_dtype).removeprefix('torch.')}"
            ),
        )
        state.add_summary(
            "Shape",
            (
                f"({self.case_name} Experts:{self.num_experts} "
                f"Capacity:{self.token_capacity} Hidden:{self.hidden_size} "
                f"Mask:{'x'.join(map(str, self.mask_values))} "
                f"Weight:{self.with_weight})"
            ),
        )

        input_element_size = torch.empty(
            (), dtype=self.dtype
        ).element_size()
        mask_element_size = torch.empty(
            (), dtype=self.mask_dtype
        ).element_size()

        read_bytes = (
            self.valid_tokens
            * self.input_hidden_size
            * input_element_size
        )
        read_bytes += self.num_experts * mask_element_size
        if self.with_weight:
            read_bytes += (
                self.valid_tokens
                * self.hidden_size
                * input_element_size
            )

        write_bytes = self.valid_tokens * (self.hidden_size + 4)

        state.add_element_count(self.valid_tokens * self.hidden_size)
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def _make_inputs(self, dev, seed):
        torch.manual_seed(seed)
        input_fp32 = torch.randn(
            self.num_experts,
            self.token_capacity,
            self.input_hidden_size,
            device=dev,
            dtype=torch.float32,
        ) * 3.0
        input_fp32[..., 0] = self.gemm1_limit + 3.0
        input_fp32[..., 1] = -self.gemm1_limit - 3.0
        input_fp32[..., self.hidden_size] = self.gemm1_limit + 4.0
        input_fp32[..., self.hidden_size + 1] = (
            -self.gemm1_limit - 4.0
        )
        input_tensor = input_fp32.to(self.dtype)

        mask = torch.tensor(
            self.mask_values, device=dev, dtype=self.mask_dtype
        )
        weight = None
        if self.with_weight:
            weight = (
                torch.randn(
                    self.hidden_size, device=dev, dtype=torch.float32
                )
                * 0.25
                + 1.0
            ).to(self.dtype)

        packed_output = torch.empty(
            self.num_experts,
            self.token_capacity,
            self.out_stride,
            device=dev,
            dtype=self.dtype,
        )
        return packed_output, input_tensor, mask, weight

    def _activation_reference(self, input_tensor, weight):
        gate = input_tensor[..., :self.hidden_size].float()
        up = input_tensor[..., self.hidden_size:].float()
        gate = gate.clamp(max=self.gemm1_limit)
        up = up.clamp(
            min=-self.gemm1_limit, max=self.gemm1_limit
        )
        output = gate * torch.sigmoid(gate * self.gemm1_alpha)
        output = output * (up + 1.0)
        if weight is not None:
            output = output * weight.float()
        return output

    @staticmethod
    def _quant_reference(activation):
        absmax = activation.abs().amax(dim=-1)
        if torch.any(absmax == 0):
            raise ValueError("reference contains an all-zero token")
        scale = absmax * INT8_SCALE_FACTOR
        quant = torch.round(
            activation * (127.0 / absmax).unsqueeze(-1)
        )
        quant = quant.clamp(-127, 127).to(torch.int8)
        return quant, scale

    def _unpack_output(self, packed_output):
        output_bytes = packed_output.view(torch.uint8)
        quant = output_bytes[..., :self.hidden_size].contiguous()
        quant = quant.view(torch.int8)
        scale = output_bytes[
            ..., self.hidden_size:self.hidden_size + 4
        ].contiguous()
        scale = scale.view(torch.float32).squeeze(-1)
        return quant, scale

    def _gather_valid(self, tensor):
        return torch.cat(
            [
                tensor[expert_id, :valid_tokens]
                for expert_id, valid_tokens in enumerate(self.mask_values)
            ],
            dim=0,
        )

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            packed_output, input_tensor, mask, weight = self._make_inputs(
                dev, seed=20260805
            )

        return self.make_launcher(
            dev_id,
            fused_silu_mul_dq_mask_quant,
            packed_output,
            input_tensor,
            mask,
            self.swiglu_limit,
            weight,
            self.gemm1_alpha,
            self.gemm1_limit,
        )

    @torch.inference_mode()
    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        packed_output, input_tensor, mask, weight = self._make_inputs(
            dev, seed=20260805
        )

        fused_silu_mul_dq_mask_quant(
            packed_output,
            input_tensor,
            mask,
            self.swiglu_limit,
            weight,
            self.gemm1_alpha,
            self.gemm1_limit,
        )
        torch.cuda.synchronize()

        activation_ref = self._activation_reference(input_tensor, weight)
        quant_ref, scale_ref = self._quant_reference(activation_ref)
        quant_actual, scale_actual = self._unpack_output(packed_output)

        quant_ref = self._gather_valid(quant_ref)
        scale_ref = self._gather_valid(scale_ref)
        activation_ref = self._gather_valid(activation_ref)
        quant_actual = self._gather_valid(quant_actual)
        scale_actual = self._gather_valid(scale_actual)

        pass_quant, diff_quant = self.check_diff(
            quant_actual, quant_ref, threshold=0.999
        )
        pass_scale, diff_scale = self.check_diff(
            scale_actual, scale_ref, threshold=0.999999
        )
        dequant_actual = (
            quant_actual.float() * scale_actual.unsqueeze(-1)
        )
        pass_dequant, diff_dequant = self.check_diff(
            dequant_actual, activation_ref, threshold=0.999
        )

        max_quant_diff = (
            quant_actual.to(torch.int16) - quant_ref.to(torch.int16)
        ).abs().max()
        quant_within_tolerance = max_quant_diff.item() <= 1
        scale_within_tolerance = torch.allclose(
            scale_actual, scale_ref, rtol=2e-3, atol=1e-5
        )

        passed = (
            pass_quant
            and pass_scale
            and pass_dequant
            and quant_within_tolerance
            and scale_within_tolerance
        )
        return passed, max(diff_quant, diff_scale, diff_dequant)
