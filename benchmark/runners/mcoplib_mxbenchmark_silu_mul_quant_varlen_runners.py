import torch

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel  # noqa: F401 - registers torch.ops.sgl_kernel
except ImportError:
    pass


FP8_MAX = 448.0
WORK_TABLE_HEADER_INT32 = 2
WORK_ITEM_INT32 = 2


def _balanced_masked_m(experts, tokens, active_routes, device):
    base, remainder = divmod(active_routes, experts)
    counts = torch.full((experts,), base, dtype=torch.int32)
    counts[:remainder] += 1
    if int(counts.max()) > tokens:
        raise ValueError("active_routes cannot fit into [E, T]")
    return counts.to(device=device)


def _reference_valid(input_tensor, masked_m, group_size, swiglu_limit):
    hidden = input_tensor.size(-1) // 2
    groups = hidden // group_size
    q_parts = []
    scale_parts = []

    limit = torch.tensor(
        swiglu_limit,
        dtype=torch.bfloat16,
        device=input_tensor.device,
    )
    for expert, count in enumerate(masked_m.cpu().tolist()):
        if count == 0:
            continue

        valid_input = input_tensor[expert, :count]
        gate = torch.minimum(valid_input[:, :hidden], limit)
        up = torch.clamp(valid_input[:, hidden:], min=-limit, max=limit)
        gate_f = gate.float()
        values = gate_f / (1.0 + torch.exp(-gate_f)) * up.float()

        grouped = values.view(count, groups, group_size)
        scales = grouped.abs().amax(dim=-1).clamp_min(1.0e-10) / FP8_MAX
        quantized = torch.clamp(
            grouped / scales.unsqueeze(-1),
            -FP8_MAX,
            FP8_MAX,
        ).to(torch.float8_e4m3fn)
        q_parts.append(quantized.view(count, hidden))
        scale_parts.append(scales.float())

    return torch.cat(q_parts), torch.cat(scale_parts)


class Silu_mul_quant_varlen_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_experts = int(config.get("num_experts", 16))
        self.num_tokens = int(config.get("num_tokens", 2048))
        self.hidden_size = int(config.get("hidden_size", 2048))
        self.topk = int(config.get("topk", 6))
        self.active_routes = int(config.get("active_routes", 768))
        self.group_size = int(config.get("group_size", 128))
        self.swiglu_limit = float(config.get("swiglu_limit", 10.0))
        self.scale_ue8m0 = bool(config.get("scale_ue8m0", False))
        self.transposed = bool(config.get("transposed", False))
        self.swizzle = bool(config.get("swizzle", False))
        self.enable_pdl = bool(config.get("enable_pdl", False))
        self.use_workspace = bool(config.get("use_workspace", False))
        self.persistent_grid = int(config.get("persistent_grid", 0))
        self.output_threshold = float(config.get("output_threshold", 0.9999))
        self.scale_threshold = float(config.get("scale_threshold", 0.999999))
        self.seed = int(config.get("seed", 7))

        if self.dtype != torch.bfloat16:
            raise ValueError("dtype must be bfloat16")
        if self.group_size != 128:
            raise ValueError("group_size must be 128")
        if not 1 <= self.num_experts <= 256:
            raise ValueError("num_experts must be in [1, 256]")
        if self.num_tokens <= 0 or self.topk <= 0:
            raise ValueError("num_tokens and topk must be positive")
        if self.hidden_size <= 0 or self.hidden_size % 256 != 0:
            raise ValueError("hidden_size must be positive and divisible by 256")
        if self.hidden_size // 8 < self.num_experts:
            raise ValueError("hidden_size / 8 must be at least num_experts")
        if self.hidden_size // 8 > 1024:
            raise ValueError("hidden_size / 8 must not exceed 1024")
        max_routes = min(
            self.num_experts * self.num_tokens,
            self.num_tokens * self.topk,
        )
        if not 0 < self.active_routes <= max_routes:
            raise ValueError(
                "active_routes must be in (0, min(E*T, T*topk)]"
            )
        if self.swiglu_limit <= 0:
            raise ValueError("swiglu_limit must be positive")
        if self.scale_ue8m0 or self.transposed or self.swizzle:
            raise ValueError(
                "this benchmark currently covers the default non-transposed, "
                "non-swizzled FP32-scale path"
            )
        if self.enable_pdl:
            raise ValueError("enable_pdl is unsupported on the MetaX backend")
        if self.use_workspace and self.persistent_grid <= 0:
            raise ValueError(
                "persistent_grid must be positive when use_workspace=true"
            )

        self.groups = self.hidden_size // self.group_size

    def define_metrics(self, state):
        launch_routes = self.num_tokens * self.topk
        active_ratio = self.active_routes / launch_routes
        path = "workspace" if self.use_workspace else "normal"

        state.add_summary("Op", self.name)
        state.add_summary("dtype", "bfloat16->float8_e4m3fn/fp32")
        state.add_summary(
            "Shape",
            (
                f"({self.num_experts} {self.num_tokens} "
                f"{self.hidden_size * 2})->({self.num_experts} "
                f"{self.num_tokens} {self.hidden_size})"
            ),
        )
        state.add_summary(
            "Routes",
            f"{self.active_routes}/{launch_routes} ({active_ratio:.3%})",
        )
        state.add_summary("Path", path)

        output_elements = self.active_routes * self.hidden_size
        state.add_element_count(output_elements)

        # Logical payload: valid BF16 gate/up rows plus the masked_m vector.
        reads = self.active_routes * 2 * self.hidden_size * 2
        reads += self.num_experts * 4

        # Valid FP8 rows and their per-group FP32 scales.
        writes = output_elements
        writes += self.active_routes * self.groups * 4

        if self.use_workspace:
            grid = min(self.persistent_grid, launch_routes)
            work_items = self.active_routes * WORK_ITEM_INT32 * 4
            reads += work_items + grid * 4
            writes += WORK_TABLE_HEADER_INT32 * 4 + work_items

        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def _prepare(self, dev_id, verification=False):
        device = f"cuda:{dev_id}"
        if verification:
            experts = min(self.num_experts, 4)
            tokens = min(self.num_tokens, 6)
            hidden = min(self.hidden_size, 512)
            topk = self.topk
            active_routes = min(10, experts * tokens, tokens * topk)
            seed = self.seed + 1
        else:
            experts = self.num_experts
            tokens = self.num_tokens
            hidden = self.hidden_size
            topk = self.topk
            active_routes = self.active_routes
            seed = self.seed

        torch.manual_seed(seed)
        input_tensor = torch.randn(
            (experts, tokens, 2 * hidden),
            dtype=torch.bfloat16,
            device=device,
        ) * 3
        masked_m = _balanced_masked_m(
            experts,
            tokens,
            active_routes,
            device,
        )
        output = torch.empty(
            (experts, tokens, hidden),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        output_scale = torch.empty(
            (experts, tokens, hidden // self.group_size),
            dtype=torch.float32,
            device=device,
        )

        workspace = None
        persistent_grid = 0
        if self.use_workspace:
            workspace = torch.empty(
                WORK_TABLE_HEADER_INT32 + WORK_ITEM_INT32 * tokens * topk,
                dtype=torch.int32,
                device=device,
            )
            persistent_grid = self.persistent_grid

        return (
            input_tensor,
            output,
            output_scale,
            masked_m,
            topk,
            workspace,
            persistent_grid,
        )

    def _op_args(self, prepared):
        (
            input_tensor,
            output,
            output_scale,
            masked_m,
            topk,
            workspace,
            persistent_grid,
        ) = prepared
        return (
            input_tensor,
            output,
            output_scale,
            masked_m,
            topk,
            self.scale_ue8m0,
            self.transposed,
            self.swizzle,
            self.swiglu_limit,
            self.enable_pdl,
            workspace,
            persistent_grid,
        )

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            prepared = self._prepare(dev_id)
        return self.make_launcher(
            dev_id,
            torch.ops.sgl_kernel.silu_mul_quant_varlen,
            *self._op_args(prepared),
        )

    @torch.inference_mode()
    def run_verification(self, dev_id):
        prepared = self._prepare(dev_id, verification=True)
        torch.ops.sgl_kernel.silu_mul_quant_varlen(*self._op_args(prepared))

        input_tensor, output, output_scale, masked_m, _, _, _ = prepared
        tokens = input_tensor.size(1)
        valid = torch.arange(tokens, device=input_tensor.device).unsqueeze(0)
        valid = valid < masked_m.unsqueeze(1)
        q_ref, scale_ref = _reference_valid(
            input_tensor,
            masked_m,
            self.group_size,
            self.swiglu_limit,
        )

        output_passed, output_diff = self.check_diff(
            output[valid],
            q_ref,
            threshold=self.output_threshold,
        )
        scale_passed, scale_diff = self.check_diff(
            output_scale[valid],
            scale_ref,
            threshold=self.scale_threshold,
        )
        return output_passed and scale_passed, max(output_diff, scale_diff)
