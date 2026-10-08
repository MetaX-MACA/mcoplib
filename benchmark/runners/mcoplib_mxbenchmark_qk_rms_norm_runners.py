import torch
import torch.nn.functional as F

from mcoplib.op import qk_rms_norm_inplace_cuda
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


class Qk_rms_norm_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.num_tokens = config.get("num_tokens", 16)
        self.q_head_num, self.kv_head_num = self._parse_head_config(
            config.get("head_config", "8x1")
        )
        self.qk_head_dim = config.get("qk_head_dim", 128)
        self.v_head_dim = config.get("v_head_dim", self.qk_head_dim)
        self.eps = config.get("eps", 1e-5)

        if self.dtype != torch.bfloat16:
            raise ValueError("qk_rms_norm only supports bfloat16 token_data")
        if self.qk_head_dim != 128:
            raise ValueError("qk_rms_norm currently only supports qk_head_dim=128")
        if self.v_head_dim != self.qk_head_dim:
            raise ValueError(
                "qk_rms_norm currently requires v_head_dim == qk_head_dim"
            )

        self.norm_head_num = self.q_head_num + self.kv_head_num
        self.total_dim = (
            self.q_head_num * self.qk_head_dim
            + self.kv_head_num * self.qk_head_dim
            + self.kv_head_num * self.v_head_dim
        )

    @staticmethod
    def _parse_head_config(head_config):
        try:
            q_head_num, kv_head_num = map(int, head_config.split("x"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "head_config must use the '<q_head_num>x<kv_head_num>' format"
            ) from exc
        if q_head_num <= 0 or kv_head_num <= 0:
            raise ValueError("q_head_num and kv_head_num must be positive")
        return q_head_num, kv_head_num

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary(
            "Shape",
            (
                f"(Tokens:{self.num_tokens} QHeads:{self.q_head_num} "
                f"KVHeads:{self.kv_head_num} QKDim:{self.qk_head_dim} "
                f"VDim:{self.v_head_dim})"
            ),
        )

        norm_elements = (
            self.num_tokens * self.norm_head_num * self.qk_head_dim
        )
        token_element_size = torch.empty((), dtype=self.dtype).element_size()
        weight_element_size = torch.empty(
            (), dtype=torch.float32
        ).element_size()

        read_bytes = norm_elements * (
            token_element_size + weight_element_size
        )
        write_bytes = norm_elements * token_element_size

        state.add_element_count(norm_elements)
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def _make_inputs(self, dev, random_weights):
        token_data = torch.randn(
            self.num_tokens,
            self.total_dim,
            dtype=self.dtype,
            device=dev,
        )
        weight_factory = torch.randn if random_weights else torch.ones
        q_norm_weight = weight_factory(
            self.qk_head_dim, dtype=torch.float32, device=dev
        )
        k_norm_weight = weight_factory(
            self.qk_head_dim, dtype=torch.float32, device=dev
        )
        return token_data, q_norm_weight, k_norm_weight

    def _reference(self, token_data, q_norm_weight, k_norm_weight):
        expected = token_data.clone()
        q_end = self.q_head_num * self.qk_head_dim
        k_end = q_end + self.kv_head_num * self.qk_head_dim

        q_data = expected[:, :q_end].view(
            self.num_tokens, self.q_head_num, self.qk_head_dim
        )
        k_data = expected[:, q_end:k_end].view(
            self.num_tokens, self.kv_head_num, self.qk_head_dim
        )
        q_data.copy_(
            F.rms_norm(
                q_data.float(),
                (self.qk_head_dim,),
                q_norm_weight,
                self.eps,
            ).to(self.dtype)
        )
        k_data.copy_(
            F.rms_norm(
                k_data.float(),
                (self.qk_head_dim,),
                k_norm_weight,
                self.eps,
            ).to(self.dtype)
        )
        return expected

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            token_data, q_norm_weight, k_norm_weight = self._make_inputs(
                dev, random_weights=False
            )

        return self.make_launcher(
            dev_id,
            qk_rms_norm_inplace_cuda,
            token_data,
            q_norm_weight,
            k_norm_weight,
            self.q_head_num,
            self.kv_head_num,
            self.qk_head_dim,
            self.eps,
        )

    @torch.inference_mode()
    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        torch.manual_seed(42)
        token_data, q_norm_weight, k_norm_weight = self._make_inputs(
            dev, random_weights=True
        )
        original = token_data.clone()
        expected = self._reference(
            original, q_norm_weight, k_norm_weight
        )

        output = qk_rms_norm_inplace_cuda(
            token_data,
            q_norm_weight,
            k_norm_weight,
            self.q_head_num,
            self.kv_head_num,
            self.qk_head_dim,
            self.eps,
        )
        torch.cuda.synchronize()

        qk_end = self.norm_head_num * self.qk_head_dim
        norm_passed, norm_diff = self.check_diff(
            output[:, :qk_end], expected[:, :qk_end], threshold=0.999
        )
        v_unchanged = torch.equal(output[:, qk_end:], original[:, qk_end:])
        aliases_input = output.data_ptr() == token_data.data_ptr()
        return norm_passed and v_unchanged and aliases_input, norm_diff
