import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

# num_candidates 超过该值时，stable_topk_gathered 将 keys 存到 global memory，
# 与 kernel 里的 MAX_SMEM_CANDIDATES = (65536 - 5000) / 8 保持一致。
MAX_SMEM_CANDIDATES = 7567


class Stable_topk_gathered_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 1)
        self.world = config.get("world", 1)
        self.topk = config.get("topk", 2048)
        self.num_candidates = self.world * self.topk
        if self.dtype != torch.float32:
            print(f"[Warning] {name} only supports float32 in kernel source. "
                  f"Forcing dtype to float32.")
            self.dtype = torch.float32

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "float32")
        state.add_summary("Shape",
                          f"({self.batch_size} {self.num_candidates} 2) -> Top{self.topk}")
        state.add_element_count(self.batch_size * self.num_candidates)

        # gathered 读：每个候选 score + token_id = 2 × float32 = 8 字节
        read_bytes = self.batch_size * self.num_candidates * 8
        # out 写：每个 slot int32 = 4 字节
        write_bytes = self.batch_size * self.topk * 4
        # world≥4 时 keys 存 global memory，额外访存 ≈ 5 × nc × 8（初始写 1 + 每 pass 读 4）
        if self.num_candidates > MAX_SMEM_CANDIDATES:
            read_bytes += self.batch_size * self.num_candidates * 8 * 4
            write_bytes += self.batch_size * self.num_candidates * 8
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def _make_gathered(self, device):
        """[bs][world×topk][2] float32: col0 score, col1 token_id 存为 float。

        token_id 落在 [0, 2^23) 内可被 float32 精确表示（与 DCP 打包布局一致）；
        随机 5% 置为 -1（无效候选）。
        """
        gathered = torch.randn(self.batch_size, self.num_candidates, 2, device=device)
        ids = torch.randint(0, 1 << 23,
                            (self.batch_size, self.num_candidates), device=device)
        invalid = torch.rand(self.batch_size, self.num_candidates,
                             device=device) < 0.05
        ids[invalid] = -1
        gathered[..., 1] = ids.to(torch.int32).to(torch.float32)
        return gathered

    def _ref(self, gathered):
        """CPU 参考：稳定排序（score 降序，token_id 升序），无效候选排最后。"""
        scores = gathered[..., 0]
        ids = gathered[..., 1].to(torch.int32)
        valid = ids >= 0
        sort_scores = scores.masked_fill(~valid, float("-inf"))
        sort_ids = ids.masked_fill(~valid, 2 ** 31 - 1)
        num_rows, num_candidates = ids.shape

        order = torch.arange(num_candidates, device=gathered.device).expand(
            num_rows, num_candidates)
        # 先按 token_id 升序（次要键，稳定），再按 score 降序（主键，稳定）
        order = order.gather(
            1, sort_ids.gather(1, order).argsort(dim=-1, stable=True))
        order = order.gather(
            1, sort_scores.gather(1, order).argsort(dim=-1, descending=True, stable=True))

        selected = ids.gather(1, order)
        k = min(self.topk, num_candidates)
        ref = torch.full((num_rows, self.topk), -1, dtype=torch.int32,
                         device=gathered.device)
        ref[:, :k] = selected[:, :k]
        return ref

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if not hasattr(torch.ops._C, "stable_topk_gathered"):
            raise RuntimeError("Operator 'torch.ops._C.stable_topk_gathered' not found.")
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            gathered = self._make_gathered(dev)
            out = torch.empty((self.batch_size, self.topk),
                              dtype=torch.int32, device=dev)
        return self.make_launcher(dev_id, torch.ops._C.stable_topk_gathered,
                                  gathered, out, self.topk)

    def run_verification(self, dev_id):
        if not hasattr(torch.ops._C, "stable_topk_gathered"):
            print("Error: torch.ops._C.stable_topk_gathered not available.")
            return False, 1.0
        dev = f'cuda:{dev_id}'
        gathered = self._make_gathered(dev)
        out = torch.empty((self.batch_size, self.topk),
                          dtype=torch.int32, device=dev)
        torch.ops._C.stable_topk_gathered(gathered, out, self.topk)
        ref = self._ref(gathered)

        # 输出顺序（radix scatter 的 bin 内顺序）与 -1 位置未指定，
        # 故排序后逐元素比较（等价于多重集 + 有效计数一致）。
        out_sorted, _ = torch.sort(out, dim=-1)
        ref_sorted, _ = torch.sort(ref, dim=-1)
        mismatch = (out_sorted != ref_sorted).sum().item()
        total = out.numel()
        passed = mismatch == 0
        diff_val = mismatch / total if total else 0.0
        return passed, diff_val
