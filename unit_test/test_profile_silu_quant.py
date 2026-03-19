import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

# from sglang.srt.layers.moe.ep.moe.kernels import silu_and_mul_masked_fwd
from mcoplib import op

def silu_and_mul_masked_fwd(
    input: torch.Tensor,
    masked_m: torch.Tensor,
):
    out_stride = (input.shape[-1]//4 + 257) // 256 * 256
    output = torch.empty((input.shape[0], input.shape[1], out_stride), device=input.device, dtype=input.dtype)
    op.fused_silu_mul_dq_mask_quant(output, input, masked_m)
    return output

# ======================
# 固定随机种子（保证每次输入一样）
# ======================
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

# ======================
# 2. 构造输入（和你日志 shape 完全对齐）
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

# gateup_output shape: [8, 4096, 4096]
gateup_output = torch.randn(8, 4096, 4096, device=device, dtype=torch.float16)

# masked_m shape: [8], 取值范围 [0, 256)
masked_m = torch.randint(low=0, high=256, size=(8,), device=device, dtype=torch.int32)

# ======================
# 3. 预热（必须，避免 CUDA 初始化干扰）
# ======================
for _ in range(10):
    _ = silu_and_mul_masked_fwd(gateup_output, masked_m)
torch.cuda.synchronize()
print("✅ 预热完成")

# ======================
# 4. Profiler 统计耗时
# ======================
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    for _ in range(50):
        out = silu_and_mul_masked_fwd(gateup_output, masked_m)
        torch.cuda.synchronize()

# ======================
# 5. 打印结果（按 CUDA 总耗时排序）
# ======================
print("\n=== Profiler 耗时统计（按 CUDA 总耗时排序）===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 可选：只看 fused 算子相关耗时
print("\n=== 仅 fused_silu_mul_dq_mask_quant 相关耗时 ===")
for e in prof.key_averages():
    if "fused_silu_mul_dq_mask_quant" in e.key or "elementwise" in e.key:
        print(f"{e.key:40} | CPU 时间: {e.cpu_time_total:8.2f} us | CUDA 时间: {e.cuda_time_total:8.2f} us")