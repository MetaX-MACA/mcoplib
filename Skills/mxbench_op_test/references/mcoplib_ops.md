# mcoplib 算子 nvbench 性能测试参考

本文档汇总已接入 benchmark 框架的算子的关键信息（签名、layout、参考实现要点、坑点），供新增算子时参考。

## 目录

- [OpBenchmarkBase 基类 API](#opbenchmarkbase-基类-api)
- [精度验证 gating 机制](#精度验证-gating-机制)
- [已接入算子的关键点](#已接入算子的关键点)
  - [rotary_embedding (default/sgl/vllm)](#rotary_embedding)
  - [fused_moe_gate_deepseek](#fused_moe_gate_deepseek)
  - [gptq_shuffle](#gptq_shuffle)
  - [paged_attention_v1 / v2](#paged_attention)
  - [sgl_topk_softmax / vllm_topk_softmax](#topk_softmax)
  - [top_k_per_row_decode](#top_k_per_row_decode)
  - [indexer_k_cache / cp_gather_indexer_k_cache](#indexer_k_cache)
- [常见 layout 陷阱](#常见-layout-陷阱)
- [阈值选择指南](#阈值选择指南)

---

## OpBenchmarkBase 基类 API

定义于 `benchmark/mcoplib_mxbenchmark_op_wrapper.py`。所有 runner 必须继承并实现三个抽象方法：

```python
class OpBenchmarkBase(ABC):
    def __init__(self, name, config):
        # name: 算子名；config: 从 JSON 加载的 dict
        # 自动解析 config["dtype"] 为 torch.dtype（默认 float16）
        # 支持 "float16" / "bfloat16" / "float32" / "int32" / 任意 torch.dtype 名
        ...

    @abstractmethod
    def define_metrics(self, state):
        """声明性能指标元信息（Shape、元素数、读写带宽等），供 nvbench 展示。"""
        ...

    @abstractmethod
    def prepare_and_get_launcher(self, dev_id, tc_s):
        """准备输入数据（在 tc_s stream 内），返回 launcher 闭包供 nvbench 反复调用。"""
        ...

    @abstractmethod
    def run_verification(self, dev_id):
        """精度验证：调用算子 + 参考实现，返回 (passed: bool, diff_val: float)。"""
        ...

    def make_launcher(self, dev_id, op_func, *args):
        """工具方法：把 op_func(*args) 包装成 launcher(launch)，在 nvbench stream 上执行。"""
        ...

    def check_diff(self, output_op, output_ref, threshold=0.999999):
        """工具方法：余弦相似度对比，返回 (passed, 1-cos_sim)。"""
        ...
```

`check_diff` 实现：`passed = (cosine_sim >= threshold)`，`diff_val = 1 - cosine_sim`。默认阈值 `0.999999`。

---

## 精度验证 gating 机制

`benchmark/mcoplib_mxbenchmark_ops.py:create_benchmark_wrapper()` 中：

```python
def benchmark_func(state):
    dev_id = state.get_device()
    # 1. 精度验证
    try:
        passed, diff_val = op_instance.run_verification(dev_id)
        state.add_summary("Acc_Pass", "Yes" if passed else "No")
        state.add_summary("Cos_Dist", f"{diff_val:.2e}")
        print(f"\n[VERIFY] {op_instance.name} -> {'PASS' if passed else 'FAIL'} (1-CosSim: {diff_val:.2e})")
        is_verified = passed
    except Exception as e:
        is_verified = False
        print(f"\n[VERIFY] Error: {e}")
        state.add_summary("Acc_Pass", "Error")

    if not is_verified:
        print(f"\n[FATAL] {op_instance.name} verification failed. Terminating all processes...")
        import os
        os._exit(0)   # ← 直接退出，不执行 nvbench 采样

    # 2. warmup + 3. nvbench 采样（仅 verify 通过才到这）
    ...
```

**关键点**：
- `os._exit(0)` 会丢弃 stdout 缓冲，所以 `[VERIFY] FAIL` 行可能不会出现在 testall.py 的捕获输出里
- testall.py 通过 `_op_in_csv()` 回退检查 CSV 是否新增该算子行来兜底判定（FAILED "no result"）
- runner 作者**无需**自己实现 gating，只需保证 `run_verification` 返回正确的 `(passed, diff_val)`

---

## 已接入算子的关键点

### rotary_embedding

**三个版本**（同名算子拆分模式）：
- `rotary_embedding`（default op，`mcoplib.op`）：`op.rotary_embedding(positions, q, k, head_size, cos_sin_cache, is_neox, rope_dim_offset, inverse)`
- `sgl_rotary_embedding`（sglang）：`torch.ops.sgl_kernel.rotary_embedding(positions, q, k, head_size, cos_sin_cache, is_neox)`
- `vllm_rotary_embedding`（vllm）：`torch.ops._C.rotary_embedding(positions, q, k, head_size, cos_sin_cache, is_neox, rope_dim_offset, inverse)`

**坑点**：
1. `cos_sin_cache` dtype **必须与 query dtype 一致**（bfloat16 not float32）——kernel dispatches on query dtype
2. default 版本的 cos/sin shape 是 `[max_seq, head_size/2]`（不是 `[max_position, head_size]`）
3. default 版本只写 QK heads，不写 V heads——ref 比对时只比 `out[:, :qk_end]`

### fused_moe_gate_deepseek

签名：`op.fused_moe_gate_deepseek(gating, bias, out_w, out_idx, topk, renormalize, num_expert_group, topk_group, num_fused_shared_experts, routed_scaling_factor, moegate_type)`

**坑点**：
- 输出是 top-k 索引（**无序**），不能用余弦相似度
- 必须按行排序后用 `torch.allclose`（weights）+ `torch.equal`（indices）比对
- 直接返回 `(passed, 0.0 if passed else 1.0)`，不走 `check_diff`

### gptq_shuffle

签名：`torch.ops._C.gptq_shuffle(q_weight(!), q_perm, bits)`

源码：`op/vllm/quantization/gptq/q_gemm.cu:2437`，调用 `shuffle_exllama_weight()`。

**两遍处理**：
1. `make_sequential_4bit_kernel`：按 `q_perm` 重排行，每个 int32 装 8 个连续 4-bit nibble
2. `shuffle_4bit_kernel`：每个 int32 内部 nibble 重排 `[q0,q1,q2,q3,q4,q5,q6,q7] -> [q0,q2,q4,q6,q1,q3,q5,q7]`（exllama layout）

**坑点**：
1. 第一遍会调用 `cudaMalloc` + `cudaMemcpyAsync` + `cudaDeviceSynchronize` + `cudaFree`——**每次调用都阻塞设备**
2. nvbench 采样 1000 次 = 1000 次 sync，总耗时 >600s，触发 testall.py 超时
3. **解决方案**：benchmark 时传空 `q_perm`（`torch.empty(0)`），跳过第一遍，只跑第二遍（快）；verification 时传真实 `q_perm`，两遍都跑（验证完整性）
4. verification 用 `torch.equal` bit-exact 比对，不走 `check_diff`

### paged_attention

签名（v1）：`torch.ops._C.paged_attention_v1(out(!), query, key_cache, value_cache, num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len, alibi_slopes?, kv_cache_dtype, k_scale, v_scale, tp_rank, blocksparse_local_blocks, blocksparse_vert_stride, blocksparse_block_size, blocksparse_head_sliding_step)`

签名（v2）：v1 + `exp_sums(!), max_logits(!), tmp_out(!)` 三个额外输出张量（在 `out` 之后、`query` 之前）

**Layout（关键）**：
- `query`: `[num_seqs, num_heads, head_size]`
- `key_cache`: `[num_blocks, num_kv_heads, head_size/x, block_size, x]`，其中 `x = 16 // sizeof(dtype)`（bf16/f16 → x=8，f32 → x=4）
- `value_cache`: **`[num_blocks, num_kv_heads, head_size, block_size]`**（不是 `[block_size, head_size]`！）
- `block_tables`: `[num_seqs, max_num_blocks_per_seq]`
- `seq_lens`: `[num_seqs]`

**参考实现**（来自 vLLM 单元测试 `test_paged_attention_v1.py`）：
```python
def ref_single_query_cached_kv_attention(output, query, num_queries_per_kv,
        key_cache, value_cache, block_tables, seq_lens, scale, alibi_slopes):
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i].cpu().tolist()
        seq_len = int(seq_lens[i])
        keys_lst, values_lst = [], []
        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size
            k = key_cache[block_number, :, :, block_offset, :]    # ← 注意：第 4 维是 block_offset
            v = value_cache[block_number, :, :, block_offset]      # ← 注意：第 3 维是 head_size，第 4 维是 block_offset
            keys_lst.append(k.reshape(num_kv_heads, head_size))
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_queries_per_kv > 1:
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)
        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        output[i].copy_(out.view(num_query_heads, head_size), non_blocking=True)
```

**坑点**：
1. `value_cache` layout 容易写反——很多人写成 `[block_size, head_size]`，导致 ref 比对失败
2. `ref_masked_attention` 用 `torch.einsum("qhd,khd->hqk", query, key)` 转置 Q
3. 阈值放宽到 `0.999`（bf16 累积误差）
4. `query` 用 `uniform_(-scale, scale)` 而非 `randn`（与 vLLM 单测一致）

### topk_softmax

**两个版本**（同名拆分）：

**sgl**：`torch.ops.sgl_kernel.topk_softmax(topk_weights(!), topk_indices(!), gating_output, renormalize, moe_softcapping, correction_bias?)`
- 源码：`op/sglang/csrc/moe/moe_topk_softmax_kernels.cu:717`
- 参数顺序：(out_w, out_idx, gating, renormalize, moe_softcapping, correction_bias)
- `correction_bias` 是 `Tensor?`（optional），传 `None`

**vllm**：`torch.ops._moe_C.topk_softmax(topk_weights(!), topk_indices(!), token_expert_indices(!), gating_output, renormalize, bias?)`
- 源码：`op/vllm/moe/topk_softmax_kernels.cu:1311`
- 参数顺序：(out_w, out_idx, out_expert_idx, gating, renormalize, bias)
- vllm 多一个 `token_expert_indices` 输出张量
- `bias` 是 `std::optional<Tensor>`，传 `None`

**坑点**：
- sgl 有 `moe_softcapping` 参数（float，0.0 表示禁用），vllm 没有
- sgl 的 bias 叫 `correction_bias`，vllm 叫 `bias`
- verification 比对 indices 时用 `torch.topk` + `(out_indices.long() == ref_idxs).all()`

### top_k_per_row_decode

签名：`torch.ops._C.top_k_per_row_decode(logits, next_n, seq_lens, indices(!), numRows, stride0, stride1, topK)`

源码：`op/vllm/sampler.cu:657`

**坑点**：
- 共 8 个位置参数，最后一个 `topK` 容易漏（int64）
- `topK` 是输出 indices 的第二维大小（`indices.shape = [numRows, topK]`）
- `seq_lens` 可以是 1D（`[num_seqs]`）或 2D（`[B, next_n]`），kernel 自动判断
- verification：每行取前 `min(top_k, row_end)` 个 token，`torch.topk` 比对值（indices 顺序不保证，先 sort）

### indexer_k_cache / cp_gather_indexer_k_cache

源码：`op/vllm/attention/...`

**要点**：
- KV cache 索引重排算子
- 输出是 index 张量，用 `torch.equal` 比对
- 参考 `benchmark/runners/mcoplib_mxbenchmark_indexer_k_cache_runners.py` 等已实现

---

## 常见 layout 陷阱

### 1. value_cache 顺序（paged_attention）

错误：`value_cache = [num_blocks, num_kv_heads, block_size, head_size]`
正确：`value_cache = [num_blocks, num_kv_heads, head_size, block_size]`

ref 中 V 切片：`value_cache[block, head, :, offset]`（在第 3 维 head_size 上切片，第 4 维 block_offset 上取标量）

### 2. cos_sin_cache dtype（rotary_embedding）

错误：`cos_sin_cache = torch.randn(..., dtype=torch.float32)`
正确：`cos_sin_cache = torch.randn(..., dtype=self.dtype)`（与 query 一致）

kernel dispatches on query dtype，cos_sin_cache dtype 不匹配会读错数据。

### 3. q_perm 空张量（gptq_shuffle benchmark）

benchmark 路径：传 `torch.empty(0, dtype=torch.int32)` 跳过慢的 make_sequential pass
verification 路径：传真实 `q_perm`，两遍都跑

### 4. top-k 无序输出

不能直接用 `check_diff`（余弦相似度对顺序不敏感）。要么：
- 排序后 `allclose` + `equal`
- bit-exact 算子用 `torch.equal`，返回 `(passed, 0.0 if passed else 1.0)`

### 5. samples 过大导致超时

`samples: 100000` 会让带 `cudaDeviceSynchronize` 的算子（gptq_shuffle 等）卡死。默认用 1000，调试时用 50-100。

### 6. set_blocking_kernel_timeout(-1)

禁止使用——会让 nvbench 无限等待。除非确认每次调用都在 15s 内，否则不要调用此方法。

---

## 阈值选择指南

| 算子类型 | 阈值 | 比对方式 |
|----|----|----|
| 普通浮点算子（rms_norm, silu_and_mul 等） | 0.999999 | `check_diff`（余弦相似度） |
| 量化算子（AWQ, GPTQ, FP8） | 0.99 - 0.999 | `check_diff` |
| Attention 类（bf16 累积误差） | 0.999 | `check_diff` |
| bit-exact 算子（gptq_shuffle, copy_blocks 等） | N/A | `torch.equal`，返回 `(passed, 0.0/1.0)` |
| 无序输出（top-k 索引） | N/A | 排序后 `allclose` + `equal` |

`check_diff` 默认 `threshold=0.999999`。需要放宽时显式传参：`self.check_diff(out, ref, threshold=0.99)`。
