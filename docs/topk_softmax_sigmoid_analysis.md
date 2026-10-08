# TopK Softmax / TopK Sigmoid 算子深度解析

## 一、算子概述与作用

### 1.1 算子是什么？

`topk_softmax` 和 `topk_sigmoid` 是 **MoE (Mixture of Experts) 模型的门控路由算子**。它们的作用是将模型的 Gate 网络输出的原始 logits 转换为专家权重，并选出 Top-K 个最合适的专家。

**一句话总结：** 输入每个 token 对所有专家的打分 (logits)，输出 Top-K 个专家的索引和对应权重。

### 1.2 在 MoE LLM 中的应用场景

```
MoE Transformer Layer 中的位置：

┌─────────────────────────────────────────────────────┐
│                 MoE Transformer Layer                │
│                                                     │
│  Input Token ──→ [Attention] ──→ hidden_state       │
│                                         │           │
│                                         ▼           │
│                               ┌─────────────────┐   │
│                               │   Gate Network   │   │
│                               │  (Linear Layer)  │   │
│                               └────────┬────────┘   │
│                                        │            │
│                                        ▼            │
│                              gating_logits [T, E]    │
│                                        │            │
│                          ┌─────────────┼──────────┐ │
│                          ▼             ▼          ▼ │
│                   ┌──────────┐  ┌──────────┐       │ │
│                   │ topk_    │  │ topk_    │  ...  │ │
│                   │ softmax  │  │ sigmoid  │       │ │
│                   └─────┬────┘  └─────┬────┘       │ │
│                         │             │            │ │
│                         ▼             ▼            │ │
│               topk_indices [T, K]     topk_weights │ │
│                         │             │            │ │
│                         ▼             ▼            │ │
│               ┌──────────────────────────┐         │ │
│               │   Expert Dispatch        │         │ │
│               │   (将 token 路由到       │         │ │
│               │    选中的 K 个专家)       │         │ │
│               └──────────┬───────────────┘         │ │
│                          ▼                         │ │
│               ┌──────────────────────────┐         │ │
│               │   Expert Computation     │         │ │
│               │   Expert_0(h) Expert_1(h)│         │ │
│               │   Expert_2(h) ...        │         │ │
│               └──────────┬───────────────┘         │ │
│                          ▼                         │ │
│               ┌──────────────────────────┐         │ │
│               │   moe_sum_reduce         │         │ │
│               │   (加权聚合专家输出)      │         │ │
│               └──────────┬───────────────┘         │ │
│                          ▼                         │ │
│                    Output Token                     │ │
└─────────────────────────────────────────────────────┘

T = num_tokens (token 数量)
E = num_experts (专家数量，如 8, 16, 64, 128, 256, 288)
K = topk (每个 token 选中的专家数，如 2, 4, 8)
```

**典型模型使用场景：**

| 模型 | num_experts | topk | 使用哪个算子 |
|------|-------------|------|-------------|
| Mixtral 8x7B | 8 | 2 | topk_softmax |
| DeepSeek V2/V3 | 160/256 | 6/8 | topk_sigmoid |
| DeepSeek V3 | 256 | 8 | topk_sigmoid |
| Qwen MoE | 64 | 8 | topk_softmax |
| Kimi K2 | 288 | 8 | topk_sigmoid |
| DBRX | 16 | 4 | topk_softmax |

### 1.3 解决了什么问题？

1. **专家选择问题**：每个 token 需要从大量专家中选择最合适的 K 个，而非使用全部专家（否则计算量与全参数模型相同）
2. **权重归一化问题**：Gate 输出的 logits 需要转换为概率分布作为权重，用于后续加权聚合专家输出
3. **效率问题**：需要在一个 GPU kernel 中融合 Softmax/Sigmoid + TopK，避免多次 kernel launch 和中间张量的显存读写

### 1.4 Softmax vs Sigmoid — 为什么需要两者？

```
Softmax 归一化 (竞争型):
  softmax(z_i) = exp(z_i) / Σ_j exp(z_j)

  特点: 所有专家权重之和 = 1，专家之间存在"竞争"关系
  适用: 专家数量较少、需要强归一化的场景 (如 Mixtral 8 experts)

  示例: logits = [2.0, 1.0, 0.5, -1.0]
        softmax → [0.56, 0.21, 0.13, 0.02]   (和=1.0)
        topk=2 → 选 [0.56, 0.21]

Sigmoid 归一化 (独立型):
  sigmoid(z_i) = 1 / (1 + exp(-z_i))

  特点: 每个专家的权重独立计算，∈(0,1)，专家之间无"竞争"
  适用: 专家数量多的场景 (如 DeepSeek 256 experts)，
        因为 softmax 在大量专家上容易导致权重过度集中或数值不稳定

  示例: logits = [2.0, 1.0, 0.5, -1.0]
        sigmoid → [0.88, 0.73, 0.62, 0.27]   (和≈2.5，非1.0)
        topk=2 → 选 [0.88, 0.73]
```

**为何 DeepSeek V3 等大专家模型选择 Sigmoid？**

| 对比维度 | Softmax | Sigmoid |
|---------|---------|---------|
| 专家间关系 | 竞争 (一个↑另一个↓) | 独立 (互不影响) |
| 权重分布 | 集中在1-2个专家 | 更均匀地分布 |
| 大专家数时 | 容易数值不稳定(指数溢出) | 每个独立计算，数值稳定 |
| 归一化 | 天然和=1 | 需额外 renormalize |
| 适用场景 | 专家数少 (≤64) | 专家数多 (≥128) |

---

## 二、算子调用流程图

### 2.1 topk_softmax 调用流程

```
Python 层调用:
  torch.ops._moe_C.topk_softmax(
      topk_weights,          # [T, K] 输出：选中专家的权重
      topk_indices,          # [T, K] 输出：选中专家的索引
      token_expert_indices,  # [T, K] 输出：token-专家映射
      gating_output,         # [T, E] 输入：Gate 网络的 logits
      renormalize,           # bool：是否重新归一化
      bias                   # Optional[float]：修正偏置
  )
      │
      ▼
C++ Host 层: topk_softmax() — op/vllm/moe/topk_softmax_kernels.cu:1311
      │
      ├── 解析参数: num_tokens, num_experts, topk
      ├── 判断是否需要 workspace (非2幂或 >256)
      ├── 分配 softmax_workspace (float32)
      │
      ├── 根据 gating_output dtype 分发:
      │   ├── float32 → dispatch_topk_launch<float, SCORING_SOFTMAX>()
      │   ├── float16 → dispatch_topk_launch<__half, SCORING_SOFTMAX>()
      │   └── bfloat16 → dispatch_topk_launch<__nv_bfloat16, SCORING_SOFTMAX>()
      │
      ▼
dispatch_topk_launch<ComputeType, SCORING_SOFTMAX>() — :1259
      │
      ├── 处理 bias (optional → raw pointer)
      ├── 根据 topk_indices dtype 分发:
      │   ├── int32  → topkGatingKernelLauncher<int, ...>()
      │   ├── uint32 → topkGatingKernelLauncher<uint32_t, ...>()
      │   └── int64  → topkGatingKernelLauncher<int64_t, ...>()
      │
      ▼
topkGatingKernelLauncher<int, ComputeType, SCORING_SOFTMAX>() — :1133
      │
      ├── [Sigmoid 快速路径] num_experts==288 && topk==8
      │   → topkGatingSigmoid288Opt (不适用于 softmax)
      │
      ├── [Sigmoid 通用路径] num_experts<=256 && topk<=16
      │   → topkGatingSigmoidCommonOpt (不适用于 softmax)
      │
      └── [Softmax 主路径] switch(num_experts)
          │
          ├── 1,2,4,8,16,32,64 → LAUNCH_TOPK (2幂优化)
          │   └── topkGatingLauncherHelper → topkGating kernel
          │
          ├── 128 → LAUNCH_SOFTMAX_OPT (Decode 优化)
          │   └── topkDecodeGatingSoftmaxLauncherHelper
          │       ├── k==8 && rows<1024 → topkGatingSoftmaxDecode
          │       └── 否则 → topkGating (通用)
          │
          ├── 256,512 → LAUNCH_TOPK (2幂优化)
          │
          ├── 192,320,384,448,576 → LAUNCH_TOPK (64倍数优化)
          │
          └── default → 分两步: moeSoftmax + moeTopK (慢路径)
```

### 2.2 topk_sigmoid 调用流程

```
Python 层调用:
  torch.ops._moe_C.topk_sigmoid(
      topk_weights,          # [T, K] 输出
      topk_indices,          # [T, K] 输出
      token_expert_indices,  # [T, K] 输出
      gating_output,         # [T, E] 输入
      renormalize,           # bool
      bias                   # Optional[float]
  )
      │
      ▼
C++ Host 层: topk_sigmoid() — :1349
      │
      ├── 参数解析同 topk_softmax
      ├── 根据 gating_output dtype 分发:
      │   ├── float32 → dispatch_topk_launch<float, SCORING_SIGMOID>()
      │   ├── float16 → dispatch_topk_launch<__half, SCORING_SIGMOID>()
      │   └── bfloat16 → dispatch_topk_launch<__nv_bfloat16, SCORING_SIGMOID>()
      │
      ▼
dispatch_topk_launch<ComputeType, SCORING_SIGMOID>() — :1259
      │
      ▼
topkGatingKernelLauncher<int, ComputeType, SCORING_SIGMOID>() — :1133
      │
      ├── [C500 专用] num_experts==288 && topk==8
      │   │  (DeepSeek V3 场景: 256路由专家+32共享专家)
      │   └── topkGatingSigmoid288Opt <<<>>> (C500 64线程Warp优化)
      │
      ├── [C500 通用] num_experts<=256 && topk<=16
      │   └── topkGatingSigmoidCommonOpt <<<>>> (64线程Warp排序)
      │
      └── [其他] switch(num_experts)
          ├── 1,2,4,...,512 → LAUNCH_TOPK → topkGating (通用kernel)
          ├── 128 → LAUNCH_SOFTMAX_OPT (Decode优化)
          └── default → 分两步: moeSigmoid + moeTopK (慢路径)
```

---

## 三、Kernel 实现详解与流程图

### 3.1 融合 Kernel: topkGating (通用路径)

这是最核心的融合 kernel，将 Softmax/Sigmoid + TopK + Renormalize 融合在单个 kernel 中。

**编译时常量 (模板参数):**
```
VPT              = Values Per Thread (每个线程处理的元素数)
NUM_EXPERTS      = 专家总数 (编译时已知)
WARPS_PER_CTA    = 每个 CTA 的 warp 数
BYTES_PER_LDG    = 每次加载的字节数
WARP_SIZE_PARAM  = Warp 大小 (32 for CUDA, 64 for MACA)
```

**线程组织:**
```
一个 CTA (线程块) 的结构:

┌───────────────────────────────────────────────────┐
│ CTA (Block) — WARPS_PER_CTA 个 Warp              │
│                                                   │
│  ┌─────────────────┐  ┌─────────────────┐        │
│  │ Warp 0           │  │ Warp 1           │  ...  │
│  │                  │  │                  │        │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │        │
│  │ │ Sub-group 0  │ │  │ │ Sub-group 0  │ │        │
│  │ │ (处理1行)    │ │  │ │ (处理1行)    │ │        │
│  │ ├──────────────┤ │  │ ├──────────────┤ │        │
│  │ │ Sub-group 1  │ │  │ │ Sub-group 1  │ │        │
│  │ │ (处理1行)    │ │  │ │ (处理1行)    │ │        │
│  │ └──────────────┘ │  │ └──────────────┘ │        │
│  └─────────────────┘  └─────────────────┘        │
└───────────────────────────────────────────────────┘

THREADS_PER_ROW = NUM_EXPERTS / VPT   (每个行需要的线程数)
ROWS_PER_WARP   = WARP_SIZE / THREADS_PER_ROW  (每个 warp 处理的行数)
ROWS_PER_CTA    = WARPS_PER_CTA * ROWS_PER_WARP  (每个 CTA 处理的行数)

例: NUM_EXPERTS=8, VPT=4, WARP_SIZE=32
    THREADS_PER_ROW = 8/4 = 2
    ROWS_PER_WARP = 32/2 = 16 (一个 warp 处理 16 行)
```

**topkGating Kernel 执行流程:**

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 数据加载 — 从 Global Memory 加载 gating logits      │
│                                                               │
│  每个线程加载 VPT 个元素:                                      │
│    - float32: 直接向量化加载                                   │
│    - bfloat16: bf162→float2 转换加载                          │
│    - float16: half2→float2 转换加载                            │
│                                                               │
│  row_chunk[VPT] ← input[thread_row * NUM_EXPERTS + offset]  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2a: Softmax 归一化 (SCORING_SOFTMAX 路径)               │
│                                                               │
│  ① Thread-local Max:                                          │
│     thread_max = max(row_chunk[0..VPT-1])                    │
│                                                               │
│  ② Warp Reduce Max (butterfly):                               │
│     for mask = THREADS_PER_ROW/2 → 1:                        │
│       thread_max = max(thread_max, shfl_xor(thread_max))     │
│     → 所有线程获得行内最大值                                   │
│                                                               │
│  ③ 减最大值 + Exp + 求和:                                     │
│     row_chunk[i] = exp(row_chunk[i] - thread_max)            │
│     row_sum = Σ row_chunk[i]                                  │
│                                                               │
│  ④ Warp Reduce Sum (butterfly):                               │
│     → 所有线程获得行内 exp 之和                                │
│                                                               │
│  ⑤ 归一化:                                                    │
│     row_chunk[i] = row_chunk[i] / row_sum                    │
│     → row_chunk 现在包含 softmax 概率                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2b: Sigmoid 归一化 (SCORING_SIGMOID 路径)               │
│                                                               │
│  每个元素独立计算:                                             │
│    row_chunk[i] = 1.0 / (1.0 + exp(-row_chunk[i]))          │
│  → row_chunk 现在包含 sigmoid 概率                             │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: NaN/Inf 钳位                                          │
│                                                               │
│  if (isnan(row_chunk[i]) || isinf(row_chunk[i])):            │
│      row_chunk[i] = 0.0f                                      │
│                                                               │
│  原因: CUDA graph padding 产生退化 hidden states →           │
│        softmax 产生全 NaN → argmax 总是选 expert 0 →         │
│        所有 topk 位选到同一个专家 → 下游 FlashInfer 排序崩溃  │
│  修复: NaN→0 后 argmax 使用索引打破平局，选 [0,1,2,...,k-1]   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 应用 Bias (可选)                                      │
│                                                               │
│  if bias != nullptr:                                          │
│    row_chunk_for_choice[i] = row_chunk[i] + bias[expert]     │
│  else:                                                        │
│    row_chunk_for_choice[i] = row_chunk[i]                     │
│                                                               │
│  注意: bias 只影响 TopK 选择过程，不影响输出权重               │
│  (输出的权重始终是 softmax/sigmoid 原始值，不含 bias)         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Top-K 选择 (循环 K 次)                                │
│                                                               │
│  for k_idx = 0 to K-1:                                        │
│    │                                                          │
│    ├── ① Thread-local ArgMax:                                 │
│    │   每个线程找自己 VPT 个元素中的最大值和索引               │
│    │                                                          │
│    ├── ② Warp ArgMax (butterfly reduce):                      │
│    │   线程组内通过 shfl_xor 共识选出全局最大值                │
│    │   平局时选索引最小的专家                                  │
│    │                                                          │
│    ├── ③ Lead Thread 写出结果:                                │
│    │   output[k*row + k_idx] = max_val (不含bias的原始权重)   │
│    │   indices[k*row + k_idx] = expert_id                     │
│    │   source_rows[k*row + k_idx] = k_idx * num_rows + row    │
│    │                                                          │
│    └── ④ 清除已选中的值:                                      │
│        将选中专家的值设为 -10000 (等效 -inf)                   │
│        下次循环不会再选中它                                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 6: Renormalize (可选)                                    │
│                                                               │
│  if renormalize:                                              │
│    selected_sum = Σ(output[0..K-1])                           │
│    output[k] = output[k] / selected_sum                      │
│                                                               │
│  作用: 使 Top-K 个选中专家的权重重新归一化为和=1               │
│  Softmax: 整行概率和=1，但截取 Top-K 后和<1，需重归一化       │
│  Sigmoid: 每个值独立∈(0,1)，K个值之和通常≠1，更需重归一化     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 分步 Kernel: moeSoftmax + moeTopK (慢路径)

当专家数不是 2 的幂且不是 64 的倍数时，退化为两步执行：

```
Step 1: moeSoftmax / moeSigmoid Kernel
┌─────────────────────────────────────────────────────────┐
│  每个 Block 处理 1 行                                       │
│                                                           │
│  moeSoftmax:                                              │
│    ① BlockReduce Max → 找行内最大值                       │
│    ② exp(x - max) + BlockReduce Sum → 归一化因子          │
│    ③ 写出: output[i] = exp(x - max) / sum               │
│                                                           │
│  moeSigmoid:                                              │
│    ① 每个线程独立: output[i] = 1/(1+exp(-input[i]))      │
│    ② NaN/Inf → 0                                         │
│                                                           │
│  输入: gating_output [T, E]                               │
│  输出: workspace [T, E] (float32 中间结果)                │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
Step 2: moeTopK Kernel
┌─────────────────────────────────────────────────────────┐
│  每个 Block 处理 1 行                                       │
│                                                           │
│  for k_idx = 0 to K-1:                                    │
│    ① BlockReduce ArgMax (cub::ArgMax) → 选最大值          │
│    ② 写出: output, indices, source_rows                   │
│    ③ 将已选专家值标记为 -1 (下次不会再选)                  │
│                                                           │
│  if renormalize: 归一化 K 个权重                           │
│                                                           │
│  输入: workspace [T, E]                                   │
│  输出: topk_weights, topk_indices, token_expert_indices   │
└─────────────────────────────────────────────────────────┘
```

### 3.3 C500 优化 Kernel: topkGatingSoftmaxDecode (Decode 场景)

针对 **Decode 阶段** (小 batch + 128 experts) 的优化 kernel。

```
topkGatingSoftmaxDecode 流程:

┌─────────────────────────────────────────────────────────────┐
│ 线程组织: 每行 4 个 Warp (128 threads)，每 CTA 2 行          │
│                                                               │
│  Warp0(32t) Warp1(32t) Warp2(32t) Warp3(32t) → 1行          │
│  Warp4(32t) Warp5(32t) Warp6(32t) Warp7(32t) → 1行          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 分段 Softmax (每个 Warp 独立计算 16 列)              │
│                                                               │
│  每个 Warp 处理 16 个 expert:                                 │
│    row_val = input[row * 128 + tid_in_row]                   │
│    ① warp 内 shfl_down_reduce_max (16→1)                     │
│    ② warp 内 shfl_down_reduce_sum (16→1)                     │
│    ③ 存入 shared memory: max_val[8], sum_val[8]              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 跨 Warp 全局归约                                     │
│                                                               │
│  前 8 个线程 (每个 Warp 的 lane 0):                           │
│    ① 从 shared mem 读取 8 组 max/sum                         │
│    ② 再次 shfl_down reduce → 全局 max, 全局 sum              │
│    ③ 计算 normalizing_factor = 1 / (sum * exp(-max))         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 归一化每个元素                                        │
│                                                               │
│  row_val = exp(row_val - global_max) * normalizing_factor    │
│  (如果 bias) row_val_for_choice = row_val + bias[expert]     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 两轮 Warp 排序选 Top-K                               │
│                                                               │
│  ① 第一轮: 每个 Warp 内排序                                  │
│    将 (weight, expert_id) 打包为 int64                        │
│    调用 warpSortDescendingNew (bitonic sort)                  │
│    每个 Warp 的 top-k 个候选存入 shared_experts              │
│                                                               │
│  ② 第二轮: 合并排序                                          │
│    从 shared_experts 加载 4*topk=32 个局部候选                │
│    再次 warpSortDescendingNew                                │
│    取最终 top-k 个结果                                        │
│                                                               │
│  ③ 写出: output, indices, source_rows                        │
│  ④ (如果 renormalize) 重归一化                               │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 C500 Sigmoid 专用 Kernel: topkGatingSigmoid288Opt

专为 **DeepSeek V3** 场景 (288 experts, topk=8) 在 MetaX C500 GPU (Warp=64) 上优化。

```
topkGatingSigmoid288Opt 流程:

┌─────────────────────────────────────────────────────────────┐
│ 线程组织:                                                    │
│   NUM_EXPERTS = 288, TOPK = 8                               │
│   WARP_SIZE_C500 = 64 (MetaX C500)                          │
│   WARPS_PER_ROW = 5 (288/64≈5)                              │
│   THREADS_PER_ROW = 5 × 64 = 320                            │
│   ROWS_PER_CTA = 2, BLOCK_THREADS = 640                     │
│                                                               │
│   Warp0(64t) Warp1(64t) Warp2(64t) Warp3(64t) Warp4(64t)   │
│        ↕           ↕           ↕           ↕           ↕     │
│   expert 0-63  expert 64-127 expert 128-191 expert 192-255  │
│   expert 256-287 (5个Warp覆盖288个expert)                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Sigmoid + Bias (每个线程处理1个 expert)               │
│                                                               │
│  if (tid_in_row < 288):                                      │
│    val = float(input[row * 288 + tid_in_row])                │
│    val = 1.0 / (1.0 + exp(-val))   ← Sigmoid               │
│    if bias: row_val_for_choice = val + bias[tid_in_row]      │
│    else:   row_val_for_choice = val                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Warp 内排序 (5 个 Warp 各自独立)                     │
│                                                               │
│  打包: int64 = (expert_id << 32) | float_bits(weight)        │
│  调用 warpSortDescendingUpdate<64bit mask>(...)               │
│  → 每个 Warp 的 64 个元素降序排列                             │
│  → 每个 Warp 的 top-8 存入 shared_experts                    │
│                                                               │
│  5 Warp × 8 = 40 个局部 Top-K 候选                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 合并排序 (Warp 0 负责)                               │
│                                                               │
│  Warp 0 的 64 个线程:                                        │
│    加载 40 个局部候选到 final_idx_weight                       │
│    再次 warpSortDescendingUpdate 排序                         │
│    取最终 top-8 个结果                                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 解包结果 + Bias 还原                                  │
│                                                               │
│  从 int64 解包:                                               │
│    expert_id_ordered = res >> 32                              │
│    max_val_ordered = get_weight(res)  ← 浮点位解读            │
│                                                               │
│  ★ Bias 还原 (关键):                                          │
│    if bias: max_val_ordered -= bias[expert_id_ordered]        │
│    → 输出权重是原始 sigmoid 值 (不含 bias)                    │
│    → bias 仅在选择过程中起作用，不影响最终权重                 │
│                                                               │
│  写出:                                                        │
│    indices[idx] = expert_id (或 NUM_EXPERTS 表示不处理)       │
│    output[idx] = max_val_ordered (or renormalize 先存 norm)   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Renormalize (可选)                                    │
│                                                               │
│  if renormalize:                                              │
│    ① 计算选中权重之和: sum = Σ(sm_norm_vals[0..7])            │
│    ② rcp_norm = 1.0 / sum (使用 __builtin_mxc_rcpf 快速倒数) │
│    ③ output[idx] = sm_norm_vals[i] * rcp_norm                │
└─────────────────────────────────────────────────────────────┘
```

### 3.5 C500 Sigmoid 通用 Kernel: topkGatingSigmoidCommonOpt

适用于 num_experts ≤ 256, topk ≤ 16 的 Sigmoid 场景，支持 MACA C500 的 64 线程 Warp。

```
topkGatingSigmoidCommonOpt 流程:

┌─────────────────────────────────────────────────────────────┐
│ 线程组织:                                                    │
│   warps_per_row = ceil(num_experts / 64)                     │
│   threads_per_row = warps_per_row * 64                       │
│   rows_per_cta = 256 / threads_per_row                       │
│                                                               │
│   例: num_experts=128 → warps_per_row=2 → threads_per_row=128│
│       rows_per_cta = 256/128 = 2                              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Sigmoid + Bias                                        │
│  (同 topkGatingSigmoid288Opt)                                │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Warp 内排序 → shared_experts                         │
│  (同 topkGatingSigmoid288Opt)                                │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 合并排序 (64 线程内)                                  │
│  (同 topkGatingSigmoid288Opt，但使用动态 shared memory)       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 解包 + Bias 还原 + 写出                              │
│  (同 topkGatingSigmoid288Opt)                                │
│                                                               │
│ Step 5: Renormalize (可选)                                    │
│  (同 topkGatingSigmoid288Opt)                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、Warp 排序算法: warpSortDescendingNew / warpSortDescendingUpdate

这两个排序函数是 C500 优化 kernel 的核心，基于 **Bitonic Sort (双调排序)** 的 Warp 级实现。

### 4.1 数据打包方式

```
int64_t packed = (expert_id << 32) | float_bits_as_int64(weight)

┌────────────────────────────────┬────────────────────────────────┐
│        高 32 位: expert_id     │       低 32 位: weight        │
│         (整数索引)             │    (float 的二进制表示)        │
└────────────────────────────────┴────────────────────────────────┘

优点: 一个 int64 可以通过 shfl_xor 一次传输 weight+index
      排序时比较 weight，相同 weight 时比较 index (小索引优先)
```

### 4.2 Bitonic Sort 流程

```
warpSortDescendingNew (32线程 Warp):

输入: 32 个线程各自持有一个 (weight, index) 对

阶段1: 构建 Bitonic 序列
  width = 2 → 4 → 8 → 16
  对每个 width:
    step = width/2 → 1
    每步: shfl_xor 交换数据，按 direction 决定升降序

阶段2: 合并排序 (全降序)
  step = 16 → 8 → 4 → 2 → 1
  每步: shfl_xor 交换数据，始终选较大值

结果: 32 个线程按 weight 降序排列
      Lane 0 = 最大值，Lane 31 = 最小值

warpSortDescendingUpdate (64线程 Warp):
  结构相同，但 WARP_SIZE=64，使用 64bit mask
  专为 MACA C500 GPU 的 64 线程 Warp 设计
```

---

## 五、Kernel 选择策略总览

```
                    topk_softmax / topk_sigmoid 调用
                              │
                              ▼
                   topkGatingKernelLauncher
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         Sigmoid?         Sigmoid?        其他
     288 experts       ≤256 experts     (Softmax/
     topk=8            topk≤16          2幂/64倍数)
              │               │               │
              ▼               ▼               ▼
     topkGating         topkGating       topkGating
     Sigmoid288Opt      SigmoidCommonOpt (融合kernel)
     (C500 专用)        (C500 通用)       │
                                         │
                              ┌──────────┼──────────┐
                              │          │          │
                          128专家    2幂专家    default
                          +Decode   (1~512)  (非2幂非64倍数)
                              │          │          │
                              ▼          ▼          ▼
                     topkGating     topkGating   moeSoftmax/
                     SoftmaxDecode  (融合)      moeSigmoid
                     (Decode优化)              + moeTopK
                                               (两步慢路径)
```

| Kernel | 适用条件 | Softmax/Sigmoid | Warp 大小 | 特点 |
|--------|---------|----------------|-----------|------|
| topkGating | 2幂/64倍数专家 | 两者都支持 | 32/64 | 通用融合 kernel |
| topkGatingSoftmaxDecode | 128 experts + Decode | Softmax | 32 | 分段 softmax + 两轮排序 |
| topkGatingSigmoid288Opt | 288 experts, topk=8 | Sigmoid | 64 | C500 专用 DeepSeek V3 |
| topkGatingSigmoidCommonOpt | ≤256 experts, topk≤16 | Sigmoid | 64 | C500 通用 Sigmoid |
| moeSoftmax + moeTopK | default | 两者都支持 | 32 | 两步分离 kernel |
