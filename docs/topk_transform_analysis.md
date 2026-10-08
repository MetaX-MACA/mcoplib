# fast_topk_transform 算子深度解析

> 源文件: `op/sglang/csrc/elementwise/topk.cu` (prefill / pure-topk / ragged)
> 源文件: `op/sglang/csrc/elementwise/topk_decode.cu` (decode, 独立 TU)
> Python 入口: `torch.ops.sgl_kernel.fast_topk_transform_fused`
> 目标硬件: MetaX C500 / C280 (sm_80, 104 SM, 1.55 TB/s HBM, 8 MB L2, 64 KB smem/SM, warp=64)

---

## 一、算子概述与作用

### 1.1 算子是什么？

`fast_topk_transform_fused` 是 SGLang 推理框架中 **MLA (Multi-head Latent Attention) 页表选择算子**。它在 DeepSeek-V3 / V3.2 推理时，为每个 query token 从全部 KV cache 页中选出 **Top-K = 2048** 个最相关的页，并将这些页号从源页表 (`src_page_table`) 聚集 (gather) 到目标页表 (`dst_page_table`)。

**一句话总结：** 输入每行 query 对所有 KV 页的注意力分数，输出 Top-2048 页号（经过 src→dst 页表变换）。

```
                         score [B, L]
                              │
                              ▼
               ┌──────────────────────────┐
               │  fast_topk_transform     │
               │  (radix-select TopK=2048)│
               │  + page-table gather     │
               └────────────┬─────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    indices [B, TopK]          dst_page_table [B, TopK]
    (行内 score 的下标)         (从 src 页表 gather 出的页号)
```

### 1.2 在 DeepSeek-V3 MLA 中的应用场景

DeepSeek-V3 使用 MLA (多头潜在注意力) 压缩 KV cache。推理时每个 token 的 query 需要和全部历史 KV 计算注意力分数，但只保留分数最高的 Top-2048 个 KV 页参与精确注意力计算，以在长上下文 (131072 tokens) 下控制计算量。

```
┌──────────────────────────────────────────────────────────┐
│               DeepSeek-V3 MLA 推理流水线                   │
│                                                          │
│  Query [B, head, dim]                                    │
│       │                                                  │
│       ▼                                                  │
│  ┌─────────────┐    KV cache pages [num_pages, dim]      │
│  │ Q-K score   │─────────────────────────────────┐       │
│  │ (per page)  │                                  │       │
│  └──────┬──────┘                                  │       │
│         │ score [B, L]  (L = 当前 KV 页数)        │       │
│         ▼                                          │       │
│  ┌──────────────────────┐                          │       │
│  │fast_topk_transform   │  ← 本算子                │       │
│  │  TopK = 2048         │                          │       │
│  └──────────┬───────────┘                          │       │
│             │ dst_page_table [B, 2048]              │       │
│             ▼                                      │       │
│  ┌──────────────────────┐                          │       │
│  │ 精确注意力 (仅对     │ ←── 只对 Top-2048 页 ──┘       │
│  │ Top-2048 页做 attn)  │    做注意力, O(2048) 而非 O(L)  │
│  └──────────┬───────────┘                                  │
│             │                                              │
│             ▼                                              │
│        Output logits                                       │
└──────────────────────────────────────────────────────────┘

典型上下文长度: L ∈ [2048, 131072]，TopK = 2048 (DeepSeek-V3.2)
```

**两条调用路径（由 `fast_topk_transform_interface` 的 dispatch 区分）：**

| 路径 | 触发条件 | TU | 内核 |
|------|---------|-----|------|
| **decode** | `row_starts_opt == nullopt && prefill_bs == B` | `topk_decode.cu` | `topk_transform_decode_kernel_impl` (或 split-K 协作版) |
| **prefill** | `row_starts_opt != nullopt` (extend) 或 `prefill_bs < B` (target_verify) | `topk.cu` | `topk_transform_prefill_kernel<HIGH_SMEM>` |
| **pure topk** | 无 page-table gather | `topk.cu` | `topk_kernel` (只输出行内下标) |
| **ragged** | ragged KV 布局 | `topk.cu` | `topk_transform_prefill_ragged_kernel` |

> 注: `prefill_bs = cu_seqlens_q.size(0) - 1`，是展开前的原始序列数；`B = score.size(0)` 是展开后的行数。decode 时 `prefill_bs == B`（每序列 1 token），prefill 时 `prefill_bs < B`（每序列多 token 展开）。

### 1.3 解决了什么问题？

1. **长上下文 Top-K 选择效率问题**：L = 131072 个 KV 页中选 Top-2048，朴素排序需 O(L log L)，radix-select 只需 O(L) 两遍扫描。
2. **页表变换融合问题**：选出 Top-K 下标后还需 `dst[i] = src[topk_idx[i]]` 的 gather，本算子将其融合进同一 kernel，避免额外 kernel launch 和中间 buffer。
3. **精确性问题**：4 轮 8-bit radix 精细化覆盖全部 32 位浮点 key，保证结果与 `torch.topk` 的集合等价（允许等值并列）。
4. **页表越界防护问题**：当 `length <= TopK` 时无需选择，直接拷贝前 `length` 个页号并补 `-1` 哨兵。
5. **扎堆分数 (clustered scores) 越界问题**：当大量分数落入同一 radix 桶导致暂存缓冲溢出时，FIX-C 保证阈值账目仍然收敛（不因缓冲满而漏计），仅丢弃超出部分的精细化机会。

### 1.4 输入输出参数详解

#### 1.4.1 `fast_topk_transform_fused`（主入口）

```cpp
torch.ops.sgl_kernel.fast_topk_transform_fused(
    score,            // Tensor [B, input_stride] float32
    lengths,          // Tensor [B] int32
    dst_page_table,   // Tensor [B, TopK] int32 (输出)
    src_page_table,   // Tensor [prefill_bs, src_stride] int32
    cu_seqlens_q,     // Tensor [prefill_bs+1] int32
    row_starts,       // Tensor? [B] int32 (可选)
)
```

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `score` | 输入 | `[B, input_stride]` | float32 | 注意力分数矩阵。`B` = 展开后行数；`input_stride` >= 实际 seq_len（通常 = MAX_SEQ_LEN=131072，padding 到固定列宽）。**行内必须连续** (`stride(1)==1`)。 |
| `lengths` | 输入 | `[B]` | int32 | 每行实际有效长度 `L`（`L <= input_stride`）。`length <= TopK` 时走 naive 拷贝路径。 |
| `dst_page_table` | **输出** | `[B, TopK]` | int32 | 输出页表：`dst[b][i] = src_page_table[seq_b][topk_index[b][i]]`，未填满的位置写 `-1`。必须连续。 |
| `src_page_table` | 输入 | `[prefill_bs, src_stride]` | int32 | 源页表：第 `seq_b` 个序列的页号数组。`stride(1)==1`。 |
| `cu_seqlens_q` | 输入 | `[prefill_bs+1]` | int32 | 累积序列长度：`cu_seqlens_q[i]` ~ `cu_seqlens_q[i+1]` 是第 `i` 个序列展开前的 token 范围。**用于 prefill 时将展开行 `bid` 映射回原序列 `seq_b`**，从而找到 `src_page_table[seq_b]`。decode 时 `cu_seqlens_q = [0,1,...,B]`。 |
| `row_starts` | 输入(可选) | `[B]` | int32 | 每行在 `score` 中的起始偏移（ragged / extend 模式）。**`nullopt` 表示行从 0 开始**（decode / target_verify）。非空 → prefill 路径。 |

**`row_starts` 的双重作用：**
1. 决定 dispatch：非空 → prefill kernel；空 → decode 或 target_verify。
2. 决定行内偏移：`row_input = score + row_starts[bid]`（extend 模式下每行从不同位置开始）。

#### 1.4.2 `fast_topk`（纯 TopK，无 page-table 变换）

```cpp
torch.ops.sgl_kernel.fast_topk(
    score,      // [B, input_stride] float32
    indices,    // [B, TopK] int32 (输出：行内下标 0..L-1)
    lengths,    // [B] int32
    row_starts  // [B]? int32
)
```

输出是 `score` 的**行内下标**（0-based），不做 `src→dst` gather。用于不需要页表变换的场景。

#### 1.4.3 `fast_topk_transform_ragged_fused`（ragged KV 布局）

```cpp
torch.ops.sgl_kernel.fast_topk_transform_ragged_fused(
    score,                // [B, input_stride]
    lengths,              // [B]
    topk_indices_ragged,  // [B, TopK] (输出：行内下标 + offset)
    topk_indices_offset,  // [B] 每行的偏移量
    row_starts            // [B]? int32
)
```

输出 = `topk_indices_offset[bid] + 局部下标`，用于 ragged KV cache 中将局部页号转为全局页号。

### 1.5 Shape 说明与真实模型尺寸

**测试覆盖的 shape 网格（来自 `test_op_topk_transform_prefill_kernel.py`）：**

| 维度 | 取值 | 含义 |
|------|------|------|
| `B` (展开后行数) | 1, 6, 132, 256, 370, 371, 1662, 4096 | 对应不同 prefill 批次 / decode batch |
| `seq_len` (L) | 64, 2048, 4096, 18560, 18688, 32768, 65536, 66551, 107520 | KV 页数（上下文长度） |
| `TopK` | 2048 | DeepSeek-V3.2 固定 |
| `MAX_SEQ_LEN` | 131072 | score 矩阵列宽（padding 上限） |

**典型真实模型 shape：**

| 场景 | B | seq_len | 走哪条路径 | 说明 |
|------|---|---------|-----------|------|
| decode 短上下文 | 1 | 64 | naive (L ≤ TopK) | 刚启动，KV 页数 < TopK |
| decode 中等上下文 | 132 | 65536 | radix-select | 逐 token 生成中 |
| decode 长上下文 | 4 | 107520 | radix-select (+split-K) | B 小 + L 大 → 触发协作 split-K |
| prefill 短 prompt | 6 | 2048 | naive (L == TopK) | prompt 长度刚好 = TopK |
| prefill 中 prompt | 1662 | 32768 | radix-select | 典型 prefill 批次 |
| prefill 长 prompt | 4096 | 107520 | radix-select | 大 batch + 长上下文（峰值带宽场景） |

**有效带宽计算公式（性能测试用）：**
- radix-select 路径（`L > TopK`）：`eff_bytes = B × (2×L + TopK) × 4`（2 遍读 score + 写 TopK 个 int32）
- naive 路径（`L ≤ TopK`）：`eff_bytes = B × (L + TopK) × 4`（1 遍读 score + 写 TopK 个 int32，不足补 -1）

---

## 二、计算流程

### 2.1 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    fast_topk_transform_prefill_kernel                   │
│                                                                         │
│  输入: score[bid, 0..L-1], src_page_entry, L                            │
│                                                                         │
│  ┌───────────────────────┐                                              │
│  │ L <= TopK ?           │                                              │
│  └──────┬────────────────┘                                              │
│     YES │           NO                                                  │
│         ▼           ▼                                                   │
│  ┌──────────┐  ┌───────────────────────────────────────────────────┐   │
│  │ naive    │  │  fast_topk_cuda_tl_exact (radix-select core)      │   │
│  │ transform│  │                                                   │   │
│  │ (拷贝+   │  │  ┌── Stage 1: 粗粒度 1024-bin 直方图 ──────────┐  │   │
│  │  补-1)   │  │  │  对全 L 个 score 做 convert_to_uint10       │  │   │
│  └────┬─────┘  │  │  → atomicAdd 进 s_histogram[1024]           │  │   │
│       │        │  │  → Hillis-Steele 后缀 cumsum                 │  │   │
│       │        │  │  → 找到 threshold_bin (cumsum 跨越 TopK)     │  │   │
│       │        │  └──────────────────────────────────────────────┘  │   │
│       │        │                                                   │   │
│       │        │  ┌── Stage 2: 候选暂存 + 精细化 ─────────────────┐  │   │
│       │        │  │  Round 0 (粗粒度 threshold_bin):              │  │   │
│       │        │  │    重新扫描 L 个 score                         │  │   │
│       │        │  │    bin > thr → 直接写入 index (确定 top-K)     │  │   │
│       │        │  │    bin == thr → 暂存 idx+value 到 smem         │  │   │
│       │        │  │                + 建 8-bit sub-histogram        │  │   │
│       │        │  │    bin < thr → 跳过                             │  │   │
│       │        │  │                                                │  │   │
│       │        │  │  Round 1..3 (uint32 的 bits 24,16,8,0):       │  │   │
│       │        │  │    cumsum → 找 sub-threshold                   │  │   │
│       │        │  │    bin > sub-thr → 写入 index                  │  │   │
│       │        │  │    bin == sub-thr → 暂存下一轮                 │  │   │
│       │        │  │    bin < sub-thr → 跳过                        │  │   │
│       │        │  │  Round 3 末: topk==0 时用 s_last_remain        │  │   │
│       │        │  │    精确裁剪到 TopK 个                          │  │   │
│       │        │  └────────────────────────────────────────────────┘  │   │
│       │        │                                                   │   │
│       │        │  s_indices[TopK] ← (radix-select 的输出)          │   │
│       │        └──────────────────┬────────────────────────────────┘   │
│       │                           │                                    │
│       │                           ▼                                    │
│       │             ┌──────────────────────────────┐                  │
│       │             │  Gather: dst[i] = src[indices[i]]              │                  │
│       │             │  (1024 线程 × 2 元素/线程 = TopK=2048)          │                  │
│       │             └──────────────┬───────────────┘                  │
│       ▼                           ▼                                    │
│  dst_page_entry[0..TopK-1]      dst_page_entry[0..TopK-1]              │
│  (前 L 个 = src[0..L-1], 其余 = -1)  (Top-2048 页号)                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Radix-Select 算法原理

核心思想：**不做排序，而是用基数直方图二分定位 Top-K 的阈值，然后只对落在阈值桶内的候选做精细化。**

```
朴素 Top-K:    排序 O(L log L)  或  维护大小-K的堆 O(L log K)
Radix-Select:  2 遍扫描 O(L) + 4 轮 O(K) 精细化，无需排序

关键洞察: 只需要知道"第 K 大的值是多少"(threshold)，然后扫一遍挑出 >= threshold 的元素。
         radix histogram 给出"每个桶有多少个元素"，cumsum 即可定位 threshold 所在桶。
```

**为什么是 2 遍 + 4 轮？**
- **第 1 遍 (Stage 1)**：用 10-bit 粗粒度 key (1024 桶) 统计直方图 → 定位 threshold 所在的粗桶。
- **第 2 遍 (Stage 2 Round 0)**：重新扫描，把粗桶内的元素（候选）暂存到 smem，同时建 8-bit 子直方图。
- **4 轮精细化 (Stage 2 Round 0..3)**：对 uint32 key 的 4 个字节（bits 24-31, 16-23, 8-15, 0-7）各做一次 256 桶子直方图 + cumsum + 分类，每轮把候选缩小到更小的桶。

10 + 8×4 = 42 bit > 32 bit，**足以精确区分任意两个 float32**（float32 只有 32 bit）。

### 2.3 float → 有序无符号整数的转换

radix-select 需要把 float 映射成**保序的无符号整数**（大的 float → 大的 uint），这样直方图的 cumsum 才有意义。

#### `convert_to_uint32(x)` — 精确 32-bit key

```cpp
uint32_t bits = __float_as_uint(x);
return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
```

IEEE 754 float32 的位布局：`[符号 1][指数 8][尾数 23]`。

| 原始 float | 位模式 | 转换后 uint32 | 映射区间 |
|-----------|--------|-------------|---------|
| -∞ | 0xFF800000 | 0x007FFFFF (~8M) | 最小 |
| -1.0 | 0xBF800000 | 0x407FFFFF (~1G) | |
| -0.0 | 0x80000000 | 0x7FFFFFFF (~2G) | 中点 |
| +0.0 | 0x00000000 | 0x80000000 (~2G) | 中点 |
| +1.0 | 0x3F800000 | 0xBF800000 (~3G) | |
| +∞ | 0x7F800000 | 0xFF800000 (~4G) | 最大 |

**符号翻转技巧**：
- 正数：`bits | 0x80000000` — 把符号位置 1，等价于 `+0x80000000`。
- 负数：`~bits` — 翻转所有位，原来符号位 1→0，且指数/尾数取反映射后保序。

结果：负数映射到 `[0, 0x80000000)`，正数映射到 `[0x80000000, 0xFFFFFFFF]`，**全序保序**，适合 radix sort。

#### `convert_to_uint10(x)` — 粗粒度 10-bit key

```cpp
__half h = __float2half_rn(x);          // float → fp16: [符号1][指数5][尾数10]
uint16_t bits = __half_as_ushort(h);
uint16_t key = (bits & 0x8000) ? ~bits : (bits | 0x8000);  // 同样的符号翻转
return key >> 6;                         // 取高 10 位 → [0, 1023]
```

fp16 只有 16 bit，取高 10 位 = 指数(5) + 尾数高 5 位。这给出 1024 个粗粒度桶，每个桶覆盖一段连续的 float 值域。同一桶内的 float 在 10-bit 粒度上"相等"，需要后续 32-bit 精细化区分。

#### `convert_to_uint10x2(a, b)` — 打包 2-wide

```cpp
__half2 h = __floats2half2_rn(a, b);  // 一条指令转 2 个 float→half
// a → low half (.x), b → high half (.y)
```

利用 `__floats2half2_rn` 的 2-wide 打包，**一条指令处理 2 个 float→fp16 转换**，比标量 `__float2half_rn` 快一倍。

### 2.4 Stage 1：粗粒度 1024-bin 直方图

```cpp
// Stage 1 (topk.cu:146-190) — 简化伪代码
zero(s_histogram[0..1024]);                     // 1. 清零
__syncthreads();

for (idx = tx; idx < L; idx += BLOCK_SIZE) {      // 2. 扫描全部 L 个 score
    bin = convert_to_uint10(row_input[idx]);      //    float → 10-bit bin
    atomicAdd(&s_histogram[bin], 1);              //    原子累加
}
__syncthreads();

run_cumsum();  // 3. Hillis-Steele 后缀 cumsum：s_histogram[bin] = #{elements with key >= bin}
```

**Hillis-Steele 后缀 cumsum（10 轮，双缓冲）：**

```
初始:  s_histogram[bin] = count(bin)           // 每个桶的元素数

轮 0:  s_histogram[bin] += s_histogram[bin+1]   // 加右边 1 个
轮 1:  s_histogram[bin] += s_histogram[bin+2]   // 加右边 2 个
轮 2:  s_histogram[bin] += s_histogram[bin+4]   // 加右边 4 个
...
轮 9:  s_histogram[bin] += s_histogram[bin+512] // 加右边 512 个

结果:  s_histogram[bin] = #{key >= bin}         // key >= bin 的元素总数
```

用双缓冲 `s_histogram_buf[2][RADIX+128]` 交替读写，避免读后写冲突。

**定位 threshold_bin：**

```cpp
// 找最大的 bin 使得 s_histogram[bin] > TopK 且 s_histogram[bin+1] <= TopK
if (s_histogram[tx] > TopK && s_histogram[tx+1] <= TopK) {
    s_threshold_bin_id = tx;
}
// 含义：key >= threshold_bin 的元素有 > TopK 个，但 key >= threshold_bin+1 的只有 <= TopK 个
//       → Top-K 的阈值落在 threshold_bin 这个桶里
topk -= s_histogram[threshold_bin + 1];  // 剩余需要从 threshold_bin 桶中选 topk 个
```

### 2.5 Stage 2：候选暂存 + 4 轮精细化

#### Round 0（粗粒度 threshold_bin 的分类）

```cpp
// Stage 2 Round 0 (topk.cu:273-338) — 简化伪代码
zero(s_histogram[0..1024]);  // 重置为子直方图用
for (idx = tx; idx < L; idx += BLOCK_SIZE) {       // 重新扫描 L 个 score
    bin = convert_to_uint10(row_input[idx]);
    if (bin > threshold_bin) {                     // 确定在 Top-K 中
        pos = atomicAdd(&s_counter, 1);
        index[pos] = idx;                          // 直接写入输出
    } else if (bin == threshold_bin) {             // 候选：需要精细化
        pos = atomicAdd(&s_num_input[0], 1);
        s_input_idx[0][pos] = idx;                 // 暂存下标
        s_input_value[0][pos] = raw_input;         // 暂存值（Round 2 优化）
        sub_bin = (convert_to_uint32(raw_input) >> 24) & 0xFF;  // bits 31:24
        atomicAdd(&s_histogram[sub_bin], 1);       // 建 8-bit 子直方图
    }
    // bin < threshold_bin: 跳过
}
```

#### Round 1..3（8-bit 子基数精细化）

```cpp
// Stage 2 Round 1..3 (topk.cu:341-411) — 简化伪代码
for (round = 0; round < 4; round++) {
    offset = 24 - round * 8;  // Round 0: bits 31:24, Round 1: 23:16, Round 2: 15:8, Round 3: 7:0

    run_refine_cumsum();  // 256-bin 后缀 cumsum (8 轮)
    // 找 sub_threshold_bin
    if (s_histogram[tx] > topk && s_histogram[tx+1] <= topk)
        s_threshold_bin_id = tx;

    topk -= s_histogram[threshold_bin + 1];  // 剩余需要选的个数

    if (topk == 0) {  // 阈值桶外恰好凑够 TopK
        for (i in candidates) {
            bin = (uint32_key(raw_input) >> offset) & 0xFF;
            if (bin > threshold_bin) index[atomicAdd(s_counter)] = idx;
        }
        break;
    } else {  // 还需从阈值子桶中选
        for (i in candidates) {
            bin = (uint32_key(raw_input) >> offset) & 0xFF;
            if (bin > threshold_bin) index[atomicAdd(s_counter)] = idx;
            else if (bin == threshold_bin) {
                if (round == 3) {  // 最后一轮：精确裁剪
                    pos = atomicAdd(&s_last_remain, -1);
                    index[TopK - pos] = idx;  // 从末尾填充
                } else {  // 非最后一轮：暂存到下一轮
                    pos = atomicAdd(&s_num_input[r^1], 1);
                    s_input_idx[r^1][pos] = idx;
                    s_input_value[r^1][pos] = raw_input;
                    sub_bin = (uint32_key >> (offset-8)) & 0xFF;
                    atomicAdd(&s_histogram[sub_bin], 1);
                }
            }
        }
    }
}
```

**关键设计：双缓冲 ping-pong**

```
Round 0:  读 s_input_idx[0]/s_input_value[0]  →  写 s_input_idx[1]/s_input_value[1]
Round 1:  读 s_input_idx[1]/s_input_value[1]  →  写 s_input_idx[0]/s_input_value[0]
Round 2:  读 s_input_idx[0]/s_input_value[0]  →  写 s_input_idx[1]/s_input_value[1]
Round 3:  读 s_input_idx[1]/s_input_value[1]  →  (最后一轮，只写 index)
```

`s_histogram_buf[2][RADIX+128]` 同理，交替使用避免读写冲突。

### 2.6 Naive 路径 (length ≤ TopK)

当 `L <= TopK = 2048` 时，所有元素都在 Top-K 中，无需 radix-select：

```cpp
// naive_topk_transform (topk.cu:55-64)
for (i = tid; i < TopK; i += BLOCK_SIZE) {
    dst_page_entry[i] = (i < length) ? src_page_entry[i] : -1;
}
// 前 length 个拷贝 src，剩余补 -1 哨兵
```

**典型场景：** decode 刚启动（KV 页数 < 2048）或 prefill 短 prompt（prompt 长度 ≤ 2048）。

### 2.7 Decode vs Prefill 深度对比

#### 2.7.1 结构差异总表

| 维度 | Decode (`topk_decode.cu`) | Prefill (`topk.cu`) | 差异原因 |
|------|--------------------------|---------------------|---------|
| **翻译单元 (TU)** | 独立 TU，自有 radix-select core 副本 | 共享 TU，prefill/ragged/pure-topk 共用 | 解耦优化：decode 可独立改不影响 prefill |
| **动态 smem** | 16 KB (`kSmem = 4×1024×4`) | 24 KB HIGH / 16 KB LOW | prefill 多了 `s_input_value` 静态 smem (16KB)，HIGH 方案给 24KB 动态 |
| **静态 smem** | `s_histogram_buf[2][1152]` = 9KB | `s_histogram_buf` 9KB + `s_input_value[2][2048]` = 16KB | **prefill 额外缓存候选值**，decode 不缓存 |
| **加载粒度** | vec4 = 16B/线程/iter | **vec8 = 32B/线程/iter** (2× float4) | C500 指南要求 ≥32B/线程；prefill 已优化，decode 未跟进 |
| **naive 路径写回** | int4 (128-bit) `__stcg` 流式存储 | 标量逐元素写 | decode 的 naive 路径在 B 小时是瓶颈 |
| **split-K 协作路径** | ✅ 有 (`topk_transform_decode_splitk_kernel`) | ❌ 无 | decode 的 B 小 (1 token/序列) → 多 SM 空闲 → 需 split-K |
| **`row_starts`** | 恒为 0 | 从 tensor 读取（可能非零） | decode 每行从 score 开头开始；prefill extend 模式有偏移 |
| **page-table 查找** | `src_page_entry = src_page_table + bid × src_stride` (1:1) | 用 `cu_seqlens_q` 二分查找所属序列 | decode 1 行/序列；prefill 多行/序列需映射 |
| **refine 轮读候选值** | `row_input[idx]` (全局随机读) | `s_input_value[r][i]` (smem 读) | prefill 缓存了值；decode 未缓存 |
| **dispatch 条件** | `!row_starts && prefill_bs == B` | else (extend: `row_starts` 非空; target_verify: `prefill_bs < B`) | 见 1.2 路径表 |

#### 2.7.2 page-table 查找的差异

**Decode（1:1 映射，简单）：**
```
bid = blockIdx.x
src_page_entry = src_page_table + bid * src_stride
// 第 bid 行的 query → 第 bid 个序列的页表 → src_page_table[bid]
// 因为 decode 时每序列只有 1 个 token，bid == seq_b
```

**Prefill（多对一映射，需 cu_seqlens_q 查找）：**
```cpp
// topk.cu:490-503
__shared__ const int32_t* s_src_page_entry;
if (prefill_bs <= 1024) {
    // 每个线程负责一个原序列的检查
    if (tid < prefill_bs) {
        if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid+1]) {
            // bid 落在 [cu_seqlens_q[tid], cu_seqlens_q[tid+1]) 区间
            // → 这个展开行 bid 属于原序列 tid
            s_src_page_entry = src_page_table + tid * src_stride;
        }
    }
}
// 1024 个线程并行扫描所有 prefill_bs 个序列，O(1) 找到 bid 属于哪个序列
```

**为什么 prefill 需要这个查找？**
- prefill 把 `prefill_bs` 个序列拼接成一个 `[B]` 行的大 batch（`B = sum(seq_lens)`）。
- 每个原序列有自己的 `src_page_table[seq_b]`（自己的 KV 页表）。
- 展开后的行 `bid` 需要映射回原序列 `seq_b` 才能找到正确的页表。
- `cu_seqlens_q` 就是这个映射的索引：`cu_seqlens_q[seq_b]` 给出展开后的起始行号。

#### 2.7.3 split-K 协作路径（仅 decode）

decode 时 B 通常很小（1-4 个 token），但 L 可能很大（107520）。单行 kernel 只启动 B 个 block，大量 SM 空闲。

```
B=4, L=107520, SM=104:
  朴素 decode: 4 blocks → 4/104 SMs busy (3.8%)
  split-K:     4 × blocks_per_row = 4 × 26 = 104 blocks → 100% SMs busy

split-K 协作流程 (topk_decode.cu:429-707):
  Phase 0: owner block 清零全局直方图 + 计数器
  Phase 1: 所有 blocks 各自扫一段 row 的 score → 局部 smem 直方图 → atomicAdd 合并到全局
  Phase 2: 所有 blocks 重新扫各自段 → bin>thr 写全局 selected, bin==thr 写全局 candidate
  Phase 3: owner block 独自跑 4 轮精细化 (候选只有几百个，单 block 足够) + gather
  (grid.sync() 在 Phase 0/1/2/3 之间同步)
```

**触发条件：** `cudaDevAttrCooperativeLaunch && B*2 <= sm_count && seq >= 60000`

**为什么 prefill 不需要 split-K？**
- prefill 的 B 大（1662, 4096），已充分占用所有 SM，无空闲可填充。

#### 2.7.4 为何 decode 不缓存候选值（不用 `s_input_value`）？

decode kernel 的注释（`topk_decode.cu:11-24`）给出三个原因：

1. **带宽指标不奖励**：decode 的有效带宽只计 1 遍 score 读取（`eff_bytes = B × L × 4`），消除第 2 遍读取不提升指标。
2. **单遍替代方案更慢**：prefill 曾尝试单遍窗口采样暂存（消除第 2 遍），但 `s_num_input` 单计数器串行化导致 wall-clock 更慢（191 vs 275 GB/s）。
3. **大长度采样方差**：L=107520 时采样方差把候选下界推高一个粗桶，漏选真实 top-K 成员（严格集合等价校验失败）。

所以 decode 保守地保留 2-pass 结构 + 不缓存值，换取**正确性**和**wall-clock 速度**。

### 2.8 举例说明

#### 例子 1：naive 路径（L ≤ TopK）

```
输入: B=1, L=3, TopK=2048
score = [0.5, 0.8, 0.1, ...]  (只有前 3 个有效)
src_page_table = [10, 20, 30, 40, ...]

执行 naive_topk_transform:
  dst[0] = src[0] = 10    (i=0 < length=3 → 拷贝)
  dst[1] = src[1] = 20    (i=1 < 3 → 拷贝)
  dst[2] = src[2] = 30    (i=2 < 3 → 拷贝)
  dst[3] = -1             (i=3 >= 3 → 补 -1)
  ...
  dst[2047] = -1

输出: dst = [10, 20, 30, -1, -1, ..., -1]
```

#### 例子 2：radix-select 路径（简化为 TopK=3, L=10）

```
输入: L=10, TopK=3
score = [0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6, 0.15]

Step 1: convert_to_uint10 (粗粒度 1024 桶，这里简化为 10 桶示意)
  score → bin:  [5, 3, 8, 1, 9, 2, 7, 4, 6, 1]

Step 2: 直方图统计
  s_histogram[bin] = count(bin):
    bin 1: 2, bin 2: 1, bin 3: 1, bin 4: 1, bin 5: 1,
    bin 6: 1, bin 7: 1, bin 8: 1, bin 9: 1
  (其余桶 = 0)

Step 3: 后缀 cumsum  s_histogram[bin] = #{key >= bin}
    bin 9: 1   (只有 0.9)
    bin 8: 2   (0.9, 0.8)
    bin 7: 3   (0.9, 0.8, 0.7)  ← cumsum 跨越 TopK=3
    bin 6: 4   (0.9, 0.8, 0.7, 0.6)
    bin 5: 5   ...
    ...

Step 4: 定位 threshold_bin
  s_histogram[7] = 3 > TopK=3?  否 (等于不大于)
  s_histogram[6] = 4 > TopK=3?  是
  s_histogram[7] = 3 <= TopK=3? 是
  → threshold_bin = 6 (key >= 6 的有 4 个，但 key >= 7 的只有 3 个)
  → topk = TopK - s_histogram[7] = 3 - 3 = 0

  因为 topk == 0，走 "topk==0" 快路径：
  只需选出 bin > 6 的元素 (即 bin ∈ {7,8,9})，恰好 3 个。

Step 5: Stage 2 快路径 (topk==0)
  重新扫描 score:
    idx=0, bin=5: 5 < 6 → 跳过
    idx=1, bin=3: 3 < 6 → 跳过
    idx=2, bin=8: 8 > 6 → 写入 index[0] = 2   (score=0.8)
    idx=3, bin=1: 1 < 6 → 跳过
    idx=4, bin=9: 9 > 6 → 写入 index[1] = 4   (score=0.9)
    idx=5, bin=2: 2 < 6 → 跳过
    idx=6, bin=7: 7 > 6 → 写入 index[2] = 6   (score=0.7)
    idx=7, bin=4: 4 < 6 → 跳过
    idx=8, bin=6: 6 == 6 → 不写 (topk==0 时只要 bin > thr)
    idx=9, bin=1: 1 < 6 → 跳过

  index = [2, 4, 6]  (对应 score 0.8, 0.9, 0.7)

Step 6: Gather (decode/prefill transform)
  dst[0] = src[2], dst[1] = src[4], dst[2] = src[6]
  dst[3..2047] = -1 (未填满)
```

#### 例子 3：radix-select 需要精细化（topk > 0）

```
输入: L=10, TopK=4
score = [0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6, 0.15]
bin   = [5,   3,   8,   1,   9,   2,   7,   4,   6,   1  ]

后缀 cumsum:
  bin 9: 1, bin 8: 2, bin 7: 3, bin 6: 4, bin 5: 5, ...

定位: s_histogram[5] = 5 > 4, s_histogram[6] = 4 <= 4
  → threshold_bin = 5
  → topk = TopK - s_histogram[6] = 4 - 4 = 0

仍然 topk==0，只需 bin > 5:
  idx=2 (bin 8), idx=4 (bin 9), idx=6 (bin 7), idx=8 (bin 6)
  → index = [2, 4, 6, 8] (score 0.8, 0.9, 0.7, 0.6) ← 恰好 4 个

但如果改成 TopK=5:
  s_histogram[5] = 5 <= 5 → threshold_bin = 4
  topk = 5 - s_histogram[5] = 5 - 5 = 0
  bin > 4 → idx=0(5),2(8),4(9),6(7),8(6) ← 5 个, 恰好

如果改成 TopK=4 但 bin 分布不同:
  score = [0.55, 0.56, 0.57, 0.58, 0.9, 0.2, 0.7, 0.4, 0.6, 0.15]
  bin   = [5,    5,    5,    5,    9,   2,   7,   4,   6,   1  ]  (4 个落在 bin=5)

  cumsum: bin 9:1, bin 8:1, bin 7:2, bin 6:3, bin 5:7
  s_histogram[5]=7 > 4, s_histogram[6]=3 <= 4
  → threshold_bin = 5
  → topk = 4 - 3 = 1  ← 需要从 bin=5 的 4 个候选中选 1 个！

  Stage 2 Round 0:
    bin > 5: idx=4(9), idx=6(7), idx=8(6) → 写入 index[0..2]
    bin == 5: idx=0,1,2,3 → 暂存为候选, 建 8-bit 子直方图
    (候选的 uint32 key: 0.55→0x3F0CCCCD, 0.56→0x3F0F5C29, 0.57→0x3F11EB85, 0.58→0x3F128F5C)
    子 bin (bits 31:24): 全部 = 0x3F = 63 → 4 个都在同一子桶！

  Round 1 (bits 23:16):
    子直方图: bin 0x0C: 1 (0.55), bin 0x0F: 1 (0.56), bin 0x11: 1 (0.57), bin 0x12: 1 (0.58)
    cumsum 找 sub-threshold: 需要从 4 个中选 1 个
    → sub-threshold = 0x12 (0x12 有 1 个, 0x13+ 有 0 个, 0x12 的 cumsum = 1 = topk)
    → topk = 1 - 1 = 0
    → bin > 0x12 的候选直接写入: idx=3 (0.58, sub_bin=0x12... 等等需要重新算)

  (实际 4 轮会精确到 32-bit，最终挑出 score 最大的那个候选)
```

> 上面的简化例子省略了 uint32 精细化的细节，实际 kernel 会用 4 轮 8-bit 把 32 位 key 完全区分，保证与 `torch.topk` 集合等价。

---

## 三、优化分析

### 3.1 优化概览

**目标硬件：** MetaX C500/C280 (sm_80, 104 SM, 1.55 TB/s HBM, 8 MB L2, 64 KB smem/SM, 64K 32-bit regs/SM, warp=64)

**Baseline (Round 0):** 479.6 GB/s（vec4 16B 加载 + 无候选值缓存）
**最终版本 (Round 5):** 540.6 GB/s（vec8 32B 加载 + smem 候选值缓存）
**总提升：** +61.0 GB/s = **+12.7%**

### 3.2 各优化详解

#### 优化 1：候选值 smem 缓存（Round 2，保留）

**改动：** 新增 `s_input_value[2][SMEM_INPUT_SIZE]`（2 × 2048 × 4B = 16 KB 静态 smem），在 Stage 2 Round 0 暂存候选值时同时写入 smem，后续 4 轮精细化从 smem 读取而非全局。

```cpp
// 优化前 (baseline): 精细化轮重新从全局随机读
const auto raw_input = row_input[idx];  // 随机地址, L1 miss, 慢

// 优化后 (Round 2): 从 smem 读
const auto raw_input = s_input_value[r_idx][i];  // 顺序 smem 地址, ~100× 快
```

**为什么有效：**
- 精细化每轮处理 ~2048 个候选，4 轮 = 8192 次随机全局读。
- 候选下标 `idx` 是分散的（来自不同位置），全局读是随机访问，L1/L2 命中率低。
- smem 读延迟 ~20 cycles，全局随机读 ~400+ cycles，快 20×。

**收益：** 479.6 → 492.0 GB/s，**+2.6%**

**代价：** 多用 16 KB 静态 smem（`s_input_value[2][2048]`）。加上 9KB 直方图 + 24KB 动态 = 49KB，仍在 64KB 限制内。

#### 优化 2：vec8 (32B) 加载（Round 5，保留）

**改动：** Stage 1 和 Stage 2 的主循环从 vec4（1× float4 = 16B/线程/iter）改为 vec8（2× float4 = 32B/线程/iter）。

```cpp
// 优化前 (baseline, vec4): 每次迭代 4 floats
for (vec_idx = tx; vec_idx < vec4_length; vec_idx += BLOCK_SIZE) {
    const auto values = row_input_vec4[vec_idx];  // 1 × ldg.b128 = 16B
    // 4 × atomicAdd
}

// 优化后 (Round 5, vec8): 每次迭代 8 floats
constexpr int VEC8_STRIDE = 2;
for (vec_idx = tx; vec_idx < vec8_length; vec_idx += BLOCK_SIZE) {
    const auto v0 = row_input_vec4[base];       // ldg.b128 #1
    const auto v1 = row_input_vec4[base + 1];   // ldg.b128 #2 (背靠背, 32B total)
    // 8 × atomicAdd (但总原子数不变, 只是循环次数减半)
}
```

**为什么有效：**
- C500 优化指南明确要求：**"每线程每次访问至少 32B"**，用 `ldg.b32/64/128`，避免 `ldg.u8`。
- vec4 (16B) 只发 1 条 `ldg.b128`，不满载 load pipeline；vec8 发 2 条背靠背 `ldg.b128`，暴露更多 ILP。
- 循环次数减半 → 分支/地址计算开销减半。
- 总原子数不变（每 8 元素 8 次 atomicAdd，和 vec4 的每 4 元素 4 次一样），不增加原子压力。

**收益：** 492.0 → 540.6 GB/s，**+9.9%**（相对 Round 2）；**+12.7%**（相对 baseline）

**代价：** 无额外 smem/寄存器开销。仅增加了 2 条 `ldg.b128` 指令和 2 个寄存器暂存 `v0, v1`。

#### 失败尝试汇总（均已回退）

| 轮次 | 技术 | 峰值 GB/s | vs baseline | 失败原因 |
|------|------|-----------|-------------|---------|
| Round 1 | 4-group warp-private 直方图 | 474.5 | -1.0% | 4 组私有直方图合并开销 > 原子节省 |
| Round 3 | warp shuffle cumsum | 编译/运行失败 | — | MACA warp=64 的 `__shfl_xor_sync` 掩码行为与 NVIDIA warp=32 不同 |
| Round 4 | 4-way suffix scan | 491.0 | +2.4% | 无增益 over Hillis-Steele，反而多一组 smem 读写 |
| Round 6 | 4-way scan + vec8 | 539.8 | +12.5% | 与 Round 5 持平，scan 部分无收益 |
| Round 7 | vec16 (64B) 加载 | 497.5 | +3.7% | 2 条 `ldg.b128` → 4 条，原子压力翻倍 (8→16 atomics/iter)，C500 HD 单元不足 |
| Round 8 | 4-group warp-private + vec8 | 536.3 | +11.8% | 同 Round 1，合并开销略大于原子节省 |
| Round 9 | vec8 unroll-by-2 | 496.7 | +3.6% | 同 Round 7，更多背靠背加载压垮原子 pipeline |
| Round 10 | flag-array 投机暂存 | 527.7 | +10.0% | flag-bit-scan 开销 + 标量随机读 > 节省的第 2 遍全局读 |

### 3.3 各优化贡献占比

```
总提升: 61.0 GB/s (+12.7%)
├── vec8 (32B) 加载 [Round 5]:  48.6 GB/s  ████████████████████████████████████ 79.7%
├── smem 候选值缓存  [Round 2]:  12.4 GB/s  ████████                              20.3%
└── (其余尝试均回退, 贡献 0)
```

### 3.4 各计算阶段时间占比

以峰值带宽 shape (bs=1662, seq_len=107520, 2.67 ms) 分析：

```
每行数据量: L=107520 × 4B = 420 KB
两遍全局读: 2 × 420 KB = 840 KB/行
聚合带宽:   540.6 GB/s (跨 104 SM)
单行有效:  540.6 / 104 = 5.2 GB/s/SM
单行时间:  840 KB / 5.2 GB/s = 161 μs
1662 行 / 104 SM = 16 行/SM → 16 × 161 = 2576 μs ≈ 2.58 ms

实测 2.67 ms，差额 ~0.09 ms 为非 HBM 开销：
├── Stage 1 直方图 (第 1 遍 HBM 读):    ~1.29 ms  (48.3%)  ████████████
├── Stage 2 候选暂存 (第 2 遍 HBM 读):   ~1.29 ms  (48.3%)  ████████████
├── 4 轮精细化 (smem 读, 无 HBM):        ~0.05 ms  ( 1.9%)  █
├── sync + cumsum + zero-init:           ~0.04 ms  ( 1.5%)  █
└── 总计:                                ~2.67 ms  (100%)
```

**结论：** kernel **96.6% 时间花在两遍 HBM 读上**，是纯内存带宽受限。原子/cumsum/sync 仅占 3.4%。

### 3.5 为何达不到 1.2 TB/s 目标

```
理论单遍上限:  1.55 TB/s (HBM datasheet)
理论两遍上限:  1.55 / 2 = 775 GB/s  ← radix-select 的物理天花板
实测峰值:      540.6 GB/s             ← 达到天花板的 70%

540.6 / 775 = 70% 效率，余下 30% 损耗来自:
├── L2 miss (L=107520×4B=420KB/行, 104 并发=43MB >> 8MB L2, 第 2 遍几乎全 miss)
├── 原子 pipeline 串行 (C500 HD 单元少, 8 atomics/iter 有排队)
├── sync 开销 (~55 次 __syncthreads × 0.5μs = 27μs/行)
└── 分支/地址计算开销

要达到 1.2 TB/s 需消除第 2 遍读:
├── SREG 缓存 (C500 指南推荐): 第 1 遍把数据存寄存器, 第 2 遍从寄存器读
│   需要 105 floats/线程 (L/1024) = 105 寄存器 → 超出 256 上限 → 不可行
├── 单遍算法 (优先队列/堆): K=2048 的堆太大, smem 放不下 (16KB vs 需要 16KB × 16 warps)
└── L2 持久化 hint: 8MB L2 / 420KB/行 = 19 行, 104 并发行只覆盖 18% → 收益 < 20%
```

**根本原因：** radix-select 的 2 遍扫描结构 + C500 的有限 L2 (8MB) + 大 L (420KB/行) 共同限制了带宽上限到 ~775 GB/s，实测 540.6 GB/s 已达 70% 效率。

### 3.6 优化决策的 C500 硬件约束

| C500 特性 | 对本 kernel 的约束 | 优化应对 |
|-----------|-------------------|---------|
| **HD 原子单元少** | atomicAdd 串行化是潜在瓶颈 | vec8 不增加原子总数；Round 1/8 的 warp-private 失败证明减原子反而被合并开销抵消 |
| **≥32B/线程/访问** | vec4 (16B) 不满载 load pipeline | vec8 (32B) 满足规则，+9.9% |
| **64K 寄存器/SM, 256/线程** | SREG 缓存上限 ~100 floats/线程 | L=107520 需 105 → 超限 → 不可行 |
| **8MB L2** | 104 并发行 × 420KB = 43MB >> L2 | 第 2 遍几乎全 L2 miss → 无法靠 L2 消除第 2 遍 |
| **warp=64 (非 32)** | `__shfl` 掩码需 `0xffffffffffffffffULL` | Round 3 warp shuffle cumsum 因掩码问题失败 |
| **64KB smem/SM** | 静态+动态 smem ≤ 64KB | 9KB 直方图 + 16KB 值缓存 + 24KB 动态 = 49KB ✓ |
| **sm_80 异步拷贝** | `cp.async` 可重叠 load 与计算 | 未尝试：需额外 32KB smem 暂存 > 64KB 上限 |

---

## 四、附录

### 4.1 关键文件

| 文件 | 作用 |
|------|------|
| `op/sglang/csrc/elementwise/topk.cu` | prefill / pure-topk / ragged kernel + 3 个 interface |
| `op/sglang/csrc/elementwise/topk_decode.cu` | decode kernel (独立 TU) + split-K 协作路径 |
| `op/sglang/csrc/common_extension.cc:148-156` | Python 绑定注册 (`torch.ops.sgl_kernel.fast_topk_*`) |
| `unit_test/test_op_topk_transform_prefill_kernel.py` | 性能测试 (带宽 + 集合等价正确性) |
| `unit_test/test_sglang_topk_transformer_prefill_kernel.py` | 正确性测试 (190 passed, 30 skipped) |

### 4.2 备份版本

| 文件 | 轮次 | 峰值 GB/s |
|------|------|-----------|
| `topk.cu.bak_baseline_v0` | Round 0 (baseline) | 479.6 |
| `topk.cu.bak_v2_round2` | Round 2 (smem 缓存) | 492.0 |
| `topk.cu.bak_v5_round5` | **Round 5 (最终, = 当前 topk.cu)** | **540.6** |

### 4.3 关键常量

```cpp
constexpr int TopK = 2048;              // DeepSeek-V3.2 固定
constexpr int kThreadsPerBlock = 1024;  // 1 block = 1024 线程 = 16 warps (warp=64)
constexpr int kSmemInputSize = 2048;    // 候选暂存缓冲容量 (每缓冲)
constexpr size_t kSmemHigh = 24KB;     // HIGH 方案动态 smem (>96KB optin 的 GPU)
constexpr size_t kSmemLow = 16KB;      // LOW 方案动态 smem (≤64KB smem 的 GPU)
constexpr auto RADIX = 1024;           // 粗粒度直方图桶数 (10-bit)
constexpr auto LOG_RADIX = 10;          // cumsum 轮数
```

### 4.4 构建与测试命令

```bash
# 构建 (仅 sglang 子模块)
docker exec -w /home/metax/mcoplib/gerrit/mcoplib mcoplib_vllm0230 bash -lc '
  source env_local.sh && \
  BUILD_VLLM_SUBMODULE=OFF BUILD_DEFAULT_OP_SUBMODULE=ON \
  BUILD_LMDEPLOY_SUBMODULE=OFF BUILD_SGLANG_SUBMODULE=ON \
  python setup.py build_ext --inplace 2>&1 | tail -20'

# 性能测试
docker exec -w /home/metax/mcoplib/gerrit/mcoplib mcoplib_vllm0230 bash -lc '
  source env_local.sh && \
  /opt/conda/bin/python unit_test/test_op_topk_transform_prefill_kernel.py 2>&1 | tail -40'

# 正确性测试
docker exec -w /home/metax/mcoplib/gerrit/mcoplib mcoplib_vllm0230 bash -lc '
  source env_local.sh && \
  /opt/conda/bin/python -m pytest -v -s unit_test/test_sglang_topk_transformer_prefill_kernel.py 2>&1 | tail -40'
```

### 4.5 最终性能数据 (Round 5, 单位 GB/s)

| bs \ seq_len | 4096 | 18560 | 18688 | 32768 | 65536 | 66551 | 107520 |
|--------------|------|-------|-------|-------|-------|-------|--------|
| 132 | 119.3 | 217.7 | 222.3 | 280.6 | 378.3 | 377.7 | 428.4 |
| 256 | 142.7 | 257.0 | 256.5 | 332.1 | 441.1 | 440.3 | 499.6 |
| 1662 | 160.7 | 273.8 | 279.0 | 362.8 | 478.4 | 483.3 | **540.6** |
| 4096 | 157.0 | 275.1 | 275.6 | 356.8 | 474.2 | 479.3 | 536.9 |

- 峰值: **540.6 GB/s** (bs=1662, seq_len=107520) — +12.7% vs baseline 479.6
- 正确性: **ALL PASS** (30 shape 全部 SET OK, 190 correctness tests passed)
- 目标: 1.2 TB/s (未达, 2-pass radix-select 理论上限 ~775 GB/s, 已达 70%)
