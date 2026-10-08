# AI 排序算子深度解析

本文系统梳理 GPU/AI 场景下常见的排序与 Top-K 算子：Bitonic Sort、Radix Sort、Max-Heap、Histogram Sort、Bucket Sort。每种算法给出原理、实现、流程图、复杂度、应用场景与举例，并在文末给出对比与选型表，以及与本项目 `op/sglang/csrc/elementwise/topk.cu` 的对照。

---

## 0. 整体定位

AI 场景下的排序（尤其 GPU 上）与经典 CPU 排序关注点不同：

- **输入规模大**：单 batch 上万~百万元素（KV-cache page、MoE expert token、top-k 候选）。
- **只关心 Top-K**：99% 的"排序"需求其实只取前 K（K=4/8/2048），不需要全序。
- **数据并行**：SIMT 模型偏好规则内存访问、低分支、可并行归约。
- **浮点转单调整数键**：把 IEEE754 float 翻成可当无符号整数比较的 key，借用整数 radix/bitonic 硬件友好性。

下面对 5 类算法逐一拆解。

---

## 一、Bitonic Sort（双调排序）

### 1.1 原理

一个序列是 **bitonic**（双调）= 先升后降（或循环移位后满足）。Bitonic sort 利用两条性质：

1. 任意两个长度 n/2 的单调（一升一降）序列交错比较交换，可生成 n 长度的 bitonic 序列；
2. 一个 n 长 bitonic 序列经 ⌈log₂n⌉ 轮 compare-swap 网络后，变成单调（全升或全降）。

核心是 **compare-swap (CS)**：`(a,b) → (min(a,b), max(a,b))`，无分支、可并行。

### 1.2 实现（GPU 朴素版）

```cuda
// n = power of 2
for (int k = 2; k <= n; k <<= 1) {        // 合并段长
  for (int j = k >> 1; j > 0; j >>= 1) {  // 比较间距
    int i = threadIdx.x;
    int a = i ^ j;                         // 配对下标
    if (a > i) {
      bool up = ((i & k) == 0);            // 本段升/降方向
      float x = d[i], y = d[a];
      if ((up && x > y) || (!up && x < y)) { swap(d[i], d[a]); }
    }
    __syncthreads();
  }
}
```

### 1.3 流程图

```
        输入 n 元素（n=2^m）
              │
   ┌──────────┴──────────┐
  合并段长 k=2,4,8,…,n   │
   └──────────┬──────────┘
              │
   对每个 k：
     间距 j=k/2,k/4,…,1
        每步并行做 n/2 次 compare-swap
        方向由 (i&k)==0 决定（升段/降段）
              │
        k=n 时全序列变单调
              │
          全序输出
```

### 1.4 复杂度

- 比较：`O(n log²n)`（n/2 次 CS × ⌈log₂n⌉ 轮 × ⌈log₂n⌉ 层）。
- 深度：`O(log²n)`，GPU 上每层一次同步。
- 空间：原地 `O(1)`，无 auxiliary buffer。

### 1.5 应用场景

- **小规模全排序**（n ≤ 1024，单 block 内）：如 attention 每头局部 score 排序。
- **确定性排序**：网络是 data-independent 的，无分支发散，适合 GPU。
- **Warp 级 / Block 级 kernel**：常作为 radix sort 的"内部块排序"组件。
- 不适合 n≥1M 单段：log²n 层数高，同步开销大。

### 1.6 举例

**例 1**：`[3,1,4,1,5,9,2,6]`（n=8）

- k=2：配对 (3,1)→(1,3)↑、(4,1)→(4,1)↓、(5,9)→(5,9)↑、(2,6)→(6,2)↓ → `[1,3,4,1,5,9,6,2]`
- k=4：j=2→配对 (1,4),(3,1) 升段→`[1,1,4,3]`；(5,6),(9,2) 降段→`[6,9,5,2]` → `[1,1,4,3,6,9,5,2]`；j=1→`[1,1,4,3,9,6,5,2]`
- k=8：j=4→升段前半/降段后半…最终 → `[1,1,2,3,4,5,6,9]`

**例 2**：`[5,2,8,1]`（n=4），K=2 取 Top-2

- k=2：(5,2)↑→(2,5)，(8,1)↓→(8,1) → `[2,5,8,1]`
- k=4：j=2：(2,8)↑、(5,1)↑ → `[2,1,8,5]`；j=1：(2,1)↑→(1,2)、(8,5)↑→(5,8) → `[1,2,5,8]`
- 取末两位 → Top-2 = `[5,8]` ✓

---

## 二、Radix Sort（基数排序）

### 2.1 原理

把 key 当作 R 进制数，从低到高（LSD）或高到低（MSD）逐位做**稳定桶分配**。每位桶分配需保证稳定——相同 key 的相对顺序跨轮保持不变。

对浮点：先用 `convert_to_uint32`（符号翻转：负数 `~bits`、非负 `bits | 0x80000000`）得到单调 uint32 key，再 radix。

### 2.2 实现（GPU LSD，关键步骤）

```cuda
// 32-bit key, R=256 (8-bit per pass, 4 passes)
for (int pass = 0; pass < 4; ++pass) {
  int shift = pass * 8;
  // 1) 直方图：count[b] = #{key with byte b}
  atomicAdd(&hist[(key >> shift) & 0xFF], 1);
  // 2) 前缀和（exclusive scan）：excl[b] = sum_{k<b} count[k]
  blelloch_scan(hist, 256);
  // 3) 散射：按 (excl[key_byte] + thread_local_offset) 写出
  out[excl[byte] + rank] = in[i];   // rank = 该线程内此 byte 出现次数
}
```

### 2.3 流程图

```
   输入 n 个 (key,val)
          │
   ┌──────┴──────┐
  外层 4 轮 (32-bit / 8-bit) LSD
   └──────┬──────┘
          │
   每轮：
   ┌──────┴──────┐
   │ 1. 直方图   │ atomicAdd 到 256 桶
   │ 2. scan     │ exclusive prefix sum → 起始偏移
   │ 3. 散射     │ 写到 out[offset+rank]
   └──────┬──────┘
          │
       交换 in/out
          │
   4 轮后得到全序
```

### 2.4 复杂度

- 比较：`O(n · w)`，w=轮数（32-bit/8-bit → w=4）。
- 空间：`O(n + R)`，需双 buffer 与直方图。
- GPU 实际：4 轮 × (n/threads + 256 scan)，对大 n 远快于 bitonic。

### 2.5 应用场景

- **大规模全排序**（n ≥ 1M）：MoE expert routing 全 token 排序、KV-cache page 排序。
- **Top-K 加速**：MSD radix-select 只走必要桶（见下文桶排序/直方图混合）。
- NVIDIA CUB `cub::DeviceRadixSort` 是工业标准。

### 2.6 举例

**例 1**：key `[170, 045, 075, 090, 802, 024, 002, 066]`（3 位十进制，R=10）LSD：

- pass1（个位）桶+稳定散 → `[170,090,802,002,024,045,075,066]`
- pass2（十位）→ `[802,002,024,045,066,170,075,090]`
- pass3（百位）→ `[002,024,045,066,075,090,170,802]`

**例 2**：浮点 `[-1.5, 2.0, -0.5, 1.0]` 转 uint32 key 后 LSD 4 轮（R=256）

- `convert_to_uint32`：`-1.5 → 0xBF C0 00 00 → ~bits = 0x40 3F FF FF`；`2.0 → 0x40 00 00 00 | 0x80 00 00 00 = 0xC0 00 00 00`；`-0.5 → 0xBF 00 00 00 → 0x40 FF FF FF`；`1.0 → 0x3F 80 00 00 | 0x80 00 00 00 = 0xBF 80 00 00`
- 排序后 key 升序：`0x40 3F FF FF, 0x40 FF FF FF, 0xBF 80 00 00, 0xC0 00 00 00`
- 还原浮点：`-1.5, -0.5, 1.0, 2.0` ✓

---

## 三、Max-Heap（最大堆）/ 堆排序

### 3.1 原理

完全二叉树，父节点 ≥ 子节点。核心两操作：

- **sift-up**：插入新元素到末尾，向上与父比较交换。
- **sift-down**：根与较大子交换向下，直到恢复堆性质。

取 Top-K 的堆维护：容量 K 的 min-heap（小顶堆），新来元素 > 堆顶则替换并 sift-down。

### 3.2 实现（Top-K with min-heap of size K）

```cuda
// 单 thread 维护 K 元素 min-heap
for (int i = 0; i < n; ++i) {
  if (heap_size < K) {
    heap[heap_size++] = x[i];
    sift_up(heap_size - 1);
  } else if (x[i] > heap[0]) {
    heap[0] = x[i];
    sift_down(0);
  }
}
// 取出时反向取 K 次 heap[0] → swap(end) → sift_down
```

### 3.3 流程图

```
   输入 n 元素，欲取 Top-K
          │
   ┌──────┴──────┐
   维护 K 元素 min-heap
   └──────┬──────┘
          │
   for each x in n:
     ├─ heap 未满 → push + sift-up
     └─ heap 满 & x>top → replace top + sift-down
          │
   取 K 次：取顶 → 与末尾交换 → 缩容 → sift-down
          │
       降序 Top-K
```

### 3.4 复杂度

- 构建：`O(n log K)`。
- 取出：`O(K log K)`。
- 单元素比较：树深 `O(log K)`，串行性质 → **GPU 不友好**。

### 3.5 应用场景

- **流式 Top-K**：元素无法全装内存时（在线推荐、实时 ranking）。
- **CPU 端小 K**：K=10/100，numpy heapq 足够。
- **K-D tree / 优先队列** 基础组件。
- GPU 上少用（分支发散、串行 sift-down）。

### 3.6 举例

**例 1**：n=8 `[3,1,4,1,5,9,2,6]`，K=3，min-heap 演化：

- push 3,1,4 → heap=[1,3,4]
- 1<top → 跳过
- 5>1 → top=5，sift-down → [3,5,4]
- 9>3 → top=9 → [4,5,9]
- 2<4 → 跳过
- 6>4 → top=6 → [5,6,9]
- 取出：5,6,9 ✓

**例 2**：流式日志取 Top-2 错误码，到达顺序 `[E404, E500, E404, E503, E500]`，按出现次数（value）维护 min-heap K=2：

- push E404(1), E500(1) → heap=[(E404,1),(E500,1)]
- E404(2)>1 → top=(E404,2)，sift-down → [(E500,1),(E404,2)]
- E503(1) 不>1 → 跳过
- E500(2)>1 → top=(E500,2) → [(E404,2),(E500,2)]
- 最终 Top-2：E404×2, E500×2 ✓

---

## 四、Histogram Sort（直方图排序）

### 4.1 原理

把 key 范围离散成 B 个桶，统计每个桶的计数（**直方图**），再做**前缀和**得到每桶起始偏移，最后把元素**散射**到目标位置。本质是「R=n 时的 radix 单轮」。

直方图排序的关键洞察：**只要知道每个 key 值在最终有序序列中的起始位置，就能直接把元素放到正确的位置上**。三步：

1. **统计（Count）**：扫一遍输入，对每个 key 值 k 累加 `count[k]++`。结果是每个 key 值的出现次数。
2. **前缀和（Exclusive Scan）**：对 `count[]` 做排他前缀和，得到 `offset[k] = sum_{i<k} count[i]`。`offset[k]` 就是 key 值 k 的元素在最终序列中的**起始下标**。
3. **散射（Scatter）**：再扫一遍输入，对每个元素 `(key=k, val=v)`，写入 `out[offset[k]] = v`，然后 `offset[k]++`（或用 thread-local rank）。所有 key=k 的元素会**连续**地排在 `[orig_offset[k], orig_offset[k]+count[k])` 区间内。

由于相同 key 的元素被写到连续区间，整体即有序。

### 4.2 实现（GPU 单 pass）

```cuda
// 假设 key ∈ [0, B)
// === Pass 1: 统计 ===
atomicAdd(&hist[key], 1);
__syncthreads();

// === Pass 2: exclusive scan ===
// excl[b] = sum_{k<b} hist[k]
blelloch_scan(hist, B);   // 就地前缀和，hist[] 变成 offset[]
__syncthreads();

// === Pass 3: 散射 ===
// 方法 A：用 atomicAdd 取写入位置（简单但有竞争）
int pos = atomicAdd(&offset[key], 1);
out[pos] = in[i];

// 方法 B：每线程本地 rank（无竞争，需双扫一次）
// 先各线程数自己内 key 出现次数，再合并 rank
```

### 4.3 流程图

```
   输入 n 个 key
       │
   ┌───┴────────────────────────────────────┐
   │ Pass 1: 直方图统计                       │
   │   for each key k: hist[k]++              │
   │   (GPU: atomicAdd)                       │
   └───┬────────────────────────────────────┘
       │
   ┌───┴────────────────────────────────────┐
   │ Pass 2: exclusive prefix sum            │
   │   offset[b] = sum_{k<b} hist[k]         │
   │   (Blelloch scan, O(B) work, O(log B))  │
   └───┬────────────────────────────────────┘
       │
   ┌───┴────────────────────────────────────┐
   │ Pass 3: 散射到输出                       │
   │   for each (key=k, val=v):              │
   │     pos = atomicAdd(&offset[k], 1)      │
   │     out[pos] = v                         │
   └───┬────────────────────────────────────┘
       │
   全序输出（若 B 覆盖 key 全域且桶内有序）
```

### 4.4 复杂度

- `O(n + B)`，单 pass。
- 空间 `O(n + B)`。
- 关键瓶颈：桶数 B 与 key 范围挂钩；若 key 是 32-bit 浮点，B 直接到 4G 不现实 → 退化成多轮 radix（每轮取 8 位，B=256）。

### 4.5 应用场景

- **整数 key 且范围小**：例如 age、rating、像素灰度值。
- **radix sort 的单轮基础**：每轮就是一个 R=256 的 histogram sort。
- **bucket quantile**：求分位数时统计后扫前缀和定位。
- AI 场景：MoE expert 直方图（expert id 通常 < 64）。

### 4.6 详细举例

**例 1（核心例，逐步画图）**：输入 `n=8`，key ∈ [0,5)，`A = [2,0,2,1,3,0,2,4]`

**Pass 1 — 统计直方图**

```
索引:   0   1   2   3   4   5   6   7
key:    2   0   2   1   3   0   2   4

扫一遍累计：
i=0  key=2 → hist[2]++
i=1  key=0 → hist[0]++
i=2  key=2 → hist[2]++
i=3  key=1 → hist[1]++
i=4  key=3 → hist[3]++
i=5  key=0 → hist[0]++
i=6  key=2 → hist[2]++
i=7  key=4 → hist[4]++

hist (B=5):
桶号:   0   1   2   3   4
计数:   2   1   3   1   1
含义:   key=0 出现 2 次
        key=1 出现 1 次
        key=2 出现 3 次
        key=3 出现 1 次
        key=4 出现 1 次
        合计 8 = n ✓
```

**Pass 2 — exclusive prefix sum**

```
对 hist 做排他前缀和（excl[b] = sum_{k<b} hist[k]）：

桶号 b:          0   1   2   3   4
hist[b]:         2   1   3   1   1
excl[b]:         0   2   3   6   7
                ↑   ↑   ↑   ↑   ↑
              key=0 key=1 key=2 key=3 key=4
              起始  起始  起始  起始  起始
              下标0 下标2 下标3 下标6 下标7

含义：
  - key=0 的元素要写到 out[0..1]（2 个）
  - key=1 的元素要写到 out[2..2]（1 个）
  - key=2 的元素要写到 out[3..5]（3 个）
  - key=3 的元素要写到 out[6..6]（1 个）
  - key=4 的元素要写到 out[7..7]（1 个）
```

**Pass 3 — 散射**

GPU 实现里 `offset[]` 是 `excl[]` 的可写副本，每写一个就 `atomicAdd(&offset[k], 1)`，所以**同一 key 的多个元素按线程到达顺序依次往后排**：

```
初始 offset = excl = [0,2,3,6,7]

i=0  key=2, val=A[0]
     pos = atomicAdd(&offset[2], 1) = 3
     out[3] = A[0] = 2
     offset[2] → 4

i=1  key=0, val=A[1]
     pos = atomicAdd(&offset[0], 1) = 0
     out[0] = A[1] = 0
     offset[0] → 1

i=2  key=2, val=A[2]
     pos = atomicAdd(&offset[2], 1) = 4
     out[4] = A[2] = 2
     offset[2] → 5

i=3  key=1, val=A[3]
     pos = atomicAdd(&offset[1], 1) = 2
     out[2] = A[3] = 1
     offset[1] → 3

i=4  key=3, val=A[4]
     pos = atomicAdd(&offset[3], 1) = 6
     out[6] = A[4] = 3
     offset[3] → 7

i=5  key=0, val=A[5]
     pos = atomicAdd(&offset[0], 1) = 1
     out[1] = A[5] = 0
     offset[0] → 2

i=6  key=2, val=A[6]
     pos = atomicAdd(&offset[2], 1) = 5
     out[5] = A[6] = 2
     offset[2] → 6

i=7  key=4, val=A[7]
     pos = atomicAdd(&offset[4], 1) = 7
     out[7] = A[7] = 4
     offset[4] → 8
```

**结果**：

```
out 索引:  0   1   2   3   4   5   6   7
out 值:    0   0   1   2   2   2   3   4
           ↑   ↑   ↑   ↑───────↑   ↑   ↑
          key=0(2个) key=1 key=2(3个) key=3 key=4

全序 ✓
```

**关键性质**：

- 同 key 的元素**相对顺序保持**（因为 atomicAdd 按到达顺序分配位置）→ **稳定排序**。
- 三次线性扫描 → `O(n + B)`。
- 若 key 范围 = n，则 `B = n`，整体 `O(n)`。

**例 2（GPU 上的 R=256 单轮，对应 radix 一轮）**：key 是 8-bit `uint8_t`，B=256

输入 `[0x03, 0xFF, 0x03, 0x80, 0x01, 0xFF, 0x03]`：

- Pass 1：`hist[0x03]=3, hist[0xFF]=2, hist[0x80]=1, hist[0x01]=1`，其余 0
- Pass 2：excl[0x01]=0, excl[0x03]=1, excl[0x80]=4, excl[0xFF]=5
- Pass 3：散射后 → `[0x01, 0x03, 0x03, 0x03, 0x80, 0xFF, 0xFF]` ✓

这就是 radix sort 每轮的内部实现。

**例 3（AI 场景：MoE expert id 直方图）**：8 个 token 被路由到 4 个专家 `[2,0,2,1,3,0,2,3]`，统计每个专家负载：

- Pass 1：`hist = [2,1,3,2]`（expert 0 收 2 个，expert 1 收 1 个，expert 2 收 3 个，expert 3 收 2 个）
- Pass 2：`excl = [0,2,3,6]`
- Pass 3：把 token id 按 expert 分组写到连续区间 → expert 0 占 `[0,2)`、expert 1 占 `[2,3)`、expert 2 占 `[3,6)`、expert 3 占 `[6,8)`，得到 token-per-expert 的连续布局，可直接喂给 grouped GEMM。

---

## 五、Bucket Sort（桶排序）

### 5.1 原理

把数据按 key 范围均匀分到 B 个桶，桶内元素**近似有序**（桶 i 的元素 > 桶 i-1），桶内再用任意排序（insertion/bitonic/recursive bucket）。

与直方图排序区别：直方图排序假设 key 离散有限；桶排序假设 key 连续、用区间划分，**桶内需二次排序**。

### 5.2 实现（GPU 桶排序 + 桶内 bitonic）

```cuda
// 1) 分桶
int b = (int)((key - key_min) / (key_max - key_min) * B);
atomicAdd(&bucket_count[b], 1);
// 2) scan → bucket_start[b]
// 3) 散射到 bucket 区
bucket_data[bucket_start[b] + rank] = val;
// 4) 桶内排序（每桶独立 kernel，bitonic / insertion）
for (b in 0..B) bitonic_sort(bucket_data + start[b], bucket_count[b]);
```

### 5.3 流程图

```
   输入 n 元素，已知 key ∈ [lo, hi]
              │
   ┌──────────┴──────────┐
   1. 分桶：b = (key-lo)/(hi-lo)*B
   2. 桶计数 + 前缀和 → bucket_start[]
   3. 散射到 bucket_data[]
   4. 桶内排序（bitonic / insertion / recursive）
   └──────────┬──────────┘
              │
   拼接各桶 → 全序输出
```

### 5.4 复杂度

- 均匀分布：`O(n + B + (n/B) log(n/B))`。
- 最坏（全聚一桶）：`O(n log n)`。
- 空间 `O(n + B)`。

### 5.5 应用场景

- **浮点近似排序**：分布均匀时近线性。
- **分位数估计**：取桶边界即可估 quantile。
- **稀疏数据归并**：按 key 区间分桶后桶内归并。
- GPU 上需谨慎：若分布倾斜，单桶过载导致 warp 发散。

### 5.6 举例

**例 1**：`[0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68]`，B=5，区间 [0,1)

- 桶0 [0,0.2)：0.17,0.12
- 桶1 [0.2,0.4)：0.39,0.26,0.21,0.23
- 桶2 [0.4,0.6)：—
- 桶3 [0.6,0.8)：0.78,0.72,0.68
- 桶4 [0.8,1.0)：0.94
- 桶内排：0.12,0.17 / 0.21,0.23,0.26,0.39 / 0.68,0.72,0.78 / 0.94
- 拼接 → `[0.12,0.17,0.21,0.23,0.26,0.39,0.68,0.72,0.78,0.94]`

**例 2**：浮点 logits 取 Top-2，`[2.3, 0.5, 1.8, 3.7, 0.9, 2.1]`，已知 min=0.5, max=3.7，B=4，区间 [0.5,3.7]

- 桶宽 = (3.7-0.5)/4 = 0.8
- 桶0 [0.5,1.3)：0.5, 0.9
- 桶1 [1.3,2.1)：1.8
- 桶2 [2.1,2.9)：2.3, 2.1
- 桶3 [2.9,3.7]：3.7
- 桶内排：0.5,0.9 / 1.8 / 2.1,2.3 / 3.7
- 拼接 `[0.5,0.9,1.8,2.1,2.3,3.7]`，取末两位 → Top-2 = `[2.3, 3.7]` ✓

---

## 六、对比与选型表

| 算子 | 比较复杂度 | GPU 友好度 | 典型 n | 输出 | AI 场景 |
|------|-----------|-----------|--------|------|---------|
| Bitonic | O(n log²n) | ★★★★★ | ≤1024/块 | 全序 | 小 block 全排、warp sort |
| Radix (LSD) | O(n·w) | ★★★★★ | 1M~100M | 全序 | MoE token sort、CUB |
| Max-Heap | O(n log K) | ★ | 流式 | Top-K | CPU 在线 ranking |
| Histogram | O(n+B) | ★★★ | key 范围小 | 全序 | radix 单轮、quantile |
| Bucket | O(n+B+(n/B)log) | ★★★ | 浮点均匀 | 近全序 | 区间分位、稀疏归并 |

---

## 七、与本项目 `topk.cu` 的对照

`fast_topk_cuda_tl` 实际是**三种算法的融合**：

1. **Histogram sort 单轮**（stage 1）：8-bit 粗直方图 + Blelloch scan + 阈值桶定位。
2. **MSD radix-select**（stage 2）：4 轮 8-bit 桶分配，但**不真正排全部**——只走「≥ TopK 配额」的桶，等价于 radix sort 的"剪枝版"。
3. **倒数写入**（round 3 `index[TopK-pos]`）：保证阈值桶恰好填满 K 配额，避免再排一次。

这正是 GPU 上 Top-K 的工程最优解：**直方图定位 + radix 细化 + 原子计数倒数填**，比纯 bitonic（log²n 层）和纯 heap（串行）都快一个数量级。这也解释了为什么 `length ≤ TopK=2048` 时反而走 naive——小规模下 radix 的 4 轮 setup 开销不划算。
