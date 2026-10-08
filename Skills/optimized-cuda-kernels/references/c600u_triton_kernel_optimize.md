<<<<<<< HEAD   (6d018d MCX-12432 and MXC-12941 add xcore1008 for wb-std)
=======
# 在 MetaX C600-U 上优化 Triton 算子 —— 实战指南

> 本文是 C600-U 上 **Triton** 算子优化的可复用方法论与技巧库，配套 `c600-optimization-guide.md`（CUDA/MACA 侧）。
> 核心信条：**先测墙、再定位瓶颈、每次只改一个变量、用余弦相似度守精度、用 back-to-back burst 计时**。
> 最重要的两条 C600-U Triton 铁律先给出，后面再展开：
>
> 1. **访存密集算子：每线程搬运 ≥32 字节 —— 用 *更少* 的 `num_warps`，不是更多。** 这是 C600-U 上最反直觉、也最高杠杆的一条。
> 2. **一切能变成 `tl.constexpr` 的循环边界 / 行数 / 维度，都要变成 `constexpr`。** 运行时 trip-count 的循环无法展开、无法被软件流水，直接锁死带宽。

---

## 目录

1. [C600-U 硬件模型与关键参数](#1-c600-u-硬件模型与关键参数)
2. [第一性原理：优化前先回答的三个问题](#2-第一性原理优化前先回答的三个问题)
3. [核心招式（按杠杆从高到低）](#3-核心招式按杠杆从高到低)
4. [Triton 通用高性能技巧速查](#4-triton-通用高性能技巧速查)
5. [反直觉的坑（Triton 侧实测踩过的）](#5-反直觉的坑triton-侧实测踩过的)
6. [案例：`_combine_kernel`（756 → 1553 GB/s，全 shape 破 1.3T 目标）](#6-案例_combine_kernel756--1553-gbs全-shape-破-13t-目标)
7. [案例：`_score_kernel`（673 → 1376 GB/s，消除 L2 请求墙 + 寄存器跨行复用）](#7-案例_score_kernel673--1376-gbs消除-l2-请求墙--寄存器跨行复用)
8. [Benchmark / 精度 / 计时模板](#8-benchmark--精度--计时模板)
9. [ReAct 优化流程 & 检查清单](#9-react-优化流程--检查清单)

---

## 1. C600-U 硬件模型与关键参数

| 项目 | 值 | 对 Triton 的含义 |
|------|-----|------------------|
| Warp（wave）宽度 | **64 lane** | `num_warps=1` 即 64 线程；block 线程数 = `num_warps × 64` |
| Shuffle 硬件子群 | **16 lane** | 跨 16 lane 的规约走原生 shuffle；`tl.sum/max` 沿归约轴由编译器映射到此 |
| Shared memory | **128 KB / SM** | Triton 的 `tl.dot`、跨轴规约、`num_stages` 缓冲都吃它，是占用率上限之一 |
| 寄存器 | **255 / thread** | 大 `BLOCK` tile × 高 `num_warps` 会触发寄存器溢出（spill）→ 崩带宽 |
| 计算单元 | 64 AP，每 AP 4 PEU、8 warp/PEU，≤2048 thread/AP | grid 必须足够大才能填满所有 AP |
| 每线程搬运门槛 | **≥32 B/thread** 才吃满合并访存 | **决定 `num_warps` 选择的第一原则** |
| 访存事务 | 128-bit（16B）向量化最佳 | `BLOCK_H` 取 2 的幂，指针自然对齐 |

**单 die HBM 实测天花板（务必以此为目标，不要拿 datasheet 3480 当靶）：**

| 访存模式 | 单 die 天花板 |
|----------|--------------|
| 只读 (read-only) | ~1474 GB/s |
| 拷贝 (1R+1W) | ~1322 GB/s |
| 原地读改写 (in-place RMW) | ~1326 GB/s |
| triad (2R+1W) | ~1294 GB/s |

> Datasheet 标称的 **2.6–4.0 TB/s 是双 die**；单 kernel 单 die 物理上够不到，别把它当成"没达标"。**目标定在 1.3 TB/s 单 die 附近（约 85% 效率）**。

---

## 2. 第一性原理：优化前先回答的三个问题

### 2.1 我改的代码真的被编译进去了吗？

Triton 有 **磁盘 JIT 缓存**。改了 kernel 但缓存命中 → 你测的是旧代码。**每次改 kernel 必须清缓存：**

```bash
rm -rf ~/.triton/cache          # 或 /root/.triton/cache（容器内 root）
# 或者
export TRITON_ALWAYS_COMPILE=1
```

> 验证签名/常量改动是否生效：改一处会导致编译报错的地方，若不报错就是没重编。首启动会现场编译特化再计时——第一发 launch 不计入 timing。

### 2.2 这块卡的墙在哪、我的算子是哪种墙？

先用 roofline 分类：**算子是 memory-bound 还是 compute-bound？**
- 逐元素 / 规约 / norm / softmax / 加权和 / 量化写：**几乎都是 memory-bound** → 目标是逼近 §1 的 HBM 天花板。
- GEMM / attention 的 QK^T：**compute-bound** → 目标是喂饱 `tl.dot`（MMA），关注 tile 尺寸、`num_stages` 流水。

memory-bound 的**理论最小流量**（用它算带宽，别数逻辑运算量）：
```
bytes = 读入的所有张量字节 + 写出的所有张量字节（各计一次）
带宽 GB/s = bytes / (latency_ms × 1e6)
```
把实测带宽 / 单 die 天花板 = 效率。**效率 < 60% 一定有结构问题**（多半是 §3 招式 1 或招式 2）。

### 2.3 离墙有多远、卡在哪一环？

C600-U 上 memory-bound Triton 算子最常见的三个"离墙"元凶，按出现频率：
1. **`num_warps` 太大 → 每线程字节数太小 → 合并访存打不满**（最常见，见招式 1）。
2. **运行时 trip-count 循环没展开 → 访存不流水**（见招式 2）。
3. **重复读共享小张量（如 `cw[H]`、scores）成了 L2 瓶颈**（见招式 4）。

---

## 3. 核心招式（按杠杆从高到低）

### 招式 1（**最高杠杆**）：用 `num_warps` 把"每线程字节数"顶到 ≥32B —— 通常是**调小**

C600-U 一个 warp = 64 lane。一个 program 处理 `BLOCK_H` 个元素时：

```
每线程元素数 = BLOCK_H / (num_warps × 64)
每线程字节数 = 每线程元素数 × dtype_bytes
```

**目标：每线程字节数 ≥ 32 B**（一次以上完整合并事务）。

实测（`_combine_kernel`，BLOCK_H=1024，bf16=2B）：

| num_warps | 线程数 | 每线程字节 | 峰值带宽 |
|-----------|--------|-----------|----------|
| 4（默认直觉） | 256 | **8 B** ❌ | ~820 GB/s |
| 2 | 128 | 16 B | ~1450 GB/s |
| **1** | **64** | **32 B** ✅ | **~1553 GB/s** |
| 8 | 512 | 4 B | ~410 GB/s（崩） |

> **反直觉但极其有效**：访存密集算子在 C600-U 上，**减少 `num_warps` 让每个线程搬更多字节**，比"多开线程"更快。这和 NVIDIA 上"尽量多线程"的直觉相反。
> 与 CUDA 侧的教训完全一致（CUDA 里表现为"降低 threads-per-head / 一个 lane 搬 64B"）。
> **落地：把 `num_warps` 做成可扫的旋钮（环境变量），从 1 开始扫 1/2/4/8，几乎总在 1 或 2 取到峰值。**

### 招式 2（次高杠杆）：把循环边界变成 `tl.constexpr`，让循环**静态展开**

Triton 只能对**编译期已知**的 trip-count 做展开与软件流水。运行时标量边界（如把行数 `NVB` 当普通参数）会让每次 `tl.load` 串行等在自己的累加后面，访存无法 overlap。

```python
# ❌ 运行时 trip-count：无法展开，访存串行
@triton.jit
def k(..., NVB, ...):
    for j in range(0, NVB + 1):      # NVB 是运行时标量
        v = tl.load(ptr + j * stride)
        acc += p[j] * v

# ✅ constexpr trip-count：静态展开，独立 load 背靠背发射、访存流水
@triton.jit
def k(..., NVB: tl.constexpr, ...):
    for j in range(0, NVB):          # constexpr → 编译期展开
        acc += p_j * tl.load(ptr + j * stride)
```

`_combine_kernel` 仅此一项：nvb=2 用例 **+37%**（600→820 GB/s），且消除了"nvb 越大每字节越慢"的病态。

> **Triton `@jit` 内的两个硬坑（写展开循环时必踩）：**
> - **不能用 `list.append()` / `vals[j]` 索引 Python list** —— AST visitor 直接报 `'append' is not in list`。**直接把 `tl.load` 累加进 `acc`**，别攒到 list 里。
> - **`tl.static_range` 不是 Python 可迭代对象**，不能用来 build list。既然 `NVB` 已是 `constexpr`，普通 `for j in range(0, NVB)` 就会展开，无需 `static_range`。
> - `tl.static_range` 适合"我明确要求展开但边界本就是 constexpr 常量"的循环体（如按 `H//BLOCK_H` 分块），语义等价于 `range` + 强制 unroll。

### 招式 3：把标量提取/规约**移出访存循环**

在加权和里逐轮做 `p_j = tl.sum(tl.where(offs==j, p, 0))` 来抽一个标量概率，是每次迭代一次 16 宽规约——纯浪费。**把它挪到"累加阶段"，让访存阶段只有 load。** 更进一步：softmax 概率一次算好存成标量，循环内只做 `acc += p_j * v`。

### 招式 4：消除对共享小张量的重复全局读（L2 瓶颈）

若每个 (token,row) program 都重复读同一份 `cw[H]`（fp32 权重），`cw` 会变成 L2 热点：bf16 行单独能到 ~1.2TB/s，一叠加 fp32 cw 直接腰斩。

**解法**：让**一个 program 拥有一个 token、把所有 row 放进 `[MAXR, BLOCK_H]` 寄存器 tile**，`cw` 每块只 load 一次并跨所有 row 复用（`_score_kernel` 的做法）。这是"数据复用 + 提高算术强度"的 Triton 版。

### 招式 5：program 粒度 —— 一行一 program vs 一 tile 一 program

- **RMSNorm / softmax / 逐行规约**：**一行（或一 token）一个 program**，整行留在寄存器/shared，规约走片内，避免跨 program 通信。
- **GEMM / 大矩阵**：**一个输出 tile 一个 program**，配 `tl.dot` + `num_stages` 流水。
- **纯逐元素**：一个 program 一段连续 `BLOCK`，只看合并访存与每线程字节数（招式 1）。

### 招式 6：全 H 并行（2D grid）掩盖延迟

把 `H` 切成 `n_h_blocks` 块、grid 做成 `(T, n_h_blocks)`，让更多 CTA 填满 AP。代价是 softmax/归约在每个 H-chunk CTA 里冗余重算——**≤16 个 fp32 元素的重算是零成本**，换来的 H 并行度远超其代价。`_combine_kernel` 用 `(T, 7)` 打满。

### 招式 7：`BLOCK_H` 取 2 的幂 + 让每线程字节数落在 32~128B

`BLOCK_H` 必须是 2 的幂（自动向量化 + 指针对齐）。选定后，`num_warps` 反推每线程字节数（招式 1）。经验区间：**每线程 32–128 B 最稳**；<32B 打不满合并，>128B 压寄存器/降占用。

### 招式 8：back-to-back burst 计时，撑住 DVFS

C600-U 有频率爬坡（DVFS）。零散计时会测到没升频的低值。**连续发射一大串 launch 夹在一对 event 之间**，取每发最优，才是真实峰值（见 §8 模板 `_bench_burst`）。

### 招式 9：`num_stages` 软件流水（compute-bound / 深 K 才有用）

`num_stages=N` 让 Triton 对含 `tl.dot` 或长 H-loop 的循环做多级预取流水。**纯 memory-bound 逐元素算子基本无收益甚至负收益**（多缓冲吃 shared 反降占用）；GEMM/attention 上从 2/3/4 扫。

---

## 4. Triton 通用高性能技巧速查

| 技巧 | 要点 | 何时用 |
|------|------|--------|
| **`tl.constexpr` 一切静态量** | 维度、行数、循环边界、tile 尺寸全部 constexpr → 展开 + 特化 | 永远优先 |
| **避免数据相关分支** | 用 `tl.where` / mask，不要用依赖数据的 `if` 造成线程发散 | 有条件写/读时 |
| **mask 处理尾块与越界** | `tl.load(ptr, mask=..., other=0.0)`，非 2 的幂维度靠 mask 补齐 | 边界/padding |
| **合并访问** | 相邻 lane 访问相邻地址；行主序张量沿最后一维展开 program | 永远 |
| **向量化 + 对齐** | `BLOCK` 取 2 的幂；指针起点对齐 16B | 永远 |
| **规约留在片内** | 沿轴 `tl.sum/max`，编译器映射到 16-lane shuffle / shared，别落回全局 | 归约/norm/softmax |
| **数据复用抬算术强度** | 共享权重 load 一次、跨多行复用（招式 4） | 有共享小张量 |
| **`@triton.autotune`** | 把 `BLOCK_*`/`num_warps`/`num_stages` 交给 autotune 扫 | 形状固定、想省手扫 |
| **少即是多的 warp** | 访存密集：从 `num_warps=1` 起扫（招式 1） | memory-bound |
| **一次 launch 做多步** | score+combine 融合、norm 融进上一步，减少 kernel 启动与中间写回 | 相邻访存算子 |
| **fp32 累加、末尾转出 dtype** | `acc` 用 fp32，`acc.to(out_ptr.dtype.element_ty)` 存回 | 保精度 |

**`@triton.autotune` 骨架：**
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': bh}, num_warps=nw, num_stages=ns)
        for bh in (512, 1024)
        for nw in (1, 2, 4)          # 访存密集从小往大，通常 1/2 胜出
        for ns in (1, 2)
    ],
    key=['H', 'NVB'],                # 这些变了才重新调优
)
@triton.jit
def kernel(...): ...
```

---

## 5. 反直觉的坑（Triton 侧实测踩过的）

1. **`num_warps` 越大越慢**（访存密集）。默认 4 往往是最差之一；答案常是 1。**先扫这个旋钮再谈别的。**
2. **忘清 `~/.triton/cache`** → 改了 kernel 却测到旧特化，得出"改动无效"的错误结论。
3. **运行时 trip-count 循环** = 隐形性能杀手，`nvb` 越大每字节越慢就是信号。变 `constexpr`。
4. **`@jit` 里 `list.append` / list 索引 / `tl.static_range` 建 list** → 直接编译报错。累加进 `acc`，别用 list。
5. **拿 datasheet 3480（双 die）当单 kernel 目标** → 永远"没达标"。用单 die ~1.3T。
6. **零散计时**没撑住 DVFS，测出偏低带宽 → 用 burst。
7. **重复读共享 fp32 小张量**（cw/scale）成 L2 墙，bf16 主体明明够快却腰斩 → 一 program 一 token + 寄存器 tile 复用。
8. **过度 `num_stages`** 在逐元素算子上吃 shared 反降占用。
9. **过拟合单一 shape**：用 shape/dtype 分支时要确认在小 shape 与大 shape 都不劣化——**若一套配置全 shape 皆优（如 `_combine_kernel` 的 NW=1），就不要引入分支**，简单即正确。

---

## 6. 案例：`_combine_kernel`（756 → 1553 GB/s，全 shape 破 1.3T 目标）

**算子**：Kimi-K3 attention residual 聚合 Step 2 —— softmax(scores) → 对 (nvb+1) 个 bf16 行加权求和 → 写 `[T,H]`。典型 memory-bound triad（多读 + 一写）。

**基线**：peak **756 GB/s**（nvb=1）/ 600（nvb=2）；效率 ~58%。病征：nvb=2 比 nvb=1 **每字节更慢**。

**仅两处改动即达标（1 个有效迭代，≤15 限制内提前停）：**

| 改动 | 原理（对应招式） | 效果 |
|------|-----------------|------|
| ① `NVB` 运行时参数 → `tl.constexpr` | 招式 2：循环静态展开，(nvb+1) 个独立行 load 背靠背发射、访存流水 | nvb=2 **+37%**（600→820） |
| ② `num_warps` 4 → **1** | 招式 1：BLOCK_H=1024 下 4 warp=8B/线程 → 1 warp=**32B/线程**，一次合并事务/lane | 全盘 **~+90%** |

附带：softmax 概率抽取移出访存循环（招式 3）。

**结果**：全部 16 个 shape 均 > 1300 GB/s（最小 1396，峰值 **1553**），`cos_sim=1.000000 ALL PASS`。小 shape（T=2048：1396–1474）与大 shape（T=16384：1517–1565）**同一份 NW=1 配置皆最优，无需分支**。

逐 shape 提升率：
| shape (T,rows,nvb) | 基线 | 最优 | 提升 |
|---|---|---|---|
| 2048,2,2 | 580 | 1455 | +151% |
| 4096,*,2 | ~594 | ~1508 | +154% |
| 16384,2,2 | 602 | **1553** | +158% |
| 16384,*,1 | ~755 | ~1520 | +101% |

**核心复用经验**：memory-bound Triton 算子在 C600-U 上，**先扫 `num_warps`（从 1 起）+ 把行/循环边界 constexpr 化**，两招常常直接从 ~55% 效率跳到 ~85%+。

---

## 7. 案例：`_score_kernel`（673 → 1376 GB/s，消除 L2 请求墙 + 寄存器跨行复用）

**算子**：Kimi-K3 attention residual 聚合 Step 1 —— 给每个 token 的 (nvb+1) 个候选行各算一个 RMS-Norm-then-project 分数：

```
s[t,j] = (r[t,j] · cw) / sqrt(mean(r[t,j]²) + eps)
```

其中 `r` 是 bank 行（`j < NVB`）或 prefix 行（`j == NVB`），`cw = norm_weight ⊙ proj_weight`，shape `[H]`、fp32，**所有行共享同一份 cw**。典型 memory-bound，且有"共享小张量"特征——是招式 4 的教科书场景。

**基线（原始 2D-grid 实现）**：peak **673 GB/s** @ T=8200 nvb=8，效率 ~52%。病征：带宽随 nvb 近线性增长（nvb=1: 645, nvb=8: 673），看似"按比例正常"，但**离单 die 墙只有一半**——L2 请求带宽被 cw 重复读饱和。

### 7.1 病灶定位：cw 是 L2 请求墙（不是容量墙）

原始实现：
- grid `(T, NVB+1)` —— **一个 CTA 处理一行**
- 每个 CTA 独立扫整个 H，每次 H-block 都 `tl.load` 一份 `cw[h0:h0+BLOCK_H]`
- **cw 总读次数 = T × (NVB+1) × H × 4B**

T=8200, nvb=8 → **73,800 个 CTA** 全涌向同一份 28 KB cw。L2 cache 容量远大于 28 KB，**命中率几乎 100%**——所以问题不是 capacity miss，而是 **request rate**：所有 SM 同时向同一些 cache line 发读请求，L2 的请求队列/端口打满，吞吐被"请求并发度"限制（不是字节带宽）。

**诊断信号**（识别"请求墙" vs "字节墙"）：
1. 带宽随 CTA 数（即 `nvb+1`）近线性，但**远低于单 die 墙**——典型"请求墙"特征。
2. 探针剥离验证：单独读 bf16 行（不带 cw）可到 ~1.2 TB/s，加进 fp32 cw 立刻腰斩到 600~700 GB/s——**cw 是元凶**。
3. 改 `BLOCK_H` / `num_warps` 都只能小幅波动，结构性瓶颈在此——单点旋钮救不了。

### 7.2 三处关键改动（招式 4 完整落地 + 招式 2 + 招式 5）

#### ① Grid `(T, NVB+1)` → `(T,)` —— 一个 program 拥有整个 token（招式 5 + 招式 4）

```python
# ❌ 原始：每行一个 CTA, cw 被每个 (t,j) CTA 各读一遍
@triton.jit
def _score_kernel(..., NVB, ...):              # NVB 是运行时标量
    pid_t = tl.program_id(0)
    j = tl.program_id(1)                       # ★ 第二维 = 行号
    if j > NVB: return
    for h0 in tl.static_range(0, H, BLOCK_H):
        cw = tl.load(cw_ptr + offs_h)          # ★ 每个 CTA 各读一遍 cw
        v = tl.load(bank_ptr + pid_t*stride_bm + j*stride_bb + offs_h)
        sumsq += tl.sum(v * v)
        dotv += tl.sum(v * cw)
    rrms = 1.0 / tl.sqrt(sumsq / H + eps)
    tl.store(scores_ptr + pid_t*stride_sm + j, dotv * rrms)

# ✅ 优化：一个 CTA 拥有一个 token, 所有 (nvb+1) 行放进 [MAXR, BLOCK_H] 寄存器 tile,
#        cw 每块只 load 一次, 跨所有行复用
@triton.jit
def _score_kernel(..., NVB: tl.constexpr, MAXR: tl.constexpr, ...):
    pid_t = tl.program_id(0)
    rj = tl.arange(0, MAXR)                    # [MAXR] 所有行的逻辑行号
    active = rj <= NVB                         # [MAXR] mask: 越界行不访存
    row_base = tl.where(
        rj < NVB,
        bank_ptr + pid_t * stride_bm + rj * stride_bb,
        prefix_ptr + pid_t * stride_pm,
    )                                          # [MAXR] 每行的基址
    sumsq = tl.zeros([MAXR, BLOCK_H], tl.float32)
    dotv = tl.zeros([MAXR, BLOCK_H], tl.float32)
    for h0 in tl.static_range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        cw = tl.load(cw_ptr + offs_h)          # ★ 一块 cw 只 load 一次
        ptrs = row_base[:, None] + offs_h[None, :]   # [MAXR, BLOCK_H]
        v = tl.load(ptrs, mask=active[:, None], other=0.0).to(tl.float32)
        sumsq += v * v
        dotv += v * cw[None, :]                # ★ cw 广播到所有 MAXR 行复用
    ssum = tl.sum(sumsq, axis=1)               # [MAXR] 一次性规约出所有行的 sumsq
    dsum = tl.sum(dotv, axis=1)                # [MAXR]
    rrms = 1.0 / tl.sqrt(ssum / H + eps)
    tl.store(scores_ptr + pid_t * stride_sm + rj, dsum * rrms, mask=active)
```

**核心机制**：
- 把 `(nvb+1)` 行的累加器从"分散在 `nvb+1` 个 CTA"改成"集中在一个 CTA 的 `[MAXR, BLOCK_H]` 寄存器 tile"。
- `cw[None, :]` 把一块 cw 广播到 tile 的所有行 → **cw 总读次数从 `T×(NVB+1)` 降到 `T`**，L2 请求墙直接消失。
- `MAXR = next_pow2(nvb+1)`：Triton tile 维度要 2 的幂；越界行（`rj > NVB`）靠 `active` mask 掉、不访存，只是占寄存器。
- `row_base = tl.where(rj < NVB, bank..., prefix...)`：把"bank 行 vs prefix 行"的分支收敛成一个指针 tile，避免循环内 `if j < NVB` 的线程发散。

#### ② NVB 从运行时参数 → `tl.constexpr`（招式 2）

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 128}, num_warps=2),
        triton.Config({'BLOCK_H': 128}, num_warps=4),
        triton.Config({'BLOCK_H': 256}, num_warps=2),
        triton.Config({'BLOCK_H': 256}, num_warps=4),
        triton.Config({'BLOCK_H': 256}, num_warps=8),
        triton.Config({'BLOCK_H': 512}, num_warps=4),
        triton.Config({'BLOCK_H': 512}, num_warps=8),
        triton.Config({'BLOCK_H': 1024}, num_warps=8),
    ],
    key=['NVB'],                                # ★ NVB 变了重新调优 + 重新特化
)
```

`NVB` 是 constexpr 后，`active = rj <= NVB` 是**编译期 mask**，编译器能确定 tile 内有效行数 → 整个 H-loop 完全展开 + 软件流水。`key=['NVB']` 不仅是"按 nvb 选 config"，更是"按 nvb 特化 mask 与循环形状"——这就是为什么 autotune 的 key 要包含所有影响 tile 形状的 constexpr。

#### ③ 规约合并：每 token 一次 barrier reduction（招式 3 的延伸）

原始实现每个 `(t,j)` CTA 在**每个 H-block 后**做 `tl.sum(v*v)` / `tl.sum(v*cw)`——共 `2×(H/BLOCK_H)` 次跨 16-lane shuffle 规约 per row，乘以 `(nvb+1)` 行就是 `2×(H/BLOCK_H)×(nvb+1)` 次。

优化版把所有行的累加器留成 `[MAXR, BLOCK_H]`，沿 `axis=1` **一次性**规约出 `[MAXR]`——**每 token 只 2 次 barrier reduction**（一次 sumsq、一次 dotv），规模是原来的 `1/(nvb+1)`。规约不是热点，但少做 `nvb+1` 倍的 shuffle 仍然减少了片内同步开销。

### 7.3 性能与精度

| 实现 | peak GB/s | 效率 vs 单 die 1300 | cos_sim |
|------|----------|-------------------|---------|
| 原始 2D-grid (per-row CTA) | 673.5 | 51.8% | 1.000000 |
| 寄存器 tile 跨行复用 | **1376.4** | **105.9%**（破 1.3T 目标） | 1.000000 |

**提升 2.04×**，全 44 个 shape `cos_sim=1.000000 ALL PASS`。带宽随 nvb 线性增长的行为保留（nvb=1: ~1300, nvb=8: ~1376），但**斜率翻倍**——因为 cw 不再随 nvb 多读，只有 bf16 行 + 标量输出随 nvb 线性。

### 7.4 关键经验总结

1. **L2 请求墙 vs L2 容量墙**：共享小张量被海量 CTA 重复读时，瓶颈是**请求并发度**而非容量。识别信号：带宽随 CTA 数线性但远低于单 die 墙；探针剥离主体读后带宽腰斩。**解法不是缩小共享张量，而是把"读它的 CTA 数"降下来**——grid 合并 + 寄存器 tile 复用。
2. **寄存器 tile 跨行复用是 Triton 的 "shared memory tiling" 等价物**：用 `[MAXR, BLOCK_H]` 的 2D tile 把多行数据同时留寄存器，让共享张量在 tile 第二维广播复用（`cw[None, :]`）。NVIDIA CUDA 上常用 smem + warp-shuffle 做，Triton 直接用 2D 寄存器 tile 更简单，但要注意寄存器预算。
3. **`MAXR = next_pow2(nvb+1)` 的代价可控**：多出来的行被 `active` mask 掉、不访存，只是占寄存器；当 `nvb ≤ 8` 时 `MAXR=16`，tile `[16, BLOCK_H]` 在 fp32 累加下仍在寄存器预算内（`BLOCK_H=256` → 16 KB/CTA，约 64 reg/thread @ num_warps=8）。`BLOCK_H` 再大会触发 spill——这就是为什么 autotune config 里 `BLOCK_H=1024` 配 `num_warps=8` 而不是更小。
4. **`NVB` constexpr 是把 `tl.where(rj < NVB, ...)` 编译期化的前提**：否则 mask 是运行时计算，编译器不能确定 tile 内有效行数，无法 aggressive 展开 H-loop。**autotune 的 `key` 要包含所有影响 tile 形状 / mask 形状的 constexpr，不只是"调旋钮"用的**。
5. **跟 `_combine_kernel` 的对比**（同算子链上、相邻两步，最优配置却不同）：
   - combine 是"加权和"——按行 unroll + `num_warps=1` 撑带宽（每线程 32B）。
   - score 是"每行独立算 norm-score"——按 token 聚合 + 寄存器跨行复用 cw，`num_warps=8` @ `BLOCK_H=1024` 胜出（tile `[16, 1024]` 大，需要更多线程分摊寄存器）。
   - **两者的 `num_warps` 最优值相反**，因为 tile 维度不同、每线程字节数公式不同。**别假设同一个算子链上共用一套 `num_warps`**——按 tile 形状独立扫。

---

## 8. Benchmark / 精度 / 计时模板

**精度：余弦相似度，阈值 0.99999（不达标打印报错退出）。**

```python
def _cos_sim(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()
```

**计时：back-to-back burst 撑住 DVFS，取每发最优（ms）。**

```python
import time, torch
def _bench_burst(fn, warm=0.6, run=0.6):
    torch.cuda.synchronize(); t0 = time.time(); n = 0
    while time.time() - t0 < warm:            # 预热 + 升频
        fn(); n += 1
        if n % 64 == 0: torch.cuda.synchronize()
    torch.cuda.synchronize()
    reps = max(64, n); best_ms = float("inf"); t0 = time.time()
    while time.time() - t0 < run:
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps): fn()            # 连续发射一整串
        e.record(); e.synchronize()
        best_ms = min(best_ms, s.elapsed_time(e) / reps)
    return best_ms
```

**带宽 = 真实 HBM 流量 / 时间**（数张量字节，别数逻辑运算）：
```python
hbm_bytes = rows_read + out_write + small_tensor_reread   # 各计一次
gbps = hbm_bytes / (ms * 1e6)
```

**每用例输出格式**（对齐 `test_op_moe_scatter_dynamic_quant.py`）：
```
[combine] T=16384 rows=2 nvb=2  cos_sim=1.000000 OK   0.6097 ms   1553.0 GB/s
...
Accuracy: ALL PASS (threshold cos_sim >= 0.99999)
Peak bandwidth: 1553.0 GB/s @ ...  (single-die target 1300 -> reached; dual-die datasheet 3480)
```

**脱离 sglang 依赖**：若 kernel 文件顶层 `import sglang`，在 `import` 前向 `sys.modules` 注入桩模块（`RMSNorm`/`ReplicatedLinear`/`is_hip`/`is_npu`），即可只加载纯 triton kernel 而无需安装 sglang。

**选空闲卡**：跑测前 `mx-smi`，挑 `GPU-Util 0%`、显存 ~540 MiB、状态 `Available` 的卡，`export CUDA_VISIBLE_DEVICES=<卡号>`（共享容器上尤其重要，避免与他人 kernel 争带宽导致读数失真）。

---

## 9. ReAct 优化流程 & 检查清单

**ReAct 环**（每轮只改一个变量）：
1. **读源码 + roofline**：memory-bound 还是 compute-bound？理论最小流量多少？离单 die 墙多远？
2. **提出一个改动**（优先级：招式 1 num_warps → 招式 2 constexpr → 招式 3/4）。
3. **改 kernel + 同步改所有 launch 点与测试的调用签名**（改了参数顺序/名字，`_mix_fused` 与测试 `_launch` 都要跟着改）。
4. **清 triton 缓存**（`rm -rf ~/.triton/cache`）。
5. **跑测**：精度（cos≥0.99999 ALL PASS）+ 逐 shape 带宽 + burst 计时。
6. **判读**：达标（全 shape 破目标）→ 停，备份最优版；否则回到 2。**≤15 轮**。

**收尾检查清单：**
- [ ] 所有 shape `cos_sim ≥ 0.99999`，无 NaN，`ALL PASS`。
- [ ] 峰值达单 die 目标（~1.3 TB/s / ~85%），或已给出"为何止步"的物理解释（探针实证，别硬追双 die 数字）。
- [ ] 小 shape 与大 shape **都**最优；一套配置全胜则不引入分支，否则按 shape/dtype 分支且各分支都不劣化。
- [ ] 未过拟合：逻辑/数学正确性、通用性完好，无 kernel trap。
- [ ] 最优版落到生产目录（如 `mcoplib/`），并留备份 + md5；记录逐 shape 提升率表。
- [ ] 计时用 burst；带宽按真实字节；缓存已清、改动确实被编译。

**C600-U Triton 硬件速记：**
- warp=64 lane，shuffle 子群=16 lane，shared 128KB，reg 255/thread。
- **访存密集：每线程 ≥32B，从 `num_warps=1` 起扫。**
- **循环边界全 `constexpr`；`@jit` 内不用 list.append / static_range 建 list。**
- 单 die 墙：只读 ~1474 / copy ~1322 / RMW ~1326 / triad ~1294 GB/s；datasheet 3480 是双 die。
>>>>>>> CHANGE (2357ed remove the dependency on cutlass)
