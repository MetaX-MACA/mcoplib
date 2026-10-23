# 在 MetaX C600-U 上优化 Triton 算子 —— 实战指南

> 本文是 C600-U 上 **Triton** 算子优化的可复用方法论与技巧库，配套 `c600-optimization-guide.md`（CUDA/MACA 侧）。
> 核心信条：**先分类算子 → 先测墙、再定位瓶颈 → 每次只改一个变量 → 用余弦相似度守精度 → 用 back-to-back burst 计时**。
>
> **先给三条最高杠杆的 C600-U Triton 铁律，后面展开：**
>
> 1. **先给算子分类，再选招式。** C600-U 上有两类完全不同的 memory-bound 算子，最优 playbook 相反：
>    - **纯流式（streaming triad / 逐元素 / 规约）**：目标逼近 HBM 流式墙（~1.3–1.5 TB/s）。核心招式：**`num_warps` 往小调（1/2）**、循环边界 `constexpr` 化。
>    - **gather + 计算混合（含 `tl.dot` / `exp2` / 掩码 scatter）**：读是**跨 head 跳步 gather**，达不到流式墙；SFU（exp2）与微 MMA 串行叠加，天花板远低于纯读。核心招式：**先消除冗余重读（read-once）**、**代数改写把逐对耦合的规约变成 GEMM**、**合并共享算子的多个 matmul**；此类 **`num_warps` 常在 4** 而非 1。
> 2. **访存密集且纯流式：每线程搬运 ≥32 字节 —— 用 *更少* 的 `num_warps`，不是更多。** C600-U 上最反直觉、最高杠杆的一条。
> 3. **一切能变成 `tl.constexpr` 的循环边界 / 行数 / 维度，都要变成 `constexpr`。** 运行时 trip-count 的循环无法展开、无法软件流水，直接锁死带宽。

---

## 目录

- [在 MetaX C600-U 上优化 Triton 算子 —— 实战指南](#在-metax-c600-u-上优化-triton-算子--实战指南)
  - [目录](#目录)
  - [1. C600-U 硬件模型与关键参数](#1-c600-u-硬件模型与关键参数)
  - [2. 第一性原理：优化前先回答的四个问题](#2-第一性原理优化前先回答的四个问题)
    - [2.1 我的算子是哪一类？（决定后面所有招式的选择）](#21-我的算子是哪一类决定后面所有招式的选择)
    - [2.2 我改的代码真的被编译进去了吗？](#22-我改的代码真的被编译进去了吗)
    - [2.3 这块卡的墙在哪、离墙多远？](#23-这块卡的墙在哪离墙多远)
    - [2.4 算术强度（AI）落在 ridge 的哪一侧？](#24-算术强度ai落在-ridge-的哪一侧)
  - [3. 核心招式（按杠杆从高到低）](#3-核心招式按杠杆从高到低)
    - [招式 1（纯流式最高杠杆）：用 `num_warps` 把"每线程字节数"顶到 ≥32B —— 通常是**调小**](#招式-1纯流式最高杠杆用-num_warps-把每线程字节数顶到-32b--通常是调小)
    - [招式 2（次高杠杆）：把循环边界变成 `tl.constexpr`，让循环**静态展开**](#招式-2次高杠杆把循环边界变成-tlconstexpr让循环静态展开)
    - [招式 3（gather+计算 **最高杠杆**）：改并行粒度，消除冗余重读（read-once）](#招式-3gather计算-最高杠杆改并行粒度消除冗余重读read-once)
    - [招式 4（gather+计算次高杠杆）：代数改写，把"逐对耦合的规约"变成 GEMM](#招式-4gather计算次高杠杆代数改写把逐对耦合的规约变成-gemm)
    - [招式 5（gather+计算）：合并"共享同一算子"的多个 matmul（stacked dot）](#招式-5gather计算合并共享同一算子的多个-matmulstacked-dot)
    - [招式 6（varlen）：把变长控制流搬到 host，kernel 只查表](#招式-6varlen把变长控制流搬到-hostkernel-只查表)
    - [招式 7：把标量提取/规约移出访存循环](#招式-7把标量提取规约移出访存循环)
    - [招式 8：消除对共享小张量的重复全局读（L2 瓶颈）](#招式-8消除对共享小张量的重复全局读l2-瓶颈)
    - [招式 9：全并行 2D grid 掩盖延迟](#招式-9全并行-2d-grid-掩盖延迟)
    - [招式 10：back-to-back burst 计时，撑住 DVFS](#招式-10back-to-back-burst-计时撑住-dvfs)
    - [招式 11：`num_stages` 软件流水（仅 compute-bound / 深 K 有用）](#招式-11num_stages-软件流水仅-compute-bound--深-k-有用)
  - [4. MACA / C600-U Triton 编译器硬限制与规避（实测踩坑）](#4-maca--c600-u-triton-编译器硬限制与规避实测踩坑)
  - [5. 分级探针法：如何找到"真实可达天花板"并证伪假设](#5-分级探针法如何找到真实可达天花板并证伪假设)
  - [6. Triton 通用高性能技巧速查](#6-triton-通用高性能技巧速查)
  - [7. 反直觉的坑（Triton 侧实测踩过的）](#7-反直觉的坑triton-侧实测踩过的)
  - [8. 案例 A：`_combine_kernel`（纯流式 triad，756 → 1553 GB/s）](#8-案例-a_combine_kernel纯流式-triad756--1553-gbs)
  - [9. 案例 B：KDA intra-chunk（gather+MMA 混合，86 → 343 GB/s，4.0×）](#9-案例-bkda-intra-chunkgathermma-混合86--343-gbs40)
  - [10. Benchmark / 精度 / 计时模板](#10-benchmark--精度--计时模板)
  - [11. ReAct 优化流程 \& 检查清单](#11-react-优化流程--检查清单)

---

## 1. C600-U 硬件模型与关键参数

| 项目 | 值 | 对 Triton 的含义 |
|------|-----|------------------|
| Warp（wave）宽度 | **64 lane** | `num_warps=1` 即 64 线程；block 线程数 = `num_warps × 64` |
| Shuffle 硬件子群 | **16 lane** | 跨 16 lane 的规约走原生 shuffle；`tl.sum/max` 沿归约轴由编译器映射到此 |
| Shared memory | **128 KB / SM** | `tl.dot`、跨轴规约、`num_stages` 缓冲都吃它，是占用率上限之一 |
| 寄存器 | **255 / thread** | 大 tile × 高 `num_warps` 触发溢出（spill）→ 崩带宽 |
| 计算单元 | 64 AP，每 AP 4 PEU、8 warp/PEU，≤2048 thread/AP | grid 必须足够大才能填满所有 AP |
| **bf16 Tensor Core** | **~270 TFLOP/s** | `tl.dot` 走 MMA 的峰值算力 |
| **fp32 / TF32 dot** | **~34 TFLOP/s** | 真 fp32 dot 是 3-pass 模拟；`allow_tf32=True` 才走硬件 |
| 每线程搬运门槛（流式） | **≥32 B/thread** 才吃满合并访存 | **决定纯流式算子 `num_warps` 选择的第一原则** |
| 访存事务 | 128-bit（16B）向量化最佳 | tile 末维取 2 的幂，指针自然对齐 |

**单 die HBM 实测天花板（务必以此为目标，不要拿 datasheet 3480 当靶）：**

| 访存模式 | 单 die 天花板 |
|----------|--------------|
| 纯流式只读 (contiguous read-only) | ~1474 GB/s |
| 拷贝 (1R+1W) | ~1322 GB/s |
| 原地读改写 (in-place RMW) | ~1326 GB/s |
| triad (2R+1W) | ~1294 GB/s |

> ⚠️ **上表是"连续流式"墙，不是"gather 墙"。** 跨 head/跨行的**跳步 gather**（如 attention 里 `[BC, K]` 每 head 只有 256 B 连续段、行间隔 `H·K`）达不到流式墙。KDA 案例实测：**同样的字节量，纯读 gather 只有 ~838 GB/s（≈流式墙的 57%）**。定目标前必须先用探针测出**你这个访问模式**的读天花板（见 §5），而不是照抄 1474。

> Datasheet 标称 **2.6–4.0 TB/s 是双 die**；单 kernel 单 die 物理够不到。纯流式算子目标 **~1.3 TB/s 单 die（~85%）**；gather+计算混合算子目标是**你探针测出的可达天花板的 ~85%**（往往只有几百 GB/s，这是物理上限而非"没优化好"）。

---

## 2. 第一性原理：优化前先回答的四个问题

### 2.1 我的算子是哪一类？（决定后面所有招式的选择）

- **纯流式**：逐元素 / norm / softmax / 加权和 / 量化写 / 拷贝。→ 招式主线：`num_warps` 往小（§3.1）、循环 `constexpr`（§3.2）。目标逼近流式墙。
- **gather + 计算混合**：内部有 `tl.dot`、`exp2/log/rsqrt` 等 SFU、掩码 scatter，且读是跨 head/跨行跳步。→ 招式主线：read-once（§3.3）、代数改写成 GEMM（§3.4）、合并 matmul（§3.5）、host 端 varlen 元数据（§3.6）。目标逼近**探针测出的复合天花板**，`num_warps` 别默认 1。

**判据**：kernel 里出现 `tl.dot` 或 `tl.math.exp2/log/...` 且带 mask 的 scatter store → 归为第二类，直接走 §3.3 起的招式，别指望达到流式墙。

### 2.2 我改的代码真的被编译进去了吗？

Triton 有**磁盘 JIT 缓存**。改了 kernel 但缓存命中 → 你测的是旧代码。**每次改 kernel 必须清缓存：**

```bash
rm -rf ~/.triton/cache          # 或 /root/.triton/cache（容器内 root）
export TRITON_ALWAYS_COMPILE=1  # 或用这个环境变量
```

> 验证改动是否生效：改一处会导致编译报错的地方，若不报错就是没重编。首启动现场编译特化，不计入 timing。

### 2.3 这块卡的墙在哪、离墙多远？

先 roofline 分类，再算**理论最小流量**（数张量字节，别数逻辑运算量）：
```
bytes = 读入的所有张量字节 + 写出的所有张量字节（各计一次）
带宽 GB/s = bytes / (latency_ms × 1e6)
效率 = 实测带宽 / 该访问模式的可达天花板
```
- 纯流式：可达天花板 = §1 流式墙。效率 < 60% 一定有结构问题（多半 §3.1/§3.2）。
- gather+计算：**可达天花板要用分级探针实测**（§5），因为 SFU 与微 MMA 串行叠加，纯读墙够不到。

### 2.4 算术强度（AI）落在 ridge 的哪一侧？

```
AI = FLOP / HBM字节
bf16 ridge = 270T / 该模式读天花板
```
KDA 案例：AI ≈ 7.4 FLOP/byte（131072 FLOP / 17.7 KB per head·sub-chunk），远小于 ridge（168）→ **memory-bound（受访存强度限制）**。但注意：即使是 memory-bound，若读是 gather 且中间夹 SFU/微 MMA，**实际墙是"读+SFU+MMA 串行叠加"的复合墙**（KDA 只有 343，而纯读 838），不是纯读墙。

---

## 3. 核心招式（按杠杆从高到低）

> §3.1–§3.2 主治**纯流式**；§3.3–§3.6 主治 **gather+计算混合**。先按 §2.1 分类再取用。

### 招式 1（纯流式最高杠杆）：用 `num_warps` 把"每线程字节数"顶到 ≥32B —— 通常是**调小**

```
每线程元素数 = BLOCK_H / (num_warps × 64)
每线程字节数 = 每线程元素数 × dtype_bytes    目标 ≥ 32 B
```

实测（`_combine_kernel`，BLOCK_H=1024，bf16=2B）：

| num_warps | 线程数 | 每线程字节 | 峰值带宽 |
|-----------|--------|-----------|----------|
| 4（默认直觉） | 256 | **8 B** ❌ | ~820 GB/s |
| 2 | 128 | 16 B | ~1450 GB/s |
| **1** | **64** | **32 B** ✅ | **~1553 GB/s** |
| 8 | 512 | 4 B | ~410 GB/s（崩） |

> **反直觉但极有效**：纯流式算子在 C600-U 上，**减少 `num_warps` 让每线程搬更多字节** 比"多开线程"更快，与 NVIDIA 直觉相反。落地：把 `num_warps` 做成可扫旋钮，从 1 起扫 1/2/4/8。
> **⚠️ 但这条只对纯流式成立。** 含 `tl.dot` + 掩码 scatter 的混合算子里 nw=1 会饿死 MMA 与 scatter 发射，最优常在 4（见 §3.5 与案例 B）。

### 招式 2（次高杠杆）：把循环边界变成 `tl.constexpr`，让循环**静态展开**

Triton 只能对**编译期已知**的 trip-count 做展开与流水。运行时标量边界会让每次 `tl.load` 串行等在累加后面，无法 overlap。

```python
# ❌ 运行时 trip-count：无法展开，访存串行
def k(..., NVB, ...):
    for j in range(0, NVB + 1):
        acc += p[j] * tl.load(ptr + j * stride)

# ✅ constexpr trip-count：静态展开，独立 load 背靠背发射
def k(..., NVB: tl.constexpr, ...):
    for j in range(0, NVB):
        acc += p_j * tl.load(ptr + j * stride)
```

`_combine_kernel` 仅此一项：nvb=2 用例 **+37%**（600→820 GB/s）。

> `@jit` 内两个硬坑：**不能 `list.append()` / `vals[j]` 索引 Python list**（AST 直接报错）；**`tl.static_range` 不是可迭代对象**，不能 build list。直接累加进 `acc`。

### 招式 3（gather+计算 **最高杠杆**）：改并行粒度，消除冗余重读（read-once）

**症状**：`token 越多 / 窗口越大，每字节越慢`——说明同一份数据被多个 program 重复读。

**KDA 实测**：token-parallel 版每个 program 只算一个 token i，内层 `for j in range(i_ts, i+1)` **反复重读 `k_j / g_j`**。BC=16 的 sub-chunk 里 `k_j` 被所有 `i≥j` 的 program 各读一次 → **平均重读 ~8.5×，占总访存 86%**。

```python
# ❌ token-parallel：一个 program 一个 token，内层循环重读 k_j/g_j
i_t = tl.program_id(0)                       # grid = (B*T, ...)
b_q = load(q[i_t]); b_g = load(g[i_t])
for j in range(i_ts, i_t + 1):
    b_kj = load(k[j]); b_gj = load(g[j])     # ← 每个 j 被反复读
    acc = tl.sum(b_q * b_kj * exp2(b_g - b_gj))   # ← 还不是 tl.dot

# ✅ sub-chunk-parallel：一个 program 一个 sub-chunk，整块只读一次
i_sc = tl.program_id(0)                      # grid = (num_sc, H/BH)
b_q = load(p_q)   # [BC, K] 整块
b_k = load(p_k); b_g = load(p_g)             # 每行只读一次，冗余 86% → 0%
```

**效果：86 → 200 GB/s（2.3×）**，且消除了"窗口越大越慢"的病态。
> 通用形态：**当输出块内元素两两共享输入时，把"一个输出元素一个 program"改成"一个输出块一个 program"，块内输入一次性载入寄存器，块内复用。**

### 招式 4（gather+计算次高杠杆）：代数改写，把"逐对耦合的规约"变成 GEMM

**症状**：想上 Tensor Core，但求和里有个同时依赖 i、j、k 的因子（如 `2^(g_ik - g_jk)`），无法分离成矩阵乘。

**KDA 的衰减因子分解**（围绕 sub-chunk 首行 `g0`，数学恒等且精确）：
```
2^(g_i − g_j) = 2^(g_i − g0) · 2^(−(g_j − g0))
```
分解后，耦合因子被"预乘"进各自算子，规约变成纯 GEMM：
```python
# ❌ 耦合因子卡在求和内 → 只能 mul+tl.sum（不上 Tensor Core）
b_A = tl.sum(q_i * (k_j * exp2(g_i - g_j)), axis=k)

# ✅ 分解：decay 预乘进 q、k → 纯矩阵乘，可上 MMA
b_dp = tl.math.exp2(b_g - b_g0);  b_dm = 1.0 / b_dp   # 倒数省一次 exp2
b_qg = b_q * b_dp                    # 左算子吸收正向衰减
b_kg = b_k * b_dm                    # 右算子吸收反向衰减
b_A  = tl.dot(b_qg, tl.trans(b_kg), allow_tf32=True)  # 现在是 GEMM
```
> **数值稳定性 + 泛化性**：围绕**每个 sub-chunk 自己的 g0**（而非整 chunk）分解，保证指数落在 (−∞, 0]，不会上溢。**不要为迎合小 gate 用例把 g0 省掉**——那是过拟合，会在大 gate 下失稳。
> **精度**：bf16 输入先 `.to(tl.float32)` 再运算，`acc` fp32，末尾转回 dtype。KDA 全 shape `cos_sim = 1.000000`。

### 招式 5（gather+计算）：合并"共享同一算子"的多个 matmul（stacked dot）

**症状**：kernel 里有两个 `tl.dot` 共享同一个左/右算子，且 M 维很小（≤16）——微 MMA 利用率低、发射两次。

**KDA**：`Aqk = qg @ kgᵀ` 与 `Akk = kbg @ kgᵀ` 共享右算子 `kgᵀ`。沿 M 维把 `qg`、`kbg` 拼成一个 `[2·BC, K]` 再做**一次** MMA：
```python
# ❌ 两次独立 MMA，各 [16,128]×[128,16]，M=16 利用率低
b_Aqk = tl.dot(b_qg,  tl.trans(b_kg), allow_tf32=True) * scale
b_Akk = tl.dot(b_kbg, tl.trans(b_kg), allow_tf32=True)

# ✅ 沿 M 拼成 [32,128]×[128,16] 一次 MMA（MACA 不支持 3D 批量 dot，用 join/split）
b_j    = tl.join(b_qg, b_kbg)                          # [BC, BK, 2]
b_qkbg = tl.reshape(tl.permute(b_j, (2, 0, 1)), (2 * BC, BK))  # [2*BC, BK]
b_A    = tl.dot(b_qkbg, tl.trans(b_kg), allow_tf32=True)      # 一次 [32,128]×[128,16]
b_A    = tl.reshape(b_A, (2, BC, BC))
b_Aqk, b_Akk = tl.split(tl.permute(b_A, (1, 2, 0)))   # split 只能拆末维=2，故先 permute
b_Aqk  = b_Aqk * scale
```
**效果：200 → 343 GB/s（1.7×）**。MMA 发射数减半、M 利用率翻倍、FLOP 不变。**dot 是本算子第一大开销（占 32%），合并它收益最大。**
> 此处 `num_warps=4` 才是最优（不是流式的 1）：MMA + 两块掩码 scatter 需要足够 lane 发射。**混合算子务必把 nw 也交给 autotune 扫 1/2/4/8。**

### 招式 6（varlen）：把变长控制流搬到 host，kernel 只查表

**症状**：kernel 内为定位 varlen 边界做 `for _ in range(20)` 二分查找 / 求 chunk 内偏移——纯标量串行，还阻塞访存。

**KDA**：host 端一次性预计算每个 sub-chunk 的 `(全局起始 token, 有效行数, 列偏移)` 三张表，kernel 里只剩 3 个 `tl.load`：
```python
# host（带缓存，按 (id(cu_seqlens), _version, B,T,BT,BC) 做 key，benchmark burst 内不重建）
for bos, eos in zip(bounds[:-1], bounds[1:]):
    for st in range(0, eos - bos, BC):
        sc_off.append(bos + st); sc_n.append(min(BC, eos-bos-st)); sc_col.append(st % BT)

# kernel：查表代替二分
off  = tl.load(sc_off + i_sc).to(tl.int32)   # 本 sub-chunk 起始 token
n    = tl.load(sc_n   + i_sc).to(tl.int32)   # 有效行数（尾块 < BC，配 boundary_check 补 0）
col0 = tl.load(sc_col + i_sc).to(tl.int32)
```
> 尾块无效行用 `boundary_check` + `row_valid = tl.arange(0,BC) < n` 掩码处理，保证泛化到任意 varlen。

### 招式 7：把标量提取/规约移出访存循环
逐轮 `p_j = tl.sum(tl.where(offs==j, p, 0))` 抽标量是纯浪费。softmax 概率一次算好存标量，循环内只 `acc += p_j * v`。

### 招式 8：消除对共享小张量的重复全局读（L2 瓶颈）
每个 program 都重读同一份 `cw[H]`（fp32）会把它变 L2 热点，bf16 主体明明够快却腰斩。解法：一 program 一 token，`cw` 每块只 load 一次跨所有 row 复用（数据复用 + 抬算术强度）。

### 招式 9：全并行 2D grid 掩盖延迟
把 `H` 切块，grid 做成 `(T, n_h_blocks)` 或 `(num_sc, H/BH)`，让更多 CTA 填满 64 个 AP。≤16 个 fp32 的冗余重算是零成本，换来的并行度远超代价。

### 招式 10：back-to-back burst 计时，撑住 DVFS
C600-U 有频率爬坡。零散计时测到没升频的低值。连续发射一大串 launch 夹在一对 event 之间取每发最优（见 §10 `_bench_burst`）。

### 招式 11：`num_stages` 软件流水（仅 compute-bound / 深 K 有用）
对含 `tl.dot` 的**大 tile** 或长 H-loop 有用；**微 tile（如 [16,128]×[128,16]）上 MACA 的流水基本 inert**，ns=1..4 实测无差（KDA 探针证实，见 §5）。纯逐元素算子上还会吃 shared 反降占用。

---

## 4. MACA / C600-U Triton 编译器硬限制与规避（实测踩坑）

> 这些在 NVIDIA Triton 上能过，在 MACA（triton 3.6.0 / maca 3.8.2）上**编译期直接报错或 launch trap**。写混合算子前务必知道。

| 想做的事 | 报错 / 现象 | ✅ 可用的规避写法 |
|----------|------------|------------------|
| **3D 批量 `tl.dot`**（`[2,M,K]×[2,K,N]`） | launch 时 `memory size or pointer value too large to fit in 32 bit` | 沿 M 维拼成 2D 单次 dot（招式 5 的 join/reshape/split） |
| **静态索引 `t[0]`** 取子张量 | 编译 `unsupported tensor index: constexpr[0]` | 用 `tl.split`（拆末维=2）或 `tl.reshape`+`tl.permute` |
| **切片 `t[:, a:b]`** | 编译 `unsupported tensor index: <slice object>` | 用 `tl.reshape` / `tl.permute` / mask，不要 Python 切片 |
| **`make_block_ptr` 的 `block_shape` 用变量算** | `Expected a list of constant integers in block_shape` | 把表达式直接内联成字面量，如 `(BC, GH * K)`，不要先存进中间变量 |
| **`num_stages` 想让微 MMA 异步重叠** | ns=1..4 完全无差（inert） | 微 tile 放弃异步幻想；靠 read-once + 合并 MMA 降总量 |
| **`@jit` 内 `list.append` / list 索引** | `'append' is not in list` | 累加进标量/张量 `acc` |

**已验证可用的高级操作**：`tl.join` / `tl.split` / `tl.permute` / `tl.reshape` / `tl.trans` / `tl.where` / `tl.make_block_ptr(boundary_check=...)` / `tl.static_range`（作展开，不作 build list）。**reshape+permute+split 是 MACA 上做"堆叠/拆分张量"的黄金组合。**

---

## 5. 分级探针法：如何找到"真实可达天花板"并证伪假设

gather+计算混合算子**不能照抄流式墙当目标**。写一个**分级微探针**，逐步叠加子操作，实测每一级的带宽，得到"复合天花板"并定位第一大开销。

**KDA 探针（T=16384，同一访问模式）逐级结果：**

| 级别 | 内容 | 带宽 | 相对全 kernel 增量 |
|------|------|------|------|
| MODE 0 | 只读 q/k/g（gather） | **838 GB/s** | 读占 41%（240 µs） |
| MODE 1 | + `exp2` 衰减 | **546 GB/s** | +exp2 22%（+129 µs） |
| MODE 2 | + stacked `tl.dot` | **360 GB/s** | +dot **32%**（+191 µs） |
| 全 kernel | + 掩码 scatter 写 | **343 GB/s** | +stores 5%（+30 µs） |

**读法**：
1. **纯读 gather 只有 838**（流式墙 1474 的 57%）——gather 本身就是第一道物理限制。
2. **exp2（SFU）与 dot（Tensor Core）串行叠加，零 overlap**（时间可加）——这是达不到纯读墙的根因。
3. **dot 是最大单项开销（32%）**——所以招式 5（合并 MMA）收益最大。
4. **343 = 546（读+exp2 墙）的 63%**，已接近该复合墙的可达上限；再往上需要硬件支持 SFU/MMA 与访存异步重叠，MACA 微 tile 不具备。

**用探针证伪优化假设（避免浪费迭代 / 避免过拟合）：**

| 假设 | 探针改动 | 实测 | 结论 |
|------|---------|------|------|
| 合并多头连续读更快 | `[BC, GH*K]` 一次读 GH 头 | **0.965×（更慢）** | 每头 256B 已充分合并，更宽反降占用 → 否决 |
| bf16 dot 比 TF32 快 | `tl.dot(...bf16, out_dtype=fp32)` | **0.77×**，cos 0.999997 | 微 tile 上 TF32 更快且更准 → 保留 TF32 |
| `num_stages` 异步隐藏 exp2/dot | `tl.range(num_stages=N)` | ns=1..4 **零变化** | MACA 微 tile 流水 inert → 否决 |
| 小 shape 需要单独配置分支 | autotune 加 size 桶 | 数字完全不变 | 小 shape 是发射开销受限，非配置受限 → 不引入死分支 |

> **方法论**：**每个"我觉得会更快"的想法，先写 10 行探针实测，再决定要不要进生产 kernel。** 探针便宜、结论确定，能挡掉大量过拟合式改动。

---

## 6. Triton 通用高性能技巧速查

| 技巧 | 要点 | 何时用 |
|------|------|--------|
| **`tl.constexpr` 一切静态量** | 维度/行数/循环边界/tile 尺寸全 constexpr | 永远优先 |
| **避免数据相关分支** | 用 `tl.where`/mask，不要数据依赖 `if` | 有条件写/读 |
| **mask 处理尾块/越界** | `boundary_check` + `row_valid` 掩码；非 2 幂维靠 mask | varlen / padding |
| **合并访问** | 相邻 lane 访相邻地址；沿最后一维展开 program | 永远 |
| **fp32 累加、末尾转 dtype** | 输入 `.to(fp32)`，`acc` fp32，`acc.to(out.dtype.element_ty)` | 保精度 |
| **`allow_tf32=True`** | 微 tile MMA 上 TF32 比 bf16 快且够准（cos=1.0） | gather+MMA |
| **read-once 粒度** | 输出块内共享输入 → 一块一 program，块内复用 | gather |
| **代数改写成 GEMM** | 分离逐对耦合因子，让 `tl.sum` 变 `tl.dot` | 有 SFU 权重的规约 |
| **合并共享算子 matmul** | join/reshape 沿 M 拼 → 单次 MMA | 多个 dot 共享算子 |
| **host 端 varlen 元数据** | 二分/偏移搬到 host，kernel 查表 | varlen / packed |
| **少即是多的 warp** | 纯流式从 `num_warps=1` 起扫 | 纯流式 |
| **`@triton.autotune`** | `BLOCK_*`/`num_warps`/`num_stages`/`BH` 交给它扫 | 形状固定 |

**`@triton.autotune` 骨架（混合算子把 nw 扫到 4/8，加 head-group `BH`）：**
```python
@triton.autotune(
    configs=[triton.Config({'BH': bh}, num_warps=nw, num_stages=ns)
             for bh in (1, 2, 3, 4, 6, 12)   # head 分组
             for nw in (1, 2, 4, 8)          # 混合算子：4 常胜；纯流式：1 常胜
             for ns in (1, 2, 3)],
    key=['K', 'H'],                          # 这些变了才重新调优
)
@triton.jit
def kernel(...): ...
```

---

## 7. 反直觉的坑（Triton 侧实测踩过的）

1. **`num_warps` 不是越小越好，也不是越大越好——分类决定方向。** 纯流式→1；含 `tl.dot`+scatter→常 4。别把一类的结论套到另一类。
2. **忘清 `~/.triton/cache`** → 改了 kernel 测到旧特化，得出"改动无效"错误结论。
3. **运行时 trip-count 循环** = 隐形杀手，"窗口越大越慢"就是信号。变 `constexpr`。
4. **`@jit` 里 `list.append` / list 索引 / `static_range` 建 list** → 编译报错。累加进 `acc`。
5. **拿 datasheet 3480（双 die）当靶** → 永远"没达标"。用单 die；gather 算子用探针墙。
6. **拿流式墙当 gather 算子的靶** → 同样永远"没达标"。gather 纯读只有流式墙的一半左右。
7. **微 tile 上 bf16 dot 未必比 TF32 快**（KDA：0.77×且更不准）。默认先试 `allow_tf32=True`。
8. **`num_stages` 在微 MMA / 逐元素上 inert 甚至负收益**。别指望它异步隐藏 SFU/MMA。
9. **合并多头连续读未必更快**（KDA 0.965×）——每头 256B 已够合并，更宽反降占用。
10. **零散计时**没撑住 DVFS，测出偏低。用 burst。
11. **过拟合单一 shape / 迎合单元测试**：为极致性能删掉 g0、砍掉泛化分支、留死逻辑，是**禁止**的。一套配置全 shape 皆优就不引入分支（简单即正确）；确需分支则每个分支都不劣化，数学与边界正确性完好，无 kernel trap。

---

## 8. 案例 A：`_combine_kernel`（纯流式 triad，756 → 1553 GB/s）

**算子**：softmax(scores) → 对 (nvb+1) 个 bf16 行加权求和 → 写 `[T,H]`。典型纯流式 triad。
**基线**：peak 756（nvb=1）/ 600（nvb=2），效率 ~58%；病征 nvb=2 比 nvb=1 每字节更慢。

**仅两处改动即达标（1 个有效迭代）：**

| 改动 | 招式 | 效果 |
|------|------|------|
| ① `NVB` 运行时参数 → `tl.constexpr` | 招式 2 静态展开 | nvb=2 **+37%**（600→820） |
| ② `num_warps` 4 → **1** | 招式 1，BLOCK_H=1024 下 8B→**32B**/线程 | 全盘 **~+90%** |

**结果**：全 16 shape > 1300 GB/s（峰 **1553**），`cos_sim=1.000000 ALL PASS`；小/大 shape 同一份 NW=1 配置皆最优，无需分支。

---

## 9. 案例 B：KDA intra-chunk（gather+MMA 混合，86 → 343 GB/s，4.0×）

**算子**：Kimi Delta Attention 的 intra-chunk 对角块——对每个 sub-chunk 内的行 i、列 j（同 head）算
```
Aqk[i,j] = scale·Σ_k q[i,k]·k[j,k]·2^(g_ik−g_jk)      (j≤i)
Akk[i,j] =        Σ_k β[i]·k[i,k]·k[j,k]·2^(g_ik−g_jk) (j<i)
```
读是**跨 head 跳步 gather**，中间夹 `exp2`（SFU）与微 MMA，末尾**掩码 scatter** 写两个对角块 → **典型 gather+计算混合**（§2.1 第二类）。

**三阶段优化（每阶段一个招式，逐 shape cos=1.000000）：**

| 阶段 | 关键改动（招式） | 峰值带宽 | 增益 |
|------|----------------|---------|------|
| 基线：token-parallel + `tl.sum` 伪点积 | — | 86 GB/s | — |
| 阶段①：sub-chunk read-once + 衰减分解（招式 3+4） | 消除 86% 冗余重读；`tl.sum`→`tl.dot`（decay 预乘进算子，可上 Tensor Core） | **200 GB/s** | **2.3×** |
| 阶段②：stacked dot（招式 5） | `qg`、`kbg` 沿 M 拼成一次 `[32,128]×[128,16]` MMA | **343 GB/s** | **1.7×** |

配套：host 端 varlen 元数据查表（招式 6）、`allow_tf32=True`、`num_warps=4`（非 1）、`BH=1`/`ns=2`（autotune 选出）。

**逐 shape（T=2048/4096/8192/16384）**：基线 83.9/84.9/85.8/86.0 → 最优 188.6/264.0/339.3/343.4 GB/s（提升 2.25×–3.99×）。

**为何止步 343（物理上限，非没优化好）**：分级探针（§5）实测 —— 纯读 gather 838、+exp2 546、+dot 360、+stores 343。SFU 与微 MMA **串行叠加零 overlap**，MACA 微 tile 无异步流水。343 已是"读+exp2"复合墙（546）的 63%。已证伪的加速路：合并多头读 0.965×、bf16 dot 0.77×、num_stages inert。

**可复用结论**：**gather+MMA 混合算子在 C600-U 上的黄金三招 = read-once（改并行粒度）→ 代数改写成 GEMM →合并共享算子的 MMA；`num_warps` 从 4 起扫；目标用探针墙而非流式墙。**

---

## 10. Benchmark / 精度 / 计时模板

**精度：余弦相似度，阈值 0.99999；参考实现用 fp64、且用与 kernel 相同的分解式（对齐数值路径）。**
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

**带宽 = 真实 HBM 流量 / 时间**（数张量字节，各计一次；gather 算子额外注意读的是"逻辑读入量"）：
```python
hbm_bytes = q_bytes + k_bytes + g_bytes + beta_bytes + Aqk_write + Akk_write
gbps = hbm_bytes / (ms * 1e6)
```

**掩码 scatter 算子必须校验写集**：只写该写的元素（如下三角 + 有效行），off-diagonal / 无效行**保持不动**——对参考结果做**逐元素写集比对**，不能只看 cos（cos 会漏掉"多写/少写到不该动的位置"）。

**每用例输出格式**（对齐 `test_op_moe_scatter_dynamic_quant.py`）：
```
[kda-intra] T=16384 H=12 K=128  cos_sim=1.000000 OK   ... ms   343.4 GB/s  eff=63%(vs 546 read+exp2 wall)
...
Accuracy: ALL PASS (threshold cos_sim >= 0.99999)
Peak: 343.4 GB/s  (probe wall: read 838 / +exp2 546 / full 343; streaming wall 1474 不适用于 gather)
```

**脱离 sglang 依赖**：kernel 顶层 `import sglang` 时，在 import 前向 `sys.modules` 注入桩（`RMSNorm`/`ReplicatedLinear`/`is_hip`/`is_npu`），只加载纯 triton kernel。

**选空闲卡**：跑测前 `mx-smi`，挑 `GPU-Util 0%`、状态 `Available` 的卡，`export CUDA_VISIBLE_DEVICES=<卡号>`（共享容器上避免与他人 kernel 争带宽导致读数失真）。

---

## 11. ReAct 优化流程 & 检查清单

**ReAct 环（每轮只改一个变量）：**
1. **分类 + roofline**：纯流式 or gather+计算？memory- or compute-bound？理论最小流量？
2. **测墙**：纯流式用 §1 表；gather+计算**写分级探针**测复合墙（§5），并定位第一大开销。
3. **提出一个改动**（纯流式：招式 1→2；gather+计算：招式 3→4→5→6）。**每个"会更快"的想法先写 10 行探针证伪/证实。**
4. **改 kernel + 同步改所有 launch 点与测试调用签名**。
5. **清 triton 缓存**（`rm -rf ~/.triton/cache`）。
6. **跑测**：cos≥0.99999 ALL PASS + 写集比对 + 逐 shape 带宽 + burst 计时。
7. **判读**：达标（全 shape 破探针墙的 ~85%，或给出物理解释）→ 停，备份最优版 + md5；否则回 3。**≤15 轮**。

**收尾检查清单：**
- [ ] 所有 shape `cos_sim ≥ 0.99999`，无 NaN，写集精确，`ALL PASS`。
- [ ] 峰值达该访问模式**探针墙**的 ~85%，或已给出"为何止步"的**探针实证**解释（别硬追双 die / 流式墙）。
- [ ] 小/大 shape 都最优；一套配置全胜则不引入分支，否则各分支都不劣化。
- [ ] 未过拟合：数学正确性（如 g0 分解的数值稳定）、通用性（varlen/尾块）、无 kernel trap。
- [ ] 最优版落生产目录（`mcoplib/`）+ 备份 + md5；记录逐 shape 提升率表。
- [ ] 计时用 burst；带宽按真实字节；缓存已清、改动确被编译。

**C600-U Triton 硬件速记：**
- warp=64 lane，shuffle 子群=16 lane，shared 128KB，reg 255/thread，bf16 MMA 270T / fp32 dot 34T。
- **先分类**：纯流式→`num_warps` 从 1 起扫 + 循环 constexpr；gather+MMA→read-once + 代数成 GEMM + 合并 MMA，`num_warps` 从 4 起扫。
- **MACA 硬限制**：无 3D 批量 dot、无 `t[0]`/切片索引 → 用 join/reshape/permute/split；微 tile num_stages inert；`@jit` 内不用 list。
- **墙**：流式只读 ~1474 / copy ~1322 / triad ~1294；**gather 纯读约只有一半**（KDA 838）；datasheet 3480 是双 die。
