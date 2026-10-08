# MetaX C600-U (MACA) CUDA Kernel 优化指南

> 面向国产 **MetaX C600-U**（MACA 架构，`sm_80` 兼容）优化 CUDA/HIP kernel 的实战指南。
> 本文按"**先建立正确心智模型 → 再套招式 → 最后看实证案例**"组织，供大模型快速提取可复用经验。

---

## 目录

1. [硬件模型（读优化前必须建立的心智模型）](#1-硬件模型)
2. [第一性原理：优化前先回答三个问题](#2-第一性原理优化前先回答三个问题)
3. [核心招式（按优先级）](#3-核心招式按优先级)
4. [反直觉的坑（实测踩过的）](#4-反直觉的坑实测踩过的)
5. [案例 A：fused_add_rmsnorm（924→1257 GB/s，+36%）](#5-案例-afused_add_rmsnorm)
6. [案例 B：Dynamic INT8 Quant（SREG + SIMD-16 单遍融合）](#6-案例-bdynamic-int8-quant)
7. [案例 C：Warp 内 Bitonic 排序](#7-案例-cwarp-内-bitonic-排序)
8. [案例 D：moe_swiglu_dynamic_quantize（token-centric 重构，101→593 GB/s，5.9×）](#8-案例-dmoe_swiglu_dynamic_quantize)
9. [编译与环境](#9-编译与环境)
10. [硬件速记表 & 检查清单](#10-硬件速记表--检查清单)
11. 案例 E–J：store_kv_cache / qk_rms_norm / reshape_and_cache / topk_transform_prefill（radix-select，采样估阈值减读取遍数，+41%） / **reshape_and_cache 量化扩展（bf16→fp8/int8，硬件 pack 指令破解“计算受限”，fp8 +66%）** / **scale_dynamic_quant（per-token 动态量化+smooth，按 dtype 分支 M 调度，fp8 974→1092 GB/s；warp-per-token 在此反而更慢）** / **moe_softmax_topk（MoE 门控 top-8，latency-bound；串行 argmax 8 轮 → 双调排序网络，关键路径 shuffle 32→3，192→277 GB/s）**

---

## 1. 硬件模型

C600-U 与 A100 参数接近（目标是达到 A100 的 ~80% 性能），CUDA 生态兼容，但**只支持 `sm_80` 及以下特性**，SM80 以上架构特性无法编译。

### 1.1 执行层级（最重要的心智模型）

| 硬件单元 | 说明 | 对应 NVIDIA |
|---|---|---|
| **AP** | 执行 block 的最小硬件单元。block 不能跨 AP。物理 **32 个 AP**，但设备 `multiProcessorCount` **报 28**——grid 计算按实测 mpc 走。 | SM |
| **PEU** | 执行 warp 的最小硬件单元，每 AP 含 **4 个 PEU**。warp 不能跨 PEU。每 PEU 最多并行 8 warp。 | SM sub-partition |
| **warp** | **64 线程**（不是 32！）。理论单 AP 上限 = 4 PEU × 8 warp × 64 = **2048 线程**。 | warp（但 NV 是 32） |
| **shuffle 子组** | **16 线程**（不是 64、不是 32）——shuffle 硬件原生就是 16-lane。见招式 1。 | — |

> ⚠️ 规律：**PEU 上并行 warp 越多，延迟掩盖越好，性能越高。** 优化主线永远是"低寄存器 + 低 shared → 高 occupancy → 让 grid 铺满所有 AP"。

### 1.2 存储层级

| 项 | 值 |
|---|---|
| Shared Memory | **128 KB / AP**（由 4 PEU 共享；多 block 驻留时平分：4 block → 每 block 16 KB） |
| L1 Cache | 32 KB |
| L2 Cache | 2 MB |
| 寄存器 | 64K 32-bit/AP，**单线程最大 255** |
| 向量化 | 支持 **128-bit（float4）** 向量访存 |
| HBM 标称带宽 | datasheet 3.4–4.0 TB/s，但**那是双 die**。见 §2.2。 |

> **shared memory 挤占 warp 的极端例子**：若一个 warp 独占 128 KB shared，则该 AP 的 4 个 PEU 上驻留 warp 数退化为 (1,0,0,0)——occupancy 崩塌。所以访存算子尽量少用甚至不用 shared。

### 1.3 硬性禁忌

- **原子操作弱**：C600-U 原子硬件较弱，热路径**避免 atomic**（一次性 setup kernel 里的小量 atomic 可接受）。
- **禁 `ldg.u8` / `ldg.i8`**：用 `ldg.b32` / `ldg.b64` 代替（小粒度 8-bit load 走慢路径）。
- **每线程至少读/写 32 字节**：小于 32B 的访存无法打满带宽，务必向量化拼成大 load/store。
- **block 必须是 warp(64) 的整数倍**：见招式 4 的半 warp 空转案例。

---

## 2. 第一性原理：优化前先回答三个问题

访存密集型 kernel 最常见的失败不是"招式不够"，而是**方向错**。按顺序回答：

### 2.1 我的改动真的被编译进去了吗？

真实教训：某次优化**前 3 轮改动完全没生效**——因为编译开关关错了，跑的一直是旧的 `.so`，数字纹丝不动却以为"优化没用"。

- **编译开关要覆盖你改的文件**：mcoplib 按子模块分开编译。改哪个子模块的文件，就要打开对应 `BUILD_*_SUBMODULE=ON`。先 `grep -rn "your_file.cu" CMakeLists.txt setup.py` 确认归属。
- **cos_sim 无法检测"改动没生效"**：余弦相似度**对缩放不变**——把归一化系数硬编码成 1.0，cos_sim 仍是 0.9999。
  > **cos_sim 通过 ≠ 你的 kernel 在跑。**
- **可靠验证法**：故意改一个**会移动带宽**的地方（注释掉归约、改 block 大小），看带宽是否跳变。带宽变了，才证明源码真的被编译并执行。

### 2.2 这块 GPU 的显存墙到底在哪？（别拿 datasheet 当目标）

datasheet 的 3.4–4.0 TB/s 是**双 die**值，**单 die 物理上只有 ~1.3–1.4 TB/s**。用微基准实测本卡、按你算子的**访存模式**定墙：

| 访存模式 | 实测（单 die） | 适用算子 |
|---|---|---|
| 只读 | ~1474 GB/s | reduce/argmax |
| copy（1R1W） | ~882–1322 GB/s | — |
| in-place RMW | ~1326 GB/s | — |
| **triad（2R1W）** | **~1294–1350 GB/s** | gather、moe_scatter |
| add_（2R1W in-place） | ~1362 GB/s | — |

> 先写微基准量出你算子对应模式的上限，再定优化目标；否则会去追一个物理上不存在的数字。达到该墙的 **90%+ 就该停**，别为最后几个百分点死磕。

PyTorch 微基准（连续 burst，撑住 DVFS，见招式 5）：

```python
import torch, time
x = torch.randn(10240, 4096, dtype=torch.bfloat16, device="cuda")
r = torch.randn(10240, 4096, dtype=torch.bfloat16, device="cuda"); o = torch.empty_like(x)
def burst(fn, bytes_, warm=1.0, run=0.8):
    torch.cuda.synchronize(); t0=time.time(); n=0
    while time.time()-t0 < warm:
        fn(); n+=1
        if n%64==0: torch.cuda.synchronize()
    torch.cuda.synchronize(); reps=max(64,n); best=0; t0=time.time()
    while time.time()-t0 < run:
        s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
        for _ in range(reps): fn()
        e.record(); e.synchronize()
        best=max(best, bytes_/(s.elapsed_time(e)/reps*1e-3)/1e9)
    return best
print("copy 1R1W:", burst(lambda: o.copy_(x),          10240*4096*2*2))
print("add  2R1W:", burst(lambda: torch.add(x,r,out=o),10240*4096*2*3))
```

也可用 **`mx-smi` 的 HBM 硬件计数器**在紧循环里读真实 DRAM 吞吐，与 effective 指标对比：
- 两者接近 → 已在访存墙，继续调 kernel 徒劳，应从**算法层减少真实流量**。
- effective ≫ HW 计数器 → 有隐藏复用（数据重读命中 L2），effective 虚高。

### 2.3 当前离墙有多远、卡在哪一环？

两遍式算子（第一遍读+归约求统计量，第二遍写）的头号瓶颈是**块级归约的 barrier 气泡**：`__syncthreads()` 期间显存通道空转。

**隔离实验**（别靠猜）：用宏临时**跳过归约/barrier**（保持访存流量不变）跑一次：

```cpp
// 探针：跳过归约，统计量直接设常量（无 barrier），只测纯访存
(void)ss; rms = 1.0f;   // 精度会错，但 cos_sim 仍 0.9999（缩放不变，见 §2.1）
```

若吞吐大涨，瓶颈就锁定在归约同步，不是访存本身。案例 A 实测：摘掉归约后 **925 → 1276 GB/s**——这 350 GB/s 就是 barrier 气泡的代价。

---

## 3. 核心招式（按优先级）

### 招式 1：16-lane shuffle 归约（**最重要**）

C600-U 的 shuffle 硬件子组是 **16 线程**。跨 16-lane 组的步长（off≥16）走**慢路径**；组内（off≤8）才是原生快路径。

- 归约优先用 **`__shfl_down_sync_16(0xffffffffffffffffULL, val, i)`**（i=8,4,2,1）做组内规约，再单独合并各组。
- **合并多组时必须用广播读**（`__shfl_sync` 显式读 lane 0/16/32/48），**不能用 xor 蝶形**——组内规约后只有各组 lane0 值正确，xor 会让其余 60 lane 拿到垃圾，导致正确性 bug。

```cpp
// 组内 16-lane 归约（掩码是 64 位全 1，步长从 8 起）
#pragma unroll
for (int i = 8; i > 0; i >>= 1)
    ss += __shfl_down_sync_16(0xffffffffffffffffULL, ss, i);
```

### 招式 2：单 barrier 归约（3 个 → 1 个）

传统"两级 shared 归约 + 广播"要 3 个 `__syncthreads`。改法：16-lane shuffle 出各组部分和 → **唯一一个** `__syncthreads` → **每个线程各自把少量部分和加起来、自己算统计量**（去掉二级归约 barrier 和广播 barrier）：

```cpp
constexpr int sm_size = NUM_THREADS >> 4;   // 16 线程一组
__shared__ float sm_sum[sm_size];
#pragma unroll
for (int i = 8; i > 0; i >>= 1)
    ss += __shfl_down_sync_16(0xffffffffffffffffULL, ss, i);
if ((threadIdx.x & 15) == 0)                // 每组 lane0 写部分和
    sm_sum[threadIdx.x >> 4] = ss;
__syncthreads();                            // ★唯一的 barrier
float tot = 0.0f;
#pragma unroll
for (int g = 0; g < sm_size; g++) tot += sm_sum[g];   // 每线程自己重算
float rms = __builtin_mxc_rcpf(sqrtf(tot / d + eps)); // 见招式 6
```

### 招式 3：小 block + 高驻留，用别的 block 掩盖 barrier 气泡

单 barrier 还不够——barrier 本身还在。真正把气泡"藏起来"靠**让每个 AP 上驻留很多 block**：A block 同步时，B/C/D block 正在流式读写，显存不空转。

做法：**减小 block、增大每线程处理量**，配 `__launch_bounds__(NT, MIN_BLOCKS)` 提高驻留。

```cpp
template<uint32_t VEC_SIZE, uint32_t NUM_REG, typename T, int NUM_THREADS, int MIN_BLOCKS = 8>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS)
FusedKernelOpt(/* ... */) { float reg[NUM_REG][VEC_SIZE]; /* ... */ }
```

> ⚠️ **寄存器预算是硬约束**：`寄存器/线程 ≈ NUM_REG × VEC_SIZE + 常量开销`，逼近 255 就掉 occupancy。案例 A 中 `NT=128, reg=4` 是甜点区（1257 GB/s），推到 `NT=64, reg=8`（64 寄存器/线程 + 单 warp 无法掩盖延迟）直接崩到 **423 GB/s**。宁可小 block 多驻留。

### 招式 4：128-bit 向量化 + load/store 位置对齐 + warp 边界对齐

- 读用 8-wide bf16（16B）、写用 16-wide int8（16B），每线程 ≥32B。
- **ownership 按统一块大小切分**，保证同一线程 load 与 store 覆盖**相同位置**，两端都 coalesced；否则会"存了没加载过的位置"，正确性 bug。
- **block 必须是 warp(64) 整数倍**：H=1536、VPT=8 → block=192=3 整 warp（最优）；VPT=16 → block=96=1.5 warp，**慢 2×**（半 warp 空转）。用 `AlignedArrayI4<scalar_t, VPT>` 做对齐向量访存。

### 招式 5：连续 back-to-back 发射，撑住 DVFS 升频

C600-U 有 DVFS：空载 XCORE ~514 MHz，持续负载才升到 ~1900 MHz。**逐次 launch + 逐次 sync 的中位数测法会低估带宽**（每次都在低频起步）。正确姿势：warmup 拉升频率后，**一个 sync 内背靠背发射 N 次**取最优（代码见 §2.2 的 `burst`）。测法不对会误判还差 30%。

### 招式 6：`__builtin_mxc_rcpf` 替除法 / 慢速倒数

C600-U 有快速倒数硬件指令，用它替 `rsqrtf` 或除法。访存密集算子里计算不是瓶颈，但顺手换掉是免费收益。

```cpp
// 原：float rms = rsqrtf(ss / d + eps);
float rms = __builtin_mxc_rcpf(sqrtf(ss / (float)d + eps));   // 1/sqrt(...)
// int8 量化里：float tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
```

### 招式 7：grid 大小按 shape 自适应，别固定倍数

- 最优 wave 数经验规律：**≈ C × √(工作量)**（某案例 C≈16，曲线在 [14,20] 很平不敏感）。
- 大 shape 需更多并发 wave 掩盖 HBM 延迟；小 shape grid 过大反因**常驻数据的 L2 冗余重载 + 尾部不均衡**掉速。
- 加下限（每 AP ≥8 wave，打满所有 AP）和上限（不超过输出行数，不空转）。

### 招式 8 MACA 三个高性能内建函数的作用

__builtin_mxc_rcpf(x) :单指令硬件倒数 1/x, 示例：127*rcpf(absmax) / 448*rcpf(absmax) 一次算出量化比例，把每元素除法变成乘法
__builtin_expf(x)：单指令 e^x， 示例：sigmoid：val*rcpf(1+expf(-val*α)) 即 x·σ(αx)（SiLU/swiglu），不走库函数
__builtin_mxc_cvt_pk4_f16tof8(v4f16)：4×fp16 → 4×fp8(e4m3) 单指令打包，返回 uint32，硬件 RNE+饱和 ±448


## 4. 反直觉的坑（实测踩过的）

1. **SPB/多行展开"越多越糟"**：每 block 处理多行（SRC_PER_BLOCK）+ 前置批量 load 在 C600-U 上**无收益甚至反效果**。实测 SPB=8 最差，SPB=1/2 最优（SPB=8 比 SPB=2 慢 ~6%）。原因：SPB 大 → 寄存器压力↑ → occupancy↓。**优先小 SPB，用 grid 数量而非单 block 工作量打满 AP。**
2. **散写优于散读**：gather 类算子中，C600-U 实测**顺序读+散写**比散读+顺序写快（散读 cache miss 惩罚更重）。别默认反转布局能救场，要 A/B 实测。
3. **访存 hint 小幅可用**：`__ldg`（只读 cache）+ `__stcg`（streaming store 绕 L2）对顺序读/散写 shape 有小而稳定的 +1%~2%，在噪声边缘，可留。
4. **"读一次 vs 读多次"重构要先算寄存器**：减少真实流量（如 MoE output-centric 8× 重读改 token-centric 读一次）理论收益大，但**整行进寄存器易溢出**——hidden=4096 一行 64 线程时每线程 128 fp32，溢出后 occupancy 崩塌反而回退（789→615/454）。两难时，访存受限场景**优先保 occupancy 和无 barrier**。
5. **小 shape 是延迟/启动受限，不是带宽受限**：token 极少（如 T=48 总流量 4.6 MB）时，launch + setup 开销摊不薄，GPU 没热起来就算完。这是固有特性，调 grid 无用。汇报要诚实区分"物理受限"与"可优化"，按大 shape 峰值评估达标。测量噪声 ±4~7%，单次高值是运气样本，要多跑取稳态。
6. **别自作聪明替换硬件超越函数**：想用 ALU 多项式（如 bit-trick `2^x` 构造）替 `expf`/`exp2f` 来"卸载 SFU"，在 C600-U 上**反而更慢**。厂商 `__builtin_expf`/`__expf`/`exp2f` 本就映射到同一条最快的硬件 exp 路径；`h2exp`/`hexp` 的 bf16/half 版本**只是两次标量 `expf` 的包装**（源码 `maca_bfloat16.hpp` 里 `h2exp = make(hexp(x), hexp(y))`），**没有真正的 packed-SFU 双发**。案例 D 实测：ALU 多项式 exp2 把 593→**499 GB/s**（-16%）。**结论：超越函数直接用 `__builtin_*` 内建，别手写近似。**
7. **int8 量化有"近似精度地板"，越界即失败**：cos_sim 阈值放宽到 0.999（int8）看似有近似空间，但 sigmoid 形状误差会被逐元素放大到量化输出。案例 D 里用免 exp 的 softsign 近似 `0.5+0.5·g·rcpf(1+|g|)` 把带宽拉到 **650 GB/s**，但 `cos_y=0.9972 < 0.999` **精度不达标**被否。**结论：能提速但破坏精度的近似一律不采用；保精度的硬件 `expf` 路径（593 GB/s）才是该算子的实际最优。** 想省算子指令，先确认近似的相对误差 << 量化步长。

---

## 5. 案例 A：fused_add_rmsnorm

**924 → 1257 GB/s（+36%，达单 die triad 墙 1276 的 98.5%）**，6 轮迭代完成。

**算子**：in-place 融合 Add+RMSNorm。`residual ← input+residual`；`input ← rmsnorm(residual)*weight`。每 token 读 input+residual、写 input+residual = 4×hidden×2B，是典型 2R2W 访存受限、两遍式（先读+归约求 rms，再写）算子。

**优化路径**：

1. **先测墙**（§2.2）：本机 2R2W 类 elementwise 墙 ≈ 1276–1300 GB/s。目标定 1.3 TB/s，而非 datasheet 3480（双 die）。
2. **隔离瓶颈**（§2.3）：`rms=1.0` 探针跳过归约 → 925 跳到 1276。锁定瓶颈是**归约 barrier 气泡**，不是访存本身。原始版用了 3 个 `__syncthreads`（部分和写回 + 二级归约 + `s_rms` 广播）。
3. **单 barrier 归约**（招式 2）：16-lane shuffle → 一个 `__syncthreads` → 每线程各自重算 rms。
4. **小 block 高驻留**（招式 3）：从原始 `NT=512, reg=2`（每线程 8 元素 / 16B load）改为 `NT=128, reg=4`（每线程 32 元素 / 128B load）+ `__launch_bounds__`。别的 block 的流式读写掩盖了 A block 的 barrier。
5. **`__builtin_mxc_rcpf`**（招式 6）替 rsqrt。

**为目标 shape 定制 dispatch**：

```cpp
constexpr int N = 16 / sizeof(T);           // bf16 → N=8
if (d == 4096 && N == 8) {                  // hidden=4096, bf16
    constexpr int NUM_THREADS = 128;        // 小 block → 每 AP 驻留更多
    FusedAddRMSNormKernelOpt<N, 4, T, NUM_THREADS>   // reg=4：32 元素/线程，128B 向量化 load
        <<<nblks, NUM_THREADS, 0, stream>>>(input, residual, weight, d,
                                            stride_input, stride_residual, weight_bias, eps);
    return 0;
}
```

**逐 shape 提升**（bf16, hidden=4096）：T=1024 +20.3%，T=2048 +28.8%，T=4096 +33.2%，T=8192 +37.0%，T=10240 +36.0%；T=16 属小 shape 延迟受限（招式 4/坑 5），不计入达标评估。精度 cos_sim ≥ 0.999998 全 shape。

**反例记录**：`NT=64, reg=8`（每线程 64 元素）→ 423 GB/s。寄存器溢出 + 单 warp 无法掩盖延迟。印证招式 3 的寄存器预算硬约束。

---

## 6. 案例 B：Dynamic INT8 Quant

C600-U 上访存受限 kernel 的范本：**把 max-finding 与量化融合进单遍全局访存**，用 SREG（源数据留寄存器）+ 16-lane SIMD 归约。`op/int8_quant_kernels.cu` 是厂商范本。

要点：源数据一次 load 后留在 `reg_src0[N]` 里，第一遍算 absmax、第二遍直接用寄存器里的值量化写出——**避免二次 load**。归约用招式 1 的 16-lane shuffle，按 `sm_size = NUM_THREADS>>4` 分档。

```cpp
template <typename scalar_t, typename scale_type, typename VT, typename VT1, int NUM_THREADS, bool WITHMASK>
__global__ void dynamic_scaled_int8_quant_kernel_sreg_opt(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, int num_tokens, int* mask_buffer=NULL) {
  if constexpr(WITHMASK) {                       // 可选：按 mask 跳过整块
    __shared__ int sm_max_token;
    if(threadIdx.x == 0) sm_max_token = mask_buffer[blockIdx.y];
    __syncthreads();
    if(blockIdx.x >= sm_max_token) return;
  }
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.y * num_tokens + blockIdx.x;
  float absmax_val = 0.0f;
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  float reg_src0[N];                              // ★源数据留寄存器（SREG）
  scalar_t const* ptr_input = input + token_idx * hidden_size;
  int length = min(hidden_size, NUM_THREADS * N);
  int index = tid * N;
  if(index < length) {
    VT reg_src = *(VT*)(ptr_input + index);       // 一次向量化 load
    scalar_t* p = (scalar_t*)&reg_src;
    #pragma unroll N
    for(int i = 0; i < N; i++) reg_src0[i] = (float)p[i];
    #pragma unroll N
    for(int i = 0; i < N; i++) absmax_val = max(absmax_val, fabsf(reg_src0[i]));
  }

  constexpr int sm_size = NUM_THREADS >> 4;       // 16 线程一组
  __shared__ float sm_max[sm_size];
  float block_absmax_val;
  // 组内 16-lane 归约 + 单 barrier（sm_size==32/16/8/4 分档，此处示 sm_size<=16 的通用形）
  for(int i = 8; i > 0; i >>= 1)
    absmax_val = max(__shfl_down_sync_16(0xffffffffffffffff, absmax_val, i), absmax_val);
  if((threadIdx.x & 15) == 0) sm_max[threadIdx.x >> 4] = absmax_val;
  __syncthreads();
  if(threadIdx.x < sm_size) {                     // 二级归约（组数≤16 时一步到位）
    float data = sm_max[threadIdx.x];
    for(int i = (sm_size>>1); i >= 1; i >>= 1)
      data = max(__shfl_down_sync_16(0xffffffffffffffff, data, i), data);
    if(threadIdx.x == 0) sm_max[0] = data;
  }
  __syncthreads();
  block_absmax_val = sm_max[0];

  if (tid == 0) scale[token_idx] = static_cast<scale_type>(block_absmax_val * 0.0078740157f); // 1/127
  float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);   // 招式 6
  int8_t* ptr_output = out + token_idx * hidden_size;
  if(index < length) {                            // 第二遍：直接用寄存器里的值写出
    VT1 vdst; int8_t* pd = (int8_t*)&vdst;
    #pragma unroll N
    for(int i = 0; i < N; i++) pd[i] = float_to_int8_rn(reg_src0[i] * tmp_scale);
    *(VT1*)(ptr_output + index) = vdst;
  }
}
```

> `sm_size==32` 时需要两级 shared（`sm_max`→`sm_max2`）再各自 16-lane 归约；`sm_size≤16` 时一步到位。分档的意义是让二级归约始终落在 16-lane 快路径内。

---

## 7. 案例 C：Warp 内 Bitonic 排序

warp=64 线程的双调排序（如 MoE top-k 权重排序），全程用 `__shfl_xor_sync`（64 位掩码），无 shared：

```cpp
// MetaX warp = 64 threads
template<uint64_t MASK=0xffffffffffffffff>
__device__ __forceinline__ void warpSortDescendingUpdate(float (&idx_and_weight)[2], int tid) {
    int64_t val = *(int64_t*)idx_and_weight;
    for (int width = 2; width < 64; width <<= 1) {          // 递增构造双调序列
        for (int step = width >> 1; step > 0; step >>= 1) {
            const bool direction = ((tid & width) == 0);
            int64_t other = __shfl_xor_sync(MASK, val, step);
            int other_tid = tid ^ step;
            bool gt = get_weight(other) > get_weight(val);
            bool eq = get_weight(other) == get_weight(val);
            bool ilt = (other >> 32) < (val >> 32);
            bool other_is_big = gt | (eq & ilt);
            bool swap = (tid < other_tid) ^ other_is_big ^ direction;
            val = swap ? other : val;
        }
    }
    for (int step = 32; step > 0; step >>= 1) {             // 最终合并
        int64_t other = __shfl_xor_sync(MASK, val, step);
        int other_tid = tid ^ step;
        bool gt = get_weight(other) > get_weight(val);
        bool eq = get_weight(other) == get_weight(val);
        bool ilt = (other >> 32) < (val >> 32);
        bool other_is_big = gt | (eq & ilt);
        bool swap = (tid < other_tid) ^ (!other_is_big);
        val = swap ? other : val;
    }
    *(int64_t*)idx_and_weight = val;
}
```

---

## 8. 案例 D：moe_swiglu_dynamic_quantize

**101 → 593 GB/s（5.9×），精度全 PASS（cos_sim=1.000000）**。范本意义：**launch 结构（block 数/占用）压倒一切**——原版慢不是招式不够，而是根本没铺满硬件。

**算子**：MoE 专家分组的 SwiGLU + per-token 动态 int8 量化。每个 routed token 行：`gate,up = scatter[t,:H],scatter[t,H:2H]`（bf16）→ `g = silu(gate)*up*smooth_scale[expert]`（fp32）→ `scale[t]=max(|g|)/127` → `y[t]=round(g/scale)` clamp int8。典型 2R1W（读 2H bf16、写 H int8 + 1 fp32），访存受限，两遍式（先求 absmax 再量化）。测试配置：bf16→int8，num_experts=128，ep_size=8（本地 16 experts），topk=8，hidden=1536，num_tokens=[16,1024,…,10240]。

**优化路径**：

1. **根因诊断（最关键）**：原版是 **expert-centric**——`grid=(16 experts, 2)` 只启动 **32 个 block**，远不够铺满 32 个 AP；且 512 线程块里只有 192 线程活跃，还带 cub `BlockReduce` 屏障。峰值仅 **101 GB/s**。**block 数 ≈ AP 数就是死刑**（招式 3/7：占用是访存受限算子的命根）。
2. **Token-centric 重构（101→533 GB/s，核心一步）**：改为**一个 64-lane warp 处理一个 routed token 行**，`grid = ceil(num_routed / WPB)` → 大 T 时启动上千 block **铺满所有 AP**。行→expert 映射：block 把 (≤64 个) expert start 偏移一次性载入 shared，每 warp 线性扫描定位自己的 expert 以索引 `smooth_scale` 行。
3. **纯 warp-shuffle absmax，零 shared / 零 syncthreads**（招式 1）：`warp_absmax` 用 4 步 intra-16 `__shfl_down_sync_16`（8,4,2,1，快路径）求组内 max，再 `__shfl_sync` 组内广播 + 2 步 `__shfl_xor_sync(...,16/32,64)` 合并 4 个 16-lane 组。彻底去掉 cub BlockReduce 的 barrier 串行链。

   ```cpp
   __device__ __forceinline__ float warp_absmax(float v) {   // 64-lane
       v = fmaxf(v, __shfl_down_sync_16(0xffffffffffffffffULL, v, 8));
       v = fmaxf(v, __shfl_down_sync_16(0xffffffffffffffffULL, v, 4));
       v = fmaxf(v, __shfl_down_sync_16(0xffffffffffffffffULL, v, 2));
       v = fmaxf(v, __shfl_down_sync_16(0xffffffffffffffffULL, v, 1));
       v = __shfl_sync(0xffffffffffffffffULL, v, (threadIdx.x & 48), 64); // 组内广播
       v = fmaxf(v, __shfl_xor_sync(0xffffffffffffffffULL, v, 16, 64));   // 合并 4 组
       v = fmaxf(v, __shfl_xor_sync(0xffffffffffffffffULL, v, 32, 64));
       return v;
   }
   ```

4. **smooth_scale 寄存器复用 RPW（533→593 GB/s，第二杠杆）**：每个 warp 把所属 expert 的 smooth 行缓存进寄存器 `sm[VPT][N]`，**连续处理 RPW=4 行复用**，仅在跨 expert 边界（`row >= next_bnd`）才重载。这把 smooth 的重复全局读摊薄到 1/4。sweep 实测 **RPW 是主导杠杆**（RPW 1→4 约 +20%），**WPB 近乎中性**，最优 `WPB=4/RPW=4`：

   ```cpp
   float sm[VPT][N];                       // 该 expert 的 smooth 行留寄存器
   for (int v = 0; v < VPT; ++v)           // 一次载入
       copy_data<sizeof(float)*N>(smooth_scale + (int64_t)e*hidden_size + (v*WARP+lane)*N, sm[v]);
   for (int r = 0; r < RPW; ++r) {         // RPW 行复用同一份 sm[][]
       if (row >= next_bnd) { /* 跨 expert 才重载 sm[][] */ }
       // ... 逐元素 silu*up*sm[v][k]，warp_absmax，量化写出 ...
   }
   ```

5. **128-bit 向量化 + 寄存器直转 int8**（招式 4）：每 lane `float4`（N=8 bf16）向量读 gate/up，fp32 结果留寄存器 `reg_g[VPT][N]`，量化后 `float2` 向量写 int8，两端 ≥32B 合并访问。VPT = H/(64·N) = 1536/512 = 3，`block=WARP*WPB`，是 64 的整数倍。
6. **`__builtin_mxc_rcpf` 替除法**（招式 6）：`silu = g·rcpf(1+expf(-g))`、`tmp_scale = 127·rcpf(absmax)`。**exp 直接用 `__builtin_expf` 硬件路径**（见 §4 坑 6：ALU 近似反而慢；§4 坑 7：softsign 免 exp 近似虽达 650 GB/s 但 cos_y=0.9972 精度不达标被否）。

**逐 shape 峰值**（bf16→int8, hidden=1536, WPB=4/RPW=4）：

| num_tokens | routes | cos_y | 带宽 GB/s | 相对 baseline |
|---|---|---|---|---|
| 16 | 128 | 1.000000 | 18.3 | 小 shape 延迟受限（不计达标） |
| 1024 | 8192 | 1.000000 | 508.0 | ~5.0× |
| 2048 | 16384 | 1.000000 | 553.8 | ~5.5× |
| 4096 | 32768 | 1.000000 | 574.2 | ~5.7× |
| 8192 | 65536 | 1.000000 | 588.8 | ~5.8× |
| 10240 | 81920 | 1.000000 | **593.1** | **5.9×** |

**为何止步 593（未达 1.3T 墙）**：诊断 probe 表明瓶颈是**逐元素 silu 计算依赖链**（convert→exp→add→rcp→3×mul），不是访存——"去掉 exp"探针可达 **940 GB/s**（访存+量化天花板，cos BAD），说明访存本身不是墙。但保精度前提下 exp 无法进一步压缩（见 §4 坑 6/7）。**教训**：访存受限算子偶尔会转成**计算依赖链受限**，此时天花板由算术链长度而非 HBM 决定，要用"摘掉最贵算子"探针把它量出来，避免去追一个被计算链挡住的带宽数字。

**核心可复用经验**：①**先看 launch 结构**——`grid` block 数远小于 AP 数（如只有 32）时，任何微优化都白费，必须先重构成能铺满 AP 的并行维度（token-centric / warp-per-row）；②**warp-per-row + 寄存器复用**是 MoE 类逐 token 算子的通用范式；③**用探针定位真实天花板**（摘归约 / 摘 exp），别盲目对照 datasheet。

---

## 9. 编译与环境

### 9.1 MACA 环境变量

```bash
DEFAULT_DIR="/opt/maca"
export MACA_PATH=${1:-$DEFAULT_DIR}
export CUDA_PATH=${HOME}/cu-bridge/CUDA_DIR
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export PATH=${CUDA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export CUCC_CMAKE_ENTRY=2
```

### 9.2 单文件编译

```bash
# 算子名为 softmax：cu 文件 softmax.cu，产物 softmax
cucc -std=c++17 -arch=sm_80 -O3 ./softmax.cu -o softmax -lcudart
./softmax   # 运行并检查
```

### 9.3 mcoplib 子模块编译（关键：见 §2.1）

改哪个子模块的文件就打开对应开关。例如改 `op/sglang/` 下的文件：

```bash
export BUILD_VLLM_SUBMODULE=OFF BUILD_DEFAULT_OP_SUBMODULE=OFF \
       BUILD_LMDEPLOY_SUBMODULE=OFF BUILD_SGLANG_SUBMODULE=ON
source env_local.sh && python setup.py develop
```

### 9.4 运行结果判读流程（ReAct）

1. 先判断是否运行成功、是否崩溃/OOM/卡死（>5 分钟无法退出则 `kill -9 <pid>`）。
2. 判断精度是否通过（cos_sim ≥ 0.9999）；不通过 = 优化失败，继续 ReAct 迭代。
3. 判断是否达到性能目标；未达到则继续 ReAct 迭代。
4. 直到精度达标 **且** 带宽达标（或已在实测墙 ~96%），汇总结果并给出最终代码路径。

---

## 10. 硬件速记表 & 检查清单

### 10.1 硬件速记

| 项 | 值 |
|---|---|
| warp | **64 线程** |
| **shuffle 硬件子组** | **16 线程**（`__shfl_*_sync_16`，掩码 `0xffffffffffffffffULL`） |
| AP（≈SM） | 物理 64，设备报 `mpc=28`（grid 按实测算） |
| 每 AP | 4 PEU × 8 warp × 64 = 2048 线程上限 |
| Shared Mem | 128 KB/AP（多 block 平分） |
| L1 / L2 | 32 KB / 2 MB |
| 最大寄存器 | 255/线程 |
| 单 die HBM 墙 | ~1.3–1.4 TB/s（datasheet 3480 是双 die，单 die 不可达） |
| 向量化 | 128-bit（float4）；每线程 ≥32B；block 为 64 倍数 |
| 快倒数 | `__builtin_mxc_rcpf(x)` |
| DVFS | 514→1900 MHz，需持续负载升频；测带宽用连续 burst |
| 忌 | 原子操作；`ldg.u8/i8`（用 b32/b64）；block 级 `__syncthreads`+shared 归约串行链 |
| 架构兼容 | `sm_80`，无 SM80 以上特性 |

### 10.2 优化流程检查清单

1. **确认编译**：`grep` 文件归属 → 打对 `BUILD_*_SUBMODULE=ON` → 用"带宽跳变"验证生效（别只信 cos_sim）。
2. **测墙**：按算子访存模式（只读/copy/triad）写微基准量本卡单 die 上限，据此定目标（通常 ~1.3 TB/s，不是 3480）。
3. **建基线**：跑单测拿原始逐 shape 带宽 + 精度（cos_sim ≥ 0.9999），`/tmp` 备份原文件（容器内无 git）。
4. **隔离瓶颈**：两遍式算子先做"摘归约"探针，量化 barrier 气泡代价。
5. **上招式**：16-lane 单 barrier 归约 → 小 block 高驻留（NT=128/reg4 甜点）→ `__builtin_mxc_rcpf` → 128-bit 对齐向量化 → grid 自适应。
6. **每轮复测**：改一处、重编、重测；精度不回退、峰值不回退才保留。达墙 ~96% 即停。
7. **保底**：始终保留当前最优版在 `op/`；`/tmp` 存一份已知最优 `.cu`。
8. **诚实汇报**：区分"物理受限的小 shape"与"可优化"，按大 shape 峰值评估达标；单 die 峰值对照 datasheet 双 die 3480 说明。

---

## 案例 E：store_kv_cache（量化 KV 写入，~10→569 GB/s，57×）

### 问题描述

将 bf16 格式的 K/V 数据读取、per-channel 量化为 int8、写入 cache。属于典型的 **Memory Bound** 算子。

**数据布局**：
- 输入 `packed_qkv`: `[total_tokens, 96, 128]` (q_head=80, kv_head=8, head_dim=128)
- 输出 `k_cache/v_cache`: `[num_blocks, kv_head_num, block_size, head_dim]` (int8)
- Scale: `[kv_head_num, head_dim]` (float32)

### 基线问题

原始 kernel 用 2D Grid `(batch, head)`，token 循环串行，SM 利用率极低（~10 GB/s）。

### 优化过程

#### 第一轮：3D Grid 分布 Token（10→435 GB/s，42×）

```cpp
// ❌ 原始: token 串行，SM 大量空闲
dim3 grid(batch_size, kv_head_num);
for (int t = 0; t < q_len; t++) { /* 串行处理 */ }

// ✅ 优化: Z 维度分布到所有 SM
int sm_count;
cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
dim3 grid(batch_size, kv_head_num, sm_count);  // Z = SM 数量
for (int t = blockIdx.z; t < q_len; t += gridDim.z) { /* 并行处理 */ }
```

**关键洞察**: 必须让 Grid 覆盖所有 SM 才能充分利用硬件并行度。

#### 第二轮：寄存器预加载 Scale（435→569 GB/s，+31%）

```cpp
// ❌ 循环内重复读取
for (int t = 0; t < q_len; t++) {
    float sk = k_scale[h * dim + tid];  // 每次循环都读全局内存
    float sv = v_scale[h * dim + tid];
}

// ✅ 循环外预加载到寄存器
const float my_k_scale = k_scale[h * dim + tid];  // 只读一次
const float my_v_scale = v_scale[h * dim + tid];
for (int t = 0; t < q_len; t++) {
    // 直接使用 my_k_scale, my_v_scale
}
```

#### 尝试但失败的优化

| 尝试 | 结果 | 原因 |
|------|------|------|
| cp.async 双缓冲 SMEM | 387 GB/s (-31%) | `__syncthreads()` 同步开销 > 收益 |
| 2x 循环展开 | 7 GB/s (-98%) | 寄存器溢出到局部内存 |
| 4x 循环展开 | 7 GB/s (-98%) | 同上 |
| 256 线程/block | 347 GB/s (-39%) | 128 线程已足够，更多线程无收益 |
| 64 线程 + 2 元素/线程 | 414 GB/s (-27%) | stride 访问破坏合并 |

### 最终 Kernel（v6，569 GB/s）

```cpp
template <typename scalar_t>
__global__ void store_kv_cache_kernel(
    const scalar_t* __restrict__ packed_qkv,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    int8_t* __restrict__ k_cache,
    int8_t* __restrict__ v_cache,
    /* ... 其他参数 ... */
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;  // 0..127 = head_dim

    // 1. 寄存器预加载 scale (避免循环内重复读全局内存)
    const float my_k_scale = k_scale[head_idx * head_dim + tid];
    const float my_v_scale = v_scale[head_idx * head_dim + tid];

    // 2. Token 循环: Z 维度分布到所有 SM
    for (int token_idx = blockIdx.z; token_idx < q_len; token_idx += gridDim.z) {
        // 3. 合并读取: 连续线程读连续 bf16 地址
        int64_t src = src_token * stride0 + head * stride1 + tid;
        scalar_t k_val = packed_qkv[src];
        scalar_t v_val = packed_qkv[src + kv_offset];

        // 4. 合并写入: 连续线程写连续 int8 地址
        k_cache[dst_k + tid] = float_to_int8_rn(k_val * my_k_scale);
        v_cache[dst_v + tid] = float_to_int8_rn(v_val * my_v_scale);
    }
}

// Host 启动
dim3 blocks(batch_size, kv_head_num, sm_count);  // 3D Grid
const int threads = 128;  // head_dim = 128
```

### 性能数据

| 配置 | 基线 | 优化后 | 提升 |
|------|------|--------|------|
| B=1, Q=32768 | ~10 GB/s | 569 GB/s | 57× |
| B=1, Q=16384 | ~10 GB/s | 548 GB/s | 55× |
| B=1, Q=8192 | ~10 GB/s | 508 GB/s | 51× |

### 经验总结

1. **3D Grid 是 memory bound 算子的第一步**: 确保所有 SM 都有工作
2. **寄存器预加载**: 对于循环内重复读取的少量数据，预加载到寄存器可带来显著收益
3. **SMEM 双缓冲不总是有效**: 当数据流已经是 streaming 模式时，`__syncthreads()` 开销可能抵消收益
4. **循环展开要谨慎**: 2x 安全，4x+ 可能导致寄存器溢出
5. **合并访存是生命线**: 跨步访问 vs 合并访问差距可达 300x
6. **量化 kernel 的峰值比例有限**: bf16→int8 涉及类型转换指令开销，纯 copy kernel 才能接近理论峰值

---

## 案例 F：qk_rms_norm（per-head QK-Norm，285→929 GB/s，3.26×）

**算子**：对融合 QKV 张量做 **per-head RMSNorm**。token_data 为 bf16 `[num_tokens, (q_head_num + 2*kv_head_num)*128]`，布局 `[Q(q) | K(kv) | V(kv)]`；对**每个 Q 头**（权重 q_norm_weight）和**每个 K 头**（权重 k_norm_weight）独立做 RMSNorm，**原地写回**，**V 区完全不碰**。典型访存受限、两遍式（先读+归约求 sum_squares，再写）算子。有效字节 = `num_tokens*(q+kv)*128*2(bf16)*2(R+W)`。测试配置：bf16，(q,kv)=(8,1)/(32,4)/(16,2)，head_dim=128，num_tokens=[16,1024,…,10240]。

### 优化路径（9 版 ReAct，逐 shape 峰值）

| 版本 | 关键改动 | 峰值 GB/s |
|---|---|---|
| baseline | **1 个 64 线程 warp / 头**，VecSize=2（**每线程仅 8B**），64-lane `__shfl_xor` 归约，`rsqrtf` | 284.6 |
| v1 | 16 线程/头 + VecSize=8（**32B/线程**）+ 16-lane 归约 + `__builtin_mxc_rcpf` + 扁平网格 | 551.4 |
| v2 | 8 线程/头（64B/线程） | 798.8 |
| **v3/v5（最优）** | **4 线程/头（128B/线程），128 线程块，packed-bf16 留寄存器** | **928.7** |
| v4 | 2 线程/头（256B/线程）→ 寄存器溢出**回退** | 536.5 |
| v6/v7/v8/v9 | min_blocks=16 / 256 线程块 / nontemporal store / 2D 网格 | ~928（**全中性**） |

### 关键优化举例

**① 每线程字节数是头号杠杆（8B→128B，×3.26 的主因）**。baseline 每线程只搬 8B（VecSize=2 的 4B 读 + 4B 写），远低于 skill 的「≥32B/线程」。改成 128-bit 向量化（bf16 float4 = 8 元素 = 16B）后，用「**每头几个线程**」控制每线程工作量：

```cpp
constexpr int kVecSize = 8;            // 128-bit：一个 bf16 float4 = 8 元素 = 16B
constexpr int kThreadsPerHead = 4;     // 128/8/4 = 4 个 float4/线程 = 32 元素 = 128B/线程
constexpr int kHeadsPerBlock = 32;     // block = 4*32 = 128 线程（case-A 甜点区）
// sweep 实证：16线程/头(16B)=551 → 8(64B)=799 → 4(128B)=928 → 2(256B)=536(寄存器溢出崩)
```
> **规律**：`每线程字节 = kVecSize*2*(128/kVecSize/kThreadsPerHead)`，从 16B 一路涨到 128B 单调提速，到 256B（2 线程/头）时寄存器溢出、occupancy 崩塌回退——与案例 A「NT=64/reg8→423」同一条硬约束。**甜点是 128 线程块 / 每线程 32 元素**。

**② 16-lane 子组归约，零 shared / 零 `__syncthreads`（招式 1）**。head_dim=128 恰好一个头由 ≤16 lane 处理，归约完全落在**原生 16-lane 快路径**，butterfly 后每 lane 都拿到全和，不需要 shared，也没有 barrier：

```cpp
// ThreadsPerHead <= 16：所有 offset 都 < 16，全程快路径；每 lane 都得到完整 sum
template <int ThreadsPerHead>
__device__ __forceinline__ float subgroup_all_reduce_sum(float v) {
#pragma unroll
  for (int off = ThreadsPerHead / 2; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffffffffffULL, v, off, ThreadsPerHead);
  return v;
}
```
> 对比 baseline 的 64-lane `__shfl_xor`（off=32/16 走**慢路径**）。**头维度 ≤16 的 per-head 归约，直接用 sub-warp 宽度的 shuffle，天然避开慢路径且无需 shared。**

**③ 源数据以 packed bf16 留寄存器（省一半寄存器）**。两遍式算子第一遍读进来别急着展开成 fp32——按 bf16 打包留寄存器（16 个 vs 32 个），只在算 sum_squares 和写回时**瞬时**转 float：

```cpp
InputVector reg_input[kRegVec];        // 打包 bf16，不是 float reg[..][..]
float ss = 0.f;
#pragma unroll
for (int r = 0; r < kRegVec; ++r) {
  reg_input[r] = head_data[lane_id + r*ThreadsPerHead];   // 一次 128-bit load
  for (int e = 0; e < VecSize; ++e) { float x = to_float(reg_input[r].data[e]); ss += x*x; }
}
// ...归约、rcpf...
for (int r = 0; r < kRegVec; ++r)      // 写回时再转 float 乘 inv_rms*weight
  for (int e = 0; e < VecSize; ++e) out.data[e] = to_bf16(to_float(reg_input[r].data[e])*inv_rms*w[e]);
```
> 本例中编译器对两种写法生成相同代码（928 == 928），但打包留寄存器是**更省寄存器**的默认写法，shape 更大或每线程元素更多时能保住 occupancy。

**④ 扁平 1D 网格枚举 (token×head)，消除尾列浪费**。baseline 用 `grid=(heads/HeadsPerBlock, tokens)` + HeadsPerBlock=2，norm_heads=9 时最后一个 block 半空转。改为把所有 (token, head) 摊平成一维、`__launch_bounds__(128,8)` 提高驻留，铺满所有 AP：

```cpp
int64_t total_heads = num_tokens * norm_head_num;
dim3 grid((total_heads + kHeadsPerBlock - 1) / kHeadsPerBlock);
// kernel 内：global_head/norm_head_num 反解 token，% 反解 head_in_token
```
> 用 `__builtin_mxc_rcpf(sqrtf(ss/d+eps))` 替 `rsqrtf`（招式 6，免费收益）。

### 为何止步 ~929（未达 1.3T 墙）——探针实证，别硬追

- **摘归约探针**（`inv_rms=1.0`，访存流量不变）→ 只从 928 到 **954 GB/s（+3%）**，证明**不是归约/barrier 瓶颈**（本就无 barrier）。
- 实测本卡**连续原地 RMW 墙 = 1195 GB/s**；但本算子每个 token 必须**跳过 V 区**（V 不参与归一化），访存天然带间隙，物理上到不了全连续墙。
- **929 / 954 = 该「跳 V 间隙」访存模式自身天花板的 97%**。9 个结构变体（线程/头比、块大小、min_blocks、store hint、2D 网格）全部收敛在 928——已到平台，符合 §2.2「达墙 90%+ 即停」。

### 逐 shape 提升率（bf16, T=10240）

| (q,kv) | baseline → best | 提升 |
|---|---|---|
| (8,1) | 254.5 → 860.2 | **3.38×** |
| (32,4) | 284.6 → 928.7 | **3.26×** |
| (16,2) | 283.6 → 902.7 | **3.18×** |

（T=16 为小 shape 启动/延迟受限，不计入达标评估，见坑 5。）

### 核心可复用经验

1. **per-head / 定长归约算子：用「每头 N 个线程」把每线程字节顶到 128B（32 元素）**，而不是「一个 64 线程 warp 处理一个 128 维头」（每线程才 8B，白白浪费向量宽度）。8B→128B 是本例 3× 提升的主因。
2. **归约宽度 ≤16 时直接用 16-lane 子组 shuffle**：无 shared、无 `__syncthreads`、全程快路径。head_dim=128 的 per-head RMSNorm 是教科书场景。
3. **两遍式源数据按原始 dtype 打包留寄存器**，瞬时转 fp32，省一半寄存器保 occupancy。
4. **有「不参与计算的区域被跳过」（如 V 区）时，先量出「带间隙」访存模式的真实天花板**（摘归约探针 + 微基准），别拿全连续 RMW 墙当目标——本例真实天花板 954 而非 1195，达到它的 97% 就该停。


## 案例 G：store_kv_cache（量化 KV 写入，dispatch 路径重写，331→940 GB/s，2.83×）

**算子**：`store_kv_cache_cuda_interface`——把融合 QKV 里的 K/V 区做 **per-(head,dim) 量化**（bf16 × float scale → int8，`float_to_int8_rn`）**写入 paged KV cache**。源 `packed_qkv[num_tokens, (q+2*kv)*head_dim]` 布局 `[Q | K | V]`；目标 4D `k/v_cache[num_blocks, kv_head, block_size, head_dim]`。**只搬 K/V、不碰 Q**，无跨 token 复用，是典型的**非对称 elementwise copy**（读 2B/元素、写 1B/元素）。测试配置：bf16，(q,kv)=(80,8)，head_dim=128，prefill Q=1k~32k / decode B16 Q=1~4。

> ⚠️ 本案例与案例 E 是**两个不同的 kernel**：E 是另一条量化 KV 路径；G 是 `store_kv.cu` 里 interface **实际派发**的 `store_kv_cache_kernel_v2`。

### 优化路径（v2 baseline → v8）

| 版本 | 关键改动 | 峰值 GB/s |
|---|---|---|
| baseline (v2) | 标量：**每线程 1 元素**（2B 读 + **1B int8 写**），grid `(B, kv_head, sm)`，128 线程 | 331.6 |
| v8-a | 128-bit 向量化（16B 读 + 8B 写/线程），head 折进 block，grid `(B,1,sm)` | 245（**回退！**） |
| v8-b | 同上 + `grid.z = 16*sm`（token 维铺满占用率） | 931.8 |
| **v8-c（最优）** | `grid.z = 32*sm` **且按 token 行数封顶**（prefill 满占用、decode 不空转） | **939.6** |

### 关键优化举例

**① 先确认「哪个 kernel 真的被 dispatch」——别被文件里的死代码误导（§2.1 的延伸）**。`store_kv.cu` 里同时躺着 `_opt_original`（向量化）、`v7`（cp.async + SMEM 双缓冲）等**从未被 interface 调用**的版本。真正派发的是最朴素的标量 `v2`。任何优化前先 grep interface 的 launch 行：

```cpp
// interface 里真正的一行——它决定了你要改谁；文件里其它 kernel 可能全是死代码
store_kv_cache_kernel_v2<maca_bfloat16><<<blocks, threads, 0, stream>>>( ... );
```
> **教训**：「这个实现基于 SMEM 64KB」之类的前提，必须对照 dispatch 路径核实。本例真实 dispatch 的 v2 **一个字节 SMEM 都没用**——纯 copy 无复用，SMEM staging 只会徒增 `__syncthreads`（死代码 v7 正是栽在这，见坑 1）。**扩大到 128KB SMEM 对本算子零收益。**

**② 每线程字节数是头号杠杆：1 元素 → 8 元素（招式 4）**。baseline 每线程只搬 2B 读 + 1B 写，事务碎到打不满 HBM。让每线程吃 N=8 个连续 dim：**16B `uint4` 读 + 8B 打包 `int64` 写**（int8×8），scale 与 token 无关、循环外预载寄存器：

```cpp
template <typename scalar_t, int N>   // N=8
__global__ void store_kv_cache_kernel_v8(...) {
  const int base = threadIdx.x * N;               // 一线程负责 8 个 dim
  const int head_idx = base / head_dim, dim0 = base % head_dim;
  float sc_k[N], sc_v[N];                          // scale 预载寄存器（token 无关）
  #pragma unroll
  for (int j=0;j<N;j++){ sc_k[j]=k_scale[head_idx*head_dim+dim0+j];
                         sc_v[j]=v_scale[head_idx*head_dim+dim0+j]; }
  for (int t = blockIdx.z; t < q_len; t += gridDim.z) {
    uint4 rk = *reinterpret_cast<const uint4*>(kp);      // 16B 向量读
    int8_t ok[N];
    #pragma unroll
    for (int j=0;j<N;j++) ok[j]=float_to_int8_rn(to_float(pk[j])*sc_k[j]);
    *reinterpret_cast<int64_t*>(kd) = *reinterpret_cast<const int64_t*>(ok); // 8B 打包写
  }
}
```

**③ 折 head 进 block 会砍掉并行度——必须用 grid.z 沿 token 维把占用率补回来**。这是本例最反直觉的一步：v8-a 把 kv_head 从 grid.y 折进 block（128 线程）后，grid 塌成 `(1,1,28)`=28 block，只有 28×128=3584 线程 ≪ 28 AP×2048，**直接从 331 掉到 245**。修复：`grid.z` 开到 `32*sm`，kernel 内 `t += gridDim.z` 天然吃掉多余 block：

```cpp
int gridz = sm_count * 32;                     // 每 AP 32 个 block，token 维铺满
dim3 blocks(batch_size, 1, gridz);
```
> **规律**：向量化把「每 block 的线程数」压小之后，一定要同步把「block 数」放大，否则占用率净亏。16*sm=931、32*sm=940（+1%），说明已从占用率受限转为**带宽受限**——到此为止。

**④ 一份 kernel 同时照顾 prefill 与 decode：grid.z 按 token 行数封顶（招式 7）**。prefill（token 上万）要满占用率；decode（B16 各 1~4 token）若也开 `32*sm=896` 个 block，绝大多数空转、纯启动开销。用 host 端已知的 `packed_qkv.size(0)` 封顶：

```cpp
int tok_hint = (int)packed_qkv.size(0);        // 总 token 行数：prefill 大 / decode 极小
int gridz = sm_count * 32;
if (tok_hint < gridz) gridz = tok_hint < 1 ? 1 : tok_hint;   // decode 不再启动上千空 block
```
> prefill 时 `tok_hint ≫ 896`，封顶不生效、保持满占用；decode 时 `gridz` 收到个位数，避免调度浪费。**无需为两种 attn_mode 写两条代码路径**。

### 为何止步 ~940（未达 1200 目标）——非对称 copy 的物理天花板

- 本算子**读 2B/元素、写 1B/元素**：int8 写只搬一半字节，却与读占同样多的事务。这类**非对称 copy** 的有效带宽天生低于对称 copy 能达到的峰值比例。
- block 倍数 16→32 只涨 ~1%（931→940），证明**已是带宽受限、非占用率受限**，launch 参数无空间可挖。
- 940 GB/s ≈ 单 die 1.3TB/s 墙的 **72%**；对「2B 读 + 1B 写」的量化写入模式，这已接近其自身访存天花板。decode 用例（Q=1~4，~0.03ms）是**启动延迟受限**，物理上无带宽可提。

### 逐 shape 提升率（bf16, (q,kv)=(80,8)）

| shape | baseline (v2) → best (v8) | 提升 |
|---|---|---|
| Prefill Q=32768 | 331.6 → 939.6 | **2.83×** |
| Prefill Q=16384 | 326.9 → 858.6 | 2.63× |
| Prefill Q=8192 | 316.1 → 805.9 | 2.55× |
| Decode B16 Q=4 | 18.9 → ~14（启动延迟受限，不计入达标） | — |

### 核心可复用经验

1. **优化前先核实 dispatch 路径**：一个 `.cu` 里可能有多份 kernel（含 cp.async/SMEM 版），但 interface 往往只 launch 最朴素那份。别对着死代码优化，也别被「基于 XX SMEM」的前提带偏——纯 copy 无复用时 SMEM 扩容零收益。
2. **标量 copy 的头号杠杆是每线程字节数**：1 元素（2B/1B）→ 8 元素（16B 读/8B 写），配 int8 打包 `int64` 写、scale 寄存器预载，是本例 2.83× 的主因（招式 4）。
3. **折维度进 block 必须补 grid**：把 head/其它维折进 block 换取向量化后，务必用 `grid.z` 沿最长维（token）放大 block 数，否则占用率净亏、比不折还慢。
4. **prefill/decode 一份代码搞定**：`grid.z` 开满再用 host 端 token 行数封顶，大 shape 满占用、小 shape 不空转，避免为不同 attn_mode 分叉代码。
5. **非对称 copy（读写字节数不等）先估自身天花板**：读 2B 写 1B 的量化写入，达墙 ~72% 已接近该模式极限，别拿对称 RMW 墙当目标（呼应案例 F 的「跳 V 间隙」思路）。


## 案例 H：reshape_and_cache（vLLM KV 写入，转置布局，210→647 GB/s，3.08×）

**算子**：vLLM 的 `reshape_and_cache`（`op/vllm/cache_kernels.cu`），把每步新算出的 K/V 写入 paged KV cache。本例 `cache_dtype="auto"`（**纯 bf16 拷贝，无量化**），是最"干净"的访存受限算子——没有任何计算，纯粹考验访存布局。测试配置：bf16，num_heads(kv)=[1,2,4]，head_dim=128，block_size=512，x=8，prefill num_tokens=10240 / decode=16。

**数据布局（读懂这个就懂了瓶颈）**：
- 源 `key/value[num_tokens, num_heads, head_size]`——最内层是 `head_size`(d)，即同一 token/head 的 d 维连续。
- 目标 `key_cache[num_blocks, num_heads, head_size/x, block_size, x]`——最内层 `x=8`，次内层 `block_size`(slot)。
- 目标 `value_cache[num_blocks, num_heads, head_size, block_size]`——**最内层是 `block_size`(slot)，不是 d！**

> ⚠️ **核心瓶颈 = value 是"d↔slot 转置写"**。源 value 沿 d 连续，但 value_cache 沿 slot(token) 连续、d 为外层跨步。于是"一个线程写一个 token 的整行 d"必然是**沿 slot 方向的大跨步散写**（stride=block_size×2B）——每个 2 字节写命中一个独立 sector，**sector 效率仅 ~6%**，带宽被打到 123~210 GB/s（且 head 越多越慢，见下）。key_cache 最内层是 x=8、d 以 x 为粒度连续，**不需要转置**，是普通合并写。**这是"源与目标最内层维度不一致 → 隐藏转置"的典型陷阱（呼应案例 G 坑 5 的非对称，但这里是布局转置）。**

### 优化路径（原始 baseline → iter6 最优，15 轮 ReAct 内收敛）

| 版本 | 关键改动 | 峰值 GB/s |
|---|---|---|
| baseline | 扁平化 flatten、`block(128)`、`grid(min(total_work, sm*4))`，**每线程标量 2B 散写** value | 209.8 (H=1) |
| iter1–4 | uint4 向量化 + grid 沿 token 铺满（key 路径提速，value 仍散写受限） | ~300–440 |
| iter5 | **SMEM 转置**：token-major tile，value 合并读入 SMEM → `__syncthreads` → 沿 slot 合并写；256 线程 | 530.8 |
| **iter6（最优）** | 在 iter5 上**把 slot_mapping 缓存进 SMEM，per-chunk 连续性/块号/偏移只预计算一次**（不再每个 d 重复读全局 slot）；512 线程，TILE_T=64 | **647.1 (H=4)** |
| iter7/8/9/11/12 | TILE_T=128→519、TILE_T=32→557、1024线程→512、grid.z 拆 head→395、d-major SMEM→351 | 全部**回退** |

### 关键优化举例

**① 识破隐藏转置，用 SMEM 把"散写"变"合并写"（本例 3× 的核心）**。value 的散写无法靠向量化救（沿 slot 跨步，uint4 也是跨步）。解法：**block 内先把一批 token 的 value 按 d 合并读进 SMEM，`__syncthreads` 后换个方向、沿 slot 合并写出**——读写两端都 coalesced，转置发生在片上。关键是"**同一个 uint4 写覆盖 8 个连续 token 的同一个 d**"（8 个 token 落在连续 slot 时）：

```cpp
// value: 源沿 d 连续 → 合并读入 SMEM（token-major tile）
//        SMEM 沿 slot 连续 → 合并写出：一个 uint4 = 8 个连续 token 在同一 d 的值
for (int t = 0; t < TILE_T; ++t)                 // 合并读：连续线程读连续 d
    s_val[t][d_vec] = *reinterpret_cast<const uint4*>(&value[(tok0+t)*H*D + h*D + d*8]);
__syncthreads();
// 合并写：8 个连续 token 拼成一个 uint4，写到 value_cache 沿 slot 连续的地址
if (slots_contiguous_and_8aligned)
    *reinterpret_cast<uint4*>(&value_cache[blk*... + d*block_size + slot0]) = pack8(s_val, d);
```
> **规律**：当源与目标"最内层维度不同"（这里 d vs slot），任何一端直接向量化都救不了另一端，**必须用 SMEM 做片上转置**，让 global 读、global 写各自沿自己的连续维合并。这与案例 E/G 的 store_kv 形成鲜明对比——那两个 kernel 目标最内层是 head_dim、与源一致，**无需转置故无需 SMEM**（940 GB/s）；本例因转置**不得不用 SMEM，天花板也因此更低**（见"止步原因"）。

**② 不同子张量走不同策略——别一刀切**。key_cache 最内层是 x=8、d 连续，是普通合并写，**直接每 token uint4 写即可，不进 SMEM**（进 SMEM 反而多一次 barrier）。只有 value 需要 SMEM 转置。同一 kernel 里 key 走直写、value 走 SMEM，是本例的关键分流：

```cpp
// key: 无转置，直接向量化写（x=8 让 d 以 8 为粒度连续）
*reinterpret_cast<uint4*>(&key_cache[blk*... + (d/8)*block_size*8 + slot*8 + d%8*0]) = key_vec;
// value: 走 ① 的 SMEM 转置路径
```

**③ slot_mapping 元数据只算一次，别每个 d 重复读全局（iter5→iter6，+22%）**。value 每行 d 都要把 token→(block, slot) 映射一遍；原来每个 d 迭代都重读全局 `slot_mapping` 并重算块号/偏移/连续性。iter6 **把整个 tile 的 slot 一次性载入 SMEM，并预计算每个 8-token chunk 的"是否连续/块号/组内偏移"存 SMEM，之后所有 d 复用**：

```cpp
__shared__ int64_t s_slot[TILE_T];               // 整 tile 的 slot 一次性载入
__shared__ int64_t s_blk[TILE_T/8], s_off[TILE_T/8];
__shared__ int      s_ctg[TILE_T/8];             // 每个 8-token chunk 是否 slot 连续
if (threadIdx.x < TILE_T) s_slot[threadIdx.x] = slot_mapping[tok0 + threadIdx.x];
__syncthreads();
if (threadIdx.x < TILE_T/8) {                    // 每 chunk 预计算一次，供所有 d 复用
    int c = threadIdx.x; int64_t s0 = s_slot[c*8];
    s_blk[c] = s0 / block_size; s_off[c] = s0 % block_size;
    s_ctg[c] = is_8_contiguous_aligned(&s_slot[c*8]);
}
__syncthreads();
```
> **规律**：两遍式/多次复用的**索引计算**（不只是数据）也应"读一次、算一次、存 SMEM 复用"——呼应招式 6 之外的"消除重复全局读"。本例 head_size=128、每行 16 个 d-vec 都复用同一份 chunk 元数据，省下 15/16 的 slot 全局读与整数运算。

**④ 为什么"提升率随 head 数暴涨"（H=4 达 5.26× vs H=1 的 2.10×）**——这是转置算子最反直觉的一点：

| shape | 原始 GB/s | iter6 GB/s | 提升 |
|---|---|---|---|
| H=1 T=10240 | 209.8 | 441.0 | **+110% (2.10×)** |
| H=2 T=10240 | 186.8 | 557.6 | **+198% (2.99×)** |
| H=4 T=10240 | 123.0 | 647.1 | **+426% (5.26×)** |

> 原始版**每个 head 独立散写** 2B 到 value_cache，head 越多、并发散写事务冲突越重，带宽**不升反降**（H=1=210 → H=4=123）。iter6 用 SMEM 合并写后，head 是天然的并行维度，**head 越多 grid 越满、带宽越高**（H=1=441 → H=4=647）。二者趋势相反，是提升率悬殊的根因。**教训：报告提升率必须逐 shape 给，单一"峰值倍数"会掩盖这种趋势反转。**

### 为何止步 ~647（未达 1300 单 die 目标）——转置的物理代价

- 本算子因 value 的 **d↔slot 转置不可避免**，必须过一遍 SMEM（读→barrier→写），**天花板天生低于无需转置的纯拷贝**。对照案例 E/G 的 store_kv（目标最内层与源一致、无转置、无 SMEM）能到 **940 GB/s**；本例结构上多了一次 SMEM 往返 + `__syncthreads`，647 已是该"转置+SMEM"设计的稳健平台。
- 结构 sweep 全部收敛：TILE_T{32,64,128}、线程{256,384,512,1024}、grid.z 拆 head、d-major SMEM 布局——**无一超过 647**，符合 §2.2"到平台即停，别为最后几个百分点死磕"。
- decode（T=16）总流量极小，**启动/延迟受限**（~6.5µs launch），两版都在个位数 GB/s，不计入达标（坑 5）。
- 精度：原始与优化版**全 shape `cos_sim = 1.000000`**（纯拷贝无近似）。

### 核心可复用经验

1. **先看源与目标的"最内层维度"是否一致**：不一致 = 隐藏转置（本例 value 源沿 d、目标沿 slot）。转置写会退化成大跨步散写、sector 效率个位数百分比，是这类 cache 写入算子的头号瓶颈——**向量化救不了转置，必须 SMEM 片上转置**让读写各自沿连续维合并。
2. **同一 kernel 内不同子张量分流**：需要转置的（value）走 SMEM，不需要的（key，x=8 让 d 连续）走直写——别为省事全塞进 SMEM，多余的 barrier 是纯损失。
3. **索引/映射也要"读一次算一次存 SMEM"**：slot_mapping 及其派生量（块号、偏移、连续性）预计算一次供所有 d 复用，是 iter5→iter6 的 +22%。
4. **转置算子的天花板低于纯拷贝**：有 SMEM 往返 + barrier 时，别拿无转置 copy 的墙（如 store_kv 的 940）当目标；用结构 sweep 找到平台（647）即停。
5. **逐 shape 报告提升率**：转置算子里 head/batch 等并行维对新旧版本可能是**相反趋势**（原始随 head 变慢、优化随 head 变快），只报峰值倍数会误导——本例 2.10×~5.26× 的跨度必须逐 shape 呈现。


## 案例 I：topk_transform_prefill（radix-select top-k，346→489 GB/s，+41%）

**算子**：`topk_transform_prefill_kernel`（`op/sglang/csrc/elementwise/topk.cu`）——对每一行 fp32 分数（`seq_len` 上万）做 **radix-select top-k（k=2048）**，输出 top-k 的索引经 page-table gather 写回。属访存受限，但**不是**前面案例那种"读+归约+写"的两遍式 elementwise，而是**多遍式选择**算子。有效字节 = `bs*(2*seq_len + k)*4`。测试配置：fp32，bs=[132,256,1662,4096]，seq_len=[66551,107520]，k=2048。

**原始结构（两次全行读）**：① 第一遍全行读，建 10-bit 粗直方图（1024 bins，shared atomicAdd）→ 后缀 cumsum 求精确阈值 bin；② 第二遍全行读，把阈值 bin 的候选暂存进 shared；③ refine 若干轮（8-bit radix）选出精确 top-k；④ gather 写回。

### 瓶颈定位：真正的成本是"两次全行读"本身

用 `#ifdef` 提前返回探针做阶段拆分（clean GPU，bs=4096/seq=107520，全程 9.34ms）：Stage1（1 次全行读 + 直方图 + cumsum）2.17ms @ **1637 GB/s 已到墙**；+Stage2 暂存（第 2 次全行读）累计 5.49ms；gather 仅 ~0.44ms。**两次全行读 ~4.4ms 就是访存地板**。先排除了几个错误方向（呼应 §3/§4）：

- **warp 聚合原子**（ballot+popc）对热计数器无效、对 107K 元素公共路径**回退到 252 GB/s**——原子争用不是瓶颈（热原子只在少量阈值候选上触发）。
- **refine 轮数几乎不影响**（4 轮 vs 1 轮差 <3%）。
- 结论：瓶颈是**读取遍数**，必须从算法层砍掉一整遍全行读（呼应 §2.2"effective≈HW 计数器时，从算法层减少真实流量"）。

### 关键优化举例

**① 单遍化：用"廉价采样"估保守阈值，取代第一遍全行读（本例 +41% 的核心杠杆）**。第一遍全行读只是为了定阈值，而**阈值不需要精确**——只采样 `1/STRIDE` 的行就能给出足够好的保守下界。把两次全行读压成 **~1.06 次**（1/16 采样 + 1 次暂存读）：

```cpp
// ① 采样建粗直方图（只读 1/SAMPLE_STRIDE 的行）
// ② 保守选候选下界 cand_lo：缩放后后缀计数 >= 1.5*k 的最大 bin
run_cumsum();                                  // s_histogram[b] = count(bin >= b)
constexpr int TARGET_CAND = 3072;              // 1.5*TopK，且 < 缓冲容量 4096
if (tx < RADIX) {
  const long cnt_ge      = (long)s_histogram[tx]     * SAMPLE_STRIDE;  // 采样计数×步长≈全量
  const long cnt_ge_next = (long)s_histogram[tx + 1] * SAMPLE_STRIDE;
  if (cnt_ge >= TARGET_CAND && cnt_ge_next < TARGET_CAND) s_threshold_bin_id = tx;
}
// ③ 仅一次全行读，暂存所有 bin>=cand_lo 的候选（~1.5*k 个）到 shared，并缓存 32-bit key
```
> **规律**：多遍式选择算子里，"为定统计量而做的整遍扫描"往往可以用**采样估计 + 保守放宽**替换。只要候选集合**能包住真正的 top-k 且装得下缓冲区**，最终结果仍是**精确的**——精度由后续 refine 保证，采样只影响候选集合大小、不影响正确性。这是把 memory-bound kernel 从"调访存"升级到"减访存流量"的算法级杠杆。

**② 采样必须"窗口化合并"，不能 strided、不能 prefix（决定成败的一步）**。这是本例最反直觉的坑：

```cpp
// ❌ strided：vec_idx = tx*STRIDE —— 不合并，仍拉取整条 cache line → 零流量节省（打平原始版）
// ❌ prefix ：只读行首 1/STRIDE —— 合并但有偏（只看行首 1/4）→ cos_sim 0.9993 失败
// ✅ windowed：连续窗口散布全行 —— 既合并又无偏
for (int base = 0; base < vec4_length; base += BLOCK_SIZE * SAMPLE_STRIDE) {
  const int vec_idx = base + tx;               // 每个窗口内 64 线程读连续 float4
  if (vec_idx < vec4_length) { /* float4 向量读 + 拆 4 元素 atomicAdd 进直方图 */ }
}
```
> **规律**：采样省流量的前提是**采样本身要合并访问**（否则 GPU 仍按 cache line 粒度拉满，等于没省）；同时采样必须**在整行上无偏**（否则粗直方图偏斜、阈值估歪，精度崩）。"连续窗口 + 窗口间大跨步"同时满足这两点，是采样类优化的通用正确写法。

**③ 保守候选范围要配足够的 refine 精度（精度门的硬约束）**。候选范围放宽到 1.5*k 后边界更"糊"，单轮 8-bit refine 分辨率不够：

```cpp
// REFINE_ROUNDS=1（8-bit 边界）：候选范围宽 → cos_sim 0.99934 失败
// REFINE_ROUNDS=2（8-bit×2=16-bit 边界）→ cos_sim >= 0.99991 通过
constexpr int REFINE_ROUNDS = 2;
```
> **规律**：**采样越激进（候选范围越宽），refine 就要越精细**来补回边界精度——二者是一对需要联调的旋钮，不能只调一个。

**④ 2 轮 refine 需要"专用第 3 块 shared key 缓存"**。原本 1 轮时第二个 ping-pong 缓冲空闲、可拿来缓存候选的 32-bit key（refine 时从 shared 读 key 而非散读 `row_input[idx]` 重载）；改 2 轮后 ping-pong 复用了该缓冲，key-cache 失效 → 掉到 414 GB/s。加一块专用缓存补回：

```cpp
constexpr size_t kSmem = 3 * 4096 * sizeof(uint32_t);   // 48KB：idx[0], idx[1], 专用 key 缓存
int* const s_key = s_input_idx[2];                       // refine round0 从 s_key 读，避免散读重载
// round0 用缓存 key；round1 才回源重载（避免同一循环内 s_key 读写竞争）
const auto key32 = (round == 0) ? (uint32_t)s_key[i] : convert_to_uint32(row_input[idx]);
```

**⑤ 采样步长 sweep：越稀越快，但有正确性硬边界（stride=32 直接崩）**：

| SAMPLE_STRIDE | 峰值 GB/s | cos_sim | 说明 |
|---|---|---|---|
| 4 | 442.9 | 0.99994 | 采样 1/4 |
| 8 | 468.9 | 0.99994 | 采样 1/8 |
| **16（最优）** | **488.8** | 0.99993 | 采样 1/16，+41.0% |
| ~~32~~ | **崩溃** | — | **非法内存访问**：采样过稀 → 候选估计不足/破坏正确性保证 |

> **规律**：采样步长是"省流量"与"估计可靠性"的权衡，存在**硬上界**——过稀会让保守估计失效（候选溢出缓冲或漏掉真值），表现为崩溃或精度失败。**逼近边界前一档（这里 16）即停**，别追最后一档。

### 逐 shape 提升率（fp32, k=2048，clean GPU）

| bs | seq_len | baseline GB/s | best(v14g) GB/s | 提升 |
|---|---|---|---|---|
| 1662 | 107520 | 346.8 | **488.8** | **+41.0%（峰值）** |
| 4096 | 107520 | ~342 | 481.4 | +40.8% |
| 256 | 107520 | ~326 | 456.6 | +40% |

（精度 cos_sim ≥ 0.99993 全 shape，>0.9999 门槛；baseline 为标量精确版 cos_sim=1.0。中间版 v8 两遍式 packed-half2+key-cache 仅 +11.7%，印证"减遍数"远胜"调单遍"。）

### 核心可复用经验

1. **memory-bound 的最大杠杆是"减少读取遍数"，不是"调单遍访存"**：多遍式选择/统计算子里，"为定阈值/统计量而做的整遍全行读"常可用**采样估计 + 保守放宽**替换，把 N 遍压成 ~1 遍。本例 2 遍→1.06 遍是 +41% 的主因，而两遍式内部微调（packed key + cache）只值 +11.7%。
2. **采样优化的两条铁律**：① 采样本身**必须合并访问**（strided 采样零收益——GPU 仍按 cache line 拉满）；② 采样**必须无偏**（prefix 采样精度崩——用"连续窗口散布全行"同时满足合并+无偏）。
3. **保守放宽 + 精确 refine = 既省流量又保精度**：只要候选集合能"包住"真值且装得下缓冲，结果仍精确；采样越激进，refine 精度就要越高来补边界（一对联调旋钮）。
4. **采样步长有正确性硬上界**：过稀会让保守估计失效（崩溃/精度失败），sweep 到边界前一档即停。
5. **先用探针排除伪瓶颈**：本例先证明"原子争用不是瓶颈"（warp 聚合反而 -60%）、"refine 轮数不重要"，才锁定"读取遍数"这个真瓶颈——呼应 §2.3 的隔离实验方法论。


## 案例 J：reshape_and_cache 量化扩展（bf16 KV → fp8 / int8，破解"计算受限")

**算子**：在案例 H 的 `reshape_and_cache`（`op/vllm/cache_kernels.cu`）之上，新增 `cache_dtype="fp8"` / `"int8"` 两条**量化写入**路径——读 bf16 K/V，per-tensor scale 量化成 fp8 (e4m3) / int8，写入 paged KV cache。`"auto"`（bf16 纯拷贝）路径保持字节不变。测试配置：bf16→{fp8,int8}，(q,kv)=(80,8) 等 7 组，head_dim=128，block_size=512，prefill T=10240。**精度全 shape `cos_sim=1.000000`**（三种 dtype 均是）。

> ⚠️ **本例与案例 H 同一个 kernel、同一个转置瓶颈**（value 的 d↔slot 转置写、走 SMEM），但**新增的量化把算子从"访存受限"推向了"计算受限"**——这是本例的核心，与前面所有纯访存案例不同。

### 关键发现：量化路径是"计算受限"，不是"访存受限"

value_cache 内层是 block_size，写是 d↔slot 转置（案例 H），**bf16 恒等路径（零转换）本身就只有 ~700 GB/s**（转置天花板，不是 1.3T 纯拷贝墙——**1.3T 在本算子物理上不可达**）。量化路径读写字节更少（读 2B、写 1B/fp8-int8），若纯访存受限**理应更快**，但初版实测反而更慢。

**判据（最关键的一步）**：fp8 与 int8 **写出字节完全相同（都 1B/元素、3B 总流量）**，但优化到某一步时 **fp8=289 GB/s 明显慢于 int8=417 GB/s**。相同访存、吞吐却差这么多 → 瓶颈**不在访存，在逐元素的量化转换指令**。fp8 的软件 narrow（`c10::Float8_e4m3fn`）比 int8 的 `float_to_int8_rn` 贵得多。

> **规律**：**同流量下不同 dtype 吞吐差异悬殊 = 计算受限信号**。把带宽 ÷ 每元素字节，换算成"**每元素速率**"（Gelem/s）再和 bf16 恒等路径的天花板比，才能看出到底卡在访存还是转换。本例 auto 转置天花板 ≈175 Gelem/s；fp8 160（**92%**）、int8 139（**79%**）——fp8 追平了转置天花板，int8 因转换指令没有硬件加速而停在 79%。

### 三个优化（全部用 `if constexpr`/`sizeof(cache_t)` 隔离，bf16 路径字节不变）

**① 倒数乘替代逐元素除法（fp8 210→277，int8 265→387）**。per-tensor scale 是标量常量，`x/scale` 在 head_size×每 token×每 head 的热循环里反复做 fp32 除法。**每次 launch 只算一次 `1/scale`**，热循环改成乘法：

```cpp
// kernel 序言：整个 launch 只算一次倒数（auto 路径不需要，置 0 省寄存器）
float k_scale_val = (kv_dt == kAuto) ? 0.f : (1.0f / *k_scale);
// ... CopyWithScaleOp 里热路径：
//   ❌ 原：q = x / scale;            —— 每元素一次 fp32 除法（慢）
//   ✅ 新：q = x * inv_scale;        —— 每元素一次乘法
```
> **规律**：热循环里除以"每 launch 不变的标量"，一律提到循环外取倒数、循环内乘。除法延迟远高于乘法，这是免费收益（呼应招式 6 的 `__builtin_mxc_rcpf`）。

**② 量化 store 加宽 uint2(8B) → uint4(16B)（fp8 277→289，int8 387→417）**。量化输出 1B/元素，8B 的 uint2 store 每线程只搬 8 个 token，未达 128-bit 满向量。用 `VST = 16/sizeof(cache_t)` 让 store 宽度随 dtype 自适应：bf16 打包 8 元素、量化打包 **16 元素**（一个 uint4）：

```cpp
constexpr int VST = 16 / sizeof(cache_t);   // bf16→8, fp8/int8→16（一个 uint4）
// value 量化写：16 个连续 token 的同一个 d 拼成一个 uint4 store
cache_t tmp[VST];
#pragma unroll
for (int i = 0; i < VST; ++i) tmp[i] = v_op(/* 第 i 个 token 的 value */);
*reinterpret_cast<uint4*>(&value_cache[...]) = *reinterpret_cast<const uint4*>(tmp);
```
> **约束**：uint4 store 要求 `slot0 % VST == 0` **且** `block_size % VST == 0` 才对齐（否则退回窄 store）。block_size=512、VST=16 天然满足。加宽访存粒度是招式 4 的直接应用，但**注意它只从 289→289 小涨**——因为此时已被 fp8 转换指令挡住，印证了"计算受限"。

**③ 硬件 fp8 pack 指令替软件 narrow（fp8 289→479，+66%，本例决定性一步）**。fp8 的瓶颈是逐元素软件 narrow。C600 有一条**一次把 4 个 float32 转成 4 个 e4m3 字节**的硬件指令 `__builtin_mxc_cvt_pk4_f32tof8`（RNE + 饱和）。封装成 helper（手动 clamp ±448 后调用），替掉软件 narrow：

```cpp
// 4 个 float32 → 4 个 e4m3 字节，一条硬件指令（替 4 次 c10::Float8_e4m3fn 软件 narrow）
__device__ __forceinline__ uint32_t rc_pack4_fp8_e4m3(float a, float b, float c, float d) {
  using v4f32 = float __attribute__((ext_vector_type(4)));
  v4f32 v = {a, b, c, d};
#pragma unroll
  for (int k = 0; k < 4; k++) v[k] = fminf(fmaxf(v[k], -448.f), 448.f);  // 手动饱和到 e4m3 范围
  return __builtin_mxc_cvt_pk4_f32tof8(v);
}
// value 写：4 次 pack 填满一个 uint4（16 个 fp8）；key 写：2 次 pack 填一个 uint2（8 个 fp8）
```
> **正确性验证（务必做，别直接上）**：C600 是 `__MACA_ARCH__==1600`（属用 `_cfg` 变体的区间），但实测**"手动 clamp ±448 + 朴素 builtin"** 与厂商 MACA 已验证的 `__cvt_rn_satfinite_e4m3x2_f32`（torch `.to(float8_e4m3fn)` 实际下沉到的实现）**逐比特一致**（8192 个含 0/次正规/±500 的值，diff=0），故无需 `_cfg` 参数。部署后单测 `cos_sim=1.0` vs torch reference 再次确认。
>
> **规律**：**发现某 dtype 是计算受限时，先查厂商有没有对应的"打包转换"硬件内建**（`__builtin_mxc_cvt_pk*`）。软件逐元素 narrow → 一条 SIMD pack 指令，是把"计算受限"打回"访存受限"的杀手锏。但硬件转换指令**必须先验证与参考实现逐比特一致**再用（尤其 fp8 的舍入/饱和语义）。

### 为何 int8 没有同款杠杆——停在 79%（诚实的天花板）

C600 **没有有符号 int8 的打包转换指令**：只有 `__builtin_mxc_cvt_pk_f32tou8`（**无符号**，且是 chained-insert 签名，不适配）。int8 路径已用厂商范本技法（`float_to_int8_rn` = `__float2int_rn` + clamp，与 `int8_quant_kernels.cu` 一致），**417 GB/s（转置天花板的 79%）就是其实际上限**。

> ⚠️ `CopyWithScaleOp` 里的 int8 clamp[-127,127] **必须保留**——该 op 与 flash kernel 共享，须对任意（非 absmax）scale 保持正确（`x*inv_scale` 可能越 127）。别为提速删 clamp。

### dispatch 注意：int8 用独立 host 分支，不进共享 DISPATCH 宏

int8 若塞进共享的 `DISPATCH_BY_KV_CACHE_DTYPE` 宏，会**强制其它 kernel 也实例化 int8**（编译膨胀 + 潜在不兼容）。用一条独立 host 分支拦截：

```cpp
if (kv_cache_dtype == "int8") {
  TORCH_CHECK(/* src 必须 bf16 */);
  CALL_RESHAPE_AND_CACHE(__nv_bfloat16, int8_t, kInt8);
  return;                                   // 不落入共享 DISPATCH 宏
}
// fp8/auto 走原有 DISPATCH_BY_KV_CACHE_DTYPE(...)
```
> `fp8::scaled_convert` 在本后端是死代码（primary template `assert(false)`）——量化内联在 `CopyWithScaleOp` 里做，别调它。

### 逐 dtype 峰值（prefill T=10240，(q,kv)=(80,8)，idle GPU）

| cache_dtype | 写字节/元素 | 峰值 GB/s | 每元素速率 Gelem/s | 占转置天花板(≈175) | 优化历程 |
|---|---|---|---|---|---|
| auto (bf16 拷贝) | 2B | ~700 | 175 | 100%（转置天花板本身） | 见案例 H |
| **fp8 (e4m3)** | 1B | **479** | **160** | **92%** | 210→277→289→**479** |
| int8 | 1B | 417 | 139 | 79% | 265→387→**417** |

> **注意"GB/s 更低不代表更慢"**：fp8/int8 写字节只有 bf16 一半，用 GB/s 直接比会误判。换成"每元素速率"才看得清——fp8 已达转置天花板的 92%，是三者里**每元素最快**的。**优化后 fp8 反超 int8**（相同 3B 流量下 479 vs 417），正是硬件 pack 指令的功劳。

### 核心可复用经验

1. **同流量下不同 dtype 吞吐差异悬殊 = 计算受限信号**：fp8 与 int8 写字节相同却差 128 GB/s，直接暴露"瓶颈在转换指令不在访存"。**把带宽换算成"每元素速率"再和恒等路径天花板比**，是区分访存受限 / 计算受限的通用判据。
2. **计算受限时先找厂商"打包转换"硬件内建**：`__builtin_mxc_cvt_pk4_f32tof8` 一条指令做 4 路 fp8 narrow，替软件逐元素 narrow → fp8 +66%，把算子打回访存受限。**但硬件转换指令必须先与参考实现逐比特验证**（clamp/舍入/饱和语义），再靠单测 cos_sim=1.0 兜底。
3. **热循环除以"每 launch 不变的标量"，一律提循环外取倒数、循环内乘**（fp8/int8 各 +30% 量级的第一步）。
4. **量化 store 用 `VST=16/sizeof(cache_t)` 随 dtype 自适应加宽**：bf16 打包 8、量化打包 16（一个 uint4），配 `slot0/block_size % VST==0` 对齐检查。
5. **没有硬件杠杆的路径要诚实收口**：int8 无有符号打包指令，79% 就是其实际天花板；别硬追、别删共享 op 的 clamp 去换速度。
6. **新增 dtype 用独立 host 分支拦截，避免污染共享 DISPATCH 宏**（防止其它 kernel 被迫实例化新 dtype）。
7. **多 dtype 一套代码用 `if constexpr`/`sizeof(cache_t)` 隔离**：bf16 恒等路径必须字节不变，量化路径的所有改动都不能碰到它。

---

## 案例 K：scale_dynamic_quant（per-token 动态量化 + per-channel smooth，fp8 974→1092 GB/s；按 dtype 分支调度）

**算子语义**：per-token 对称动态量化，叠加 per-channel smooth。对每个 token `t`、通道 `c`：
`v = hidden[t,c] * smooth[c]`；`absmax = max_c|v|`；`scale[t] = absmax/QMAX`；
`out[t,c] = round/convert(v * QMAX/absmax)` 饱和。QMAX=127(int8)/448(fp8-e4m3)。
输入 bf16、smooth 是 **fp32**、输出 int8/fp8。

**Roofline**：连续流式 **2R1W**（读 2B bf16 + 写 1B out + 每 token 4B scale），且 smooth
向量 `H*4B` = bf16 hidden 读的 **2 倍**，是最大的**次级流量**。单 die triad 墙 ~1294 GB/s。
—— 注意与案例 H/J 的 reshape_and_cache 不同：**本算子是连续拷贝（非转置/散读）**，天花板是
triad 墙本身，有真实上冲空间，不是转置天花板（~175 Gelem/s）那种结构性受限。

### ① 多 token 寄存器常驻，摊薄 fp32 smooth 的 L2 重读（核心结构）

每个 block 固定负责一组列，把这组列的 smooth 切片**一次性读进寄存器**，然后**跨 M 个 token 行复用**
（`vreg` 每行重算，smooth 常驻不动）。smooth 占 bf16 hidden 读的 2 倍，跨 M 行复用即把这块次级
流量摊薄到 1/M。这是本算子 GB/s 的**头号杠杆**（比去 barrier 更值钱，见③）。

```cpp
// smooth 一次入寄存器，跨 M 行复用（smv 只读一次，M 行不重读）
float smv[NR][VPT];
#pragma unroll
for (int r = 0; r < NR; ++r) { /* 从 smooth_scales 读 VPT 个 fp32 进 smv[r] */ }
#pragma unroll
for (int m = 0; m < M; ++m) {          // 同一 block 连做 M 个 token 行
    float vreg[NR][VPT]; float local_max = 0;
    /* 读第 (t0+m) 行 hidden，乘 smv（复用！），求 local_max */
    float absmax = /* 16-lane shuffle 块归约（招式 1/2） */;
    /* 用 absmax 量化写出该行 */
}
```

### ② 按 dtype 分支的 M 调度（第 13 类"按类型分支不同逻辑"的实证）

M 的最优值**取决于输出 dtype**，因为量化 store 的成本天差地别：

| dtype | store 成本 | 大 T 最优 M | 峰值 GB/s | 原因 |
|---|---|---|---|---|
| **int8** | 逐元素标量 round+clamp（**无硬件有符号 pack**） | **M=4** | 945 | store 重 → smooth 摊薄收益大，值得半 grid |
| **fp8** | 4-wide 硬件 pack `cvt_pk4_f32tof8`（招式 8） | **M=2** | **1092** | store 便宜 → 瓶颈转向 grid 占用，M=4 腰斩 grid 饿死 28 AP |

fp8 从 M=4 改 M=2：**974→1092 GB/s（+12%）**。分支用 `constexpr bool is_i8 = std::is_same_v<T2,int8_t>`：

```cpp
constexpr bool is_i8 = std::is_same_v<T2, int8_t>;
int M = 1;
if (is_i8) { if (token_num>=8192) M=4; else if (token_num>=1024) M=2; }   // 标量 store：敢 ramp 到 4
else       { if (token_num>=1024) M=2; }                                  // HW-pack store：封顶 2
```
> **小 T 两者都 M=1**：grid 本就填满 AP，占用 > smooth 复用。M 调度只在大 T 分叉。
> **规律**：**store 便宜的 dtype（有 HW pack）瓶颈在 grid 占用，store 贵的 dtype 瓶颈在次级流量**——
> 同一 kernel 对不同 dtype 要给不同的 tiling，别用一套 M 通吃。

### ③ warp-per-token 在这里**失败**了（与案例 moe_scatter 正相反的教训）

试过把 moe_scatter 的赢家搬来：**一个 64-lane warp 独占一整行**，absmax 纯 warp 内 shuffle
（`warpReduceMax64`），**零 `__syncthreads`**。结果**反而更慢**：int8 945→840，fp8 974→943。

原因：warp-per-token 让每行独立，就**放弃了跨 token 的 smooth 寄存器复用**（①）——每行都要重读
fp32 smooth。**丢掉这块次级流量摊薄的代价 > 去掉 barrier 省下的**。

> **可复用判据**：**barrier 去除只在"barrier 受限"时才是净胜**。本算子是**次级流量受限**（fp32 smooth
> 重读），不是 barrier 受限，所以 warp-per-token（用独立性换掉 barrier、但也换掉了复用）净亏。
> 上一个新结构前，先用招式 §2.3 的探针确认瓶颈到底是不是 barrier——是 gather/散读（moe_scatter）
> 才轮到 warp-per-row 发威，连续 + 有可复用常驻数据的算子别盲搬。

### ④ 测量完整性坑：非法 tiling 会伪造出"超墙"数字

调 M 时试了 `M=3`：读出 **2000+ GB/s**（远超 1294 单 die 墙）——**假的**。因为 dispatch 只实例化了
`M∈{1,2,4}` 的模板，`M=3` 落入 `else`→跑 M=1 的 kernel 体，但 grid 却按 M=3 划分，**2/3 的 token
根本没被写**，于是"用 1/3 的时间做了 1/3 的活"被算成 3× 带宽。

> **规律**：**任何"超过物理墙"的读数先当 bug**，八成是漏写/少算了流量（tiling 与模板不匹配、grid
> 越界 early-return、shape 没覆盖全）。只 emit 有对应模板的 tiling 值，并对非法值 clamp/兜底；
> 汇报前用"读数 ≤ 微基准实测墙"做一次 sanity gate。

### 逐 dtype 峰值（全 38×2 shape ALL PASS；idle GPU10/11）

| dtype | 峰值 GB/s | shape | cos_sim | 占单 die 墙(~1294) | 历程 |
|---|---|---|---|---|---|
| int8 | 945 | T=65536,H=4096 | 0.99994 | ~73% | schedule 已优，维持 |
| **fp8** | **1092** | T=65536,H=1024 | 0.99965 | **~84%** | 974→**1092**（M=4→M=2） |

> int8 停在 73% 与案例 J 同因——**C600 无有符号 int8 打包指令**，标量 store 是其上限；fp8 靠
> 硬件 pack 达 84%，是三者里每元素最快。fp8 cos_sim 天花板 ~0.9995（e4m3 仅 3 位尾数），阈值
> 0.999 已是该类型的物理合理线，别拿 0.99999 要求 8-bit 浮点。

### 核心可复用经验

1. **连续流式量化算子的头号杠杆是"次级流量摊薄"，不是去 barrier**：per-channel 参数（此处 fp32
   smooth，占主读 2×）进寄存器跨 M 行复用，比任何同步优化都值钱。先看**次级流量**再看 barrier。
2. **同一 kernel 按输出 dtype 给不同 tiling**：有 HW-pack 的 dtype（fp8）store 便宜 → grid 占用受限 →
   小 M；无 HW-pack 的 dtype（int8）store 贵 → 次级流量受限 → 大 M。`if constexpr(is_same_v<...>)` 分叉。
3. **新结构（warp-per-row 等）搬运前先验证瓶颈匹配**：它在 gather/散读（barrier/散读受限）赢，在
   连续+可复用常驻数据（次级流量受限）输。瓶颈错配，去 barrier 也净亏。
4. **超墙读数一律先判 bug**：tiling 与模板不匹配导致漏写 token，会伪造 N× 带宽；只 emit 有模板的
   tiling 值，汇报前用微基准墙做 sanity gate。

---

## 案例 L：moe_softmax_topk（MoE 门控 top-8，192→277 GB/s，+44%；串行 argmax → 双调网络）

**算子**：MoE 门控。每行(token)读 E=128 个 expert 的 fp32 logit，做 softmax → 取 top-8 → 对 8 个权重重归一化，写 `[T,8]` 权重 + `[T,8]` 索引。目标 shape：E=128 k=8 T=65536。**精度全 shape `cos_sim=1.000000 idx=1.0000`**（逐位与 torch 一致，非近似）。

> ⚠️ 本例与前面所有案例最大的不同：它是 **latency-bound（延迟受限），不是 bandwidth-bound**。每行只读 128×4=512B，roofline AI≈1.2 « ridge 22.7，HBM 只用了约 13%。**瓶颈是 top-k 选择里跨 lane shuffle 的串行依赖链，不是显存带宽。** 所以本例的优化主线不是"打满带宽"，而是"缩短 shuffle 关键路径"。

### 先标定天花板：用 shuffle 数量当北极星，而不是 GB/s

延迟受限算子测"墙"的方式不同——不是测 triad 带宽，而是**用探针隔离出 shuffle 开销**。实测：

| 探针 | shuffle 数 | 带宽 |
|---|---|---|
| read-probe（只读 + 6 次 shuffle 求 max，不选 top-k） | 6 | **329 GB/s** |
| 完整 top-8（两种 32-shuffle kernel） | 32 | **171 GB/s** |

> **结论：带宽由"关键路径上的 shuffle 数×宽度"决定。** 把 shuffle 从 32 逼近 6，就能把 171 拉向 329。这是整个优化的目标函数——**先找到延迟受限算子的真正标尺，再动手**。

### 基线为什么慢：串行 8 轮 argmax-and-mask（32 长依赖链）

基线 `fusedSoftmaxTopk16Vpt8Tournament`：SUBW=16（16 lane/行，一 warp 跑 4 行，每 lane 持 8 expert）。核心是**串行取 8 次最大值**：

```cpp
for (int kk = 0; kk < MAX_K; ++kk) {            // 8 轮，严格串行
    const PackedTopK local  = maxPackedTopK(head0, head1);
    const PackedTopK winner = subgroupArgmax16(local);   // 每轮 4 步蝶形 = 4 shuffle
    if (lane == winner_lane) { v0.x = -FLT_MAX; head0 = argmax4(v0, base0); } // 划掉赢家、重算
}
```

两个致命点：**(1) 8 轮 × 4 shuffle = 32 次，且第 kk 轮必须等第 kk-1 轮划掉赢家、重算 head 才能开始 → 一条 32 长的串行依赖链，延迟全部累加无法重叠；(2) SUBW=16 时 `__shfl_xor` width=16 在 64 宽 warp 上会跨 8-lane 硬件边界，走慢路径。**这就是 171~192 的来源。

### 核心改动：把"串行选 8 次"换成"排序一次"——`fusedSoftmaxTopk8Bitonic`

用**双调排序网络(bitonic network)** 从根本上改变依赖结构。三个改动叠加：

**① SUBW 16→8，每 lane 持 16 expert，一 warp 跑 8 行**——让所有 shuffle 落在硬件原生 8-lane 快路径内（m=1,2,4 全 intra-8），且行并发翻倍、延迟互相掩盖：

```cpp
constexpr int SUBW = 8;
const int lane = threadIdx.x & (SUBW - 1);   // 0..7
const int sub  = (threadIdx.x >> 3) & 7;     // 一个 64-warp 跑 8 行
```

**② 先在寄存器内本地排序，把跨 lane 工作前移**（用充裕的算力换稀缺的 shuffle）：

```cpp
sort16_desc(a);   // Batcher 网络 63 次 compare-exchange，纯寄存器、零 shuffle
                  // 之后 a[0..7] 即本 lane 私有的降序 top-8
```

**③ 3 步双调归约合并全局 top-8——每步内 8 个 shuffle 相互独立**（关键路径从 32 降到 3）：

```cpp
__device__ void mergeKeepTop8_w8(PackedTopK a[16], int m) {
    PackedTopK b[8];
    for (int i = 0; i < 8; ++i) b[i].raw = SHFL_XOR_8(a[i].raw, m); // 8 个独立 shuffle
    PackedTopK c[8];
    for (int i = 0; i < 8; ++i) c[i] = maxPackedTopK(a[i], b[7 - i]); // 两降序序列拼双调
    HALVE(0,4);HALVE(1,5);HALVE(2,6);HALVE(3,7);   // 标准双调 halver
    HALVE(0,2);HALVE(1,3);HALVE(4,6);HALVE(5,7);
    HALVE(0,1);HALVE(2,3);HALVE(4,5);HALVE(6,7);
    for (int i = 0; i < 8; ++i) a[i] = c[i];
}
// 主体只调 3 次：
for (int m = 1; m < SUBW; m <<= 1) mergeKeepTop8_w8(a, m);   // m=1,2,4，之后每 lane 都持全局 top-8
```

> **为什么快**：总 shuffle 24 次（< 32），但真正的杀手锏是**依赖结构**——每步内 8 个 `SHFL_XOR_8` 互相独立可流水并发，**关键路径只有 log₂(8)=3 个 shuffle 延迟**，而不是基线的 32。从"32 长的串行链"变成"3 层、每层 8 路并行"。双调数学保证结果逐位精确（cos=1.0），非近似。

**辅助手法**（都在削关键路径上的额外开销）：

```cpp
union alignas(8) PackedTopK { struct{ float value; int32_t expert; } fields; int64_t raw; };
// value+index 打包进一个 int64 → 一次 shuffle 同时搬两字段，index 不用再发一轮 shuffle
const float row_max = a[0].fields.value;                       // softmax 单调 → rank-0 即行 max
const float my_e = (lane<MAX_K)? __builtin_expf(a[lane].fields.value - row_max):0.f; // 每 lane 只算 1 次 expf（8/行，非 64/行）
const float inv = __builtin_mxc_rcpf(sum);                     // 招式 6 硬件倒数
```

### gate 必须严格：双调网络只对 top-8 数学成立

`mergeKeepTop8` 只保 8 个候选——它的正确性**只对 k=8 成立**。dispatch 卡死三个条件，实测验证边界：

```cpp
const bool auto_w8 = (stk_mode == 0 && topk == 8 && sizeof(scalar_t) == sizeof(float));
if ((stk_mode == 12 || auto_w8) && num_experts == 128) { /* 走 w8bitonic */ }
```

- **k=16**：用 w8bitonic → cos=0.90 idx=0.0（**错**，第 9~16 名已被丢弃）→ 留在原路径。
- **k=4**：Tournament 更快（330 vs 264，k=4 时串行链本就只 4 轮，双调固定网络反而不划算）→ 留在原路径。
- **k=8**：唯一甜点，277 vs 192 = **1.44×**，cos=1.0。实测三路互不影响：k=8→277、k=4→330.8、k=16→112，全 cos=1.0。

### 核心可复用经验

1. **延迟受限算子先换标尺**：不是测 triad 带宽，而是用探针隔离出主导延迟（此处 shuffle 数：6-shuffle 探针 329 vs 32-shuffle 171）。**标尺错了，优化方向就错。** 呼应 §2.3 的"隔离实验"，但隔离对象从 barrier 换成 shuffle 链。
2. **串行选择 → 并行网络是延迟受限 top-k 的根本解**：把 O(k) 串行的 argmax-and-mask（32 长依赖链）换成"寄存器本地排序 + O(log SUBW) 双调归约"（关键路径 3）。**减少的不只是 shuffle 数量，更是依赖链长度**——每步内多个 shuffle 独立可并发才是提速主因。
3. **缩小 SUBW 让 shuffle 全落在 16-lane 快域内**（此处 16→8，全 intra-8），同时一 warp 多跑几行加倍延迟掩盖。与招式 1 同源（16-lane 快路径），但这里进一步用更窄的 8-lane 子域换取更短的归约。
4. **算力换 shuffle**：本算子算力只用 0.7%、极度充裕。把跨 lane 的工作（贵、在关键路径）尽量前移成寄存器内 compare-exchange（便宜、可乱序）。
5. **优化前先 sweep 现有变体，未必要写新 kernel**：本次 192→277 的净收益是 **6 行 dispatch 改动**——接上一个**早已编译进二进制、但 auto 从没命中的**更优 kernel（`MOE_STK_MODE` 环境变量后面藏着 10+ 个变体）。之前 microbench 得出的"TPR32→16 是增益"对生产其实是 **no-op**——因为 auto 路径**本来就是** 16-lane 的 Tournament，真正的增益来自一个**不同的算法**（双调）。**先摸清现有 dispatch 里已有什么，再决定写不写。**
6. **近似型 top-k 的 gate 要卡死数学成立域**：双调 top-8 网络对 k≠8 会静默算错（k=16 cos=0.90）。凡"只对特定 k/E/dtype 正确或更快"的快路径，dispatch 必须 `topk==8 && E==128 && fp32` 三条件齐备，其余 k 回退——用全 shape 单测（含 k=4/16）验证边界，别只测目标 shape。
