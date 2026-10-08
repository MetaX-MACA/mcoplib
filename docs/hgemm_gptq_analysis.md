# launch_gemm_gptq_kernel 深度分析报告

---

## 1. 功能解读

### 1.1 核心功能

`launch_gemm_gptq_kernel` 是一个专为 **MetaX C500 GPU** 优化的 **GPTQ 量化权重矩阵乘法** CUDA kernel。它计算：

```
C[m × n] = A[m × k] × B_dequant[k × n]
```

其中 B 是 GPTQ 量化后的权重矩阵（4-bit 或 8-bit），在计算过程中**在线反量化**（dequantize）后与激活矩阵 A 做矩阵乘法。

### 1.2 输入输出

| 参数 | 说明 |
|------|------|
| `A` | 激活矩阵 [m × k]，FP16 或 BF16 |
| `B` | 量化权重矩阵 [k × n]，4-bit packed（uint32_t 打包，每 32-bit 存 8 个 4-bit 值）或 8-bit packed |
| `C` | 输出矩阵 [m × n]，FP16/BF16 |
| `C_temp` | FP32 临时缓冲区（BF16 高精度模式下用于 FP32 累加后 reduce） |
| `scales` | 量化缩放因子 [(k/groupsize) × n]，FP16 |
| `zeros` | 量化零点 [(k/groupsize) × n/8]，4-bit packed（仅 kU4 类型） |
| `g_idx` | 分组索引 [k]，int32 |
| `quant_group` | 量化组大小，必须为 2^n (n≥5)，支持 32/64/128 等 |

### 1.3 支持的量化类型

| 类型 ID | 说明 | 反量化公式 |
|---------|------|-----------|
| `kU4` | 4-bit 量化，带显式零点 | `value = (quant_val - zero) × scale` |
| `kU4B8` | 4-bit 量化，零点固定为 8 | `value = (quant_val - 8) × scale` |
| `kU8` | 8-bit 量化，无零点 | `value = quant_val × scale` |
| `kU8B128` | 8-bit 量化，零点固定为 128 | `value = (quant_val - 128) × scale` |

### 1.4 解决的问题

1. **GPTQ 推理性能**：GPTQ 4-bit 量化是大模型推理的关键技术，将权重压缩到 4-bit 可节省 ~8x 显存和带宽。但反量化与矩阵乘法的融合是性能瓶颈——先反量化再乘法会导致额外显存开销和带宽浪费。该 kernel 将**反量化和矩阵乘法融合**在同一个 kernel 中。

2. **C500 硬件适配**：
   - 使用 C500 专有的 MMA 指令（`mma_16x16x16f16`/`mma_16x16x16bf16`）
   - 使用 C500 专有的向量加载指令（`ldg_b32/b64/b128_reg_noasync`）
   - 使用 C500 专有的 barrier 指令（`barrier_bsm`）
   - 使用 C500 专有的 bit-cast 指令（`CVT_B0TOF32` 等 `__builtin_mxc_*`）
   - 使用 C500 专有的 packed FMA（`__builtin_mxc_pk_fma_f32`）
   - 针对 C500 的 PEU（Processing Element Unit）数量（416 = 13×8×4）调度 block

3. **BF16 高精度模式**：C500 没有 BF16 atomicAdd 指令，因此对 BF16 输出采用 FP32 累加 + 后处理 reduce 的策略，避免精度损失。

### 1.5 应用场景

| 场景 | 说明 |
|------|------|
| GPTQ 4-bit 量化模型推理 | LLaMA、Qwen 等模型的 GPTQ INT4 推理 |
| DeepSeek-V3/R1 MoE 推理 | MoE 模型中大量的小 GEMM 计算 |
| GPTQ 8-bit 量化推理 | 精度要求更高的场景 |
| 融合 MoE GEMM | `hgemm_gptq_fused_moe` 变体，融合了路由和 GEMM |

---

## 2. 实现流程图

### 2.1 整体架构

该 kernel 采用 **Fused Dequant+GEMM** 架构，包含 4 个命名空间下的变体：

```
hgemm_marlin_gptq
  ├── __hgemm_singular_blocks_k    (BLOCKS_K 为奇数，使用 atomicAdd 写回)
  ├── __hgemm_even_blocks_k        (BLOCKS_K 为偶数，使用 atomicAdd 写回)
  ├── __hgemm_singular_opt1_blocks_k (BLOCKS_K 为奇数，无 atomicAdd 写回)
  └── __hgemm_even_opt1_blocks_k   (BLOCKS_K 为偶数，无 atomicAdd 写回)
```

`launch_gemm_gptq_kernel` 根据 `tiles_n * chunks >= PEUS` 决定使用 atomic 路径还是 no-atomic 路径。

### 2.2 核心 Kernel 执行流程（以 __hgemm_singular_blocks_k 为例）

```
┌─────────────────────────────────────────────────────────────────┐
│              hgemm_gptq Kernel 启动                               │
│  Grid: dim3(blocks, chunks, 1), Block: 256 threads               │
│  Shared Memory: 16KB (0x4000)                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 0: 初始化                                                  │
│  1. 计算 blockIdx.y 对应的 A/C 偏移（M 维度分块）                  │
│  2. 初始化 LoadingManager：设置地址、TileManager、ThreadView       │
│  3. 清零累加器 output[BLOCKS_M][N_ITERS][4] = 0                  │
│  4. 预加载：ldg_scales + ldg_zp + ldg_b(0) + ldg_a(0)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Main Loop: while (max_iters > 0)                                 │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ Phase 1: Tile 初始化                                       │   │
│  │   init_bsm_addr()  → 重置共享内存地址指针                    │   │
│  │   ldg_scales()     → 从全局内存加载 scales 到寄存器          │   │
│  │   ldg_zp()         → 从全局内存加载 zeros 到寄存器          │   │
│  │   ldg_b(0)         → 从全局内存加载第一块量化权重 B          │   │
│  │   ldg_a(0)         → 从全局内存加载第一块激活 A              │   │
│  │   sts_scales()     → scales 从寄存器写入共享内存             │   │
│  │   sts_zeros()      → zeros 解压后写入共享内存                │   │
│  │   barrier_bsm      → 同步等待共享内存写入完成                │   │
│  │   lds_scales()     → 从共享内存读取 scales 到寄存器          │   │
│  │   pack_scales()    → 打包为 v2f 格式 (scale, -zero*scale)   │   │
│  └───────────────────────────────────────────────────────────┘   │
│                           │                                       │
│                           ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ Phase 2: K 维度循环 (BLOCKS_K-1 次非尾迭代 + 1 次尾迭代)    │   │
│  │                                                             │   │
│  │   对每个 k_idx:                                              │   │
│  │   ┌─────────────────────────────────────────────────────┐ │   │
│  │   │ on_dequant<KTAIL>(k_idx):                           │ │   │
│  │   │   ① swap_b_cache: 交换 B 的 cache buffer           │ │   │
│  │   │   ② sts_a: 将 A 从寄存器写入共享内存               │ │   │
│  │   │   ③ dequant: 反量化 B（4-bit→FP16/BF16）           │ │   │
│  │   │   ④ next_k_pre + ldg_b + ldg_a: 预取下一块数据     │ │   │
│  │   │   ⑤ barrier_bsm: 等待 A 写入共享内存完成           │ │   │
│  │   │   ⑥ lds_a: 从共享内存读取 A 到寄存器               │ │   │
│  │   └─────────────────────────────────────────────────────┘ │   │
│  │                                                             │   │
│  │   M 维度内循环:                                              │   │
│  │   ┌─────────────────────────────────────────────────────┐ │   │
│  │   │ for m_idx = 0 to BLOCKS_M-1:                        │ │   │
│  │   │   lds_a(m_idx+1): 预取下一块 A（双缓冲）            │ │   │
│  │   │   matmul(m_idx):  执行 MMA 16×16×16                 │ │   │
│  │   │     → mma(local_a[0], dequant_b, output)            │ │   │
│  │   │     → mma(local_a[1], dequant_b+1, output)          │ │   │
│  │   └─────────────────────────────────────────────────────┘ │   │
│  │                                                             │   │
│  │   barrier_bsm → next_k → matmul(last_m)                    │   │
│  └───────────────────────────────────────────────────────────┘   │
│                           │                                       │
│                           ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ Phase 3: 写回结果                                            │   │
│  │   if (need_save_data):                                       │   │
│  │     write_c_pre(): 将 output 写回全局内存                    │   │
│  │       → BF16: atomicAdd(C_temp, fp32_val) + 后续 reduce    │   │
│  │       → FP16: atomicAdd(C, fp16_val)                        │   │
│  │       → no-atomic: 直接写入 C[offset]                       │   │
│  │     clear_c(): 清零累加器                                    │   │
│  │                                                             │   │
│  │   barrier_bsm                                                │   │
│  │   max_iters--                                                │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Post-Processing (仅 BF16 高精度模式)                             │
│  1. clean_zero: 清零 C_temp 缓冲区                                │
│  2. all_reduce: C_temp (FP32) → C (BF16) 求和                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 反量化流程（4-bit）

```
输入: packed_uint32 (8 个 4-bit 值打包)
      scale (FP16), zero (4-bit packed → FP32)

┌──────────────────────────────────────────────────────┐
│  Step 1: 提取低 4-bit                                │
│    p0 = packed & 0x0f0f0f0f                         │
│                                                      │
│  Step 2: 使用 C500 专用 bit-cast 指令转为 FP32       │
│    CVT_B0TOF32(p0, tmp[0])  // byte 0 的 4-bit → FP32│
│    CVT_B2TOF32(p0, tmp[1])  // byte 2 的 4-bit → FP32│
│                                                      │
│  Step 3: 使用 packed FMA 完成反量化                   │
│    result = val × scale + (-zero × scale)            │
│    即: __builtin_mxc_pk_fma_f32(val, scale, zero)    │
│                                                      │
│  Step 4: BF16 模式下转换为 BF16                      │
│    f32x2_cvt_bf16x2: 自定义 FP32→BF16 转换           │
│                                                      │
│  Step 5: 处理高 4-bit                                │
│    p0 = (packed >> 4) & 0x0f0f0f0f                  │
│    重复 Step 2~4                                     │
└──────────────────────────────────────────────────────┘

输出: 8 个 FP16/BF16 反量化值
```

### 2.4 共享内存布局

```
smem_base (16KB = 0x4000):
┌──────────────────────────────────────────┐
│ 0x0000 ~ 0x1FFF: 矩阵 A (8KB)           │
│   格式: [BLOCKS_M × SLICE_M][PAD_SLICE_K]│
│   PAD_SLICE_K = 40 (SLICE_K=32 + 8 pad)  │
│   双缓冲: 奇偶 K 段交替使用                │
├──────────────────────────────────────────┤
│ 0x2000 ~ 0x2FFF: Scales (4KB)            │
│   格式: half2[TILE_N]                    │
│   每个 half2 包含相邻两个 N 位置的 scale   │
├──────────────────────────────────────────┤
│ 0x3000 ~ 0x33FF: Zeros (1KB)             │
│   格式: float[TILE_N/8 × 8]             │
│   解压后的 -zero 值，以 FP32 存储         │
└──────────────────────────────────────────┘
```

---

## 3. 典型 Shape 举例

### 场景：LLaMA-2-70B GPTQ INT4, M=1, N=4096, K=8192, group_size=128

**问题维度**：
- `prob_m = 1`（decode 阶段，单 token）
- `prob_n = 4096`
- `prob_k = 8192`
- `quant_group = 128`
- 量化类型：kU4（4-bit，带显式零点）
- 标量类型：BF16

**编译期常量**（假设 BLOCKS_M=1, BLOCKS_N=4, BLOCKS_K=4）：
```
SLICE_M=16, SLICE_N=16, SLICE_K=32
TILE_M = 1×16 = 16
TILE_N = 4×16 = 64
TILE_K = 4×32 = 128
WAVE = 64, SLOT = 16, WAVES_PER_BLOCK = 256/64 = 4
N_ITERS = 64 / (4 × 16) = 1
PACK_RATIO_4BITS = 32/4 = 8
```

**Grid 配置计算**：
```
tiles_m = ceil(1/16) = 1
tiles_n = ceil(4096/64) = 64
tiles_k = ceil(8192/128) = 64
total_tiles = 64 × 64 = 4096

PEUS = 416 (C500)
iters = ceil(4096/416) = 10
blocks = 416

Grid: dim3(416, chunks, 1), 每个block处理 10 个 tile
```

### 单个 Block 执行细节（BLOCKS_K=4, N_ITERS=1, BF16, kU4）

#### 初始化

```
tid = 0..255
wave_idx = tid/64   → 0,1,2,3
wave_tid = tid%64   → 0..63
slot_idx = wave_tid/16  → 0,1,2,3
slot_tid = wave_tid%16  → 0..15

TileManager: 管理当前 block 负责的 tile 序列
  tile_start_col, tile_start_row → 对应 N 和 K 维度的 tile 索引
  my_iters = 10 → 该 block 需要处理 10 个 tile

LoadingManager:
  output[1][1][4] = {0}  // 1 个 M 块 × 1 个 N 迭代 × 4 个 FP32 累加值
  local_a[1][2]     // 1 个 M 块 × 2 个 PackTypeInt2（A 片段）
  local_b[1]        // 1 个 uint32_t（量化权重）
  local_b_cache[1]  // 双缓冲
  local_dequanted_b[1][8]  // 反量化后的 8 个 BF16 值
  local_scales[1]   // v2f {scale, scale}
  local_zeros[1]    // v2f {-zero*scale, -zero*scale}
```

#### Tile 循环迭代 1（tile_start_col=0, tile_start_row=0）

**Phase 1: 加载 Scales + Zeros + A + B**

```
1. init_bsm_addr()
   bsm_a_ptr = smem_base + slot_tid × (40/8) + slot_idx
   bsm_scales_ptr = smem_base + 0x2000 + (wave_idx×16 + slot_tid) × 1
   bsm_zeros_ptr = smem_base + 0x3000 + (wave_idx×16 + slot_tid) × 1

2. ldg_scales()
   加载 scales[0×4096 + 0..63] 到 temp_scales (half2)
   每个 tid 加载一个 half2（2 个 BF16 scale 值）

3. ldg_zp()
   加载 zeros[0×4096/8 + 0..7] 到 temp_zeros (uint32_t)
   每个 tid 加载一个 uint32_t（8 个 4-bit 零点打包）

4. ldg_b(0)
   加载 B 的第一段量化权重
   B 地址 = B + (0/8 × 4096 + 0) + slot_idx × 4096 + (wave_idx×16+slot_tid)×1
   → 加载 1 个 uint32_t 到 local_b_cache[0]

5. ldg_a(0)
   加载 A[0..0][0..31] 到 temp_a[]
   每 256 线程协同加载 16×32 个 BF16 值
   LOADING_A_LOOP = 32×16/2/256 = 1

6. sts_scales()
   将 temp_scales 写入 smem_base + 0x2000 + tid

7. sts_zeros()
   decompress_zero_4bits(temp_zeros, temp[8])
   → 使用 CVT_B0TOF32 等 C500 专用指令解压 4-bit 零点为 FP32
   → 取负值: temp[i] = -zero_val
   写入 smem_base + 0x3000 + tid × 8

8. barrier_bsm

9. lds_scales()
   从共享内存加载 scale 到 local_dequanted_b[0]

10. pack_scales()
    对 kU4 类型:
      s = __bfloat162float(local_dequanted_b[0][0])  // scale 转 FP32
      z = *(bsm_zeros_ptr + 0)                        // -zero × scale
      z = z × s
      local_scales[0] = {s, s}
      local_zeros[0] = {z, z}
```

**Phase 2: K 维度循环 (BLOCKS_K=4, 即 3 次非尾 + 1 次尾)**

```
k_idx=0 (非尾):
  on_dequant<false>(0):
    ① swap_b_cache(0): local_b[0] = local_b_cache[0]
    ② sts_a(): temp_a → smem_base (A 写入共享内存)
    ③ dequant(0):
       dequant_gptq_4bits(local_b[0], out[8], scales, zeros)
       → 提取 4-bit 值 → CVT_B0TOF32 转 FP32
       → pk_fma_f32: result = val × scale + (-zero×scale)
       → f32x2_cvt_bf16x2 转 BF16
       → 得到 8 个反量化 BF16 值
    ④ next_k_pre(): 更新 B_loading 和 A_loading 地址
    ⑤ ldg_b(1): 预取下一段 B
    ⑥ ldg_a(1): 预取下一段 A
    ⑦ barrier_bsm: 等待 A 写入完成
    ⑧ lds_a(0): 从共享内存读取 A 到 local_a[0]

  M 维度循环 (BLOCKS_M=1, 仅一次):
    matmul(0):
      mma_16x16x16bf16(local_a[0][0], dequant_b[0:1], output[0][0])
      mma_16x16x16bf16(local_a[0][1], dequant_b[2:3], output[0][0])
      → 每次 mma 计算 16×16×16 = 4096 个乘加
      → 2 次 mma 覆盖 K=32 维度

  barrier_bsm → next_k() → matmul(0)  // 最后一个 M 块

k_idx=1, k_idx=2: 同理

k_idx=3 (尾迭代):
  on_dequant<true>(3):
    不预取下一段（KTAIL=true）
    其余步骤相同

  next_tile_pre(): 准备下一个 tile
  matmul(0)
```

**Phase 3: 写回**

```
write_c_pre():
  store_m = slot_idx × 4 + 0 × 16 + miter2  (0..15)
  store_n = (wave_idx × 16 + slot_tid) × 1 + tile_col × 64

  BF16 高精度模式:
    atomicAdd(C_temp + store_m × 4096 + store_n, output[0][0][miter2])
    → 写入 FP32 临时缓冲区

clear_c():
  output[0][0][0..3] = 0
```

#### 后处理（BF16 高精度模式）

```
1. clean_zero<<<N, 512>>>(C_temp, m×n)
   → 清零 C_temp 缓冲区（在 kernel 执行前，用于累积多个 block 的部分和）

2. all_reduce<<<N, 512>>>(C_temp, C, m×n)
   → 将 C_temp 中的 FP32 值转换为 BF16 并写入 C
   → 如果 USE_C=true，还会与 C 中已有的 BF16 值相加
```

### 数据流总结

```
Global Memory                  Registers                     Shared Memory
─────────────                  ─────────                     ─────────────

B[quantized] ──ldg_b──→ local_b_cache ──swap──→ local_b
                                                    │
                                                dequant()
                                                    │
                                                    ▼
                                           local_dequanted_b[8]
                                                    │
scales ──ldg──→ temp_scales ──sts──→ smem ──lds──→ pack_scales() → local_scales
zeros  ──ldg──→ temp_zeros  ──sts──→ smem ──lds──→ pack_scales() → local_zeros

A ──ldg_a──→ temp_a ──sts_a──→ smem_a ──lds_a──→ local_a[2]

                                                    │
                                               matmul() (MMA)
                                                    │
                                                    ▼
                                           output[BLOCKS_M][N_ITERS][4]
                                                    │
                                               write_c_pre()
                                                    │
                                                    ▼
                                              C / C_temp (Global Memory)
```

---

## 4. 反复论证与验证

### 4.1 分块正确性验证

**TILE 大小**: M=16, N=64, K=128（以 BLOCKS_M=1, BLOCKS_N=4, BLOCKS_K=4 为例）

- 每个 block 处理输出矩阵 C 的一个 16×64 子块
- K 维度分 128 个元素，分 4 个 SLICE_K=32 的子块
- 每个 SLICE_K 内，256 线程协同工作：4 个 wave × 16 slot × 16 slot_tid
- 每个 slot_tid 负责加载 1 个 uint32_t（8 个 4-bit 值）的权重
- 4 个 slot_idx 对应 K 维度的 4 行 packed 权重（32/8=4）

**验证**:
- SLICE_K=32, PACK_RATIO=8 → 每 SLICE_K 有 32/8=4 行 packed B
- slot_idx=0..3 正好覆盖 4 行 ✓
- N_ITERS = TILE_N / (WAVES_PER_BLOCK × SLOT) = 64/(4×16) = 1
- 每个 slot_tid 处理 N 维度的 1 个连续位置 ✓

### 4.2 反量化正确性验证

GPTQ 4-bit 反量化公式：`value = (quant_val - zero) × scale`

代码中实现为：
```
dequant_gptq_4bits():
  result = val × scale + scale_zero
  其中 scale_zero = -zero × scale（预计算）
```

等价于：`result = val × scale - zero × scale = (val - zero) × scale` ✓

### 4.3 MMA 正确性验证

```
matmul(mdx):
  mma_16x16x16(local_a[mdx][0], dequant_b[0:1], output[mdx][i])
  mma_16x16x16(local_a[mdx][1], dequant_b[2:3], output[mdx][i])
```

- `local_a[mdx][0]` 和 `local_a[mdx][1]` 分别对应 K 维度的前 16 和后 16 元素
- 4-bit 反量化后，一个 uint32_t 解出 8 个值，两个 PackTypeInt2 组成 16×16 的 B 矩阵
- 两次 mma 覆盖 SLICE_K=32 的完整 K 维度 ✓

### 4.4 共享内存 Bank Conflict 分析

- A 矩阵在共享内存中按 PAD_SLICE_K=40 排列（而非 32），8 个元素的 padding 恰好避免 bank conflict
- 每个 slot_tid 读取连续的 128-bit（PackTypeInt4），对应 8 个 half，跨 8 个 bank
- 16 个 slot_tid 读取 16 个连续的 PackTypeInt4，覆盖 128 个 bank（C500 有 32 个 bank，4 轮迭代无冲突）✓

### 4.5 BF16 高精度路径验证

C500 没有 BF16 的 atomicAdd 指令。代码的处理策略：

1. **累加阶段**：在 FP32 临时缓冲区 `C_temp` 中用 `atomicAdd(C_temp + offset, v)` 累加（FP32 atomicAdd C500 支持）
2. **Reduce 阶段**：`all_reduce` kernel 将 `C_temp` 中的 FP32 值转换为 BF16 写入最终输出 `C`
3. 这比直接用 FP16 atomicAdd 精度更高（BF16 → FP16 会丢失指数范围）

### 4.6 Tile 调度正确性验证

```
TileManager::init():
  tile_idx = iters × bidx  // 每个 block 从第 iters×bidx 个 tile 开始
  tiles_n = ceil(n/TILE_N)
  tiles_k = ceil(k/TILE_K)
  tile_col = tile_idx / tiles_k  // N 维度 tile 索引
  tile_row = tile_idx % tiles_k  // K 维度 tile 索引
```

- 按 K 优先排序 tile：先沿 K 维度遍历，再沿 N 维度遍历
- 好处：沿 K 维度遍历时，scales/zeros 可以复用（同一 quant_group 内的 scale 相同）
- `need_save_data()` 在 K 维度最后一个 tile 或 my_iters==1 时触发写回 ✓

### 4.7 双缓冲（Double Buffering）验证

代码中的 `local_b` / `local_b_cache` 实现了 B 矩阵的双缓冲：

```
ldg_b(k_idx) → local_b_cache  // 异步加载到 cache
swap_b_cache(i) → local_b[i] = local_b_cache[i]  // 交换使用
dequant(local_b[i])  // 反量化当前 buffer
```

在 `on_dequant` 中：
1. 先 swap 当前 buffer 并开始反量化
2. 同时预取下一段 B 到 cache buffer
3. 反量化和加载重叠执行

对 A 矩阵也实现了双缓冲：`ldg_a → temp_a → sts_a → smem_a → lds_a → local_a`，当前段 A 在 smem 中被消费时，下一段 A 正在从全局内存加载。✓

### 4.8 性能关键点总结

| 维度 | 设计选择 | 原因 |
|------|---------|------|
| 分块大小 | 16×64×128 (M×N×K) | 适配 C500 16KB 共享内存限制 |
| 线程组织 | 4 wave × 4 slot × 16 slot_tid | 匹配 C500 PEU 结构 (13 AP × 4 PEU × 8 DPC) |
| 反量化 | 使用 `__builtin_mxc_pk_fma_f32` | C500 专用 packed FMA，2 个 FP32 FMA 同时执行 |
| Bit-cast | 使用 `CVT_B0TOF32` 等内置函数 | C500 专用指令，比通用移位+转换快 |
| Barrier | `barrier_bsm` 而非 `__syncthreads()` | C500 专用 barrier，更轻量 |
| 加载 | `ldg_b32/b64/b128_reg_noasync` | C500 专用带谓词的向量化加载 |
| BF16 转换 | `f32x2_cvt_bf16x2` 自定义 | 使用 `__builtin_mxc_ubfe` 和 `__builtin_mxc_byte_perm`，比标准转换更高效 |
| Block 调度 | tiles_n × chunks ≥ PEUS 时用 no-atomic | 避免 atomicAdd 开销；tiles 不足时用 atomic 确保正确性 |
