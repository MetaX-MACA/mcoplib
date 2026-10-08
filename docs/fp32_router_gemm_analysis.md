# fp32_router_gemm CUDA Kernel 深度分析报告

---

## 1. 功能解读：该 Kernel 实现了什么？

### 1.1 核心功能

`fp32_router_gemm` 是一个专为 **MoE（Mixture of Experts）路由层**设计的 FP32 精度矩阵乘法 CUDA kernel。它计算：

```
Output[M x E] = Activation[M x H] × Weight^T[E x H]   →  FP32 结果
```

其中：
- **M (kNumTokens)**：token 数量，范围 1~32（推理 decode 阶段的小 batch）
- **E (kNumExperts)**：专家数量，固定为 **256**
- **H (kHiddenDim)**：隐藏维度，固定为 **3072**
- **Activation**：支持 **FP32** 或 **BF16** 输入
- **Weight**：始终为 **FP32**
- **Output**：始终为 **FP32**

### 1.2 它解决了什么问题？

该 kernel 专为 **DeepSeek-V3** 等超大 MoE 模型的 **路由（Router）层**设计，解决的核心问题是：

1. **路由精度问题**：MoE 路由需要高精度的 gate logits 来正确选择专家。使用 FP16/BF16 的通用 GEMM（如 cuBLAS）在路由计算中可能产生精度损失，导致专家选择错误，影响模型质量。该 kernel 保证权重始终以 FP32 参与计算，输出也是 FP32。

2. **小 M 维度下的性能问题**：在 decode 阶段，M 通常只有 1~32 个 token。通用 GEMM 库（cuBLAS）针对大 M 优化，对小 M 场景（尤其是 M≤32, K=3072）启动开销和线程利用率极低。该 kernel 采用 **一个 block 计算一列（一个专家）** 的策略，256 个 block 完美映射到 256 个专家，规避了小 M 的 GEMM 效率问题。

3. **避免混合精度累加误差**：直接在 FP32 域内做乘加，避免 BF16 累加的精度损失，同时支持 BF16 输入（节省显存和带宽）。

### 1.3 应用场景

| 场景 | 说明 |
|------|------|
| **DeepSeek-V3 推理** | 256 专家 MoE 架构，Router 层 gate 计算 |
| **DeepSeek-R1 推理** | 同架构，MoE 路由 |
| **其他大规模 MoE 模型** | 具有类似 H=3072, E=256 配置的模型 |
| **Decode 阶段路由** | 小 batch（M≤32）的高精度路由计算 |
| **需要 FP32 精度路由的场景** | 任何对路由精度敏感的 MoE 推理 |

---

## 2. 实现流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    invokeFp32RouterGemm 调用                      │
│  输入: mat_a[M×H], mat_b[E×H], output[M×E]                      │
│  Grid: <<<E=256 blocks, 128 threads/block>>>                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: 编译期常量计算                                           │
│  VPT = 16/sizeof(InputT)  →  fp32:4, bf16:8                     │
│  k_elems_per_k_iteration = VPT × kBlockSize = VPT × 128         │
│  fp32: 4×128=512, bf16: 8×128=1024                               │
│  k_iterations = H / k_elems_per_k_iteration                      │
│  fp32: 3072/512=6, bf16: 3072/1024=3                             │
│  kNumWarps = 128/32 = 4                                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: 线程索引映射                                             │
│  n_idx = blockIdx.x  (当前 block 负责第 n_idx 个专家)             │
│  tid = threadIdx.x   (0~127)                                     │
│  warpId = tid/32     (0~3)                                       │
│  laneId = tid%32     (0~31)                                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: 初始化                                                   │
│  acc[M] = {0}  (每个线程维护 M 个 FP32 累加器)                    │
│  sm_reduction[M][4]  (shared memory, 用于 warp 间规约)            │
│  b_col = mat_b + n_idx × H  (指向当前专家的权重列)                 │
│  预计算 k_bases[ki] = ki × k_elems_per_k_iteration + tid × VPT   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: 主循环 — 沿 K 维度分块计算                                │
│  for ki = 0 to k_iterations-1:                                   │
│    ┌───────────────────────────────────────────────────────┐     │
│    │ 4a. 加载权重: load_weight<VPT>(b_col + k_base, b)     │     │
│    │     - 每线程从全局内存加载 VPT 个 FP32 权重值           │     │
│    │     - 使用 float4 向量化加载（16字节/次）               │     │
│    └───────────────────────────────────────────────────────┘     │
│    ┌───────────────────────────────────────────────────────┐     │
│    │ 4b. 对每个 token 循环:                                 │     │
│    │   for m_idx = 0 to kNumTokens-1:                      │     │
│    │     load_activation<VPT>(mat_a + m_idx*H + k_base, a) │     │
│    │     - bf16: 加载8个bf16并转换为FP32                    │     │
│    │     - fp32: 直接加载4个FP32                            │     │
│    └───────────────────────────────────────────────────────┘     │
│    ┌───────────────────────────────────────────────────────┐     │
│    │ 4c. 逐元素乘加:                                       │     │
│    │   for k = 0 to VPT-1:                                 │     │
│    │     acc[m_idx] += a_float[k] × b_float[k]             │     │
│    └───────────────────────────────────────────────────────┘     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: Warp 级蝴蝶规约 (Warp-level Butterfly Reduction)        │
│  对每个 token m:                                                  │
│    sum = acc[m]                                                   │
│    sum += __shfl_xor_sync(sum, 16)  // 0↔16, 1↔17, ...          │
│    sum += __shfl_xor_sync(sum, 8)   // 0↔8,  1↔9,  ...          │
│    sum += __shfl_xor_sync(sum, 4)   // 0↔4,  1↔5,  ...          │
│    sum += __shfl_xor_sync(sum, 2)   // 0↔2,  1↔3,  ...          │
│    sum += __shfl_xor_sync(sum, 1)   // 0↔1                     │
│    if (laneId == 0) sm_reduction[m][warpId] = sum                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: __syncthreads() — 确保 sm_reduction 写入完成             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 7: Block 级规约 (仅 tid==0 执行)                           │
│  对每个 token m:                                                  │
│    final_sum = Σ(w=0 to kNumWarps-1) sm_reduction[m][w]          │
│    out[m × E + n_idx] = final_sum                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  输出: output[m][e] = Σ_k activation[m][k] × weight[e][k]       │
│  即: Output = Activation × Weight^T                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 逐句代码解读

### 3.1 加载辅助函数

```cuda
// 第18-19行: 模板声明 - VPT(Values Per Thread)控制每线程加载数量
template <int VPT>
__device__ __forceinline__ void load_weight(float const* ptr, float* dst);
```

**VPT=4 特化（第21-28行）**：FP32 激活时使用，一次 `float4` 加载 4 个 FP32 值：
```cuda
template <>
__device__ __forceinline__ void load_weight<4>(float const* ptr, float* dst) {
  float4 v = *reinterpret_cast<float4 const*>(ptr);  // 128-bit 向量化加载，单次事务
  dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;  // 解包到 4 个标量
}
```

**VPT=8 特化（第30-42行）**：BF16 激活时使用，两次 `float4` 加载 8 个 FP32 值：
```cuda
template <>
__device__ __forceinline__ void load_weight<8>(float const* ptr, float* dst) {
  float4 v0 = *reinterpret_cast<float4 const*>(ptr);     // 加载前 4 个 FP32
  float4 v1 = *reinterpret_cast<float4 const*>(ptr + 4); // 加载后 4 个 FP32
  dst[0..3] = v0.xyzw; dst[4..7] = v1.xyzw;
}
```

**BF16 激活加载（第60-67行）**：
```cuda
template <>
__device__ __forceinline__ void load_activation<__nv_bfloat16, 8>(
    __nv_bfloat16 const* ptr, float* dst) {
  uint4 v = *reinterpret_cast<uint4 const*>(ptr);  // 128-bit 加载 = 8 × bf16 = 16 字节
  __nv_bfloat16 const* bf16_ptr = reinterpret_cast<__nv_bfloat16 const*>(&v);
  #pragma unroll
  for (int i = 0; i < 8; i++) dst[i] = __bfloat162float(bf16_ptr[i]); // 逐个转 FP32
}
```

### 3.2 核心 Kernel 逐句解读

```cuda
// 第76-78行: 模板参数
// InputT: 激活类型(float或__nv_bfloat16)
// kBlockSize: 块大小=128
// kNumTokens: token数M(1~32), 编译期常量
// kNumExperts: 专家数E=256
// kHiddenDim: 隐藏维度H=3072
template <typename InputT, int kBlockSize, int kNumTokens, int kNumExperts,
          int kHiddenDim>
__global__ __launch_bounds__(128, 1)  // 固定128线程/block, 最少1个block/SM
void fp32_router_gemm_kernel(
    float* out, InputT const* mat_a, float const* mat_b) {

  // 第80行: VPT = 16/sizeof(InputT)
  //   fp32: 16/4=4, bf16: 16/2=8
  constexpr int VPT = 16 / sizeof(InputT);

  // 第81行: 每次K迭代, 整个block处理的元素数
  //   fp32: 4×128=512, bf16: 8×128=1024
  constexpr int k_elems_per_k_iteration = VPT * kBlockSize;

  // 第82行: K维度迭代次数
  //   fp32: 3072/512=6, bf16: 3072/1024=3
  constexpr int k_iterations = kHiddenDim / k_elems_per_k_iteration;

  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;  // 128/32=4

  // 第86-89行: 索引计算
  int const n_idx = blockIdx.x;   // 当前block负责第几个专家(0~255)
  int const tid = threadIdx.x;    // 线程ID(0~127)
  int const warpId = tid / kWarpSize;   // warp ID(0~3)
  int const laneId = tid % kWarpSize;   // lane ID(0~31)

  // 第91行: 每个线程维护M个FP32累加器
  float acc[kNumTokens] = {};  // 零初始化

  // 第92行: 共享内存, 用于warp间规约
  // 大小: kNumTokens × 4 × 4字节 = M×16字节
  __shared__ float sm_reduction[kNumTokens][kNumWarps];

  // 第94行: 指向当前专家的权重行(行优先, 长度H)
  float const* b_col = mat_b + n_idx * kHiddenDim;

  // 第96-100行: 预计算K维度的基地址偏移, 避免循环内重复计算
  int k_bases[k_iterations];
  #pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) {
    k_bases[ki] = ki * k_elems_per_k_iteration + tid * VPT;
  }

  // 第102-104行: SM90+ 的程序化网格依赖控制
  // C500 (sm_80) 上不生效, 编译时会被优化掉
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;");
  #endif

  // 第106-122行: 主计算循环 — 沿K维度分块
  for (int ki = 0; ki < k_iterations; ki++) {
    int const k_base = k_bases[ki];

    // 加载权重: 每线程加载VPT个FP32权重值
    float b_float[VPT];
    load_weight<VPT>(b_col + k_base, b_float);

    // 对每个token做点积
    #pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
      // 加载激活值: bf16→fp32转换 或 直接fp32加载
      float a_float[VPT];
      load_activation<InputT, VPT>(mat_a + m_idx * kHiddenDim + k_base, a_float);

      // 逐元素FP32乘加
      #pragma unroll
      for (int k = 0; k < VPT; k++) {
        acc[m_idx] += a_float[k] * b_float[k];
      }
    }
  }

  // 第124-134行: Warp级蝴蝶规约
  // 使用 __shfl_xor_sync 进行 warp 内全规约
  #pragma unroll
  for (int m = 0; m < kNumTokens; m++) {
    float sum = acc[m];
    sum += __shfl_xor_sync(0xffffffff, sum, 16); // 高16与低16交换并相加
    sum += __shfl_xor_sync(0xffffffff, sum, 8);  // 8对交换并相加
    sum += __shfl_xor_sync(0xffffffff, sum, 4);  // 4对交换并相加
    sum += __shfl_xor_sync(0xffffffff, sum, 2);  // 2对交换并相加
    sum += __shfl_xor_sync(0xffffffff, sum, 1);  // 相邻交换并相加
    // 规约完成后, lane 0 持有该 warp 的完整部分和
    if (laneId == 0) sm_reduction[m][warpId] = sum;
  }

  // 第136行: 同步, 确保所有 warp 写入 sm_reduction 完成
  __syncthreads();

  // 第138-146行: Block级最终规约 — 仅 tid==0 执行
  if (tid == 0) {
    #pragma unroll
    for (int m = 0; m < kNumTokens; m++) {
      float final_sum = 0.0f;
      #pragma unroll
      for (int w = 0; w < kNumWarps; w++) {
        final_sum += sm_reduction[m][w];  // 累加4个warp的部分和
      }
      // 写入输出: 行优先, [m, n_idx]
      out[m * kNumExperts + n_idx] = final_sum;
    }
  }

  // 第148-150行: SM90+ 网格依赖(同上, C500不生效)
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
  #endif
}
```

### 3.3 Launcher 函数解读

```cuda
// 第157-164行: 调用接口
template <typename InputT, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeFp32RouterGemm(float* output, InputT const* mat_a,
                          float const* mat_b, cudaStream_t stream) {
  constexpr int kBlockSize = 128;
  // 启动 256 个 block, 每个 block 128 线程
  // 256 个 block 对应 256 个专家, 每个block计算一列输出
  fp32_router_gemm_kernel<InputT, kBlockSize, kNumTokens, kNumExperts,
                          kHiddenDim><<<kNumExperts, kBlockSize, 0, stream>>>(
      output, mat_a, mat_b);
}
```

### 3.4 Entry 层解读 (fp32_router_gemm_entry.cu)

```cuda
// 编译期循环展开器 — 将运行时 num_tokens 映射到编译期模板参数
// 这是关键优化: kNumTokens 作为模板参数, 编译器可以完全展开内层循环
template <typename InputT, int kBegin, int kEnd>
struct Fp32LoopUnroller {
  static void unroll(int num_tokens, float* output, InputT const* mat_a,
                     float const* mat_b, cudaStream_t stream) {
    if (num_tokens == kBegin) {
      invokeFp32RouterGemm<InputT, kBegin, 256, 3072>(...);
    } else {
      Fp32LoopUnroller<InputT, kBegin + 1, kEnd>::unroll(...);  // 递归
    }
  }
};

// 终止条件
template <typename InputT, int kEnd>
struct Fp32LoopUnroller<InputT, kEnd, kEnd> { ... };

// 主入口: 校验参数 → 选择类型 → 循环展开
void fp32_router_gemm(torch::Tensor& output, torch::Tensor const& mat_a,
                      torch::Tensor const& mat_b) {
  // 校验: 2D, CUDA, contiguous, H=3072, E=256, M∈[0,32]
  // bf16: Fp32LoopUnroller<__nv_bfloat16, 1, 32>
  // fp32: Fp32LoopUnroller<float, 1, 32>
}
```

---

## 4. 典型 Shape 举例详细阐述

### 4.1 场景：DeepSeek-V3 Decode, M=8, BF16 激活

**输入**：
- `mat_a`: shape [8, 3072], dtype=bfloat16
- `mat_b`: shape [256, 3072], dtype=float32
- `output`: shape [8, 256], dtype=float32

**编译期常量**：
```
VPT = 16 / sizeof(__nv_bfloat16) = 16 / 2 = 8
k_elems_per_k_iteration = 8 × 128 = 1024
k_iterations = 3072 / 1024 = 3
kNumWarps = 128 / 32 = 4
```

**Kernel 启动**：`<<<256, 128>>>`, 即 256 个 block, 每个 block 128 线程。

**单个 block（假设 blockIdx.x = 42，即第 42 号专家）的执行过程**：

#### Step 1: 初始化
```
n_idx = 42
b_col = mat_b + 42 × 3072  → 指向专家42的权重

acc[0..7] = {0, 0, 0, 0, 0, 0, 0, 0}  // 8个FP32累加器

sm_reduction[8][4]  → 32 × 4 = 128 字节共享内存

k_bases[0] = 0 × 1024 + tid × 8    // tid=0: 0, tid=1: 8, ..., tid=127: 1016
k_bases[1] = 1 × 1024 + tid × 8    // tid=0: 1024, tid=1: 1032, ...
k_bases[2] = 2 × 1024 + tid × 8    // tid=0: 2048, tid=1: 2056, ...
```

**K 维度分工**：128 线程 × 8 元素/线程 = 1024 元素/迭代 × 3 迭代 = 3072 = H ✓

#### Step 2: ki=0（K 维度第 0 段, k∈[0, 1023]）

以 **tid=5** 为例：
```
k_base = k_bases[0] = 0 + 5 × 8 = 40

加载权重: load_weight<8>(b_col + 40, b_float)
  → 加载 mat_b[42][40..47] 这 8 个 FP32 值

对 token 0 (m_idx=0):
  加载激活: load_activation<bf16,8>(mat_a + 0×3072 + 40, a_float)
    → 从全局内存加载 8 个 BF16 = 16 字节 (uint4 一次加载)
    → 转换为 FP32: a_float[0..7]
  乘加: acc[0] += Σ(a_float[k] × b_float[k])  k=0..7

对 token 1 (m_idx=1):
  加载激活: load_activation<bf16,8>(mat_a + 1×3072 + 40, a_float)
  乘加: acc[1] += Σ(a_float[k] × b_float[k])

... (同理 token 2~7)
```

**全局内存访问模式分析**：
- 权重: 128 线程连续访问 b_col[0..1023], 完美合并 (coalesced)
  - 每 warp 32 线程 × 8 元素 = 256 元素 = 1KB 连续读取 ✓
- 激活: 128 线程连续访问 mat_a[...][0..1023], 完美合并 ✓

#### Step 3: ki=1 和 ki=2

同理处理 K 维度的 [1024, 2047] 和 [2048, 3071] 段。

#### Step 4: Warp 蝴蝶规约

ki 循环结束后，每个线程持有部分和 `acc[0..7]`。Warp 内需要规约：

```
以 token 0 为例, warp 0 内 (laneId 0~31):

初始: lane 0 持 acc[0]_0, lane 1 持 acc[0]_1, ..., lane 31 持 acc[0]_31

Round 1: __shfl_xor_sync(sum, 16)
  lane 0 ↔ lane 16 交换并相加
  lane 1 ↔ lane 17 交换并相加
  ... 结果: 每个lane持2个值的和

Round 2: __shfl_xor_sync(sum, 8)
  lane 0 ↔ lane 8 交换并相加
  ... 结果: 每个lane持4个值的和

Round 3: __shfl_xor_sync(sum, 4)
  ... 结果: 每个lane持8个值的和

Round 4: __shfl_xor_sync(sum, 2)
  ... 结果: 每个lane持16个值的和

Round 5: __shfl_xor_sync(sum, 1)
  ... 结果: 每个lane持32个值的和 = warp 0 的完整部分和

lane 0 将结果写入 sm_reduction[0][0]
```

4 个 warp 各自规约后，`sm_reduction[m][0..3]` 存有 4 个部分和。

#### Step 5: Block 最终规约

```
仅 tid=0 执行:
  对 token 0: final_sum = sm[0][0] + sm[0][1] + sm[0][2] + sm[0][3]
  对 token 1: final_sum = sm[1][0] + sm[1][1] + sm[1][2] + sm[1][3]
  ...
  对 token 7: final_sum = sm[7][0] + sm[7][1] + sm[7][2] + sm[7][3]

  写入: out[0×256+42] = token0 对专家42 的得分
        out[1×256+42] = token1 对专家42 的得分
        ...
        out[7×256+42] = token7 对专家42 的得分
```

#### Step 6: 256 个 block 并行

所有 256 个 block 同时执行上述过程，最终 `output[8][256]` 被完整填满。

### 4.2 FP32 激活场景对比

若 `mat_a` 为 FP32：
```
VPT = 16 / sizeof(float) = 4
k_elems_per_k_iteration = 4 × 128 = 512
k_iterations = 3072 / 512 = 6  (多一倍迭代, 但每次加载量相同)

每线程: 6 次迭代 × 8 tokens × 4 乘加 = 192 FP32 FMA
vs BF16: 3 次迭代 × 8 tokens × 8 乘加 = 192 FP32 FMA
→ 计算量相同, 但 FP32 需要更多全局内存带宽(激活体积×2)
```

---

## 5. 基于 Metax C500 的优化空间分析

### 5.1 算子瓶颈分析

首先判断该算子属于 Memory Bound 还是 Compute Bound：

**以 M=8, BF16 为例**：
```
计算量 (FMA):
  256 blocks × 128 threads × 3 iterations × 8 tokens × 8 FMA = 6,291,456 FMA
  = 6.29M FP32 FMA ≈ 12.58 GFLOP

数据读取量:
  权重: 256 × 3072 × 4B = 3.14 MB (FP32)
  激活: 8 × 3072 × 2B = 49.15 KB (BF16)
  输出: 8 × 256 × 4B = 8.19 KB
  总计: ~3.20 MB

算术强度: 12.58 GFLOP / 3.20 MB ≈ 3.93 FLOP/Byte

C500 峰值:
  FP32 算力: ~19.5 TFLOP/s (非Tensor Core)
  带宽: 1.55 TB/s

Roofline 转折点: 19.5 TFLOP / 1.55 TB/s ≈ 12.6 FLOP/Byte
```

**结论: 算术强度 3.93 << 12.6, 该算子是典型的 Memory Bound 算子。**

权重矩阵 (3.14 MB) 占总数据量的 98%，是带宽瓶颈的核心。

### 5.2 优化空间详细分析

#### 优化点 1: 激活值缓存复用 — 高优先级

**问题**: 当前实现中，128 个线程都从全局内存读取同一组激活值。对于同一个 K 段，所有线程读取不同的激活元素（按 tid 分工），这是正确的。但 **权重被 128 个线程加载后，每个 token 都重新加载同一组权重**——权重在每个线程内被复用（存入 `b_float[VPT]`），这已经是最优的。

**但激活值存在复用机会**：当前每个线程独立加载所有 M 个 token 的激活。如果将激活先加载到共享内存，可以让 warp 内的线程共享，减少全局内存访问。不过由于每个线程处理不同的 K 元素，激活的 K 维度没有重叠，所以当前方式已经是合理的。

**实际优化机会**：当 M 较小时（如 M=1~4），权重读取占比极高。可以考虑 **多个 block 合并处理同一列**，但当前架构已经是一个 block 一列，改动的收益有限。

#### 优化点 2: 权重预取到 Shared Memory — 高优先级

**问题**: 当前权重直接从全局内存加载。对于 M=8, BF16 场景，每个线程在 3 次 K 迭代中，每次加载 8 个 FP32 权重（32 字节），然后对 8 个 token 乘加。权重只在当前 K 段的 8 个 token 计算中复用。

**优化方案**: 使用 **cp.async** 异步预取权重到共享内存，实现计算与加载的重叠：

```cuda
// C500 支持 cp.async (sm_80)
// 在计算第 ki 轮时, 异步预取第 ki+1 轮的权重到 smem
__shared__ float sm_weight[k_elems_per_k_iteration]; // 1024 × 4B = 4KB

// 预取第一轮
__pipeline_memcpy_async(sm_weight, b_col, k_elems_per_k_iteration * sizeof(float));
__pipeline_commit();

for (int ki = 0; ki < k_iterations; ki++) {
  __pipeline_wait_prior(0);  // 等待预取完成

  // 预取下一轮 (如果还有)
  if (ki + 1 < k_iterations) {
    __pipeline_memcpy_async(sm_weight, b_col + (ki+1)*k_elems_per_k_iteration, ...);
    __pipeline_commit();
  }

  // 从 smem 读取权重 (而非 global mem)
  float b_float[VPT];
  load_from_smem<VPT>(sm_weight + tid * VPT, b_float);

  // ... 乘加计算不变 ...
}
```

**预期收益**: 隐藏权重加载延迟，特别是当 M 较小时（权重加载占比高）。估计提升 10-20%。

#### 优化点 3: 使用 TF32 Tensor Core — 高优先级

**问题**: 当前使用 FP32 标量乘加，C500 FP32 吞吐为每 SM 每 cycle 64 FMA。C500 的第三代 Tensor Core 支持 TF32，吞吐可达 FP32 的 8 倍。

**优化方案**: 将权重和激活转换为 TF32 格式，使用 `wmma::load_matrix_sync` + `wmma::mma_sync` 进行 Tensor Core 矩阵乘：

```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// 每个 block 计算 16×16 (或 16×8) 的小块
// 对于 M=8, 可以用 8×16 的 fragment
fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> b_frag;
fragment<accumulator, 16, 16, 8, precision::tf32> c_frag;
```

**预期收益**: TF32 Tensor Core 的吞吐远高于 FP32 标量，估计提升 2-4 倍。但需要注意：
- M 很小（1~32），Tensor Core 的 16×16 分块可能利用率不高
- 需要仔细设计分块策略来适应小 M
- TF32 精度略低于 FP32（10 bit 尾数 vs 23 bit），需要验证路由精度是否满足要求

#### 优化点 4: 避免单线程最终规约 — 中优先级

**问题**: Step 7 中，只有 `tid==0` 执行最终规约，存在两个问题：
1. **线程利用率低**: 128 线程中只有 1 个在工作
2. **C500 上避免原子操作**: 当前不用原子操作是正确的（C500 原子操作性能差），但串行化也低效

**优化方案 A — Warp Shuffle 最终规约**:

```cuda
// 替代 tid==0 串行规约
// 使用一个完整 warp 来做最终规约
if (warpId == 0) {
  float val = 0.0f;
  if (laneId < kNumWarps) val = sm_reduction[m][laneId];
  // warp 内规约
  val += __shfl_xor_sync(0xffffffff, val, 2);
  val += __shfl_xor_sync(0xffffffff, val, 1);
  if (laneId == 0) out[m * kNumExperts + n_idx] = val;
}
```

**优化方案 B — 向量化写入**:

如果 M 较大，可以让多个线程各负责一部分 token 的写入。

**预期收益**: 微小（最终规约只涉及 4 个加法），但更优雅。

#### 优化点 5: 激活值的 L2 缓存利用 — 中优先级

**问题**: 256 个 block 各自独立读取完整的激活矩阵。对于 M=8, BF16，激活矩阵只有 ~49 KB，远小于 C500 的 8 MB L2 缓存。

**现状分析**: 实际上，由于所有 block 读取同一份激活，L2 缓存自然会把激活缓存下来。256 个 block 大致对应 256/2.4 ≈ 107 个 SM（C500 有 104 个 SM），激活矩阵会在 L2 中被高效复用。

**潜在优化**: 使用 `cudaFuncSetAttribute` 设置 L2 驻留策略：

```cuda
cudaFuncSetAttribute(fp32_router_gemm_kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxL1);  // 最大化 L1/共享内存
```

或者使用 `cudaAccessPolicyWindow` 设置 L2 持久化：

```cuda
cudaAccessPolicyWindow window;
window.base_ptr = reinterpret_cast<void*>(const_cast<InputT*>(mat_a));
window.num_bytes = M * kHiddenDim * sizeof(InputT);
window.hitRatio = 1.0;
cudaStreamAttrValue stream_attribute;
stream_attribute.accessPolicyWindow = window;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                       &stream_attribute);
```

**预期收益**: 小（L2 自然缓存已经不错），但在 M 较大时可能有 5-10% 的提升。

#### 优化点 6: Grid 大小优化 — 低优先级

**问题**: C500 有 104 个 SM，当前 256 个 block 可以充分填充。但 M=1 时每个 block 的计算量极小，可能导致 SM 空闲等待。

**优化方案**: 当 M 很小时，可以考虑 **将多个专家的权重合并在一个 block 中计算**，增加单 block 计算量，减少总 block 数：

```
M=1 时: 原始 256 blocks × 极小计算量
优化: 64 blocks, 每个计算 4 个专家 → 更好的 SM 利用率
```

但这需要修改 kernel 结构（增加内层专家循环），复杂度较高。

#### 优化点 7: BF16 激活的向量化加载优化 — 低优先级

**问题**: 当前 `load_activation<bf16,8>` 使用 `uint4` 加载 8 个 BF16，然后逐个转换。这个方式已经是 128-bit 向量化加载，基本是最优的。

**微优化**: 可以使用 `__bfloat1622float2` 一次转换 2 个 BF16 为 2 个 FP32，减少转换指令数：

```cuda
uint4 v = *reinterpret_cast<uint4 const*>(ptr);
__nv_bfloat162 const* bf16x2_ptr =
    reinterpret_cast<__nv_bfloat162 const*>(&v);
#pragma unroll
for (int i = 0; i < 4; i++) {
  float2 f2 = __bfloat1622float2(bf16x2_ptr[i]);
  dst[i*2] = f2.x;
  dst[i*2+1] = f2.y;
}
```

**预期收益**: 微小（减少少量转换指令），但实现简单。

#### 优化点 8: 寄存器压力优化 — 低优先级

**问题**: 当 M=32 时，每个线程需要 32 个 FP32 累加器 + VPT 个临时变量，寄存器压力较大。`__launch_bounds__(128, 1)` 要求最少 1 block/SM，意味着最多 2048/128 = 16 blocks/SM，但寄存器限制可能降低实际占用率。

**分析**: 32 个累加器 × 4 字节 = 128 字节，加上其他变量约 60 个寄存器/线程。C500 每 SM 64K 寄存器，128 线程 × 60 = 7680 寄存器，远小于 64K，不会成为瓶颈。

---

## 6. 综合优化建议优先级

| 优先级 | 优化方向 | 预期提升 | 实现难度 | 风险 |
|--------|----------|----------|----------|------|
| **P0** | cp.async 异步预取权重到 smem | 10-20% | 中 | 低 |
| **P0** | TF32 Tensor Core 加速 | 2-4x | 高 | 中（精度需验证） |
| **P1** | Warp Shuffle 最终规约 | <5% | 低 | 低 |
| **P1** | L2 缓存持久化 | 5-10% | 低 | 低 |
| **P2** | bf1622float2 双路转换 | <3% | 低 | 低 |
| **P2** | 小M时多专家/block | 10-30% | 高 | 中 |

**最关键的优化**是 **cp.async 异步预取**（立即可做，风险低）和 **TF32 Tensor Core**（收益最大，但需精度验证）。对于 C500 上的 Memory Bound 算子，减少全局内存访问延迟是最直接的提升手段。
