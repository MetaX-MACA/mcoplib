# minimax_allreduce_rms_qk Kernel 深度解读

> 源文件：`op/vllm/minimax_reduce_rms_kernel.cu`
> 头文件：`op/vllm/minimax_reduce_rms_kernel.h`
> 适用平台：Metax C500 / C280 (sm_80, warp-32)，CUDA 兼容

---

## 目录

1. [算子总览](#1-算子总览)
2. [源文件结构与命名空间](#2-源文件结构与命名空间)
3. [LamportComm 结构体 —— 跨卡通信的核心](#3-lamportcomm-结构体--跨卡通信的核心)
4. [负零哨兵与 volatile 加载](#4-负零哨兵与-volatile-加载)
5. [RMS rsqrt 助手](#5-rms-rsqrt-助手)
6. [Warp / Block 归约原语](#6-warp--block-归约原语)
7. [IndexHelper —— 索引映射](#7-indexhelper--索引映射)
8. [标量 Q-only Kernel —— 通用 fallback 路径](#8-标量-q-only-kernel--通用-fallback-路径)
9. [Float4 QK 融合 Kernel —— MiniMax-M2 专用优化路径](#9-float4-qk-融合-kernel--minimax-m2-专用优化路径)
10. [Launcher 与 NRanks 分发](#10-launcher-与-nranks-分发)
11. [Host 入口 minimax_allreduce_rms_qk](#11-host-入口-minimax_allreduce_rms_qk)
12. [跨卡 AllReduce 实现机制详解](#12-跨卡-allreduce-实现机制详解)
13. [RMSNorm 实现详解](#13-rmsnorm-实现详解)
14. [整体流程图](#14-整体流程图)
15. [MACA 平台适配注意事项](#15-maca-平台适配注意事项)

---

## 1. 算子总览

`minimax_allreduce_rms_qk` 是 **MiniMax-M2 模型 attention 前的 fused QK-RMSNorm 算子**，把"跨卡 AllReduce 方差 + RMSNorm"两步融合进一个 kernel，避免单独 AllReduce 的 launch 开销和中间 buffer 的往返读写。

### 输入输出

| 参数 | 类型 | shape | 含义 |
|---|---|---|---|
| `qkv` | bf16/fp16/fp32 | `[num_tokens, q_size + 2*kv_size]` | 融合张量，前 `q_size` 是 Q 分片，接着 `kv_size` 是 K 分片 |
| `norm_weight_q` | 同上 | `[q_size]` | Q 的 RMSNorm 权重（本卡分片） |
| `norm_weight_k` | 同上 | `[kv_size]` | K 的 RMSNorm 权重（本卡分片） |
| `workspace` | int64 | `[3N+2]` 或更大 | void** 指针数组，存 Lamport buffer 指针和 flag/layout |
| `q_size` | int | - | 本卡 Q 维度（如 1536） |
| `kv_size` | int | - | 本卡 K 维度（如 256） |
| `rank` | int | `0..N-1` | 本卡 TP rank |
| `nranks` | int | `2/4/8/16` | TP size |
| `eps` | float | `1e-6` | RMSNorm eps |

输出：`q_out [num_tokens, q_size]` 和 `k_out [num_tokens, kv_size]`，均已 RMSNorm 过。

### 核心数学

对每个 token 的全维度向量 `x`（维度 `OriginQDim = q_size * nranks`）：

```
variance_local = sum(x_shard²)              # 每卡本地算（仅本卡分片）
variance_full  = AllReduce(variance_local)  # 跨卡求和 → 全维度 variance
rms_scale = rsqrt(variance_full / OriginQDim + eps)
out = x_shard * rms_scale * norm_weight
```

### 关键设计点

- **跨卡 AllReduce 不用 NCCL**，而用 **Lamport 三缓冲协议** + 原始 device 指针 + `ld_global_volatile` 自旋等待。AllReduce 在 kernel 内部完成，省掉 host 端 NCCL 调用和中间 buffer 读写。
- **NRanks 是编译期模板参数**（只支持 2/4/8/16），因为 `LamportComm<NRanks>` 用了定长数组 `uint8_t* data_bufs[NRanks]`，且 warp shuffle 的 active_mask 依赖 NRanks。
- **Float4 QK 融合路径**：当 `q_size * nranks == 6144 && kv_size * nranks == 1024`（即 MiniMax-M2 的 Q=6144、K=1024）时启用优化路径，每个 thread 处理 4 个 token，Q 和 K 在同一个 kernel 内并行处理。

---

## 2. 源文件结构与命名空间

```cpp
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <torch/cuda.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "cuda_compat.h"
#include "cuda_utils.h"
#include "core/registration.h"
#include "minimax_reduce_rms_kernel.h"
#include <algorithm>

#define FINAL_MASK 0xffffffff
#define MINIMAX_REDUCE_RMS_WARP_SIZE 32

namespace vllm { namespace tensorrt_llm {
```

- `cooperative_groups.h` —— 用于 sm_90+ 的 cluster/grid 协同（Hopper）；sm_80（MACA C500/C280）走 `#else` 分支，退化为 `blockIdx.x` / `threadIdx.x`。
- `FINAL_MASK = 0xffffffff` —— warp shuffle 的全活跃掩码（32 线程全参与）。
- `MINIMAX_REDUCE_RMS_WARP_SIZE = 32` —— 一个 warp 32 线程，按 CUDA 标准 warp 语义编码。

---

## 3. LamportComm 结构体 —— 跨卡通信的核心

```cpp
template <int NRanks>
struct LamportComm {
  __device__ __forceinline__ LamportComm(void** workspace, int rank) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr    = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
    clear_ptr   = &reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[0];
    flag_value  = *flag_ptr;
    auto comm_size = reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[1];
    clear_size  = *clear_ptr;
    int data_offset  = flag_value % 3;
    int clear_offset = (flag_value + 2) % 3;
    for (int r = 0; r < NRanks; ++r) {
      data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r])
                     + data_offset * comm_size;
    }
    clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank])
                + clear_offset * comm_size;
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int64_t new_clear_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {}
      *flag_ptr = (flag_value + 1) % 3;
      *clear_ptr = new_clear_size;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  int64_t* clear_ptr;
  uint8_t* data_bufs[NRanks];
  uint8_t* clear_buf;
  int64_t clear_size;
  int flag_value;
};
```

### workspace 的 void** 指针数组布局

以 NRanks=4 为例（共 `3N+2 = 14` 个槽位，实际 workspace 张量通常 over-allocate 到 `(50,)`）：

| 索引 | 内容 | 说明 |
|---|---|---|
| `[0 .. N-1]` | ipc_buffers placeholder | 预留给 IPC buffer 指针（本实现未用） |
| `[N .. 2N-1]` | ipc_barriers placeholder | 预留给 IPC barrier（本实现未用） |
| `[2N .. 3N-1]` | 各 rank 的 Lamport 数据 buffer 指针 | **真正用到的部分** |
| `[3N]` | flag_buf: `int32[3] = {counter, unused, flag}` | counter 用于 block 间同步，flag 是三缓冲轮转索引 |
| `[3N+1]` | layout_buf: `int64[2] = {clear_size, comm_size}` | clear_size 是上一轮需要清除的字节数，comm_size 是单槽字节数 |

### 构造函数逐句解读

- **`counter_ptr` / `flag_ptr` / `clear_ptr`**：从 workspace 取出三个控制字段的地址。
- **`flag_value = *flag_ptr`**：读取当前 Lamport 轮转索引（0/1/2），所有 rank 在同一轮里读同一个值。
- **`comm_size`**：单槽字节数（从 layout_buf[1] 读）。
- **`clear_size = *clear_ptr`**：上一轮写入的字节数（从 layout_buf[0] 读），本轮需要清除这么多字节。
- **`data_offset = flag_value % 3`**：本轮要读/写的数据槽索引。
- **`clear_offset = (flag_value + 2) % 3`**：上一轮用过的槽，本轮要清零（轮转角度：上一轮 = `(flag - 1 + 3) % 3 = (flag + 2) % 3`）。
- **`data_bufs[r]`**：rank r 的 buffer 在本轮数据槽的起点。**注意：本 rank 的 kernel 同时持有所有 rank 的 buffer 指针**——这是跨卡 AllReduce 的基础（本 rank 要把数据写到所有 peer 的 buffer，也要从自己的 buffer 读出所有 peer 写进来的数据）。
- **`clear_buf`**：本 rank 自己的 buffer 在上一轮数据槽的起点，本轮负责清零。
- **`__syncthreads() + atomicAdd(counter_ptr, 1)`**：block 内所有线程同步后，由 thread 0 原子地累加 counter，用于 `update()` 中判断"所有 block 都到达了这一轮的末尾"。

### update() —— 轮转推进协议

- 只有 block 0 的 thread 0 执行（避免重复写）。
- 自旋等待 `counter == gridDim.x`，即所有 block 都跑完了本轮主体。
- 把 flag 推进到下一轮 `(flag+1) % 3`。
- 设置 `clear_ptr = new_clear_size`：告诉下一轮的 kernel "你要清除多少字节"。
- 重置 `counter = 0`，为下一轮做准备。

**这是 Lamport 三缓冲的核心**：通过共享的 `flag` + `counter`，多张卡并行跑同一个 kernel 时，所有 block 在本轮开始时读同一个 `flag_value`，本轮结束时由一个 block 把 `flag` 推进。下一轮 kernel 调用进来读新的 `flag_value`，会用一个不同的数据槽，避免覆盖上一轮还没读完的数据。

---

## 4. 负零哨兵与 volatile 加载

### 4.1 负零哨兵

```cpp
__device__ __forceinline__ bool is_neg_zero(float v) {
  return *reinterpret_cast<uint32_t*>(&v) == 0x80000000;
}
__device__ __forceinline__ bool is_neg_zero(float4 v) {
  return is_neg_zero(v.x) || is_neg_zero(v.y) ||
         is_neg_zero(v.z) || is_neg_zero(v.w);
}
__device__ __forceinline__ float4 get_neg_zero() {
  float4 vec;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    reinterpret_cast<uint32_t*>(&vec)[i] = 0x80000000;
  return vec;
}
```

- **`-0.0f` 的 IEEE-754 位模式 = `0x80000000`**。Lamport 协议用 -0.0f 作为"槽位空"哨兵：本 rank 自旋等待 peer 把数据写进来，写进来后位模式不再是 `0x80000000`，`is_neg_zero` 返回 false，循环退出。
- **为什么用 -0.0f 而不是 0.0f**：正常的 variance 总是 ≥ 0，但可能是 +0.0f（`0x00000000`）。如果用 0.0f 当哨兵，遇到真实 variance=+0 的 token 就会误判为"空槽"。用 -0.0f（`0x80000000`）和真实 +0.0f（`0x00000000`）位模式不同，可以区分。
- **`get_neg_zero()`**：返回一个全 -0.0f 的 float4，用于清空缓冲区。

### 4.2 volatile 加载

```cpp
__device__ __forceinline__ float4 ld_global_volatile(float4* addr) {
    volatile float4* vaddr = (volatile float4*)addr;
    float4 res;
    // 分量读取，确保 volatile 语义生效于每个分量
    res.x = vaddr->x;
    res.y = vaddr->y;
    res.z = vaddr->z;
    res.w = vaddr->w;
    return res;
}

__device__ __forceinline__ float ld_global_volatile(float *addr) {
  // MACA cucc miscompiles ((volatile int*)(addr))[0] -- it returns a stale
  // register value instead of re-reading global memory, which breaks the
  // Lamport spin-wait (is_neg_zero never sees the -0.0f sentinel and the
  // loop exits with garbage variance).  Casting through (volatile float*)
  // is verified to produce a true volatile global load on the C500/C280
  // sm_80 path.
  float val;
  __threadfence();
  val = *((volatile float *)addr);
  return val;
}
```

- **`volatile`** 关键字强制每次重新从全局内存读取，不允许编译器把值缓存到寄存器。否则自旋循环 `while(!done)` 会被优化成只读一次，永远看不到 peer 的写入，循环死锁或读到脏值。
- **`__threadfence()`**：在 volatile 读之前插入一次 GPU 内存的栅栏，保证本 GPU 之前的写操作都已经对其他线程可见。
- **float4 版本**：分量读取确保 volatile 语义作用于每个分量（C++ 标准不保证 struct 整体 volatile，必须分量操作）。

### 4.3 MACA 平台的特殊修复

原 vLLM 代码用 `((volatile int*)(addr))[0]`，在 MACA cucc 下被错误编译，读到 `0xcf000000` 之类的脏寄存器值，导致 `is_neg_zero` 永远返回 false，自旋循环立即退出，输出 NaN。

通过独立的 CUDA 测试程序验证了 5 种写法：

| 写法 | MACA 结果 |
|---|---|
| `(volatile int*)addr[0]` （原代码） | ❌ 读到 `0xcf000000` |
| `(volatile uint32_t*)addr` | ❌ 读到 `0x4f000000` |
| `*((volatile float*)addr)` | ✅ 正确读到 `0x80000000` |
| `__ldcg(addr)` | ✅ 正确 |
| `__threadfence() + *((volatile float*)addr)` | ✅ 正确 |

**修复方法是改用 `*((volatile float*)addr)`**——这是反复实验验证后找到的可行写法。

---

## 5. RMS rsqrt 助手

```cpp
template <int Dim>
__device__ __forceinline__ float rms_rsqrt(float& v, float eps) {
  constexpr float kInvDim = 1.0F / static_cast<float>(Dim);
  v = rsqrtf((v * kInvDim) + eps);
  return v;
}

template <int Dim>
__device__ __forceinline__ float4 rms_rsqrt(float4& v, float eps) {
  constexpr float kInvDim = 1.0F / static_cast<float>(Dim);
  v.x = rsqrtf((v.x * kInvDim) + eps);
  v.y = rsqrtf((v.y * kInvDim) + eps);
  v.z = rsqrtf((v.z * kInvDim) + eps);
  v.w = rsqrtf((v.w * kInvDim) + eps);
  return v;
}
```

- **`Dim` 是模板参数**（编译期常量），让 `1/Dim` 在编译期算好，运行时只做一次乘法 + 一次 `rsqrtf`。
- **`rsqrtf(x) = 1/sqrt(x)`**，比先 `sqrt` 再除快得多。
- 注意 `Dim` 是**完整维度**（如 6144），不是单卡分片维度。`v` 已经是跨卡求和后的总 variance。
- float4 版本对 4 个分量分别计算，对应 4 个 token 的 4 个 scale。

---

## 6. Warp / Block 归约原语

### 6.1 Warp 蝴蝶归约

```cpp
template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
  }
  return (T)(0.0f);
}
```

- **蝴蝶归约**：`mask = 16, 8, 4, 2, 1`，每步与相距 `mask` 的 lane 交换并累加。5 步后整个 warp 的 32 个 lane 都有总和。
- **`__shfl_xor_sync(mask, val, lane_offset, warp_size)`**：与 lane `xor(lane_id, offset)` 交换数据。
- `NUM` 模板参数支持同时归约多个值（如 4 个 token 的 variance）。

### 6.2 Block 两阶段归约

```cpp
template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val) {
  static __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSumV2<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) shared[i][wid] = val[i];
  }
  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++)
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  warpReduceSumV2<T, NUM>(val);
  return (T)(0.0f);
}
```

- **两阶段归约**：先 warp 内归约（lane 0 拿到 warp 总和），把每个 warp 的总和写到 shared memory 的 `shared[i][wid]`；同步后，前 `blockDim/32` 个 lane 从 shared memory 读出 warp 总和，再做一次 warp 归约，lane 0 拿到 block 总和。
- **`shared[NUM][33]`**：用 33 而不是 32 是为了避免 bank conflict（shared memory 32 个 bank，多 1 列做 padding）。
- **`is_mask` 判断**：只有前 `NumWarp` 个 lane 参与第二阶段（避免越界读）。

### 6.3 局部 warp 归约（float4 版）

```cpp
template <uint32_t kNumThreads, typename T, int ArraySize = 4>
__device__ __forceinline__ void local_warp_reduce_sum_array(
    T* value_ptr, uint32_t active_mask = 0xffffffffu) {
  static_assert(kNumThreads >= 1 && kNumThreads <= MINIMAX_REDUCE_RMS_WARP_SIZE);
#pragma unroll
  for (int i = 0; i < ArraySize; ++i) {
#pragma unroll
    for (int mask = kNumThreads / 2; mask > 0; mask >>= 1) {
      value_ptr[i] += __shfl_xor_sync(active_mask, value_ptr[i], mask,
                                      MINIMAX_REDUCE_RMS_WARP_SIZE);
    }
  }
}

constexpr int next_pow2(int val) {
  int result = 1;
  while (result < val) result <<= 1;
  return result;
}
```

- **`kNumThreads`** 是模板参数——只有前 `kNumThreads` 个 lane 参与（用 `active_mask` 控制），其他 lane 的值在 shuffle 中不参与。
- **`active_mask`**：例如 NRanks=4 时 `active_mask = 0b1111 = 0xF`，只有 lane 0-3 有效。这用于"前 NRanks 个 lane 各持有一个 rank 的方差，做完 warp 归约后所有 NRanks 个 lane 都有总和"的并行 AllReduce 模式。
- **`next_pow2`**：返回 ≥val 的最小 2 的幂，用于把非 2 幂的 NumWarp 凑成 2 幂做蝴蝶归约。

---

## 7. IndexHelper —— 索引映射

```cpp
template <typename DType>
class IndexHelper {
 public:
  __device__ __forceinline__ IndexHelper(MiniMaxReduceRMSParams const& params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();
    token_id = grid.cluster_rank();
    access_id_in_token = cluster.thread_rank();
    token_stride = grid.num_clusters();
#else
    token_id = blockIdx.x;
    access_id_in_token = threadIdx.x;
    token_stride = gridDim.x;
#endif
    access_id = token_id * params.hidden_dim / kElemsPerAccess<DType> +
                access_id_in_token;
    access_stride = token_stride * params.hidden_dim / kElemsPerAccess<DType>;
    tot_access = params.size_q / kElemsPerAccess<DType>;
  }
  ...
};
```

- **sm_90+**：用 cooperative groups 的 cluster_rank / thread_rank（Hopper 集群并行）。
- **sm_80（MACA C500）**：用经典的 `blockIdx.x` 当 token_id，`threadIdx.x` 当 access_id_in_token，`gridDim.x` 当 token_stride。
- **`access_id`**：当前线程要处理的第一个 float4 在 `allreduce_in` 中的索引。
- **`access_stride`**：grid-stride loop 的步长。
- **`tot_access`**：总共多少个 float4 access（= `size_q / kElemsPerAccess`）。
- **`kElemsPerAccess`**：bf16/fp16=8（一个 float4 = 8 个 bf16），fp32=4（一个 float4 = 4 个 float）。

---

## 8. 标量 Q-only Kernel —— 通用 fallback 路径

源码：`minimax_reduce_rms_kernel_lamport<DType, NRanks>`（L261-343）

当 `use_float4 == false`（即 shape 不是 MiniMax-M2 的 6144/1024）时走这条路径。每个 block 处理一个 token 的所有 float4 access，串行处理 NRanks 个 rank。

### 8.1 入口与构造

```cpp
template <typename DType, int NRanks>
__global__ void __launch_bounds__(1024)
    minimax_reduce_rms_kernel_lamport(MiniMaxReduceRMSParams params) {
  IndexHelper<DType> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;
  int tot_tokens = params.size_q / params.hidden_dim;
  float4 clear_vec = get_neg_zero();

  LamportComm<NRanks> comm(params.workspace, params.rank);
  int clear_access = comm.clear_size / kElemsPerAccess<DType>;
```

- **`__launch_bounds__(1024)`**：限制每个 block 最多 1024 线程，提示编译器寄存器分配。
- **`tot_tokens`**：当前 rank 的 token 数（= `size_q / hidden_dim`，hidden_dim 是单卡分片维度）。
- **`clear_vec`**：用于清空上一轮槽位的 -0.0f 哨兵。
- **`LamportComm` 构造**：完成 flag 读取、buffer 指针计算、counter++（每个 block 进入时 counter++，标志着"我这个 block 进入了本轮"）。
- **`clear_access`**：本轮要清空多少个 float4 access（从上一轮的 `clear_size` 继承）。

### 8.2 主体 grid-stride loop

```cpp
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  for (int idx = access_id; idx < tot_access;
       idx += access_stride, token_id += token_stride) {
    alignas(16) DType vals[kElemsPerAccess<DType>];
    float sum_variance = 0.F;
    *reinterpret_cast<float4*>(vals) =
        reinterpret_cast<float4*>(params.allreduce_in)[idx];
#pragma unroll
    for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
      sum_variance += static_cast<float>(vals[i]) * static_cast<float>(vals[i]);
    }
    blockReduceSumV2<float, 1>(&sum_variance);
    if (is_neg_zero(sum_variance)) {
      sum_variance = 0.F;
    }
```

**步骤 1：本地 variance 归约**
- 读一个 float4（8 个 bf16）到 `vals[]`。
- 每个 thread 算自己这 8 个元素的平方和（`sum_variance`）。
- `blockReduceSumV2<float, 1>(&sum_variance)`：block 内归约，最终 lane 0 拿到整行的 `sum(x²)`。
- **负零防护**：如果归约后 `sum_variance` 是 -0.0f（边界情况下可能产生），强制改成 +0.0f，避免被 Lamport 自旋循环误判为哨兵。

### 8.3 跨卡广播写

```cpp
    if (threadIdx.x == 0) {
      for (int r = 0; r < NRanks; ++r) {
        reinterpret_cast<float*>(
            comm.data_bufs[r])[(params.rank * tot_tokens) + token_id] =
            (sum_variance);
      }
    }
```

**步骤 2：广播写** —— 本 rank 的 block 0（thread 0）把自己的 variance 写到 **所有 NRanks 的 buffer** 在本轮数据槽的位置。
- 写到 `comm.data_bufs[r]` 即 rank r 的 buffer。
- 索引 `[rank * tot_tokens + token_id]`：每个 rank 在 buffer 里占 `tot_tokens` 个 float 的槽位，本 rank 写自己的槽位（offset = `rank * tot_tokens`）。
- **关键**：本 rank 写到所有 peer 的 buffer，意思是"我的 variance 在每张卡的 buffer 里都有一份副本"。这样 peer 卡的 kernel 读自己的 buffer 就能拿到所有 rank 的 variance。

### 8.4 跨卡聚合读（自旋等待）

```cpp
    bool done = false;
    float vars_all_ranks[NRanks];
    while (!done) {
      done = true;
#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        vars_all_ranks[r] = ld_global_volatile(&reinterpret_cast<float*>(
            comm.data_bufs[params.rank])[(r * tot_tokens) + token_id]);
        done &= !is_neg_zero(vars_all_ranks[r]);
      }
    }
    sum_variance = 0.F;
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      sum_variance += vars_all_ranks[r];
    }
```

**步骤 3：自旋等待 + 聚合读** —— 本 rank 的 thread 0 从**自己的 buffer**（`data_bufs[params.rank]`）读所有 NRanks 的 variance。
- 索引 `[(r * tot_tokens) + token_id]`：rank r 在本 rank buffer 里的槽位。
- **`ld_global_volatile`**：volatile 读，每次重新从 global memory 读，保证看到 peer 的最新写入。
- **`is_neg_zero` 检测**：如果是 -0.0f 哨兵，说明该 rank 还没写入，继续自旋。所有 NRanks 都不是 -0.0f 时 `done` 保持 true，循环退出。
- **`done &= !is_neg_zero(...)`**：用 `&=` 累积判断，只要有一个 rank 还没写入就 `done = false`。

**步骤 4：跨卡求和** —— 把所有 NRanks 的 variance 加起来，得到 `sum_variance`（这是跨卡 AllReduce 的结果）。

### 8.5 RMSNorm 计算与写出

```cpp
    DType norm_weight[kElemsPerAccess<DType>];
    *reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(norm_weight) =
        reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(
            params.rms_gamma)[access_id_in_token];

#pragma unroll
    for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
      vals[i] = static_cast<DType>(
          static_cast<float>(vals[i]) *
          rsqrtf(
              (sum_variance / static_cast<float>(params.hidden_dim) / NRanks) +
              params.rms_eps) *
          static_cast<float>(norm_weight[i]));
    }

    reinterpret_cast<float4*>(params.rms_norm_out)[idx] =
        *reinterpret_cast<float4*>(vals);
  }
```

**步骤 5：RMSNorm 计算 + 写出**
- 读 norm_weight（一个 float4，与 `vals` 同维度）。
- 对每个元素：`out = x * rsqrt(sum_variance / hidden_dim / NRanks + eps) * weight`
- **`hidden_dim * NRanks`** = 完整维度（如 6144），所以除以 `hidden_dim / NRanks` 等价于除以完整维度。这是 RMSNorm 的标准公式 `rsqrt(mean(x²) + eps)`。
- `static_cast<float>` 提升精度，避免 bf16 累加误差。
- 写到 `rms_norm_out[idx]`，一个 float4 写回。

### 8.6 清理与轮转推进

```cpp
  for (int idx = access_id; idx < clear_access; idx += access_stride) {
    reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
  }
  comm.update(params.size_q * NRanks);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}
```

- **清理上一轮的槽**：把 `clear_buf`（上一轮用过的数据槽）填回 -0.0f 哨兵，供两轮后的下一次调用使用。
- **`comm.update(params.size_q * NRanks)`**：把本轮写入的字节数（`size_q * NRanks`，因为本 rank 写了 NRanks 个 buffer，每个 `size_q` 字节）传给下一轮作为 `clear_size`。同时推进 flag、重置 counter。

---

## 9. Float4 QK 融合 Kernel —— MiniMax-M2 专用优化路径

源码：`minimax_reduce_qk_rms_kernel_lamport_float4<DType, NRanks, OriginQDim, OriginKDim>`（L352-636）

当 `hidden_dim * NRanks == 6144 && hidden_dim_k * NRanks == 1024` 时启用（即 MiniMax-M2 的 Q=6144、K=1024 维度）。把 Q 和 K 在同一个 kernel 里处理，每个 thread 处理 4 个 token（float4）。

### 9.1 编译期常量

```cpp
template <typename DType, int NRanks, int OriginQDim, int OriginKDim>
__global__ void __launch_bounds__(1024)
    minimax_reduce_qk_rms_kernel_lamport_float4(MiniMaxReduceRMSParams params) {
  constexpr int RankQDim = OriginQDim / NRanks;  // 每卡 Q 维度，如 6144/4=1536
  constexpr int RankKDim = OriginKDim / NRanks;  // 每卡 K 维度，如 1024/4=256
  constexpr int ThreadsPerRowQ = RankQDim / kElemsPerAccess<DType>;
  constexpr int ThreadsPerRowK = RankKDim / kElemsPerAccess<DType>;
  constexpr int NumWarpQ = (ThreadsPerRowQ + MINIMAX_REDUCE_RMS_WARP_SIZE - 1) /
                           MINIMAX_REDUCE_RMS_WARP_SIZE;
  constexpr int NumWarpK = (ThreadsPerRowK + MINIMAX_REDUCE_RMS_WARP_SIZE - 1) /
                           MINIMAX_REDUCE_RMS_WARP_SIZE;
```

bf16 时 `kElemsPerAccess=8`，所以 `ThreadsPerRowQ = 1536/8 = 192`，`NumWarpQ = 192/32 = 6`；`ThreadsPerRowK = 256/8 = 32`，`NumWarpK = 1`。总共 `NumWarpQ + NumWarpK = 7` 个 warp = 224 线程。

### 9.2 内存步长计算

```cpp
  int access_stride_q = (params.stride_q > 0 ? params.stride_q : RankQDim) /
                        kElemsPerAccess<DType>;
  int access_stride_k = (params.stride_k > 0 ? params.stride_k : RankKDim) /
                        kElemsPerAccess<DType>;
  int access_stride_q_out =
      (params.stride_q_out > 0 ? params.stride_q_out : params.hidden_dim) /
      kElemsPerAccess<DType>;
  int access_stride_k_out =
      (params.stride_k_out > 0 ? params.stride_k_out : params.hidden_dim_k) /
      kElemsPerAccess<DType>;
```

- 当 `stride_q > 0` 时，Q 是 qkv 张量的一部分，行 stride 等于整个 qkv 最后一维（如 `q_size + 2*kv_size = 1536 + 2*256 = 2048`）。
- 当 `stride_q = 0` 时，Q 是 contiguous 张量，行 stride 等于 `RankQDim`。
- 输出 `q_out` / `k_out` 通常 contiguous（`stride_q_out = 0`），用 `hidden_dim` 当 stride。

### 9.3 Grid/Block 索引

```cpp
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  cg::grid_group grid = cg::this_grid();
  int group_id = grid.cluster_rank();
  int access_id_in_token = cluster.thread_rank();
  int group_stride = grid.num_clusters();
#else
  int group_id = blockIdx.x;
  int access_id_in_token = threadIdx.x;
  int group_stride = gridDim.x;
#endif

  bool is_q = (access_id_in_token < NumWarpQ * MINIMAX_REDUCE_RMS_WARP_SIZE);
  int k_thread_idx =
      access_id_in_token - (NumWarpQ * MINIMAX_REDUCE_RMS_WARP_SIZE);
  bool is_valid_q = (access_id_in_token < ThreadsPerRowQ);
  bool is_valid_k = (k_thread_idx >= 0 && k_thread_idx < ThreadsPerRowK);
```

- **`group_id`**：当前 block 处理哪一组 4 个 token。
- **`is_q`**：当前 thread 是否属于 Q leader warp 区（前 `NumWarpQ` 个 warp）。
- **`k_thread_idx`**：K 区内的相对索引（从 `access_id_in_token` 减去 Q 区大小）。
- **`is_valid_q` / `is_valid_k`**：边界判断，防止 thread 越界读数据。

### 9.4 Norm weight 预加载

```cpp
  DType norm_weight[kElemsPerAccess<DType>]{};
  if (is_q) {
    if (is_valid_q) {
      *reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(
          norm_weight) =
          reinterpret_cast<typename ElemsPerAccess<DType>::vec_type const*>(
              params.rms_gamma)[access_id_in_token];
    }
  } else {
    if (is_valid_k) {
      *reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(
          norm_weight) =
          reinterpret_cast<typename ElemsPerAccess<DType>::vec_type const*>(
              params.rms_gamma_k)[k_thread_idx];
    }
  }
```

- 在主循环外预加载 norm_weight 到寄存器，避免循环内重复加载。
- Q 区线程从 `rms_gamma` 读，K 区线程从 `rms_gamma_k` 读。

### 9.5 主组循环 —— 4 个 token 一组

```cpp
  for (int g = group_id; g < tot_groups; g += group_stride) {
    alignas(16) DType vals[4][kElemsPerAccess<DType>]{};
    float warp_sum_variance[4]{0.F, 0.F, 0.F, 0.F};

    if (is_q) {
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        int token_r = g * 4 + row;
        if (token_r >= tot_tokens || !is_valid_q) continue;
        int idx_r = token_r * access_stride_q + access_id_in_token;
        *reinterpret_cast<float4*>(&vals[row][0]) =
            reinterpret_cast<float4 const*>(params.allreduce_in)[idx_r];
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          float x = static_cast<float>(vals[row][i]);
          warp_sum_variance[row] += x * x;
        }
      }
    } else {
      // K 分支类似，读 allreduce_in_k
    }

    local_warp_reduce_sum_array<MINIMAX_REDUCE_RMS_WARP_SIZE, float, 4>(
        warp_sum_variance);
```

- 每次迭代处理 **4 个 token**（一组），`warp_sum_variance[4]` 存 4 个 token 的方差。
- 每个 thread 读 4 个 token 的 float4 数据（每个 token 一行），算每个 token 的本地 variance。
- **warp 内归约**：每个 warp 的 lane 0 拿到该 warp 的 4 个 token 的本地 variance。

### 9.6 Block 内归约（两阶段）

```cpp
    int lane = threadIdx.x & (MINIMAX_REDUCE_RMS_WARP_SIZE - 1);
    if (lane == 0) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        block_reduce_sum[t][threadIdx.x / MINIMAX_REDUCE_RMS_WARP_SIZE] =
            warp_sum_variance[t];
      }
    }
    __syncthreads();

    int tid = threadIdx.x;
```

- warp lane 0 把 4 个 token 的 variance 写到 shared memory `block_reduce_sum[t][warp_id]`，shape `[4][WarpCount]`。
- `__syncthreads()` 保证所有 warp 都写完。

### 9.7 Q leader warp 的跨卡 AllReduce（并行模式）

```cpp
    if (tid < MINIMAX_REDUCE_RMS_WARP_SIZE) {
      constexpr int kNumWarpQPow2 =
          (next_pow2(NumWarpQ) > NRanks) ? next_pow2(NumWarpQ) : NRanks;
      float local_sum[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        local_sum[t] = (tid < NumWarpQ) ? block_reduce_sum[t][tid] : 0.F;
      }
      local_warp_reduce_sum_array<kNumWarpQPow2, float, 4>(local_sum);
```

- **第一个 warp（tid 0-31）**负责 Q 的跨卡归约。
- **`local_sum[t]`**：从 shared memory 读出本 warp 的 4 个 token 的本地 variance（前 `NumWarpQ` 个 lane 有效，其他填 0）。
- **`local_warp_reduce_sum_array<kNumWarpQPow2, float, 4>`**：warp 内归约，所有 `kNumWarpQPow2` 个 lane 都拿到 block 内所有 warp 的总和。`kNumWarpQPow2 = max(next_pow2(NumWarpQ), NRanks)`，确保至少有 `NRanks` 个 lane 参与（因为后面要用前 NRanks 个 lane 做并行 AllReduce）。

#### 并行 Push —— 广播写

```cpp
      if (tid < NRanks) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          if (is_neg_zero(local_sum[t])) local_sum[t] = 0.F;
        }
        // Parallel push: thread tid writes this rank's Q sum to rank tid's buf
        reinterpret_cast<float4*>(
            comm.data_bufs[tid])[(params.rank * tot_groups * 2) + (2 * g)] =
            *reinterpret_cast<float4*>(local_sum);
```

**并行 Push** —— 前 `NRanks` 个 lane（tid 0..NRanks-1）各负责一个 rank：
- lane `tid` 把本 rank 的 4 个 token 的 variance（打包成 float4）写到 **rank tid 的 buffer**。
- 索引 `[(rank * tot_groups * 2) + (2 * g)]`：本 rank 在 buffer 里的偏移，`* 2` 是因为 Q 和 K 交替存放（Q 在偶数槽，K 在奇数槽）。
- **NRanks 个 lane 并行写 NRanks 个 buffer**，每个 buffer 写一份本 rank 的 variance——这是 Lamport 协议的"广播写"。

#### 并行 Pull —— 聚合读（自旋等待）

```cpp
        // Parallel pull: thread tid reads rank tid's contribution from
        // this rank's (params.rank's) buffer
        bool done = false;
        float4 var_all_ranks;
        while (!done) {
          done = true;
          var_all_ranks = ld_global_volatile(&reinterpret_cast<float4*>(
              comm.data_bufs[params.rank])[(tid * tot_groups * 2) + (2 * g)]);
          done &= !is_neg_zero(var_all_ranks);
        }
```

**并行 Pull** —— 前 `NRanks` 个 lane 各负责一个 rank：
- lane `tid` 从**本 rank 的 buffer** 读 rank tid 的 variance（float4，4 个 token 一起）。
- 自旋等待直到不是 -0.0f 哨兵（即 peer tid 已经写入）。
- **NRanks 个 lane 并行读**，每个 lane 拿到不同 rank 的 variance。

#### Warp 级 AllReduce

```cpp
        // Warp-level allreduce: each of the NRanks threads holds one rank's
        // partial sum; after this all NRanks threads have the global total.
        constexpr uint32_t kQActiveMask = (1u << NRanks) - 1u;
        local_warp_reduce_sum_array<NRanks, float, 4>(
            reinterpret_cast<float*>(&var_all_ranks), kQActiveMask);

        // Thread 0 computes rsqrt with compile-time Dim and writes to smem
        if (tid == 0) {
          *reinterpret_cast<float4*>(global_scale_q) =
              rms_rsqrt<OriginQDim>(var_all_ranks, params.rms_eps);
        }
      }
    }
```

**Warp 级 All Reduce** —— 把 NRanks 个 lane 各自持有的不同 rank 的 variance 做蝴蝶归约（`kQActiveMask = (1<<NRanks)-1`，前 NRanks 个 lane 全活跃），归约后所有 NRanks 个 lane 都拿到总和。
- **lane 0 算 rsqrt**（用编译期 `OriginQDim = 6144`），写到 shared memory `global_scale_q[4]`（4 个 token 各一个 scale）。

### 9.8 K leader warp 的跨卡 AllReduce

```cpp
    } else if (tid >= MINIMAX_REDUCE_RMS_WARP_SIZE * NumWarpQ &&
               tid < MINIMAX_REDUCE_RMS_WARP_SIZE * (NumWarpQ + 1)) {
      // --- K leader warp ---
      constexpr int kNumWarpKPow2 =
          (next_pow2(NumWarpK) > NRanks) ? next_pow2(NumWarpK) : NRanks;
      ...
      // 索引用 (2 * g + 1)，OriginKDim = 1024
      ...
    }
    __syncthreads();
```

完全对称，只是用 K 的索引（`2*g+1`）和 K 的 `OriginKDim = 1024`。**关键**：K 部分由第二个 warp（tid 在 `[WARP_SIZE * NumWarpQ, WARP_SIZE * (NumWarpQ+1))`）处理，因为 Q 和 K 是分开的 leader warp。

### 9.9 RMSNorm 写出

```cpp
    if (is_q) {
#pragma unroll
      for (int t = 0; t < 4; ++t)
        warp_sum_variance[t] = global_scale_q[t];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          vals[r][i] = static_cast<DType>(static_cast<float>(vals[r][i]) *
                                          warp_sum_variance[r] *
                                          static_cast<float>(norm_weight[i]));
        }
        int token_r = g * 4 + r;
        if (token_r >= tot_tokens || !is_valid_q) continue;
        int idx_out = token_r * access_stride_q_out + access_id_in_token;
        reinterpret_cast<float4*>(params.rms_norm_out)[idx_out] =
            *reinterpret_cast<float4*>(&vals[r][0]);
      }
    } else {
      // K 分支写出 rms_norm_out_k
    }
  }  // end group loop
```

- 从 shared memory 读 `global_scale_q[4]`。
- 对每个 token：`out = x * scale * weight`，写回 `rms_norm_out`。
- 边界检查：`token_r >= tot_tokens` 时不写（最后一组可能不足 4 个 token）。

### 9.10 清理与轮转推进

```cpp
  int clear_access = static_cast<int>(comm.clear_size / kElemsPerAccess<DType>);
  int clear_stride = group_stride * blockDim.x;
  for (int idx = group_id * blockDim.x + threadIdx.x; idx < clear_access;
       idx += clear_stride) {
    reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
  }

  comm.update(static_cast<int64_t>(2) * tot_groups * kElemsPerAccess<DType> *
              NRanks);
```

- 清理上一轮槽位（所有线程参与，按 grid-stride 分配）。
- `update(2 * tot_groups * kElemsPerAccess * NRanks)`：本轮写入了 `2 * tot_groups * kElemsPerAccess * NRanks` 字节（Q+K 交替存放，每组 2 个 float4，每个 float4 = `kElemsPerAccess` 字节，跨 NRanks 个 buffer）。

---

## 10. Launcher 与 NRanks 分发

### 10.1 标量 kernel launcher

```cpp
template <typename DType, int NRanks>
void minimax_reduce_rms_kernel_launcher(MiniMaxReduceRMSParams const& params) {
  static int SM = getSMVersion();
  int token_num = params.size_q / params.hidden_dim;
  int sm_count = get_sm_count();
  int cluster_size = 1;
  int cluster_num = token_num;
  int threads_per_token = params.hidden_dim / kElemsPerAccess<DType>;
  int block_size = threads_per_token;

  int max_blocks_per_sm = get_max_active_blocks(
      minimax_reduce_rms_kernel_lamport<DType, NRanks>, block_size);
  int max_grid = max_blocks_per_sm * sm_count;

  int grid_size =
      (std::min(max_grid, cluster_num * cluster_size) / cluster_size) *
      cluster_size;
  minimax_reduce_rms_kernel_lamport<DType, NRanks>
      <<<grid_size, block_size, 0, params.stream>>>(params);
}
```

- **block_size = threads_per_token**：一个 block 处理一个 token 的所有 float4 access。如 bf16 Q 维度 1536 → 192 threads = 6 warps。
- **grid_size**：`min(max_grid, token_num)`，不超过 GPU 能并发跑的 block 数，避免空转。

### 10.2 Float4 kernel launcher

```cpp
template <typename DType, int NRanks, int OriginQDim, int OriginKDim>
void minimax_reduce_rms_kernel_launcher_float4(...) {
  ...
  int block_size = divUp(access_per_row_q, WARP_SIZE) +
                   divUp(access_per_row_k, WARP_SIZE);
  ...
}
```

float4 版的 block_size 是 Q 和 K 各自需要的 warp 数之和（向上取整到 warp 边界）。

### 10.3 NRanks 分发

```cpp
void minimax_reduce_rms_op(MiniMaxReduceRMSParams const& params) {
  if (params.nranks == 2) dispatch_dtype<2>(params);
  else if (params.nranks == 4) dispatch_dtype<4>(params);
  else if (params.nranks == 8) dispatch_dtype<8>(params);
  else if (params.nranks == 16) dispatch_dtype<16>(params);
  else TORCH_CHECK(false, "minimax_reduce_rms_op: unsupported ranks number!");
}
```

**NRanks 必须是模板参数**（编译期常量），因为 `LamportComm<NRanks>` 用了 `uint8_t* data_bufs[NRanks]` 数组，且 warp shuffle 的 active_mask 依赖 NRanks。所以只支持 2/4/8/16 四种 TP size。

### 10.4 DType 与 float4 路径分发

```cpp
template <int NRanks>
void dispatch_dtype(MiniMaxReduceRMSParams const& params) {
  bool use_float4 = (params.allreduce_in_k != nullptr) &&
                    (params.hidden_dim * params.nranks == 6144) &&
                    (params.hidden_dim_k * params.nranks == 1024);

  if (params.dtype == at::ScalarType::Half) {
    if (use_float4) minimax_reduce_rms_kernel_launcher_float4<half, NRanks, 6144, 1024>(params);
    else            minimax_reduce_rms_kernel_launcher<half, NRanks>(params);
  } else if (params.dtype == at::ScalarType::BFloat16) {
    if (use_float4) minimax_reduce_rms_kernel_launcher_float4<__nv_bfloat16, NRanks, 6144, 1024>(params);
    else            minimax_reduce_rms_kernel_launcher<__nv_bfloat16, NRanks>(params);
  } else if (params.dtype == at::ScalarType::Float) {
    if (use_float4) minimax_reduce_rms_kernel_launcher_float4<float, NRanks, 6144, 1024>(params);
    else            minimax_reduce_rms_kernel_launcher<float, NRanks>(params);
  } else {
    TORCH_CHECK(false, "Unsupported data type for minimax_reduce_rms_op");
  }
}
```

- **float4 路径触发条件**：K 输入存在 + Q 全维度 = 6144 + K 全维度 = 1024（MiniMax-M2 shape）。
- 否则走标量 Q-only kernel（通用 fallback）。

---

## 11. Host 入口 minimax_allreduce_rms_qk

```cpp
std::tuple<torch::Tensor, torch::Tensor> minimax_allreduce_rms_qk(
    torch::Tensor qkv, torch::Tensor const& norm_weight_q,
    torch::Tensor const& norm_weight_k, torch::Tensor workspace,
    int64_t const q_size, int64_t const kv_size, int64_t const rank,
    int64_t const nranks, double const eps) {
  TORCH_CHECK(qkv.dim() == 2, "minimax_allreduce_rms_qk: qkv must be 2D");
  TORCH_CHECK(qkv.is_contiguous(), "minimax_allreduce_rms_qk: qkv must be contiguous");
  int64_t qkv_dim = qkv.size(-1);
  TORCH_CHECK(qkv_dim == q_size + 2 * kv_size, ...);
  TORCH_CHECK(rank < nranks, ...);

  int64_t num_tokens = qkv.size(0);
  int elem_bytes = qkv.element_size();

  torch::Tensor q_out = torch::empty({num_tokens, q_size}, qkv.options());
  torch::Tensor k_out = torch::empty({num_tokens, kv_size}, qkv.options());

  auto params = vllm::tensorrt_llm::MiniMaxReduceRMSParams();
  params.nranks = static_cast<int>(nranks);
  params.rank = static_cast<int>(rank);
  params.dtype = qkv.scalar_type();
  params.size_q = static_cast<int>(num_tokens * q_size);
  params.hidden_dim = static_cast<int>(q_size);
  params.size_k = static_cast<int>(num_tokens * kv_size);
  params.hidden_dim_k = static_cast<int>(kv_size);
  params.stride_q = static_cast<int>(qkv_dim);  // q 行 stride = q_size + 2*kv_size
  params.stride_k = static_cast<int>(qkv_dim);
  params.stride_q_out = 0;  // 输出 contiguous
  params.stride_k_out = 0;
  params.workspace = reinterpret_cast<void**>(workspace.mutable_data_ptr());

  uint8_t* base = static_cast<uint8_t*>(qkv.data_ptr());
  params.allreduce_in   = base;                          // Q 起始
  params.allreduce_in_k = base + q_size * elem_bytes;   // K 起始（跳过 Q）
  params.rms_gamma = norm_weight_q.data_ptr();
  params.rms_gamma_k = norm_weight_k.data_ptr();
  params.rms_eps = static_cast<float>(eps);
  params.stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  params.rms_norm_out = q_out.mutable_data_ptr();
  params.rms_norm_out_k = k_out.mutable_data_ptr();

  vllm::tensorrt_llm::minimax_reduce_rms_op(params);
  return {q_out, k_out};
}
```

### 关键点

- **qkv 张量布局**：`[num_tokens, q_size + 2*kv_size]`，Q 在前 `q_size` 列，K 在接下来 `kv_size` 列（再接下来 `kv_size` 是 V，但本算子不用 V）。
- **`stride_q = qkv_dim`**：Q 的行 stride 等于整个 qkv 最后一维（因为 Q 是 qkv 的前 q_size 列，下一行的 Q 起点要跳过整个 qkv_dim）。
- **`allreduce_in_k = base + q_size * elem_bytes`**：K 的起点 = Q 起点偏移 `q_size` 个元素。
- **输出 `q_out [num_tokens, q_size]` 和 `k_out [num_tokens, kv_size]`**：contiguous 布局（stride_out=0 表示用 hidden_dim）。

---

## 12. 跨卡 AllReduce 实现机制详解

### 12.1 为什么不用 NCCL？

NCCL AllReduce 是一个独立的 kernel launch，需要：
1. 算完本地 variance → 写到中间 buffer
2. 启动 NCCL kernel 做跨卡 AllReduce
3. 再启动下一个 kernel 用 AllReduce 结果做 RMSNorm

3 次 kernel launch + 2 次中间 buffer 读写，延迟很大。**Lamport 协议把 AllReduce 融合进 RMSNorm kernel 内部**，跨卡通信通过共享 buffer + volatile 自旋等待完成，省掉 2 次 launch 和中间 buffer。

### 12.2 Lamport 三缓冲协议

**核心思想**：每张卡预先分配一个 triple-buffer（3 个槽位），所有卡的 buffer 通过 CUDA IPC（或本实现中的 peer access）相互可见。每张卡的 kernel：
1. **广播写**：把自己的 variance 写到所有 NRanks 张卡的 buffer（包括自己的）。
2. **聚合读**：从**自己的 buffer** 读所有 NRanks 的 variance，自旋等待直到所有 rank 都写入。

**三缓冲的作用**：避免上一轮的数据还没被读完就被下一轮覆盖。`flag_value % 3` 决定本轮用哪个槽，`(flag_value + 2) % 3` 是上一轮用过的槽（本轮清零），`(flag_value + 1) % 3` 是下一轮将用的槽（当前不被任何 kernel 读写）。这样三轮轮转，读-写-清分离，无冲突。

### 12.3 Lamport 通信的详细数据流

假设 NRanks=4，token_id=0，tot_tokens=512：

**写入阶段**（每个 rank 的 thread 0 执行）：

```
rank 0: 写 data_bufs[0][0*512+0], data_bufs[1][0*512+0],
              data_bufs[2][0*512+0], data_bufs[3][0*512+0] = rank 0 variance
rank 1: 写 data_bufs[0..3][1*512+0] = rank 1 variance
rank 2: 写 data_bufs[0..3][2*512+0] = rank 2 variance
rank 3: 写 data_bufs[0..3][3*512+0] = rank 3 variance
```

**读取阶段**（每个 rank 的 thread 0 自旋等待）：

```
rank 0 读 data_bufs[0] 的 [0*512+0], [1*512+0], [2*512+0], [3*512+0]
       → 拿到所有 rank 的 variance
rank 1 读 data_bufs[1] 的相同槽位 → 拿到所有 rank 的 variance
...
```

**关键点**：每张卡只需要从**自己的 buffer** 读，不需要跨卡读 peer 的 buffer。所有跨卡的写入在 Lamport 协议下都是写到 peer 的 buffer（"广播写"），读只读自己的 buffer（"聚合读"）。这样避免了跨卡读的延迟（跨卡读延迟更高）。

### 12.4 两种并行 AllReduce 模式

#### 标量 Q-only Kernel 的串行模式

```cpp
if (threadIdx.x == 0) {
  for (int r = 0; r < NRanks; ++r) {
    data_bufs[r][(rank * tot_tokens) + token_id] = sum_variance;  // 串行写 NRanks 个 buffer
  }
}
// 自旋读 NRanks 次
while (!done) {
  for (int r = 0; r < NRanks; ++r) {
    vars_all_ranks[r] = ld_global_volatile(...);
  }
}
// 串行加 NRanks 次
for (int r = 0; r < NRanks; ++r) sum_variance += vars_all_ranks[r];
```

只由 thread 0 串行处理 NRanks 个 rank，简单但慢。

#### Float4 QK Kernel 的并行模式

```cpp
if (tid < NRanks) {
  // 并行 push: NRanks 个 lane 各写一个 rank 的 buffer
  data_bufs[tid][(rank * tot_groups * 2) + (2 * g)] = ...;
  // 并行 pull: NRanks 个 lane 各读一个 rank 的贡献
  var_all_ranks = ld_global_volatile(&data_bufs[rank][(tid * tot_groups * 2) + ...]);
  // warp shuffle 归约: NRanks 个 lane 蝴蝶归约，每个 lane 都拿到总和
  local_warp_reduce_sum_array<NRanks, float, 4>(..., kQActiveMask);
}
```

前 NRanks 个 lane 并行处理 NRanks 个 rank，然后 warp shuffle 蝴蝶归约（比串行加法快 log2(NRanks) 倍）。这是性能优化的关键。

### 12.5 跨卡可见性保证

**`ld_global_volatile` + `__threadfence()`** 是跨卡可见性的关键：
- **`volatile`** 强制每次重新从 global memory 读，不允许编译器缓存到寄存器。
- **`__threadfence()`** 在 volatile 读之前插入 GPU 内存栅栏，保证本 GPU 之前的写都已经对其他 thread 可见。
- **CUDA IPC / peer access** 保证跨卡的 global memory 是同一物理地址空间，`__threadfence()` 的可见性可以延伸到 peer GPU。

### 12.6 轮转推进协议

```
时间轴：
  iter 0 (flag=0)     iter 1 (flag=1)     iter 2 (flag=2)     iter 3 (flag=0)
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │ 本轮数据槽 0  │    │ 本轮数据槽 1  │    │ 本轮数据槽 2  │    │ 本轮数据槽 0  │
  │ 上轮清理槽 2  │    │ 上轮清理槽 0  │    │ 上轮清理槽 1  │    │ 上轮清理槽 2  │
  │ 下轮预留槽 1  │    │ 下轮预留槽 2  │    │ 下轮预留槽 0  │    │ 下轮预留槽 1  │
  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

- 本轮开始：所有 block 读 `flag_value`，使用 `flag_value % 3` 槽做数据通信。
- 本轮结束：block 0 thread 0 等所有 block 完成（`counter == gridDim.x`），然后 `flag = (flag+1) % 3`，把本轮写入字节数记到 `clear_size` 供下一轮清理用。
- 下一轮：所有 block 读新的 `flag_value`，本轮清理上一轮的槽（`clear_offset = (flag+2) % 3`）。

---

## 13. RMSNorm 实现详解

### 13.1 RMSNorm 数学公式

对每个 token 的 hidden_dim 维向量 `x`：

```
variance = sum(x²) / hidden_dim
rms = sqrt(variance + eps)
out = x / rms * weight = x * rsqrt(variance + eps) * weight
```

### 13.2 跨卡场景下的 hidden_dim 处理

每张卡只持有 `x` 的分片（`hidden_dim = OriginQDim / NRanks`），但 variance 需要全维度求和。所以：
- **本地 variance**：`sum(x_shard²)`，仅本卡分片
- **跨卡 AllReduce**：所有卡的本地 variance 求和 = 全维度 variance
- **RMSNorm**：`rsqrt(sum_variance / OriginQDim + eps) * weight`

代码中 `sum_variance / params.hidden_dim / NRanks` 等价于 `sum_variance / OriginQDim`（因为 `hidden_dim = OriginQDim / NRanks`）。float4 版直接用模板参数 `OriginQDim` 做编译期除法，更精确。

### 13.3 计算精度

```cpp
for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
  vals[i] = static_cast<DType>(
      static_cast<float>(vals[i]) *
      rsqrtf((sum_variance / hidden_dim / NRanks) + eps) *
      static_cast<float>(norm_weight[i]));
}
```

- `static_cast<float>(vals[i])`：bf16 提升到 fp32 做乘法，避免 bf16 累加误差。
- `rsqrtf`：硬件快速倒数平方根。
- 最终 `static_cast<DType>` 截断回 bf16。

### 13.4 编译期 Dim 优化

```cpp
template <int Dim>
__device__ __forceinline__ float rms_rsqrt(float& v, float eps) {
  constexpr float kInvDim = 1.0F / static_cast<float>(Dim);  // 编译期常量
  v = rsqrtf((v * kInvDim) + eps);
  return v;
}
```

`Dim = 6144` 或 `1024` 是编译期常量，`1/Dim` 在编译期算好为 `kInvDim`，运行时只做一次乘法 + 一次 `rsqrtf`，比运行时除法快。

### 13.5 4-token 并行处理（float4 版）

float4 QK kernel 每次处理 4 个 token：
- `warp_sum_variance[4]`：4 个 token 各一个 variance。
- 跨卡 AllReduce 时把 4 个 variance 打包成一个 float4 一起传输，**减少 Lamport buffer 读写次数 4 倍**。
- RMSNorm 时 4 个 token 各算各的 scale，互不干扰。

这是性能优化的关键：把"4 个 token × NRanks 个 rank"的 AllReduce 转换成"1 个 float4 × NRanks 个 rank"，吞吐量提升 4 倍。

---

## 14. 整体流程图

### 14.1 高层流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Host: minimax_allreduce_rms_qk                 │
│  - 解析 qkv 张量布局 [num_tokens, q_size + 2*kv_size]              │
│  - 构造 MiniMaxReduceRMSParams                                      │
│  - 分发到 minimax_reduce_rms_op                                     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              minimax_reduce_rms_op: 按 NRanks 分发                  │
│   NRanks=2 → dispatch_dtype<2>                                      │
│   NRanks=4 → dispatch_dtype<4>                                      │
│   NRanks=8 → dispatch_dtype<8>                                      │
│   NRanks=16 → dispatch_dtype<16>                                    │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              dispatch_dtype: 选 float4 vs 标量 kernel                │
│   if (K input存在 && q_size*N==6144 && kv_size*N==1024):           │
│       → minimax_reduce_qk_rms_kernel_lamport_float4 (MiniMax-M2)   │
│   else:                                                              │
│       → minimax_reduce_rms_kernel_lamport (通用 fallback)           │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GPU Kernel 执行                                   │
│  每个 block 处理一组 token (标量: 1 token; float4: 4 tokens)       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 1. 构造 LamportComm                                          │  │
│  │    - 读 flag_value, 算 data_offset = flag % 3                │  │
│  │    - 准备所有 rank 的 buffer 指针                            │  │
│  │    - atomicAdd(counter, 1)  ← block 间同步信号               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 2. 本地 variance 归约                                        │  │
│  │    - 读 float4 数据 (8 bf16)                                 │  │
│  │    - 每 thread 算 sum(x²)                                    │  │
│  │    - warp shuffle + shared memory 两阶段归约                  │  │
│  │    - block 内 lane 0 拿到 sum(x²)                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 3. 跨卡 AllReduce (Lamport 三缓冲协议)                       │  │
│  │    ┌────────────────────────────────────────────────────┐   │  │
│  │    │ Push: 本 rank 的 variance 写到所有 NRanks 的 buffer │   │  │
│  │    │   (标量: thread 0 串行写; float4: NRanks 个 lane  │   │  │
│  │    │    并行写, float4 打包 4 个 token)                   │   │  │
│  │    └────────────────────────────────────────────────────┘   │  │
│  │    ┌────────────────────────────────────────────────────┐   │  │
│  │    │ Pull: 从本 rank buffer 自旋读所有 NRanks variance   │   │  │
│  │    │   - ld_global_volatile + __threadfence              │   │  │
│  │    │   - is_neg_zero(-0.0f) 哨兵检测                      │   │  │
│  │    │   - 所有 rank 都写入后退出                          │   │  │
│  │    └────────────────────────────────────────────────────┘   │  │
│  │    ┌────────────────────────────────────────────────────┐   │  │
│  │    │ Reduce: NRanks 个 variance 求和                     │   │  │
│  │    │   (标量: 串行加; float4: warp shuffle 蝴蝶归约)     │   │  │
│  │    └────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 4. RMSNorm 计算                                              │  │
│  │    - scale = rsqrt(sum_var / OriginQDim + eps)              │  │
│  │    - out = x * scale * norm_weight                          │  │
│  │    - bf16 → fp32 算乘法, 截断回 bf16                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 5. 写出 + 清理 + 轮转推进                                    │  │
│  │    - 写 rms_norm_out / rms_norm_out_k                       │  │
│  │    - 清理上一轮槽位 (填 -0.0f 哨兵)                         │  │
│  │    - block 0 thread 0:                                      │  │
│  │        while(counter != gridDim.x) {} // 等所有 block 完成  │  │
│  │        flag = (flag+1) % 3                                   │  │
│  │        clear_size = 本轮写入字节数                          │  │
│  │        counter = 0                                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.2 Lamport 三缓冲轮转图

```
                buffer slot 0      buffer slot 1      buffer slot 2
              ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
  iter 0      │   ★ 读+写       │ │   预留          │ │   清理上一轮   │
  (flag=0)    │   本轮数据      │ │   (不动)        │ │   (填 -0.0f)   │
              └─────────────────┘ └─────────────────┘ └─────────────────┘
  iter 1      │   清理上一轮   │ │   ★ 读+写       │ │   预留          │
  (flag=1)    │   (填 -0.0f)   │ │   本轮数据      │ │   (不动)        │
              └─────────────────┘ └─────────────────┘ └─────────────────┘
  iter 2      │   预留          │ │   清理上一轮   │ │   ★ 读+写       │
  (flag=2)    │   (不动)        │ │   (填 -0.0f)   │ │   本轮数据      │
              └─────────────────┘ └─────────────────┘ └─────────────────┘
  iter 3      │   ★ 读+写       │ │   预留          │ │   清理上一轮   │
  (flag=0)    │   本轮数据      │ │   (不动)        │ │   (填 -0.0f)   │
              └─────────────────┘ └─────────────────┘ └─────────────────┘
                       ↑                                          ↑
                  下一轮 flag 推进后, 本轮的"读+写"槽变成"清理"槽
```

### 14.3 跨卡通信数据流图（NRanks=4，float4 版）

```
              ┌─────────── rank 0 buffer ───────────┐
              │ slot [0*groups*2 + 2g]   = rank0 var │  ← rank0 写
              │ slot [1*groups*2 + 2g]   = rank1 var │  ← rank1 写
              │ slot [2*groups*2 + 2g]   = rank2 var │  ← rank2 写
              │ slot [3*groups*2 + 2g]   = rank3 var │  ← rank3 写
              └────────────────────────────────────────┘
                                ↑
                                │ rank 0 自旋读
                                │ (lane 0..3 各读一个 rank)
              ┌─────────── rank 1 buffer ───────────┐
              │ slot [0*groups*2 + 2g]   = rank0 var │  ← rank0 写
              │ slot [1*groups*2 + 2g]   = rank1 var │  ← rank1 写
              │ slot [2*groups*2 + 2g]   = rank2 var │  ← rank2 写
              │ slot [3*groups*2 + 2g]   = rank3 var │  ← rank3 写
              └────────────────────────────────────────┘
                                ↑
                                │ rank 1 自旋读
                                │ (lane 0..3 各读一个 rank)
              ... (rank 2, rank 3 同理)

  并行 Push:                      并行 Pull + Warp AllReduce:
  rank 0 lane 0 ──┐                lane 0 读 rank0 var ──┐
  rank 0 lane 1 ──┤                lane 1 读 rank1 var ──┤ shuffle
  rank 0 lane 2 ──┤                lane 2 读 rank2 var ──┤ xor 归约
  rank 0 lane 3 ──┘                lane 3 读 rank3 var ──┘
    写 rank 0..3                    每个 lane 都拿到 sum(var0..3)
                                    lane 0 算 rsqrt
```

---

## 15. MACA 平台适配注意事项

### 15.1 `ld_global_volatile` 的 MACA 修复

原 vLLM 代码用 `((volatile int*)(addr))[0]`，在 MACA cucc 下被错误编译，读到脏寄存器值，导致 `is_neg_zero` 永远返回 false，自旋循环立即退出，输出 NaN。**修复方法是改用 `*((volatile float*)addr)`**——这是反复实验验证后找到的可行写法。

详见 `op/vllm/minimax_reduce_rms_kernel.cu` L138-149 的注释。

### 15.2 sm_90+ 特性的条件编译

所有 sm_90+ 特性（`cooperative_groups::cluster_group`、`griddepcontrol` PTX）都用 `#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))` 包裹，sm_80（MACA C500/C280）走 `#else` 分支，退化为经典 `blockIdx.x` / `threadIdx.x`。

### 15.3 跨卡执行的现实限制

Lamport 协议要求**所有 rank 的 kernel 真正并发执行**，否则自旋等待会死锁。原 vLLM 设计使用 `torch.multiprocessing.spawn` + NCCL + CUDA IPC，每个进程绑定一个 GPU，4 张卡真正并行。

**MACA 平台的限制**：
1. **CUDA IPC 完全坏**：`cudaIpcOpenMemHandle` 在 MACA 上直接返回 `mcErrorInvalidValue`，跨进程 IPC 不可用。
2. **单 GPU 多 stream 不并发**：MACA C280 上即使 1-block 的小 kernel 也会被序列化（实测 speedup 0.91x~1.03x），无法让 4 个 rank 的 kernel 真正并发执行 → spin-wait 死锁。
3. **跨 GPU 写不可见**：多 GPU + peer access + `__threadfence_system()` 也无法让 cuda:1 的写对 cuda:0 的 `ld_global_volatile` 可见（实测 1 亿次循环都没看到写）。

因此本算子在 MACA 上单进程下无法正常工作。生产环境需要通过多进程 + NCCL + 替代 IPC 机制（如共享主机内存映射到 device 地址空间）来实现真正的并发执行，这部分由上层框架（vLLM-metax）的 `LamportWorkspace` 模块处理。

### 15.4 NRanks 模板参数的编译期约束

NRanks 必须是 2/4/8/16 之一，因为是 `LamportComm<NRanks>` 的模板参数。这要求上层调用方在调用 `minimax_reduce_rms_op` 之前确定 TP size，且 TP size 不能动态变化。这是 vLLM-TRT-LLM 设计的硬约束，不是 MACA 特有。

---

## 附录：源文件清单

| 文件 | 行数 | 作用 |
|---|---|---|
| `op/vllm/minimax_reduce_rms_kernel.cu` | 876 | kernel 实现 + launcher + host 入口 |
| `op/vllm/minimax_reduce_rms_kernel.h` | 79 | `MiniMaxReduceRMSParams` 结构 + `kElemsPerAccess` 模板 |
| `op/vllm/torch_bindings.cpp` | - | pybind11 注册 `minimax_allreduce_rms_qk` op |

### 关键函数索引

| 函数 | 行号 | 作用 |
|---|---|---|
| `LamportComm<NRanks>::LamportComm` | L40-59 | 构造 Lamport 通信器，读 flag、准备 buffer 指针、atomicAdd counter |
| `LamportComm<NRanks>::update` | L61-69 | 推进 flag 轮转，重置 counter，记录 clear_size |
| `is_neg_zero(float)` | L80-82 | 检测 -0.0f 哨兵 |
| `get_neg_zero()` | L89-96 | 返回全 -0.0f 的 float4 |
| `rms_rsqrt<Dim>(float&, float)` | L98-103 | 编译期 Dim 的 RMS rsqrt |
| `ld_global_volatile(float*)` | L138-149 | MACA 修复后的 volatile 加载 |
| `warpReduceSumV2<T, NUM>` | L157-166 | warp 蝴蝶归约 |
| `blockReduceSumV2<T, NUM>` | L168-192 | block 两阶段归约 |
| `local_warp_reduce_sum_array<kNumThreads, T, ArraySize>` | L195-208 | float4 版局部 warp 归约 |
| `IndexHelper<DType>` | L220-248 | 索引映射（sm_80 / sm_90 双路径） |
| `minimax_reduce_rms_kernel_lamport<DType, NRanks>` | L261-343 | 标量 Q-only kernel（通用 fallback） |
| `minimax_reduce_qk_rms_kernel_lamport_float4<DType, NRanks, QDim, KDim>` | L352-636 | Float4 QK 融合 kernel（MiniMax-M2 专用） |
| `minimax_reduce_rms_kernel_launcher<DType, NRanks>` | L675-694 | 标量 kernel launcher |
| `minimax_reduce_rms_kernel_launcher_float4<DType, NRanks, QDim, KDim>` | L696-747 | float4 kernel launcher |
| `dispatch_dtype<NRanks>` | L749-782 | DType + float4 路径分发 |
| `minimax_reduce_rms_op` | L785-797 | NRanks 分发 |
| `minimax_allreduce_rms` | L800-826 | host 入口（Q-only） |
| `minimax_allreduce_rms_qk` | L828-876 | host 入口（Q+K 融合） |
