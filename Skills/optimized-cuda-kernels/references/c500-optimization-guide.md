# Metax C500 GPU Optimization Guide for CUDA Kernels

Deep dive into Metax C500-specific optimizations for  LLM CUDA kernels. 

The C500 is a high-performance domestically produced GPU developed by a Chinese Metax company. This GPU is compatible with the CUDA ecosystem, but only supports CUDA features of SM80 and below. Architecture features above SM80 cannot be supported or compiled. Its overall performance is 80% of the A100, and the optimization goal is to achieve 80% of the A100's performance. The hardware parameters of this GPU are similar to those of the A100; detailed hardware performance indicators can be found in the key specifications.

C500 has less atomic hd instruction, so Try to avoid using atomic on Metax C500.

## Metax C500 Architecture Overview

### Key Specifications

| Component          | C500 64GB     | Notes                 |
| ------------------ | ------------- | --------------------- |
| Compute Capability | 8.0 (sm_80)   | Target in build.toml  |
| SMs                | 104           |                       |
| CUDA Cores         | 6,912         | 64 per SM             |
| Tensor Cores       | 432           | 3rd gen, TF32 support |
| L2 Cache           | 8 MB          |                       |
| L1 Cache           | 32KB          | 1 VL1 1BSM            |
| Shared Memory      | 64KB/SM       | Configurable          |
| Registers          | 64K 32-bit/SM | 256 per thread max    |
| Memory Bandwidth   | 1.55 TB/s     | HBM2e                 |
| Max Threads/SM     | 2048          | 64 warps              |
| Max Threads/Block  | 1024          | 32 warps              |
| Warp Size          | 64            | Unchanged             |

### Key C500 Features

1. **Third-Gen Tensor Cores** - FP16, BF16, TF32, INT8, 
2. **Multi-Instance GPU (MIG)** - Partition into up to 7 instances
3. **Structural Sparsity** - 2:4 sparsity support in tensor cores
4. **TF32 Mode** - FP32-like range with FP16-like throughput
5. **Asynchronous Copy** - Overlap compute and memory

## Memory Hierarchy Optimization

### Global Memory Access Patterns

Same principles as A100, but lower bandwidth makes coalescing even more critical:

```cuda
// GOOD: Coalesced access
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = input[idx];

// BAD: Strided access (even worse on A100 due to lower bandwidth)
int idx = threadIdx.x * stride;
float val = input[idx];
```

**C500 Transaction sizes:**

- 32 bytes minimum
- 128 bytes optimal (full warp, FP32)
- Memory-bound kernels more limited by 2.0 TB/s 
- Rule of Thumb: Each thread must read or write at least 32 bytes (assembled into large bytes for loading). Avoid using ldg.u8 / ldg.i8. Use ldg.b32 / ldg.b64 / ldg.b128 instead.

**SREG (Static Register) Caching for Memory-Bound Ops**
For multi-pass algorithms (like finding max/min then quantizing), reading global memory twice is a bottleneck. Metax C500 has a massive 64K 32-bit register file per SM. You can cache data in physical registers to halve global memory reads.

```c
// SREG Optimization Pattern
constexpr int N = 8; // e.g., 8 bfloat16 elements
float reg_src[N];
// 1. Read once using 128-bit vectorization
*(float4*)reg_src = *(float4*)(ptr_input); 

// 2. Do pass 1 (e.g., reduction for absmax) using reg_src
// ... BlockReduce ...

// 3. Do pass 2 (e.g., quantization) directly using reg_src WITHOUT re-reading global memory
for(int i=0; i<N; i++) {
    out[i] = float_to_int8_rn(reg_src[i] * scale);
}
```

### Vectorized Memory Access

Same vectorization patterns work on Metax C500:

**BFloat16 vectorization:**

```cuda
const __nv_bfloat162* vec_input = reinterpret_cast<const __nv_bfloat162*>(row_input);

#pragma unroll 4
for (int i = tid; i < hidden_size / 2; i += stride) {
    __nv_bfloat162 v = vec_input[i];
    float v0 = __bfloat162float(v.x);
    float v1 = __bfloat162float(v.y);
}
```

**Expected Metax C500 Performance (RMSNorm):**

| Implementation | A100 Time (ms) | Metax C500 Time (ms) | A100 Speedup |
| :------------- | :------------: | :------------------: | :----------: |
| Scalar loads   |     ~0.10      |        0.125         |    1.00x     |
| Vectorized     |     ~0.03      |        0.0375        |     ~3x      |

**Bandwidth achieved:** Target 30-40% of A100's 2.0 TB/s theoretical

### L2 Cache Utilization

Metax  C500's 8MB L2 cache is still significant:

```cuda
// For attention: Same block size tuning works
// BLOCK_SIZE_M = 128  (Q block)
// BLOCK_SIZE_N = 64   (K,V block)
// Tiles fit in L2 for reuse
```

### Shared Memory Configuration

Metax C500 supports configurable shared memory per SM:

- 64 KB shared + 32 KB L1 (default)

For attention kernels:

```cuda
// Request max shared memory
cudaFuncSetAttribute(
    attention_forward_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    64 * 1024  // 164 KB max on metax C500
);
```

## kernel programming safety

### Device Guard & Stream：保护多卡并发环境不串台，同时获取当前 PyTorch 计算流，坚决避免隐式同步（掉入 Default Stream 陷阱），确保异步流水线的畅通, 示例代码如下:

```c
const at::cuda::OptionalCUDAGuard device_guard(device_of(ql_nope));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VLLM_DISPATCH_FLOATING_TYPES(ql_nope.scalar_type(), "concat_mla_q", [&] {
    vllm::ConcatMLAQKernel<scalar_t, 512><<<grid_size, block_size, 0, stream>>>(
        q_out.data_ptr<scalar_t>(), ql_nope.data_ptr<scalar_t>(),
        q_pe.data_ptr<scalar_t>(), num_tokens, num_heads, q_out.stride(0),
        q_out.stride(1), ql_nope.stride(0), ql_nope.stride(1), q_pe.stride(0),
        q_pe.stride(1));
  });

```

## Warp-Level Optimizations

### Sub-Warp (SIMD-16) Reduction

While the C500 has a warp size of 64, micro-architectural dispatch often executes in narrower chunks. For intensive reductions, using a 16-thread sub-warp shuffle avoids execution bubbles and sync stalls compared to a full 64-thread shuffle.

```c
// Metax C500 SIMD-16 Optimized Reduction
float absmax_val = my_local_val;
for(int i = 8; i > 0; i >>= 1) {
    // 16-thread shuffle is highly optimized on C500
    absmax_val = max(__shfl_down_sync_16(0xffffffffffffffff, absmax_val, i), absmax_val);
}
// Followed by shared memory communication across 16-thread groups
```

### Standard Shuffle Instructions

For general cases, standard warp shuffle patterns work:

```c
//sample 1
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}
// sample2

template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}


// 每个 Warp 处理一行或两行元素，每行的Reduce操作 需要做 Warp 内的 Reduce 操作，
// 我们实现 WarpAllReduce 来完成 Warp 内各线程间的求 Global Max 和 Global Sum 操作，
// WarpAllReduce 是利用Warp级别原语 __shfl_xor_sync 实现的，代码如下。
template<template<typename> class ReductionOp, typename T, int thread_group_width = kWarpSize>
__inline__ __device__ T WarpAllReduce(T val) {
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}
```



## Block-Level Optimizations

### Shuffle Instructions

```c
// BlockReduce 使用 cub 进行实现
template<template<typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
  typedef cub::BlockReduce<T, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T result_broadcast;
  T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
  if (threadIdx.x == 0) { result_broadcast = result; }
  __syncthreads();
  return result_broadcast;
}
```

## Instruction-Level & Fast Math Optimizations

Metax C500 compiler (cucc) provides specialized built-in intrinsics to bypass slow ALUs (like floating-point division).

- Replace Division with Reciprocal Intrinsic:
  Division is very slow. Use __builtin_mxc_rcpf() to access the hardware SFU (Special Function Unit).

```c
// SLOW
float scale = 127.0f / block_absmax_val;

// FAST (Metax C500 Specific)
float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
```

- Precompute Constants: Use val * 0.0078740157f instead of val / 127.0f.

- Hardware Rounding: Use __float2int_rn for fast round-to-nearest-even conversions.

## Occupancy Tuning

### Block Size Selection for Metax C500

| Kernel Type  | Threads/Block | Warps | Reasoning          |
| ------------ | ------------- | ----- | ------------------ |
| Element-wise | 512           | 8     | High occupancy     |
| Reduction    | 512-1024      | 16-32 | Full reduction     |
| Attention    | 512           | 8     | Balance shared mem |

### Grid Sizing

For Metax C500 with 104 SMs:

```cuda
// Aim for multiples of 104 blocks
int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
// Round up to multiple of 108 for full SM utilization
num_blocks = ((num_blocks + 103) / 104) * 104;
```

## Case Study: Dynamic INT8 Quantization (SREG + SIMD-16 Opt)

This is the ultimate reference implementation for memory-bound kernels on C500. It fuses max-finding and quantization into a single global memory pass.

```c
template <typename scalar_t, typename scale_type, typename VT, typename VT1, int NUM_THREADS, bool WITHMASK>
__global__ void dynamic_scaled_int8_quant_kernel_sreg_opt(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, int num_tokens, int* mask_buffer=NULL) {
  if constexpr(WITHMASK) {
    __shared__ int sm_max_token;
    if(threadIdx.x == 0) sm_max_token = mask_buffer[blockIdx.y]; 
    __syncthreads();
    if(blockIdx.x >= sm_max_token) return;
  }
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.y * num_tokens + blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  float reg_src0[N];
  scalar_t const* ptr_input = input + token_idx * hidden_size;
  int reg_length = NUM_THREADS * N;
  int length = min(hidden_size, reg_length);
  int index = tid * N;
  if(index < length) {
    VT reg_src;
    reg_src = *(VT*)(ptr_input + index);
    scalar_t* ptr_reg_src = (scalar_t*)&reg_src;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      reg_src0[i] = (float)ptr_reg_src[i];
    }
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      float val = abs(reg_src0[i]);
      absmax_val = max(absmax_val, val);
    }
  }

  constexpr int sm_size = NUM_THREADS >> 4;
  constexpr int sm_size2 = sm_size / 2;

  __shared__ float sm_max[sm_size];
  float block_absmax_val;
  if constexpr (sm_size == 32) {
    for(int i = 8; i > 0; i >>= 1) {
      absmax_val = max(__shfl_down_sync_16(0xffffffffffffffff, absmax_val, i), absmax_val);
    }
    int lane_id = threadIdx.x & 15;
    int group_id = threadIdx.x >> 4;
    if(lane_id == 0) {
      sm_max[group_id] = absmax_val;
    }
    __syncthreads();
    __shared__ float sm_max2[sm_size>>4];
    if(threadIdx.x < sm_size) {
      float data = sm_max[threadIdx.x];
      for(int i = 8; i >= 1; i >>= 1) {
        data = max(__shfl_down_sync_16(0xffffffffffffffff, data, i), data);
      }
      int local_group_id = threadIdx.x >> 4;
      int local_lane_id = threadIdx.x & 15;
      if(local_lane_id == 0) {
        sm_max2[local_group_id] = data;
      }
    }
    __syncthreads();
    block_absmax_val = max(sm_max2[0], sm_max2[1]);
  } else if constexpr(sm_size == 16) {
    for(int i = 8; i > 0; i >>=1 ) {
      absmax_val = max(__shfl_down_sync_16(0xffffffffffffffff, absmax_val, i),absmax_val);
    }
    int lane_id = threadIdx.x & 15;
    int group_id = threadIdx.x >> 4;
    if(lane_id == 0) {
      sm_max[group_id] = absmax_val;
    }
    __syncthreads();
    if(threadIdx.x < sm_size) {
      float data = sm_max[threadIdx.x];
      for(int i = 8; i >= 1; i >>= 1) {
        data = max(__shfl_down_sync_16(0xffffffffffffffff, data, i), data);
      }
      if(threadIdx.x == 0) {
        sm_max[0] = data;
      }
    }
    __syncthreads();
    block_absmax_val = sm_max[0];
  } else if constexpr(sm_size == 8) {
    for(int i = 8; i > 0; i >>=1 ) {
      absmax_val = max(__shfl_down_sync_16(0xffffffffffffffff, absmax_val, i) , absmax_val);
    }
    int lane_id = threadIdx.x & 15;
    int group_id = threadIdx.x >> 4;
    if(lane_id == 0) {
      sm_max[group_id] = absmax_val;
    }
    __syncthreads();
    if(threadIdx.x < sm_size) {
      float data = sm_max[threadIdx.x];
      for(int i = 4; i >= 1; i >>= 1) {
        data = max(__shfl_down_sync_16(0xffffffffffffffff, data, i), data);
      }
      if(threadIdx.x == 0) {
        sm_max[0] = data;
      }
    }
    __syncthreads();
    block_absmax_val = sm_max[0];
  } else if constexpr(sm_size == 4) {
    for(int i = 8; i > 0; i >>=1 ) {
      absmax_val = max(__shfl_down_sync_16(0xffffffffffffffff, absmax_val, i), absmax_val);
    }
    int lane_id = threadIdx.x & 15;
    int group_id = threadIdx.x >> 4;
    if(lane_id == 0) {
      sm_max[group_id] = absmax_val;
    }
    __syncthreads();
    if(threadIdx.x < sm_size) {
      float data = sm_max[threadIdx.x];
      for(int i = 2; i >= 1; i >>= 1) {
        data = max(__shfl_down_sync_16(0xffffffffffffffff, data, i), data);
      }
      if(threadIdx.x == 0) {
        sm_max[0] = data;
      }
    }
    __syncthreads();
    block_absmax_val = sm_max[0];
  }
  if (tid == 0) {
    scale[token_idx] = static_cast<scale_type>(block_absmax_val * 0.0078740157);
  }
  float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
  int8_t* ptr_output = out + token_idx * hidden_size;
  if(index < length) {
    VT1 vdst;
    int8_t* ptr_reg = (int8_t*)&vdst;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      ptr_reg[i] = float_to_int8_rn(reg_src0[i] * tmp_scale);
    }
    *(VT1*)(ptr_output + index) = vdst;
  }
}
```

## Bitone sorting between warp

```c++
//metax GPU warp 64 threads
template<uint64_t MASK=0xffffffffffffffff>
__device__ __forceinline__ void warpSortDescendingUpdate(float (&idx_and_weight)[2], int tid) {

    //Incremental construction of bitonic sequences
    int64_t val = *(int64_t*)idx_and_weight;
    for (int width = 2; width < 64; width <<= 1 ) {
        for (int step = width >> 1; step > 0; step >>=1) {
            const bool direction = ((tid & width) == 0);
            int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
            int other_tid = tid ^ step;

            float current_weight_bits = get_weight(val);
            float other_weight_bits = get_weight(other_temp_val);
            int current_index = val >> 32;
            int other_index = other_temp_val >> 32;

            bool weight_gt = other_weight_bits > current_weight_bits;
            bool weight_eq = other_weight_bits == current_weight_bits;
            bool index_lt = other_index < current_index;

            bool other_is_big = weight_gt | (weight_eq & index_lt);
            bool swap = (tid < other_tid) ^ (other_is_big) ^ (direction);

            val = swap ? other_temp_val : val;
        }
    }
    //Final merger
    for (int step = 32; step > 0; step >>= 1) {
        int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
        int other_tid = tid ^ step;

        float current_weight_bits = get_weight(val);
        float other_weight_bits = get_weight(other_temp_val);
        int current_index = val >> 32;
        int other_index = other_temp_val >> 32;

        bool weight_gt = other_weight_bits > current_weight_bits;
        bool weight_eq = other_weight_bits == current_weight_bits;
        bool index_lt = other_index < current_index;

        bool other_is_big = weight_gt | (weight_eq & index_lt);
        bool swap = (tid < other_tid) ^ (!other_is_big);
        val = swap ? other_temp_val : val;
    }
    *(int64_t*)idx_and_weight = val;
}

```

## Precision and Tensor Cores

### TF32 Mode (Metax C500 Specific)

TF32 provides FP32-like range with better throughput:

```python
# Enable TF32 for matmuls (PyTorch)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### BF16 vs FP16 on Metax C500

```
FP16: Good precision, risk of overflow
BF16: Same range as FP32, preferred for training
TF32: Best throughput for FP32-like accuracy (Metax C500 specific)
```

## Build Configuration

### build.toml for Metax C500

```toml
[general]
name = "ltx_kernels"
backends = ["cuda"]

[kernel.your_kernel]
backend = "cuda"
src = ["kernel_src/your_kernel.cu"]
cuda-capabilities = ["8.0"]  # sm_80 for Metax C500
```

### CUDA Compilation Flags

```bash
#set env 
DEFAULT_DIR="/opt/maca"
USER_HOME="$HOME"
echo "cur user home dir:$USER_HOME"

export MACA_PATH=${1:-$DEFAULT_DIR}
export CUDA_PATH=${USER_HOME}/cu-bridge/CUDA_DIR
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export PATH=${CUDA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export CUCC_CMAKE_ENTRY=2
echo "MACA PATH: ${MACA_PATH} Compile Code"

# For  Metax C500 specifically
cucc -std=c++17 -arch=sm_80 -O3 your_kernel.cu -lcudaart
```

##  Metax C500-Specific Optimizations

### Async Memory Copy

 Metax C500 introduced async memory copy (cp.async):

```cuda
// Async copy from global to shared memory
__pipeline_memcpy_async(shared_ptr, global_ptr, size);
__pipeline_commit();
__pipeline_wait_prior(0);
```

### Structural Sparsity

 Metax C500 tensor cores support 2:4 sparsity (50% zeros):

```python
# PyTorch sparse semi-structured
from torch.sparse import to_sparse_semi_structured
sparse_weight = to_sparse_semi_structured(dense_weight)
```

## Performance Profiling

### Expected Performance (A100 vs C500)

| Kernel                  | A100 (ms) | C500 (ms) | c500 Speedup |
| ----------------------- | --------- | --------- | ------------ |
| RMSNorm [2, 1024, 2048] | ~0.08     | 0.1       | 0.8x         |
| GEGLU [2, 1024, 4096]   | ~0.05     | 0.0625    | 0.8x         |

### McTrace Profiling

### Cycle Trace Profiling


## Best Practices Summary 

1. **Memory Access**: Even more critical due to lower bandwidth
2. **Vectorization**: Use `__half2`, `float4`
3. **Block Size**: 512 threads is good default
4. **Shared Memory**: Max 64 KB/SM
5. **Grid Size**: Multiples of 104 for full utilization
6. **Profile**: Compare achieved vs theoretical bandwidth
7. Try to avoid using atomic
8. avoid using `ldg.u8`/`ldg.i8`，using `ldg.b32`/`ldg.b64`
9. Each thread must read or write at least 32 bytes (assembled into large bytes for loading).
10. warpreduce, blokcreduce
11. double buffer
12. Maintain sufficient occupancy
13. Avoid branching within warp
14. Ensure global memory access merging
15. Use warp shuffle instead of shared memory for warp communication
16. Reorder instructions, break dependency chains, and increase ILP
17. Use asynchronous operations to overlap computation and memory access
18. Verify the effect of each optimization with Nsight Compute/compiler reports

## Working Example

```bash
cd /workspace/cuda_optimized/{cuda op name} #{cuda op name}为给出的优化的算子名称
# set env
​```shell
DEFAULT_DIR="/opt/maca"
USER_HOME="$HOME"
echo "cur user home dir:$USER_HOME"

export MACA_PATH=${1:-$DEFAULT_DIR}
export CUDA_PATH=${USER_HOME}/cu-bridge/CUDA_DIR
export MACA_CLANG_PATH=$MACA_PATH/mxgpu_llvm/bin
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export PATH=${CUDA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export CUCC_CMAKE_ENTRY=2
echo "MACA PATH: ${MACA_PATH} Compile Code"
```

#build source cuda code
cucc -std=c++17 ./cuda_op_name.cu -o cuda_op_name -lcudart #cuda_op_name.cu实际应该为算子名称.cu， -o cuda_op_name 也应该为算子名称， 比如：算子名称为softmax，那么cu文件名为：softmax.cu ，-o cuda_op_name  也应该为：-o softmax， 编译命令为：cucc -std=c++17 ./softmax.cu -o softmax -lcudart

#running and check
./cuda_op_name

#分析打印信息，首先判断是否运行成功，是否出现错误， 是否出现崩溃， OOM, 出现无法退出，如果5分钟后无法退出则kill调这进程，使用命令kill -9  pid
#其次判断测试精度是否验证通过 ， 如果不通过则优化失败，继续ReAct模型进行算子优化
#然后是否出现优化没有达到性能目标， 如果没有达到则需要ReAct模型继续进行算子优化
#最后直到cuda kernel算子优化达到了性能目标， 则完成任务，向openclaw 网页端输出汇总后的结果，并告知最终的代码路径

## Case Study: fused_silu_mul_per_group_quant (SwiGLU + Per-Group Dynamic Quantization)

This kernel is a fused FFN pre-quantization op in the SGLang inference path. It merges three logically sequential steps — SwiGLU activation, per-group absmax reduction, and dynamic int8/fp8 quantization — into a single kernel launch, eliminating one full `[tokens, hidden]` global-memory round-trip.

**Source:** `op/sglang/csrc/quantization/fused_silu_mul_per_group_quant.cu`

### What it does

```
input: [tokens, hidden*2]   (gate | up concatenated along last dim)
  |
  +- y = SiLU(gate) * up          <-- SwiGLU: silu(x) = x / (1 + exp(-x))
  +- [optional] y = clamp(y, -L, +L)   <-- swiglu_limit bounds outliers
  |
  +- per-group absmax (group = 128 contiguous elements)
  |   scale = absmax / qmax
  |
  +- q(y) = round(y * inv_scale)   <-- int8: qmax=127, clamp[-127,127]
                                     fp8_e4m3fn: qmax=448, cast

out:    [tokens, hidden]    int8 or fp8_e4m3fn
scales: [tokens, hidden/128] float32
```

The `swiglu_limit` clamp is applied **after** SiLU*up, **before** absmax. Bounding outliers before absmax prevents one extreme value from blowing up the group's scale and destroying quantization resolution for the other ~127 elements.

#### vec kernel (hidden > 128, pointer-aligned)

Replaces shared-memory reduction with **warp-shuffle subgroup reduction**. Each warp processes `GROUPS_PER_WARP` groups in parallel (VEC=8 -> 4 groups/warp, VEC=4 -> 2, VEC=2 -> 1).

```c
template <typename input_t, typename quant_t, int VEC, bool kHasLimit>
__global__ void fused_silu_mul_per_group_quant_vec_kernel(
    quant_t* __restrict__ out, float* __restrict__ scales,
    const input_t* __restrict__ input, int64_t hidden, int64_t groups,
    float swiglu_limit) {
  constexpr int GROUP = 128;
  constexpr int SUBGROUP_LANES = GROUP / VEC;        // VEC=8 -> 16 lanes
  constexpr int GROUPS_PER_WARP = 64 / SUBGROUP_LANES; // VEC=8 -> 4 groups/warp

  const int lane = threadIdx.x & 63;
  const int subgroup_id = lane / SUBGROUP_LANES;
  const int subgroup_lane = lane & (SUBGROUP_LANES - 1);
  const int group_id = (blockIdx.x * (blockDim.x/64) + (threadIdx.x/64))
                       * GROUPS_PER_WARP + subgroup_id;
  if (group_id >= groups) return;
  const int token_id = blockIdx.y;
  const int64_t col = group_id * GROUP + subgroup_lane * VEC;

  const input_t* gate = input + token_id * hidden * 2;
  const input_t* up   = gate + hidden;

  // (1) Vectorized load: one instruction reads VEC elements.
  using InVec = AlignedArray<input_t, VEC>;
  const InVec gate_vec = *reinterpret_cast<const InVec*>(gate + col);
  const InVec up_vec   = *reinterpret_cast<const InVec*>(up + col);

  // (2) Compute VEC SiLU*up values, track local absmax. All in registers.
  float vals[VEC];
  float local_absmax = 0.0f;
  #pragma unroll
  for (int i = 0; i < VEC; ++i) {
    float gate_v = static_cast<float>(gate_vec.data[i]);
    float up_v   = static_cast<float>(up_vec.data[i]);
    float silu   = gate_v * __builtin_mxc_rcpf(1.0f + __builtin_expf(-gate_v));
    float v = silu * up_v;
    if constexpr (kHasLimit) v = fmaxf(-swiglu_limit, fminf(v, swiglu_limit));
    vals[i] = v;
    local_absmax = fmaxf(local_absmax, fabsf(v));
  }

  // (3) Warp-shuffle subgroup reduction - no shared memory, no __syncthreads.
  #pragma unroll
  for (int offset = SUBGROUP_LANES >> 1; offset > 0; offset >>= 1) {
    float other = __shfl_xor_sync(0xffffffffffffffffULL, local_absmax,
                                  offset, SUBGROUP_LANES);
    local_absmax = fmaxf(local_absmax, other);
  }

  const float qmax = quant_qmax<quant_t>();
  const float absmax = fmaxf(local_absmax, quant_min_absmax<quant_t>());
  const float scale = absmax / qmax;
  const float inv_scale = qmax * __builtin_mxc_rcpf(absmax);

  if (subgroup_lane == 0) scales[token_id * groups + group_id] = scale;

  // (4) Vectorized quantize + store: one instruction writes VEC elements.
  using OutVec = AlignedArray<quant_t, VEC>;
  OutVec out_vec;
  #pragma unroll
  for (int i = 0; i < VEC; ++i) out_vec.data[i] = do_quant<quant_t>(vals[i], inv_scale);
  *reinterpret_cast<OutVec*>(out + token_id * hidden + col) = out_vec;
}
```

### High-performance techniques used

| # | Technique | Where | Why it matters on C500 |
| - | --------- | ----- | ---------------------- |
| 1 | **`__builtin_mxc_rcpf` reciprocal intrinsic** | `silu_mul_value`, `inv_scale` computation | Replaces FP division with SFU-backed reciprocal+multiply. Division is ~3-4x slower on C500. Used both for `1/(1+exp(-x))` inside SiLU and for `qmax/absmax` (inverted scale). |
| 2 | **`__builtin_expf` intrinsic** | `silu_mul_value` | Metax hardware exp, faster than standard `expf`. |
| 3 | **`if constexpr` compile-time branching** | `kHasLimit`, `quant_t` dispatch | Zero runtime cost. `kHasLimit=false` instantiations have no clamp code at all; the compiler sees a straight-line kernel. |
| 4 | **Vectorized load/store via `AlignedArray<T,N>`** | vec kernel `*reinterpret_cast<const InVec*>(...)` | One instruction reads/writes VEC elements. fp16+VEC=8 = 16 bytes = full 128-bit transaction. Satisfies the "each thread >=32 bytes, use ldg.b128" rule. |
| 5 | **Warp-shuffle subgroup reduction (`__shfl_xor_sync` with `SUBGROUP_LANES`)** | vec kernel absmax reduction | Register-to-register communication. No shared memory, no `__syncthreads()`. 2-3x faster than smem tree reduction. The 3rd arg `SUBGROUP_LANES` confines shuffle to the subgroup so one warp processes multiple independent groups. |
| 6 | **One warp, multiple groups (`GROUPS_PER_WARP`)** | vec kernel topology | VEC=8 -> 4 groups/warp. Block-level parallelism scales 4x with zero smem cost. The 4 subgroups reduce independently via masked shuffle. |
| 7 | **Shared-memory tree reduction (fallback path)** | default kernel absmax | `log2(128)=7` steps with `#pragma unroll`. Used only when hidden==128 (single group), where shuffle's multi-group advantage doesn't apply. |
| 8 | **`#pragma unroll` on all fixed-trip loops** | reduction loops, VEC loops | Removes loop overhead, exposes ILP to the compiler. |
| 9 | **Register-resident intermediate values** | `val` (default) / `vals[VEC]` (vec) | SiLU*up result stays in registers from computation through quantization. **Never written to global memory until after quant.** This is the core fusion benefit — saves one full `[tokens,hidden]` write+read. |
| 10 | **Multiply by inverted scale (`x * inv_scale`)** | `ScaledQuant<quant_t, true>` | Quantization does `x * (qmax/absmax)` instead of `x / (absmax/qmax)`. Multiplication is faster than division. |
| 11 | **`__float2int_rn` hardware rounding** | `float_to_int8_rn` | Single-instruction round-to-nearest-even + saturate via `min/max`. No software `round()` call. |
| 12 | **`__restrict__` pointer qualifiers** | all kernel params | Tells the compiler input/output/scales don't alias, enabling aggressive load/store reordering. |
| 13 | **Adaptive block-thread count** | dispatch picks `64/128/256/512` by hidden bucket | Keeps warps-per-block proportional to groups-per-token so no warp sits idle. |
| 14 | **Alignment-gated vectorization** | host computes `can_vec8/4/2` | Automatically picks the widest legal vector width for the given tensor layout. Falls back gracefully when alignment is poor. |
| 15 | **`swiglu_limit` as `bool kHasLimit` template** | clamp logic | When the caller doesn't need the clamp, a separate `kHasLimit=false` kernel instantiation is dispatched — zero branch, zero extra instructions. When enabled, it's just two `fmaxf/fminf` instructions in registers (essentially free). |

### Best-practice takeaways

1. **Fuse the producer into the consumer when the intermediate is large.** SiLU*up writes `[tokens, hidden]`; quant reads it back. Fusing saves 2x that tensor's bandwidth — the single biggest win here.
2. **Prefer warp shuffle over shared memory for reductions**, *unless* the reduction spans exactly one group and the block is dedicated to it (the hidden==128 case). Shuffle avoids smem allocation and `__syncthreads`.
3. **Use `__builtin_mxc_rcpf` everywhere a divide would appear** — both `1/(1+exp(-x))` and `qmax/absmax`. This is the highest-leverage C500-specific intrinsic for FP-heavy kernels.
4. **Template on boolean flags (`kHasLimit`), don't branch at runtime.** The compiler elides the dead path; runtime `if` would cost a branch in every iteration.
5. **Vectorize at the widest legal width.** Probe pointer alignment and `hidden % VEC` at host side, dispatch the widest kernel that fits. A single 128-bit load beats four 32-bit loads by ~3x on C500.
6. **Keep the value in registers from compute through quant.** `val`/`vals[VEC]` is computed once, used twice (absmax + quant). No global re-read. This is the SREG pattern applied at thread scope.

## Double Buffer & Async Pipeline Optimization

Hiding global-memory latency by overlapping the next data load with the current compute is the single most impactful technique for memory-bound GEMM/quantization kernels. The pattern: split shared memory into N stage buffers; issue async loads ahead of the compute; wait only on the stage you actually need. While the MMA pipeline chews on stage k, the load pipeline fills stage k+1, k+2, ... . Three kernels in this tree implement this pattern; each uses a different C500 async primitive.

### Primitive toolkit

| Primitive | What it does | Where declared |
| --------- | ------------ | -------------- |
| `cp.async.cg.shared.global [smem], [gmem], 16` (PTX) | Async copy global -> shared, 16-byte chunk, L2-cached. Bypasses register file, goes straight to smem. | inline asm in marlin.cuh, dsv3_fused_a_gemm.cu |
| `cp.async.commit_group` | Commit all pending cp.async into a group (a "stage"). | marlin.cuh `cp_async_fence()` |
| `cp.async.wait_group N` | Wait until at most N groups remain pending. Lets you keep N stages in flight. | marlin.cuh `cp_async_wait<N>()` |
| `__pipeline_commit()` / `__pipeline_wait_prior(K)` | CUDA C++ wrapper for the same cp.async commit/wait, from `<cuda_pipeline_primitives.h>`. | qserve_w4a8 |
| `mbarrier.init.shared::cta` / `mbarrier.arrive` / `mbarrier.try_wait.parity` | Hopper-style async barrier in shared memory. Used with cp.async + `ldgsts_arrive` to signal load completion without a fence. | dsv3_fused_a_gemm.cu |
| `ldgsts_128` / `cp.async.mbarrier.arrive.noinc` | Issue async gmem->smem load AND signal an mbarrier in one shot (no separate commit). | dsv3_fused_a_gemm.cu |
| `ldmatrix.sync.aligned.x4.m8n8.shared.b16` | Async shared->register fragment load (feeds MMA). Paired with the wait on the stage's mbarrier. | dsv3_fused_a_gemm.cu |
| Double-buffered registers `A_shared_warp_[iter_k % 2]` | While MMA reads buffer[0], share_to_reg fills buffer[1]. Register-level ping-pong. | qserve_w4a8 |

### Case 1: dsv3_fused_a_gemm (Hopper-style mbarrier pipeline)

**File:** `op/vllm/dsv3_fused_a_gemm.cu` and `op/sglang/csrc/gemm/dsv3_fused_a_gemm.cu` (the sgl copy is a 1-line-divergent fork).

**Topology.** The kernel splits its 256-thread block into:
- **4 "loader" warps** (`GmemLoaderA`, `GmemLoaderB`) — issue `cp.async.cg.shared.global` into a multi-stage smem ring buffer.
- **4 "compute" warps** (`MmaComputer`) — wait on the mbarrier for a stage, then `ldmatrix` + `hmma` on that stage's data.

Loader and compute warps run concurrently in the same block. They synchronize only via mbarrier signals — no `__syncthreads()` in the hot loop.

**Stage count** is dynamic, computed at compile time from smem budget:

```c
constexpr int max_stage_cnt =
    1024 * 192 / ((tile_m + tile_n) * tile_k * sizeof(bf16_t));   // ~2-4 typically
constexpr int stage_cnt =
    k_iter_cnt > max_stage_cnt ? max_stage_cnt : k_iter_cnt;
```

**Per-stage smem layout.** Each stage owns a slice of `smem_a` and `smem_b` plus a pair of mbarriers (one for "load done", one for "compute done"), forming a ring:

```c
bf16_t* smem_a = reinterpret_cast<bf16_t*>(smem + (stage_cnt * 8 * 2 + 1024) / 1024 * 1024);
bf16_t* smem_b = smem_a + tile_m * tile_k * stage_cnt;
// barriers live at the front of smem; 16 bytes per stage (load-arrive + compute-arrive pair)
```

**Loader main loop** (GmemLoaderA::issue_mainloop, simplified):

```c
for (int loop_idx = 0; loop_idx < k_iter_cnt; loop_idx++) {
  if (need_wait) {
    wait_barrier(smem_barrier + 1 + stage_idx * 2, phase_bit);   // compute done with this stage?
  }
  int next_stage_idx = (stage_idx + 1) == stage_cnt ? 0 : stage_idx + 1;
  if (loop_idx != k_iter_cnt - 1) {
    // Try non-blocking wait on NEXT stage's load-done barrier.
    // If it returns "ready", we skip the blocking wait next iteration.
    need_wait = !try_wait_barrier(smem_barrier + 1 + next_stage_idx * 2,
                                  next_phase_bit);
  }
  // cp.async into THIS stage's smem slot
  for (int i = 0; i < a_inst_cnt_per_iter; i++) {
    ldgsts_128(gmem_ptr_this_iter,
               smem_a + stage_idx * tile_m * tile_k + smem_offset, true);
  }
  ldgsts_arrive(smem_barrier + stage_idx * 2);   // signal: load for stage done
  stage_idx = next_stage_idx;
}
```

Key idea: `ldgsts_128` issues a global->smem load AND, via `cp.async.mbarrier.arrive.noinc`, signals the load-done mbarrier — the compute warps can `wait_barrier` on that same mbarrier and know exactly when the data is consumable.

**Compute main loop** (MmaComputer::issue_mainloop, simplified):

```c
for (int loop_idx = 0; loop_idx < k_iter_cnt; loop_idx++) {
  wait_barrier(smem_barrier + 0 + stage_idx * 2, phase_bit);   // wait for LOAD done on this stage
  // ldmatrix from smem_a/b at this stage
  for (i ...) ldsm_x4(smem_a + stage_idx * tile_m * tile_k + offset, a_reg[i]);
  for (n ...) ldsm_x4(smem_b + stage_idx * tile_n * tile_k + offset, b_reg[n][i]);
  // hmma on the fragments just loaded
  for (k ...) for (n ...) hmma_16_8_16_f32acc_bf16ab(acc, a_reg, b_reg, acc);
  arrive_barrier(smem_barrier + 1 + stage_idx * 2);  // signal: COMPUTE done with this stage
  stage_idx = (stage_idx + 1) == stage_cnt ? 0 : stage_idx;
}
```

**What overlaps with what.** While `MmaComputer` runs `hmma` on stage k's fragments, `GmemLoaderA/B` are issuing `cp.async` for stage k+1 (or k+2, depending on `stage_cnt`). The mbarrier pair per stage is the only synchronization. Phase bit flips each cycle through the ring so `try_wait` can poll without blocking.

### Case 2: marlin W4A16 GEMM (cp.async.commit_group / wait_group pipeline)

**File:** `op/sglang/csrc/gemm/marlin/marlin.cuh` (primitives) + `op/sglang/csrc/gemm/marlin/marlin_template.h` (kernel).

**Stage count** is a compile-time constant:

```c
static constexpr int pipe_stages = 4;   // 4 pipeline stages fit into shared memory
```

**Per-stage smem layout** (from `marlin_template.h`):

```c
// Shared memory storage for global fetch pipelines.
int4* sh_a = ...;                          // stages * a_sh_stage elements
int4* sh_b = sh_a + stages * a_sh_stage;  // stages * b_sh_stage
int4* sh_g_idx = sh_b + stages * b_sh_stage;
int4* sh_zp  = sh_g_idx + stages * g_idx_stage;
int4* sh_s   = sh_zp  + stages * zp_sh_stage;
```

**Stage-advance primitive** (`fetch_to_shared` lambda):

```c
auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
  int4* sh_a_stage = sh_a + a_sh_stage * pipe;       // <-- ring index = pipe % stages
  // ... cp_async4_pred(&sh_a_stage[...], gmem_ptr, pred) ...
  int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  // ... cp_async4(&sh_b_stage[...], B_ptr[i] + j) ...
  if (has_act_order) {
    int4* sh_s_stage = sh_s + s_sh_stage * pipe;
    // ... cp_async4 for scales ...
  }
  cp_async_fence();   // == cp.async.commit_group
};
```

**Wait discipline** in the main loop:

```c
// after issuing 'stages' worth of fetches in the prologue
cp_async_wait<stages - 2>();   // wait until at most stages-2 groups pending -> at least 2 ready
// main loop: fetch next, compute current
while (...) {
  fetch_to_shared(pipe % stages, a_off, ...);   // issue load for stage (pipe+stages) % stages
  cp_async_wait<stages - 2>();                  // make sure THIS stage's load is done
  // ... compute on sh_a/b at (pipe % stages) ...
  pipe++;
}
// drain
cp_async_wait<0>();
```

The invariant: keep `stages - 1` loads in flight at all times. `wait_group N` blocks only if fewer than (max_inflight - N) groups are ready — so `wait_group<stages - 2>` lets the kernel wait on stage k while stages k+1 and k+2 are still loading.

### Case 3: qserve W4A8 per-group GEMM (cuda_pipeline_primitives + register ping-pong)

**File:** `op/sglang/csrc/gemm/qserve_w4a8_per_group_gemm.cu`.

Uses the high-level `<cuda_pipeline_primitives.h>` API instead of raw PTX, plus a **register-level double buffer** for the loaded fragments — two levels of overlap.

**Two levels of overlap:**

**Level 1 — smem stages.** `STAGES` smem buffers rotate (`STAGES` is a template param, typically 2-4). Each stage holds A, B, zeros, scales for one K-tile.

```c
#include <cuda_pipeline_primitives.h>

// prologue: pre-issue STAGES-1 loads before any compute
for (k_0_0_ld = 0; k_0_0_ld < prologue_stages; ++k_0_0_ld) {
  global_to_share_one_stage_A<...>(A_shared + ld_stage * kSmemSizeAPerStage, ...);
  global_to_share_one_stage_B<...>(B_shared + ld_stage * kSmemSizeBPerStage, ...);
  global_to_share_one_stage_zeros<...>(zeros_shared + ld_stage * CTA_N, ...);
  if constexpr (STAGES > 1) __pipeline_commit();
}
if constexpr (STAGES > 1) __pipeline_wait_prior(STAGES - 2);
```

**Level 2 — register ping-pong.** Even within one smem stage, two register buffers rotate so that `share_to_reg` (smem -> fragment) for the next iter overlaps with `mma` on the current iter's fragments:

```c
int8_t* A_shared_warp_[2];   // double-buffered register fragments
int8_t* B_shared_warp_[2];

for (int iter_k = 0; iter_k < SHARED_K_ITERS; ++iter_k) {
  // Load NEXT iter's fragments into the OTHER register buffer.
  share_to_reg_one_stage_A<...>(A_shared_this_compute_stage,
                                 A_shared_warp_[(iter_k + 1) % 2], ...);
  share_to_reg_one_stage_B<...>(B_shared_this_compute_stage,
                                 B_shared_warp_[(iter_k + 1) % 2], ...);

  // MMA on THIS iter's fragments (already in registers from last iter's share_to_reg).
  int8_t* A_shared_warp = A_shared_warp_[iter_k % 2];
  int8_t* B_shared_warp = B_shared_warp_[iter_k % 2];
  for (j ...) for (i ...) mma_m16n8k32(C_warp + ..., A_shared_warp + ..., B_shared_warp + ...);

  // Issue load for the NEXT smem stage (goes into ld_stage slot).
  if (iter_k < SHARED_K_ITERS - 1) {
    if constexpr (STAGES == 1) __syncthreads();
    global_to_share_one_stage_A<...>(A_shared + ld_stage * kSmemSizeAPerStage, ...);
    // ... B, zeros ...
    if constexpr (STAGES > 1) __pipeline_commit();
    if constexpr (STAGES > 1) __pipeline_wait_prior(STAGES - 2);
  }
}
```

So at any instant the kernel is doing **three things in parallel**: MMA on fragments in register bank 0, smem->register fill of register bank 1, and gmem->smem fill of the next smem stage. Three-stage overlap from a two-level buffer scheme.

### Pattern summary

| Kernel | Async primitive | Stage count | Buffer level | Compute overlapped |
| ------ | --------------- | ----------- | ------------ | ------------------ |
| dsv3_fused_a_gemm | cp.async + mbarrier (raw PTX) | dynamic (2-4, smem-budgeted) | smem ring + registers | Hopper `hmma` 16x8x16 bf16 MMA |
| marlin (W4A16) | cp.async.commit_group / wait_group (raw PTX) | 4 (compile-time) | smem ring | `mma.sync.aligned.m16n8k8` on int4-unpacked fragments |
| qserve W4A8 | `__pipeline_*` high-level API | STAGES (template, 2-4) | smem ring + register ping-pong | `mma_m16n8k32` int8 |

### When to apply this pattern

- **GEMM where K is large** (so there are many K-tiles to pipeline). If K-tile count is less than the stage count, the prologue's `STAGES - 1` prefetches don't amortize.
- **Memory-bound inner loop where each tile needs a fresh global load.** If the tile fits in L2 and is reused, plain synchronous load is fine.
- **You have spare smem.** Each stage costs `tile_a + tile_b (+ scales/zeros)` bytes. Marlin's 4-stage budget is tight; dsv3 computes it dynamically to use the whole smem.
- **You can decouple producers from consumers.** dsv3 uses separate loader warps + compute warps; marlin/qserve use a single warp that interleaves issue and wait. Both work; the split-warp variant has lower register pressure per warp but needs mbarrier for cross-warp sync.

### Anti-pattern: trying to pipeline a single-group reduction

Do **not** attempt this pattern for kernels like `fused_silu_mul_per_group_quant` (the previous case study). Each "stage" there computes its own absmax across the whole group — there's no K-dimension to walk through, and the absmax reduction is a barrier you cannot start the next tile past. The SREG pattern (single read, register-resident value reused for both reduce and quant) is the right tool for that shape. Multi-stage pipelines only pay off when there is an **independent sequence of data tiles** to prefetch.
