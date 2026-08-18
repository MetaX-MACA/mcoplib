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

