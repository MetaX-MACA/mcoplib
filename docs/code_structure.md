# mcOpLib 项目代码结构

## 一、项目定位

mcOpLib 是面向 LLM 推理框架的自定义算子库，为 MetaX GPU（MACA 架构）提供高性能 CUDA 内核。同时支持 vLLM、SGLang、LMDeploy 三大框架，并包含一套通用默认算子。

```
项目规模: 175 个 .cu 文件 / 12 个 .cpp 文件 / 138 个 .cuh/.h 头文件
```

## 二、顶层目录结构

```
mcoplib/                          ← 项目根目录
├── op/                           ← ★ 算子源码（核心，按框架分子目录）
│   ├── vllm/                     ← vLLM 算子（47 个 .cu）
│   ├── sglang/                   ← SGLang 算子（88 个 .cu）
│   ├── lmdeploy/                 ← LMDeploy 算子（3 个 .cu）
│   ├── cv/                       ← CV 视觉算子（独立 deb 包构建）
│   ├── pybind.cpp                ← 默认算子 pybind11 绑定
│   ├── *.cu                      ← 默认算子内核（30 个 .cu）
│   └── *.h                       ← 默认算子头文件
├── mcoplib/                      ← ★ Python 包（编译产物 + Python 封装）
│   ├── __init__.py               ← 版本检查 & MACA 兼容性校验
│   ├── version                   ← 构建时写入的版本信息
│   ├── *.so                      ← 编译后的扩展模块
│   ├── custom_ops.py             ← 自定义算子 Python 接口
│   ├── profiler.py               ← 性能分析工具
│   ├── quant_utils.py            ← 量化工具函数
│   ├── triton_fused_moe.py       ← Triton MoE 融合算子
│   └── triton_utils/             ← Triton 工具
├── include/                      ← 默认算子头文件（.h）
├── kernel/                       ← 默认算子内核头文件（.cuh/.h）
├── cmake/                        ← CMake 工具函数（utils.cmake）
├── CMakeLists.txt                ← ★ CMake 构建配置（核心）
├── setup.py                      ← ★ Python 构建入口（核心）
├── env.sh                        ← 通用 MACA 环境设置
├── env_local.sh                  ← 容器内 MACA 环境设置
├── requirements/                 ← Python 依赖
│   └── build.txt                 ← 构建时依赖
├── unit_test/                    ← 单元测试（70+ 测试文件）
├── benchmark/                    ← 性能基准测试（mxbench 框架）
├── docs/                         ← 项目文档
├── Skills/                       ← CUDA 优化技能参考
├── mxbench/                      ← mxbench 基准测试工具
├── .deps/                        ← FetchContent 下载的依赖
└── build/                        ← CMake 构建临时目录
```

## 三、算子源码结构（op/ 目录 — 项目核心）

### 3.1 总览

```
op/
├── pybind.cpp                    ← 默认算子 Python 绑定入口
├── [30 个 .cu]                   ← 默认算子 CUDA 内核
├── [对应 .h]                     ← 默认算子头文件
│
├── vllm/                         ← vLLM 子模块
│   ├── torch_bindings.cpp        ← vLLM _C 模块绑定 (TORCH_LIBRARY)
│   ├── [47 个 .cu]               ← vLLM 算子内核
│   ├── attention/                ← 注意力机制
│   ├── moe/                      ← MoE 算子
│   │   └── torch_bindings.cpp    ← _moe_C 模块绑定
│   ├── quantization/             ← 量化算子
│   │   ├── awq/                  ← AWQ 量化
│   │   ├── cutlass_w8a8/         ← W8A8 量化 GEMM
│   │   ├── fp4/                  ← FP4 量化
│   │   ├── fused_kernels/        ← 融合量化内核
│   │   ├── gguf/                 ← GGUF 格式
│   │   ├── gptq/                 ← GPTQ 量化
│   │   └── w8a8/                 ← W8A8 量化
│   ├── mamba/                    ← Mamba SSM
│   ├── cutlass_extensions/       ← CUTLASS 扩展
│   └── sparse/                   ← 稀疏算子
│
├── sglang/                       ← SGLang 子模块
│   ├── csrc/                     ← CUDA 源码
│   │   ├── common_extension.cc   ← sgl_kernel 绑定 (TORCH_LIBRARY_FRAGMENT)
│   │   ├── flash_extension.cc    ← Flash Attention 绑定
│   │   ├── flashmla_extension.cc ← Flash MLA 绑定
│   │   ├── spatial_extension.cc  ← Spatial 算子绑定
│   │   ├── moe/                  ← MoE 算子（最大子目录）
│   │   ├── attention/            ← 注意力机制
│   │   ├── elementwise/          ← 逐元素算子
│   │   ├── quantization/         ← 量化算子
│   │   ├── allreduce/            ← AllReduce 通信
│   │   ├── grammar/              ← 语法约束
│   │   ├── memory/               ← 内存管理
│   │   ├── kvcacheio/            ← KV Cache 传输
│   │   ├── speculative/          ← 推测解码
│   │   ├── gemm/                 ← GEMM 算子
│   │   ├── mamba/                ← Mamba SSM
│   │   ├── sgl_diffusion/        ← 扩散模型算子
│   │   └── spatial/              ← 空间算子
│   └── include/                  ← SGLang 头文件
│       ├── sgl_kernel_ops.h      ← ★ 核心算子声明（100+ 函数）
│       ├── sgl_flash_kernel_ops.h ← Flash 内核算子声明
│       └── [工具头文件]
│
├── lmdeploy/                     ← LMDeploy 子模块
│   └── ops/                      ← LMDeploy 算子
│       ├── pybind.cpp            ← Python 绑定
│       ├── attention/            ← 注意力机制
│       └── [3 个 .cu]            ← 内核文件
│
└── cv/                           ← CV 视觉算子（独立 deb 包构建）
    ├── csrc/
    └── include/
```

### 3.2 算子分类与功能映射

| 算子类别 | 默认算子 (op/) | vLLM | SGLang | 典型算子 |
|---------|--------------|------|--------|---------|
| **注意力** | - | paged_attention_v1/v2, merge_attn_states | cascade, fused_mla, vertical_slash_index | Paged Attention, MLA, Flash Attention |
| **MoE** | moe_swiglu_dq, moe_softmax_topk, moe_gather, moe_scatter | topk_softmax, grouped_topk, moe_align_sum | moe_sum_reduce, moe_fused_gate, grouped_gemm, moe_fused_w4a16 | TopK选择、MoE求和、分组GEMM |
| **归一化** | fused_rms_norm_dq, rms_norm_dynamic_per_token_quant | layernorm_kernels | fused_add_rms_norm, fused_layernorm_dynamic_per_*_quant | RMS Norm, LayerNorm |
| **量化** | int8_quant, scale_dynamic_quant, silu_mul_quant_fp8 | fp8/w8a8/int8/gptq/awq/gguf quant | fp8_quantize, int8_quant, fused_silu_mul_per_group_quant | FP8/W8A8/INT8/GPTQ/AWQ 量化 |
| **融合算子** | fused_rope, fused_bias_swiglu, fused_bias_gelu, fused_softplus_sqrt | fused_qknorm_rope, activation_kernels | dsv4_norm_rope, fused_rotary_emb, concat_mla | RoPE、SiLU+Mul、Gate融合 |
| **通信** | all_reduce | - | custom_all_reduce | AllReduce |
| **KV Cache** | store_kv, send/recv_to_attention_node | cache_kernels, reshape_and_cache | transfer_kv_*, store | KV Cache 读写/传输 |
| **推测解码** | - | - | speculative_sampling, eagle_utils, packbit | 推测采样 |
| **Mamba** | - | selective_scan_fwd | causal_conv1d | Mamba SSM |

### 3.3 默认算子（op/*.cu）完整清单

```
核心融合算子:
  fused_rope.cu             ← Rotary Position Embedding
  fused_bias_swiglu.cu     ← SwiGLU 激活 + 偏置
  fused_bias_gelu.cu       ← GELU 激活 + 偏置
  fused_bias_dropout.cu    ← Dropout + 偏置
  fused_repeat_kv.cu       ← Repeat KV for GQA
  fused_softplus_sqrt.cu   ← Softplus + Sqrt 激活
  fused_rms_norm_dq.cu     ← RMS Norm + Dynamic Quant
  fused_moe_gate_deepseek.cu  ← DeepSeek MoE Gate 融合
  fused_add_layernorm_per_token_quant_padding_output.cu
  fused_add_gemma_rmsnorm_per_token_quant_padding_output.cu
  fused_deepseekv4_qkv_rms_norm_rope.cu
  fused_split_qkv_gemma_rmsnorm_rope.cu

MoE 算子:
  moe_swiglu_dq.cu          ← MoE SwiGLU + Dynamic Quant
  moe_softmax_topk.cu       ← MoE Softmax TopK
  moe_gather.cu             ← MoE Gather
  moe_scatter_dynamic_quant.cu ← MoE Scatter + DQ
  moe_fused_gate_opt.cu     ← MoE Gate 优化版

量化算子:
  int8_quant_kernels.cu     ← INT8 量化
  scale_dynamic_quant.cu    ← Scale 动态量化
  silu_mul_quant_fp8_nopack.cu ← SiLU+Mul FP8 量化

位置编码:
  rotary_embedding.cu       ← RoPE
  rope_train.cu             ← RoPE 训练版

KV Cache:
  store_kv.cu               ← KV Cache 存储
  send_to_attention_node_pre_process.cu   ← 发送到注意力节点
  recv_from_attention_node_post_process.cu ← 从注意力节点接收

其他:
  all_reduce.cu             ← AllReduce 通信
  glm_attention_prepare.cu  ← GLM 注意力准备
  gptq_marlin.cu            ← GPTQ Marlin 量化
```

## 四、Python 包结构（mcoplib/ 目录）

### 4.1 包内容

```
mcoplib/
├── __init__.py               ← 导入时: 版本打印 + MACA 兼容性检查
├── version                   ← 构建版本信息（setup.py 写入）
│
├── 编译产物 (.so 扩展模块):
│   ├── op.cpython-310-x86_64-linux-gnu.so        ← 默认算子
│   ├── _C.abi3.so                                 ← vLLM 核心算子 (Stable ABI)
│   ├── _moe_C.abi3.so                             ← vLLM MoE 算子 (Stable ABI)
│   ├── sgl_kernel.cpython-310-x86_64-linux-gnu.so ← SGLang 核心算子
│   ├── sgl_grouped_gemm_cuda.*.so                 ← SGLang 分组 GEMM
│   ├── sgl_moe_fused_w4a16.*.so                   ← SGLang MoE W4A16 融合
│   ├── sgl_grouped_gemm_mctlass_int8.*.so         ← SGLang INT8 分组 GEMM
│   └── lmdeploy.cpython-310-x86_64-linux-gnu.so   ← LMDeploy 算子
│
├── Python 封装:
│   ├── custom_ops.py         ← 自定义算子 Python 接口
│   ├── profiler.py           ← GPU 性能分析器
│   ├── quant_utils.py        ← 量化工具函数
│   ├── fused_bias_dropout.py ← Bias+Dropout Python 封装
│   ├── fused_bias_swiglu.py  ← Bias+SwiGLU Python 封装
│   ├── fused_gelu.py         ← GELU Python 封装
│   ├── fused_repeat_kv.py    ← Repeat KV Python 封装
│   ├── fused_mla.py          ← MLA Python 封装
│   ├── fused_router_drop.py  ← Router Drop Python 封装
│   ├── triton_fused_moe.py   ← Triton MoE 融合算子
│   ├── triton_utils/         ← Triton 工具
│   └── marlin_*.py           ← Marlin 量化工具
```

### 4.2 Python 调用方式

```python
# 方式1: 通过 torch.ops 命名空间调用（最常用）
import mcoplib.sgl_kernel
torch.ops.sgl_kernel.moe_sum_reduce(input, output, scale)
torch.ops.sgl_kernel.per_token_cast_to_fp8(out, scale, x)

import mcoplib._C
torch.ops._C.silu_and_mul(out, input)

import mcoplib._moe_C
torch.ops._moe_C.topk_softmax(topk_weights, topk_indices, ...)

# 方式2: 通过 Python 封装调用
from mcoplib.fused_bias_swiglu import fused_bias_swiglu
result = fused_bias_swiglu(input, bias)
```

### 4.3 torch.ops 命名空间映射

| 扩展模块 | torch.ops 命名空间 | 注册方式 | 典型算子 |
|---------|-------------------|---------|---------|
| _C.abi3.so | `torch.ops._C` | TORCH_LIBRARY_EXPAND | paged_attention, rms_norm, silu_and_mul |
| _moe_C.abi3.so | `torch.ops._moe_C` | TORCH_LIBRARY_EXPAND | topk_softmax, moe_align_sum |
| sgl_kernel.so | `torch.ops.sgl_kernel` | TORCH_LIBRARY_FRAGMENT | fused_add_rmsnorm, moe_sum_reduce, per_token_cast_to_fp8 |
| op.so | `torch.ops.op` | pybind11 | fused_rope, moe_swiglu_dq |
| lmdeploy.so | 直接导入 | pybind11 | attention, rotary_embedding |

## 五、构建系统架构

### 5.1 构建入口与流程

```
setup.py (Python)
    │
    ├── 1. 前置检查: Python版本、MACA版本、依赖包
    ├── 2. 写入 version 文件
    ├── 3. 根据 BUILD_*_SUBMODULE 构建 ext_modules 列表
    ├── 4. 调用 cmake_build_ext:
    │       ├── configure()  → cmake_maca 配置（只执行一次）
    │       ├── build()      → cmake_maca --build --target=...
    │       └── install()    → cmake_maca --install --component=...
    └── 5. 复制 .so 到 mcoplib/ 目录

CMakeLists.txt (CMake)
    │
    ├── find_package(Python, pybind11, Torch)
    ├── GPU 架构检测与编译标志设置
    ├── 读取 BUILD_*_SUBMODULE 控制子模块
    ├── FetchContent(CUTLASS → MATLASS)
    │
    ├── vLLM 子模块:
    │     ├── define_gpu_extension_target(_C)
    │     └── define_gpu_extension_target(_moe_C)
    │
    ├── SGLang 子模块:
    │     ├── define_gpu_extension_target(sgl_kernel)
    │     ├── define_gpu_extension_target(sgl_grouped_gemm_cuda)
    │     ├── define_gpu_extension_target(sgl_moe_fused_w4a16)
    │     └── define_gpu_extension_target(sgl_grouped_gemm_mctlass_int8)
    │
    ├── LMDeploy 子模块:
    │     └── define_gpu_extension_target(lmdeploy)
    │
    └── 默认算子:
          └── define_gpu_extension_target(op)
```

### 5.2 子模块编译控制

```
BUILD_VLLM_SUBMODULE=ON/OFF        → _C.abi3.so, _moe_C.abi3.so
BUILD_SGLANG_SUBMODULE=ON/OFF      → sgl_kernel.so, sgl_grouped_gemm_cuda.so,
                                      sgl_moe_fused_w4a16.so,
                                      sgl_grouped_gemm_mctlass_int8.so
BUILD_LMDEPLOY_SUBMODULE=ON/OFF    → lmdeploy.so
BUILD_DEFAULT_OP_SUBMODULE=ON/OFF  → op.so (其他框架可能依赖)
```

## 六、代码分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Python 调用层                               │
│  torch.ops.sgl_kernel.xxx()  /  torch.ops._C.xxx()            │
│  mcoplib.fused_xxx.xxx()     /  mcoplib.custom_ops             │
├─────────────────────────────────────────────────────────────────┤
│                      Python 绑定层                               │
│  torch_bindings.cpp (TORCH_LIBRARY)                             │
│  common_extension.cc (TORCH_LIBRARY_FRAGMENT)                   │
│  pybind.cpp (PYBIND11_MODULE)                                   │
├─────────────────────────────────────────────────────────────────┤
│                      算子调度层                                  │
│  Host 函数: 输入检查 → 参数计算 → Grid配置 → Kernel启动          │
│  例: moe_sum_reduce() → 检查输入 → 选择kernel → <<<grid,block>>>│
├─────────────────────────────────────────────────────────────────┤
│                      GPU 内核层                                  │
│  .cu 文件: __global__ kernel 函数                               │
│  例: moe_sum_reduce_c500_kernel<WARPS, TOPK>                    │
│  技术点: 向量化加载(uint4)、BF16Vec8、warp级并行、grid-stride   │
├─────────────────────────────────────────────────────────────────┤
│                      工具/辅助层                                 │
│  include/*.h    ← 函数声明                                      │
│  kernel/*.cuh   ← 内核工具函数 (dispatch_utils, utils)          │
│  cmake/utils.cmake ← CMake工具函数                              │
│  Skills/        ← CUDA 优化参考                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 七、测试与基准

### 7.1 单元测试（unit_test/）

70+ 测试文件，覆盖所有主要算子。命名规则: `test_<算子名>.py`

```
分类:
  - 注意力: test_paged_attention_v1.py, test_fused_mla.py
  - MoE: test_moe_sum_reduce.py, test_moe_softmax_topk.py, test_grouped_topk.py
  - 归一化: test_rms_norm.py, test_fused_add_rms_norm.py
  - 量化: test_per_token_cast_to_fp8.py, test_int8_quant, test_fp8_*
  - 融合: test_fused_rope.py, test_fused_silu_mul_*.py
  - 位置编码: test_rope.py, test_dsv4_norm_rope.py
  - KV Cache: test_store_kv.py
  - 性能: test_profiler.py, benchmark_fp32_router_gemm.py
```

### 7.2 性能基准（benchmark/）

基于 mxbench 框架的性能测试，包含:
- C++ nvbench 基准测试
- Python 基准测试封装
- 自动化测试脚本 (testall.py)

## 八、依赖关系图

```
                    ┌─────────────┐
                    │  mcoplib    │ (Python 包)
                    └──────┬──────┘
                           │ import
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────┴──────┐ ┌──────┴──────┐ ┌──────┴──────┐
    │  vLLM 插件   │ │ SGLang 插件  │ │LMDeploy插件 │
    │ vllm-metax  │ │   sglang    │ │  lmdeploy   │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
           │    ┌──────────┼──────────┐    │
           ▼    ▼          ▼          ▼    ▼
    ┌──────────────────────────────────────────┐
    │           mcoplib 算子库                   │
    │  _C.so  _moe_C.so  op.so  sgl_kernel.so │
    │  lmdeploy.so  sgl_grouped_*.so           │
    └──────────────────┬───────────────────────┘
                       │ 依赖
           ┌───────────┼───────────┐
           │           │           │
    ┌──────┴──────┐ ┌──┴───┐ ┌────┴────┐
    │ PyTorch     │ │MATLASS│ │flashinfer│
    │ (CUDA扩展)  │ │(GEMM) │ │(Attention)│
    └─────────────┘ └──────┘ └─────────┘
           │
    ┌──────┴──────┐
    │  MACA SDK   │
    │  (编译器+驱动)│
    └─────────────┘
```

## 九、关键设计决策

### 9.1 为什么分成多个 .so 模块？
- **按框架隔离**: 不同框架的算子独立编译，避免符号冲突
- **可选编译**: 通过 BUILD_*_SUBMODULE 按需编译，加速开发循环
- **Stable ABI**: vLLM 模块使用 `abi3` 后缀，兼容多 Python 版本

### 9.2 为什么 vLLM 用 TORCH_LIBRARY 而 SGLang 用 TORCH_LIBRARY_FRAGMENT？
- `TORCH_LIBRARY` 定义一个全新的 ops 命名空间，用于独占注册
- `TORCH_LIBRARY_FRAGMENT` 允许多个编译单元向同一命名空间添加算子，SGLang 有多个 extension 文件

### 9.3 CUTLASS → MATLASS 替换
MACA SDK 提供了 CUTLASS 兼容层 MATLASS，位于 `$MACA_PATH/include`。构建时通过 FetchContent 将 CUTLASS 源码路径指向 MATLASS。

### 9.4 GPU 架构支持策略
- 每个 .cu 文件独立设置 gencode flags（`set_gencode_flags_for_srcs`）
- 避免全局设置导致所有文件编译所有架构，大幅减少编译时间
- MACA 额外添加 `--offload-arch=xcore1000/1500/1600` 支持 MetaX GPU
