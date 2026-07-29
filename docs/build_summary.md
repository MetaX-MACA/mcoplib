# mcOpLib 编译构建方式总结与流程图

## 一、项目概述

mcOpLib 是一个面向 LLM 推理框架（vLLM、SGLang、LMDeploy）的自定义算子库，提供优化的 CUDA/MACA GPU 内核。项目通过 Python setuptools + CMake 混合构建系统，将多个框架的算子编译为独立的 Python 扩展模块（.so 文件）。

## 二、构建环境要求

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| Python | 3.9 - 3.12 | 3.9+ 必须 |
| CMake | >= 3.26 | 容器中需用 /opt/cmake-3.30.4 |
| MACA SDK | >= 3.7.0 | MetaX GPU 编译工具链 |
| PyTorch | 2.6+ / 2.8+ | 提供 CUDA 扩展构建支持 |
| pybind11 | - | Python/C++ 绑定 |
| Ninja | - | 构建加速（可选） |
| sccache/ccache | - | 编译缓存（可选） |

## 三、环境变量设置

### env.sh（通用环境）
```bash
export MACA_PATH=/opt/maca
export CUDA_PATH=$HOME/cu-bridges/CUDA_DIR
export CUCC_PATH=$MACA_PATH/tools/cu-bridge
export PATH=$CUDA_PATH/bin:$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$CUCC_PATH/tools:$CUCC_PATH/bin:$PATH
export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/ompi/lib:$MACA_PATH/mxgpu_llvm/lib:$LD_LIBRARY_PATH
export CUCC_CMAKE_ENTRY=2
export ENABLE_BUILD_GPTQ_MARLIN_OP=1
```

### env_local.sh（容器内环境）
```bash
# 与 env.sh 类似，但 ENABLE_BUILD_GPTQ_MARLIN_OP=0
# 适配容器内路径
```

### 构建控制环境变量
| 变量 | 默认值 | 说明 |
|------|--------|------|
| BUILD_VLLM_SUBMODULE | ON | 编译 vLLM 算子（_C, _moe_C） |
| BUILD_SGLANG_SUBMODULE | ON | 编译 SGLang 算子（sgl_kernel 等 4 个模块） |
| BUILD_LMDEPLOY_SUBMODULE | ON | 编译 LMDeploy 算子 |
| BUILD_DEFAULT_OP_SUBMODULE | ON | 编译默认算子（op 模块，其他框架依赖） |
| ENABLE_BUILD_CUTLASS_OP | ON | 编译 CUTLASS/MATLASS GEMM 算子 |
| ENABLE_BUILD_GPTQ_MARLIN_OP | OFF | 编译 GPTQ Marlin 量化算子 |
| DEBUG_LINE_INFO | - | 添加内核调试行号信息 |
| MAX_JOBS | CPU 核数 | 并行编译任务数 |

## 四、扩展模块清单

| 模块名 | 框架 | 源码目录 | 输出文件 | API 类型 |
|--------|------|---------|---------|---------|
| mcoplib.op | 默认 | op/*.cu | op.cpython-310-x86_64-linux-gnu.so | py_limited_api=False |
| mcoplib._C | vLLM | op/vllm/*.cu | _C.abi3.so | py_limited_api=True (stable ABI) |
| mcoplib._moe_C | vLLM MoE | op/vllm/moe/*.cu | _moe_C.abi3.so | py_limited_api=True |
| mcoplib.sgl_kernel | SGLang | op/sglang/csrc/*.cu | sgl_kernel.cpython-310-x86_64-linux-gnu.so | py_limited_api=False |
| mcoplib.sgl_grouped_gemm_cuda | SGLang | op/sglang/csrc/moe/grouped_gemm*.cu | sgl_grouped_gemm_cuda.cpython-310-x86_64-linux-gnu.so | py_limited_api=False |
| mcoplib.sgl_moe_fused_w4a16 | SGLang | op/sglang/csrc/moe/moe_fused_w4a16*.cu | sgl_moe_fused_w4a16.cpython-310-x86_64-linux-gnu.so | py_limited_api=False |
| mcoplib.sgl_grouped_gemm_mctlass_int8 | SGLang | op/sglang/csrc/moe/grouped_gemm_mctlass_int8*.cu | sgl_grouped_gemm_mctlass_int8.cpython-310-x86_64-linux-gnu.so | py_limited_api=False |
| mcoplib.lmdeploy | LMDeploy | op/lmdeploy/ops/*.cu | lmdeploy.cpython-310-x86_64-linux-gnu.so | py_limited_api=False |

## 五、编译构建命令

### 容器内编译（推荐）
```bash
# 进入项目目录
cd /home/metax/mcoplib/github_mcoplib/mcoplib

# 设置环境变量
export BUILD_VLLM_SUBMODULE=OFF BUILD_DEFAULT_OP_SUBMODULE=OFF BUILD_LMDEPLOY_SUBMODULE=OFF
source env_local.sh
export PATH=/opt/conda/bin:$PATH
export CMAKE_PREFIX_PATH=/opt/conda

# 增量编译
python setup.py build_ext --inplace

# 全量安装
pip install -e . --no-build-isolation
```

### Docker 外部执行
```bash
docker exec -t -w /home/metax/mcoplib/github_mcoplib/mcoplib vllm-dsv4 /bin/bash -c "
  export BUILD_VLLM_SUBMODULE=OFF BUILD_DEFAULT_OP_SUBMODULE=OFF BUILD_LMDEPLOY_SUBMODULE=OFF
  source env_local.sh
  export PATH=/opt/conda/bin:\$PATH
  export CMAKE_PREFIX_PATH=/opt/conda
  python setup.py build_ext --inplace
"
```

### 打包 wheel
```bash
python -m build --no-isolation
# 输出: dist/mcoplib-{version}+maca{maca_ver}.torch{torch_ver}-{python_tag}-{abi_tag}-{platform}.whl
```

## 六、编译流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    mcOpLib 编译构建流程                               │
└─────────────────────────────────────────────────────────────────────┘

用户执行: python setup.py build_ext --inplace
          │
          ▼
┌─────────────────────────────────────┐
│  1. setup.py 入口                    │
│  ┌─────────────────────────────────┐│
│  │ Python 版本检查 (>=3.9)         ││
│  │ MACA 版本兼容性检查 (>=3.7.0)   ││
│  │ 依赖包检查 (requirements/build) ││
│  │ 写入 version 文件               ││
│  └─────────────────────────────────┘│
│          │                          │
│          ▼                          │
│  2. 根据 BUILD_*_SUBMODULE 环境变量  │
│     构建扩展模块列表 ext_modules      │
│  ┌─────────────────────────────────┐│
│  │ BUILD_VLLM_SUBMODULE=ON?       ││
│  │   → _C, _moe_C                ││
│  │ BUILD_SGLANG_SUBMODULE=ON?     ││
│  │   → sgl_kernel, sgl_grouped_  ││
│  │     gemm_cuda, sgl_moe_fused_ ││
│  │     w4a16, sgl_grouped_gemm_  ││
│  │     mctlass_int8              ││
│  │ BUILD_LMDEPLOY_SUBMODULE=ON?  ││
│  │   → lmdeploy                  ││
│  │ BUILD_DEFAULT_OP_SUBMODULE=ON?││
│  │   → op                        ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  3. cmake_build_ext.configure()      │
│  ┌─────────────────────────────────┐│
│  │ 调用 cmake_maca 配置            ││
│  │                                 ││
│  │ 输入参数:                       ││
│  │  - CMAKE_BUILD_TYPE            ││
│  │  - MCOPLIB_TARGET_DEVICE=cuda  ││
│  │  - USE_MACA=ON                 ││
│  │  - MACA_VERSION_MAJOR/MINOR/.. ││
│  │  - VLLM_PYTHON_EXECUTABLE      ││
│  │  - FETCHCONTENT_BASE_DIR       ││
│  │  - EXT_SUFFIX                  ││
│  │  - NVCC_THREADS=8              ││
│  │  - Generator: Ninja            ││
│  │                                 ││
│  │ 只执行一次（所有扩展共享配置）   ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. CMakeLists.txt 配置阶段                                         │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ find_package: Python, pybind11, Torch                         │ │
│  │                                                               │ │
│  │ GPU 架构检测:                                                  │ │
│  │   CUDA >= 12.8 → sm_70~sm_120                                │ │
│  │   CUDA < 12.8  → sm_70~sm_90                                 │ │
│  │   MACA 额外: --offload-arch=xcore1000/1500/1600              │ │
│  │                                                               │ │
│  │ 读取 BUILD_*_SUBMODULE 环境变量                               │ │
│  │                                                               │ │
│  │ ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐ │ │
│  │ │ BUILD_VLLM=ON   │  │ BUILD_SGLANG=ON │  │BUILD_LMDEPLOY  │ │ │
│  │ │                 │  │                 │  │    =ON         │ │ │
│  │ │ FetchContent:   │  │                 │  │                │ │ │
│  │ │  CUTLASS/MATLASS│  │ flashinfer      │  │                │ │ │
│  │ │                 │  │ include paths   │  │                │ │ │
│  │ │ 收集 .cu 源文件 │  │ 收集 .cu 源文件 │  │ 收集 .cu 源文件│ │ │
│  │ │                 │  │                 │  │                │ │ │
│  │ │ define_gpu_     │  │ define_gpu_     │  │ define_gpu_    │ │ │
│  │ │ extension_      │  │ extension_      │  │ extension_     │ │ │
│  │ │ target(_C)      │  │ target(sgl_     │  │ target         │ │ │
│  │ │                 │  │  kernel等)      │  │ (lmdeploy)     │ │ │
│  │ └─────────────────┘  └─────────────────┘  └────────────────┘ │ │
│  │                                                               │ │
│  │ ┌─────────────────┐                                          │ │
│  │ │BUILD_DEFAULT=ON │                                          │ │
│  │ │                 │                                          │ │
│  │ │ 收集 op/*.cu    │                                          │ │
│  │ │ define_gpu_     │                                          │ │
│  │ │ extension_      │                                          │ │
│  │ │ target(op)      │                                          │ │
│  │ └─────────────────┘                                          │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
          │
          │
          ▼
┌─────────────────────────────────────┐
│  5. CMake 编译阶段                    │
│  ┌─────────────────────────────────┐│
│  │ cmake_maca --build . -j=N       ││
│  │   --target=_C                   ││
│  │   --target=_moe_C               ││
│  │   --target=sgl_kernel           ││
│  │   --target=sgl_grouped_gemm_... ││
│  │   --target=sgl_moe_fused_w4a16  ││
│  │   --target=sgl_grouped_gemm_... ││
│  │   --target=lmdeploy             ││
│  │   --target=op                   ││
│  │                                 ││
│  │ 编译流程:                       ││
│  │  .cu → nvcc/mcc → .o           ││
│  │  .cpp → g++ → .o               ││
│  │  .o → linker → .so             ││
│  │                                 ││
│  │ 每个 .cu 文件按 GPU 架构编译:    ││
│  │  sm_75, sm_80, sm_89, sm_90    ││
│  │  + xcore1000/1500/1600 (MACA)  ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  6. CMake 安装阶段                    │
│  ┌─────────────────────────────────┐│
│  │ cmake_maca --install .          ││
│  │   --prefix=build/lib.../mcoplib ││
│  │   --component=<target_name>     ││
│  │                                 ││
│  │ 对每个扩展模块分别安装:          ││
│  │  .so → build/lib.../mcoplib/    ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  7. setuptools 复制阶段               │
│  ┌─────────────────────────────────┐│
│  │ build/lib.../mcoplib/*.so       ││
│  │         │                       ││
│  │         ▼  (copy)               ││
│  │ mcoplib/*.so                    ││
│  │                                 ││
│  │ 最终产物:                       ││
│  │  mcoplib/                       ││
│  │   ├── op.cpython-310-...so      ││
│  │   ├── _C.abi3.so               ││
│  │   ├── _moe_C.abi3.so           ││
│  │   ├── sgl_kernel.cpython-...so ││
│  │   ├── sgl_grouped_gemm_...so   ││
│  │   ├── sgl_moe_fused_w4a16.so   ││
│  │   ├── sgl_grouped_gemm_...so   ││
│  │   ├── lmdeploy.cpython-...so   ││
│  │   ├── version                  ││
│  │   └── __init__.py              ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  8. Python 中使用                     │
│  ┌─────────────────────────────────┐│
│  │ import mcoplib                  ││
│  │ import mcoplib.sgl_kernel       ││
│  │ import mcoplib._C               ││
│  │                                 ││
│  │ torch.ops.sgl_kernel.xxx(...)  ││
│  │ torch.ops._C.xxx(...)          ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
```

## 七、关键构建细节

### 1. 单次 CMake 配置
所有扩展模块共享一次 `cmake_maca` 配置，`configure()` 只在第一个扩展调用时执行，后续扩展跳过。

### 2. GPU 架构编译策略
- 每个 .cu 文件通过 `set_gencode_flags_for_srcs()` 设置独立的 GPU 架构编译标志
- MACA 额外添加 `--offload-arch=xcore1000/1500/1600` 支持 MetaX GPU
- 特定文件有额外编译优化（如 scaled_mm 添加 `-mllvm -metaxgpu-igroup=true`）

### 3. CUTLASS/MATLASS 替换
- MACA 环境下，CUTLASS 源码路径被替换为 `$MACA_PATH/include`（即 MATLASS）
- 通过 `FetchContent` 机制加载

### 4. 版本号生成
版本格式: `{mcoplib_version}+maca{maca_ver}.torch{torch_major.minor}`
示例: `0.4.9+maca3.7.1.8.dsv4.torch2.8`

### 5. 增量编译
`python setup.py build_ext --inplace` 支持增量编译，只重新编译修改过的 .cu/.cpp 文件。需先 `rm -rf build/` 才能全量重编。

## 八、常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| cmake_maca not found | 未 source env.sh/env_local.sh | `source env_local.sh` |
| Python < 3.9 error | 容器默认 python 版本低 | `export PATH=/opt/conda/bin:$PATH` |
| mc_runtime_types.h not found | CMake 找到错误的 torch | `export CMAKE_PREFIX_PATH=/opt/conda` |
| sgl_kernel.so not found (install) | lmdeploy install 失败导致整体中断 | `BUILD_LMDEPLOY_SUBMODULE=OFF` |
| __bfloat162 undefined | MACA 不支持 CUDA 专用类型 | 使用 `__nv_bfloat16` + `__bfloat162float` |
