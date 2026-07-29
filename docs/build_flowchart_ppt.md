# mcOpLib 编译构建流程图 (PPT版)

## 整体编译流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                       mcOpLib 编译构建总流程                         │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ ① 环境准备│───▶│ ② 前置检查│───▶│ ③ CMake  │───▶│ ④ CMake  │───▶│ ⑤ CMake  │───▶│ ⑥ 产物部署│
  │          │    │          │    │   配置    │───▶│   编译    │───▶│   安装    │───▶│          │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

---

## ① 环境准备

```
  source env_local.sh              ← 设置 MACA 编译器路径
  export PATH=/opt/conda/bin:$PATH ← Python/CMake 路径
  export CMAKE_PREFIX_PATH=/opt/conda ← 指定 torch/pybind11 位置

  可选: BUILD_VLLM_SUBMODULE=OFF
        BUILD_DEFAULT_OP_SUBMODULE=OFF
        BUILD_LMDEPLOY_SUBMODULE=OFF
```

---

## ② 前置检查

```
  python setup.py build_ext --inplace
       │
       ▼
  ┌─────────────────────────────┐
  │  Python >= 3.9 ?            │──No──▶ 报错退出
  │  MACA >= 3.7.0 ?            │──No──▶ 报错退出
  │  依赖包满足? (build.txt)     │──No──▶ 报错退出
  │  写入 mcoplib/version 文件   │
  └─────────────────────────────┘
       │ Yes
       ▼
  根据 BUILD_*_SUBMODULE 构建扩展模块列表
```

---

## ③ CMake 配置

```
  cmake_maca <源码目录> -G Ninja [参数...]
       │
       ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                    CMakeLists.txt                           │
  │                                                             │
  │  find_package(Python, pybind11, Torch)                     │
  │                                                             │
  │  GPU 架构: sm_75, sm_80, sm_89, sm_90                     │
  │  MACA 架构: xcore1000, xcore1500, xcore1600               │
  │                                                             │
  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
  │  │  vLLM 子模块  │ │ SGLang 子模块│ │ LMDeploy子模块│       │
  │  │  (BUILD_VLLM)│ │(BUILD_SGLANG)│ │(BUILD_LMDEP) │       │
  │  │              │ │              │ │              │       │
  │  │ _C:          │ │ sgl_kernel:  │ │ lmdeploy:    │       │
  │  │  attention   │ │  attention   │ │  attention   │       │
  │  │  quantization│ │  moe         │ │  pos_encoding│       │
  │  │  layernorm   │ │  quantization│ │  cache       │       │
  │  │  moe(topk)   │ │  elementwise │ │              │       │
  │  │  ...         │ │  ...         │ │              │       │
  │  │              │ │              │ │              │       │
  │  │ _moe_C:      │ │ sgl_grouped_ │ │              │       │
  │  │  moe_align   │ │  gemm_cuda   │ │              │       │
  │  │  topk_softmax│ │ sgl_moe_     │ │              │       │
  │  │  grouped_topk│ │  fused_w4a16 │ │              │       │
  │  └──────────────┘ │ sgl_grouped_ │ └──────────────┘       │
  │                    │  gemm_mctlass│                        │
  │  ┌──────────────┐ │  _int8       │                        │
  │  │ 默认算子子模块 │ └──────────────┘                        │
  │  │(BUILD_DEFAULT)│                                         │
  │  │              │  依赖:                                    │
  │  │ op:          │  • CUTLASS → MATLASS ($MACA_PATH/include)│
  │  │  fused_rope  │  • flashinfer (Python site-packages)     │
  │  │  rms_norm    │  • mcblas (MACA BLAS)                    │
  │  │  moe_swiglu  │                                          │
  │  │  ...         │                                          │
  │  └──────────────┘                                          │
  └─────────────────────────────────────────────────────────────┘
```

---

## ④ CMake 编译

```
  cmake_maca --build . -j=<N> --target=<模块名>
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  每个 .cu 文件编译流程:                                   │
  │                                                         │
  │    .cu ──▶ [nvcc/mcc] ──▶ .o                           │
  │              │                                          │
  │              ├── -gencode=arch=compute_75,code=sm_75    │
  │              ├── -gencode=arch=compute_80,code=sm_80    │
  │              ├── -gencode=arch=compute_89,code=sm_89    │
  │              ├── -gencode=arch=compute_90,code=sm_90    │
  │              ├── --offload-arch=xcore1000               │
  │              ├── --offload-arch=xcore1500               │
  │              ├── --offload-arch=xcore1600               │
  │              ├── -O3 -std=c++17                         │
  │              └── --use_fast_math --expt-relaxed-constexpr│
  │                                                         │
  │  链接:                                                   │
  │    .o ──▶ [linker] ──▶ <模块名>.cpython-310-x86_64-linux-gnu.so
  └─────────────────────────────────────────────────────────┘
```

---

## ⑤⑥ 安装与部署

```
  cmake_maca --install . --component=<模块名>
       │
       ▼
  ┌──────────────────────────────────────────────────────┐
  │  编译产物 → 安装路径映射:                              │
  │                                                      │
  │  build/temp.../                                      │
  │    ├── _C.abi3.so              → mcoplib/_C.abi3.so │
  │    ├── _moe_C.abi3.so          → mcoplib/_moe_C.abi3.so│
  │    ├── sgl_kernel.cpython-310-...so                  │
  │    │                           → mcoplib/sgl_kernel...so│
  │    ├── sgl_grouped_gemm_...so  → mcoplib/sgl_grouped_...│
  │    ├── sgl_moe_fused_w4a16.so  → mcoplib/sgl_moe_...   │
  │    ├── sgl_grouped_gemm_mctlass_int8.so              │
  │    │                           → mcoplib/sgl_grouped_...│
  │    ├── lmdeploy.cpython-310-...so                    │
  │    │                           → mcoplib/lmdeploy...so  │
  │    └── op.cpython-310-...so    → mcoplib/op...so       │
  │                                                      │
  │  Python 调用:                                         │
  │    import mcoplib.sgl_kernel                         │
  │    torch.ops.sgl_kernel.moe_sum_reduce(input, out, s)│
  └──────────────────────────────────────────────────────┘
```

---

## 扩展模块架构总览

```
                    mcoplib Python 包
                         │
          ┌──────────────┼──────────────┐──────────────┐
          │              │              │              │
     ┌────┴────┐   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
     │  vLLM   │   │ SGLang  │   │LMDeploy │   │ 默认算子 │
     │  模块   │   │  模块   │   │  模块   │   │  模块   │
     └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘
          │              │              │              │
    ┌─────┴─────┐  ┌────┴─────┐  ┌────┴────┐  ┌────┴────┐
    │ _C.abi3.so│  │sgl_kernel│  │lmdeploy │  │  op.so  │
    │           │  │   .so    │  │  .so    │  │         │
    │_moe_C     │  │sgl_group │  │         │  │fused_rope│
    │ .abi3.so  │  │ed_gemm.so│  │         │  │rms_norm │
    │           │  │sgl_moe_  │  │         │  │moe_swiglu│
    │           │  │fused.so  │  │         │  │...      │
    │           │  │sgl_group │  │         │  │         │
    │           │  │ed_gemm_  │  │         │  │         │
    │           │  │int8.so   │  │         │  │         │
    └───────────┘  └──────────┘  └─────────┘  └─────────┘
```

---

## 快速参考: 编译命令

```
# ===== 仅编译 SGLang 算子 (推荐开发模式) =====
cd /home/metax/mcoplib/github_mcoplib/mcoplib
export BUILD_VLLM_SUBMODULE=OFF BUILD_DEFAULT_OP_SUBMODULE=OFF BUILD_LMDEPLOY_SUBMODULE=OFF
source env_local.sh
export PATH=/opt/conda/bin:$PATH CMAKE_PREFIX_PATH=/opt/conda
python setup.py build_ext --inplace

# ===== 全量编译 =====
source env.sh
python setup.py build_ext --inplace

# ===== 打包 wheel =====
python -m build --no-isolation
```
