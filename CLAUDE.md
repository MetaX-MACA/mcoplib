# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

mcOpLib is a custom CUDA operator library for MetaX GPU (MACA architecture) LLM inference. It provides optimized kernels for three inference frameworks — vLLM, SGLang, LMDeploy — plus a set of shared "default" operators. Kernels compile via the MACA SDK (`cmake_maca`) targeting MetaX xcore GPUs, and are exposed to Python through several `pybind11`/`TORCH_LIBRARY` extension modules.

Most kernels require real MetaX C-series hardware to run. The build enforces a minimum MACA SDK compatibility version (currently `3.7.0`, defined in `setup.py` as `MIN_COMPATIBILITY_MACA_VERSION`) at both build time and runtime.

## Build & Install

Always source the environment first — `cmake_maca`, the MACA toolchain, and `MACA_PATH` must be on PATH:

```bash
cd /home/metax/mcoplib/gerrit/mcoplib
source env.sh        # CI/online build (ENABLE_BUILD_GPTQ_MARLIN_OP=1)
source env_local.sh  # local container build (ENABLE_BUILD_GPTQ_MARLIN_OP=0)
```

Both `env.sh` and `env_local.sh` set `MACA_PATH` (default `/opt/maca`), `CUDA_PATH` (`~/cu-bridge/CUDA_DIR`), and `CUCC_PATH`. The only difference is `ENABLE_BUILD_GPTQ_MARLIN_OP`.

Build commands (run from the repo root after sourcing env):

```bash
# Editable install, no incremental compilation; -v/-vv/-vvv for verbose
pip install -e . --no-build-isolation

# Editable install WITH incremental compilation support
python setup.py develop
# Or extension-only build (in-place)
python setup.py build_ext --inplace

# Wheel packaging -> dist/mcoplib-{ver}+maca{maca_ver}.torch{torch_ver}-{pytag}-{abi}-{platform}.whl
python -m build --no-isolation
```

Note: `pip install -e . -v` captures stdout via a pipe, so `setup.py`'s `print` output only appears after the build fully succeeds or fails. Use `python setup.py develop` if you need live build progress.

`WCUDA_DEBUG=1` prints WCUDA details during compilation.

### Build-control environment variables

All default to `ON`. Set to `OFF` to skip a submodule's kernels:

- `BUILD_VLLM_SUBMODULE` — vLLM ops (`mcoplib._C`, `mcoplib._moe_C`)
- `BUILD_SGLANG_SUBMODULE` — SGLang ops (`mcoplib.sgl_kernel`, `sgl_grouped_gemm_cuda`, `sgl_moe_fused_w4a16`, `sgl_grouped_gemm_mctlass_int8`)
- `BUILD_LMDEPLOY_SUBMODULE` — LMDeploy ops (`mcoplib.lmdeploy`)
- `BUILD_DEFAULT_OP_SUBMODULE` — default ops (`mcoplib.op`). **Must stay ON** — other modules reuse these kernels and `import mcoplib` imports `mcoplib.op` unconditionally.

Additional flags:

- `ENABLE_BUILD_CUTLASS_OP` (default `1`) — compile CUTLASS OP API kernels
- `ENABLE_BUILD_GPTQ_MARLIN_OP` — set by `env.sh`/`env_local.sh`
- `CUCC_TARGETS="xcore1000, xcore1089, xcore1500, xcore1501"` — multi-platform (C600/C600U/C588) compilation
- `MAX_JOBS` — override parallel compile job count
- `FETCHCONTENT_BASE_DIR` — override cutlass/dependency download dir (default `ROOT/.deps`)

`sccache`/`ccache` and `ninja` are auto-detected and used when available.

## Testing

Python unit tests live in `unit_test/` (70+ files, pytest-based, require MetaX hardware):

```bash
cd /home/metax/mcoplib/gerrit/mcoplib
pytest unit_test/test_rms_norm.py          # single test file
pytest unit_test/ -k "test_name"           # by name pattern
```

Tests import directly from compiled extensions, e.g. `from mcoplib.op import rms_norm` or `torch.ops._C.paged_attention_v1`. A failed import of `mcoplib._C`/`mcoplib._moe_C` usually means the corresponding `BUILD_*_SUBMODULE` was off at build time.

### CV op kernels (separate deb package)

CV ops build independently as a deb (not part of the main wheel):

```bash
cd op/cv
cmake_maca -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake_maca --build build -j$(nproc)
cd build && cpack -G DEB                    # -> mcoplib_cv-*.deb
sudo dpkg -i mcoplib_cv-0.2.0-Linux.deb    # installs to /opt/maca-ai/mcoplib
```

### C++ kernel tests (require installed mcoplib_cv deb)

```bash
cd unit_test/cpp
mkdir build && cd build
cmake_maca .. && make_maca
```

## Version & runtime checks

```bash
mcoplib_version          # print build/runtime MACA version info (entry point in pyproject.toml)
```

At import time `mcoplib/__init__.py` runs three checks against the running MACA SDK: build-vs-runtime version compatibility (release major.minor must match; master/date builds must be within 30 days), and the minimum-compatibility floor (`Min_Compatibility_Maca_Version` written into `mcoplib/version` at build time). A mismatch calls `sys.exit(1)`. To bypass while debugging, edit `mcoplib/version`'s `Min_Compatibility_Maca_Version` field or `/opt/maca/Version.txt` — neither is guaranteed to work correctly.

## Debugging operator inputs

Set env vars to trace or dump operator inputs at runtime:

```bash
export MCOP_DEBUG_TRACE=1                  # print dtype/shape of each op's inputs to stderr
export MCOP_DEBUG_PARAMS_DUMP=1            # dump full input params to disk as JSON
export MCOP_TENSOR_DUMP_SAMPLE_SIZE=20     # sample N elements per tensor (default)
export MCOP_TENSOR_DUMP_FULL=1             # dump entire tensor contents
```

## Architecture

### Source layout (`op/` is the core)

Operators are organized by target framework. Each framework subdir builds into its own extension module with its own pybind/TORCH_LIBRARY bindings:

| Dir | Extension module | Binding entry | Framework |
|-----|------------------|---------------|-----------|
| `op/*.cu` + `op/pybind.cpp` | `mcoplib.op` | `pybind.cpp` (`PYBIND11_MODULE`) | shared/default ops |
| `op/vllm/torch_bindings.cpp` | `mcoplib._C` | `TORCH_LIBRARY` | vLLM |
| `op/vllm/moe/torch_bindings.cpp` | `mcoplib._moe_C` | `TORCH_LIBRARY` | vLLM MoE |
| `op/sglang/csrc/common_extension.cc` etc. | `mcoplib.sgl_kernel` (+ grouped_gemm, moe_fused_w4a16, grouped_gemm_mctlass_int8) | `TORCH_LIBRARY_FRAGMENT` | SGLang |
| `op/lmdeploy/ops/pybind.cpp` | `mcoplib.lmdeploy` | pybind | LMDeploy |
| `op/cv/` | `libmcoplib_cv.so` (deb, not wheel) | — | CV/vision |

`include/` and `kernel/` hold headers for the **default** ops only; vLLM/SGLang/LMDeploy keep their headers inside their own subdirs (`op/vllm/`, `op/sglang/include/`, `op/lmdeploy/ops/`).

### Build system

`setup.py` is the entry point. It defines `CMakeExtension` subclasses (one per extension module) and a `cmake_build_ext` command that:

1. Reads MACA version from `$MACA_PATH/Version.txt`, enforces `MIN_COMPATIBILITY_MACA_VERSION` (release builds compared by major.minor.patch; master/date builds skip the check).
2. Runs `cmake_maca` configure once per extension, passing `MACA_VERSION_*`, `USE_MACA=ON`, `EXT_SUFFIX`, and the build-type options above.
3. Builds all configured targets, then `cmake --install` copies the `.so` files into the `mcoplib/` package dir.

Extension ordering in `setup.py` matters: extensions needing `EXT_SUFFIX` (lmdeploy, op, sgl_*) are registered before the `abi3` ones (`_moe_C`, `_C`), and `mcoplib.op` is forced first because `configure()` runs once per `CMakeLists.txt`.

`CMakeLists.txt` orchestrates the actual compilation: it conditionally `add_subdirectory`s each framework block based on `BUILD_*_SUBMODULE`, fetches cutlass via `FetchContent` (into `.deps`), and sets per-target `--offload-arch=xcore1000/xcore1500/xcore1600` compile flags. SGLang MoE/cutlass and vLLM quantization paths are gated by `GLOBAL_ENABLE_BUILD_CUTLASS_OP`.

### Python package (`mcoplib/`)

- `__init__.py` — version parsing + MACA compatibility checks (runs at import)
- `version` — generated at build time by `setup.py:write_git_info_file` with `Mcoplib_Version`, `Build_Maca_Version`, `Min_Compatibility_Maca_Version`, `GIT_BRANCH`, `GIT_COMMIT`, framework op versions
- `custom_ops.py` — high-level Python wrappers over `mcoplib.op` kernels
- `triton_fused_moe.py`, `triton_dsv4.py`, `triton_utils/` — Triton fallback/alternative kernels
- `quant_utils.py`, `marlin_*.py`, `fused_*.py` — quantization and fusion helpers

### Calling kernels

Default ops: `from mcoplib.op import rms_norm` / `ops.rms_norm(...)`.

vLLM ops: `torch.ops._C.paged_attention_v1(...)` and `torch.ops._moe_C.<op>`.

SGLang ops: `import mcoplib.sgl_kernel` then `torch.ops.sgl_kernel.<op>` (plus `sgl_grouped_gemm_cuda`, `sgl_moe_fused_w4a16`).

### Benchmarking (`benchmark/`, `mxbench/`)

`benchmark/` runs the mxbench harness for op performance/accuracy testing. `benchmark/build_env_local.sh` auto-installs both `mcoplib` and `mxbench` (use `build_env_local.sh` locally; `build_env.sh` is for Jenkins CI). Output metrics include `Acc_Pass`, `Cos_Dist`, `GPU Time`, `Noise`, `Elem/s`, `GlobalMem BW`, `BWUtil`. See `benchmark/How_To_Add_Op_Benchtest.md` and `benchmark/README.md`.

`Skills/mxbench_op_test/` and `Skills/optimized-cuda-kernels/` hold reference material for adding benchmarks and CUDA optimization techniques.

## Common gotchas

- **MACA version mismatch at build**: `ERROR: MACA minimum compatibility version mismatch, aborting` — the running MACA SDK is below `3.7.0`. Compare by major.minor.patch only; build/suffix components (`.38.dsv4`, `.c600u`) are ignored. Master/date versions (YYYYMMDD) skip the check.
- **`cmake_maca: No such file or directory`**: forgot to `source env.sh`.
- **`libssl.so.1.1: cannot open shared object file`**: cmake too new for the image's openssl. Pin `pip3 install cmake==3.26.3`.
- **Wheel missing version info / no `version` file**: git not installed in the build env (`setup.py` falls back to `"unknown"` for branch/commit but still writes the version file).
- **`python -m build` "No module named build.__main__"**: the `build` package in the env is not PyPA's. `pip install --force-reinstall build`.
- **Disabling `BUILD_DEFAULT_OP_SUBMODULE`** breaks `import mcoplib` — the default ops are a hard dependency for other modules.

## Repo conventions

- Build tool is `cmake_maca`, never plain `cmake`.
- Always `source env.sh` (or `env_local.sh`) before any build or test invocation.
- The Python package dir is `mcoplib/`; the operator source dir is `op/`. They share the name `mcoplib.op` (the compiled extension) but live in different trees.
- `docs/` (Chinese) has detailed architecture write-ups: `code_structure.md`, `build_summary.md`, `op_kernel_classification.md`, and per-op analyses (`fused_add_rmsnorm_analysis.md`, `topk_softmax_sigmoid_analysis.md`, `minimax_allreduce_rms_qk_analysis.md`). Consult these before non-trivial kernel work.
