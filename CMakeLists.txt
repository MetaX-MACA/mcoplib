cmake_minimum_required(VERSION 3.26)
project(mcoplib LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# the default CMAKE_BUILD_TYPE is Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
#python
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)
message(STATUS  "PYTHON CMAKE ENVS Python_EXECUTABLE:${Python_EXECUTABLE} Python_LIBRARIES:${Python_LIBRARIES} Python_SITELIB:${Python_SITELIB} Python_INCLUDE_DIRS:${Python_INCLUDE_DIRS} Python_SITEARCH:${Python_SITEARCH} Python_VERSION:${Python_VERSION}")


# 自动获取 Python 的 site-packages 目录
if(DEFINED Python_SITELIB)
    set(PYTHON_PACKAGE ${Python_SITELIB})
elseif(DEFINED Python_SITEARCH)
    set(PYTHON_PACKAGE ${Python_SITEARCH})
else()
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
        OUTPUT_VARIABLE PYTHON_PACKAGE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
message(STATUS "Python package path: ${PYTHON_PACKAGE}")

if(NOT DEFINED VENDOR_TYPE)
    set(VENDOR_TYPE "Metax")
endif()

set(pybind11_DIR "${PYTHON_PACKAGE}/pybind11/share/cmake/pybind11")
message("pybind11_DIR = ${pybind11_DIR}")
find_package(pybind11 CONFIG REQUIRED)
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
find_package(Torch REQUIRED)


message(STATUS "torch compile env TORCH_INSTALL_PREFIX:${TORCH_INSTALL_PREFIX}")

option(USE_MACA "Enable MACA Support" ON)
option(ENABLE_BLAS_API "Enable blas api for fused moe" ON)

if (USE_MACA)
  message(STATUS "###########################")
  message(STATUS "Detected Using MACA")
  message(STATUS "###########################")

  add_compile_definitions(
    MACA_VERSION_MAJOR=${MACA_VERSION_MAJOR}
    MACA_VERSION_MINOR=${MACA_VERSION_MINOR}
    MACA_VERSION_PATCH=${MACA_VERSION_PATCH}
    MACA_VERSION_BUILD=${MACA_VERSION_BUILD}
  )
  message(STATUS "MACA version: ${MACA_VERSION_MAJOR}.${MACA_VERSION_MINOR}.${MACA_VERSION_PATCH}.${MACA_VERSION_BUILD}")

  set(MACA_PATH "$ENV{MACA_PATH}")
  add_compile_definitions(USE_MACA)

  if (MACA_PATH AND EXISTS ${MACA_PATH})
    message(STATUS "MACA found at ${MACA_PATH}")
  else()
    message(FATAL_ERROR "MACA not found or invalid path, please check your MACA_PATH")
  endif()
endif()


# Prevent installation of dependencies (cutlass) by default.
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12")

set(TORCH_SUPPORTED_VERSION_CUDA "2.6.0")

find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")

# Supported NVIDIA architectures.
# This check must happen after find_package(Torch) because that's when CMAKE_CUDA_COMPILER_VERSION gets defined
if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0")
else()
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0")
endif()

set(MCOPLIB_GPU_LANG "CUDA")


# For cuda we want to be able to control which architectures we compile for on
# a per-file basis in order to cut down on compile time. So here we extract
# the set of architectures we want to compile for and remove the from the
# CMAKE_CUDA_FLAGS so that they are not applied globally.
#
clear_cuda_arches(CUDA_ARCH_FLAGS)
clear_torch_arches(CUDA_ARCH_FLAGS)
extract_unique_cuda_archs_ascending(CUDA_ARCHS "${CUDA_ARCH_FLAGS}")
message(STATUS "CUDA target architectures CUDA_ARCHS: ${CUDA_ARCHS} CUDA_ARCH_FLAGS:${CUDA_ARCH_FLAGS} CMAKE_CUDA_FLAGS:${CMAKE_CUDA_FLAGS}")
# Filter the target architectures by the supported supported archs
# since for some files we will build for all CUDA_ARCHS.
cuda_archs_loose_intersection(CUDA_ARCHS
  "${CUDA_SUPPORTED_ARCHS}" "${CUDA_ARCHS}")
message(STATUS "CUDA supported target architectures: CUDA_ARCHS: ${CUDA_ARCHS} CUDA_SUPPORTED_ARCHS:${CUDA_SUPPORTED_ARCHS}")

get_torch_gpu_compiler_flags(MCOPLIB_GPU_FLAGS ${MCOPLIB_GPU_LANG} USE_MACA)

#
# Set nvcc parallelism.
#
if(NVCC_THREADS AND MCOPLIB_GPU_LANG STREQUAL "CUDA")
  list(APPEND MCOPLIB_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()

####build sub project ENV

option(BUILD_VLLM_SUBMODULE "Build the vllm op kernels subproject" ON)
option(BUILD_SGLANG_SUBMODULE "Build the sglang op kernels subproject" ON)
option(BUILD_LMDEPLOY_SUBMODULE "Build the lmdeploy op kernels subproject" ON)
option(BUILD_DEFAULT_OP_SUBMODULE "Build the default op kernels subproject" ON)


# 如果存在环境变量，则覆盖默认设置
if(DEFINED ENV{BUILD_VLLM_SUBMODULE})
    set(BUILD_VLLM_SUBMODULE "$ENV{BUILD_VLLM_SUBMODULE}")
endif()

if(DEFINED ENV{BUILD_SGLANG_SUBMODULE})
    set(BUILD_SGLANG_SUBMODULE "$ENV{BUILD_SGLANG_SUBMODULE}")
endif()

if(DEFINED ENV{BUILD_LMDEPLOY_SUBMODULE})
    set(BUILD_LMDEPLOY_SUBMODULE "$ENV{BUILD_LMDEPLOY_SUBMODULE}")
endif()

if(DEFINED ENV{BUILD_DEFAULT_OP_SUBMODULE})
    set(BUILD_DEFAULT_OP_SUBMODULE "$ENV{BUILD_DEFAULT_OP_SUBMODULE}")
endif()


#########trap kernel debug#############
if(DEFINED ENV{CUDA_PATH})
    set(CUDA_PATH "$ENV{CUDA_PATH}")
else()
    message(FATAL_ERROR "CUDA_PATH env var not set before compile project, please set.")
endif()
set(CUCC_PATH "$ENV{CUCC_PATH}")
include_directories(${CUCC_PATH}/include)
include_directories(${CUDA_PATH}/include)

#add trap kernel debug source code line info
if(DEFINED ENV{DEBUG_LINE_INFO})
  message(STATUS "Add compile params for debug source code line info when happen trap kernel")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info> 
  )
endif()

#
# Use FetchContent for C++ dependencies that are compiled as part of vLLM's build process.
# setup.py will override FETCHCONTENT_BASE_DIR to play nicely with sccache.
# Each dependency that produces build artifacts should override its BINARY_DIR to avoid
# conflicts between build types. It should instead be set to ${CMAKE_BINARY_DIR}/<dependency>.
#
include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

set(VLLM_SRC_DIR ${CMAKE_SOURCE_DIR}/op/vllm)

if (BUILD_VLLM_SUBMODULE) 
  message(STATUS "Starting to build vllm op kernels submodule...")

  #
  # _C extension
  #

  set(VLLM_EXT_SRC
    "${VLLM_SRC_DIR}/mamba/mamba_ssm/selective_scan_fwd.cu"
    "${VLLM_SRC_DIR}/cache_kernels.cu"
    "${VLLM_SRC_DIR}/attention/paged_attention_v1.cu"
    "${VLLM_SRC_DIR}/attention/paged_attention_v2.cu"
    "${VLLM_SRC_DIR}/attention/merge_attn_states.cu"
    "${VLLM_SRC_DIR}/attention/vertical_slash_index.cu"
    "${VLLM_SRC_DIR}/pos_encoding_kernels.cu"
    "${VLLM_SRC_DIR}/activation_kernels.cu"
    "${VLLM_SRC_DIR}/layernorm_kernels.cu"
    "${VLLM_SRC_DIR}/layernorm_quant_kernels.cu"
    "${VLLM_SRC_DIR}/fused_qknorm_rope_kernel.cu"
    "${VLLM_SRC_DIR}/sampler.cu"
    "${VLLM_SRC_DIR}/cuda_view.cu"
    "${VLLM_SRC_DIR}/quantization/gptq/q_gemm.cu"
    "${VLLM_SRC_DIR}/quantization/compressed_tensors/int8_quant_kernels.cu"
    "${VLLM_SRC_DIR}/quantization/fp8/common.cu"
    "${VLLM_SRC_DIR}/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu"
    "${VLLM_SRC_DIR}/quantization/gguf/gguf_kernel.cu"
    "${VLLM_SRC_DIR}/quantization/activation_kernels.cu"
    "${VLLM_SRC_DIR}/cuda_utils_kernels.cu"
    "${VLLM_SRC_DIR}/torch_bindings.cpp")

  # support opt of gptq-marlin
  set_source_files_properties(
    "${VLLM_SRC_DIR}/quantization/gptq/q_gemm.cu"
    PROPERTIES
    COMPILE_FLAGS "-mllvm -metaxgpu-igroup=true -mllvm -metaxgpu-igroup-config=B8 -maxmregcount=128"
  )

  if(MCOPLIB_GPU_LANG STREQUAL "CUDA")
    SET(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

    # Set CUTLASS_REVISION. Used for FetchContent. Also fixes some bogus messages when building.
    set(CUTLASS_REVISION "v4.0.0" CACHE STRING "CUTLASS revision to use")

    # Use the specified CUTLASS source directory for compilation if VLLM_CUTLASS_SRC_DIR is provided
    if (DEFINED ENV{VLLM_CUTLASS_SRC_DIR})
      set(VLLM_CUTLASS_SRC_DIR $ENV{VLLM_CUTLASS_SRC_DIR})
    endif()

    # Substitue CUTLASS with MATLASS 
    if (USE_MACA)
      message(WARNING "Use MACA, Overwrite VLLM_CUTLASS_SRC_DIR.")
      set(VLLM_CUTLASS_SRC_DIR $ENV{MACA_PATH}/include)
    endif()

    if(VLLM_CUTLASS_SRC_DIR)
      if(NOT IS_ABSOLUTE VLLM_CUTLASS_SRC_DIR)
        get_filename_component(VLLM_CUTLASS_SRC_DIR "${VLLM_CUTLASS_SRC_DIR}" ABSOLUTE)
      endif()
      message(STATUS "The VLLM_CUTLASS_SRC_DIR is set, using ${VLLM_CUTLASS_SRC_DIR} for compilation")
      FetchContent_Declare(cutlass SOURCE_DIR ${VLLM_CUTLASS_SRC_DIR})
    else()
      message(FATAL_ERROR "Error: Unable to download cutlass automatically. Please download cutlass manually and set the 'VLLM_CUTLASS_SRC_DIR' environment variable.")
    endif()
    FetchContent_MakeAvailable(cutlass)

    list(APPEND VLLM_EXT_SRC
      "${VLLM_SRC_DIR}/quantization/awq/gemm_kernels.cu"
      "${VLLM_SRC_DIR}/permute_cols.cu"
      "${VLLM_SRC_DIR}/quantization/cutlass_w8a8/scaled_mm_entry.cu"
      "${VLLM_SRC_DIR}/quantization/fp4/nvfp4_quant_entry.cu"
      "${VLLM_SRC_DIR}/quantization/fp4/nvfp4_scaled_mm_entry.cu"
      "${VLLM_SRC_DIR}/sparse/cutlass/sparse_scaled_mm_entry.cu"
      "${VLLM_SRC_DIR}/cutlass_extensions/common.cpp"
      "${VLLM_SRC_DIR}/attention/mla/cutlass_mla_entry.cu")

    set_gencode_flags_for_srcs(
      SRCS "${VLLM_EXT_SRC}"
      CUDA_ARCHS "${CUDA_ARCHS}")

    # support opt of gptq-marlin
    set_source_files_properties(
      "${VLLM_SRC_DIR}/quantization/awq/gemm_kernels.cu"
      PROPERTIES
      COMPILE_FLAGS " -mllvm -metaxgpu-igroup=true -mllvm -metaxgpu-igroup-config=B8 -maxmregcount=128"
    )

    # support opt of cutlass w8a8 scale mm
    set_source_files_properties(
      "${VLLM_SRC_DIR}/quantization/cutlass_w8a8/scaled_mm_entry.cu"
      PROPERTIES
      COMPILE_FLAGS " -mllvm -metaxgpu-igroup=true -mllvm -misched-postra=true"
    )  


    # For the cutlass_scaled_mm kernels we want to build the c2x (CUTLASS 2.x)
    # kernels for the remaining archs that are not already built for 3x.
    # (Build 8.9 for FP8)
    cuda_archs_loose_intersection(SCALED_MM_2X_ARCHS
      "7.5;8.0;8.7;8.9+PTX" "${CUDA_ARCHS}")
    # subtract out the archs that are already built for 3x
    list(REMOVE_ITEM SCALED_MM_2X_ARCHS ${SCALED_MM_3X_ARCHS})
    if (SCALED_MM_2X_ARCHS)
      set(SRCS "${VLLM_SRC_DIR}/quantization/cutlass_w8a8/scaled_mm_c2x.cu")
      set_gencode_flags_for_srcs(
        SRCS "${SRCS}"
        CUDA_ARCHS "${SCALED_MM_2X_ARCHS}")
      list(APPEND VLLM_EXT_SRC "${SRCS}")
      list(APPEND MCOPLIB_GPU_FLAGS "-DENABLE_SCALED_MM_C2X=1")

      # support opt of cutlass w8a8 scale mm
      set_source_files_properties(
        "${VLLM_SRC_DIR}/quantization/cutlass_w8a8/scaled_mm_c2x.cu"
        PROPERTIES
        COMPILE_FLAGS " -mllvm -metaxgpu-igroup=true -mllvm -misched-postra=true"
      )
      message(STATUS "Building scaled_mm_c2x for archs: ${SCALED_MM_2X_ARCHS}")
    endif()
  # if CUDA endif
  endif()


  message(STATUS "Enabling C extension.")
  define_gpu_extension_target(
    _C
    DESTINATION mcoplib
    LANGUAGE ${MCOPLIB_GPU_LANG}
    SOURCES ${VLLM_EXT_SRC}
    COMPILE_FLAGS ${MCOPLIB_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR} 
    INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
    INCLUDE_DIRECTORIES ${VLLM_SRC_DIR}
    USE_SABI 3
    WITH_SOABI)

  # If CUTLASS is compiled on NVCC >= 12.5, it by default uses
  # cudaGetDriverEntryPointByVersion as a wrapper to avoid directly calling the
  # driver API. This causes problems when linking with earlier versions of CUDA.
  # Setting this variable sidesteps the issue by calling the driver directly.
  target_compile_definitions(_C PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
  
  #添加编译参数
  target_compile_options(_C PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:
  --Wno-maca-min-blocks-per-multiprocessor
  --Wno-ignored-attributes
  --Wno-macro-redefined
  --Wno-pass-failed
  --Wno-maca-compat
  >)

  #
  # _moe_C extension
  #

  set(VLLM_MOE_EXT_SRC
    "${VLLM_SRC_DIR}/moe/torch_bindings.cpp"
    "${VLLM_SRC_DIR}/moe/moe_align_sum_kernels.cu"
    "${VLLM_SRC_DIR}/moe/moe_lora_align_sum_kernels.cu"
    "${VLLM_SRC_DIR}/moe/topk_softmax_kernels.cu")

  if (USE_MACA AND ENABLE_BLAS_API)
    list(APPEND VLLM_MOE_EXT_SRC "${VLLM_SRC_DIR}/moe/moe_ops.cpp")

    set(MCBLAS_INCLUDE_DIR $ENV{MACA_PATH}/include/mcblas)
    set(MCBLAS_LIB $ENV{MACA_PATH}/lib/libmcblas.so)
  endif()

  #if(MCOPLIB_GPU_LANG STREQUAL "CUDA")
  #  list(APPEND VLLM_MOE_EXT_SRC "${VLLM_SRC_DIR}/moe/moe_wna16.cu")
  #endif()

  set_gencode_flags_for_srcs(
    SRCS "${VLLM_MOE_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")

  #if(MCOPLIB_GPU_LANG STREQUAL "CUDA")
  #  set(VLLM_MOE_WNA16_SRC
  #    "${VLLM_SRC_DIR}/moe/moe_wna16.cu")

  #  set_gencode_flags_for_srcs(
  #    SRCS "${VLLM_MOE_WNA16_SRC}"
  #    CUDA_ARCHS "${CUDA_ARCHS}")

  #  list(APPEND VLLM_MOE_EXT_SRC "${VLLM_MOE_WNA16_SRC}")
  #  # 9.0 for latest bf16 atomicAdd PTX
  #  cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0;8.7;9.0+PTX" "${CUDA_ARCHS}")
  #endif()

  if(MCOPLIB_GPU_LANG STREQUAL "CUDA")
    set(MOE_PERMUTE_SRC
        "${VLLM_SRC_DIR}/moe/moe_permute_unpermute_op.cu")

    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_PERMUTE_SRC}"
      CUDA_ARCHS "${MOE_PERMUTE_ARCHS}")

    list(APPEND VLLM_MOE_EXT_SRC "${MOE_PERMUTE_SRC}")
  endif()
  message(STATUS "Enabling moe extension.")

  set(BLAS_API_ARGS "")
  if(USE_MACA AND ENABLE_BLAS_API)
    message(STATUS "Blas API for fused moe enabled")
    list(APPEND BLAS_API_ARGS 
        INCLUDE_DIRECTORIES ${MCBLAS_INCLUDE_DIR}
        LIBRARIES ${MCBLAS_LIB})
  endif()

  define_gpu_extension_target(
    _moe_C
    DESTINATION mcoplib
    LANGUAGE ${MCOPLIB_GPU_LANG}
    SOURCES ${VLLM_MOE_EXT_SRC}
    COMPILE_FLAGS ${MCOPLIB_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
    INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
    INCLUDE_DIRECTORIES ${VLLM_SRC_DIR}
    ${BLAS_API_ARGS}
    USE_SABI 3
    WITH_SOABI)

endif()


if(NOT DEFINED EXT_SUFFIX)
  set(EXT_SUFFIX ".so")
endif()

if (BUILD_LMDEPLOY_SUBMODULE)
  message(STATUS "Starting to build lmdeploy op kernels submodule...")
  #############lmdeploy ops start ###############
  set(LMDEPLOY_SRCS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/op/lmdeploy/ops)
  set(LMDEPLOY_EXT_SRC
      "${LMDEPLOY_SRCS_DIR}/pybind.cpp"
      "${LMDEPLOY_SRCS_DIR}/cache_kernels.cu"
      "${LMDEPLOY_SRCS_DIR}/attention/attention_kernels.cu"
      "${LMDEPLOY_SRCS_DIR}/pos_encoding_kernels.cu"
      )
  set_gencode_flags_for_srcs(
    SRCS "${LMDEPLOY_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")
  #
  message(STATUS "Enabling lmdeploy_ops extension.")
  define_gpu_extension_target(
    lmdeploy
    DESTINATION mcoplib
    LANGUAGE ${MCOPLIB_GPU_LANG}
    SOURCES ${LMDEPLOY_EXT_SRC}
    COMPILE_FLAGS ${MCOPLIB_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${LMDEPLOY_SRCS_DIR}
    INCLUDE_DIRECTORIES ${VLLM_SRC_DIR}/attention
    INCLUDE_DIRECTORIES ${VLLM_SRC_DIR})

  set_target_properties(lmdeploy PROPERTIES PREFIX "" SUFFIX "${EXT_SUFFIX}")

  #############lmdeploy ops end ###############
endif()


if (BUILD_DEFAULT_OP_SUBMODULE)
  message(STATUS "Starting to build default op kernels submodule...")
  ############mcoplib##############
  set(MCOPLIB_SRCS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/op)
  set(MCOPLIB_EXT_SRC
      "${MCOPLIB_SRCS_DIR}/pybind.cpp"
      "${MCOPLIB_SRCS_DIR}/fused_bias_dropout.cu"
      "${MCOPLIB_SRCS_DIR}/fused_rope.cu"
      "${MCOPLIB_SRCS_DIR}/fused_bias_swiglu.cu"
      "${MCOPLIB_SRCS_DIR}/fused_repeat_kv.cu"
      "${MCOPLIB_SRCS_DIR}/fused_bias_gelu.cu"
      "${MCOPLIB_SRCS_DIR}/fused_rms_norm_dq.cu"
      "${MCOPLIB_SRCS_DIR}/moe_swiglu_dq.cu"
      "${MCOPLIB_SRCS_DIR}/moe_softmax_topk.cu"
      "${MCOPLIB_SRCS_DIR}/all_reduce.cu"
      "${MCOPLIB_SRCS_DIR}/rotary_embedding.cu"
      "${MCOPLIB_SRCS_DIR}/moe_gather.cu"
      "${MCOPLIB_SRCS_DIR}/store_kv.cu"
      "${MCOPLIB_SRCS_DIR}/moe_scatter_dynamic_quant.cu"
      "${MCOPLIB_SRCS_DIR}/scale_dynamic_quant.cu"
      "${MCOPLIB_SRCS_DIR}/rope_train.cu"
      "${MCOPLIB_SRCS_DIR}/recv_from_attention_node_post_process.cu"
      "${MCOPLIB_SRCS_DIR}/send_to_attention_node_pre_process.cu"
      "${MCOPLIB_SRCS_DIR}/int8_quant_kernels.cu"
      "${MCOPLIB_SRCS_DIR}/fused_add_layernorm_per_token_quant_padding_output.cu"
      "${MCOPLIB_SRCS_DIR}/rms_norm_dynamic_per_token_quant.cu"
      "${MCOPLIB_SRCS_DIR}/fused_moe_gate_deepseek.cu"
      "${MCOPLIB_SRCS_DIR}/gptq_marlin.cu"
      "${MCOPLIB_SRCS_DIR}/moe_fused_gate_opt.cu")

  set_source_files_properties(
    "${MCOPLIB_SRCS_DIR}/gptq_marlin.cu"
    "${MCOPLIB_SRCS_DIR}/fused_moe_gate_deepseek.cu"
    PROPERTIES
    COMPILE_FLAGS "-mllvm -metaxgpu-igroup=true -mllvm -metaxgpu-igroup-config=B8 -maxmregcount=128"
  )
  set_gencode_flags_for_srcs(
    SRCS "${MCOPLIB_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")
  message(STATUS "Enabling lmdeploy_ops extension.")
  define_gpu_extension_target(
    op
    DESTINATION mcoplib
    LANGUAGE ${MCOPLIB_GPU_LANG}
    SOURCES ${MCOPLIB_EXT_SRC}
    COMPILE_FLAGS ${MCOPLIB_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${MCOPLIB_SRCS_DIR}
    INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(op PROPERTIES PREFIX "" SUFFIX "${EXT_SUFFIX}")

endif()


if (BUILD_SGLANG_SUBMODULE)
  message(STATUS "Starting to build sglang op kernels submodule...")
  ############sglang-kernel##############
  set(SGLANG_KERNELS_SRCS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/op/sglang)
  set(MCOPLIB_SGLANG_KERNEL_SRC
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/allreduce/custom_all_reduce.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/attention/cascade.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/attention/merge_attn_states.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/attention/vertical_slash_index.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/attention/fused_mla.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/elementwise/activation.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/elementwise/fused_add_rms_norm_kernel.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/elementwise/rope.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/elementwise/pos_enc.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/elementwise/topk.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/elementwise/copy.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/elementwise/concat_mla.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/gemm/awq_kernel.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/moe_align_kernel.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/moe_fused_gate.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/moe_topk_softmax_kernels.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/prepare_moe_input.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/moe_sum.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/moe_sum_reduce.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/speculative/eagle_utils.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/speculative/speculative_sampling.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/speculative/packbit.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/speculative/ngram_utils.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/grammar/apply_token_bitmask_inplace_cuda.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/kvcacheio/transfer.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/memory/store.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/common_extension.cc"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/moe_fused_gate_opt.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/memory/weak_ref_tensor.cpp"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/quantization/int8_quant_kernels.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/quantization/quantize_kernel.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/sgl_diffusion/elementwise/timestep_embedding.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/elementwise/fused_layernorm_dynamic_per_token_quant_custom.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/elementwise/fused_rotary_emb.cu"
      #"${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/cutlass_moe/scaled_mm_c2x.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/kimi_k2_moe_fused_gate.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/fused_qknorm_rope_kernel.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/moe_topk_sigmoid_kernels.cu"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/mamba/causal_conv1d.cu")
      #"${SGLANG_KERNELS_SRCS_DIR}/csrc/cutlass_w8a8/scaled_mm_entry.cu")

  set_gencode_flags_for_srcs(
    SRCS "${MCOPLIB_SGLANG_KERNEL_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")
  message(STATUS "Enabling mcoplib  sgl_kernel extension.")

  #flashinfer 
  set(FLASHINFER_INCLUDE 
          ${PYTHON_PACKAGE}/flashinfer/data/include
          ${PYTHON_PACKAGE}/flashinfer/data/gemm
          ${PYTHON_PACKAGE}/flashinfer/data/include
          ${PYTHON_PACKAGE}/flashinfer/data/csrc
          cublas)
  define_gpu_extension_target(
    sgl_kernel
    DESTINATION mcoplib
    LANGUAGE ${MCOPLIB_GPU_LANG}
    SOURCES ${MCOPLIB_SGLANG_KERNEL_SRC}
    #COMPILE_FLAGS ${MCOPLIB_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${SGLANG_KERNELS_SRCS_DIR}/include
    INCLUDE_DIRECTORIES ${SGLANG_KERNELS_SRCS_DIR}/csrc
    INCLUDE_DIRECTORIES ${FLASHINFER_INCLUDE}
    INCLUDE_DIRECTORIES $ENV{MACA_PATH}/include
    INCLUDE_DIRECTORIES ${SGLANG_KERNELS_SRCS_DIR}/csrc/fast_hadamard_transform)

  set_target_properties(sgl_kernel PROPERTIES PREFIX "" SUFFIX "${EXT_SUFFIX}")
  target_link_libraries(sgl_kernel PRIVATE mctlassEx)

  #添加编译参数
  target_compile_options(sgl_kernel PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3 -std=c++17 -Xcompiler -fPIC -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90>
      $<$<COMPILE_LANGUAGE:CUDA>:-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_VERSIONS_GENERATED -DCUTE_USE_PACKED_TUPLE=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 >
      $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v --expt-relaxed-constexpr  -use-fast-math  -DTORCH_API_INCLUDE_EXTENSION_H >
      $<$<COMPILE_LANGUAGE:CUDA>:-DFLASHINFER_ENABLE_BF16 -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_HALF_OPERATORS__ >
  )

  ############sglang grouped_gemm_cuda##############
  set(MCOPLIB_SGLANG_GROUPED_GEMM_SRC
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/grouped_gemm.cpp"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/grouped_gemm_cuda.cu"
  )

  set_gencode_flags_for_srcs(
    SRCS "${MCOPLIB_SGLANG_GROUPED_GEMM_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")
  message(STATUS "Enabling mcoplib  grouped_gemm_cuda extension.")

  define_gpu_extension_target(
    sgl_grouped_gemm_cuda
    DESTINATION mcoplib
    LANGUAGE ${MCOPLIB_GPU_LANG}
    SOURCES ${MCOPLIB_SGLANG_GROUPED_GEMM_SRC}
    COMPILE_FLAGS ${MCOPLIB_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${SGLANG_KERNELS_SRCS_DIR}/include
    INCLUDE_DIRECTORIES ${SGLANG_KERNELS_SRCS_DIR}/csrc)
  set_target_properties(sgl_grouped_gemm_cuda PROPERTIES PREFIX "" SUFFIX "${EXT_SUFFIX}")

  target_compile_options(sgl_grouped_gemm_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__MACA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_HALF_OPERATORS__ >
  )

  ############sglang moe_fused_w4a16##############
  set(MCOPLIB_SGLANG_MOE_FUSED_SRC
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/moe_fused_w4a16.cpp"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/moe_fused_w4a16_cuda.cu"
  )

  set_gencode_flags_for_srcs(
    SRCS "${MCOPLIB_SGLANG_MOE_FUSED_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")
  message(STATUS "Enabling mcoplib  grouped_gemm_cuda extension.")

  define_gpu_extension_target(
    sgl_moe_fused_w4a16
    DESTINATION mcoplib
    LANGUAGE ${MCOPLIB_GPU_LANG}
    SOURCES ${MCOPLIB_SGLANG_MOE_FUSED_SRC}
    COMPILE_FLAGS ${MCOPLIB_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${SGLANG_KERNELS_SRCS_DIR}/include
    INCLUDE_DIRECTORIES ${SGLANG_KERNELS_SRCS_DIR}/csrc)
  set_target_properties(sgl_moe_fused_w4a16 PROPERTIES PREFIX "" SUFFIX "${EXT_SUFFIX}")

  #添加编译参数
  target_compile_options(sgl_moe_fused_w4a16 PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:-U__MACA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_HALF_OPERATORS__ >
  )

  ############sglang grouped_gemm_mctlass_int8##############
  set(MCOPLIB_SGLANG_GROUPED_GEMM_MCTLASS_SRC
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/grouped_gemm_mctlass_int8.cpp"
      "${SGLANG_KERNELS_SRCS_DIR}/csrc/moe/grouped_gemm_mctlass_int8_cuda.cu"
  )

  set_gencode_flags_for_srcs(
    SRCS "${MCOPLIB_SGLANG_GROUPED_GEMM_MCTLASS_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")
  message(STATUS "Enabling mcoplib  grouped_gemm_cuda extension.")

  define_gpu_extension_target(
    sgl_grouped_gemm_mctlass_int8
    DESTINATION mcoplib
    LANGUAGE ${MCOPLIB_GPU_LANG}
    SOURCES ${MCOPLIB_SGLANG_GROUPED_GEMM_MCTLASS_SRC}
    COMPILE_FLAGS ${MCOPLIB_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${SGLANG_KERNELS_SRCS_DIR}/include
    INCLUDE_DIRECTORIES ${SGLANG_KERNELS_SRCS_DIR}/csrc)
  set_target_properties(sgl_grouped_gemm_mctlass_int8 PROPERTIES PREFIX "" SUFFIX "${EXT_SUFFIX}")

  #添加编译参数
  target_compile_options(sgl_grouped_gemm_mctlass_int8 PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:-U__MACA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_HALF_OPERATORS__ > 
  )
endif()