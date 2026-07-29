cmake_minimum_required(VERSION 3.30.4)

project(mxbench LANGUAGES C CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 80)

# For CUDA via nvcc:
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-unused-parameter -Xcompiler=-std=gnu++17 -Xcompiler=-Wno-error -Xcompiler=-Wno-cast-qual -Xcompiler=-Wno-implicit-float-conversion --expt-relaxed-constexpr --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3 -Xcompiler -fPIC \
                    -mllvm -metaxgpu-disable-bsm-offset=0 \
                    -mllvm -metaxgpu-combine-intrinsic-bfe=false \
                    -mllvm -metaxgpu-igroup=true \
                    -mllvm -metaxgpu-igroup-config=B8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17 -O2 -Wno-error -Wno-c++17-extensions -Wno-unused-parameter -DUSE_MACA")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error -Wno-unused-parameter ")

#PYTHON3
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python3 Include: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 Libraries: ${Python3_LIBRARIES}")

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


#maca compiler setting 
option(USE_MACA "Enable MACA Support" ON)

if(NOT DEFINED VENDOR_TYPE)
    set(VENDOR_TYPE "Metax")
endif()


#pybind11
set(pybind11_DIR "${PYTHON_PACKAGE}/pybind11/share/cmake/pybind11")
message(STATUS "pybind11_DIR = ${pybind11_DIR}")
find_package(pybind11 CONFIG REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# --------------------------
# libtorch
# --------------------------
set(Torch_DIR "${PYTHON_PACKAGE}/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
message(STATUS "Torch include dirs = ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries    = ${TORCH_LIBRARIES}")

# --------------------------
# CUDA
# --------------------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDAToolkit include dirs = ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit libraries = ${CUDAToolkit_LIBRARIES}")

set(mxbench_ops
  softmax_benchmark.cu
  all_reduce.cc
)

# Metatarget for all examples:
add_custom_target(mxbench.ops.all)

set(MXBENCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(MXBENCH_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/install/lib/)


function (mxbench_add_examples_target target_prefix)
  add_custom_target(${target_prefix}.all)
  add_dependencies(mxbench.ops.all ${target_prefix}.all)

  foreach(example_src IN LISTS mxbench_ops)
    get_filename_component(example_name "${example_src}" NAME_WLE)
    string(PREPEND example_name "${target_prefix}.")


    add_executable(${example_name} "${example_src}")

    target_include_directories(${example_name}
        PRIVATE
            ${MXBENCH_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Link with NVBench
    
    target_link_directories(${example_name} PRIVATE ${MXBENCH_LIB_DIR})
    target_link_libraries(${example_name} PRIVATE 
    "nvbench")

    if ("${example_src}" MATCHES ".*\\.(cc|cpp)$")
        message(STATUS, "setting compiler for C plus plus source code ")
        target_compile_definitions(${example_name}  PRIVATE USE_MACA)

        target_include_directories(${example_name}  PRIVATE
            ${Python3_INCLUDE_DIRS}
            ${TORCH_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
        )

        target_link_libraries(${example_name}  PRIVATE
            ${Python3_LIBRARIES}
            ${TORCH_LIBRARIES}
        )
        #cuda
        target_link_libraries(${example_name} PRIVATE CUDA::cudart CUDA::nvrtc)
        #pybind11
        target_link_libraries(${example_name} PRIVATE pybind11::embed)

    endif()

    set(example_args --timeout 0.1)
    list(APPEND example_args --min-time 1e-5)
    add_test(NAME ${example_name}
      COMMAND "$<TARGET_FILE:${example_name}>" ${example_args})

    # These should not deadlock. If they do, it may be that the CUDA context was created before
    set_tests_properties(${example_name} PROPERTIES
      FAIL_REGULAR_EXPRESSION "Possible Deadlock Detected"
    )

    add_dependencies(${target_prefix}.all ${example_name})
  endforeach()
endfunction()



mxbench_add_examples_target(mxbench)
