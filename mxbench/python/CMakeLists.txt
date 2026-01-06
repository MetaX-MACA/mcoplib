cmake_minimum_required(VERSION 3.26)

# CUDA is transitive dependency of nvbench
project(nvbench LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Development.Module)
find_package(CUDAToolkit REQUIRED)

# Get CMake package manager
#set(_cpm_download_location ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
set(CPM_LOCAL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake)
message(STATUS "=====> ${CPM_LOCAL_PATH}")
# file(
#   DOWNLOAD
#   https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.42.0/CPM.cmake
#   ${_cpm_download_location}
#   EXPECTED_HASH SHA256=2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a
# )
include(${CPM_LOCAL_PATH})

# CPMAddPackage(
#    NAME nvbench
#    #SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..
#    SOURCE_DIR /home/metax/yiyu/nvbench/nvbench
#    OPTIONS "NVBench_INSTALL_RULES ON"
#    FIND_PACKAGE_ARGS CONFIG REQUIRED
# )
set(EXTRA_SEARCH_PATH $ENV{NVBENCH_INSTALL_PATH})

if(NOT DEFINED ENV{NVBENCH_INSTALL_PATH})
    set (EXTRA_SEARCH_PATH "${NVBENCH_INSTALL_PATH}")
endif()
message(STATUS "nvbench install path:${EXTRA_SEARCH_PATH}")
if(EXTRA_SEARCH_PATH)
    # 将其添加到 CMAKE_PREFIX_PATH 中
    # list(APPEND ...) 确保不覆盖已有的其他搜索路径
    list(APPEND CMAKE_PREFIX_PATH "${EXTRA_SEARCH_PATH}")
    message(STATUS "已添加环境变量路径到搜索列表: ${EXTRA_SEARCH_PATH}")
endif()

find_package(nvbench CONFIG REQUIRED HINTS)

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


pybind11_add_module(_nvbench MODULE src/py_nvbench.cpp)
target_link_libraries(_nvbench PUBLIC nvbench::nvbench)
target_link_libraries(_nvbench PRIVATE CUDA::cudart_static)
target_link_libraries(_nvbench PRIVATE CUDA::cuda_driver)

set_target_properties(_nvbench PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(_nvbench PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
set_target_properties(_nvbench PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS _nvbench DESTINATION cuda/bench)

# Install nvbench shared library next to the Python module
# nvbench::nvbench is an IMPORTED target, so we directly copy the .so file
install(FILES $ENV{NVBENCH_INSTALL_PATH}/lib/libnvbench.so
        DESTINATION cuda/bench)
