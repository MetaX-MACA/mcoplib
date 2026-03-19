cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 设置项目名称和使用的语言（需要包含CUDA）
project(mcoplib_cv LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA架构
set(CUDA_ARCHITECTURES "native")  # 自动检测当前GPU架构

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3 -Xcompiler -fPIC  -Xcompiler=-Wno-unused-parameter -Xcompiler=-std=gnu++17 -Xcompiler=-Wno-error -Xcompiler=-Wno-cast-qual -Xcompiler=-Wno-implicit-float-conversion \
                    -mllvm -metaxgpu-disable-bsm-offset=0 \
                    -mllvm -metaxgpu-combine-intrinsic-bfe=false \
                    -mllvm -metaxgpu-igroup=true \
                    -mllvm -metaxgpu-igroup-config=B8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17 -O2 -Wno-error -Wno-c++17-extensions -Wno-unused-parameter ")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error -Wno-unused-parameter ")

option(BUILD_LIBRARY "Build the library" ON)

# 设置CUDA编译选项
set(CUDA_COMPILE_OPTIONS
    -O3
)

# 添加库目标
if(BUILD_LIBRARY)
    
    add_library(mcoplib_cv SHARED
        csrc/postprocess.cu
        csrc/preprocess.cu
        csrc/arithm.cu
        csrc/calsum.cu
        csrc/countNozero.cu
        csrc/split.cu
        csrc/meanstdev.cu
    )
    
    # 设置包含目录
    target_include_directories(mcoplib_cv PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    
    # 设置CUDA编译选项
    target_compile_options(mcoplib_cv PRIVATE
        ${CUDA_COMPILE_OPTIONS}
    )
    if(BUILD_MACA_PLATFORM STREQUAL "C600")
      target_compile_options(mcoplib_cv PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:--offload-arch=xcore1500>
        )
    endif()
    
    # 安装规则
    #install(TARGETS mcoplib_cv
    #    EXPORT ${PROJECT_NAME}Targets
    #    LIBRARY DESTINATION lib
    #    ARCHIVE DESTINATION lib
    #    RUNTIME DESTINATION bin
    #    INCLUDES DESTINATION include
    #)
    
    # 安装头文件
    install(DIRECTORY include/ DESTINATION include)

    install(TARGETS mcoplib_cv
      LIBRARY DESTINATION opt/maca-ai/mcoplib/lib  # note: no leading slash
    )
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION /opt/maca-ai/mcoplib/include)
    set(CPACK_GENERATOR "DEB")
    set(CPACK_PACKAGE_NAME "mcoplib_cv")
    set(CPACK_PACKAGE_VERSION "0.2.0")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "yixiong.yu@metax-tech.com")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.29), libstdc++6")
    set(CPACK_PACKAGING_INSTALL_PREFIX "/")
    include(CPack)
endif()