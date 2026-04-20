---
name: optimized-cuda-kernels
description: "Provides guidance for writing and benchmarking optimized CUDA kernels for Metax C500 GPU, 阅读CUDA kernels 的源码，根据Metax C500 GPU硬件参数极致优化CUDA算子，通过ReAct方式来benchmark 优化后算子的精度及性能，通过对比优化前后的性能，达到优化指标后即完成优化任务"
disable-model-invocation: false
user-invocable: true
---

# CUDA Kernels

This skill provides patterns and guidance for developing optimized CUDA kernels targeting Metax C500 GPU
## Quick Start

### 深度阅读CUDA kernel源码
1: 你是一个CUDA算子优化权威专家， 非常擅长于CUDA高性能编码, 深度阅读给出的CUDA op kernels源码，分析该kernel 存在哪些优化空间，分析该算子属于Memory Bound, 还是Compute Bound
### 优化CUDA Kernel，编写CUDA kernel benchmark测试代码
2：根据./references./c500-optimization-guide.md GPU硬件参数及优化手段，在Metax C500 gpu上优化该CUDA 算子
3：编写一个以该算子名称为文件名的cu代码文件， 该代码文件需要包含优化后的cuda实现，以及优化前的CUDA 实现，还需要包含一个该算子的CPU实现
4：测试代码需要包含10次的warmop,以及100次实际运行平均耗时，需要计算并打印如下信息：
```bash
// =========================================================================
// Calculate bandwidth
// =========================================================================
double data_read_gb = input_size_bytes / 1e9;
double data_written_gb = output_size_bytes / 1e9;
double total_data_gb = data_read_gb + data_written_gb;

double before_optimized_avg_time_sec = before_optimized_avg_time_ms / 1000.0;
double after_optimized_avg_time_sec = after_optimized_avg_time_ms / 1000.0;
double bandwidth_gbps = total_data_gb / avg_time_sec;
float theoretical_bandwidth = GetDeviceMemoryBandwidthGBps();
float efficiency = (bandwidth_gbps / theoretical_bandwidth) * 100.0f;

//Print performance data before optimization
std::cout << "\n=====================Print performance data before optimization==================" << std::endl;
std::cout << "[Performance Results]" << std::endl;
std::cout << "=======================================" << std::endl;
std::cout << "  Total time for " << config.test_iterations << " iterations: " << total_time_ms << " ms" << std::endl;
std::cout << "  Average time per iteration: " << avg_time_ms << " ms" << std::endl;
std::cout << "  Data read: " << data_read_gb << " GB" << std::endl;
std::cout << "  Data written: " << data_written_gb << " GB" << std::endl;
std::cout << "  Total data transferred: " << total_data_gb << " GB" << std::endl;
std::cout << "  Achieved bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;
std::cout << "  Theoretical bandwidth: " << theoretical_bandwidth << " GB/s" << std::endl;
std::cout << "  Bandwidth efficiency: " << efficiency << "%" << std::endl;

double after_optimized_avg_time_sec = after_optimized_avg_time_ms / 1000.0;
double after_optimized_bandwidth_gbps = total_data_gb / after_optimized_avg_time_sec;
float after_optimized_efficiency = (after_optimized_bandwidth_gbps / theoretical_bandwidth) * 100.0f;
std::cout<<"Print performance data after optimization"<<std::endl;
std::cout << "\n===================Print performance data after optimization====================" << std::endl;
std::cout << "[Performance Results]" << std::endl;
std::cout << "=======================================" << std::endl;
std::cout << "  Total time for " << config.test_iterations << " iterations: " << total_time_ms << " ms" << std::endl;
std::cout << "  Average time per iteration: " << after_optimized_avg_time_sec << " ms" << std::endl;
std::cout << "  Data read: " << data_read_gb << " GB" << std::endl;
std::cout << "  Data written: " << data_written_gb << " GB" << std::endl;
std::cout << "  Total data transferred: " << total_data_gb << " GB" << std::endl;
std::cout << "  Achieved bandwidth: " << after_optimized_bandwidth_gbps << " GB/s" << std::endl;
std::cout << "  Theoretical bandwidth: " << theoretical_bandwidth << " GB/s" << std::endl;
std::cout << "  Bandwidth efficiency: " << after_optimized_efficiency << "%" << std::endl;

//Percentage improvement in printing performance

// =========================================================================
std::cout << "\n===================Percentage improvement in printing performance====================" << std::endl;
double rate = before_optimized_avg_time_ms/after_optimized_avg_time_ms;
std::cout << "kernel optimization Percentage improvement: " << rate << "%" << std::endl;
```
5:cu代码中必须要包含CUDA算子的优化前后的结果的精度验证，至少保证8组典型的输出数据，使用余弦相似度去校验最终结果， 必须要保证精度达到0.9999及以上， 不然就打印报错退出，优化失败
6：保存该cu代码文件
### 编译benchmark 测试代码文件
1：根据给出的服务器IP使用ssh登录服务器
2：根据给出的容器名称， 使用docker命令登录该容器
3：在容器中执行shell命令：mkdir -p /workspace/cuda_optimized/{cuda op name}， {cuda op name}为给出的优化的算子名称
4: 将cuda kernel 优化后的cu源代码文件远程发送到远程服务器的/tmp目录中，并将该cu文件拷贝到容器的/workspace/cuda_optimized/{cuda op name}目录下
5：在容器中设置如下所示的环境变量：
```shell
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
6: 设置环境变量后，运行编译命令：cucc -std=c++17 ./cuda_op_name.cu -o cuda_op_name -lcudart , cuda_op_name.cu实际应该为算子名称.cu， -o cuda_op_name 也应该为算子名称， 比如：算子名称为softmax，那么cu文件名为：softmax.cu ，-o cuda_op_name  也应该为：-o softmax， 编译命令为：cucc -std=c++17 ./softmax.cu -o softmax -lcudart
###  容器中运行编译后的可执行文件
1：运行编译后的可执行文件
2：获取运行后的打印信息
3：分析远程服务器中容器中执行后的终端的输出信息，并总结下优化后的结果
4：如果效果未达到目标或者编译失败，或者测试精度校验失败，则需要继续根据以上步骤来继续优化该CUDA 算子，采用ReAct模型来继续优化算子，不断地进行优化，直到优化目标完成，ReAct CUDA算子优化流程为：首先：深度阅读CUDA kernel源码。 然后：优化CUDA Kernel，编写CUDA kernel benchmark测试代码。 然后：编译benchmark 测试代码文件。 然后：容器中运行编译后的可执行文件， 不断地根据以上步骤进行优化，修改错误，编译通过后，验证，失败了再重复根据以上步骤进行优化验证。
5：持续迭代， 直到该算子优化达到优化目标（1：精度验证通过 2：算子性能提升达到了目标）

###
1：优化完整后， 输出最终优化完整后的打印信息

### GPU Optimization Guides
- [c500-optimization-guide.md](references/c500-optimization-guide.md)

### Reference