// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
// #include <Python.h>
// #include <iostream>
// #include <vector>
// #include <stdexcept>

// // 辅助宏：检查 Python 调用是否出错
// #define CHECK_PY_ERR() \
//     if (PyErr_Occurred()) { \
//         PyErr_Print(); \
//         throw std::runtime_error("Python API error occurred."); \
//     }

// int main() {
//     std::cout << "[Info] Initializing Python Interpreter..." << std::endl;
    
//     // 1. 初始化 Python 解释器
//     Py_Initialize();
    
//     try {
//         // 2. 导入模块 "mcoplib.op"
//         // 相当于 Python: import mcoplib.op
//         PyObject* pModuleName = PyUnicode_FromString("mcoplib.op");
//         PyObject* pModule = PyImport_Import(pModuleName);
//         Py_DECREF(pModuleName);
        
//         if (!pModule) {
//             throw std::runtime_error("Failed to import mcoplib.op. Did you pip install the whl?");
//         }

//         // 3. 获取 softmax 函数对象
//         // 相当于 Python: func = mcoplib.op.softmax
//         PyObject* pFunc = PyObject_GetAttrString(pModule, "all_reduce_max");
//         if (!pFunc || !PyCallable_Check(pFunc)) {
//             throw std::runtime_error("Cannot find function 'softmax' or it is not callable.");
//         }

//         // 4. 准备输入数据 (这里比较繁琐，因为要转换 C++ 数组到 Python 列表/Numpy)
//         // 假设 softmax 接受一个 list 或者 numpy array
//         // 相当于 Python: input_data = [1.0, 2.0, 3.0]
//         std::vector<float> cpp_data = {1.0f, 2.0f, 3.0f};
//         PyObject* pArgs = PyTuple_New(1); // 这是一个参数元组
//         PyObject* pList = PyList_New(cpp_data.size());
        
//         for (size_t i = 0; i < cpp_data.size(); ++i) {
//             PyObject* pValue = PyFloat_FromDouble(cpp_data[i]);
//             PyList_SetItem(pList, i, pValue);
//         }
        
//         // 将列表放入参数元组的第0个位置
//         PyTuple_SetItem(pArgs, 0, pList);

//         // 5. 调用函数
//         // 相当于 Python: result = op.softmax(input_data)
//         std::cout << "[Info] Calling mcoplib.op.softmax..." << std::endl;
//         PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
//         CHECK_PY_ERR();

//         // 6. 解析返回值 (假设返回也是一个 list)
//         if (pResult) {
//             // 简单的打印逻辑，实际测试中你需要在这里做数值对比
//             PyObject* pIter = PyObject_GetIter(pResult);
//             PyObject* pItem;
//             std::cout << "[Info] Result: [ ";
//             while ((pItem = PyIter_Next(pIter))) {
//                 double val = PyFloat_AsDouble(pItem);
//                 std::cout << val << " ";
//                 Py_DECREF(pItem);
//             }
//             std::cout << "]" << std::endl;
//             Py_DECREF(pIter);
//             Py_DECREF(pResult);
//         } else {
//             std::cerr << "[Error] Function returned NULL" << std::endl;
//         }

//         // 清理引用
//         Py_DECREF(pArgs); // pList 会自动减少引用
//         Py_DECREF(pFunc);
//         Py_DECREF(pModule);

//     } catch (const std::exception& e) {
//         std::cerr << "[Fatal Error] " << e.what() << std::endl;
//     }

//     // 7. 关闭解释器
//     Py_Finalize();
//     return 0;
// }

// #include <Python.h>
// #include <torch/torch.h>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <vector>

// void run_python_function(PyObject* func, torch::Tensor input, torch::Tensor output) {
//     std::cout<<"======================>run_python_function 1"<<std::endl;
//     PyObject* args = PyTuple_New(2);
//     PyTuple_SetItem(args, 0, PyCapsule_New(input.data_ptr(), "torch.Tensor", nullptr));
//     PyTuple_SetItem(args, 1, PyCapsule_New(output.data_ptr(), "torch.Tensor", nullptr));
//     std::cout<<"======================>run_python_function 2"<<std::endl;
//     PyObject* result = PyObject_CallObject(func, args);
//     std::cout<<"======================>run_python_function 3"<<std::endl;
//     Py_DECREF(args);

//     if (!result) {
//         PyErr_Print();
//         throw std::runtime_error("Python function call failed");
//     }

//     Py_DECREF(result);
// }

// std::vector<float> benchmark(PyObject* func,
//                              torch::Tensor input,
//                              torch::Tensor output,
//                              int warmup,
//                              int rep)
// {
//     std::cout<<"======================>benchmark 1"<<std::endl;
//     for (int i = 0; i < warmup; i++) {
//         run_python_function(func, input, output);
//     }
//     std::cout<<"======================>benchmark 2"<<std::endl;
//     std::vector<cudaEvent_t> start(rep), end(rep);
//     for (int i = 0; i < rep; i++) {
//         cudaEventCreate(&start[i]);
//         cudaEventCreate(&end[i]);
//     }

//     for (int i = 0; i < rep; i++) {
//         cudaEventRecord(start[i]);
//         run_python_function(func, input, output);
//         cudaEventRecord(end[i]);
//     }

//     cudaDeviceSynchronize();

//     std::vector<float> duration(rep);
//     for (int i = 0; i < rep; i++) {
//         cudaEventElapsedTime(&duration[i], start[i], end[i]);
//         cudaEventDestroy(start[i]);
//         cudaEventDestroy(end[i]);
//     }

//     return duration;
// }

// int main() {
//     // ========== 初始化 Python ==========
//     Py_Initialize();

//     PyRun_SimpleString("import torch");
//     PyObject* module = PyImport_ImportModule("mcoplib.op");

//     if (!module) {
//         PyErr_Print();
//         throw std::runtime_error("Failed to import mcoplib.op");
//     }
//     std::cout<<"======================>1"<<std::endl;
//     PyObject* all_reduce_max = PyObject_GetAttrString(module, "all_reduce_max");
//     PyObject* all_reduce_sum = PyObject_GetAttrString(module, "all_reduce_sum");

//     if (!all_reduce_max || !all_reduce_sum) {
//         throw std::runtime_error("Could not find required function");
//     }
//     std::cout<<"======================>2"<<std::endl;
//     // ========== 准备数据 ==========
//     int m = 4096, n = 4096;

//     torch::Tensor input = torch::randn({m, n},
//         torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));

//     torch::Tensor output = torch::zeros({m},
//         torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
//     std::cout<<"======================>3"<<std::endl;
//     // ========== benchmark ==========
//     auto dur1 = benchmark(all_reduce_max, input, output, 10, 100);
//     std::cout<<"======================>4"<<std::endl;
//     float avg1 = std::accumulate(dur1.begin(), dur1.end(), 0.0f) / dur1.size();
//     std::cout << "all_reduce_max avg time = " << avg1 << " ms\n";

//     auto dur2 = benchmark(all_reduce_sum, input, output, 10, 100);
//     float avg2 = std::accumulate(dur2.begin(), dur2.end(), 0.0f) / dur2.size();
//     std::cout << "all_reduce_sum avg time = " << avg2 << " ms\n";

//     Py_Finalize();
//     return 0;
// }

// bench_all_reduce_pyembed.cpp
#include <nvbench/nvbench.cuh>
#include <nvbench/main.cuh>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>

namespace py = pybind11;

std::vector<float> benchmark_py_func(py::object func, py::object input, py::object output, int warmup = 2, int rep = 10) {
    // warmup
    for (int i = 0; i < warmup; ++i) {
        func(input, output);
    }

    // create python torch.cuda.Events
    py::list start_events;
    py::list end_events;
    for (int i = 0; i < rep; ++i) {
        // torch.cuda.Event(enable_timing=True)
        py::object se = py::module_::import("torch").attr("cuda").attr("Event")(true);
        py::object ee = py::module_::import("torch").attr("cuda").attr("Event")(true);
        start_events.append(se);
        end_events.append(ee);
    }

    for (int i = 0; i < rep; ++i) {
        start_events[i].attr("record")();
        func(input, output);
        end_events[i].attr("record")();
    }

    // synchronize
    py::module_::import("torch").attr("cuda").attr("synchronize")();

    std::vector<float> durations;
    durations.reserve(rep);
    for (int i = 0; i < rep; ++i) {
        float dt = start_events[i].attr("elapsed_time")(end_events[i]).cast<float>();
        durations.push_back(dt);
    }
    return durations;
}



void all_reduce_benchmark(nvbench::state &state)
{

    std::cout<<"======================>2"<<std::endl;
    py::object torch = py::module_::import("torch");
    py::object mcoplib = py::module_::import("mcoplib.op");
    
    std::cout<<"======================>3"<<std::endl;
    // get functions
    py::object all_reduce_max = mcoplib.attr("all_reduce_max");
    py::object all_reduce_sum = mcoplib.attr("all_reduce_sum");

    std::cout<<"======================>4"<<std::endl;
    // prepare data on GPU (use Python torch API, not libtorch)
    int m = 4096, n = 4096;
    py::object input = torch.attr("rand")(py::make_tuple(m, n),
                                            py::arg("device")="cuda",
                                            py::arg("dtype")=torch.attr("bfloat16"));
    py::object output = torch.attr("zeros")(py::make_tuple(m),
                                            py::arg("device")="cuda",
                                            py::arg("dtype")=torch.attr("bfloat16"));
    std::cout<<"======================>5"<<std::endl;


    const auto warmup = static_cast<int>(state.get_int64("WarmUp"));
    const auto rep = static_cast<int>(state.get_int64("Rep"));
    std::cout<<"======================>6"<<std::endl;


    state.exec([all_reduce_max, input, output, warmup,rep]
               (nvbench::launch &launch) {
        benchmark_py_func(all_reduce_max, input, output, warmup, rep);
    });
    std::cout<<"======================>7"<<std::endl;
}


// Register benchmarks
NVBENCH_BENCH(all_reduce_benchmark)
    .add_int64_axis("WarmUp", {2})
    .add_int64_axis("Rep", {10});

NVBENCH_MAIN