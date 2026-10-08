#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <cstring>
#include <cstdlib>  // for std::getenv
#include <optional> // 必须包含这个头文件
#include <chrono>
#include <ctime>
#include <atomic>
#include <cstdio>      // for snprintf
#include <sys/stat.h>  // for mkdir
#include <unistd.h>    // for getpid
namespace mcop {
namespace debug {

// 运行期环境变量检查 - 完全通过环境变量控制，无需编译期宏
// 使用 static 变量缓存，确保只检查一次环境变量，性能开销极小
// 第一层开关（默认关）：只有显式设为 1/ON/on 才 dump
inline bool is_debug_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("MCOP_DEBUG_PARAMS_DUMP");
        if (env == nullptr) return false;  // 默认关
        std::string v(env);
        return (v == "1" || v == "ON" || v == "on");
    }();
    return enabled;
}

// 第二层过滤开关：MCOP_DEBUG_FILTER 未设置 -> 全关（不打印、不 dump）；
// 设 "all" -> 全部算子；否则按逗号分隔的函数名子串匹配
inline bool name_matches(const char* func_name) {
    static const char* filter_p = std::getenv("MCOP_DEBUG_FILTER");
    if (filter_p == nullptr || filter_p[0] == '\0') return false;
    const std::string name(func_name ? func_name : "");
    const std::string filter(filter_p);
    if (filter == "all") return true;
    size_t start = 0;
    while (start <= filter.size()) {
        size_t comma = filter.find(',', start);
        std::string token = filter.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        // 去掉首尾空白，支持 "op1, op2" 带空格的写法
        size_t b = token.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) {
            token.clear();
        } else {
            size_t e = token.find_last_not_of(" \t\r\n");
            token = token.substr(b, e - b + 1);
        }
        if (!token.empty() && name.find(token) != std::string::npos) return true;
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return false;
}

// 是否该对这个算子 dump：第一层开关(默认关) && 第二层过滤 FILTER
inline bool should_dump(const char* func_name) {
    if (!is_debug_enabled()) return false;
    if (!name_matches(func_name)) return false;
    return true;
}

// 获取 tensor 数据采样数量（通过环境变量配置）
inline size_t get_tensor_dump_sample_size() {
    static const size_t sample_size = []() -> size_t {  // 显式指定返回类型
        const char* env = std::getenv("MCOP_TENSOR_DUMP_SAMPLE_SIZE");
        if (!env) return 20;  // 默认采样前20个和后20个（共40个）
        try {
            size_t size = std::stoul(env);
            return size > 0 ? size : 20;
        } catch (...) {
            return 20;
        }
    }();
    return sample_size;
}

// 是否 dump 全部 tensor 数据（通过环境变量配置）
inline bool dump_full_tensor() {
    static const bool dump_full = []() {
        const char* env = std::getenv("MCOP_TENSOR_DUMP_FULL");
        return env != nullptr && std::string(env) != "0";
    }();
    return dump_full;
}

// 基础模板：默认使用 typeid
template<typename T>
struct TypeInfo {
    static std::string name() {
        return typeid(T).name();
    }
};

// 宏：快速定义基础类型的名称
#define DEFINE_TYPE_NAME(Type, StringName) \
template<> struct TypeInfo<Type> { static std::string name() { return StringName; } }

// 常用基础类型特化 (解决 "i", "f", "b" 问题)
DEFINE_TYPE_NAME(int, "int");
DEFINE_TYPE_NAME(long, "long");
DEFINE_TYPE_NAME(float, "float");
DEFINE_TYPE_NAME(double, "double");
DEFINE_TYPE_NAME(bool, "bool");
DEFINE_TYPE_NAME(size_t, "size_t");

// Tensor 特化
template<>
struct TypeInfo<at::Tensor> {
    static std::string name() { return "at::Tensor"; }
};

// 检测是否为 std::optional 的 Type Trait
template<typename T>
struct is_optional : std::false_type {};

template<typename T>
struct is_optional<std::optional<T>> : std::true_type {};

// 参数序列化器
class ParamSerializer {
public:
    struct ParamInfo {
        std::string name;
        std::string type;
        std::string value;
        std::string shape;
        std::string dtype;
        size_t bytes;
    };

    static std::string escape_string(const std::string& s) {
        std::string result;
        result.reserve(s.length() * 1.2);
        for (char c : s) {
            switch (c) {
                case '"': result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default:
                    if (c < 32) {
                        char buf[8];
                        snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                        result += buf;
                    } else {
                        result += c;
                    }
            }
        }
        return result;
    }

    // Tensor序列化 - dump 实际数据内容
    static ParamInfo serialize_tensor(const char* name, const at::Tensor& tensor) {
        ParamInfo info;
        info.name = name;
        info.type = TypeInfo<at::Tensor>::name();
        info.dtype = toString(tensor.scalar_type());

        // 获取shape
        std::ostringstream shape_ss;
        shape_ss << "[";
        for (size_t i = 0; i < tensor.dim(); ++i) {
            if (i > 0) shape_ss << ", ";
            shape_ss << tensor.size(i);
        }
        shape_ss << "]";
        info.shape = shape_ss.str();

        // 获取数据指针和基本信息
        std::ostringstream ptr_ss;
        ptr_ss << "0x" << std::hex << reinterpret_cast<uintptr_t>(tensor.data_ptr());
        std::string data_ptr_str = ptr_ss.str();

        info.bytes = tensor.nbytes();

        // Dump tensor 内容
        size_t total_elements = tensor.numel();
        std::ostringstream value_ss;

        if (total_elements == 0) {
            value_ss << "[]";
        } else {
            if (dump_full_tensor()) {
                // Dump 所有数据：整块拷回 CPU（要全量，不可避免）
                at::Tensor cpu_tensor = tensor.device().is_cpu() ? tensor : tensor.to(torch::kCPU);
                value_ss << dump_tensor_full(cpu_tensor);
            } else {
                // Dump 采样数据（前N个和后N个）：只拷被采样的少量元素回 CPU，
                // 避免把整个 tensor 从显存搬回内存（这是之前 dump 开销的大头）
                size_t sample_size = get_tensor_dump_sample_size();
                if (total_elements <= 2 * sample_size) {
                    // 元素太少，直接全量拷（本身就没多大）
                    at::Tensor cpu_tensor = tensor.device().is_cpu() ? tensor : tensor.to(torch::kCPU);
                    value_ss << dump_tensor_full(cpu_tensor);
                } else {
                    at::Tensor flat = tensor.flatten();
                    at::Tensor head = flat.slice(0, 0, static_cast<int64_t>(sample_size)).to(torch::kCPU);
                    at::Tensor tail = flat.slice(0, static_cast<int64_t>(total_elements - sample_size),
                                                  static_cast<int64_t>(total_elements)).to(torch::kCPU);
                    value_ss << dump_tensor_sample(head, tail);
                }
                value_ss << " (showing " << std::min(2 * sample_size, total_elements)
                          << " of " << total_elements << " elements, set MCOP_TENSOR_DUMP_FULL=1 for all)";
            }
        }

        // 添加数据指针信息
        value_ss << " [data_ptr=" << data_ptr_str << "]";
        info.value = value_ss.str();

        return info;
    }

    // Helper: 按 dtype 把 [start, start+count) 的元素打印进 ss（仅限 CPU 连续 tensor）
    static void dump_tensor_range(const at::Tensor& tensor, size_t start, size_t count, std::ostringstream& ss) {
        const size_t end = start + count;
        if (tensor.dtype() == at::ScalarType::BFloat16) {
            const at::BFloat16* data = tensor.data_ptr<at::BFloat16>();
            for (size_t i = start; i < end; ++i) {
                if (i > start) ss << ", ";
                ss << static_cast<float>(data[i]);
            }
        } else if (tensor.dtype() == at::ScalarType::Float) {
            const float* data = tensor.data_ptr<float>();
            for (size_t i = start; i < end; ++i) {
                if (i > start) ss << ", ";
                ss << data[i];
            }
        } else if (tensor.dtype() == at::ScalarType::Half) {
            const at::Half* data = tensor.data_ptr<at::Half>();
            for (size_t i = start; i < end; ++i) {
                if (i > start) ss << ", ";
                ss << static_cast<float>(data[i]);
            }
        } else if (tensor.dtype() == at::ScalarType::Double) {
            const double* data = tensor.data_ptr<double>();
            for (size_t i = start; i < end; ++i) {
                if (i > start) ss << ", ";
                ss << data[i];
            }
        } else if (tensor.dtype() == at::ScalarType::Int) {
            const int* data = tensor.data_ptr<int>();
            for (size_t i = start; i < end; ++i) {
                if (i > start) ss << ", ";
                ss << data[i];
            }
        } else if (tensor.dtype() == at::ScalarType::Long) {
            const int64_t* data = tensor.data_ptr<int64_t>();
            for (size_t i = start; i < end; ++i) {
                if (i > start) ss << ", ";
                ss << data[i];
            }
        } else {
            ss << "... (unsupported dtype: " << toString(tensor.scalar_type()) << ")";
        }
    }

    // Helper: Dump tensor 全部数据
    static std::string dump_tensor_full(const at::Tensor& tensor) {
        std::ostringstream ss;
        ss << "[";
        dump_tensor_range(tensor, 0, tensor.numel(), ss);
        ss << "]";
        return ss.str();
    }

    // Helper: Dump tensor 采样数据（前 N + 后 N，head/tail 已是拷回 CPU 的小 tensor）
    static std::string dump_tensor_sample(const at::Tensor& head, const at::Tensor& tail) {
        std::ostringstream ss;
        ss << "[";
        dump_tensor_range(head, 0, head.numel(), ss);
        ss << ", ..., ";
        dump_tensor_range(tail, 0, tail.numel(), ss);
        ss << "]";
        return ss.str();
    }

    // 标量类型序列化
    template<typename T>
    static typename std::enable_if<std::is_arithmetic<T>::value, ParamInfo>::type
    serialize(const char* name, T value) {
        ParamInfo info;
        info.name = name;
        info.type = TypeInfo<T>::name();

        std::ostringstream ss;
        ss << value;
        info.value = ss.str();
        info.shape = "[]";
        info.dtype = TypeInfo<T>::name();
        info.bytes = sizeof(T);

        return info;
    }

    // 指针类型序列化 - dump 指针内容
    template<typename T>
    static ParamInfo serialize_pointer(const char* name, T* ptr) {
        ParamInfo info;
        info.name = name;
        info.type = std::string(TypeInfo<T>::name()) + "*";

        std::ostringstream ss;
        ss << "0x" << std::hex << reinterpret_cast<uintptr_t>(ptr);

        // 特殊处理：字符指针（char*, const char*）
        if constexpr (std::is_same_v<T, char> || std::is_same_v<T, const char>) {
            if (ptr != nullptr) {
                ss << " \"" << ptr << "\"";  // Dump 字符串内容
            } else {
                ss << " (null)";
            }
        }
        // 特殊处理：void*
        else if constexpr (std::is_same_v<T, void>) {
            // void* 只能显示地址，无法解引用
        }
        // 其他指针类型：尝试 dump 数组内容（前几个元素）
        else if constexpr (std::is_arithmetic_v<T>) {
            // 数值类型指针：dump 前5个元素
            if (ptr != nullptr) {
                ss << " [";
                for (int i = 0; i < 5; ++i) {
                    if (i > 0) ss << ", ";
                    ss << ptr[i];
                }
                ss << ", ...]";
            } else {
                ss << " (null)";
            }
        } else {
            // 其他类型指针：只显示地址
        }

        info.value = ss.str();
        info.shape = "[]";
        info.dtype = TypeInfo<T>::name();
        info.bytes = sizeof(T*);

        return info;
    }

    // std::string 特化
    static ParamInfo serialize_string(const char* name, const std::string& str) {
        ParamInfo info;
        info.name = name;
        info.type = "std::string";
        info.dtype = "string";
        info.shape = "[]";

        std::ostringstream ss;
        ss << "\"" << str << "\" (length=" << str.size() << ")";
        info.value = ss.str();
        info.bytes = sizeof(std::string) + str.size();

        return info;
    }
};

// JSON生成器
class JsonGenerator {
public:
    static std::string generate(const std::vector<ParamSerializer::ParamInfo>& params,
                                 const char* function_name) {
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"function\": \"" << ParamSerializer::escape_string(function_name) << "\",\n";
        ss << "  \"parameters\": [\n";

        for (size_t i = 0; i < params.size(); ++i) {
            const auto& param = params[i];
            ss << "    {\n";
            ss << "      \"name\": \"" << ParamSerializer::escape_string(param.name) << "\",\n";
            ss << "      \"type\": \"" << ParamSerializer::escape_string(param.type) << "\",\n";
            ss << "      \"dtype\": \"" << ParamSerializer::escape_string(param.dtype) << "\",\n";
            ss << "      \"shape\": \"" << ParamSerializer::escape_string(param.shape) << "\",\n";
            ss << "      \"value\": \"" << ParamSerializer::escape_string(param.value) << "\",\n";
            ss << "      \"bytes\": " << param.bytes << "\n";
            ss << "    }";
            if (i < params.size() - 1) {
                ss << ",";
            }
            ss << "\n";
        }

        ss << "  ]\n";
        ss << "}\n";
        return ss.str();
    }
};

// 获取当前时间戳字符串（秒级）: YYYYMMDD_HHMMSS
inline std::string get_timestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm_buf;
    ::localtime_r(&t, &tm_buf);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_buf);
    return std::string(buf);
}

// 批次目录名：进程内只生成一次，一次运行 = 一个文件夹。
// 用 inline 函数内 static，保证整个进程（跨所有 .cu 编译单元）共享同一个目录。
inline const std::string& batch_dir() {
    static const std::string dir = []() {
        // 若显式指定 MCOP_DEBUG_DUMP_DIR，则所有进程共用该目录（父进程如 testall.py 会带上时间戳）
        std::string d;
        const char* env = std::getenv("MCOP_DEBUG_DUMP_DIR");
        if (env && env[0]) {
            d = env;
        } else {
            d = "mcoplib_op_params_dump_" + get_timestamp();
        }
        ::mkdir(d.c_str(), 0755);
        return d + "/";
    }();
    return dir;
}

// 每个算子最多记录多少次调用（默认 20，第 21 次起不再写）。可通过 MCOP_DEBUG_DUMP_MAX_CALLS 覆盖。
inline size_t get_max_dump_calls() {
    static const size_t limit = []() {
        const char* env = std::getenv("MCOP_DEBUG_DUMP_MAX_CALLS");
        if (!env) return size_t(20);
        try { size_t v = std::stoul(env); return v > 0 ? v : size_t(20); }
        catch (...) { return size_t(20); }
    }();
    return limit;
}

// 每个算子一个计数器，限制最多记录 N 次调用。
// inline 函数内 static 保证跨编译单元共享；mutex 保证多线程调用线程安全。
// 返回本次是第几次（从 1 起），返回 0 表示已超上限、应跳过。
inline size_t record_dump_call(const char* function_name) {
    static std::mutex mtx;
    static std::unordered_map<std::string, size_t> counts;
    std::lock_guard<std::mutex> lock(mtx);
    size_t& c = counts[function_name];
    if (c >= get_max_dump_calls()) return 0;
    return ++c;
}

// 一个算子一个固定文件名（不带序号/时间戳），放在批次目录下
inline std::string make_dump_filename(const char* function_name) {
    return batch_dir() + function_name + ".json";
}

// 文件写入器：append=true 追加，否则 trunc 新建
inline void write_to_file(const std::string& content, const std::string& filename, bool append) {
    std::ofstream file(filename, append ? (std::ios::out | std::ios::app)
                                        : (std::ios::out | std::ios::trunc));
    if (file.is_open()) {
        file << content;
        file.close();
    }
}

// template<typename T>
// struct TypeInfo {
//     static std::string name() {
//         // 如果想在 Linux 下对未知类型也显示得好看，可以使用 abi::__cxa_demangle
//         // 但为了简单且满足你的 int/bool/float 需求，我们主要靠下面的特化
//         return typeid(T).name();
//     }
// };

// // 宏定义简化特化代码
// #define DEFINE_TYPE_NAME(Type, StringName) \
// template<> struct TypeInfo<Type> { static std::string name() { return StringName; } }

// // 常用基础类型特化 (满足你的需求)
// DEFINE_TYPE_NAME(int, "int");
// DEFINE_TYPE_NAME(long, "long");
// DEFINE_TYPE_NAME(float, "float");
// DEFINE_TYPE_NAME(double, "double");
// DEFINE_TYPE_NAME(bool, "bool");
// DEFINE_TYPE_NAME(size_t, "size_t");

// // PyTorch 特有类型特化
// DEFINE_TYPE_NAME(at::BFloat16, "bfloat16");
// DEFINE_TYPE_NAME(at::Half, "half");

// // Tensor 特化 (保留原有的)
// template<>
// struct TypeInfo<at::Tensor> {
//     static std::string name() { return "at::Tensor"; }
// };

// // 1. 新增：定义一个 Type Trait 来检测是否为 std::optional
// template<typename T>
// struct is_optional : std::false_type {};

// template<typename T>
// struct is_optional<std::optional<T>> : std::true_type {};

// 2. 修改：serialize_single_param_impl 函数
// 增加了对 std::optional 的处理逻辑
// 单个参数序列化实现
// template<typename T>
// ParamSerializer::ParamInfo serialize_single_param_impl(const T& arg) {
//     // 使用 decay_t 去除 const 和引用，确保类型判断准确
//     using DecayT = std::decay_t<T>; 

//     if constexpr (std::is_same_v<DecayT, at::Tensor>) {
//         return ParamSerializer::serialize_tensor("", arg);
//     } 
//     else if constexpr (std::is_same_v<DecayT, std::string>) {
//         return ParamSerializer::serialize_string("", arg);
//     } 
//     else if constexpr (std::is_arithmetic_v<DecayT>) {
//         return ParamSerializer::serialize("", arg);
//     } 
//     else if constexpr (std::is_pointer_v<DecayT>) {
//         return ParamSerializer::serialize_pointer("", arg);
//     } 
//     // =======================================================
//     // 修复点：新增对 std::optional 的分支处理
//     // =======================================================
//     else if constexpr (is_optional<DecayT>::value) {
//         if (arg.has_value()) {
//             // 递归调用：如果 optional 有值，取出值并再次序列化
//             // 这样既能支持 optional<int>，也能支持 optional<Tensor>
//             auto info = serialize_single_param_impl(arg.value());
//             info.type = "std::optional<" + info.type + ">"; // 修正类型显示
//             return info;
//         } else {
//             // 如果是 nullopt (无值)
//             ParamSerializer::ParamInfo info;
//             info.type = "std::optional";
//             info.dtype = "nullopt";
//             info.shape = "[]";
//             info.value = "nullopt";
//             info.bytes = 0;
//             return info;
//         }
//     }
//     // =======================================================
//     else {
//         // 其他类型使用std::ostringstream (兜底)
//         ParamSerializer::ParamInfo info;
//         info.type = TypeInfo<T>::name();
//         info.dtype = TypeInfo<T>::name();
//         info.shape = "[]";

//         std::ostringstream ss;
//         // 只有非 optional 且未被前面捕获的类型才会走到这里
//         // 如果这里还报错，说明还有其他不支持 << 的自定义类型
//         ss << arg; 
        
//         info.value = ss.str();
//         info.bytes = sizeof(T);

//         return info;
//     }
// }

// ==========================================
// 修改点 2: 序列化实现逻辑优化
// ==========================================

template<typename T>
ParamSerializer::ParamInfo serialize_single_param_impl(const T& arg) {
    using DecayT = std::decay_t<T>; // 去除引用和const

    if constexpr (std::is_same_v<DecayT, at::Tensor>) {
        return ParamSerializer::serialize_tensor("", arg);
    } 
    else if constexpr (std::is_same_v<DecayT, std::string>) {
        return ParamSerializer::serialize_string("", arg);
    } 
    else if constexpr (std::is_arithmetic_v<DecayT>) {
        // 数值类型：现在会调用我们特化过的 TypeInfo，返回 "int" 而不是 "i"
        return ParamSerializer::serialize("", arg);
    } 
    else if constexpr (std::is_pointer_v<DecayT>) {
        return ParamSerializer::serialize_pointer("", arg);
    } 
    // --- std::optional 处理逻辑优化 ---
    else if constexpr (is_optional<DecayT>::value) {
        // 获取 optional 内部的类型，例如 optional<float> -> InnerType 是 float
        using InnerType = typename DecayT::value_type;
        // 获取内部类型的名称，例如 "float"
        std::string inner_type_name = TypeInfo<InnerType>::name();

        if (arg.has_value()) {
            // 递归序列化内部值
            auto info = serialize_single_param_impl(arg.value());
            
            // 修正 type 字段：std::optional<float>
            info.type = "std::optional<" + inner_type_name + ">";
            // 修正 dtype 字段：确保它显示内部数据的类型 (例如 "float")
            // 如果递归返回的 dtype 已经是正确的，这里其实是冗余的，但为了保险：
            if (info.dtype == "nullopt" || info.dtype.empty()) {
                info.dtype = inner_type_name;
            }
            return info;
        } else {
            // nullopt 情况
            ParamSerializer::ParamInfo info;
            // 即使没有值，我们也知道它的类型定义是 std::optional<float>
            info.type = "std::optional<" + inner_type_name + ">";
            // dtype 应该是内部类型，例如 "float" (需求中 routed_scaling_factor 是 optional<float>)
            info.dtype = inner_type_name; 
            info.shape = "[]";
            info.value = "nullopt";
            info.bytes = 0;
            return info;
        }
    }
    // -------------------------------
    else {
        // 兜底逻辑
        ParamSerializer::ParamInfo info;
        info.type = TypeInfo<T>::name();
        info.dtype = TypeInfo<T>::name();
        info.shape = "[]";

        std::ostringstream ss;
        ss << arg; 
        
        info.value = ss.str();
        info.bytes = sizeof(T);

        return info;
    }
}

// 单个参数序列化分发
template<typename T>
ParamSerializer::ParamInfo serialize_single_param(const T& arg, const char* name = "") {
    ParamSerializer::ParamInfo info = serialize_single_param_impl(arg);
    info.name = name;
    return info;
}

// 使用fold expression收集参数（支持最多20个参数）
template<typename... Args>
auto collect_params(const Args&... args) {
    std::vector<ParamSerializer::ParamInfo> params;
    params.reserve(sizeof...(Args));

    // 使用fold expression展开每个参数
    (void)std::initializer_list<int>{
        (params.push_back(serialize_single_param(args)), 0)...
    };

    return params;
}

// 带参数名的版本
template<typename... Args>
auto collect_params_with_names(const std::vector<const char*>& names, const Args&... args) {
    std::vector<ParamSerializer::ParamInfo> params;
    params.reserve(sizeof...(Args));

    size_t index = 0;
    // 使用fold expression展开每个参数
    (void)std::initializer_list<int>{
        ([&]() {
            auto info = serialize_single_param(args);
            if (index < names.size()) {
                info.name = names[index];
            }
            params.push_back(info);
            ++index;
        }(), 0)...
    };

    return params;
}

// 主要的dump函数
template<typename... Args>
inline void dump_params(const char* function_name,
                       
    
    const std::vector<const char*>& param_names,
                        const Args&... args) {
    if (!is_debug_enabled()) {
        return;  // 零开销：编译器会优化掉
    }

    try {
        // 每个算子最多记录 N 次调用，超限直接跳过（不序列化、零开销）
        size_t call_no = record_dump_call(function_name);
        if (call_no == 0) return;

        // 收集参数信息
        auto params = collect_params_with_names(param_names, args...);

        // 生成JSON
        std::string json = JsonGenerator::generate(params, function_name);

        // 生成文件名（一个算子一个固定文件）
        std::string filename = make_dump_filename(function_name);

        // 写入文件：第 1 次 trunc 新建，之后 append，记录之间留空行
        write_to_file(json + "\n", filename, /*append=*/ call_no > 1);
    } catch (const std::exception& e) {
        // 静默失败，不影响kernel执行
        (void)e;
    }
}

// 简化版本：不带参数名
template<typename... Args>
inline void dump_params_simple(const char* function_name, const Args&... args) {
    if (!is_debug_enabled()) {
        return;  // 零开销：编译器会优化掉
    }
 
    try {
        // 每个算子最多记录 N 次调用，超限直接跳过（不序列化、零开销）
        size_t call_no = record_dump_call(function_name);
        if (call_no == 0) return;

        // 收集参数信息（不带参数名）
        auto params = collect_params(args...);
        // 自动生成参数名 arg0, arg1, ...
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i].name.empty()) {
                params[i].name = "arg" + std::to_string(i);
            }
        }
      
        // 生成JSON
        std::string json = JsonGenerator::generate(params, function_name);
        // 生成文件名
        std::string filename = make_dump_filename(function_name);

        // 写入文件：第 1 次 trunc 新建，之后 append，记录之间留空行
        write_to_file(json + "\n", filename, /*append=*/ call_no > 1);
    } catch (const std::exception& e) {
        // 静默失败，不影响kernel执行
        (void)e;
    }
}

} // namespace debug
} // namespace mcop

// ============================================================================
// 统一的宏接口 - DEBUG_DUMP_PARAMS
// 支持最多 20 个参数，自动生成参数名 arg0, arg1, ...
// ============================================================================

// 获取函数名的宏
#define MCOP_GET_FUNC_NAME() __FUNCTION__

// 参数计数宏（计算可变参数的个数）
#define PP_NARG(...) \
    PP_NARG_(__VA_ARGS__, PP_RSEQ_N())

#define PP_NARG_(...) \
    PP_ARG_N(__VA_ARGS__)

#define PP_ARG_N( \
          _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, \
         _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, \
         N, ...) N

#define PP_RSEQ_N() \
         20, 19, 18, 17, 16, 15, 14, 13, 12, 11, \
         10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

// 宏重载选择器
#define SELECT_MACRO_IMPL(count, ...) MACRO_CONCATENATE_(DEBUG_DUMP_PARAMS_IMPL_, count)(__VA_ARGS__)
#define MACRO_CONCATENATE_(x, y) x ## y

// 1-20 个参数的实现 - 使用字符串化操作符捕获实际变量名
#define DEBUG_DUMP_PARAMS_IMPL_1(x1) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1}; \
            std::vector<const char*> _names_vec(_names, _names + 1); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_2(x1, x2) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2}; \
            std::vector<const char*> _names_vec(_names, _names + 2); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_3(x1, x2, x3) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3}; \
            std::vector<const char*> _names_vec(_names, _names + 3); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_4(x1, x2, x3, x4) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4}; \
            std::vector<const char*> _names_vec(_names, _names + 4); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_5(x1, x2, x3, x4, x5) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5}; \
            std::vector<const char*> _names_vec(_names, _names + 5); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_6(x1, x2, x3, x4, x5, x6) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5, #x6}; \
            std::vector<const char*> _names_vec(_names, _names + 6); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5, x6); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_7(x1, x2, x3, x4, x5, x6, x7) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5, #x6, #x7}; \
            std::vector<const char*> _names_vec(_names, _names + 7); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5, x6, x7); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_8(x1, x2, x3, x4, x5, x6, x7, x8) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5, #x6, #x7, #x8}; \
            std::vector<const char*> _names_vec(_names, _names + 8); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5, x6, x7, x8); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_9(x1, x2, x3, x4, x5, x6, x7, x8, x9) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5, #x6, #x7, #x8, #x9}; \
            std::vector<const char*> _names_vec(_names, _names + 9); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5, x6, x7, x8, x9); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_10(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5, #x6, #x7, #x8, #x9, #x10}; \
            std::vector<const char*> _names_vec(_names, _names + 10); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10); \
        } \
    } while(0)

// 为了节省空间，11-20 个参数的实现 - 使用字符串化操作符捕获实际变量名
#define DEBUG_DUMP_PARAMS_IMPL_11(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5, #x6, #x7, #x8, #x9, #x10, #x11}; \
            std::vector<const char*> _names_vec(_names, _names + 11); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_12(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5, #x6, #x7, #x8, #x9, #x10, #x11, #x12}; \
            std::vector<const char*> _names_vec(_names, _names + 12); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_13(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5, #x6, #x7, #x8, #x9, #x10, #x11, #x12, #x13}; \
            std::vector<const char*> _names_vec(_names, _names + 13); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_14(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5, #x6, #x7, #x8, #x9, #x10, #x11, #x12, #x13, #x14}; \
            std::vector<const char*> _names_vec(_names, _names + 14); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_15(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5, #x6, #x7, #x8, #x9, #x10, #x11, #x12, #x13, #x14, #x15}; \
            std::vector<const char*> _names_vec(_names, _names + 15); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_16(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5, #x6, #x7, #x8, #x9, #x10, #x11, #x12, #x13, #x14, #x15, #x16}; \
            std::vector<const char*> _names_vec(_names, _names + 16); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_17(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5, #x6, #x7, #x8, #x9, #x10, #x11, #x12, #x13, #x14, #x15, #x16, #x17}; \
            std::vector<const char*> _names_vec(_names, _names + 17); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_18(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5, #x6, #x7, #x8, #x9, #x10, #x11, #x12, #x13, #x14, #x15, #x16, #x17, #x18}; \
            std::vector<const char*> _names_vec(_names, _names + 18); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_19(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5, #x6, #x7, #x8, #x9, #x10, #x11, #x12, #x13, #x14, #x15, #x16, #x17, #x18, #x19}; \
            std::vector<const char*> _names_vec(_names, _names + 19); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS_IMPL_20(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20) \
    do { \
        if (::mcop::debug::should_dump(MCOP_GET_FUNC_NAME())) { \
            constexpr const char* _names[] = {#x1, #x2, #x3, #x4, #x5, #x6, #x7, #x8, #x9, #x10, #x11, #x12, #x13, #x14, #x15, #x16, #x17, #x18, #x19, #x20}; \
            std::vector<const char*> _names_vec(_names, _names + 20); \
            ::mcop::debug::dump_params(MCOP_GET_FUNC_NAME(), _names_vec, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20); \
        } \
    } while(0)

#define DEBUG_DUMP_PARAMS(...) \
    SELECT_MACRO_IMPL(PP_NARG(__VA_ARGS__), __VA_ARGS__)


