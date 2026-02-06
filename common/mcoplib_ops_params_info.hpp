#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <optional>
#include <ATen/ATen.h> // 确保包含 ATen
#include <utility>
#include <iostream>

#ifdef scalar_type
#undef scalar_type
#endif

// ==========================================
// Part 1: 辅助工具函数 (保持不变)
// ==========================================
namespace std {
    // 为 tuple 提供流支持
    template<class... Args>
    std::ostream& operator<<(std::ostream& os, const std::tuple<Args...>& t) {
        os << "(";
        std::apply([&os](auto&&... args) {
            size_t n = 0;
            ((os << (n++ == 0 ? "" : ", ") << args), ...);
        }, t);
        return os << ")";
    }
}
namespace debug_utils {
    inline bool is_trace_enabled() {
        static const char* env_p = std::getenv("MCOP_DEBUG_TRACE");
        static const bool enabled = (env_p != nullptr && (std::string(env_p) == "1" || std::string(env_p) == "ON"));
        return enabled;
    }

    template <typename T>
    void print_value(std::ostream& os, const T& val) {
        os << val;
    }

    // 针对 Tensor 的特化
    inline void print_value(std::ostream& os, const at::Tensor& tensor) {
        if (tensor.defined()) {
            os << "Tensor(Shape=[";
            auto sizes = tensor.sizes();
            for (size_t i = 0; i < sizes.size(); ++i) {
                os << sizes[i] << (i < sizes.size() - 1 ? ", " : "");
            }
            os << "], Dtype=" << tensor.scalar_type() 
               << ", Device=" << tensor.device() << ")";
        } else {
            os << "Tensor(Undefined)";
        }
    }

    // 针对 Optional 的特化
    template <typename T>
    void print_value(std::ostream& os, const std::optional<T>& opt) {
        if (opt.has_value()) {
            os << "Optional(";
            print_value(os, opt.value());
            os << ")";
        } else {
            os << "Optional(nullopt)";
        }
    }

    template <typename T>
    void log_argument(const char* arg_name, const T& arg_value, bool is_last) {
        std::cout << "  " << arg_name << " = ";
        print_value(std::cout, arg_value);
        if (!is_last) std::cout << ",\n";
    }
}

// ==========================================
// Part 2: 宏定义 (已扩展支持 20 个参数)
// ==========================================

// 1. 计数器宏：支持自动推导 0~20 个参数
#define GET_ARG_COUNT(...) GET_ARG_COUNT_INNER(__VA_ARGS__, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define GET_ARG_COUNT_INNER(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, N, ...) N

// 2. 拼接宏
#define CONCAT(A, B) CONCAT_INNER(A, B)
#define CONCAT_INNER(A, B) A ## B

// 3. 递归展开宏 (扩展至 20)
// 最后一个参数 (is_last=true)
#define FOR_EACH_1(x)      debug_utils::log_argument(#x, x, true);

// 中间参数 (is_last=false)，然后递归调用 N-1
#define FOR_EACH_2(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_1(__VA_ARGS__)
#define FOR_EACH_3(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_2(__VA_ARGS__)
#define FOR_EACH_4(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_3(__VA_ARGS__)
#define FOR_EACH_5(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_4(__VA_ARGS__)
#define FOR_EACH_6(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_5(__VA_ARGS__)
#define FOR_EACH_7(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_6(__VA_ARGS__)
#define FOR_EACH_8(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_7(__VA_ARGS__)
#define FOR_EACH_9(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_8(__VA_ARGS__)
#define FOR_EACH_10(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_9(__VA_ARGS__)
#define FOR_EACH_11(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_10(__VA_ARGS__)
#define FOR_EACH_12(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_11(__VA_ARGS__)
#define FOR_EACH_13(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_12(__VA_ARGS__)
#define FOR_EACH_14(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_13(__VA_ARGS__)
#define FOR_EACH_15(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_14(__VA_ARGS__)
#define FOR_EACH_16(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_15(__VA_ARGS__)
#define FOR_EACH_17(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_16(__VA_ARGS__)
#define FOR_EACH_18(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_17(__VA_ARGS__)
#define FOR_EACH_19(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_18(__VA_ARGS__)
#define FOR_EACH_20(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_19(__VA_ARGS__)
#define FOR_EACH_21(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_20(__VA_ARGS__)
#define FOR_EACH_22(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_21(__VA_ARGS__)
#define FOR_EACH_23(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_22(__VA_ARGS__)
#define FOR_EACH_24(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_23(__VA_ARGS__)
#define FOR_EACH_25(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_24(__VA_ARGS__)
#define FOR_EACH_26(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_25(__VA_ARGS__)
#define FOR_EACH_27(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_26(__VA_ARGS__)
#define FOR_EACH_28(x, ...) debug_utils::log_argument(#x, x, false); FOR_EACH_27(__VA_ARGS__)

// 4. 最终分发宏
#define FOR_EACH_(N, ...) CONCAT(FOR_EACH_, N)(__VA_ARGS__)
#define FOR_EACH(...) FOR_EACH_(GET_ARG_COUNT(__VA_ARGS__), __VA_ARGS__)

// 5. 用户接口宏
#define DEBUG_TRACE_PARAMS(...) \
    do { \
        if (debug_utils::is_trace_enabled()) { \
            std::cout << "[MCOP_DEBUG] Call: " << __func__ << "\n"; \
            FOR_EACH(__VA_ARGS__) \
            std::cout << std::endl; \
        } \
    } while (0)