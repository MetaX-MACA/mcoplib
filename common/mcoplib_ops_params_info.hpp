#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <optional>
#include <mutex>
#include <unordered_map>
#include <ATen/ATen.h> // 确保包含 ATen

// ==========================================
// Part 1: 辅助工具函数 (保持不变)
// ==========================================
namespace debug_utils {
    // 第一层开关（默认关）：只有显式设为 1/ON/on 才 trace
    inline bool is_trace_enabled() {
        static const bool enabled = []() {
            const char* env = std::getenv("MCOP_DEBUG_TRACE");
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

    // 每个算子最多打印多少次调用（默认 20，第 21 次起不再打印）。
    // 与 dump 共用 MCOP_DEBUG_DUMP_MAX_CALLS 这个上限，保证终端和 json 打印的是同一批前 N 次。
    inline size_t get_max_trace_calls() {
        static const size_t limit = []() {
            const char* env = std::getenv("MCOP_DEBUG_DUMP_MAX_CALLS");
            if (!env) return size_t(20);
            try { size_t v = std::stoul(env); return v > 0 ? v : size_t(20); }
            catch (...) { return size_t(20); }
        }();
        return limit;
    }

    // 每个算子一个计数器，限制最多打印 N 次。
    // inline 函数内 static 保证跨编译单元共享；mutex 保证多线程调用线程安全。
    // 返回本次是第几次（从 1 起），返回 0 表示已超上限、应跳过。
    inline size_t record_trace_call(const char* function_name) {
        static std::mutex mtx;
        static std::unordered_map<std::string, size_t> counts;
        std::lock_guard<std::mutex> lock(mtx);
        size_t& c = counts[function_name];
        if (c >= get_max_trace_calls()) return 0;
        return ++c;
    }

    // 是否该对这个算子输出 trace：第一层开关 && 第二层过滤 && 未超调用次数上限
    inline bool should_trace(const char* func_name) {
        if (!is_trace_enabled()) return false;
        if (!name_matches(func_name)) return false;
        return record_trace_call(func_name) != 0;
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
#define GET_ARG_COUNT(...) GET_ARG_COUNT_INNER(__VA_ARGS__, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define GET_ARG_COUNT_INNER(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, N, ...) N

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

// 4. 最终分发宏
#define FOR_EACH_(N, ...) CONCAT(FOR_EACH_, N)(__VA_ARGS__)
#define FOR_EACH(...) FOR_EACH_(GET_ARG_COUNT(__VA_ARGS__), __VA_ARGS__)

// 5. 用户接口宏
#define DEBUG_TRACE_PARAMS(...) \
    do { \
        if (debug_utils::should_trace(__func__)) { \
            std::cout << "[MCOP_DEBUG] Call: " << __func__ << "\n"; \
            FOR_EACH(__VA_ARGS__) \
            std::cout << std::endl; \
        } \
    } while (0)