#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace tf {

[[noreturn]] inline void die(const std::string& msg) {
    throw std::runtime_error(msg);
}

inline void check(bool cond, const std::string& msg) {
    if (!cond) die(msg);
}

inline uint16_t rd_u16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}
inline uint32_t rd_u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
inline uint64_t rd_u64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}
inline void wr_u16(uint8_t* p, uint16_t v) { std::memcpy(p, &v, 2); }
inline void wr_u32(uint8_t* p, uint32_t v) { std::memcpy(p, &v, 4); }
inline void wr_u64(uint8_t* p, uint64_t v) { std::memcpy(p, &v, 8); }

// SplitMix64. Used for init and sampling. Not cryptographic.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t u32() { return static_cast<uint32_t>(next() >> 32); }
    // [0, 1)
    float uni() { return static_cast<float>(next() >> 40) * (1.0f / 16777216.0f); }
    float uni(float a, float b) { return a + (b - a) * uni(); }
    int uni_int(int n) { return static_cast<int>(next() % static_cast<uint64_t>(n)); }
    float normal() {
        float u1 = std::max(uni(), 1e-12f);
        float u2 = uni();
        return std::sqrt(-2.f * std::log(u1)) * std::cos(6.283185307179586f * u2);
    }
};

inline int hardware_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n ? static_cast<int>(n) : 1;
}

// Chunked parallel loop. `fn(begin, end)` covers a half-open index range.
template <class Fn>
void parallel_for_range(int64_t n, Fn fn, int threads = 0) {
    if (n <= 0) return;
    if (threads <= 0) threads = hardware_threads();
    if (threads > n) threads = static_cast<int>(n);
    if (threads <= 1 || n < 4096) {
        fn(int64_t{0}, n);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    std::exception_ptr eptr = nullptr;
    std::vector<std::exception_ptr> errs(threads);
    int64_t chunk = (n + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
        int64_t a = std::min(n, t * chunk);
        int64_t b = std::min(n, a + chunk);
        if (t == threads - 1) {
            try {
                fn(a, b);
            } catch (...) {
                errs[t] = std::current_exception();
            }
        } else {
            pool.emplace_back([&, t, a, b] {
                try {
                    fn(a, b);
                } catch (...) {
                    errs[t] = std::current_exception();
                }
            });
        }
    }
    for (auto& th : pool) th.join();
    for (auto& e : errs) {
        if (e) std::rethrow_exception(e);
    }
}

inline float dot(const float* a, const float* b, int n) {
    float s = 0.f;
    int i = 0;
    for (; i + 3 < n; i += 4) s += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

}  // namespace tf
