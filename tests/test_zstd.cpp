#include "tf_zstd.hpp"
#include "tf_util.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

static std::vector<uint8_t> slurp(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) tf::die(std::string("cannot open ") + path);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_zstd file.zst [expected]\n");
        return 2;
    }
    auto src = slurp(argv[1]);
    std::vector<uint8_t> out;
    try {
        tf::zstd_decompress(src.data(), src.size(), out);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "decompress failed: %s\n", e.what());
        return 1;
    }
    if (argc >= 3) {
        auto exp = slurp(argv[2]);
        if (exp != out) {
            std::fprintf(stderr, "mismatch got %zu expected %zu\n", out.size(), exp.size());
            size_t n = std::min(out.size(), exp.size());
            for (size_t i = 0; i < n; ++i) {
                if (out[i] != exp[i]) {
                    std::fprintf(stderr, "first diff at %zu got %02x exp %02x\n", i, out[i], exp[i]);
                    break;
                }
            }
            return 1;
        }
    }
    std::fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
