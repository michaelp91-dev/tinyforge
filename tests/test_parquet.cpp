#include "tf_parquet.hpp"
#include "tf_util.hpp"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_parquet shard.parquet\n");
        return 2;
    }
    tf::ParquetFile pf(argv[1]);
    std::printf("rows %lld groups %d\n", (long long)pf.num_rows(), pf.num_row_groups());
    if (pf.num_row_groups() < 1 || pf.num_rows() < 1) {
        std::fprintf(stderr, "empty parquet\n");
        return 1;
    }
    auto docs = pf.read_row_group(0);
    std::printf("group0 docs %zu\n", docs.size());
    if (docs.empty()) return 1;
    int nonempty = 0;
    size_t bytes = 0;
    for (auto& d : docs) {
        bytes += d.size();
        if (!d.empty()) ++nonempty;
    }
    std::printf("nonempty %d bytes %zu\n", nonempty, bytes);
    std::printf("doc0: %.120s\n", docs[0].c_str());
    if (nonempty < docs.size() / 2) {
        std::fprintf(stderr, "too many empty docs\n");
        return 1;
    }
    // A second group, to catch offset bugs.
    if (pf.num_row_groups() > 1) {
        auto d2 = pf.read_row_group(1);
        if (d2.empty()) return 1;
        std::printf("group1 docs %zu first %.80s\n", d2.size(), d2[0].c_str());
    }
    if (pf.num_row_groups() > 2) {
        auto last = pf.read_row_group(pf.num_row_groups() - 1);
        std::printf("last group docs %zu\n", last.size());
        if (last.empty()) return 1;
    }
    std::printf("parquet tests passed\n");
    return 0;
}
