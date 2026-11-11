#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tf {

// Reads the single optional BYTE_ARRAY column produced by parquet-cpp-arrow for
// karpathy/climbmix-400b-shuffle: PLAIN encoding, ZSTD (or uncompressed) pages,
// one data page per row group. Other layouts throw.
class ParquetFile {
public:
    explicit ParquetFile(const std::string& path);
    ~ParquetFile();
    ParquetFile(const ParquetFile&) = delete;
    ParquetFile& operator=(const ParquetFile&) = delete;

    int64_t num_rows() const { return num_rows_; }
    int num_row_groups() const { return static_cast<int>(groups_.size()); }
    // Documents in one row group, in file order. Nulls become empty strings.
    std::vector<std::string> read_row_group(int index) const;

private:
    struct Group {
        int64_t num_rows = 0;
        int64_t data_offset = 0;
        int64_t compressed = 0;
        int64_t uncompressed = 0;
        int codec = 0;  // 0 uncompressed, 6 zstd
    };
    int fd_ = -1;
    const uint8_t* map_ = nullptr;
    size_t size_ = 0;
    int64_t num_rows_ = 0;
    std::vector<Group> groups_;
};

}  // namespace tf
