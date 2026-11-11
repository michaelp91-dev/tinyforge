#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tf {

// Decompress a single zstd frame (or a concatenation of frames) from [src, src+src_len).
// Supports the subset ClimbMix / parquet-cpp actually emits: no dictionaries,
// blocks up to the spec maximum (min(window, 128 KiB)), Huffman and FSE.
// If expected_size != 0, the output must be exactly that many bytes.
void zstd_decompress(const uint8_t* src, size_t src_len, std::vector<uint8_t>& out, size_t expected_size = 0);

}  // namespace tf
