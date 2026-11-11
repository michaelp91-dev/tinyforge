#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tf {

// GPT-4-style pre-tokenization, nanochat's variant (numbers are 1–2 digits):
//   '(?i:[sdmt]|ll|ve|re)
//   |[^\r\n\p{L}\p{N}]?+\p{L}+
//   |\p{N}{1,2}
//   | ?[^\s\p{L}\p{N}]++[\r\n]*
//   |\s*[\r\n]
//   |\s+(?!\S)
//   |\s+
// Splits cover the input exactly. Each span is [begin, end) in bytes.
std::vector<std::pair<size_t, size_t>> pretok_split(const uint8_t* s, size_t n);
inline std::vector<std::pair<size_t, size_t>> pretok_split(const std::string& s) {
    return pretok_split(reinterpret_cast<const uint8_t*>(s.data()), s.size());
}

}  // namespace tf
