#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tf {

// Byte-level BPE. Special tokens are appended after the merge vocabulary,
// nanochat-style, and are not learned. <|bos|> is the first special token.
class Tokenizer {
public:
    static constexpr int kNumSpecial = 9;
    static const char* special_name(int i);

    int vocab_size = 0;
    int bos_id = 0;
    std::vector<std::string> tokens;   // raw bytes, or the special-token spelling
    std::vector<int> token_bytes;      // UTF-8 length; 0 for specials (excluded from bpb)
    std::unordered_map<uint64_t, int> merge_to;  // (left<<32|right) -> new id

    void train(const uint8_t* data, size_t n, int vocab, int threads);
    void train(const std::string& text, int vocab, int threads) {
        train(reinterpret_cast<const uint8_t*>(text.data()), text.size(), vocab, threads);
    }

    std::vector<int32_t> encode(const uint8_t* data, size_t n, bool prepend_bos) const;
    std::vector<int32_t> encode(const std::string& text, bool prepend_bos) const {
        return encode(reinterpret_cast<const uint8_t*>(text.data()), text.size(), prepend_bos);
    }
    // Parallel over documents. Each document is encoded independently.
    void encode_batch(const std::vector<std::string>& docs, std::vector<std::vector<int32_t>>& out, bool prepend_bos,
                      int threads) const;

    std::string decode(const int32_t* ids, size_t n) const;
    std::string decode(const std::vector<int32_t>& ids) const { return decode(ids.data(), ids.size()); }

    void save(const std::string& path) const;
    void load(const std::string& path);
};

}  // namespace tf
