#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tf {

// Flat uint16 token file. Every document already starts with BOS.
struct TokenCorpus {
    std::vector<uint16_t> tokens;
    std::vector<uint64_t> offsets;  // size = num_docs + 1, in tokens
    int vocab = 0;
    int bos = 0;
    void save(const std::string& path) const;
    void load(const std::string& path);
    int64_t num_docs() const { return (int64_t)offsets.size() - 1; }
};

// BOS-aligned best-fit packer. Every row is exactly T+1 tokens with no padding;
// a document that does not fit is cropped (the tail is dropped), matching nanochat.
class BestFitLoader {
public:
    BestFitLoader(const TokenCorpus& corpus, int buffer_docs = 1000);
    void next(std::vector<int32_t>& x, std::vector<int32_t>& y, int B, int T);
    uint64_t docs_consumed() const { return consumed_; }
    void seek_docs(uint64_t doc_index);

private:
    const TokenCorpus* corpus_;
    std::vector<std::vector<int32_t>> buf_;
    uint64_t next_doc_ = 0;
    uint64_t consumed_ = 0;
    int buffer_docs_;
    void refill();
};

}  // namespace tf
