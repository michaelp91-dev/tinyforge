#include "tf_data.hpp"
#include "tf_util.hpp"

#include <cstring>
#include <fstream>

namespace tf {

void TokenCorpus::save(const std::string& path) const {
    std::ofstream o(path, std::ios::binary);
    check(o.good(), "cannot write token corpus");
    o.write("TFBIN001", 8);
    uint32_t v = vocab, b = bos;
    uint64_t nd = num_docs(), nt = tokens.size();
    o.write((char*)&v, 4);
    o.write((char*)&b, 4);
    o.write((char*)&nd, 8);
    o.write((char*)&nt, 8);
    o.write((char*)tokens.data(), sizeof(uint16_t) * tokens.size());
    o.write((char*)offsets.data(), sizeof(uint64_t) * offsets.size());
}

void TokenCorpus::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    check(in.good(), "cannot read token corpus " + path);
    char magic[8];
    in.read(magic, 8);
    check(std::memcmp(magic, "TFBIN001", 8) == 0, "bad token corpus magic");
    uint32_t v = 0, b = 0;
    uint64_t nd = 0, nt = 0;
    in.read((char*)&v, 4);
    in.read((char*)&b, 4);
    in.read((char*)&nd, 8);
    in.read((char*)&nt, 8);
    vocab = v;
    bos = b;
    tokens.resize(nt);
    offsets.resize(nd + 1);
    in.read((char*)tokens.data(), sizeof(uint16_t) * nt);
    in.read((char*)offsets.data(), sizeof(uint64_t) * (nd + 1));
    check(in.good(), "truncated token corpus");
}

BestFitLoader::BestFitLoader(const TokenCorpus& corpus, int buffer_docs) : corpus_(&corpus), buffer_docs_(buffer_docs) {}

void BestFitLoader::seek_docs(uint64_t doc_index) {
    next_doc_ = doc_index % std::max<uint64_t>(corpus_->num_docs(), 1);
    consumed_ = doc_index;
    buf_.clear();
}

void BestFitLoader::refill() {
    int64_t nd = corpus_->num_docs();
    check(nd > 0, "token corpus is empty");
    while ((int)buf_.size() < buffer_docs_) {
        if (next_doc_ >= (uint64_t)nd) next_doc_ = 0;
        uint64_t a = corpus_->offsets[next_doc_];
        uint64_t b = corpus_->offsets[next_doc_ + 1];
        std::vector<int32_t> doc;
        doc.reserve(b - a);
        for (uint64_t i = a; i < b; ++i) doc.push_back(corpus_->tokens[i]);
        if (!doc.empty()) buf_.push_back(std::move(doc));
        ++next_doc_;
    }
}

void BestFitLoader::next(std::vector<int32_t>& x, std::vector<int32_t>& y, int B, int T) {
    int cap = T + 1;
    x.assign((size_t)B * T, 0);
    y.assign((size_t)B * T, 0);
    std::vector<int32_t> row(cap);
    for (int r = 0; r < B; ++r) {
        int pos = 0;
        while (pos < cap) {
            if ((int)buf_.size() < buffer_docs_) refill();
            int remain = cap - pos;
            int best = -1, best_len = 0;
            for (int i = 0; i < (int)buf_.size(); ++i) {
                int n = (int)buf_[i].size();
                if (n <= remain && n > best_len) {
                    best = i;
                    best_len = n;
                }
            }
            if (best >= 0) {
                std::memcpy(&row[pos], buf_[best].data(), sizeof(int32_t) * best_len);
                pos += best_len;
                buf_.erase(buf_.begin() + best);
                ++consumed_;
            } else {
                int s = 0;
                for (int i = 1; i < (int)buf_.size(); ++i)
                    if (buf_[i].size() < buf_[s].size()) s = i;
                int n = std::min(remain, (int)buf_[s].size());
                std::memcpy(&row[pos], buf_[s].data(), sizeof(int32_t) * n);
                pos += n;
                buf_.erase(buf_.begin() + s);
                ++consumed_;
            }
        }
        std::memcpy(&x[(size_t)r * T], row.data(), sizeof(int32_t) * T);
        std::memcpy(&y[(size_t)r * T], row.data() + 1, sizeof(int32_t) * T);
    }
}

}  // namespace tf
