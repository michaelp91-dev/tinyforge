#include "tf_bpe.hpp"
#include "tf_pretok.hpp"
#include "tf_util.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <queue>
#include <unordered_map>

namespace tf {
namespace {

const char* kSpecial[Tokenizer::kNumSpecial] = {
    "<|bos|>",           "<|user_start|>",  "<|user_end|>",     "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>",  "<|python_end|>",  "<|output_start|>", "<|output_end|>",
};

uint64_t pair_key(uint32_t a, uint32_t b) { return (uint64_t(a) << 32) | b; }

struct HeapItem {
    int64_t count;
    uint32_t left, right;
    bool operator<(const HeapItem& o) const {
        if (count != o.count) return count < o.count;
        if (left != o.left) return left > o.left;
        return right > o.right;
    }
};

}  // namespace

const char* Tokenizer::special_name(int i) { return kSpecial[i]; }

void Tokenizer::train(const uint8_t* data, size_t n, int vocab, int threads) {
    check(vocab >= 256 + kNumSpecial, "vocab must cover bytes plus special tokens");
    int n_merge = vocab - 256 - kNumSpecial;
    if (threads <= 0) threads = hardware_threads();

    std::unordered_map<std::string, int64_t> freq;
    if (n > 0 && threads > 1) {
        std::vector<std::unordered_map<std::string, int64_t>> parts(threads);
        parallel_for_range(static_cast<int64_t>(threads), [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; ++t) {
                size_t begin = static_cast<size_t>(t) * n / threads;
                size_t end = static_cast<size_t>(t + 1) * n / threads;
                if (t + 1 != threads) {
                    while (end < n && data[end] != '\n') ++end;
                }
                if (t > 0) {
                    while (begin < end && data[begin] != '\n') ++begin;
                    if (begin < end && data[begin] == '\n') ++begin;
                }
                auto spans = pretok_split(data + begin, end - begin);
                auto& m = parts[t];
                for (auto [s0, s1] : spans) {
                    m[std::string(reinterpret_cast<const char*>(data + begin + s0), s1 - s0)] += 1;
                }
            }
        }, threads);
        for (auto& m : parts) {
            for (auto& kv : m) freq[kv.first] += kv.second;
        }
    } else {
        auto spans = pretok_split(data, n);
        for (auto [s0, s1] : spans) freq[std::string(reinterpret_cast<const char*>(data + s0), s1 - s0)] += 1;
    }

    struct Word {
        std::vector<int> sym;
        int64_t freq = 0;
    };
    std::vector<Word> words;
    words.reserve(freq.size());
    for (auto& kv : freq) {
        Word w;
        w.freq = kv.second;
        w.sym.reserve(kv.first.size());
        for (unsigned char c : kv.first) w.sym.push_back(c);
        if (w.sym.size() >= 2) words.push_back(std::move(w));
    }
    freq.clear();

    std::unordered_map<uint64_t, int64_t> counts;
    std::unordered_map<uint64_t, std::vector<int>> locs;
    counts.reserve(words.size() * 2);
    std::priority_queue<HeapItem> heap;
    auto touch = [&](uint32_t L, uint32_t R, int64_t delta, int widx, bool locate) {
        uint64_t k = pair_key(L, R);
        int64_t& c = counts[k];
        c += delta;
        if (locate) locs[k].push_back(widx);
        if (c > 0) heap.push(HeapItem{c, L, R});
    };
    for (int wi = 0; wi < static_cast<int>(words.size()); ++wi) {
        const auto& sym = words[wi].sym;
        for (size_t i = 0; i + 1 < sym.size(); ++i) touch(sym[i], sym[i + 1], words[wi].freq, wi, true);
    }

    merge_to.clear();
    std::vector<std::pair<uint32_t, uint32_t>> ordered;
    ordered.reserve(n_merge);
    std::vector<int> seen(words.size(), 0);
    int stamp = 1;
    std::fprintf(stderr, "bpe: %zu unique pieces, training %d merges\n", words.size(), n_merge);
    for (int m = 0; m < n_merge; ++m) {
        uint32_t best_l = 0, best_r = 0;
        int64_t best_c = 0;
        bool found = false;
        while (!heap.empty()) {
            HeapItem it = heap.top();
            heap.pop();
            uint64_t k = pair_key(it.left, it.right);
            auto cit = counts.find(k);
            if (cit == counts.end() || cit->second != it.count || it.count <= 0) continue;
            best_l = it.left;
            best_r = it.right;
            best_c = it.count;
            found = true;
            break;
        }
        if (!found) break;
        int new_id = 256 + m;
        merge_to[pair_key(best_l, best_r)] = new_id;
        ordered.emplace_back(best_l, best_r);
        auto lit = locs.find(pair_key(best_l, best_r));
        if (lit != locs.end()) {
            for (int wi : lit->second) {
                if (wi < 0 || static_cast<size_t>(wi) >= words.size()) continue;
                if (seen[wi] == stamp) continue;
                seen[wi] = stamp;
                Word& w = words[wi];
                bool has = false;
                for (size_t i = 0; i + 1 < w.sym.size(); ++i)
                    if (w.sym[i] == static_cast<int>(best_l) && w.sym[i + 1] == static_cast<int>(best_r)) {
                        has = true;
                        break;
                    }
                if (!has) continue;
                for (size_t i = 0; i + 1 < w.sym.size(); ++i) touch(w.sym[i], w.sym[i + 1], -w.freq, wi, false);
                std::vector<int> neu;
                neu.reserve(w.sym.size());
                for (size_t i = 0; i < w.sym.size();) {
                    if (i + 1 < w.sym.size() && w.sym[i] == static_cast<int>(best_l) &&
                        w.sym[i + 1] == static_cast<int>(best_r)) {
                        neu.push_back(new_id);
                        i += 2;
                    } else {
                        neu.push_back(w.sym[i]);
                        ++i;
                    }
                }
                w.sym.swap(neu);
                for (size_t i = 0; i + 1 < w.sym.size(); ++i) touch(w.sym[i], w.sym[i + 1], w.freq, wi, true);
            }
        }
        ++stamp;
        if (stamp == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            stamp = 1;
        }
        if ((m + 1) % 2000 == 0) std::fprintf(stderr, "bpe: merge %d/%d (count %lld)\n", m + 1, n_merge, (long long)best_c);
        (void)best_c;
    }

    if (static_cast<int>(ordered.size()) < n_merge) {
        std::fprintf(stderr, "bpe: corpus exhausted after %zu merges (requested %d)\n", ordered.size(), n_merge);
    }
    vocab_size = 256 + static_cast<int>(ordered.size()) + kNumSpecial;
    bos_id = 256 + static_cast<int>(ordered.size());
    tokens.assign(vocab_size, {});
    token_bytes.assign(vocab_size, 0);
    for (int i = 0; i < 256; ++i) {
        tokens[i] = std::string(1, static_cast<char>(i));
        token_bytes[i] = 1;
    }
    for (int i = 0; i < static_cast<int>(ordered.size()); ++i) {
        int id = 256 + i;
        uint32_t L = ordered[i].first, R = ordered[i].second;
        tokens[id] = tokens[L] + tokens[R];
        token_bytes[id] = static_cast<int>(tokens[id].size());
    }
    for (int i = 0; i < kNumSpecial; ++i) {
        tokens[bos_id + i] = kSpecial[i];
        token_bytes[bos_id + i] = 0;
    }
    std::fprintf(stderr, "bpe: vocab %d ready (bos id %d)\n", vocab_size, bos_id);
}

std::vector<int32_t> Tokenizer::encode(const uint8_t* data, size_t n, bool prepend_bos) const {
    check(vocab_size > 0, "tokenizer is not loaded");
    std::vector<int32_t> out;
    if (prepend_bos) out.push_back(bos_id);
    size_t i = 0;
    auto encode_span = [&](size_t b, size_t e) {
        if (b >= e) return;
        auto pieces = pretok_split(data + b, e - b);
        for (auto [s0, s1] : pieces) {
            std::vector<int> sym;
            sym.reserve(s1 - s0);
            for (size_t k = s0; k < s1; ++k) sym.push_back(data[b + k]);
            while (sym.size() >= 2) {
                int best_id = vocab_size + 1;
                int best_i = -1;
                for (size_t t = 0; t + 1 < sym.size(); ++t) {
                    auto it = merge_to.find(pair_key(sym[t], sym[t + 1]));
                    if (it != merge_to.end() && it->second < best_id) {
                        best_id = it->second;
                        best_i = static_cast<int>(t);
                    }
                }
                if (best_i < 0) break;
                sym[best_i] = best_id;
                sym.erase(sym.begin() + best_i + 1);
            }
            for (int id : sym) out.push_back(id);
        }
    };
    while (i < n) {
        int which = -1;
        size_t slen = 0;
        for (int s = 0; s < kNumSpecial; ++s) {
            size_t L = std::strlen(kSpecial[s]);
            if (i + L <= n && std::memcmp(data + i, kSpecial[s], L) == 0 && L > slen) {
                which = s;
                slen = L;
            }
        }
        if (which >= 0) {
            out.push_back(bos_id + which);
            i += slen;
            continue;
        }
        // Take a run up to the next special token.
        size_t j = i + 1;
        while (j < n) {
            bool hit = false;
            for (int s = 0; s < kNumSpecial; ++s) {
                size_t L = std::strlen(kSpecial[s]);
                if (j + L <= n && std::memcmp(data + j, kSpecial[s], L) == 0) {
                    hit = true;
                    break;
                }
            }
            if (hit) break;
            ++j;
        }
        encode_span(i, j);
        i = j;
    }
    return out;
}

void Tokenizer::encode_batch(const std::vector<std::string>& docs, std::vector<std::vector<int32_t>>& out, bool prepend_bos,
                             int threads) const {
    out.resize(docs.size());
    parallel_for_range(static_cast<int64_t>(docs.size()), [&](int64_t a, int64_t b) {
        for (int64_t i = a; i < b; ++i) out[i] = encode(docs[i], prepend_bos);
    }, threads);
}

std::string Tokenizer::decode(const int32_t* ids, size_t n) const {
    std::string s;
    for (size_t i = 0; i < n; ++i) {
        int id = ids[i];
        check(id >= 0 && id < vocab_size, "token id out of range");
        s += tokens[id];
    }
    return s;
}

void Tokenizer::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    check(f, "cannot write tokenizer");
    char magic[8] = {'T', 'F', 'T', 'K', '0', '0', '0', '1'};
    std::fwrite(magic, 1, 8, f);
    uint32_t v = vocab_size, nb = static_cast<uint32_t>(bos_id - 256), ns = kNumSpecial;
    std::fwrite(&v, 4, 1, f);
    std::fwrite(&nb, 4, 1, f);
    std::fwrite(&ns, 4, 1, f);
    std::vector<std::pair<uint32_t, uint32_t>> merges(nb);
    for (const auto& kv : merge_to) {
        int id = kv.second;
        check(id >= 256 && id < bos_id, "bad merge id");
        merges[id - 256] = {static_cast<uint32_t>(kv.first >> 32), static_cast<uint32_t>(kv.first)};
    }
    for (auto [L, R] : merges) {
        std::fwrite(&L, 4, 1, f);
        std::fwrite(&R, 4, 1, f);
    }
    std::fclose(f);
}

void Tokenizer::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    check(f, "cannot read tokenizer " + path);
    char magic[8];
    check(std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "TFTK0001", 8) == 0, "bad tokenizer magic");
    uint32_t v = 0, nb = 0, ns = 0;
    check(std::fread(&v, 4, 1, f) == 1 && std::fread(&nb, 4, 1, f) == 1 && std::fread(&ns, 4, 1, f) == 1, "truncated tokenizer");
    check(ns == kNumSpecial, "unexpected special-token count");
    vocab_size = static_cast<int>(v);
    bos_id = vocab_size - kNumSpecial;
    check(bos_id == 256 + static_cast<int>(nb), "tokenizer header is inconsistent");
    merge_to.clear();
    tokens.assign(vocab_size, {});
    token_bytes.assign(vocab_size, 0);
    for (int i = 0; i < 256; ++i) {
        tokens[i] = std::string(1, static_cast<char>(i));
        token_bytes[i] = 1;
    }
    for (uint32_t i = 0; i < nb; ++i) {
        uint32_t L = 0, R = 0;
        check(std::fread(&L, 4, 1, f) == 1 && std::fread(&R, 4, 1, f) == 1, "truncated merges");
        int id = 256 + static_cast<int>(i);
        check(L < static_cast<uint32_t>(id) && R < static_cast<uint32_t>(id), "merge refers to a future token");
        tokens[id] = tokens[L] + tokens[R];
        token_bytes[id] = static_cast<int>(tokens[id].size());
        merge_to[pair_key(L, R)] = id;
    }
    for (int i = 0; i < kNumSpecial; ++i) {
        tokens[bos_id + i] = kSpecial[i];
        token_bytes[bos_id + i] = 0;
    }
    std::fclose(f);
}

}  // namespace tf
