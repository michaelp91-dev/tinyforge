#include "tf_bpe.hpp"
#include "tf_parquet.hpp"
#include "tf_util.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    try {
        int vocab = 32768;
        int64_t max_bytes = 100'000'000;
        int threads = 0;
        const char* out = "tokenizer.bin";
        std::vector<std::string> shards;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--vocab") && i + 1 < argc) vocab = std::atoi(argv[++i]);
            else if (!std::strcmp(argv[i], "--max-bytes") && i + 1 < argc) max_bytes = std::atoll(argv[++i]);
            else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
            else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
            else shards.emplace_back(argv[i]);
        }
        tf::check(!shards.empty(), "usage: tf_tok_train [--vocab 32768] [--max-bytes N] [--out tokenizer.bin] shard.parquet ...");
        std::string text;
        text.reserve((size_t)std::min(max_bytes, (int64_t)64 << 20));
        for (auto& shard : shards) {
            if ((int64_t)text.size() >= max_bytes) break;
            tf::ParquetFile pf(shard);
            std::fprintf(stderr, "reading %s (%d row groups)\n", shard.c_str(), pf.num_row_groups());
            for (int g = 0; g < pf.num_row_groups() && (int64_t)text.size() < max_bytes; ++g) {
                auto docs = pf.read_row_group(g);
                for (auto& d : docs) {
                    if ((int64_t)text.size() >= max_bytes) break;
                    text.append(d);
                    text.push_back('\n');
                }
            }
        }
        std::fprintf(stderr, "training BPE on %zu bytes, vocab %d\n", text.size(), vocab);
        tf::Tokenizer tok;
        tok.train(text, vocab, threads);
        tok.save(out);
        std::fprintf(stderr, "wrote %s (vocab %d, bos %d)\n", out, tok.vocab_size, tok.bos_id);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
