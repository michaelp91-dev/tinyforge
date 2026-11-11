#include "tf_bpe.hpp"
#include "tf_data.hpp"
#include "tf_parquet.hpp"
#include "tf_util.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        const char* tok_path = nullptr;
        const char* out = "tokens.bin";
        int threads = 0;
        std::vector<std::string> shards;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--tokenizer") && i + 1 < argc) tok_path = argv[++i];
            else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
            else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
            else shards.emplace_back(argv[i]);
        }
        tf::check(tok_path && !shards.empty(), "usage: tf_tok_encode --tokenizer tokenizer.bin --out tokens.bin shard.parquet ...");
        tf::Tokenizer tok;
        tok.load(tok_path);
        tf::TokenCorpus corpus;
        corpus.vocab = tok.vocab_size;
        corpus.bos = tok.bos_id;
        corpus.offsets.push_back(0);
        for (auto& shard : shards) {
            tf::ParquetFile pf(shard);
            std::fprintf(stderr, "encoding %s (%d groups, %lld rows)\n", shard.c_str(), pf.num_row_groups(), (long long)pf.num_rows());
            for (int g = 0; g < pf.num_row_groups(); ++g) {
                auto docs = pf.read_row_group(g);
                std::vector<std::vector<int32_t>> ids;
                tok.encode_batch(docs, ids, true, threads);
                for (auto& row : ids) {
                    for (int id : row) {
                        tf::check(id >= 0 && id < 65536, "token id does not fit in uint16");
                        corpus.tokens.push_back((uint16_t)id);
                    }
                    corpus.offsets.push_back(corpus.tokens.size());
                }
                std::fprintf(stderr, "  group %d/%d  docs %zu  tokens %zu\n", g + 1, pf.num_row_groups(), docs.size(), corpus.tokens.size());
            }
        }
        corpus.save(out);
        std::fprintf(stderr, "wrote %s  docs %lld  tokens %zu\n", out, (long long)corpus.num_docs(), corpus.tokens.size());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
