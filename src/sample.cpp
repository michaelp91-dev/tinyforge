#include "tf_bpe.hpp"
#include "tf_ckpt.hpp"
#include "tf_model.hpp"
#include "tf_util.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    try {
        const char* ckpt = nullptr;
        const char* tokp = nullptr;
        const char* prompt = "The capital of France is";
        const char* backend = "cpu";
        int ntok = 64, topk = 40;
        float temp = 0.8f;
        uint64_t seed = 1;
        for (int i = 1; i + 1 < argc; ++i) {
            if (!std::strcmp(argv[i], "--ckpt")) ckpt = argv[++i];
            else if (!std::strcmp(argv[i], "--tokenizer")) tokp = argv[++i];
            else if (!std::strcmp(argv[i], "--prompt")) prompt = argv[++i];
            else if (!std::strcmp(argv[i], "--tokens")) ntok = std::atoi(argv[++i]);
            else if (!std::strcmp(argv[i], "--temp")) temp = std::atof(argv[++i]);
            else if (!std::strcmp(argv[i], "--topk")) topk = std::atoi(argv[++i]);
            else if (!std::strcmp(argv[i], "--seed")) seed = std::strtoull(argv[++i], nullptr, 10);
            else if (!std::strcmp(argv[i], "--backend")) backend = argv[++i];
        }
        tf::check(ckpt && tokp, "usage: tf_sample --ckpt ckpt.bin --tokenizer tok.bin [--prompt ...] [--backend cpu|cuda]");
        tf::Tokenizer tok;
        tok.load(tokp);
        tf::Checkpoint c = tf::load_checkpoint(ckpt);
        tf::Engine* eng = std::strcmp(backend, "cuda") == 0 ? tf::create_engine_cuda(c.cfg, tok.token_bytes.data())
                                                           : tf::create_engine_cpu(c.cfg, tok.token_bytes.data());
        tf::check(eng, "CUDA engine unavailable");
        eng->load_params(c.params);
        auto ids = tok.encode(prompt, true);
        auto out = eng->generate(ids, ntok, temp, topk, seed);
        std::string text = tok.decode(out);
        std::printf("%s\n", text.c_str());
        delete eng;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
