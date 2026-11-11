#include "tf_ckpt.hpp"
#include "tf_data.hpp"
#include "tf_model.hpp"
#include "tf_bpe.hpp"
#include "tf_util.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

static int argi(int argc, char** argv, const char* key, int def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return std::atoi(argv[i + 1]);
    return def;
}
static float argf(int argc, char** argv, const char* key, float def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return std::atof(argv[i + 1]);
    return def;
}
static const char* args(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
}
static bool flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return true;
    return false;
}

int main(int argc, char** argv) {
    try {
        tf::GPTConfig cfg;
        cfg.n_layer = argi(argc, argv, "--depth", 6);
        int dim = argi(argc, argv, "--dim", 0);
        int head_dim = argi(argc, argv, "--head-dim", 128);
        if (dim <= 0) {
            int base = cfg.n_layer * 64;
            dim = ((base + head_dim - 1) / head_dim) * head_dim;
        }
        cfg.n_embd = dim;
        int heads = argi(argc, argv, "--heads", 0);
        cfg.n_head = heads > 0 ? heads : dim / head_dim;
        cfg.n_kv_head = argi(argc, argv, "--kv-heads", cfg.n_head);
        cfg.seq_len = argi(argc, argv, "--seq", 2048);
        cfg.vocab = argi(argc, argv, "--vocab", 32768);
        cfg.window_pattern = args(argc, argv, "--window", "SSSL");
        cfg.seed = (uint64_t)argi(argc, argv, "--seed", 1337);
        cfg.data_ratio = argf(argc, argv, "--ratio", 20.f);
        cfg.matrix_lr = argf(argc, argv, "--matrix-lr", cfg.matrix_lr);
        cfg.embedding_lr = argf(argc, argv, "--embedding-lr", cfg.embedding_lr);
        cfg.unembedding_lr = argf(argc, argv, "--unembedding-lr", cfg.unembedding_lr);
        cfg.weight_decay = argf(argc, argv, "--wd", cfg.weight_decay);
        cfg.warmup_steps = argi(argc, argv, "--warmup", cfg.warmup_steps);

        const char* backend = args(argc, argv, "--backend", "cpu");
        const char* tok_path = args(argc, argv, "--tokenizer", "");
        const char* data_path = args(argc, argv, "--data", "");
        const char* val_path = args(argc, argv, "--val", "");
        const char* out_dir = args(argc, argv, "--out", "runs/d6");
        const char* resume = args(argc, argv, "--resume", "");
        bool synthetic = flag(argc, argv, "--synthetic");
        int device_batch = argi(argc, argv, "--device-batch", 4);
        int total_batch_arg = argi(argc, argv, "--total-batch", 0);
        int steps_arg = argi(argc, argv, "--steps", 0);
        int eval_every = argi(argc, argv, "--eval-every", 0);
        int eval_tokens = argi(argc, argv, "--eval-tokens", 0);
        int save_every = argi(argc, argv, "--save-every", 0);
        double peak_flops = argf(argc, argv, "--peak-flops", 312e12f);

        tf::Tokenizer tok;
        std::vector<int> token_bytes(cfg.vocab, 1);
        if (!synthetic) {
            tf::check(tok_path[0], "--tokenizer is required unless --synthetic");
            tok.load(tok_path);
            cfg.vocab = tok.vocab_size;
            token_bytes = tok.token_bytes;
        } else {
            token_bytes.assign(cfg.vocab, 1);
        }

        tf::Layout layout = tf::make_layout(cfg);
        tf::GPTConfig d12 = cfg;
        d12.n_layer = 12;
        int hd = layout.head_dim;
        int base = 12 * 64;
        int ddim = ((base + hd - 1) / hd) * hd;
        d12.n_embd = ddim;
        d12.n_head = ddim / hd;
        d12.n_kv_head = d12.n_head;
        int64_t d12_scaling = tf::make_layout(d12).n_scaling;
        int64_t total_batch = total_batch_arg;
        if (total_batch <= 0) {
            double ratio = (double)layout.n_scaling / (double)d12_scaling;
            double predicted = 524288.0 * std::pow(ratio, 0.383);
            int64_t p2 = 1;
            while (p2 * 2 <= predicted * 1.414) p2 *= 2;  // nearest power of two, rough
            // pick the closer of p2 and p2*2
            total_batch = (std::fabs(predicted - p2) < std::fabs(predicted - p2 * 2)) ? p2 : p2 * 2;
            if (total_batch < cfg.seq_len) total_batch = cfg.seq_len;
        }
        int micro = device_batch * cfg.seq_len;
        tf::check(total_batch % micro == 0, "total batch must be a multiple of device_batch * seq");
        int accum = (int)(total_batch / micro);
        int64_t target_tokens = (int64_t)(cfg.data_ratio * layout.n_scaling);
        int steps = steps_arg > 0 ? steps_arg : (int)(target_tokens / total_batch);
        if (steps < 1) steps = 1;

        std::fprintf(stderr, "tinyforge %s  depth=%d dim=%d heads=%d kv=%d seq=%d vocab=%d\n", backend, cfg.n_layer, cfg.n_embd,
                     cfg.n_head, cfg.n_kv_head, cfg.seq_len, cfg.vocab);
        std::fprintf(stderr, "params %d  scaling %d  flops/token %.3e\n", layout.n_param, layout.n_scaling, tf::flops_per_token(layout));
        std::fprintf(stderr, "batch %lld (micro %d x accum %d)  steps %d  tokens %.3e  ratio %.2f\n", (long long)total_batch, micro, accum,
                     steps, (double)total_batch * steps, (double)total_batch * steps / layout.n_scaling);

        tf::Engine* eng = nullptr;
        if (std::strcmp(backend, "cuda") == 0) {
            eng = tf::create_engine_cuda(cfg, token_bytes.data());
            tf::check(eng, "this binary has no CUDA engine; build with `make cuda`");
        } else {
            eng = tf::create_engine_cpu(cfg, token_bytes.data());
        }
        int step0 = 0;
        tf::TokenCorpus data, val;
        tf::BestFitLoader* loader = nullptr;
        tf::BestFitLoader* vloader = nullptr;
        if (!synthetic) {
            tf::check(data_path[0], "--data is required unless --synthetic");
            data.load(data_path);
            loader = new tf::BestFitLoader(data, 256);
            if (val_path[0]) {
                val.load(val_path);
                vloader = new tf::BestFitLoader(val, 128);
            }
        }
        if (resume[0]) {
            tf::Checkpoint ck = tf::load_checkpoint(resume);
            eng->load_params(ck.params);
            step0 = ck.step;
            if (loader) loader->seek_docs(ck.doc_cursor);
            std::string op = std::string(resume) + ".optim";
            FILE* f = std::fopen(op.c_str(), "rb");
            if (f) {
                std::fclose(f);
                eng->load_optim(op);
            }
            std::fprintf(stderr, "resumed from step %d\n", step0);
        }

        std::string mkdir = std::string("mkdir -p ") + out_dir;
        if (std::system(mkdir.c_str()) != 0) { /* best effort */ }
        std::string log_path = std::string(out_dir) + "/train.log";
        FILE* log = std::fopen(log_path.c_str(), step0 ? "a" : "w");
        std::vector<int32_t> x, y;
        tf::Rng rng(cfg.seed + 99);
        double t_train = 0;
        for (int step = step0; step < steps; ++step) {
            tf::TrainSched sch = tf::schedule_for(eng->layout(), total_batch, step, steps);
            eng->zero_grad();
            double t0 = 0;
            timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            t0 = ts.tv_sec + ts.tv_nsec * 1e-9;
            float loss_acc = 0;
            for (int a = 0; a < accum; ++a) {
                if (synthetic) {
                    x.resize((size_t)device_batch * cfg.seq_len);
                    y.resize(x.size());
                    for (size_t i = 0; i < x.size(); ++i) {
                        x[i] = rng.uni_int(cfg.vocab);
                        y[i] = rng.uni_int(cfg.vocab);
                    }
                } else {
                    loader->next(x, y, device_batch, cfg.seq_len);
                }
                float loss = 0;
                eng->train_step(x.data(), y.data(), device_batch, cfg.seq_len, 1.f / accum, loss);
                loss_acc += loss / accum;
            }
            eng->optim_step(sch);
            clock_gettime(CLOCK_MONOTONIC, &ts);
            double t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
            double dt = t1 - t0;
            if (step > 2) t_train += dt;
            double tps = total_batch / dt;
            double flops = tf::flops_per_token(eng->layout()) * total_batch / dt;
            double mfu = 100.0 * flops / peak_flops;
            std::fprintf(stderr, "step %05d/%d  loss %.4f  lr x%.3f  %.0f tok/s  mfu %.1f%%  %.2fs\n", step, steps, loss_acc, sch.lr_mult, tps,
                         mfu, dt);
            if (log)
                std::fprintf(log, "%d\t%.6f\t%.6f\t%.1f\t%.3f\n", step, loss_acc, sch.lr_mult, tps, mfu);
            bool do_eval = vloader && eval_every > 0 && ((step + 1) % eval_every == 0 || step + 1 == steps);
            if (do_eval) {
                int et = eval_tokens > 0 ? eval_tokens : device_batch * cfg.seq_len * 4;
                int eb = et / (device_batch * cfg.seq_len);
                if (eb < 1) eb = 1;
                double nats = 0;
                int64_t bytes = 0;
                for (int i = 0; i < eb; ++i) {
                    vloader->next(x, y, device_batch, cfg.seq_len);
                    eng->eval_forward(x.data(), y.data(), device_batch, cfg.seq_len, token_bytes.data(), nats, bytes);
                }
                double bpb = tf::bits_per_byte(nats, bytes);
                std::fprintf(stderr, "  val bpb %.4f over %lld bytes\n", bpb, (long long)bytes);
                if (log) std::fprintf(log, "eval %d\t%.6f\t%lld\n", step + 1, bpb, (long long)bytes);
            }
            bool do_save = (save_every > 0 && (step + 1) % save_every == 0) || step + 1 == steps;
            if (do_save) {
                tf::Checkpoint ck;
                ck.cfg = cfg;
                ck.step = step + 1;
                ck.doc_cursor = loader ? loader->docs_consumed() : 0;
                eng->copy_params(ck.params);
                std::string cp = std::string(out_dir) + "/ckpt.bin";
                tf::save_checkpoint(cp, ck);
                eng->save_optim(cp + ".optim");
                std::fprintf(stderr, "  saved %s\n", cp.c_str());
            }
            if (log) std::fflush(log);
        }
        if (log) std::fclose(log);
        std::fprintf(stderr, "done. train time after warmup %.1f min\n", t_train / 60.0);
        delete loader;
        delete vloader;
        delete eng;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
