#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tf {

// GPT config. Defaults are the nanochat d6 geometry: depth 6, width 384,
// head dim 128 (3 heads), context 2048, vocab 32768.
struct GPTConfig {
    int n_layer = 6;
    int n_embd = 384;
    int n_head = 3;
    int n_kv_head = 3;
    int seq_len = 2048;
    int vocab = 32768;
    float rope_base = 100000.f;
    float softcap = 15.f;
    float qk_scale = 1.2f;
    float rms_eps = 1e-6f;
    std::string window_pattern = "SSSL";
    // 0 keeps the nanochat-style derived learning rates.
    float matrix_lr = 0.02f;
    float embedding_lr = 0.3f;
    float unembedding_lr = 0.008f;
    float scalar_lr = 0.5f;
    float weight_decay = 0.28f;
    float smear_lr = 0.2f;
    int warmup_steps = 40;
    float warmdown_ratio = 0.65f;
    float final_lr_frac = 0.05f;
    float data_ratio = 20.f;
    uint64_t seed = 1337;
};

struct LayerOff {
    int q = 0, k = 0, v = 0, proj = 0, fc = 0, fc_proj = 0;
    int ve = -1;        // vocab * kv_dim, or -1
    int ve_gate = -1;   // n_kv * 12, or -1
    int q_n = 0, k_n = 0, v_n = 0, proj_n = 0, fc_n = 0, fc_proj_n = 0, ve_n = 0, gate_n = 0;
    int window = 0;  // how many tokens to the left (inclusive of that distance)
    bool has_ve = false;
};

struct Layout {
    GPTConfig cfg;
    int head_dim = 0;
    int kv_dim = 0;
    int mlp_dim = 0;
    int wte = 0, wte_n = 0;
    int lm = 0, lm_n = 0;
    int resid = 0;  // n_layer
    int x0 = 0;
    int smear_lambda = 0;
    int backout = 0;
    int smear_gate = 0;  // 24
    std::vector<LayerOff> layers;
    int n_param = 0;
    // Muon matrices are every 2D tensor inside the blocks (including ve_gate).
    // Embeddings, lm_head, and the scalars/smear gate are AdamW.
    int n_scaling = 0;  // block matrices + lm_head, the nanochat ratio denominator
    int n_matmul = 0;
};

Layout make_layout(GPTConfig cfg);

// Bits-per-byte from a sum of nats over tokens whose byte length is > 0.
inline double bits_per_byte(double nats, int64_t bytes) {
    if (bytes <= 0) return 0;
    return nats / (0.6931471805599453 * static_cast<double>(bytes));
}

struct TrainSched {
    float lr_mult = 1.f;
    float muon_momentum = 0.95f;
    float muon_wd = 0.f;
    float embedding_lr = 0.f;
    float unembedding_lr = 0.f;
    float value_lr = 0.f;
    float resid_lr = 0.f;
    float x0_lr = 0.f;
    float smear_lr = 0.f;
    float matrix_lr = 0.f;
};

// nanochat's batch/depth scaling, evaluated at this model's shape.
TrainSched schedule_for(const Layout& L, int64_t total_batch, int step, int num_steps);

int64_t scaling_params(const Layout& L);
// Training FLOPs per token, nanochat's accounting (6 * matmul params + attention).
double flops_per_token(const Layout& L);

// Parameter vector helpers shared by CPU and CUDA engines.
void init_params(const Layout& L, float* params, uint64_t seed);

class Engine {
public:
    virtual ~Engine() = default;
    virtual const Layout& layout() const = 0;
    virtual void zero_grad() = 0;
    // scale multiplies the loss before backward (gradient accumulation).
    virtual void train_step(const int32_t* x, const int32_t* y, int B, int T, float scale, float& loss) = 0;
    virtual void optim_step(const TrainSched& sch) = 0;
    // nats/bytes use token_bytes[id] (0 skips the token, e.g. specials).
    virtual void eval_forward(const int32_t* x, const int32_t* y, int B, int T, const int* token_bytes, double& nats,
                              int64_t& bytes) = 0;
    virtual std::vector<int32_t> generate(const std::vector<int32_t>& prompt, int max_new, float temperature, int top_k,
                                          uint64_t seed) = 0;
    virtual void copy_params(std::vector<float>& host) const = 0;
    virtual void copy_grads(std::vector<float>& host) const = 0;
    virtual void load_params(const std::vector<float>& host) = 0;
    virtual void save_optim(const std::string& path) const = 0;
    virtual void load_optim(const std::string& path) = 0;
};

Engine* create_engine_cpu(const GPTConfig& cfg, const int* token_bytes);
// Defined in the CUDA translation unit. Returns null if this binary has no CUDA.
Engine* create_engine_cuda(const GPTConfig& cfg, const int* token_bytes);

}  // namespace tf
