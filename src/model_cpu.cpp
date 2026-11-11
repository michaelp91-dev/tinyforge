#include "tf_model.hpp"
#include "tf_ops.hpp"
#include "tf_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace tf {
namespace {

int div_up_window(int seq) {
    // nanochat: short_window = -(-seq // 4 // 128) * 128, which floors seq/4 to a multiple of 128
    // via the double negation of floor division. For positive seq it equals ((seq/4)/128)*128.
    int q = seq / 4;
    return (q / 128) * 128;
}

void add_scaled(float* dst, const float* src, int n, float s) {
    for (int i = 0; i < n; ++i) dst[i] += src[i] * s;
}

void polar_express(std::vector<float>& X, int rows, int cols) {
    static const double coeff[5][3] = {
        {8.156554524902461, -22.48329292557795, 15.878769915207462},
        {4.042929935166739, -2.808917465908714, 0.5000178451051316},
        {3.8916678022926607, -2.772484153217685, 0.5060648178503393},
        {3.285753657755655, -2.3681294933425376, 0.46449024233003106},
        {2.3465413258596377, -1.7097828382687081, 0.42323551169305323},
    };
    auto gemm_sq = [&](const std::vector<float>& A, std::vector<float>& C, int n) {
        // C = A @ A, both n x n.
        std::fill(C.begin(), C.end(), 0.f);
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < n; ++k) {
                float a = A[static_cast<int64_t>(i) * n + k];
                for (int j = 0; j < n; ++j) C[static_cast<int64_t>(i) * n + j] += a * A[static_cast<int64_t>(k) * n + j];
            }
    };
    bool tall = rows > cols;
    int nside = tall ? cols : rows;
    std::vector<float> A(static_cast<size_t>(nside) * nside), AA(A.size()), B(A.size()), Y(X.size());
    for (int step = 0; step < 5; ++step) {
        double a = coeff[step][0], b = coeff[step][1], c = coeff[step][2];
        if (tall) {
            // A = X^T @ X  (cols, cols)
            std::fill(A.begin(), A.end(), 0.f);
            for (int r = 0; r < rows; ++r)
                for (int i = 0; i < cols; ++i) {
                    float x = X[static_cast<int64_t>(r) * cols + i];
                    for (int j = 0; j < cols; ++j) A[static_cast<int64_t>(i) * cols + j] += x * X[static_cast<int64_t>(r) * cols + j];
                }
        } else {
            // A = X @ X^T (rows, rows)
            std::fill(A.begin(), A.end(), 0.f);
            for (int i = 0; i < rows; ++i)
                for (int k = 0; k < cols; ++k) {
                    float x = X[static_cast<int64_t>(i) * cols + k];
                    for (int j = 0; j < rows; ++j) A[static_cast<int64_t>(i) * rows + j] += x * X[static_cast<int64_t>(j) * cols + k];
                }
        }
        gemm_sq(A, AA, nside);
        for (int i = 0; i < nside * nside; ++i) B[i] = static_cast<float>(b) * A[i] + static_cast<float>(c) * AA[i];
        if (tall) {
            // Y = a X + X @ B
            for (int r = 0; r < rows; ++r)
                for (int j = 0; j < cols; ++j) {
                    float s = 0.f;
                    for (int k = 0; k < cols; ++k) s += X[static_cast<int64_t>(r) * cols + k] * B[static_cast<int64_t>(k) * cols + j];
                    Y[static_cast<int64_t>(r) * cols + j] = static_cast<float>(a) * X[static_cast<int64_t>(r) * cols + j] + s;
                }
        } else {
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j) {
                    float s = 0.f;
                    for (int k = 0; k < rows; ++k) s += B[static_cast<int64_t>(i) * rows + k] * X[static_cast<int64_t>(k) * cols + j];
                    Y[static_cast<int64_t>(i) * cols + j] = static_cast<float>(a) * X[static_cast<int64_t>(i) * cols + j] + s;
                }
        }
        X.swap(Y);
    }
}

void muon_update(float* p, float* g, float* mom, float* sec, int rows, int cols, float momentum, float lr, float wd) {
    const float beta2 = 0.9f;
    int n = rows * cols;
    float one_m = 1.f - momentum;
    for (int i = 0; i < n; ++i) mom[i] = mom[i] * momentum + g[i] * one_m;
    for (int i = 0; i < n; ++i) g[i] = g[i] * one_m + mom[i] * momentum;
    std::vector<float> X(g, g + n);
    double fro2 = 0;
    for (float v : X) fro2 += static_cast<double>(v) * v;
    float fro = std::sqrt(static_cast<float>(fro2));
    float target = fro / std::sqrt(static_cast<float>(rows));
    for (int r = 0; r < rows; ++r) {
        double rn2 = 0;
        for (int c = 0; c < cols; ++c) {
            float v = X[static_cast<int64_t>(r) * cols + c];
            rn2 += static_cast<double>(v) * v;
        }
        float rn = std::sqrt(static_cast<float>(std::max(rn2, 1e-12)));
        float s = target / rn;
        for (int c = 0; c < cols; ++c) X[static_cast<int64_t>(r) * cols + c] *= s;
    }
    fro2 = 0;
    for (float v : X) fro2 += static_cast<double>(v) * v;
    fro = std::sqrt(static_cast<float>(fro2));
    float invn = 1.f / (fro * 1.01f + 1e-6f);
    for (float& v : X) v *= invn;
    if (rows > 1 && cols > 1) polar_express(X, rows, cols);
    fro2 = 0;
    for (float v : X) fro2 += static_cast<double>(v) * v;
    fro = std::sqrt(static_cast<float>(std::max(fro2, 1e-12)));
    float renorm = std::sqrt(static_cast<float>(std::min(rows, cols))) / fro;
    for (float& v : X) v *= renorm;

    bool per_row = rows >= cols;
    int red = per_row ? cols : rows;
    int nstat = per_row ? rows : cols;
    std::vector<float> vmean(nstat);
    if (per_row) {
        for (int r = 0; r < rows; ++r) {
            double s = 0;
            for (int c = 0; c < cols; ++c) {
                float v = X[static_cast<int64_t>(r) * cols + c];
                s += static_cast<double>(v) * v;
            }
            vmean[r] = static_cast<float>(s / cols);
        }
    } else {
        for (int c = 0; c < cols; ++c) {
            double s = 0;
            for (int r = 0; r < rows; ++r) {
                float v = X[static_cast<int64_t>(r) * cols + c];
                s += static_cast<double>(v) * v;
            }
            vmean[c] = static_cast<float>(s / rows);
        }
    }
    double vnorm_sq = 0;
    for (float v : vmean) vnorm_sq += v;
    vnorm_sq *= red;
    float vnorm = std::sqrt(static_cast<float>(vnorm_sq));
    for (int i = 0; i < nstat; ++i) sec[i] = sec[i] * beta2 + vmean[i] * (1.f - beta2);
    std::vector<float> step(nstat), scaled(nstat);
    double new_sum = 0;
    for (int i = 0; i < nstat; ++i) {
        step[i] = 1.f / std::sqrt(std::max(sec[i], 1e-10f));
        scaled[i] = (vmean[i] * static_cast<float>(red)) * step[i] * step[i];
        new_sum += scaled[i];
    }
    float vnorm_new = std::sqrt(static_cast<float>(std::max(new_sum, 1e-20)));
    float ratio = vnorm / std::max(vnorm_new, 1e-10f);
    float lr_eff = lr * std::sqrt(std::max(1.f, static_cast<float>(rows) / static_cast<float>(cols)));
    if (per_row) {
        for (int r = 0; r < rows; ++r) {
            float fs = step[r] * ratio;
            for (int c = 0; c < cols; ++c) {
                int i = r * cols + c;
                float upd = X[i] * fs;
                float mask = (upd * p[i] >= 0.f) ? 1.f : 0.f;
                p[i] -= lr_eff * upd + lr_eff * wd * p[i] * mask;
            }
        }
    } else {
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                int i = r * cols + c;
                float upd = X[i] * step[c] * ratio;
                float mask = (upd * p[i] >= 0.f) ? 1.f : 0.f;
                p[i] -= lr_eff * upd + lr_eff * wd * p[i] * mask;
            }
    }
}

void adamw(float* p, const float* g, float* m, float* v, int n, float lr, float b1, float b2, float eps, float wd, int step) {
    float bc1 = 1.f - std::pow(b1, static_cast<float>(step));
    float bc2 = 1.f - std::pow(b2, static_cast<float>(step));
    float step_size = lr / bc1;
    for (int i = 0; i < n; ++i) {
        p[i] *= (1.f - lr * wd);
        m[i] = m[i] * b1 + g[i] * (1.f - b1);
        v[i] = v[i] * b2 + g[i] * g[i] * (1.f - b2);
        float denom = std::sqrt(v[i] / bc2) + eps;
        p[i] -= step_size * m[i] / denom;
    }
}

struct AdamSlot {
    int off = 0, n = 0;
    float b1 = 0.8f, b2 = 0.95f, eps = 1e-10f, wd = 0.f;
    int kind = 0;  // 0 emb, 1 lm, 2 ve, 3 resid, 4 x0, 5 smear
    std::vector<float> m, v;
    int step = 0;
};

struct MuonSlot {
    int off = 0, rows = 0, cols = 0;
    std::vector<float> mom, sec;
};

}  // namespace

Layout make_layout(GPTConfig cfg) {
    check(cfg.n_layer >= 1 && cfg.n_layer <= 64, "bad depth");
    check(cfg.n_embd >= 24 && cfg.n_embd % cfg.n_head == 0, "width must divide into heads and be >= 24");
    check(cfg.n_kv_head > 0 && cfg.n_head % cfg.n_kv_head == 0, "kv heads must divide query heads");
    check(cfg.vocab >= 2 && cfg.seq_len >= 2, "bad vocab or sequence");
    Layout L;
    L.cfg = cfg;
    L.head_dim = cfg.n_embd / cfg.n_head;
    L.kv_dim = cfg.n_kv_head * L.head_dim;
    L.mlp_dim = 4 * cfg.n_embd;
    int at = 0;
    auto take = [&](int n) {
        int o = at;
        at += n;
        return o;
    };
    L.wte_n = cfg.vocab * cfg.n_embd;
    L.wte = take(L.wte_n);
    L.lm_n = cfg.vocab * cfg.n_embd;
    L.lm = take(L.lm_n);
    L.resid = take(cfg.n_layer);
    L.x0 = take(cfg.n_layer);
    L.smear_lambda = take(1);
    L.backout = take(1);
    L.smear_gate = take(24);
    L.layers.resize(cfg.n_layer);
    int long_w = cfg.seq_len;
    int short_w = div_up_window(cfg.seq_len);
    if (short_w < 1) short_w = long_w;
    std::string pat = cfg.window_pattern.empty() ? "L" : cfg.window_pattern;
    for (char c : pat) check(c == 'S' || c == 'L' || c == 's' || c == 'l', "window pattern must be S/L");
    for (int i = 0; i < cfg.n_layer; ++i) {
        LayerOff& a = L.layers[i];
        char ch = pat[i % static_cast<int>(pat.size())];
        if (ch == 's') ch = 'S';
        if (ch == 'l') ch = 'L';
        a.window = (ch == 'S') ? short_w : long_w;
        a.q_n = cfg.n_embd * cfg.n_embd;
        a.k_n = L.kv_dim * cfg.n_embd;
        a.v_n = a.k_n;
        a.proj_n = cfg.n_embd * cfg.n_embd;
        a.fc_n = L.mlp_dim * cfg.n_embd;
        a.fc_proj_n = cfg.n_embd * L.mlp_dim;
        a.q = take(a.q_n);
        a.k = take(a.k_n);
        a.v = take(a.v_n);
        a.proj = take(a.proj_n);
        a.fc = take(a.fc_n);
        a.fc_proj = take(a.fc_proj_n);
        a.has_ve = (i % 2) == ((cfg.n_layer - 1) % 2);
        if (a.has_ve) {
            a.ve_n = cfg.vocab * L.kv_dim;
            a.gate_n = cfg.n_kv_head * 12;
            a.ve = take(a.ve_n);
            a.ve_gate = take(a.gate_n);
        }
    }
    L.layers.back().window = long_w;
    L.n_param = at;
    int matmul = L.lm_n;
    for (auto& a : L.layers) matmul += a.q_n + a.k_n + a.v_n + a.proj_n + a.fc_n + a.fc_proj_n + a.gate_n;
    L.n_scaling = matmul;
    L.n_matmul = matmul;
    return L;
}

int64_t scaling_params(const Layout& L) { return L.n_scaling; }

double flops_per_token(const Layout& L) {
    double attn = 0;
    int H = L.cfg.n_head, D = L.head_dim, T = L.cfg.seq_len;
    for (auto& a : L.layers) {
        int w = std::min(a.window, T);
        attn += 12.0 * H * D * w;
    }
    return 6.0 * static_cast<double>(L.n_matmul) + attn;
}

TrainSched schedule_for(const Layout& L, int64_t total_batch, int step, int num_steps) {
    GPTConfig ref = L.cfg;
    ref.n_layer = 12;
    int hd = L.head_dim;
    int base = 12 * 64;
    int dim = ((base + hd - 1) / hd) * hd;
    ref.n_embd = dim;
    ref.n_head = dim / hd;
    ref.n_kv_head = ref.n_head;
    Layout d12 = make_layout(ref);
    double batch_scale = std::sqrt(static_cast<double>(total_batch) / 524288.0);
    double dmodel_scale = std::pow(static_cast<double>(L.cfg.n_embd) / 768.0, -0.5);
    double wd_scaled = static_cast<double>(L.cfg.weight_decay) * batch_scale *
                       (static_cast<double>(d12.n_scaling) / static_cast<double>(L.n_scaling));
    int warmdown = static_cast<int>(std::llround(L.cfg.warmdown_ratio * num_steps));
    if (warmdown < 1) warmdown = 1;
    float lrm;
    if (step < L.cfg.warmup_steps) lrm = (step + 1.f) / static_cast<float>(std::max(L.cfg.warmup_steps, 1));
    else if (step <= num_steps - warmdown) lrm = 1.f;
    else {
        float progress = (num_steps - step) / static_cast<float>(warmdown);
        lrm = progress + (1.f - progress) * L.cfg.final_lr_frac;
    }
    int warmdown_start = num_steps - warmdown;
    float mom;
    if (step < 400) {
        float frac = step / 400.f;
        mom = (1.f - frac) * 0.85f + frac * 0.97f;
    } else if (step >= warmdown_start) {
        float progress = (step - warmdown_start) / static_cast<float>(warmdown);
        mom = 0.97f * (1.f - progress) + 0.90f * progress;
    } else mom = 0.97f;
    float wd = static_cast<float>(wd_scaled * 0.5 * (1.0 + std::cos(3.141592653589793 * step / std::max(num_steps, 1))));
    TrainSched s;
    s.lr_mult = lrm;
    s.muon_momentum = mom;
    s.muon_wd = wd;
    s.embedding_lr = static_cast<float>(L.cfg.embedding_lr * batch_scale * dmodel_scale) * lrm;
    s.unembedding_lr = static_cast<float>(L.cfg.unembedding_lr * batch_scale * dmodel_scale) * lrm;
    s.value_lr = s.embedding_lr * 0.5f;
    s.resid_lr = static_cast<float>(L.cfg.scalar_lr * batch_scale * 0.01) * lrm;
    s.x0_lr = static_cast<float>(L.cfg.scalar_lr * batch_scale) * lrm;
    s.smear_lr = L.cfg.smear_lr * lrm;
    s.matrix_lr = static_cast<float>(L.cfg.matrix_lr * batch_scale) * lrm;
    return s;
}

void init_params(const Layout& L, float* p, uint64_t seed) {
    std::memset(p, 0, sizeof(float) * L.n_param);
    Rng rng(seed);
    int C = L.cfg.n_embd;
    float s = std::sqrt(3.f) / std::sqrt(static_cast<float>(C));
    for (int i = 0; i < L.wte_n; ++i) p[L.wte + i] = rng.normal() * 0.8f;
    for (int i = 0; i < L.lm_n; ++i) p[L.lm + i] = rng.normal() * 0.001f;
    int n = L.cfg.n_layer;
    for (int i = 0; i < n; ++i) {
        float denom = static_cast<float>(std::max(n - 1, 1));
        p[L.resid + i] = 1.15f - 0.10f * i / denom;
        p[L.x0 + i] = 0.20f - 0.15f * i / denom;
    }
    p[L.smear_lambda] = 0.f;
    p[L.backout] = 0.2f;
    for (int i = 0; i < 24; ++i) p[L.smear_gate + i] = rng.uni(0.f, 0.02f);
    auto uni = [&](int off, int count, float lo, float hi) {
        for (int i = 0; i < count; ++i) p[off + i] = rng.uni(lo, hi);
    };
    for (auto& a : L.layers) {
        uni(a.q, a.q_n, -s, s);
        uni(a.k, a.k_n, -s, s);
        uni(a.v, a.v_n, -s, s);
        uni(a.fc, a.fc_n, -0.4f * s, 0.4f * s);
        if (a.has_ve) {
            uni(a.ve, a.ve_n, -s, s);
            uni(a.ve_gate, a.gate_n, 0.f, 0.02f);
        }
    }
}

namespace {

struct CpuEngine : Engine {
    Layout L;
    std::vector<float> P, G;
    std::vector<int> tok_bytes;
    std::vector<float> rope_cos, rope_sin;
    int rope_T = 0;
    std::vector<AdamSlot> adam;
    std::vector<MuonSlot> muon;

    explicit CpuEngine(const GPTConfig& cfg, const int* token_bytes) : L(make_layout(cfg)) {
        P.assign(L.n_param, 0.f);
        G.assign(L.n_param, 0.f);
        init_params(L, P.data(), cfg.seed);
        tok_bytes.assign(cfg.vocab, 1);
        if (token_bytes)
            for (int i = 0; i < cfg.vocab; ++i) tok_bytes[i] = token_bytes[i];
        auto add_adam = [&](int off, int n, float b1, float b2, float wd, int kind) {
            AdamSlot s;
            s.off = off;
            s.n = n;
            s.b1 = b1;
            s.b2 = b2;
            s.wd = wd;
            s.kind = kind;
            s.m.assign(n, 0.f);
            s.v.assign(n, 0.f);
            adam.push_back(std::move(s));
        };
        auto add_muon = [&](int off, int rows, int cols) {
            MuonSlot m;
            m.off = off;
            m.rows = rows;
            m.cols = cols;
            m.mom.assign(static_cast<size_t>(rows) * cols, 0.f);
            m.sec.assign(rows >= cols ? rows : cols, 0.f);
            muon.push_back(std::move(m));
        };
        add_adam(L.wte, L.wte_n, 0.8f, 0.995f, 0.001f, 0);
        add_adam(L.lm, L.lm_n, 0.8f, 0.96f, 0.01f, 1);
        add_adam(L.resid, L.cfg.n_layer, 0.8f, 0.95f, 0.05f, 3);
        add_adam(L.x0, L.cfg.n_layer, 0.96f, 0.95f, 0.f, 4);
        add_adam(L.smear_gate, 24, 0.8f, 0.95f, 0.f, 5);
        add_adam(L.smear_lambda, 1, 0.8f, 0.95f, 0.f, 5);
        add_adam(L.backout, 1, 0.8f, 0.95f, 0.f, 5);
        int C = L.cfg.n_embd;
        for (auto& a : L.layers) {
            add_muon(a.q, C, C);
            add_muon(a.k, L.kv_dim, C);
            add_muon(a.v, L.kv_dim, C);
            add_muon(a.proj, C, C);
            add_muon(a.fc, L.mlp_dim, C);
            add_muon(a.fc_proj, C, L.mlp_dim);
            if (a.has_ve) {
                add_adam(a.ve, a.ve_n, 0.8f, 0.995f, 0.01f, 2);
                add_muon(a.ve_gate, L.cfg.n_kv_head, 12);
            }
        }
        ensure_rope(8);
    }

    const Layout& layout() const override { return L; }
    void zero_grad() override { std::fill(G.begin(), G.end(), 0.f); }
    void copy_params(std::vector<float>& h) const override { h = P; }
    void copy_grads(std::vector<float>& h) const override { h = G; }
    void load_params(const std::vector<float>& h) override {
        check((int)h.size() == L.n_param, "param size mismatch");
        P = h;
    }

    void ensure_rope(int T) {
        if (T <= rope_T) return;
        int half = L.head_dim / 2;
        rope_cos.assign((size_t)T * half, 0.f);
        rope_sin.assign((size_t)T * half, 0.f);
        for (int i = 0; i < half; ++i) {
            float freq = 1.f / std::pow(L.cfg.rope_base, (2.f * i) / (float)L.head_dim);
            for (int t = 0; t < T; ++t) {
                float ang = t * freq;
                rope_cos[(size_t)t * half + i] = std::cos(ang);
                rope_sin[(size_t)t * half + i] = std::sin(ang);
            }
        }
        rope_T = T;
    }

    static void acc(float* dst, const float* src, int n, float s = 1.f) {
        for (int i = 0; i < n; ++i) dst[i] += src[i] * s;
    }

    // Transpose (B,T,H,D) <-> (B,H,T,D).
    static void to_bhtd(const float* in, float* out, int B, int T, int H, int D) {
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < H; ++h)
                    for (int d = 0; d < D; ++d)
                        out[(((size_t)b * H + h) * T + t) * D + d] = in[(((size_t)b * T + t) * H + h) * D + d];
    }
    static void to_bthd(const float* in, float* out, int B, int T, int H, int D) {
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < H; ++h)
                for (int t = 0; t < T; ++t)
                    for (int d = 0; d < D; ++d)
                        out[(((size_t)b * T + t) * H + h) * D + d] = in[(((size_t)b * H + h) * T + t) * D + d];
    }

    struct LSave {
        std::vector<float> x_in, x_before, q_pre, k_pre, v, probs, attn, ve_g, x_mid, hpre;
        std::vector<float> rstd_a, rstd_q, rstd_k, rstd_m;
    };

    // One forward. Optionally backprops mean CE (scaled) and/or records bpb pieces and last-step logits.
    float run(const int32_t* idx, const int32_t* y, int B, int T, bool backward, float scale, double* nats, int64_t* nbytes,
              float* last_logits) {
        ensure_rope(T);
        const int C = L.cfg.n_embd, H = L.cfg.n_head, Hk = L.cfg.n_kv_head, D = L.head_dim, V = L.cfg.vocab;
        const int M = L.mlp_dim, BT = B * T;
        const float eps = L.cfg.rms_eps, qks = L.cfg.qk_scale, softcap = L.cfg.softcap;
        std::vector<float> xe((size_t)BT * C), rstd_e(BT), xpre((size_t)BT * C), x0((size_t)BT * C), smear_g(BT, 0.f);
        for (int i = 0; i < BT; ++i) {
            int id = idx[i];
            check(id >= 0 && id < V, "token id out of range");
            std::memcpy(&xe[(size_t)i * C], &P[L.wte + (size_t)id * C], sizeof(float) * C);
        }
        rmsnorm_fwd(xe.data(), xpre.data(), rstd_e.data(), BT, C, eps);
        for (int b = 0; b < B; ++b) {
            std::memcpy(&x0[(size_t)(b * T) * C], &xpre[(size_t)(b * T) * C], sizeof(float) * C);
            for (int t = 1; t < T; ++t) {
                const float* xt = &xpre[(size_t)(b * T + t) * C];
                float pre = 0.f;
                for (int k = 0; k < 24; ++k) pre += P[L.smear_gate + k] * xt[k];
                float sig = 1.f / (1.f + std::exp(-pre));
                float g = P[L.smear_lambda] * sig;
                smear_g[b * T + t] = g;
                float* dst = &x0[(size_t)(b * T + t) * C];
                const float* prev = &xpre[(size_t)(b * T + t - 1) * C];
                for (int c = 0; c < C; ++c) dst[c] = xt[c] + g * prev[c];
            }
        }
        std::vector<LSave> sv(L.cfg.n_layer);
        std::vector<float> x = x0;
        const int back_i = L.cfg.n_layer / 2;
        std::vector<float> x_saved;
        for (int li = 0; li < L.cfg.n_layer; ++li) {
            const LayerOff& a = L.layers[li];
            LSave& S = sv[li];
            S.x_before = x;
            S.x_in.resize((size_t)BT * C);
            float rl = P[L.resid + li], xl = P[L.x0 + li];
            for (int i = 0; i < BT * C; ++i) S.x_in[i] = rl * S.x_before[i] + xl * x0[i];
            std::vector<float> xn((size_t)BT * C);
            S.rstd_a.resize(BT);
            rmsnorm_fwd(S.x_in.data(), xn.data(), S.rstd_a.data(), BT, C, eps);
            std::vector<float> qbt((size_t)BT * C), kbt((size_t)BT * L.kv_dim), vbt((size_t)BT * L.kv_dim);
            gemm_nt(xn.data(), &P[a.q], qbt.data(), BT, C, C);
            gemm_nt(xn.data(), &P[a.k], kbt.data(), BT, L.kv_dim, C);
            gemm_nt(xn.data(), &P[a.v], vbt.data(), BT, L.kv_dim, C);
            S.q_pre.resize((size_t)B * H * T * D);
            S.k_pre.resize((size_t)B * Hk * T * D);
            S.v.resize((size_t)B * Hk * T * D);
            to_bhtd(qbt.data(), S.q_pre.data(), B, T, H, D);
            to_bhtd(kbt.data(), S.k_pre.data(), B, T, Hk, D);
            to_bhtd(vbt.data(), S.v.data(), B, T, Hk, D);
            if (a.has_ve) {
                S.ve_g.resize((size_t)BT * Hk);
                for (int i = 0; i < BT; ++i) {
                    int id = idx[i];
                    int b = i / T, t = i % T;
                    for (int h = 0; h < Hk; ++h) {
                        const float* w = &P[a.ve_gate + (size_t)h * 12];
                        float pre = 0.f;
                        for (int k = 0; k < 12; ++k) pre += w[k] * xn[(size_t)i * C + k];
                        float g = 3.f / (1.f + std::exp(-pre));
                        S.ve_g[(size_t)i * Hk + h] = g;
                        const float* ve = &P[a.ve + ((size_t)id * Hk + h) * D];
                        float* vv = &S.v[(((size_t)b * Hk + h) * T + t) * D];
                        for (int d = 0; d < D; ++d) vv[d] += g * ve[d];
                    }
                }
            }
            std::vector<float> q = S.q_pre, k = S.k_pre;
            rope_fwd(q.data(), rope_cos.data(), rope_sin.data(), B, H, T, D);
            rope_fwd(k.data(), rope_cos.data(), rope_sin.data(), B, Hk, T, D);
            S.rstd_q.resize((size_t)B * H * T);
            S.rstd_k.resize((size_t)B * Hk * T);
            std::vector<float> qn(q.size()), kn(k.size());
            rmsnorm_fwd(q.data(), qn.data(), S.rstd_q.data(), B * H * T, D, eps);
            rmsnorm_fwd(k.data(), kn.data(), S.rstd_k.data(), B * Hk * T, D, eps);
            for (float& v : qn) v *= qks;
            for (float& v : kn) v *= qks;
            S.probs.resize((size_t)B * H * T * T);
            S.attn.resize((size_t)B * H * T * D);
            int window = a.window >= T ? -1 : a.window;
            attn_fwd(qn.data(), kn.data(), S.v.data(), S.attn.data(), S.probs.data(), B, H, Hk, T, D, window);
            std::vector<float> attn_bt((size_t)BT * C), y((size_t)BT * C);
            to_bthd(S.attn.data(), attn_bt.data(), B, T, H, D);
            gemm_nt(attn_bt.data(), &P[a.proj], y.data(), BT, C, C);
            S.x_mid.resize((size_t)BT * C);
            for (int i = 0; i < BT * C; ++i) S.x_mid[i] = S.x_in[i] + y[i];
            std::vector<float> xn2((size_t)BT * C);
            S.rstd_m.resize(BT);
            rmsnorm_fwd(S.x_mid.data(), xn2.data(), S.rstd_m.data(), BT, C, eps);
            S.hpre.resize((size_t)BT * M);
            gemm_nt(xn2.data(), &P[a.fc], S.hpre.data(), BT, M, C);
            std::vector<float> h(S.hpre.size()), y2((size_t)BT * C);
            relu2_fwd(S.hpre.data(), h.data(), BT * M);
            gemm_nt(h.data(), &P[a.fc_proj], y2.data(), BT, C, M);
            x.resize((size_t)BT * C);
            for (int i = 0; i < BT * C; ++i) x[i] = S.x_mid[i] + y2[i];
            if (li == back_i) x_saved = x;
        }
        std::vector<float> x_before_back = x;
        float bl = P[L.backout];
        for (int i = 0; i < BT * C; ++i) x[i] = x_before_back[i] - bl * x_saved[i];
        std::vector<float> xf((size_t)BT * C), rstd_f(BT);
        rmsnorm_fwd(x.data(), xf.data(), rstd_f.data(), BT, C, eps);
        std::vector<float> logits((size_t)BT * V);
        gemm_nt(xf.data(), &P[L.lm], logits.data(), BT, V, C);
        if (last_logits) std::memcpy(last_logits, &logits[(size_t)(BT - 1) * V], sizeof(float) * V);
        float loss = 0.f;
        std::vector<float> dlogits;
        if (y) {
            dlogits.assign((size_t)BT * V, 0.f);
            loss = softcap_ce(logits.data(), y, backward ? dlogits.data() : nullptr, BT, V, softcap, backward);
            if (backward)
                for (float& v : dlogits) v *= scale;
            if (nats && nbytes) {
                // Per-token nll, same softcap, summed only where the target has a positive byte length.
                for (int r = 0; r < BT; ++r) {
                    int tgt = y[r];
                    if (tgt < 0) continue;
                    int nb = tok_bytes[tgt];
                    if (nb <= 0) continue;
                    const float* row = &logits[(size_t)r * V];
                    float m = -1e30f;
                    for (int v = 0; v < V; ++v) m = std::max(m, softcap * std::tanh(row[v] / softcap));
                    float sum = 0.f, zt = 0.f;
                    for (int v = 0; v < V; ++v) {
                        float e = std::exp(softcap * std::tanh(row[v] / softcap) - m);
                        sum += e;
                        if (v == tgt) zt = e;
                    }
                    *nats += -std::log(std::max(zt / sum, 1e-20f));
                    *nbytes += nb;
                }
            }
        }
        if (!backward || !y) return loss;

        // ---- backward ----
        std::vector<float> dW, dX;
        auto add_nt_grads = [&](const float* dY, const float* X, int offW, int Mrows, int N, int K) {
            // dY (M,N), X (M,K), W (N,K)
            dW.assign((size_t)N * K, 0.f);
            gemm_tn(dY, X, dW.data(), N, K, Mrows);
            acc(&G[offW], dW.data(), N * K);
            dX.assign((size_t)Mrows * K, 0.f);
            gemm_nn(dY, &P[offW], dX.data(), Mrows, K, N);
            return dX;
        };
        std::vector<float> dxf = add_nt_grads(dlogits.data(), xf.data(), L.lm, BT, V, C);
        std::vector<float> dx_backin((size_t)BT * C);
        rmsnorm_bwd(x.data(), rstd_f.data(), dxf.data(), dx_backin.data(), BT, C);
        // x = x_before_back - bl * x_saved
        std::vector<float> dx_last = dx_backin;
        std::vector<float> dx_saved((size_t)BT * C, 0.f);
        for (int i = 0; i < BT * C; ++i) {
            dx_saved[i] += -bl * dx_backin[i];
            G[L.backout] += -x_saved[i] * dx_backin[i];
        }
        std::vector<float> dx0_extra((size_t)BT * C, 0.f);
        std::vector<float> dx = dx_last;
        for (int li = L.cfg.n_layer - 1; li >= 0; --li) {
            const LayerOff& a = L.layers[li];
            LSave& S = sv[li];
            if (li == back_i)
                for (int i = 0; i < BT * C; ++i) dx[i] += dx_saved[i];
            // MLP
            std::vector<float> h((size_t)BT * M);
            relu2_fwd(S.hpre.data(), h.data(), BT * M);
            std::vector<float> xn2((size_t)BT * C);
            for (int i = 0; i < BT; ++i)
                for (int c = 0; c < C; ++c) xn2[(size_t)i * C + c] = S.x_mid[(size_t)i * C + c] * S.rstd_m[i];
            std::vector<float> dh = add_nt_grads(dx.data(), h.data(), a.fc_proj, BT, C, M);
            std::vector<float> dhpre((size_t)BT * M);
            relu2_bwd(S.hpre.data(), dh.data(), dhpre.data(), BT * M);
            std::vector<float> dxn2 = add_nt_grads(dhpre.data(), xn2.data(), a.fc, BT, M, C);
            std::vector<float> dx_mid((size_t)BT * C);
            rmsnorm_bwd(S.x_mid.data(), S.rstd_m.data(), dxn2.data(), dx_mid.data(), BT, C);
            for (int i = 0; i < BT * C; ++i) dx_mid[i] += dx[i];
            // Attention projection
            std::vector<float> attn_bt((size_t)BT * C);
            to_bthd(S.attn.data(), attn_bt.data(), B, T, H, D);
            std::vector<float> dattn_bt = add_nt_grads(dx_mid.data(), attn_bt.data(), a.proj, BT, C, C);
            std::vector<float> dattn((size_t)B * H * T * D);
            to_bhtd(dattn_bt.data(), dattn.data(), B, T, H, D);
            // Rebuild q, k used by attention.
            std::vector<float> q = S.q_pre, k = S.k_pre;
            rope_fwd(q.data(), rope_cos.data(), rope_sin.data(), B, H, T, D);
            rope_fwd(k.data(), rope_cos.data(), rope_sin.data(), B, Hk, T, D);
            std::vector<float> qn(q.size()), kn(k.size());
            rmsnorm_fwd(q.data(), qn.data(), S.rstd_q.data(), B * H * T, D, eps);
            rmsnorm_fwd(k.data(), kn.data(), S.rstd_k.data(), B * Hk * T, D, eps);
            for (float& v : qn) v *= qks;
            for (float& v : kn) v *= qks;
            std::vector<float> dq((size_t)B * H * T * D), dk((size_t)B * Hk * T * D), dv((size_t)B * Hk * T * D);
            int window = a.window >= T ? -1 : a.window;
            attn_bwd(qn.data(), kn.data(), S.v.data(), S.probs.data(), dattn.data(), dq.data(), dk.data(), dv.data(), B, H, Hk, T,
                     D, window);
            for (float& v : dq) v *= qks;
            for (float& v : dk) v *= qks;
            std::vector<float> dq_rope((size_t)B * H * T * D), dk_rope((size_t)B * Hk * T * D);
            rmsnorm_bwd(q.data(), S.rstd_q.data(), dq.data(), dq_rope.data(), B * H * T, D);
            rmsnorm_bwd(k.data(), S.rstd_k.data(), dk.data(), dk_rope.data(), B * Hk * T, D);
            rope_bwd(dq_rope.data(), rope_cos.data(), rope_sin.data(), B, H, T, D);
            rope_bwd(dk_rope.data(), rope_cos.data(), rope_sin.data(), B, Hk, T, D);
            // Value embedding grad before the projection grad.
            std::vector<float> dv_proj = dv;
            if (a.has_ve) {
                for (int i = 0; i < BT; ++i) {
                    int id = idx[i];
                    int b = i / T, t = i % T;
                    for (int h = 0; h < Hk; ++h) {
                        float* vv = &dv[(((size_t)b * Hk + h) * T + t) * D];
                        float* ve = &P[a.ve + ((size_t)id * Hk + h) * D];
                        float* gve = &G[a.ve + ((size_t)id * Hk + h) * D];
                        float gmul = S.ve_g[(size_t)i * Hk + h];
                        float dgate = 0.f;
                        for (int d = 0; d < D; ++d) {
                            gve[d] += vv[d] * gmul;
                            dgate += vv[d] * ve[d];
                        }
                        // g = 3 * sigmoid(pre), dpre = dgate * 3 * sig * (1-sig) = dgate * g * (1 - g/3)
                        float sig = gmul / 3.f;
                        float dpre = dgate * gmul * (1.f - sig);
                        float* gw = &G[a.ve_gate + (size_t)h * 12];
                        // xn recomputed below; stash dpre on ve_g slot by accumulating into a side buffer
                        // Use the high end of ve_g vector? Store dpre in ve_g as a second pass. We'll recompute xn now.
                        (void)gw;
                        (void)dpre;
                    }
                }
            }
            // Recompute xn and finish ve_gate, then QKV projection grads.
            std::vector<float> xn((size_t)BT * C);
            for (int i = 0; i < BT; ++i)
                for (int c = 0; c < C; ++c) xn[(size_t)i * C + c] = S.x_in[(size_t)i * C + c] * S.rstd_a[i];
            std::vector<float> dxn((size_t)BT * C, 0.f);
            if (a.has_ve) {
                for (int i = 0; i < BT; ++i) {
                    int b = i / T, t = i % T;
                    for (int h = 0; h < Hk; ++h) {
                        float* vv = &dv[(((size_t)b * Hk + h) * T + t) * D];
                        const float* ve = &P[a.ve + ((size_t)idx[i] * Hk + h) * D];
                        float gmul = S.ve_g[(size_t)i * Hk + h];
                        float dgate = 0.f;
                        for (int d = 0; d < D; ++d) dgate += vv[d] * ve[d];
                        float sig = gmul / 3.f;
                        float dpre = dgate * gmul * (1.f - sig);
                        float* gw = &G[a.ve_gate + (size_t)h * 12];
                        const float* w = &P[a.ve_gate + (size_t)h * 12];
                        for (int k = 0; k < 12; ++k) {
                            gw[k] += dpre * xn[(size_t)i * C + k];
                            dxn[(size_t)i * C + k] += dpre * w[k];
                        }
                    }
                }
            }
            std::vector<float> qbt((size_t)BT * C), kbt((size_t)BT * L.kv_dim), vbt((size_t)BT * L.kv_dim);
            std::vector<float> dqbt((size_t)BT * C), dkbt((size_t)BT * L.kv_dim), dvbt((size_t)BT * L.kv_dim);
            to_bthd(dq_rope.data(), dqbt.data(), B, T, H, D);
            to_bthd(dk_rope.data(), dkbt.data(), B, T, Hk, D);
            to_bthd(dv_proj.data(), dvbt.data(), B, T, Hk, D);
            auto add_dx = [&](const float* dY, int offW, int N, int K) {
                std::vector<float> tmp((size_t)BT * K);
                gemm_nn(dY, &P[offW], tmp.data(), BT, K, N);
                for (int i = 0; i < BT * K; ++i) dxn[i] += tmp[i];
                std::vector<float> dw((size_t)N * K);
                gemm_tn(dY, xn.data(), dw.data(), N, K, BT);
                acc(&G[offW], dw.data(), N * K);
            };
            add_dx(dqbt.data(), a.q, C, C);
            add_dx(dkbt.data(), a.k, L.kv_dim, C);
            add_dx(dvbt.data(), a.v, L.kv_dim, C);
            std::vector<float> dx_in((size_t)BT * C);
            rmsnorm_bwd(S.x_in.data(), S.rstd_a.data(), dxn.data(), dx_in.data(), BT, C);
            for (int i = 0; i < BT * C; ++i) dx_in[i] += dx_mid[i];
            float rl = P[L.resid + li], xl = P[L.x0 + li];
            dx.assign((size_t)BT * C, 0.f);
            for (int i = 0; i < BT * C; ++i) {
                dx[i] += rl * dx_in[i];
                dx0_extra[i] += xl * dx_in[i];
                G[L.resid + li] += S.x_before[i] * dx_in[i];
                G[L.x0 + li] += x0[i] * dx_in[i];
            }
            (void)qbt;
            (void)kbt;
            (void)vbt;
        }
        // dx is grad of x0 from the residual stream. Add the x0-skip grads.
        for (int i = 0; i < BT * C; ++i) dx[i] += dx0_extra[i];
        // Smear backward.
        std::vector<float> dxpre((size_t)BT * C, 0.f);
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) dxpre[(size_t)(b * T) * C + c] += dx[(size_t)(b * T) * C + c];
            for (int t = 1; t < T; ++t) {
                const float* dxt = &dx[(size_t)(b * T + t) * C];
                const float* prev = &xpre[(size_t)(b * T + t - 1) * C];
                const float* xt = &xpre[(size_t)(b * T + t) * C];
                float g = smear_g[b * T + t];
                float* dprev = &dxpre[(size_t)(b * T + t - 1) * C];
                float* dcur = &dxpre[(size_t)(b * T + t) * C];
                float dg = 0.f;
                for (int c = 0; c < C; ++c) {
                    dcur[c] += dxt[c];
                    dprev[c] += dxt[c] * g;
                    dg += dxt[c] * prev[c];
                }
                float pre = 0.f;
                for (int k = 0; k < 24; ++k) pre += P[L.smear_gate + k] * xt[k];
                float sig = 1.f / (1.f + std::exp(-pre));
                G[L.smear_lambda] += dg * sig;
                float dpre = dg * P[L.smear_lambda] * sig * (1.f - sig);
                for (int k = 0; k < 24; ++k) {
                    G[L.smear_gate + k] += dpre * xt[k];
                    dcur[k] += dpre * P[L.smear_gate + k];
                }
            }
        }
        std::vector<float> dxe((size_t)BT * C);
        rmsnorm_bwd(xe.data(), rstd_e.data(), dxpre.data(), dxe.data(), BT, C);
        for (int i = 0; i < BT; ++i) {
            int id = idx[i];
            acc(&G[L.wte + (size_t)id * C], &dxe[(size_t)i * C], C);
        }
        return loss;
    }

    void train_step(const int32_t* x, const int32_t* y, int B, int T, float scale, float& loss) override {
        loss = run(x, y, B, T, true, scale, nullptr, nullptr, nullptr);
    }
    void eval_forward(const int32_t* x, const int32_t* y, int B, int T, const int* tb, double& nats, int64_t& bytes) override {
        if (tb)
            for (int i = 0; i < L.cfg.vocab; ++i) tok_bytes[i] = tb[i];
        run(x, y, B, T, false, 1.f, &nats, &bytes, nullptr);
    }
    void optim_step(const TrainSched& sch) override {
        for (auto& s : adam) {
            float lr = sch.embedding_lr;
            if (s.kind == 1) lr = sch.unembedding_lr;
            else if (s.kind == 2) lr = sch.value_lr;
            else if (s.kind == 3) lr = sch.resid_lr;
            else if (s.kind == 4) lr = sch.x0_lr;
            else if (s.kind == 5) lr = sch.smear_lr;
            ++s.step;
            adamw(&P[s.off], &G[s.off], s.m.data(), s.v.data(), s.n, lr, s.b1, s.b2, s.eps, s.wd, s.step);
        }
        for (auto& m : muon)
            muon_update(&P[m.off], &G[m.off], m.mom.data(), m.sec.data(), m.rows, m.cols, sch.muon_momentum, sch.matrix_lr,
                        sch.muon_wd);
    }
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt, int max_new, float temperature, int top_k,
                                  uint64_t seed) override {
        std::vector<int32_t> ids = prompt;
        if (ids.empty()) ids.push_back(0);
        Rng rng(seed);
        int V = L.cfg.vocab;
        std::vector<float> logits(V);
        for (int n = 0; n < max_new; ++n) {
            int T = (int)ids.size();
            check(T < 8192, "generate context too long for the CPU reference");
            run(ids.data(), nullptr, 1, T, false, 1.f, nullptr, nullptr, logits.data());
            int next = 0;
            if (temperature <= 0.f) {
                next = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
            } else {
                std::vector<int> order(V);
                for (int i = 0; i < V; ++i) order[i] = i;
                if (top_k > 0 && top_k < V) {
                    std::nth_element(order.begin(), order.begin() + top_k, order.end(),
                                     [&](int a, int b) { return logits[a] > logits[b]; });
                    order.resize(top_k);
                }
                float m = -1e30f;
                for (int i : order) m = std::max(m, logits[i] / temperature);
                double sum = 0;
                std::vector<double> p(order.size());
                for (size_t i = 0; i < order.size(); ++i) {
                    p[i] = std::exp(logits[order[i]] / temperature - m);
                    sum += p[i];
                }
                double r = rng.uni() * sum;
                next = order.back();
                for (size_t i = 0; i < order.size(); ++i) {
                    r -= p[i];
                    if (r <= 0) {
                        next = order[i];
                        break;
                    }
                }
            }
            ids.push_back(next);
        }
        return ids;
    }
    void save_optim(const std::string& path) const override {
        std::ofstream o(path, std::ios::binary);
        check(o.good(), "cannot write optim state");
        o.write("TFOP0001", 8);
        int32_t na = (int)adam.size(), nm = (int)muon.size();
        o.write((char*)&na, 4);
        o.write((char*)&nm, 4);
        for (auto& s : adam) {
            o.write((char*)&s.step, 4);
            o.write((char*)s.m.data(), sizeof(float) * s.m.size());
            o.write((char*)s.v.data(), sizeof(float) * s.v.size());
        }
        for (auto& m : muon) {
            o.write((char*)m.mom.data(), sizeof(float) * m.mom.size());
            o.write((char*)m.sec.data(), sizeof(float) * m.sec.size());
        }
    }
    void load_optim(const std::string& path) override {
        std::ifstream in(path, std::ios::binary);
        check(in.good(), "cannot read optim state");
        char magic[8];
        in.read(magic, 8);
        check(std::memcmp(magic, "TFOP0001", 8) == 0, "bad optim magic");
        int32_t na = 0, nm = 0;
        in.read((char*)&na, 4);
        in.read((char*)&nm, 4);
        check(na == (int)adam.size() && nm == (int)muon.size(), "optim state does not match the model");
        for (auto& s : adam) {
            in.read((char*)&s.step, 4);
            in.read((char*)s.m.data(), sizeof(float) * s.m.size());
            in.read((char*)s.v.data(), sizeof(float) * s.v.size());
        }
        for (auto& m : muon) {
            in.read((char*)m.mom.data(), sizeof(float) * m.mom.size());
            in.read((char*)m.sec.data(), sizeof(float) * m.sec.size());
        }
    }
};

}  // namespace

Engine* create_engine_cpu(const GPTConfig& cfg, const int* token_bytes) { return new CpuEngine(cfg, token_bytes); }

}  // namespace tf
