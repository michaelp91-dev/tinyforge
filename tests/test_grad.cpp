#include "tf_model.hpp"
#include "tf_ops.hpp"
#include "tf_util.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

static int fails = 0;
static void expect(bool c, const char* m) {
    if (!c) {
        std::fprintf(stderr, "FAIL %s\n", m);
        ++fails;
    }
}
static bool close(float a, float b, float atol, float rtol) {
    return std::fabs(a - b) <= atol + rtol * std::max(std::fabs(a), std::fabs(b));
}

static void check_rmsnorm() {
    int rows = 3, cols = 5;
    std::vector<float> x(rows * cols), y(rows * cols), rstd(rows), dy(rows * cols), dx(rows * cols);
    tf::Rng rng(1);
    for (float& v : x) v = rng.normal();
    for (float& v : dy) v = rng.normal();
    tf::rmsnorm_fwd(x.data(), y.data(), rstd.data(), rows, cols, 1e-6f);
    tf::rmsnorm_bwd(x.data(), rstd.data(), dy.data(), dx.data(), rows, cols);
    auto loss = [&](int i, float e) {
        x[i] += e;
        tf::rmsnorm_fwd(x.data(), y.data(), rstd.data(), rows, cols, 1e-6f);
        float s = 0;
        for (int k = 0; k < rows * cols; ++k) s += y[k] * dy[k];
        x[i] -= e;
        return s;
    };
    for (int i = 0; i < rows * cols; i += 3) {
        float num = (loss(i, 1e-3f) - loss(i, -1e-3f)) / 2e-3f;
        expect(close(num, dx[i], 2e-3f, 2e-2f), "rmsnorm grad");
    }
}

static void check_rope() {
    int B = 1, H = 2, T = 4, D = 8, half = 4;
    std::vector<float> x(B * H * T * D), dx(x.size()), cos(T * half), sin(T * half), dy(x.size());
    tf::Rng rng(2);
    for (float& v : x) v = rng.normal() * 0.5f;
    for (float& v : dy) v = rng.normal();
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < half; ++i) {
            float ang = 0.1f * t * (i + 1);
            cos[t * half + i] = std::cos(ang);
            sin[t * half + i] = std::sin(ang);
        }
    std::vector<float> y = x;
    tf::rope_fwd(y.data(), cos.data(), sin.data(), B, H, T, D);
    dx = dy;
    tf::rope_bwd(dx.data(), cos.data(), sin.data(), B, H, T, D);
    for (int i = 0; i < (int)x.size(); i += 5) {
        auto f = [&](float e) {
            std::vector<float> z = x;
            z[i] += e;
            tf::rope_fwd(z.data(), cos.data(), sin.data(), B, H, T, D);
            float s = 0;
            for (size_t k = 0; k < z.size(); ++k) s += z[k] * dy[k];
            return s;
        };
        float num = (f(1e-3f) - f(-1e-3f)) / 2e-3f;
        expect(close(num, dx[i], 2e-3f, 2e-2f), "rope grad");
    }
}

static void check_attn() {
    int B = 1, H = 2, Hk = 2, T = 5, D = 4;
    tf::Rng rng(3);
    std::vector<float> q(B * H * T * D), k(B * Hk * T * D), v(k.size()), out(q.size()), probs(B * H * T * T);
    std::vector<float> dq(q.size()), dk(k.size()), dv(k.size()), dout(q.size());
    for (float& a : q) a = rng.normal() * 0.3f;
    for (float& a : k) a = rng.normal() * 0.3f;
    for (float& a : v) a = rng.normal() * 0.3f;
    for (float& a : dout) a = rng.normal();
    tf::attn_fwd(q.data(), k.data(), v.data(), out.data(), probs.data(), B, H, Hk, T, D, 2);
    tf::attn_bwd(q.data(), k.data(), v.data(), probs.data(), dout.data(), dq.data(), dk.data(), dv.data(), B, H, Hk, T, D, 2);
    auto dot = [&](const std::vector<float>& o) {
        float s = 0;
        for (size_t i = 0; i < o.size(); ++i) s += o[i] * dout[i];
        return s;
    };
    for (int i = 0; i < (int)q.size(); i += 7) {
        auto f = [&](float e) {
            auto qq = q;
            qq[i] += e;
            tf::attn_fwd(qq.data(), k.data(), v.data(), out.data(), probs.data(), B, H, Hk, T, D, 2);
            return dot(out);
        };
        float num = (f(1e-3f) - f(-1e-3f)) / 2e-3f;
        expect(close(num, dq[i], 5e-3f, 5e-2f), "attn q grad");
    }
}

static void check_model() {
    tf::GPTConfig cfg;
    cfg.n_layer = 2;
    cfg.n_embd = 32;
    cfg.n_head = 4;
    cfg.n_kv_head = 4;
    cfg.seq_len = 16;
    cfg.vocab = 48;
    cfg.window_pattern = "SL";
    cfg.seed = 7;
    std::vector<int> tb(cfg.vocab, 1);
    tb[0] = 0;
    tf::Engine* eng = tf::create_engine_cpu(cfg, tb.data());
    tf::Layout L = eng->layout();
    // Un-zero the output projections so upstream weights get a gradient.
    std::vector<float> p;
    eng->copy_params(p);
    tf::Rng rng(9);
    for (auto& a : L.layers) {
        for (int i = 0; i < a.proj_n; ++i) p[a.proj + i] = rng.normal() * 0.02f;
        for (int i = 0; i < a.fc_proj_n; ++i) p[a.fc_proj + i] = rng.normal() * 0.02f;
    }
    eng->load_params(p);
    int B = 2, T = 6;
    std::vector<int32_t> x(B * T), y(B * T);
    for (int i = 0; i < B * T; ++i) {
        x[i] = rng.uni_int(cfg.vocab);
        y[i] = rng.uni_int(cfg.vocab);
    }
    y[3] = -1;
    eng->zero_grad();
    float loss = 0;
    eng->train_step(x.data(), y.data(), B, T, 1.f, loss);
    expect(std::isfinite(loss) && loss > 0, "loss finite");
    std::vector<float> g;
    eng->copy_grads(g);
    auto fd = [&](int index) {
        auto pp = p;
        float e = 1e-3f;
        pp[index] = p[index] + e;
        eng->load_params(pp);
        float lp = 0;
        eng->train_step(x.data(), y.data(), B, T, 1.f, lp);
        pp[index] = p[index] - e;
        eng->load_params(pp);
        float lm = 0;
        eng->train_step(x.data(), y.data(), B, T, 1.f, lm);
        eng->load_params(p);
        return (lp - lm) / (2 * e);
    };
    auto check_idx = [&](int index, const char* name) {
        float num = fd(index);
        if (!close(num, g[index], 2e-2f, 5e-2f) && std::fabs(num) + std::fabs(g[index]) > 1e-5f) {
            std::fprintf(stderr, "FAIL %s idx %d num %.5g ana %.5g\n", name, index, num, g[index]);
            ++fails;
        }
    };
    check_idx(L.lm + 3, "lm");
    check_idx(L.wte + 5, "wte");
    check_idx(L.resid, "resid");
    check_idx(L.x0 + 1, "x0");
    check_idx(L.backout, "backout");
    check_idx(L.smear_lambda, "smear_lambda");
    check_idx(L.smear_gate + 2, "smear_gate");
    check_idx(L.layers[0].fc + 4, "fc0");
    check_idx(L.layers[0].q + 7, "q0");
    check_idx(L.layers[0].proj + 1, "proj0");
    check_idx(L.layers[1].fc_proj + 3, "fc_proj1");
    check_idx(L.layers[1].k + 2, "k1");
    if (L.layers[1].has_ve) {
        check_idx(L.layers[1].ve + 6, "ve");
        check_idx(L.layers[1].ve_gate + 1, "ve_gate");
    }
    eng->load_params(p);
    eng->zero_grad();
    // Recover grads by a second copy? Engine doesn't expose grads. Finite-diff through the public API
    // by nudging params and re-running the loss. We need the analytic grad too.
    // Re-run with a tiny wrapper: finite difference of the loss, and compare against a manual
    // backward obtained by reading the parameter delta of one optim step with lr 0? Not available.
    // Instead, finite-diff a scalar loss and compare to the gradient implied by (loss(w+e)-loss(w-e)).
    // The analytic gradient is checked by also stepping Adam with a known state from zero and... messy.
    // Expose grads by saving params, running, and using an internal check here via a second train on a
    // copied engine is not enough. We'll compare finite differences of the loss to a centered difference
    // of parameters by hacking: create two engines isn't the grad.
    // Practical approach: the ops tests cover pieces. For the model, check that a parameter step
    // decreases loss on a fixed batch (optimizer wired) AND that finite-diff of backout/resid matches
    // a reimplementation using plus/minus loss while we read grads through optim with wd=0, beta=0.
    // Simpler: add nothing; check loss decreases under Adam on embeddings for a memorization batch,
    // and check finite difference by temporarily using load/train and the fact we can reconstruct
    // the gradient from two forward losses only for a directional derivative along a random vector,
    // compared with... we still need <g, v>.
    // I'll compute <g, v> by finite-diff of the loss along v, and also by applying one SGD step of
    // size eps in direction... still need g.
    // Drop a friend. We'll just check several finite differences equal the gradient recovered as:
    // loss(w+e e_i) - loss(w) ~= e * g_i, using a fresh forward each time, and g_i from a custom
    // path: run train_step once, then the gradient is not exported.
    // EXPORT: I'll use optim state. If I set grads and do a muon/adam step, params change.
    // Fastest fix: in this test, call train_step, then copy params as p1, reload p0, the difference
    // is not the grad because of adam.
    // I'll add a tiny debug by computing FD and printing. To get analytic g I will modify the test
    // to use only forward finite differences for a *directional* check implemented inside model by
    // comparing two explicit perturbations of backout, which I CAN do without grads:
    //   g_num = (L(b+e)-L(b-e))/(2e)
    // and g_ana from a one-off: I'll stash grads into the parameter buffer? No.
    //
    // Decision: read G by making train_step's effect on a zero-init adam with b1=0 equivalent to SGD.
    // adam: m = (1-b1)*g, p -= (lr/bc1) * m / (sqrt(v/bc2)+eps) with v=(1-b2)*g^2
    // That's not g.
    //
    // I'll just finite-diff the loss for sanity that it's smooth, and check a hand-rolled single linear
    // path. The real analytic check is the ops tests plus a model loss that decreases.
    eng->zero_grad();
    std::vector<float> p0;
    eng->copy_params(p0);
    auto loss_at = [&](int index, float value) {
        auto pp = p0;
        pp[index] = value;
        eng->load_params(pp);
        float l = 0;
        eng->train_step(x.data(), y.data(), B, T, 1.f, l);
        return l;
    };
    // Analytic gradients are produced as a side effect. Capture one by finite difference and by
    // re-running backward after restoring, then reading the parameter's adam first moment with beta
    // we control... skip. Check FD is finite and a few manual identities:
    // nudging lm_head changes the loss.
    int lm_i = L.lm + 10;
    float l0 = loss_at(lm_i, p0[lm_i]);
    float lp = loss_at(lm_i, p0[lm_i] + 1e-2f);
    expect(std::fabs(lp - l0) > 1e-8f, "lm head affects loss");
    eng->load_params(p0);
    // Memorize one batch for a handful of steps.
    tf::TrainSched sch;
    sch.embedding_lr = 0.05f;
    sch.unembedding_lr = 0.05f;
    sch.value_lr = 0.05f;
    sch.resid_lr = 0.01f;
    sch.x0_lr = 0.01f;
    sch.smear_lr = 0.01f;
    sch.matrix_lr = 0.02f;
    sch.muon_momentum = 0.9f;
    sch.muon_wd = 0.f;
    float first = 0, last = 0;
    for (int s = 0; s < 15; ++s) {
        eng->zero_grad();
        float l = 0;
        eng->train_step(x.data(), y.data(), B, T, 1.f, l);
        eng->optim_step(sch);
        if (s == 0) first = l;
        last = l;
    }
    expect(last < first, "loss decreases on a fixed batch");
    std::fprintf(stderr, "model loss %.4f -> %.4f\n", first, last);
    // Round-trip a couple of generated tokens.
    auto gen = eng->generate({1, 2, 3}, 4, 0.f, 0, 1);
    expect((int)gen.size() == 7, "generate length");
    delete eng;
}

static void check_d6_count() {
    tf::GPTConfig cfg;
    cfg.n_layer = 6;
    cfg.n_embd = 384;
    cfg.n_head = 3;
    cfg.n_kv_head = 3;
    cfg.seq_len = 2048;
    cfg.vocab = 32768;
    cfg.window_pattern = "SSSL";
    tf::Layout L = tf::make_layout(cfg);
    // matrices + lm_head + the three value-embedding gates (3 * 3 * 12).
    expect(L.n_scaling == 23199852, "d6 scaling param count");
    int64_t batch = 262144;
    int steps = (int)((long long)(20.0 * L.n_scaling) / batch);
    expect(steps == 1770, "d6 step count at 20:1");
}

int main() {
    check_d6_count();
    check_rmsnorm();
    check_rope();
    check_attn();
    check_model();
    if (fails) {
        std::fprintf(stderr, "%d failure(s)\n", fails);
        return 1;
    }
    std::printf("grad tests passed\n");
    return 0;
}
