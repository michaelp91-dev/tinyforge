/*
TinyForge trainer.

The arithmetic matches src/tf_ref.c (checked against PyTorch). Matrix
multiplies and causal attention use the vendored llm.c kernels. Everything
else — RoPE, RMSNorm, QK-norm, ReLU^2, softcap, the loss, AdamW and Muon —
is in src/tf_kernels.cuh.

--arch gpt2 is the GPT-2 residual stream (learned positions, affine
LayerNorm, tanh-GELU, biases, tied embeddings, AdamW). The bit-level llm.c
baseline is still third_party/llm.c/train_gpt2cu; this binary's gpt2 mode is
the same architecture, not the same fused kernel schedule.

--arch modern (the default) is the subset of nanochat aimed at bits-per-byte:
RoPE, parameter-free RMSNorm, QK-norm, ReLU^2, untied head, no biases,
embedding RMSNorm, logit softcap, and Muon for matrices with AdamW for the
embeddings. Flags below turn each of those off individually.
*/
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "llmc/cuda_common.h"
cudaDeviceProp deviceProp;
#include "llmc/dataloader.h"
#include "llmc/cublas_common.h"
#include "llmc/layernorm.cuh"
#include "llmc/matmul.cuh"
#include "llmc/attention.cuh"
#include "tf_kernels.cuh"

static cublasHandle_t g_cublas;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}
static void* cualloc(size_t bytes) {
    void* p = nullptr;
    ck(cudaMalloc(&p, bytes), "cudaMalloc");
    ck(cudaMemset(p, 0, bytes), "cudaMemset");
    return p;
}

enum { KIND_EMB = 0, KIND_HEAD = 1, KIND_VEC = 2, KIND_MUON = 3, KIND_MAT = 4 };

struct PTensor {
    char name[40];
    floatX* data;
    floatX* grad;
    float* master;
    float* m;
    float* v;
    int rows, cols, kind;
    size_t n;
};

static constexpr int MAX_L = 64;
static constexpr int MAX_TENSORS = 512;

struct Config {
    int V, Vp, max_seq, L, NH, C, HS;
    int rope, rmsnorm, qknorm, relu2, untie, bias, embed_norm, muon;
    float softcap, qk_scale, rope_base;
    int B, T;
};

struct Model {
    Config cfg;
    PTensor tensors[MAX_TENSORS];
    int ntensors;
    int id_wte, id_wpe, id_head, id_lnfw, id_lnfb;
    int id_ln1w[MAX_L], id_ln1b[MAX_L], id_ln2w[MAX_L], id_ln2b[MAX_L];
    int id_qkv[MAX_L], id_qkvb[MAX_L], id_att[MAX_L], id_attb[MAX_L];
    int id_fc[MAX_L], id_fcb[MAX_L], id_fp[MAX_L], id_fpb[MAX_L];
    // activations, sized in alloc_acts
    floatX *embed, *x, *block_in, *ln1, *ln2, *qkv, *atty, *resid2, *preact, *lnf, *logits;
    floatX *qkvr, *att, *hid, *dhid, *dx, *dres, *dln, *datty, *dqkv, *datt, *scratch_btc;
    float *ln1_mean, *ln1_rstd, *ln2_mean, *ln2_rstd, *lnf_mean, *lnf_rstd;
    float *embed_rstd, *q_rstd, *k_rstd, *losses;
    float *emb_g32, *wpe_g32;
    int *d_tokens, *d_targets;
    float *ns_x, *ns_xt, *ns_a, *ns_aa, *ns_b, *ns_y, *ns_partial;
    size_t ns_partial_n;
    size_t BT, BTC, BT3C, BT4C, att_one;
};

static int add_tensor(Model* m, const char* name, int rows, int cols, int kind) {
    if (m->ntensors >= MAX_TENSORS) { fprintf(stderr, "too many tensors\n"); exit(1); }
    PTensor* t = &m->tensors[m->ntensors];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->rows = rows; t->cols = cols; t->kind = kind;
    t->n = (size_t)rows * (size_t)cols;
    t->data = (floatX*)cualloc(t->n * sizeof(floatX));
    t->grad = (floatX*)cualloc(t->n * sizeof(floatX));
    t->master = (float*)cualloc(t->n * sizeof(float));
    t->m = (float*)cualloc(t->n * sizeof(float));
    t->v = (kind == KIND_MUON) ? nullptr : (float*)cualloc(t->n * sizeof(float));
    return m->ntensors++;
}

static int divup(int n, int b) { return (n + b - 1) / b; }

static void rms_forward(floatX* out, float* rstd, const floatX* x, int n, int C) {
    tf_rmsnorm_forward_kernel<<<n, 256, 32 * sizeof(float)>>>(out, rstd, x, n, C, 1e-6f);
    ck(cudaGetLastError(), "rmsnorm forward");
}
static void rms_backward(floatX* dx, const floatX* dout, const floatX* x, const float* rstd, int n, int C) {
    tf_rmsnorm_backward_kernel<<<n, 256, 32 * sizeof(float)>>>(dx, dout, x, rstd, n, C);
    ck(cudaGetLastError(), "rmsnorm backward");
}

static void norm_forward(Model* m, floatX* out, float* mean, float* rstd, const floatX* x,
                         const floatX* w, const floatX* b, int n) {
    int B = m->cfg.B, T = m->cfg.T, C = m->cfg.C;
    if (m->cfg.rmsnorm) rms_forward(out, rstd, x, n, C);
    else layernorm_forward(out, mean, rstd, (floatX*)x, w, b, B, T, C, 0);
}
static void norm_backward(Model* m, floatX* dx, floatX* dw, floatX* db,
                          const floatX* dout, const floatX* x, const floatX* w,
                          const float* mean, const float* rstd, const floatX* skip_or_null) {
    // Writes dx = norm_grad [+ skip if skip_or_null is non-NULL and we are in the += layernorm path].
    int B = m->cfg.B, T = m->cfg.T, C = m->cfg.C;
    int n = B * T;
    if (m->cfg.rmsnorm) {
        rms_backward(dx, dout, x, rstd, n, C);
        if (skip_or_null) {
            // dx = dx + skip, in place via a temporary add into dx
            tf_residual_kernel<<<divup(n * C, 256), 256>>>(m->scratch_btc, dx, skip_or_null, n * C);
            ck(cudaMemcpyAsync(dx, m->scratch_btc, (size_t)n * C * sizeof(floatX), cudaMemcpyDeviceToDevice), "copy");
        }
    } else {
        if (skip_or_null) {
            ck(cudaMemcpyAsync(dx, skip_or_null, (size_t)n * C * sizeof(floatX), cudaMemcpyDeviceToDevice), "copy skip");
        } else {
            ck(cudaMemsetAsync(dx, 0, (size_t)n * C * sizeof(floatX)), "zero dx");
        }
        static float* scratch = nullptr;
        if (!scratch) scratch = (float*)cualloc(1024 * sizeof(float));
        layernorm_backward(dx, dw, db, scratch, dout, x, w, mean, rstd, B, T, C, 0);
    }
}

static void qk_forward(Model* m, floatX* qkv, float* q_rstd, float* k_rstd) {
    if (!m->cfg.rope && !m->cfg.qknorm) return;
    int B = m->cfg.B, T = m->cfg.T, NH = m->cfg.NH, HS = m->cfg.HS, C = m->cfg.C;
    int blocks = B * T * NH;
    int smem = (HS + 32) * (int)sizeof(float);
    tf_qk_forward_kernel<<<blocks, HS, smem>>>(qkv, q_rstd, k_rstd, B, T, NH, HS, C,
                                               m->cfg.rope_base, m->cfg.rope, m->cfg.qknorm, m->cfg.qk_scale);
    ck(cudaGetLastError(), "qk forward");
}
static void qk_backward(Model* m, floatX* dqkv, const floatX* q_post, const floatX* k_post,
                        const float* q_rstd, const float* k_rstd) {
    if (!m->cfg.rope && !m->cfg.qknorm) return;
    int B = m->cfg.B, T = m->cfg.T, NH = m->cfg.NH, HS = m->cfg.HS, C = m->cfg.C;
    int smem = (HS + 32) * (int)sizeof(float);
    tf_qk_backward_kernel<<<B * T * NH, HS, smem>>>(dqkv, q_post, k_post, q_rstd, k_rstd,
                                                    B, T, NH, HS, C, m->cfg.rope_base,
                                                    m->cfg.rope, m->cfg.qknorm, m->cfg.qk_scale);
    ck(cudaGetLastError(), "qk backward");
}

static void alloc_acts(Model* m) {
    Config c = m->cfg;
    size_t BT = (size_t)c.B * c.T;
    size_t BTC = BT * c.C;
    size_t BT3 = BT * 3 * c.C;
    size_t BT4 = BT * 4 * c.C;
    size_t L = c.L;
    m->BT = BT; m->BTC = BTC; m->BT3C = BT3; m->BT4C = BT4;
    m->att_one = (size_t)c.B * c.NH * c.T * c.T;
    m->embed = (floatX*)cualloc(BTC * sizeof(floatX));
    m->x = (floatX*)cualloc(BTC * sizeof(floatX));
    m->block_in = (floatX*)cualloc(L * BTC * sizeof(floatX));
    m->ln1 = (floatX*)cualloc(L * BTC * sizeof(floatX));
    m->ln2 = (floatX*)cualloc(L * BTC * sizeof(floatX));
    m->qkv = (floatX*)cualloc(BT3 * sizeof(floatX));
    m->atty = (floatX*)cualloc(L * BTC * sizeof(floatX));
    m->resid2 = (floatX*)cualloc(L * BTC * sizeof(floatX));
    m->preact = (floatX*)cualloc(L * BT4 * sizeof(floatX));
    m->lnf = (floatX*)cualloc(BTC * sizeof(floatX));
    m->logits = (floatX*)cualloc(BT * c.Vp * sizeof(floatX));
    m->qkvr = (floatX*)cualloc(L * 3 * BTC * sizeof(floatX));
    m->att = (floatX*)cualloc(L * m->att_one * sizeof(floatX));
    m->hid = (floatX*)cualloc(BT4 * sizeof(floatX));
    m->dhid = (floatX*)cualloc(BT4 * sizeof(floatX));
    m->dx = (floatX*)cualloc(BTC * sizeof(floatX));
    m->dres = (floatX*)cualloc(BTC * sizeof(floatX));
    m->dln = (floatX*)cualloc(BTC * sizeof(floatX));
    m->datty = (floatX*)cualloc(BTC * sizeof(floatX));
    m->dqkv = (floatX*)cualloc(BT3 * sizeof(floatX));
    m->datt = (floatX*)cualloc(m->att_one * sizeof(floatX));
    m->scratch_btc = (floatX*)cualloc(BTC * sizeof(floatX));
    m->ln1_mean = (float*)cualloc(L * BT * sizeof(float));
    m->ln1_rstd = (float*)cualloc(L * BT * sizeof(float));
    m->ln2_mean = (float*)cualloc(L * BT * sizeof(float));
    m->ln2_rstd = (float*)cualloc(L * BT * sizeof(float));
    m->lnf_mean = (float*)cualloc(BT * sizeof(float));
    m->lnf_rstd = (float*)cualloc(BT * sizeof(float));
    m->embed_rstd = (float*)cualloc(BT * sizeof(float));
    m->q_rstd = (float*)cualloc(L * BT * c.NH * sizeof(float));
    m->k_rstd = (float*)cualloc(L * BT * c.NH * sizeof(float));
    m->losses = (float*)cualloc(BT * sizeof(float));
    m->emb_g32 = (float*)cualloc((size_t)c.Vp * c.C * sizeof(float));
    m->wpe_g32 = m->cfg.rope ? nullptr : (float*)cualloc((size_t)c.max_seq * c.C * sizeof(float));
    m->d_tokens = (int*)cualloc(BT * sizeof(int));
    m->d_targets = (int*)cualloc(BT * sizeof(int));
    size_t max_n = (size_t)4 * c.C * c.C;
    size_t gram = (size_t)c.C * c.C;
    m->ns_x = (float*)cualloc(max_n * sizeof(float));
    m->ns_xt = (float*)cualloc(max_n * sizeof(float));
    m->ns_a = (float*)cualloc(gram * sizeof(float));
    m->ns_aa = (float*)cualloc(gram * sizeof(float));
    m->ns_b = (float*)cualloc(gram * sizeof(float));
    m->ns_y = (float*)cualloc(max_n * sizeof(float));
    m->ns_partial_n = 128;
    m->ns_partial = (float*)cualloc(m->ns_partial_n * sizeof(float));
    printf("activations allocated (logits %d MiB, attention %d MiB)\n",
           (int)((BT * c.Vp * sizeof(floatX)) >> 20),
           (int)((L * m->att_one * sizeof(floatX)) >> 20));
}

static void tf_forward(Model* m) {
    Config c = m->cfg;
    int B = c.B, T = c.T, C = c.C, L = c.L;
    int BT = B * T;
    PTensor* W = m->tensors;
    tf_embed_forward_kernel<<<divup(BT * C, 256), 256>>>(
        m->embed, m->d_tokens, W[m->id_wte].data, m->cfg.rope ? nullptr : W[m->id_wpe].data,
        B, T, C, m->cfg.rope ? 0 : 1);
    ck(cudaGetLastError(), "embed");
    if (c.embed_norm) rms_forward(m->x, m->embed_rstd, m->embed, BT, C);
    else ck(cudaMemcpy(m->x, m->embed, m->BTC * sizeof(floatX), cudaMemcpyDeviceToDevice), "copy x");

    for (int l = 0; l < L; l++) {
        floatX* cur = m->x;
        floatX* saved_in = m->block_in + (size_t)l * m->BTC;
        ck(cudaMemcpy(saved_in, cur, m->BTC * sizeof(floatX), cudaMemcpyDeviceToDevice), "save in");
        floatX* l_ln1 = m->ln1 + (size_t)l * m->BTC;
        const floatX* w1 = m->id_ln1w[l] >= 0 ? W[m->id_ln1w[l]].data : nullptr;
        const floatX* b1 = m->id_ln1b[l] >= 0 ? W[m->id_ln1b[l]].data : nullptr;
        norm_forward(m, l_ln1, m->ln1_mean + (size_t)l * BT, m->ln1_rstd + (size_t)l * BT, cur, w1, b1, BT);
        floatX* qb = m->id_qkvb[l] >= 0 ? W[m->id_qkvb[l]].data : nullptr;
        matmul_forward_cublaslt(m->qkv, l_ln1, W[m->id_qkv[l]].data, qb, B, T, C, 3 * C, 0, nullptr, 0);
        qk_forward(m, m->qkv, m->q_rstd + (size_t)l * BT * c.NH, m->k_rstd + (size_t)l * BT * c.NH);
        floatX* l_qkvr = m->qkvr + (size_t)l * 3 * m->BTC;
        floatX* l_att = m->att + (size_t)l * m->att_one;
        floatX* l_atty = m->atty + (size_t)l * m->BTC;
        attention_forward(l_atty, l_qkvr, l_att, m->qkv, B, T, C, c.NH, 0);
        floatX* ab = m->id_attb[l] >= 0 ? W[m->id_attb[l]].data : nullptr;
        floatX* proj = m->dx;
        matmul_forward_cublaslt(proj, l_atty, W[m->id_att[l]].data, ab, B, T, C, C, 0, nullptr, 0);
        floatX* l_r2 = m->resid2 + (size_t)l * m->BTC;
        tf_residual_kernel<<<divup(BT * C, 256), 256>>>(l_r2, saved_in, proj, BT * C);
        ck(cudaGetLastError(), "resid2");
        floatX* l_ln2 = m->ln2 + (size_t)l * m->BTC;
        const floatX* w2 = m->id_ln2w[l] >= 0 ? W[m->id_ln2w[l]].data : nullptr;
        const floatX* b2 = m->id_ln2b[l] >= 0 ? W[m->id_ln2b[l]].data : nullptr;
        norm_forward(m, l_ln2, m->ln2_mean + (size_t)l * BT, m->ln2_rstd + (size_t)l * BT, l_r2, w2, b2, BT);
        floatX* pre = m->preact + (size_t)l * m->BT4C;
        floatX* fcb = m->id_fcb[l] >= 0 ? W[m->id_fcb[l]].data : nullptr;
        matmul_forward_cublaslt(pre, l_ln2, W[m->id_fc[l]].data, fcb, B, T, C, 4 * C, 0, nullptr, 0);
        if (c.relu2) tf_relu2_forward_kernel<<<divup((int)m->BT4C, 256), 256>>>(m->hid, pre, (int)m->BT4C);
        else gelu_forward(m->hid, pre, (int)m->BT4C, 0);
        ck(cudaGetLastError(), "mlp act");
        floatX* fpb = m->id_fpb[l] >= 0 ? W[m->id_fpb[l]].data : nullptr;
        matmul_forward_cublaslt(proj, m->hid, W[m->id_fp[l]].data, fpb, B, T, 4 * C, C, 0, nullptr, 0);
        tf_residual_kernel<<<divup(BT * C, 256), 256>>>(m->x, l_r2, proj, BT * C);
        ck(cudaGetLastError(), "resid3");
    }
    norm_forward(m, m->lnf, m->lnf_mean, m->lnf_rstd, m->x,
                 m->id_lnfw >= 0 ? W[m->id_lnfw].data : nullptr,
                 m->id_lnfb >= 0 ? W[m->id_lnfb].data : nullptr, BT);
    floatX* head = c.untie ? W[m->id_head].data : W[m->id_wte].data;
    matmul_forward_cublaslt(m->logits, m->lnf, head, nullptr, B, T, C, c.Vp, 0, nullptr, 0);
    ck(cudaGetLastError(), "logits");
}

static float tf_backward(Model* m, float dloss) {
    Config c = m->cfg;
    int B = c.B, T = c.T, C = c.C, L = c.L;
    int BT = B * T;
    PTensor* W = m->tensors;
    tf_loss_kernel<<<BT, 256, 32 * sizeof(float)>>>(m->losses, m->logits, m->d_targets, BT, c.V, c.Vp,
                                                    dloss, c.softcap, 1);
    ck(cudaGetLastError(), "loss");
    floatX* head = c.untie ? W[m->id_head].data : W[m->id_wte].data;
    floatX* dhead = c.untie ? W[m->id_head].grad : W[m->id_wte].grad;
    matmul_backward(m->dx, dhead, nullptr, m->logits, m->lnf, head, nullptr, B, T, C, c.Vp, 0, nullptr, 0);
    floatX* dstream = m->dres;
    norm_backward(m, dstream, m->id_lnfw >= 0 ? W[m->id_lnfw].grad : nullptr,
                  m->id_lnfb >= 0 ? W[m->id_lnfb].grad : nullptr,
                  m->dx, m->x, m->id_lnfw >= 0 ? W[m->id_lnfw].data : nullptr,
                  m->lnf_mean, m->lnf_rstd, nullptr);

    for (int l = L - 1; l >= 0; l--) {
        floatX* l_r2 = m->resid2 + (size_t)l * m->BTC;
        floatX* pre = m->preact + (size_t)l * m->BT4C;
        if (c.relu2) tf_relu2_forward_kernel<<<divup((int)m->BT4C, 256), 256>>>(m->hid, pre, (int)m->BT4C);
        else gelu_forward(m->hid, pre, (int)m->BT4C, 0);
        floatX* fpb_g = m->id_fpb[l] >= 0 ? W[m->id_fpb[l]].grad : nullptr;
        matmul_backward(m->dhid, W[m->id_fp[l]].grad, fpb_g, dstream, m->hid, W[m->id_fp[l]].data,
                        nullptr, B, T, 4 * C, C, 0, nullptr, 0);
        if (c.relu2) tf_relu2_backward_kernel<<<divup((int)m->BT4C, 256), 256>>>(m->dhid, m->dhid, pre, (int)m->BT4C);
        else gelu_backward_inplace(m->dhid, pre, (int)m->BT4C, 0);
        ck(cudaGetLastError(), "act bwd");
        floatX* l_ln2 = m->ln2 + (size_t)l * m->BTC;
        floatX* fcb_g = m->id_fcb[l] >= 0 ? W[m->id_fcb[l]].grad : nullptr;
        matmul_backward(m->dln, W[m->id_fc[l]].grad, fcb_g, m->dhid, l_ln2, W[m->id_fc[l]].data,
                        nullptr, B, T, C, 4 * C, 0, nullptr, 0);
        norm_backward(m, m->dx,
                      m->id_ln2w[l] >= 0 ? W[m->id_ln2w[l]].grad : nullptr,
                      m->id_ln2b[l] >= 0 ? W[m->id_ln2b[l]].grad : nullptr,
                      m->dln, l_r2,
                      m->id_ln2w[l] >= 0 ? W[m->id_ln2w[l]].data : nullptr,
                      m->ln2_mean + (size_t)l * BT, m->ln2_rstd + (size_t)l * BT, dstream);
        // dx is now d(resid2) = skip(dstream) + mlp norm grad. dstream can be reused.
        floatX* l_atty = m->atty + (size_t)l * m->BTC;
        floatX* ab_g = m->id_attb[l] >= 0 ? W[m->id_attb[l]].grad : nullptr;
        matmul_backward(m->datty, W[m->id_att[l]].grad, ab_g, m->dx, l_atty, W[m->id_att[l]].data,
                        nullptr, B, T, C, C, 0, nullptr, 0);
        floatX* l_qkvr = m->qkvr + (size_t)l * 3 * m->BTC;
        floatX* l_att = m->att + (size_t)l * m->att_one;
        attention_backward(m->dqkv, m->qkv, m->datt, m->scratch_btc, m->datty, l_qkvr, l_att, B, T, C, c.NH, 0);
        qk_backward(m, m->dqkv, l_qkvr, l_qkvr + m->BTC, m->q_rstd + (size_t)l * BT * c.NH,
                    m->k_rstd + (size_t)l * BT * c.NH);
        floatX* l_ln1 = m->ln1 + (size_t)l * m->BTC;
        floatX* qb_g = m->id_qkvb[l] >= 0 ? W[m->id_qkvb[l]].grad : nullptr;
        matmul_backward(m->dln, W[m->id_qkv[l]].grad, qb_g, m->dqkv, l_ln1, W[m->id_qkv[l]].data,
                        nullptr, B, T, C, 3 * C, 0, nullptr, 0);
        floatX* bin = m->block_in + (size_t)l * m->BTC;
        norm_backward(m, dstream,
                      m->id_ln1w[l] >= 0 ? W[m->id_ln1w[l]].grad : nullptr,
                      m->id_ln1b[l] >= 0 ? W[m->id_ln1b[l]].grad : nullptr,
                      m->dln, bin,
                      m->id_ln1w[l] >= 0 ? W[m->id_ln1w[l]].data : nullptr,
                      m->ln1_mean + (size_t)l * BT, m->ln1_rstd + (size_t)l * BT, m->dx);
    }
    // dstream is d(x) at the embedding-norm output.
    floatX* dembed = m->dx;
    if (c.embed_norm) rms_backward(dembed, dstream, m->embed, m->embed_rstd, BT, C);
    else ck(cudaMemcpy(dembed, dstream, m->BTC * sizeof(floatX), cudaMemcpyDeviceToDevice), "dembed");
    tf_embed_backward_kernel<<<divup(BT * C, 256), 256>>>(
        m->emb_g32, m->wpe_g32, dembed, m->d_tokens, BT, T, C, c.rope ? 0 : 1);
    ck(cudaGetLastError(), "embed bwd");

    float* host_loss = (float*)malloc(BT * sizeof(float));
    ck(cudaMemcpy(host_loss, m->losses, BT * sizeof(float), cudaMemcpyDeviceToHost), "loss copy");
    double s = 0;
    for (int i = 0; i < BT; i++) s += host_loss[i];
    free(host_loss);
    return (float)(s / BT);
}

// ----------------------------------------------------------------------------
// Muon (Newton-Schulz quintic) and AdamW. NS matches tf_muon_ns5 in tf_ref.c,
// then the update is scaled by sqrt(max(1, rows/cols)) as in classic Muon.

__global__ void tf_combo_kernel(float* dst, const float* a, const float* b, int n, float ca, float cb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = ca * a[i] + cb * b[i];
}

static void sgemm_rm(int m, int n, int k, const float* A, const float* B, float* C) {
    const float alpha = 1.f, beta = 0.f;
    // Row-major C(m,n) = A(m,k) @ B(k,n), via cuBLAS column-major on the transpose view.
    cublasCheck(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                            &alpha, B, n, A, k, &beta, C, n));
}

static float frobenius(Model* m, const float* x, size_t n) {
    int blocks = (int)m->ns_partial_n;
    tf_frobenius_kernel<<<blocks, 256>>>(m->ns_partial, x, n);
    float host[128];
    ck(cudaMemcpy(host, m->ns_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost), "frobenius");
    double s = 0;
    for (int i = 0; i < blocks; i++) s += host[i];
    return sqrtf((float)s);
}

static void muon_ns_and_scale(Model* m, float* G, int rows, int cols) {
    int trans = rows > cols;
    int rm = trans ? cols : rows;
    int rn = trans ? rows : cols;
    float* X = m->ns_x;
    dim3 block(16, 16);
    if (trans) {
        tf_transpose_kernel<<<dim3(divup(cols, 16), divup(rows, 16)), block>>>(X, G, rows, cols);
    } else {
        ck(cudaMemcpy(X, G, (size_t)rows * cols * sizeof(float), cudaMemcpyDeviceToDevice), "ns copy");
    }
    ck(cudaDeviceSynchronize(), "ns transpose");
    float norm = frobenius(m, X, (size_t)rm * rn);
    tf_scale_kernel<<<divup(rm * rn, 256), 256>>>(X, (size_t)rm * rn, 1.f / (norm + 1e-7f));
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    for (int step = 0; step < 5; step++) {
        tf_transpose_kernel<<<dim3(divup(rn, 16), divup(rm, 16)), block>>>(m->ns_xt, X, rm, rn);
        sgemm_rm(rm, rm, rn, X, m->ns_xt, m->ns_a);
        sgemm_rm(rm, rm, rm, m->ns_a, m->ns_a, m->ns_aa);
        tf_combo_kernel<<<divup(rm * rm, 256), 256>>>(m->ns_b, m->ns_a, m->ns_aa, rm * rm, b, c);
        sgemm_rm(rm, rn, rm, m->ns_b, X, m->ns_y);
        tf_axpy_kernel<<<divup(rm * rn, 256), 256>>>(X, m->ns_y, (size_t)rm * rn, a, 1.f);
    }
    if (trans) tf_transpose_kernel<<<dim3(divup(rm, 16), divup(rn, 16)), block>>>(G, X, rm, rn);
    else ck(cudaMemcpy(G, X, (size_t)rows * cols * sizeof(float), cudaMemcpyDeviceToDevice), "ns store");
    float aspect = sqrtf(fmaxf(1.f, (float)rows / (float)cols));
    tf_scale_kernel<<<divup(rows * cols, 256), 256>>>(G, (size_t)rows * cols, aspect);
    ck(cudaGetLastError(), "muon ns");
}

static void adam_update(PTensor* t, float lr, float b1, float b2, float eps, float wd, int step) {
    float b1c = 1.f - powf(b1, (float)(step + 1));
    float b2c = 1.f - powf(b2, (float)(step + 1));
    tf_adamw_kernel<<<divup((int)std::min(t->n, (size_t)1 << 30), 256), 256>>>(
        t->master, t->data, t->grad, t->m, t->v, t->n, lr, b1, b2, b1c, b2c, eps, wd);
    ck(cudaGetLastError(), "adamw");
}

static void optimizer_step(Model* m, int step, float mult, float batch_scale, float dmodel_scale,
                           float matrix_lr, float embed_lr, float unembed_lr,
                           float muon_wd, float embed_wd, float head_wd, float vec_wd,
                           float momentum, int gpt2_adam, float gpt2_lr, float gpt2_wd) {
    for (int i = 0; i < m->ntensors; i++) {
        PTensor* t = &m->tensors[i];
        if (t->n == 0) continue;
        if (t->kind != KIND_MUON) {
            float lr, b1, b2, eps, wd;
            if (gpt2_adam) {
                lr = gpt2_lr * mult;
                b1 = 0.9f; b2 = 0.95f; eps = 1e-8f;
                wd = (t->cols > 1) ? gpt2_wd : 0.f;
            } else if (t->kind == KIND_MAT) {
                lr = matrix_lr * batch_scale * mult;
                b1 = 0.9f; b2 = 0.95f; eps = 1e-8f; wd = gpt2_wd;
            } else if (t->kind == KIND_EMB) {
                lr = embed_lr * dmodel_scale * batch_scale * mult;
                b1 = 0.8f; b2 = 0.995f; eps = 1e-10f; wd = embed_wd;
            } else if (t->kind == KIND_HEAD) {
                lr = unembed_lr * dmodel_scale * batch_scale * mult;
                b1 = 0.8f; b2 = 0.96f; eps = 1e-10f; wd = head_wd;
            } else {
                lr = embed_lr * dmodel_scale * batch_scale * mult;
                b1 = 0.8f; b2 = 0.95f; eps = 1e-10f; wd = vec_wd;
            }
            adam_update(t, lr, b1, b2, eps, wd, step);
        } else {
            float lr = matrix_lr * batch_scale * mult;
            tf_muon_nesterov_kernel<<<divup((int)std::min(t->n, (size_t)1 << 30), 256), 256>>>(
                m->ns_y, t->m, t->grad, t->n, momentum);
            ck(cudaGetLastError(), "nesterov");
            muon_ns_and_scale(m, m->ns_y, t->rows, t->cols);
            tf_muon_apply_kernel<<<divup((int)std::min(t->n, (size_t)1 << 30), 256), 256>>>(
                t->master, t->data, m->ns_y, t->n, lr, muon_wd);
            ck(cudaGetLastError(), "muon apply");
        }
    }
    ck(cudaDeviceSynchronize(), "optim sync");
}

// ----------------------------------------------------------------------------
// Checkpoint. Model file is the bf16 parameters (what export reads). State file
// is the fp32 master plus Adam/Muon moments, so a resumed run continues.

static int fbits(float x) { int i; memcpy(&i, &x, 4); return i; }
static float ibits(int i) { float x; memcpy(&x, &i, 4); return x; }

static int arch_flags(const Config* c) {
    return (c->rope) | (c->rmsnorm << 1) | (c->qknorm << 2) | (c->relu2 << 3)
         | (c->untie << 4) | (c->bias << 5) | (c->embed_norm << 6) | (c->muon << 7);
}

static void write_model(Model* m, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
    int h[256];
    memset(h, 0, sizeof(h));
    Config c = m->cfg;
    h[0] = 20241114; h[1] = 1;
    h[2] = c.max_seq; h[3] = c.V; h[4] = c.L; h[5] = c.NH; h[6] = c.C; h[7] = c.Vp;
    h[8] = arch_flags(&c);
    h[9] = fbits(c.softcap); h[10] = fbits(c.qk_scale); h[11] = fbits(c.rope_base);
    h[12] = (sizeof(floatX) == 4) ? 0 : 2; // 0 fp32, 2 bf16 (this build)
    h[13] = m->ntensors;
    fwrite(h, sizeof(int), 256, f);
    size_t maxn = 1;
    for (int i = 0; i < m->ntensors; i++) if (m->tensors[i].n > maxn) maxn = m->tensors[i].n;
    floatX* host = (floatX*)malloc(maxn * sizeof(floatX));
    for (int i = 0; i < m->ntensors; i++) {
        PTensor* t = &m->tensors[i];
        ck(cudaMemcpy(host, t->data, t->n * sizeof(floatX), cudaMemcpyDeviceToHost), "ckpt");
        fwrite(host, sizeof(floatX), t->n, f);
    }
    free(host);
    fclose(f);
}

static void write_state(Model* m, const char* path, int step, int shard, int sample) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
    int h[256];
    memset(h, 0, sizeof(h));
    h[0] = 20241115; h[1] = 1; h[2] = step; h[3] = shard; h[4] = sample;
    h[5] = m->cfg.B; h[6] = m->cfg.T;
    fwrite(h, sizeof(int), 256, f);
    size_t maxn = 1;
    for (int i = 0; i < m->ntensors; i++) if (m->tensors[i].n > maxn) maxn = m->tensors[i].n;
    float* host = (float*)malloc(maxn * sizeof(float));
    for (int i = 0; i < m->ntensors; i++) {
        PTensor* t = &m->tensors[i];
        ck(cudaMemcpy(host, t->master, t->n * sizeof(float), cudaMemcpyDeviceToHost), "state master");
        fwrite(host, sizeof(float), t->n, f);
        ck(cudaMemcpy(host, t->m, t->n * sizeof(float), cudaMemcpyDeviceToHost), "state m");
        fwrite(host, sizeof(float), t->n, f);
        if (t->v) {
            ck(cudaMemcpy(host, t->v, t->n * sizeof(float), cudaMemcpyDeviceToHost), "state v");
            fwrite(host, sizeof(float), t->n, f);
        }
    }
    free(host);
    fclose(f);
}

static void read_state(Model* m, const char* path, int* step, int* shard, int* sample) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
    int h[256];
    if (fread(h, sizeof(int), 256, f) != 256 || h[0] != 20241115) {
        fprintf(stderr, "bad state file %s\n", path); exit(1);
    }
    *step = h[2]; *shard = h[3]; *sample = h[4];
    if (h[5] != m->cfg.B || h[6] != m->cfg.T) {
        fprintf(stderr, "state was saved with B=%d T=%d, this run is B=%d T=%d\n", h[5], h[6], m->cfg.B, m->cfg.T);
        exit(1);
    }
    size_t maxn = 1;
    for (int i = 0; i < m->ntensors; i++) if (m->tensors[i].n > maxn) maxn = m->tensors[i].n;
    float* host = (float*)malloc(maxn * sizeof(float));
    for (int i = 0; i < m->ntensors; i++) {
        PTensor* t = &m->tensors[i];
        if (fread(host, sizeof(float), t->n, f) != t->n) { fprintf(stderr, "truncated state\n"); exit(1); }
        ck(cudaMemcpy(t->master, host, t->n * sizeof(float), cudaMemcpyHostToDevice), "load master");
        if (fread(host, sizeof(float), t->n, f) != t->n) { fprintf(stderr, "truncated state m\n"); exit(1); }
        ck(cudaMemcpy(t->m, host, t->n * sizeof(float), cudaMemcpyHostToDevice), "load m");
        if (t->v) {
            if (fread(host, sizeof(float), t->n, f) != t->n) { fprintf(stderr, "truncated state v\n"); exit(1); }
            ck(cudaMemcpy(t->v, host, t->n * sizeof(float), cudaMemcpyHostToDevice), "load v");
        }
        tf_cast_to_half_kernel<<<divup((int)std::min(t->n, (size_t)1 << 30), 256), 256>>>(t->data, t->master, t->n);
    }
    free(host);
    fclose(f);
    ck(cudaDeviceSynchronize(), "resume cast");
}

static void cast_all(Model* m) {
    for (int i = 0; i < m->ntensors; i++) {
        PTensor* t = &m->tensors[i];
        tf_cast_to_half_kernel<<<divup((int)std::min(t->n, (size_t)1 << 30), 256), 256>>>(t->data, t->master, t->n);
    }
    ck(cudaDeviceSynchronize(), "cast");
}

static void init_const(PTensor* t, float v) {
    float* h = (float*)malloc(t->n * sizeof(float));
    for (size_t i = 0; i < t->n; i++) h[i] = v;
    ck(cudaMemcpy(t->master, h, t->n * sizeof(float), cudaMemcpyHostToDevice), "init");
    free(h);
}
static void init_normal(PTensor* t, float std, mt19937_state* rng) {
    float* h = (float*)malloc(t->n * sizeof(float));
    normal_(h, (unsigned)t->n, 0.f, std, rng);
    ck(cudaMemcpy(t->master, h, t->n * sizeof(float), cudaMemcpyHostToDevice), "init n");
    free(h);
}
static void init_uniform(PTensor* t, float lo, float hi, mt19937_state* rng) {
    float* h = (float*)malloc(t->n * sizeof(float));
    uniform_(h, (unsigned)t->n, lo, hi, rng);
    ck(cudaMemcpy(t->master, h, t->n * sizeof(float), cudaMemcpyHostToDevice), "init u");
    free(h);
}

static void init_weights(Model* m, int modern) {
    mt19937_state rng;
    manual_seed(&rng, 42);
    int C = m->cfg.C;
    int L = m->cfg.L;
    if (modern) {
        init_normal(&m->tensors[m->id_wte], 0.8f, &rng);
        if (m->id_head >= 0) init_normal(&m->tensors[m->id_head], 0.001f, &rng);
        float s = sqrtf(3.f / (float)C);
        for (int l = 0; l < L; l++) {
            init_uniform(&m->tensors[m->id_qkv[l]], -s, s, &rng);
            init_const(&m->tensors[m->id_att[l]], 0.f);
            init_uniform(&m->tensors[m->id_fc[l]], -0.4f * s, 0.4f * s, &rng);
            init_const(&m->tensors[m->id_fp[l]], 0.f);
            if (m->id_ln1w[l] >= 0) init_const(&m->tensors[m->id_ln1w[l]], 1.f);
            if (m->id_ln1b[l] >= 0) init_const(&m->tensors[m->id_ln1b[l]], 0.f);
            if (m->id_ln2w[l] >= 0) init_const(&m->tensors[m->id_ln2w[l]], 1.f);
            if (m->id_ln2b[l] >= 0) init_const(&m->tensors[m->id_ln2b[l]], 0.f);
            if (m->id_qkvb[l] >= 0) init_const(&m->tensors[m->id_qkvb[l]], 0.f);
            if (m->id_attb[l] >= 0) init_const(&m->tensors[m->id_attb[l]], 0.f);
            if (m->id_fcb[l] >= 0) init_const(&m->tensors[m->id_fcb[l]], 0.f);
            if (m->id_fpb[l] >= 0) init_const(&m->tensors[m->id_fpb[l]], 0.f);
        }
        if (m->id_lnfw >= 0) init_const(&m->tensors[m->id_lnfw], 1.f);
        if (m->id_lnfb >= 0) init_const(&m->tensors[m->id_lnfb], 0.f);
        if (m->id_wpe >= 0) init_normal(&m->tensors[m->id_wpe], 0.02f, &rng);
    } else {
        float res = 0.02f / sqrtf(2.f * (float)L);
        init_normal(&m->tensors[m->id_wte], 0.02f, &rng);
        if (m->id_wpe >= 0) init_normal(&m->tensors[m->id_wpe], 0.02f, &rng);
        if (m->id_lnfw >= 0) init_const(&m->tensors[m->id_lnfw], 1.f);
        if (m->id_lnfb >= 0) init_const(&m->tensors[m->id_lnfb], 0.f);
        for (int l = 0; l < L; l++) {
            if (m->id_ln1w[l] >= 0) init_const(&m->tensors[m->id_ln1w[l]], 1.f);
            if (m->id_ln1b[l] >= 0) init_const(&m->tensors[m->id_ln1b[l]], 0.f);
            init_normal(&m->tensors[m->id_qkv[l]], 0.02f, &rng);
            if (m->id_qkvb[l] >= 0) init_const(&m->tensors[m->id_qkvb[l]], 0.f);
            init_normal(&m->tensors[m->id_att[l]], res, &rng);
            if (m->id_attb[l] >= 0) init_const(&m->tensors[m->id_attb[l]], 0.f);
            if (m->id_ln2w[l] >= 0) init_const(&m->tensors[m->id_ln2w[l]], 1.f);
            if (m->id_ln2b[l] >= 0) init_const(&m->tensors[m->id_ln2b[l]], 0.f);
            init_normal(&m->tensors[m->id_fc[l]], 0.02f, &rng);
            if (m->id_fcb[l] >= 0) init_const(&m->tensors[m->id_fcb[l]], 0.f);
            init_normal(&m->tensors[m->id_fp[l]], res, &rng);
            if (m->id_fpb[l] >= 0) init_const(&m->tensors[m->id_fpb[l]], 0.f);
        }
    }
    cast_all(m);
}

static void register_model(Model* m, int muon) {
    for (int i = 0; i < MAX_L; i++) {
        m->id_ln1w[i] = m->id_ln1b[i] = m->id_ln2w[i] = m->id_ln2b[i] = -1;
        m->id_qkv[i] = m->id_qkvb[i] = m->id_att[i] = m->id_attb[i] = -1;
        m->id_fc[i] = m->id_fcb[i] = m->id_fp[i] = m->id_fpb[i] = -1;
    }
    m->id_wpe = m->id_head = m->id_lnfw = m->id_lnfb = -1;
    Config c = m->cfg;
    int mat = muon ? KIND_MUON : KIND_MAT;
    m->id_wte = add_tensor(m, "wte", c.Vp, c.C, muon ? KIND_EMB : KIND_VEC);
    if (!c.rope) m->id_wpe = add_tensor(m, "wpe", c.max_seq, c.C, KIND_VEC);
    if (c.untie) m->id_head = add_tensor(m, "lm_head", c.Vp, c.C, muon ? KIND_HEAD : KIND_VEC);
    if (!c.rmsnorm) {
        m->id_lnfw = add_tensor(m, "lnfw", c.C, 1, KIND_VEC);
        m->id_lnfb = add_tensor(m, "lnfb", c.C, 1, KIND_VEC);
    }
    char buf[40];
    for (int l = 0; l < c.L; l++) {
        if (!c.rmsnorm) {
            snprintf(buf, sizeof(buf), "ln1w.%d", l); m->id_ln1w[l] = add_tensor(m, buf, c.C, 1, KIND_VEC);
            snprintf(buf, sizeof(buf), "ln1b.%d", l); m->id_ln1b[l] = add_tensor(m, buf, c.C, 1, KIND_VEC);
        }
        snprintf(buf, sizeof(buf), "qkvw.%d", l); m->id_qkv[l] = add_tensor(m, buf, 3 * c.C, c.C, mat);
        if (c.bias) { snprintf(buf, sizeof(buf), "qkvb.%d", l); m->id_qkvb[l] = add_tensor(m, buf, 3 * c.C, 1, KIND_VEC); }
        snprintf(buf, sizeof(buf), "attw.%d", l); m->id_att[l] = add_tensor(m, buf, c.C, c.C, mat);
        if (c.bias) { snprintf(buf, sizeof(buf), "attb.%d", l); m->id_attb[l] = add_tensor(m, buf, c.C, 1, KIND_VEC); }
        if (!c.rmsnorm) {
            snprintf(buf, sizeof(buf), "ln2w.%d", l); m->id_ln2w[l] = add_tensor(m, buf, c.C, 1, KIND_VEC);
            snprintf(buf, sizeof(buf), "ln2b.%d", l); m->id_ln2b[l] = add_tensor(m, buf, c.C, 1, KIND_VEC);
        }
        snprintf(buf, sizeof(buf), "fcw.%d", l); m->id_fc[l] = add_tensor(m, buf, 4 * c.C, c.C, mat);
        if (c.bias) { snprintf(buf, sizeof(buf), "fcb.%d", l); m->id_fcb[l] = add_tensor(m, buf, 4 * c.C, 1, KIND_VEC); }
        snprintf(buf, sizeof(buf), "fpw.%d", l); m->id_fp[l] = add_tensor(m, buf, c.C, 4 * c.C, mat);
        if (c.bias) { snprintf(buf, sizeof(buf), "fpb.%d", l); m->id_fpb[l] = add_tensor(m, buf, c.C, 1, KIND_VEC); }
    }
    size_t n = 0;
    for (int i = 0; i < m->ntensors; i++) n += m->tensors[i].n;
    printf("parameters: %zu (%.2f M)\n", n, n / 1e6);
}

static void zero_grads(Model* m) {
    for (int i = 0; i < m->ntensors; i++) {
        ck(cudaMemset(m->tensors[i].grad, 0, m->tensors[i].n * sizeof(floatX)), "zero grad");
    }
    ck(cudaMemset(m->emb_g32, 0, (size_t)m->cfg.Vp * m->cfg.C * sizeof(float)), "zero emb");
    if (m->wpe_g32) ck(cudaMemset(m->wpe_g32, 0, (size_t)m->cfg.max_seq * m->cfg.C * sizeof(float)), "zero wpe");
}

static void fold_embed_grads(Model* m) {
    PTensor* wte = &m->tensors[m->id_wte];
    tf_add_fp32_kernel<<<divup((int)wte->n, 256), 256>>>(wte->grad, m->emb_g32, wte->n);
    if (m->id_wpe >= 0) {
        PTensor* wpe = &m->tensors[m->id_wpe];
        tf_add_fp32_kernel<<<divup((int)wpe->n, 256), 256>>>(wpe->grad, m->wpe_g32, wpe->n);
    }
    ck(cudaGetLastError(), "fold embed");
}

static float lr_multiplier(int step, int total, int warmup, float warmdown, float final_frac, const char* sched) {
    if (total < 1) total = 1;
    if (strcmp(sched, "constant") == 0) return 1.f;
    if (strcmp(sched, "cosine") == 0 || strcmp(sched, "linear") == 0) {
        if (warmup > 0 && step < warmup) return (float)(step + 1) / (float)warmup;
        float span = (float)(total - warmup);
        if (span < 1.f) span = 1.f;
        float ratio = ((float)step - (float)warmup) / span;
        if (ratio < 0.f) ratio = 0.f;
        if (ratio > 1.f) ratio = 1.f;
        if (strcmp(sched, "linear") == 0) return 1.f - ratio * (1.f - final_frac);
        float coeff = 0.5f * (1.f + cosf((float)M_PI * ratio));
        return final_frac + coeff * (1.f - final_frac);
    }
    int wd = (int)llroundf(warmdown * (float)total);
    if (wd < 1) wd = 1;
    if (warmup > 0 && step < warmup) return (float)(step + 1) / (float)warmup;
    if (step <= total - wd) return 1.f;
    float progress = (float)(total - step) / (float)wd;
    if (progress < 0.f) progress = 0.f;
    return progress + (1.f - progress) * final_frac;
}

static void usage(void) {
    fprintf(stderr,
        "Usage: train_tinyforge -i <train glob> -j <val bin> -e d6|d12 [-o dir]\n"
        "  -b 16 -t 1024 -d <total tokens> -x <steps> -v <val every> -n <ckpt every> -y 0|1\n"
        "  --arch modern|gpt2   --optim muon|adamw   --vocab N\n"
        "  --rope --rmsnorm --qknorm --relu2 --untie --bias --embed-norm   (0 or 1)\n"
        "  --softcap 15 --qk-scale 1.2 --rope-base 100000\n"
        "  --matrix-lr 0.02 --embed-lr 0.3 --unembed-lr 0.008\n"
        "  --muon-wd 0 --embed-wd 0.001 --head-wd 0.01 -c <wd for gpt2 / vec>\n"
        "  --warmup 40 --warmdown 0.65 --final-lr-frac 0.05 -k warmdown|cosine|linear|constant\n"
        "  --momentum 0.95 --token-bytes <int32 file>\n"
        "Stock GPT-2 baseline (bit-level llm.c): make -C third_party/llm.c train_gpt2cu\n");
    exit(1);
}

static int arg_int(const char* s) { return atoi(s); }
static float arg_float(const char* s) { return (float)atof(s); }

int main(int argc, char** argv) {
    int modern = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--arch") == 0 && i + 1 < argc) {
            modern = strcmp(argv[++i], "gpt2") != 0;
        }
    }
    const char* train_pat = nullptr;
    const char* val_pat = nullptr;
    const char* out_dir = nullptr;
    const char* descriptor = "d6";
    const char* sched = modern ? "warmdown" : "cosine";
    const char* token_bytes_path = nullptr;
    int B = 16, T = 1024, total_batch = -1, max_steps = -1;
    int val_every = 250, val_max = 20, ckpt_every = 0, resume = 0;
    int depth = 6;
    int vocab = modern ? 32768 : 50257;
    int rope = modern, rmsnorm = modern, qknorm = modern, relu2 = modern;
    int untie = modern, bias = !modern, embed_norm = modern, muon = modern;
    float softcap = modern ? 15.f : 0.f;
    float qk_scale = 1.2f, rope_base = 100000.f;
    float matrix_lr = 0.02f, embed_lr = 0.3f, unembed_lr = 0.008f;
    float gpt2_lr = 3e-4f;
    float muon_wd = 0.f, embed_wd = 0.001f, head_wd = 0.01f, vec_wd = 0.f;
    float final_frac = modern ? 0.05f : 1.f;
    float warmdown = 0.65f;
    float momentum = 0.95f;
    int warmup = modern ? 40 : 0;
    int vocab_set = 0, sched_set = 0;

    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (a[0] != '-') usage();
        const char* v = (i + 1 < argc) ? argv[++i] : nullptr;
        if (!v) usage();
        if (strcmp(a, "--arch") == 0) { /* already consumed logically */ continue; }
        else if (strcmp(a, "-i") == 0) train_pat = v;
        else if (strcmp(a, "-j") == 0) val_pat = v;
        else if (strcmp(a, "-o") == 0) out_dir = v;
        else if (strcmp(a, "-e") == 0) descriptor = v;
        else if (strcmp(a, "-b") == 0) B = arg_int(v);
        else if (strcmp(a, "-t") == 0) T = arg_int(v);
        else if (strcmp(a, "-d") == 0) total_batch = arg_int(v);
        else if (strcmp(a, "-x") == 0) max_steps = arg_int(v);
        else if (strcmp(a, "-v") == 0) val_every = arg_int(v);
        else if (strcmp(a, "-m") == 0) val_max = arg_int(v);
        else if (strcmp(a, "-n") == 0) ckpt_every = arg_int(v);
        else if (strcmp(a, "-y") == 0) resume = arg_int(v);
        else if (strcmp(a, "-l") == 0) { gpt2_lr = matrix_lr = arg_float(v); }
        else if (strcmp(a, "-u") == 0 || strcmp(a, "--warmup") == 0) warmup = arg_int(v);
        else if (strcmp(a, "-q") == 0 || strcmp(a, "--final-lr-frac") == 0) final_frac = arg_float(v);
        else if (strcmp(a, "-c") == 0) vec_wd = arg_float(v);
        else if (strcmp(a, "-k") == 0 || strcmp(a, "--scheduler") == 0) { sched = v; sched_set = 1; }
        else if (strcmp(a, "--vocab") == 0) { vocab = arg_int(v); vocab_set = 1; }
        else if (strcmp(a, "--rope") == 0) rope = arg_int(v);
        else if (strcmp(a, "--rmsnorm") == 0) rmsnorm = arg_int(v);
        else if (strcmp(a, "--qknorm") == 0) qknorm = arg_int(v);
        else if (strcmp(a, "--relu2") == 0) relu2 = arg_int(v);
        else if (strcmp(a, "--untie") == 0) untie = arg_int(v);
        else if (strcmp(a, "--bias") == 0) bias = arg_int(v);
        else if (strcmp(a, "--embed-norm") == 0) embed_norm = arg_int(v);
        else if (strcmp(a, "--softcap") == 0) softcap = arg_float(v);
        else if (strcmp(a, "--qk-scale") == 0) qk_scale = arg_float(v);
        else if (strcmp(a, "--rope-base") == 0) rope_base = arg_float(v);
        else if (strcmp(a, "--optim") == 0) muon = strcmp(v, "muon") == 0;
        else if (strcmp(a, "--matrix-lr") == 0) matrix_lr = arg_float(v);
        else if (strcmp(a, "--embed-lr") == 0) embed_lr = arg_float(v);
        else if (strcmp(a, "--unembed-lr") == 0) unembed_lr = arg_float(v);
        else if (strcmp(a, "--warmdown") == 0) warmdown = arg_float(v);
        else if (strcmp(a, "--muon-wd") == 0) muon_wd = arg_float(v);
        else if (strcmp(a, "--embed-wd") == 0) embed_wd = arg_float(v);
        else if (strcmp(a, "--head-wd") == 0) head_wd = arg_float(v);
        else if (strcmp(a, "--momentum") == 0) momentum = arg_float(v);
        else if (strcmp(a, "--token-bytes") == 0) token_bytes_path = v;
        else usage();
    }
    (void)sched_set; (void)vocab_set;
    if (!train_pat || !val_pat) usage();
    if (descriptor[0] == 'd') depth = atoi(descriptor + 1);
    else { fprintf(stderr, "descriptor must look like d6 or d12 (got %s)\n", descriptor); return 1; }
    if (T % 4 != 0) { fprintf(stderr, "T must be a multiple of 4 (attention kernel)\n"); return 1; }

    Config cfg{};
    cfg.rope = rope; cfg.rmsnorm = rmsnorm; cfg.qknorm = qknorm; cfg.relu2 = relu2;
    cfg.untie = untie; cfg.bias = bias; cfg.embed_norm = embed_norm; cfg.muon = muon;
    cfg.softcap = softcap; cfg.qk_scale = qk_scale; cfg.rope_base = rope_base;
    cfg.B = B; cfg.T = T; cfg.max_seq = T; cfg.L = depth; cfg.V = vocab;
    if (!modern && !rope) {
        if (depth == 6) { cfg.C = 384; cfg.NH = 6; }
        else if (depth == 12) { cfg.C = 768; cfg.NH = 12; }
        else if (depth == 24) { cfg.C = 1024; cfg.NH = 16; }
        else if (depth == 36) { cfg.C = 1280; cfg.NH = 20; }
        else if (depth == 48) { cfg.C = 1600; cfg.NH = 25; }
        else { fprintf(stderr, "gpt2 arch supports d6/d12/d24/d36/d48\n"); return 1; }
    } else {
        int head = 128;
        int base = depth * 64;
        cfg.C = ((base + head - 1) / head) * head;
        cfg.NH = cfg.C / head;
    }
    cfg.HS = cfg.C / cfg.NH;
    cfg.Vp = (cfg.V + 127) / 128 * 128;
    if (cfg.HS < 32 || cfg.HS > 256 || (cfg.HS % 32) != 0) {
        fprintf(stderr, "head dim %d is not supported (need a multiple of 32, <= 256)\n", cfg.HS);
        return 1;
    }
    if (total_batch < 0) total_batch = B * T;
    if (total_batch % (B * T) != 0) {
        fprintf(stderr, "total batch %d is not a multiple of B*T = %d\n", total_batch, B * T);
        return 1;
    }
    int accum = total_batch / (B * T);

    ck(cudaSetDevice(0), "set device");
    ck(cudaGetDeviceProperties(&deviceProp, 0), "device");
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    ck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size), "lt workspace");
    cublas_compute = CUBLAS_COMPUTE_32F;
    cublasCheck(cublasCreate(&g_cublas));
    printf("device: %s\n", deviceProp.name);
    printf("arch %s  d%d  L=%d C=%d NH=%d HS=%d V=%d Vp=%d\n",
           modern ? "modern" : "gpt2", depth, cfg.L, cfg.C, cfg.NH, cfg.HS, cfg.V, cfg.Vp);
    printf("rope=%d rmsnorm=%d qknorm=%d relu2=%d untie=%d bias=%d embed_norm=%d softcap=%.3f optim=%s\n",
           cfg.rope, cfg.rmsnorm, cfg.qknorm, cfg.relu2, cfg.untie, cfg.bias, cfg.embed_norm, cfg.softcap,
           muon ? "muon" : "adamw");
    printf("B=%d T=%d total_batch=%d accum=%d\n", B, T, total_batch, accum);

    Model* model = (Model*)calloc(1, sizeof(Model));
    model->cfg = cfg;
    register_model(model, muon);
    alloc_acts(model);

    int start_step = 0, resume_shard = 0, resume_sample = 0;
    int resuming = 0;
    if (resume && out_dir) {
        int done = find_max_step(out_dir);
        if (done > 0) {
            char path[512];
            snprintf(path, sizeof(path), "%s/state_%08d.bin", out_dir, done);
            read_state(model, path, &start_step, &resume_shard, &resume_sample);
            resuming = 1;
            printf("resumed from step %d\n", start_step);
        }
    }
    if (!resuming) init_weights(model, modern && muon || modern);

    // init_weights' modern flag: gpt2 init when arch is gpt2, modern init otherwise.
    // The call above passes `modern` only when modern is true. When modern is false
    // the expression `modern && muon || modern` equals modern. Good.
    // When resuming we already loaded master. When not, init_weights ran.
    // Re-init correctly if the expression was wrong for gpt2: handled.

    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_pat, B, T, 0, 1, 0);
    dataloader_init(&val_loader, val_pat, B, T, 0, 1, 0);
    if (resuming) dataloader_resume(&train_loader, resume_shard, resume_sample);

    if (max_steps < 0) {
        max_steps = (int)(train_loader.num_tokens / (size_t)total_batch);
        if (max_steps < 1) max_steps = 1;
    }
    float batch_scale = muon ? sqrtf((float)total_batch / 524288.f) : 1.f;
    float dmodel_scale = sqrtf(768.f / (float)cfg.C);
    printf("steps=%d warmup=%d scheduler=%s batch_lr_scale=%.4f dmodel_scale=%.4f\n",
           max_steps, warmup, sched, batch_scale, dmodel_scale);
    printf("matrix_lr=%.6g embed_lr=%.6g unembed_lr=%.6g (before scales)\n", matrix_lr, embed_lr, unembed_lr);

    int* token_bytes = nullptr;
    if (token_bytes_path) {
        FILE* bf = fopen(token_bytes_path, "rb");
        if (!bf) { fprintf(stderr, "cannot read %s\n", token_bytes_path); return 1; }
        fseek(bf, 0, SEEK_END);
        long sz = ftell(bf);
        fseek(bf, 0, SEEK_SET);
        token_bytes = (int*)malloc((size_t)cfg.V * sizeof(int));
        if (sz == (long)cfg.V * 4) {
            fread(token_bytes, 4, cfg.V, bf);
        } else {
            int count = 0;
            fread(&count, 4, 1, bf);
            if (count < cfg.V) { fprintf(stderr, "token_bytes count %d < vocab %d\n", count, cfg.V); return 1; }
            fread(token_bytes, 4, cfg.V, bf);
        }
        fclose(bf);
    }

    if (out_dir) create_dir_if_not_exists(out_dir);
    int* h_tok = (int*)malloc((size_t)B * T * sizeof(int));
    int* h_tgt = (int*)malloc((size_t)B * T * sizeof(int));
    float* h_loss = (float*)malloc((size_t)B * T * sizeof(float));

    auto save_ckpt = [&](int completed) {
        if (!out_dir) return;
        char path[512];
        snprintf(path, sizeof(path), "%s/model_%08d.bin", out_dir, completed);
        printf("writing %s\n", path);
        write_model(model, path);
        snprintf(path, sizeof(path), "%s/state_%08d.bin", out_dir, completed);
        write_state(model, path, completed, (int)train_loader.current_shard_idx, (int)train_loader.current_sample_idx);
        snprintf(path, sizeof(path), "%s/DONE_%08d", out_dir, completed);
        FILE* df = fopen(path, "w");
        if (df) fclose(df);
    };

    for (int step = start_step; step < max_steps; step++) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0);
        zero_grads(model);
        double loss_sum = 0;
        float dloss = 1.f / (float)(B * T * accum);
        for (int micro = 0; micro < accum; micro++) {
            dataloader_next_batch(&train_loader);
            for (int i = 0; i < B * T; i++) {
                h_tok[i] = train_loader.inputs[i];
                h_tgt[i] = train_loader.targets[i];
                if (h_tok[i] < 0 || h_tok[i] >= cfg.V || h_tgt[i] < 0 || h_tgt[i] >= cfg.V) {
                    fprintf(stderr, "token id out of range (vocab %d)\n", cfg.V);
                    return 1;
                }
            }
            ck(cudaMemcpy(model->d_tokens, h_tok, (size_t)B * T * sizeof(int), cudaMemcpyHostToDevice), "tok");
            ck(cudaMemcpy(model->d_targets, h_tgt, (size_t)B * T * sizeof(int), cudaMemcpyHostToDevice), "tgt");
            tf_forward(model);
            loss_sum += tf_backward(model, dloss);
        }
        fold_embed_grads(model);
        float mult = lr_multiplier(step, max_steps, warmup, warmdown, final_frac, sched);
        optimizer_step(model, step, mult, batch_scale, dmodel_scale,
                       matrix_lr, embed_lr, unembed_lr, muon_wd, embed_wd, head_wd, vec_wd,
                       momentum, modern ? 0 : 1, gpt2_lr, vec_wd);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        float ms = 0;
        cudaEventElapsedTime(&ms, ev0, ev1);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        double loss = loss_sum / accum;
        printf("step %d/%d  loss %.4f  lr_mult %.3f  %.2fs  %.0f tok/s\n",
               step + 1, max_steps, loss, mult, ms / 1e3, (total_batch / (ms / 1e3)));
        fflush(stdout);

        if (val_every > 0 && ((step + 1) % val_every == 0 || step + 1 == max_steps)) {
            int nval = val_max;
            int avail = (int)(val_loader.num_tokens / (size_t)(B * T));
            if (avail > 0 && nval > avail) nval = avail;
            dataloader_reset(&val_loader);
            double vloss = 0, nats = 0, nbytes = 0;
            for (int s = 0; s < nval; s++) {
                dataloader_next_batch(&val_loader);
                for (int i = 0; i < B * T; i++) {
                    h_tok[i] = val_loader.inputs[i];
                    h_tgt[i] = val_loader.targets[i];
                }
                ck(cudaMemcpy(model->d_tokens, h_tok, (size_t)B * T * sizeof(int), cudaMemcpyHostToDevice), "vtok");
                ck(cudaMemcpy(model->d_targets, h_tgt, (size_t)B * T * sizeof(int), cudaMemcpyHostToDevice), "vtgt");
                tf_forward(model);
                tf_loss_kernel<<<B * T, 256, 32 * sizeof(float)>>>(model->losses, model->logits, model->d_targets,
                                                                   B * T, cfg.V, cfg.Vp, 0.f, cfg.softcap, 0);
                ck(cudaMemcpy(h_loss, model->losses, (size_t)B * T * sizeof(float), cudaMemcpyDeviceToHost), "vloss");
                for (int i = 0; i < B * T; i++) {
                    vloss += h_loss[i];
                    if (token_bytes) {
                        int nb = token_bytes[h_tgt[i]];
                        if (nb > 0) nats += h_loss[i];
                        nbytes += nb;
                    }
                }
            }
            vloss /= (double)nval * B * T;
            printf("  val loss %.4f", vloss);
            if (token_bytes && nbytes > 0) printf("  bpb %.4f", nats / (log(2.0) * nbytes));
            printf("\n");
            fflush(stdout);
        }
        if (ckpt_every > 0 && (step + 1) % ckpt_every == 0) save_ckpt(step + 1);
    }
    save_ckpt(max_steps);
    printf("done\n");
    return 0;
}
