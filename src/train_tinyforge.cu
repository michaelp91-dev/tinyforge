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
#include <ctime>
#include <dirent.h>
#include <unistd.h>

#include "llmc/cuda_common.h"
cudaDeviceProp deviceProp;
#include "llmc/dataloader.h"
#include "llmc/cublas_common.h"
#include "llmc/layernorm.cuh"
#include "llmc/matmul.cuh"
#include "llmc/attention.cuh"
#include "tf_kernels.cuh"

static cublasHandle_t g_cublas;

// Data parallel over one node: one process per GPU, gradients averaged with
// NCCL before the optimizer step. Build with -DTF_NCCL -lnccl (make USE_NCCL=1).
static int g_rank = 0, g_world = 1;
#ifdef TF_NCCL
#include <nccl.h>
static ncclComm_t g_nccl;
#if defined(ENABLE_FP32)
#define TF_NCCL_FLOATX ncclFloat
#elif defined(ENABLE_FP16)
#define TF_NCCL_FLOATX ncclHalf
#else
#define TF_NCCL_FLOATX ncclBfloat16
#endif
static void nccl_check(ncclResult_t r, const char* what) {
    if (r != ncclSuccess) {
        fprintf(stderr, "NCCL error in %s: %s\n", what, ncclGetErrorString(r));
        exit(1);
    }
}
#endif
static float* g_scalar = nullptr;

// --profile: time sections of one step with CUDA events (adds syncs only on that step).
static int g_prof = 0;
static const int PROF_MAX = 4096;
static cudaEvent_t g_prof_ev[PROF_MAX];
static const char* g_prof_name[PROF_MAX];
static int g_prof_n = 0;
static void prof_mark(const char* name) {
    if (!g_prof || g_prof_n >= PROF_MAX) return;
    if (!g_prof_ev[g_prof_n]) cudaEventCreate(&g_prof_ev[g_prof_n]);
    cudaEventRecord(g_prof_ev[g_prof_n]);
    g_prof_name[g_prof_n++] = name;
}
static void prof_report(void) {
    if (!g_prof || g_prof_n < 2) return;
    cudaEventSynchronize(g_prof_ev[g_prof_n - 1]);
    const char* names[PROF_MAX]; float tot[PROF_MAX]; int k = 0;
    for (int i = 1; i < g_prof_n; i++) {
        float ms = 0; cudaEventElapsedTime(&ms, g_prof_ev[i - 1], g_prof_ev[i]);
        int j = 0; while (j < k && strcmp(names[j], g_prof_name[i]) != 0) j++;
        if (j == k) { names[k] = g_prof_name[i]; tot[k++] = 0; }
        tot[j] += ms;
    }
    printf("profile:");
    for (int j = 0; j < k; j++) printf("  %s %.0fms", names[j], tot[j]);
    printf("\n");
    g_prof_n = 0;
}
#ifdef ENABLE_FP16
static int g_ns_fast16 = 1; // Muon Newton-Schulz GEMMs on fp16 tensor cores (fp32 in/out); --ns-fp16 0 to disable
#else
static int g_ns_fast16 = 0;
#endif

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

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void dist_init(const char* sync_dir) {
#ifdef TF_NCCL
    if (g_world <= 1) return;
    char path[512], tmp[512];
    snprintf(path, sizeof(path), "%s/nccl_id.bin", sync_dir);
    snprintf(tmp, sizeof(tmp), "%s/nccl_id.bin.tmp", sync_dir);
    ncclUniqueId id;
    if (g_rank == 0) {
        nccl_check(ncclGetUniqueId(&id), "unique id");
        FILE* f = fopen(tmp, "wb");
        if (!f || fwrite(&id, sizeof(id), 1, f) != 1) { fprintf(stderr, "cannot write %s\n", tmp); exit(1); }
        fclose(f);
        if (rename(tmp, path) != 0) { fprintf(stderr, "cannot rename %s\n", tmp); exit(1); }
    } else {
        int ok = 0;
        for (int i = 0; i < 6000 && !ok; i++) {
            FILE* f = fopen(path, "rb");
            if (f) {
                ok = fread(&id, sizeof(id), 1, f) == 1;
                fclose(f);
            }
            if (!ok) usleep(100000);
        }
        if (!ok) { fprintf(stderr, "rank %d: no %s after 10 min\n", g_rank, path); exit(1); }
    }
    nccl_check(ncclCommInitRank(&g_nccl, g_world, id, g_rank), "comm init");
    printf("rank %d/%d: NCCL ready\n", g_rank, g_world);
#else
    (void)sync_dir;
    if (g_world > 1) {
        fprintf(stderr, "--nproc %d needs a build with NCCL (make USE_NCCL=1)\n", g_world);
        exit(1);
    }
#endif
}

// Sum (op 0) or max (op 1) of one float across ranks.
static float dist_reduce(float x, int op) {
#ifdef TF_NCCL
    if (g_world <= 1) return x;
    ck(cudaMemcpy(g_scalar, &x, sizeof(float), cudaMemcpyHostToDevice), "scalar up");
    nccl_check(ncclAllReduce(g_scalar, g_scalar, 1, ncclFloat, op ? ncclMax : ncclSum, g_nccl, 0), "scalar");
    ck(cudaMemcpy(&x, g_scalar, sizeof(float), cudaMemcpyDeviceToHost), "scalar down");
#else
    (void)op;
#endif
    return x;
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
    float2* rope_cs; // (T, HS/2) cos/sin table for the fused path
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
static int g_fused_loss = 1;
static int g_fuse = 1; // --fuse: fused residual+RMSNorm, stored ReLU^2, RoPE table, vectorized elementwise
static int use_fused(const Model* m) {
    const Config& c = m->cfg;
    return g_fuse && c.rmsnorm && c.relu2 && !c.bias && c.C % 8 == 0 && (c.HS % 2) == 0;
}
static void loss_launch(float* losses, floatX* logits, const int* targets, int n, int V, int Vp,
                        float dloss, float softcap, int write_grad) {
    if (g_fused_loss && V % x128::size == 0 && Vp % x128::size == 0) {
        tf_loss_fused_kernel<<<n, 256>>>(losses, logits, targets, n, V, Vp, dloss, softcap, write_grad);
    } else {
        tf_loss_kernel<<<n, 256, 32 * sizeof(float)>>>(losses, logits, targets, n, V, Vp, dloss, softcap, write_grad);
    }
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
    if (m->cfg.rmsnorm && use_fused(m)) {
        tf_rmsnorm_backward_skip_kernel<<<n, 96>>>(dx, dout, x, rstd, skip_or_null, n, C);
        ck(cudaGetLastError(), "rms bwd skip");
    } else if (m->cfg.rmsnorm) {
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
static int use_fused(const Model* m);
static void qk_backward(Model* m, floatX* dqkv, const floatX* q_post, const floatX* k_post,
                        const float* q_rstd, const float* k_rstd) {
    if (!m->cfg.rope && !m->cfg.qknorm) return;
    int B = m->cfg.B, T = m->cfg.T, NH = m->cfg.NH, HS = m->cfg.HS, C = m->cfg.C;
    int smem = (HS + 32) * (int)sizeof(float);
    if (use_fused(m)) {
        tf_qk_backward_tab_kernel<<<B * T * NH, HS, smem>>>(dqkv, q_post, k_post, q_rstd, k_rstd, m->rope_cs,
                                                           B, T, NH, HS, C, m->cfg.rope, m->cfg.qknorm, m->cfg.qk_scale);
        ck(cudaGetLastError(), "qk backward tab");
        return;
    }
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
    // llm.c's attention_forward reuses its qkv input as the (B, NH, T, T)
    // pre-softmax scratch, so this buffer must hold max(B*T*3C, B*NH*T*T).
    // It used to be B*T*3C only: at d12/T=512 the scores overran it by
    // B*T*C elements into the next allocation (layer 0's saved attention
    // output), corrupting layer 0's attention-projection weight gradient
    // (and crashing the fp16 build).
    m->qkv = (floatX*)cualloc(std::max(BT3, (size_t)c.B * c.NH * c.T * c.T) * sizeof(floatX));
    m->atty = (floatX*)cualloc(L * BTC * sizeof(floatX));
    m->resid2 = (floatX*)cualloc(L * BTC * sizeof(floatX));
    m->preact = (floatX*)cualloc(L * BT4 * sizeof(floatX));
    m->lnf = (floatX*)cualloc(BTC * sizeof(floatX));
    m->logits = (floatX*)cualloc(BT * c.Vp * sizeof(floatX));
    m->qkvr = (floatX*)cualloc(L * 3 * BTC * sizeof(floatX));
    m->att = (floatX*)cualloc(L * m->att_one * sizeof(floatX));
    // L slices: the fused path keeps every layer's ReLU^2 output for the
    // backward pass instead of recomputing it (unfused path uses slice 0).
    m->hid = (floatX*)cualloc(L * BT4 * sizeof(floatX));
    {
        int half = c.HS / 2;
        float2* h = (float2*)malloc((size_t)c.T * half * sizeof(float2));
        for (int t = 0; t < c.T; t++)
            for (int j = 0; j < half; j++) {
                double f = (double)t / pow((double)c.rope_base, (2.0 * j) / (double)c.HS);
                h[(size_t)t * half + j] = make_float2((float)cos(f), (float)sin(f));
            }
        m->rope_cs = (float2*)cualloc((size_t)c.T * half * sizeof(float2));
        ck(cudaMemcpy(m->rope_cs, h, (size_t)c.T * half * sizeof(float2), cudaMemcpyHostToDevice), "rope table");
        free(h);
    }
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

static void tf_forward_fused(Model* m) {
    Config c = m->cfg;
    int B = c.B, T = c.T, C = c.C, L = c.L;
    int BT = B * T;
    PTensor* W = m->tensors;
    tf_embed_forward_kernel<<<divup(BT * C, 256), 256>>>(
        m->embed, m->d_tokens, W[m->id_wte].data, m->cfg.rope ? nullptr : W[m->id_wpe].data,
        B, T, C, m->cfg.rope ? 0 : 1);
    ck(cudaGetLastError(), "embed");
    // layer l reads its input from block_in[l] (no copy); the last layer writes m->x.
    if (c.embed_norm) rms_forward(m->block_in, m->embed_rstd, m->embed, BT, C);
    else ck(cudaMemcpyAsync(m->block_in, m->embed, m->BTC * sizeof(floatX), cudaMemcpyDeviceToDevice), "copy x");
    rms_forward(m->ln1, m->ln1_rstd, m->block_in, BT, C);
    prof_mark("embed");
    int smem = (c.HS + 32) * (int)sizeof(float);
    for (int l = 0; l < L; l++) {
        floatX* cur = m->block_in + (size_t)l * m->BTC;
        floatX* l_ln1 = m->ln1 + (size_t)l * m->BTC;
        matmul_forward_cublaslt(m->qkv, l_ln1, W[m->id_qkv[l]].data, nullptr, B, T, C, 3 * C, 0, nullptr, 0);
        prof_mark("f_gemm");
        if (c.rope || c.qknorm) {
            tf_qk_forward_tab_kernel<<<BT * c.NH, c.HS, smem>>>(m->qkv, m->q_rstd + (size_t)l * BT * c.NH,
                m->k_rstd + (size_t)l * BT * c.NH, m->rope_cs, B, T, c.NH, c.HS, C, c.rope, c.qknorm, c.qk_scale);
            ck(cudaGetLastError(), "qk fwd");
        }
        prof_mark("f_rope_qknorm");
        floatX* l_qkvr = m->qkvr + (size_t)l * 3 * m->BTC;
        floatX* l_att = m->att + (size_t)l * m->att_one;
        floatX* l_atty = m->atty + (size_t)l * m->BTC;
        attention_forward(l_atty, l_qkvr, l_att, m->qkv, B, T, C, c.NH, 0);
        prof_mark("f_attention");
        floatX* proj = m->dx;
        matmul_forward_cublaslt(proj, l_atty, W[m->id_att[l]].data, nullptr, B, T, C, C, 0, nullptr, 0);
        prof_mark("f_gemm");
        floatX* l_r2 = m->resid2 + (size_t)l * m->BTC;
        floatX* l_ln2 = m->ln2 + (size_t)l * m->BTC;
        tf_add_rmsnorm_kernel<<<BT, 96>>>(l_r2, l_ln2, m->ln2_rstd + (size_t)l * BT, cur, proj, BT, C, 1e-6f);
        ck(cudaGetLastError(), "add+rms 2");
        prof_mark("f_norm/copy");
        floatX* pre = m->preact + (size_t)l * m->BT4C;
        floatX* hid = m->hid + (size_t)l * m->BT4C;
        matmul_forward_cublaslt(pre, l_ln2, W[m->id_fc[l]].data, nullptr, B, T, C, 4 * C, 0, nullptr, 0);
        prof_mark("f_gemm");
        tf_relu2_forward_vec_kernel<<<(unsigned)((m->BT4C / 8 + 255) / 256), 256>>>(hid, pre, m->BT4C);
        ck(cudaGetLastError(), "relu2");
        prof_mark("f_relu2");
        matmul_forward_cublaslt(proj, hid, W[m->id_fp[l]].data, nullptr, B, T, 4 * C, C, 0, nullptr, 0);
        prof_mark("f_gemm");
        int last = (l + 1 == L);
        floatX* nxt = last ? m->x : m->block_in + (size_t)(l + 1) * m->BTC;
        floatX* nln = last ? m->lnf : m->ln1 + (size_t)(l + 1) * m->BTC;
        float* nrs = last ? m->lnf_rstd : m->ln1_rstd + (size_t)(l + 1) * BT;
        tf_add_rmsnorm_kernel<<<BT, 96>>>(nxt, nln, nrs, l_r2, proj, BT, C, 1e-6f);
        ck(cudaGetLastError(), "add+rms 1");
        prof_mark("f_norm/copy");
    }
    floatX* head = c.untie ? W[m->id_head].data : W[m->id_wte].data;
    matmul_forward_cublaslt(m->logits, m->lnf, head, nullptr, B, T, C, c.Vp, 0, nullptr, 0);
    ck(cudaGetLastError(), "logits");
    prof_mark("f_head");
}

static void tf_forward_unfused(Model* m);
static void tf_forward(Model* m) {
    if (use_fused(m)) tf_forward_fused(m);
    else tf_forward_unfused(m);
}

static void tf_forward_unfused(Model* m) {
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

    prof_mark("embed");
    for (int l = 0; l < L; l++) {
        floatX* cur = m->x;
        floatX* saved_in = m->block_in + (size_t)l * m->BTC;
        ck(cudaMemcpy(saved_in, cur, m->BTC * sizeof(floatX), cudaMemcpyDeviceToDevice), "save in");
        floatX* l_ln1 = m->ln1 + (size_t)l * m->BTC;
        const floatX* w1 = m->id_ln1w[l] >= 0 ? W[m->id_ln1w[l]].data : nullptr;
        const floatX* b1 = m->id_ln1b[l] >= 0 ? W[m->id_ln1b[l]].data : nullptr;
        norm_forward(m, l_ln1, m->ln1_mean + (size_t)l * BT, m->ln1_rstd + (size_t)l * BT, cur, w1, b1, BT);
        floatX* qb = m->id_qkvb[l] >= 0 ? W[m->id_qkvb[l]].data : nullptr;
        prof_mark("f_norm/copy");
        matmul_forward_cublaslt(m->qkv, l_ln1, W[m->id_qkv[l]].data, qb, B, T, C, 3 * C, 0, nullptr, 0);
        prof_mark("f_gemm");
        qk_forward(m, m->qkv, m->q_rstd + (size_t)l * BT * c.NH, m->k_rstd + (size_t)l * BT * c.NH);
        prof_mark("f_rope_qknorm");
        floatX* l_qkvr = m->qkvr + (size_t)l * 3 * m->BTC;
        floatX* l_att = m->att + (size_t)l * m->att_one;
        floatX* l_atty = m->atty + (size_t)l * m->BTC;
        attention_forward(l_atty, l_qkvr, l_att, m->qkv, B, T, C, c.NH, 0);
        prof_mark("f_attention");
        floatX* ab = m->id_attb[l] >= 0 ? W[m->id_attb[l]].data : nullptr;
        floatX* proj = m->dx;
        matmul_forward_cublaslt(proj, l_atty, W[m->id_att[l]].data, ab, B, T, C, C, 0, nullptr, 0);
        prof_mark("f_gemm");
        floatX* l_r2 = m->resid2 + (size_t)l * m->BTC;
        tf_residual_kernel<<<divup(BT * C, 256), 256>>>(l_r2, saved_in, proj, BT * C);
        ck(cudaGetLastError(), "resid2");
        floatX* l_ln2 = m->ln2 + (size_t)l * m->BTC;
        const floatX* w2 = m->id_ln2w[l] >= 0 ? W[m->id_ln2w[l]].data : nullptr;
        const floatX* b2 = m->id_ln2b[l] >= 0 ? W[m->id_ln2b[l]].data : nullptr;
        norm_forward(m, l_ln2, m->ln2_mean + (size_t)l * BT, m->ln2_rstd + (size_t)l * BT, l_r2, w2, b2, BT);
        floatX* pre = m->preact + (size_t)l * m->BT4C;
        floatX* fcb = m->id_fcb[l] >= 0 ? W[m->id_fcb[l]].data : nullptr;
        prof_mark("f_norm/copy");
        matmul_forward_cublaslt(pre, l_ln2, W[m->id_fc[l]].data, fcb, B, T, C, 4 * C, 0, nullptr, 0);
        prof_mark("f_gemm");
        if (c.relu2) tf_relu2_forward_kernel<<<divup((int)m->BT4C, 256), 256>>>(m->hid, pre, (int)m->BT4C);
        else gelu_forward(m->hid, pre, (int)m->BT4C, 0);
        ck(cudaGetLastError(), "mlp act");
        floatX* fpb = m->id_fpb[l] >= 0 ? W[m->id_fpb[l]].data : nullptr;
        prof_mark("f_relu2");
        matmul_forward_cublaslt(proj, m->hid, W[m->id_fp[l]].data, fpb, B, T, 4 * C, C, 0, nullptr, 0);
        prof_mark("f_gemm");
        tf_residual_kernel<<<divup(BT * C, 256), 256>>>(m->x, l_r2, proj, BT * C);
        ck(cudaGetLastError(), "resid3");
        prof_mark("f_norm/copy");
    }
    norm_forward(m, m->lnf, m->lnf_mean, m->lnf_rstd, m->x,
                 m->id_lnfw >= 0 ? W[m->id_lnfw].data : nullptr,
                 m->id_lnfb >= 0 ? W[m->id_lnfb].data : nullptr, BT);
    floatX* head = c.untie ? W[m->id_head].data : W[m->id_wte].data;
    prof_mark("f_norm/copy");
    matmul_forward_cublaslt(m->logits, m->lnf, head, nullptr, B, T, C, c.Vp, 0, nullptr, 0);
    ck(cudaGetLastError(), "logits");
    prof_mark("f_head");
}

static float tf_backward(Model* m, float dloss) {
    Config c = m->cfg;
    int B = c.B, T = c.T, C = c.C, L = c.L;
    int BT = B * T;
    PTensor* W = m->tensors;
    loss_launch(m->losses, m->logits, m->d_targets, BT, c.V, c.Vp, dloss, c.softcap, 1);
    ck(cudaGetLastError(), "loss");
    prof_mark("loss");
    floatX* head = c.untie ? W[m->id_head].data : W[m->id_wte].data;
    floatX* dhead = c.untie ? W[m->id_head].grad : W[m->id_wte].grad;
    matmul_backward(m->dx, dhead, nullptr, m->logits, m->lnf, head, nullptr, B, T, C, c.Vp, 0, nullptr, 0);
    prof_mark("b_head");
    floatX* dstream = m->dres;
    norm_backward(m, dstream, m->id_lnfw >= 0 ? W[m->id_lnfw].grad : nullptr,
                  m->id_lnfb >= 0 ? W[m->id_lnfb].grad : nullptr,
                  m->dx, m->x, m->id_lnfw >= 0 ? W[m->id_lnfw].data : nullptr,
                  m->lnf_mean, m->lnf_rstd, nullptr);

    for (int l = L - 1; l >= 0; l--) {
        floatX* l_r2 = m->resid2 + (size_t)l * m->BTC;
        floatX* pre = m->preact + (size_t)l * m->BT4C;
        const int fused = use_fused(m);
        floatX* hid = fused ? m->hid + (size_t)l * m->BT4C : m->hid;
        if (fused) {}
        else if (c.relu2) tf_relu2_forward_kernel<<<divup((int)m->BT4C, 256), 256>>>(m->hid, pre, (int)m->BT4C);
        else gelu_forward(m->hid, pre, (int)m->BT4C, 0);
        prof_mark("b_relu2");
        floatX* fpb_g = m->id_fpb[l] >= 0 ? W[m->id_fpb[l]].grad : nullptr;
        matmul_backward(m->dhid, W[m->id_fp[l]].grad, fpb_g, dstream, hid, W[m->id_fp[l]].data,
                        nullptr, B, T, 4 * C, C, 0, nullptr, 0);
        prof_mark("b_gemm");
        if (fused) tf_relu2_backward_vec_kernel<<<(unsigned)((m->BT4C / 8 + 255) / 256), 256>>>(m->dhid, m->dhid, pre, m->BT4C);
        else if (c.relu2) tf_relu2_backward_kernel<<<divup((int)m->BT4C, 256), 256>>>(m->dhid, m->dhid, pre, (int)m->BT4C);
        else gelu_backward_inplace(m->dhid, pre, (int)m->BT4C, 0);
        ck(cudaGetLastError(), "act bwd");
        prof_mark("b_relu2");
        floatX* l_ln2 = m->ln2 + (size_t)l * m->BTC;
        floatX* fcb_g = m->id_fcb[l] >= 0 ? W[m->id_fcb[l]].grad : nullptr;
        matmul_backward(m->dln, W[m->id_fc[l]].grad, fcb_g, m->dhid, l_ln2, W[m->id_fc[l]].data,
                        nullptr, B, T, C, 4 * C, 0, nullptr, 0);
        prof_mark("b_gemm");
        norm_backward(m, m->dx,
                      m->id_ln2w[l] >= 0 ? W[m->id_ln2w[l]].grad : nullptr,
                      m->id_ln2b[l] >= 0 ? W[m->id_ln2b[l]].grad : nullptr,
                      m->dln, l_r2,
                      m->id_ln2w[l] >= 0 ? W[m->id_ln2w[l]].data : nullptr,
                      m->ln2_mean + (size_t)l * BT, m->ln2_rstd + (size_t)l * BT, dstream);
        // dx is now d(resid2) = skip(dstream) + mlp norm grad. dstream can be reused.
        prof_mark("b_norm/copy");
        floatX* l_atty = m->atty + (size_t)l * m->BTC;
        floatX* ab_g = m->id_attb[l] >= 0 ? W[m->id_attb[l]].grad : nullptr;
        matmul_backward(m->datty, W[m->id_att[l]].grad, ab_g, m->dx, l_atty, W[m->id_att[l]].data,
                        nullptr, B, T, C, C, 0, nullptr, 0);
        prof_mark("b_gemm");
        floatX* l_qkvr = m->qkvr + (size_t)l * 3 * m->BTC;
        floatX* l_att = m->att + (size_t)l * m->att_one;
        attention_backward(m->dqkv, m->qkv, m->datt, m->scratch_btc, m->datty, l_qkvr, l_att, B, T, C, c.NH, 0);
        prof_mark("b_attention");
        qk_backward(m, m->dqkv, l_qkvr, l_qkvr + m->BTC, m->q_rstd + (size_t)l * BT * c.NH,
                    m->k_rstd + (size_t)l * BT * c.NH);
        prof_mark("b_rope_qknorm");
        floatX* l_ln1 = m->ln1 + (size_t)l * m->BTC;
        floatX* qb_g = m->id_qkvb[l] >= 0 ? W[m->id_qkvb[l]].grad : nullptr;
        matmul_backward(m->dln, W[m->id_qkv[l]].grad, qb_g, m->dqkv, l_ln1, W[m->id_qkv[l]].data,
                        nullptr, B, T, C, 3 * C, 0, nullptr, 0);
        prof_mark("b_gemm");
        floatX* bin = m->block_in + (size_t)l * m->BTC;
        norm_backward(m, dstream,
                      m->id_ln1w[l] >= 0 ? W[m->id_ln1w[l]].grad : nullptr,
                      m->id_ln1b[l] >= 0 ? W[m->id_ln1b[l]].grad : nullptr,
                      m->dln, bin,
                      m->id_ln1w[l] >= 0 ? W[m->id_ln1w[l]].data : nullptr,
                      m->ln1_mean + (size_t)l * BT, m->ln1_rstd + (size_t)l * BT, m->dx);
        prof_mark("b_norm/copy");
    }
    // dstream is d(x) at the embedding-norm output.
    floatX* dembed = m->dx;
    if (c.embed_norm) rms_backward(dembed, dstream, m->embed, m->embed_rstd, BT, C);
    else ck(cudaMemcpy(dembed, dstream, m->BTC * sizeof(floatX), cudaMemcpyDeviceToDevice), "dembed");
    tf_embed_backward_kernel<<<divup(BT * C, 256), 256>>>(
        m->emb_g32, m->wpe_g32, dembed, m->d_tokens, BT, T, C, c.rope ? 0 : 1);
    ck(cudaGetLastError(), "embed bwd");
    prof_mark("b_embed");

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
    if (g_ns_fast16) {
        // fp32 in/out, inputs rounded to fp16 for the tensor cores (Turing has no TF32).
        cublasCheck(cublasGemmEx(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                 &alpha, B, CUDA_R_32F, n, A, CUDA_R_32F, k, &beta, C, CUDA_R_32F, n,
                                 CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));
    } else {
        cublasCheck(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                &alpha, B, n, A, k, &beta, C, n));
    }
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

// fp16 builds (Turing T4: no bf16, no TF32) use dynamic loss scaling: the
// backward pass runs on loss * g_loss_scale so small gradients survive fp16,
// the optimizer multiplies gradients by g_grad_unscale, and a step whose
// gradients overflow is skipped with the scale halved.
static float g_grad_unscale = 1.f;

static void adam_update(PTensor* t, float lr, float b1, float b2, float eps, float wd, int step) {
    float b1c = 1.f - powf(b1, (float)(step + 1));
    float b2c = 1.f - powf(b2, (float)(step + 1));
    tf_adamw_kernel<<<divup((int)std::min(t->n, (size_t)1 << 30), 256), 256>>>(
        t->master, t->data, t->grad, t->m, t->v, t->n, lr, b1, b2, b1c, b2c, eps, wd, g_grad_unscale);
    ck(cudaGetLastError(), "adamw");
}

// Muon for all matrices of one shape at once: strided-batched Newton-Schulz GEMMs
// (one launch per GEMM per iteration instead of one per matrix) and no host syncs.
// Same math as muon_ns_and_scale + tf_muon_apply_kernel.
static int g_ns_batch = 1;
static void sgemm_batched(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                          const float* A, int lda, long long sA, const float* B, int ldb, long long sB,
                          float* C, int ldc, long long sC, int nb) {
    const float alpha = 1.f, beta = 0.f;
    cublasCheck(cublasGemmStridedBatchedEx(g_cublas, ta, tb, m, n, k, &alpha,
        A, CUDA_R_32F, lda, sA, B, CUDA_R_32F, ldb, sB, &beta, C, CUDA_R_32F, ldc, sC, nb,
        g_ns_fast16 ? CUBLAS_COMPUTE_32F_FAST_16F : CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
static void muon_group(Model* m, int* ids, int nb, float lr, float wd, float momentum) {
    static float *X = nullptr, *Y = nullptr, *A = nullptr, *AA = nullptr, *Bm = nullptr, *ss = nullptr;
    static size_t capXY = 0, capA = 0;
    PTensor* t0 = &m->tensors[ids[0]];
    int rows = t0->rows, cols = t0->cols;
    int trans = rows > cols;
    int rm = trans ? cols : rows, rn = trans ? rows : cols;
    size_t per = (size_t)rm * rn, perA = (size_t)rm * rm;
    if (per * nb > capXY) {
        if (X) { cudaFree(X); cudaFree(Y); }
        capXY = per * nb;
        X = (float*)cualloc(capXY * sizeof(float)); Y = (float*)cualloc(capXY * sizeof(float));
    }
    if (perA * nb > capA) {
        if (A) { cudaFree(A); cudaFree(AA); cudaFree(Bm); }
        capA = perA * nb;
        A = (float*)cualloc(capA * sizeof(float)); AA = (float*)cualloc(capA * sizeof(float));
        Bm = (float*)cualloc(capA * sizeof(float));
    }
    if (!ss) ss = (float*)cualloc(MAX_TENSORS * sizeof(float));
    dim3 block(16, 16);
    for (int b = 0; b < nb; b++) {
        PTensor* t = &m->tensors[ids[b]];
        float* dst = trans ? Y + b * per : X + b * per;
        tf_muon_nesterov_kernel<<<divup((int)t->n, 256), 256>>>(dst, t->m, t->grad, t->n, momentum, g_grad_unscale);
        if (trans) tf_transpose_kernel<<<dim3(divup(cols, 16), divup(rows, 16)), block>>>(X + b * per, dst, rows, cols);
    }
    ck(cudaMemsetAsync(ss, 0, nb * sizeof(float)), "ns ss");
    tf_batched_sumsq_kernel<<<dim3(64, nb), 256>>>(ss, X, per);
    tf_batched_normalize_kernel<<<dim3(64, nb), 256>>>(X, ss, per);
    const float a = 3.4445f, bcoef = -4.7750f, c = 2.0315f;
    for (int it = 0; it < 5; it++) {
        // A = X X^T (row-major rm x rn X is a column-major rn x rm matrix Xc; A = Xc^T Xc)
        sgemm_batched(CUBLAS_OP_T, CUBLAS_OP_N, rm, rm, rn, X, rn, per, X, rn, per, A, rm, perA, nb);
        // AA = A A (symmetric)
        sgemm_batched(CUBLAS_OP_N, CUBLAS_OP_N, rm, rm, rm, A, rm, perA, A, rm, perA, AA, rm, perA, nb);
        tf_combo_kernel<<<divup((int)(perA * nb), 256), 256>>>(Bm, A, AA, (int)(perA * nb), bcoef, c);
        // Y = B X (row-major), via column-major Y^T = X^T B^T
        sgemm_batched(CUBLAS_OP_N, CUBLAS_OP_N, rn, rm, rm, X, rn, per, Bm, rm, perA, Y, rn, per, nb);
        tf_axpy_kernel<<<divup((int)(per * nb), 256), 256>>>(X, Y, per * nb, a, 1.f);
    }
    float aspect = sqrtf(fmaxf(1.f, (float)rows / (float)cols));
    for (int b = 0; b < nb; b++) {
        PTensor* t = &m->tensors[ids[b]];
        const float* upd = X + b * per;
        if (trans) {
            tf_transpose_kernel<<<dim3(divup(rm, 16), divup(rn, 16)), block>>>(Y + b * per, X + b * per, rm, rn);
            upd = Y + b * per;
        }
        tf_muon_apply_scaled_kernel<<<divup((int)t->n, 256), 256>>>(t->master, t->data, upd, t->n, lr, wd, aspect);
    }
    ck(cudaGetLastError(), "muon group");
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
        } else if (g_ns_batch) {
            continue; // batched below
        } else {
            float lr = matrix_lr * batch_scale * mult;
            tf_muon_nesterov_kernel<<<divup((int)std::min(t->n, (size_t)1 << 30), 256), 256>>>(
                m->ns_y, t->m, t->grad, t->n, momentum, g_grad_unscale);
            ck(cudaGetLastError(), "nesterov");
            muon_ns_and_scale(m, m->ns_y, t->rows, t->cols);
            tf_muon_apply_kernel<<<divup((int)std::min(t->n, (size_t)1 << 30), 256), 256>>>(
                t->master, t->data, m->ns_y, t->n, lr, muon_wd);
            ck(cudaGetLastError(), "muon apply");
        }
    }
    if (g_ns_batch) {
        // group Muon tensors by shape, in registration order
        int done[MAX_TENSORS] = {0};
        int ids[MAX_TENSORS];
        float lr = matrix_lr * batch_scale * mult;
        for (int i = 0; i < m->ntensors; i++) {
            PTensor* t = &m->tensors[i];
            if (t->kind != KIND_MUON || t->n == 0 || done[i]) continue;
            int nb = 0;
            for (int j = i; j < m->ntensors; j++) {
                PTensor* u = &m->tensors[j];
                if (!done[j] && u->kind == KIND_MUON && u->rows == t->rows && u->cols == t->cols) {
                    ids[nb++] = j; done[j] = 1;
                }
            }
            muon_group(m, ids, nb, lr, muon_wd, momentum);
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
#ifdef ENABLE_FP16
    const int write_master = 1; // fp16 build: write the fp32 master weights
#else
    const int write_master = 0;
#endif
    h[12] = (sizeof(floatX) == 4 || write_master) ? 0 : 2; // 0 fp32, 2 bf16
    h[13] = m->ntensors;
    fwrite(h, sizeof(int), 256, f);
    size_t maxn = 1;
    for (int i = 0; i < m->ntensors; i++) if (m->tensors[i].n > maxn) maxn = m->tensors[i].n;
    float* host = (float*)malloc(maxn * sizeof(float));
    for (int i = 0; i < m->ntensors; i++) {
        PTensor* t = &m->tensors[i];
        if (write_master) {
            ck(cudaMemcpy(host, t->master, t->n * sizeof(float), cudaMemcpyDeviceToHost), "ckpt");
            fwrite(host, sizeof(float), t->n, f);
        } else {
            ck(cudaMemcpy(host, t->data, t->n * sizeof(floatX), cudaMemcpyDeviceToHost), "ckpt");
            fwrite(host, sizeof(floatX), t->n, f);
        }
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
    h[5] = m->cfg.B; h[6] = m->cfg.T; h[7] = g_world;
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

static void read_state(Model* m, const char* path, int* step, int* shard, int* sample, int* world) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
    int h[256];
    if (fread(h, sizeof(int), 256, f) != 256 || h[0] != 20241115) {
        fprintf(stderr, "bad state file %s\n", path); exit(1);
    }
    *step = h[2]; *shard = h[3]; *sample = h[4];
    *world = h[7] > 0 ? h[7] : 1;
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

static void dist_average_grads(Model* m) {
#ifdef TF_NCCL
    if (g_world <= 1) return;
    nccl_check(ncclGroupStart(), "group start");
    for (int i = 0; i < m->ntensors; i++) {
        PTensor* t = &m->tensors[i];
        nccl_check(ncclAllReduce(t->grad, t->grad, t->n, TF_NCCL_FLOATX, ncclAvg, g_nccl, 0), "grads");
    }
    nccl_check(ncclGroupEnd(), "group end");
    ck(cudaStreamSynchronize(0), "grad sync");
#else
    (void)m;
#endif
}

// Write to a temp name and rename, so a killed session never leaves a
// half-written file under the real name. DONE_ is written last.
static void finish_file(const char* tmp, const char* path) {
    if (rename(tmp, path) != 0) { fprintf(stderr, "cannot rename %s\n", tmp); exit(1); }
}

// Keep only the newest `keep` checkpoints in dir.
static void prune_ckpts(const char* dir, int keep) {
    if (keep <= 0) return;
    int steps[4096];
    int n = 0;
    DIR* d = opendir(dir);
    if (!d) return;
    struct dirent* e;
    while ((e = readdir(d)) != NULL && n < 4096) {
        if (strncmp(e->d_name, "DONE_", 5) == 0) steps[n++] = atoi(e->d_name + 5);
    }
    closedir(d);
    std::sort(steps, steps + n);
    for (int i = 0; i + keep < n; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/DONE_%08d", dir, steps[i]); remove(path);
        snprintf(path, sizeof(path), "%s/model_%08d.bin", dir, steps[i]); remove(path);
        snprintf(path, sizeof(path), "%s/state_%08d.bin", dir, steps[i]); remove(path);
    }
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
        "  --time-limit <seconds: checkpoint and exit>  --keep-ckpts <N newest kept>\n"
        "  --nproc N --rank R --sync-dir <dir>   (one process per GPU, NCCL build)\n"
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
    double time_limit = 0;
    int prof_steps = 0;
#ifdef ENABLE_FP16
    float loss_scale = 65536.f; // dynamic; halves on overflow, doubles after 200 clean steps
#else
    float loss_scale = 1.f;
#endif
    int keep_ckpts = 0;
    const char* sync_dir = nullptr;

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
        else if (strcmp(a, "--time-limit") == 0) time_limit = atof(v);
        else if (strcmp(a, "--loss-scale") == 0) loss_scale = arg_float(v);
        else if (strcmp(a, "--profile") == 0) prof_steps = arg_int(v);
        else if (strcmp(a, "--ns-fp16") == 0) g_ns_fast16 = arg_int(v);
        else if (strcmp(a, "--fused-loss") == 0) g_fused_loss = arg_int(v);
        else if (strcmp(a, "--fuse") == 0) g_fuse = arg_int(v);
        else if (strcmp(a, "--ns-batch") == 0) g_ns_batch = arg_int(v);
        else if (strcmp(a, "--keep-ckpts") == 0) keep_ckpts = arg_int(v);
        else if (strcmp(a, "--nproc") == 0) g_world = arg_int(v);
        else if (strcmp(a, "--rank") == 0) g_rank = arg_int(v);
        else if (strcmp(a, "--sync-dir") == 0) sync_dir = v;
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
    if (g_world < 1 || g_rank < 0 || g_rank >= g_world) { fprintf(stderr, "bad --nproc/--rank\n"); return 1; }
    if (g_world > 1 && !sync_dir) sync_dir = out_dir ? out_dir : ".";
    if (total_batch < 0) total_batch = B * T * g_world;
    if (total_batch % (B * T * g_world) != 0) {
        fprintf(stderr, "total batch %d is not a multiple of B*T*nproc = %d\n", total_batch, B * T * g_world);
        return 1;
    }
    int accum = total_batch / (B * T * g_world);
    // Only rank 0 talks; the other ranks print errors only.
    if (g_rank != 0 && !freopen("/dev/null", "w", stdout)) return 1;

    ck(cudaSetDevice(g_rank), "set device");
    ck(cudaGetDeviceProperties(&deviceProp, g_rank), "device");
    g_scalar = (float*)cualloc(sizeof(float));
    dist_init(sync_dir);
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    ck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size), "lt workspace");
    cublas_compute = CUBLAS_COMPUTE_32F;
    cublasCheck(cublasCreate(&g_cublas));
    cublasCheck(cublasCreate(&cublas_handle_ex));
    cublasCheck(cublasSetWorkspace(cublas_handle_ex, cublaslt_workspace, cublaslt_workspace_size));
    printf("device: %s\n", deviceProp.name);
    printf("arch %s  d%d  L=%d C=%d NH=%d HS=%d V=%d Vp=%d\n",
           modern ? "modern" : "gpt2", depth, cfg.L, cfg.C, cfg.NH, cfg.HS, cfg.V, cfg.Vp);
    printf("rope=%d rmsnorm=%d qknorm=%d relu2=%d untie=%d bias=%d embed_norm=%d softcap=%.3f optim=%s\n",
           cfg.rope, cfg.rmsnorm, cfg.qknorm, cfg.relu2, cfg.untie, cfg.bias, cfg.embed_norm, cfg.softcap,
           muon ? "muon" : "adamw");
    printf("B=%d T=%d nproc=%d total_batch=%d accum=%d\n", B, T, g_world, total_batch, accum);

    Model* model = (Model*)calloc(1, sizeof(Model));
    model->cfg = cfg;
    register_model(model, muon);
    alloc_acts(model);
    {
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "meminfo");
        printf("GPU memory after allocation: %.2f GiB used of %.2f GiB\n",
               (total_b - free_b) / 1073741824.0, total_b / 1073741824.0);
    }

    {
        // Startup self-check of the GEMM path (cuBLASLt or the fp16 GemmEx path) in the
        // three transpose modes the trainer uses, against a CPU reference.
        const int M = 96, N = 64, K = 128;
        float worst = 0.f;
        for (int mode = 0; mode < 3; mode++) {
            bool tA = mode == 0, tB = mode == 2;
            floatX *ha = (floatX*)malloc(M * K * sizeof(floatX)), *hb = (floatX*)malloc(K * N * sizeof(floatX));
            floatX* hd = (floatX*)malloc(M * N * sizeof(floatX));
            mt19937_state r; manual_seed(&r, 11 + mode);
            for (int i = 0; i < M * K; i++) ha[i] = (floatX)(randfloat32(&r) - 0.5f);
            for (int i = 0; i < K * N; i++) hb[i] = (floatX)(randfloat32(&r) - 0.5f);
            floatX *da = (floatX*)cualloc(M * K * sizeof(floatX)), *db = (floatX*)cualloc(K * N * sizeof(floatX));
            floatX* dd = (floatX*)cualloc(M * N * sizeof(floatX));
            ck(cudaMemcpy(da, ha, M * K * sizeof(floatX), cudaMemcpyHostToDevice), "g");
            ck(cudaMemcpy(db, hb, K * N * sizeof(floatX), cudaMemcpyHostToDevice), "g");
            matmul_cublaslt(dd, da, db, nullptr, M, N, K, 0, tA, tB);
            ck(cudaMemcpy(hd, dd, M * N * sizeof(floatX), cudaMemcpyDeviceToHost), "g");
            for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
                float acc = 0.f;
                for (int kk = 0; kk < K; kk++) {
                    float av = tA ? (float)ha[i * K + kk] : (float)ha[kk * M + i];
                    float bv = tB ? (float)hb[kk * N + j] : (float)hb[j * K + kk];
                    acc += av * bv;
                }
                worst = std::max(worst, fabsf(acc - (float)hd[j * M + i]));
            }
            cudaFree(da); cudaFree(db); cudaFree(dd); free(ha); free(hb); free(hd);
        }
        printf("gemm check: max abs error %.2e\n", worst);
        if (worst > 2e-2f) { fprintf(stderr, "GEMM self-check failed\n"); return 1; }
    }
    if (g_fused_loss) {
        // Startup self-check: fused loss kernel against the reference kernel on random logits.
        int n = 64, V = cfg.V, Vp = cfg.Vp;
        size_t ne = (size_t)n * Vp;
        floatX* h = (floatX*)malloc(ne * sizeof(floatX));
        int* ht = (int*)malloc(n * sizeof(int));
        mt19937_state r; manual_seed(&r, 7);
        for (size_t i = 0; i < ne; i++) h[i] = (floatX)(8.f * (randfloat32(&r) - 0.5f) * (i % 97 == 0 ? 6.f : 1.f));
        for (int i = 0; i < n; i++) ht[i] = (int)(randfloat32(&r) * (V - 1));
        floatX* a = (floatX*)cualloc(ne * sizeof(floatX));
        floatX* b = (floatX*)cualloc(ne * sizeof(floatX));
        float* la = (float*)cualloc(n * sizeof(float));
        float* lb = (float*)cualloc(n * sizeof(float));
        int* dt = (int*)cualloc(n * sizeof(int));
        ck(cudaMemcpy(a, h, ne * sizeof(floatX), cudaMemcpyHostToDevice), "chk");
        ck(cudaMemcpy(b, h, ne * sizeof(floatX), cudaMemcpyHostToDevice), "chk");
        ck(cudaMemcpy(dt, ht, n * sizeof(int), cudaMemcpyHostToDevice), "chk");
        float dl = 1024.f / n;
        tf_loss_kernel<<<n, 256, 32 * sizeof(float)>>>(la, a, dt, n, V, Vp, dl, cfg.softcap, 1);
        tf_loss_fused_kernel<<<n, 256>>>(lb, b, dt, n, V, Vp, dl, cfg.softcap, 1);
        ck(cudaDeviceSynchronize(), "loss check");
        float hla[64], hlb[64];
        ck(cudaMemcpy(hla, la, n * sizeof(float), cudaMemcpyDeviceToHost), "chk");
        ck(cudaMemcpy(hlb, lb, n * sizeof(float), cudaMemcpyDeviceToHost), "chk");
        floatX* ga = (floatX*)malloc(ne * sizeof(floatX));
        ck(cudaMemcpy(h, a, ne * sizeof(floatX), cudaMemcpyDeviceToHost), "chk");
        ck(cudaMemcpy(ga, b, ne * sizeof(floatX), cudaMemcpyDeviceToHost), "chk");
        double dmax = 0, gmax = 0, gref = 0;
        for (int i = 0; i < n; i++) dmax = std::max(dmax, (double)fabsf(hla[i] - hlb[i]));
        for (size_t i = 0; i < ne; i++) {
            gmax = std::max(gmax, (double)fabsf((float)h[i] - (float)ga[i]));
            gref = std::max(gref, (double)fabsf((float)h[i]));
        }
        printf("fused loss check: max |dloss| %.2e (loss ~%.2f), max |dgrad| %.2e of %.2e\n", dmax, hla[0], gmax, gref);
        if (dmax > 1e-3 || gmax > 1e-2 * gref) { fprintf(stderr, "fused loss kernel mismatch; use --fused-loss 0\n"); return 1; }
        cudaFree(a); cudaFree(b); cudaFree(la); cudaFree(lb); cudaFree(dt); free(h); free(ht); free(ga);
    }

    int start_step = 0, resume_shard = 0, resume_sample = 0, resume_world = 1;
    int resuming = 0;
    if (resume && out_dir) {
        int done = find_max_step(out_dir);
        if (done > 0) {
            char path[512];
            snprintf(path, sizeof(path), "%s/state_%08d.bin", out_dir, done);
            read_state(model, path, &start_step, &resume_shard, &resume_sample, &resume_world);
            resuming = 1;
            if (resume_world != g_world) {
                // A loader sample is nproc*B*T tokens, so rescale the cursor.
                resume_sample = (int)((long long)resume_sample * resume_world / g_world);
                printf("checkpoint was nproc=%d, now nproc=%d: data cursor rescaled\n", resume_world, g_world);
            }
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
    dataloader_init(&train_loader, train_pat, B, T, g_rank, g_world, 0);
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
        size_t got = 0;
        // train_tokenizer.py writes an int32 count header and then `count`
        // int32 byte lengths. With --vocab one larger than the tokenizer
        // (the usual case) that file is exactly V*4 bytes, the same size as a
        // headerless table, so the size alone cannot tell them apart. The old
        // code read it as headerless: every id was shifted by one and id 0
        // (BOS) got the header value (~32767 bytes), which inflated the byte
        // count ~9x and made bpb ~9x too small. Check the header first.
        int count = 0;
        if (fread(&count, 4, 1, bf) != 1) count = 0;
        memset(token_bytes, 0, (size_t)cfg.V * sizeof(int));
        if (count > 0 && count <= cfg.V && sz == (long)(count + 1) * 4) {
            got = fread(token_bytes, 4, count, bf);
            got = (got == (size_t)count) ? (size_t)cfg.V : 0;
        } else if (sz == (long)cfg.V * 4) {
            fseek(bf, 0, SEEK_SET);
            got = fread(token_bytes, 4, cfg.V, bf);
        }
        int maxb = 0;
        for (int i = 0; i < cfg.V; i++) maxb = std::max(maxb, token_bytes[i]);
        if (got == (size_t)cfg.V && maxb > 256) {
            fprintf(stderr, "%s: a token of %d bytes looks wrong\n", token_bytes_path, maxb);
            return 1;
        }
        if (got != (size_t)cfg.V) { fprintf(stderr, "short read on %s\n", token_bytes_path); return 1; }
        fclose(bf);
    }

    if (out_dir) create_dir_if_not_exists(out_dir);
    int* h_tok = (int*)malloc((size_t)B * T * sizeof(int));
    int* h_tgt = (int*)malloc((size_t)B * T * sizeof(int));
    float* h_loss = (float*)malloc((size_t)B * T * sizeof(float));

    int last_saved = resuming ? start_step : -1;
    auto save_ckpt = [&](int completed) {
        if (!out_dir || g_rank != 0 || completed == last_saved) return;
        char path[512], tmp[520];
        snprintf(path, sizeof(path), "%s/model_%08d.bin", out_dir, completed);
        snprintf(tmp, sizeof(tmp), "%s.tmp", path);
        printf("writing %s\n", path);
        write_model(model, tmp);
        finish_file(tmp, path);
        snprintf(path, sizeof(path), "%s/state_%08d.bin", out_dir, completed);
        snprintf(tmp, sizeof(tmp), "%s.tmp", path);
        write_state(model, tmp, completed, (int)train_loader.current_shard_idx, (int)train_loader.current_sample_idx);
        finish_file(tmp, path);
        snprintf(path, sizeof(path), "%s/DONE_%08d", out_dir, completed);
        FILE* df = fopen(path, "w");
        if (df) fclose(df);
        last_saved = completed;
        prune_ckpts(out_dir, keep_ckpts);
        fflush(stdout);
    };
    int* d_nonfinite = (int*)cualloc(sizeof(int));
    int clean_steps = 0, skipped_steps = 0;
    if (loss_scale != 1.f) printf("fp16 dynamic loss scale, start %.0f\n", loss_scale);
    double t_start = now_seconds();
    double step_time_sum = 0;
    int step_time_n = 0;

    for (int step = start_step; step < max_steps; step++) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0);
        g_prof = (prof_steps > 0 && step >= start_step + 2 && step < start_step + 2 + prof_steps);
        prof_mark("start");
        zero_grads(model);
        prof_mark("zero");
        double loss_sum = 0;
        float dloss = loss_scale / (float)(B * T * accum);
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
            prof_mark("data");
            tf_forward(model);
            loss_sum += tf_backward(model, dloss);
        }
        fold_embed_grads(model);
        dist_average_grads(model);
        prof_mark("fold+allreduce");
        float mult = lr_multiplier(step, max_steps, warmup, warmdown, final_frac, sched);
        int overflow = 0;
        if (loss_scale != 1.f) {
            ck(cudaMemsetAsync(d_nonfinite, 0, sizeof(int)), "flag");
            for (int i = 0; i < model->ntensors; i++) {
                PTensor* t = &model->tensors[i];
                if (t->n == 0) continue;
                int blocks = (int)std::min((size_t)1024, (t->n + 255) / 256);
                tf_nonfinite_kernel<<<blocks, 256>>>(d_nonfinite, t->grad, t->n);
            }
            ck(cudaMemcpy(&overflow, d_nonfinite, sizeof(int), cudaMemcpyDeviceToHost), "flag read");
        }
        if (overflow) {
            // Gradients are identical on every rank after the all-reduce, so all ranks skip together.
            loss_scale = std::max(1.f, loss_scale * 0.5f);
            clean_steps = 0;
            skipped_steps++;
            printf("step %d: fp16 gradient overflow, step skipped, loss scale -> %.0f\n", step + 1, loss_scale);
        } else {
            g_grad_unscale = 1.f / loss_scale;
            optimizer_step(model, step, mult, batch_scale, dmodel_scale,
                           matrix_lr, embed_lr, unembed_lr, muon_wd, embed_wd, head_wd, vec_wd,
                           momentum, modern ? 0 : 1, gpt2_lr, vec_wd);
            if (loss_scale != 1.f && ++clean_steps >= 200 && loss_scale < 16777216.f) {
                loss_scale *= 2.f;
                clean_steps = 0;
            }
        }
        prof_mark("optimizer");
        prof_report();
        if (g_world > 1 && (step == start_step || (step + 1) % 100 == 0)) {
            // Every rank applies the same averaged gradient, so weights must match.
            PTensor* probe = &model->tensors[model->id_qkv[0]];
            float f = frobenius(model, probe->master, probe->n);
            float fmax = dist_reduce(f, 1), fsum = dist_reduce(f, 0);
            if (fabsf(fmax - fsum / g_world) > 1e-5f * fmaxf(fmax, 1e-20f)) {
                fprintf(stderr, "ranks out of sync at step %d (%g vs mean %g)\n", step + 1, fmax, fsum / g_world);
                return 3;
            }
        }
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        float ms = 0;
        cudaEventElapsedTime(&ms, ev0, ev1);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        double loss = dist_reduce((float)(loss_sum / accum), 0) / g_world;
        if (step > start_step) { step_time_sum += ms / 1e3; step_time_n++; }
        double avg = step_time_n ? step_time_sum / step_time_n : ms / 1e3;
        printf("step %d/%d  loss %.4f  lr_mult %.3f  %.2fs  %.0f tok/s  eta %.1fh\n",
               step + 1, max_steps, loss, mult, ms / 1e3, (total_batch / (ms / 1e3)),
               avg * (max_steps - step - 1) / 3600.0);
        fflush(stdout);
        if (!std::isfinite(loss)) {
            fprintf(stderr, "loss is not finite at step %d; stopping (last checkpoint kept)\n", step + 1);
            return 2;
        }

        if (g_rank == 0 && val_every > 0 && ((step + 1) % val_every == 0 || step + 1 == max_steps)) {
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
                loss_launch(model->losses, model->logits, model->d_targets, B * T, cfg.V, cfg.Vp, 0.f, cfg.softcap, 0);
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
            if (token_bytes && nbytes > 0) printf("  bpb %.4f  (%.3f bytes/token)", nats / (log(2.0) * nbytes), nbytes / ((double)nval * B * T));
            printf("\n");
            fflush(stdout);
        }
        if (ckpt_every > 0 && (step + 1) % ckpt_every == 0) save_ckpt(step + 1);
        if (time_limit > 0 && step + 1 < max_steps) {
            float over = (now_seconds() - t_start >= time_limit) ? 1.f : 0.f;
            if (dist_reduce(over, 1) > 0.f) {
                save_ckpt(step + 1);
                printf("time limit %.0fs reached after step %d; rerun with -y 1 to continue\n", time_limit, step + 1);
                fflush(stdout);
                return 0;
            }
        }
    }
    save_ckpt(max_steps);
    printf("done\n");
    return 0;
}
