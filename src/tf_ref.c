#include "tf_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *tf_alloc(size_t n) {
    float *p = (float *)malloc(n * sizeof(float));
    if (!p) {
        fprintf(stderr, "tf_ref: out of memory (%zu floats)\n", n);
        exit(1);
    }
    return p;
}

static void tf_zero(float *p, size_t n) {
    if (p && n) memset(p, 0, n * sizeof(float));
}

void tf_rmsnorm_forward(float *out, float *rstd, const float *x, int n, int c, float eps) {
    for (int i = 0; i < n; i++) {
        const float *row = x + (size_t)i * c;
        float ss = 0.f;
        for (int j = 0; j < c; j++) ss += row[j] * row[j];
        float s = 1.f / sqrtf(ss / (float)c + eps);
        rstd[i] = s;
        float *orow = out + (size_t)i * c;
        for (int j = 0; j < c; j++) orow[j] = row[j] * s;
    }
}

void tf_rmsnorm_backward(float *dx, const float *dout, const float *x, const float *rstd, int n, int c) {
    float inv_c = 1.f / (float)c;
    for (int i = 0; i < n; i++) {
        const float *xr = x + (size_t)i * c;
        const float *dr = dout + (size_t)i * c;
        float *dxr = dx + (size_t)i * c;
        float s = rstd[i];
        float mean_dy = 0.f;
        for (int j = 0; j < c; j++) mean_dy += dr[j] * (xr[j] * s);
        mean_dy *= inv_c;
        for (int j = 0; j < c; j++) {
            float y = xr[j] * s;
            dxr[j] = s * (dr[j] - y * mean_dy);
        }
    }
}

void tf_layernorm_forward(float *out, float *mean, float *rstd, const float *x,
                          const float *weight, const float *bias, int n, int c, float eps) {
    for (int i = 0; i < n; i++) {
        const float *row = x + (size_t)i * c;
        float m = 0.f;
        for (int j = 0; j < c; j++) m += row[j];
        m /= (float)c;
        float v = 0.f;
        for (int j = 0; j < c; j++) {
            float d = row[j] - m;
            v += d * d;
        }
        v /= (float)c;
        float s = 1.f / sqrtf(v + eps);
        mean[i] = m;
        rstd[i] = s;
        float *orow = out + (size_t)i * c;
        for (int j = 0; j < c; j++) {
            float norm = (row[j] - m) * s;
            orow[j] = norm * weight[j] + bias[j];
        }
    }
}

void tf_layernorm_backward(float *dx, float *dweight, float *dbias,
                           const float *dout, const float *x, const float *weight,
                           const float *mean, const float *rstd, int n, int c) {
    float inv_c = 1.f / (float)c;
    for (int i = 0; i < n; i++) {
        const float *xr = x + (size_t)i * c;
        const float *dr = dout + (size_t)i * c;
        float *dxr = dx + (size_t)i * c;
        float m = mean[i];
        float s = rstd[i];
        float sum_dy = 0.f;
        float sum_dy_norm = 0.f;
        for (int j = 0; j < c; j++) {
            float norm = (xr[j] - m) * s;
            float dy = dr[j] * weight[j];
            sum_dy += dy;
            sum_dy_norm += dy * norm;
            dweight[j] += dr[j] * norm;
            dbias[j] += dr[j];
        }
        sum_dy *= inv_c;
        sum_dy_norm *= inv_c;
        for (int j = 0; j < c; j++) {
            float norm = (xr[j] - m) * s;
            dxr[j] = s * (dr[j] * weight[j] - sum_dy - norm * sum_dy_norm);
        }
    }
}

static void tf_rope_cs(int t, int i, int HS, float base, float *cos_out, float *sin_out) {
    /* channel_range = 0,2,4,... and inv_freq = base ** -(channel_range / HS) */
    float freq = (float)t / powf(base, (2.f * (float)i) / (float)HS);
    *cos_out = cosf(freq);
    *sin_out = sinf(freq);
}

void tf_rope_forward(float *out, const float *x, int B, int T, int NH, int HS, float base) {
    int half = HS / 2;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float *src = x + (((size_t)b * T + t) * NH + h) * HS;
                float *dst = out + (((size_t)b * T + t) * NH + h) * HS;
                for (int i = 0; i < half; i++) {
                    float c, s;
                    tf_rope_cs(t, i, HS, base, &c, &s);
                    float x1 = src[i];
                    float x2 = src[half + i];
                    dst[i] = x1 * c + x2 * s;
                    dst[half + i] = -x1 * s + x2 * c;
                }
            }
        }
    }
}

void tf_rope_backward(float *dx, const float *dout, int B, int T, int NH, int HS, float base) {
    int half = HS / 2;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float *src = dout + (((size_t)b * T + t) * NH + h) * HS;
                float *dst = dx + (((size_t)b * T + t) * NH + h) * HS;
                for (int i = 0; i < half; i++) {
                    float c, s;
                    tf_rope_cs(t, i, HS, base, &c, &s);
                    float y1 = src[i];
                    float y2 = src[half + i];
                    dst[i] = y1 * c - y2 * s;
                    dst[half + i] = y1 * s + y2 * c;
                }
            }
        }
    }
}

void tf_relu2_forward(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i] > 0.f ? x[i] : 0.f;
        out[i] = v * v;
    }
}

void tf_relu2_backward(float *dx, const float *dout, const float *x, int n) {
    for (int i = 0; i < n; i++) {
        dx[i] = x[i] > 0.f ? dout[i] * 2.f * x[i] : 0.f;
    }
}

void tf_gelu_forward(float *out, const float *x, int n) {
    const float k = sqrtf(2.f / (float)M_PI);
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.f + tanhf(k * (xi + cube)));
    }
}

void tf_gelu_backward(float *dx, const float *dout, const float *x, int n) {
    const float k = sqrtf(2.f / (float)M_PI);
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        float cube = 0.044715f * xi * xi * xi;
        float tanh_arg = k * (xi + cube);
        float tanh_out = tanhf(tanh_arg);
        float sech2 = 1.f - tanh_out * tanh_out;
        float local = 0.5f * (1.f + tanh_out) + xi * 0.5f * sech2 * k * (1.f + 3.f * 0.044715f * xi * xi);
        dx[i] = local * dout[i];
    }
}

void tf_softcap_forward(float *out, const float *x, int n, float cap) {
    for (int i = 0; i < n; i++) out[i] = cap * tanhf(x[i] / cap);
}

void tf_softcap_backward(float *dx, const float *dout, const float *x, int n, float cap) {
    for (int i = 0; i < n; i++) {
        float th = tanhf(x[i] / cap);
        dx[i] = dout[i] * (1.f - th * th);
    }
}

void tf_linear_forward(float *y, const float *x, const float *w, const float *bias,
                       int n, int cin, int cout) {
    for (int i = 0; i < n; i++) {
        const float *xr = x + (size_t)i * cin;
        float *yr = y + (size_t)i * cout;
        for (int o = 0; o < cout; o++) {
            const float *wr = w + (size_t)o * cin;
            float acc = bias ? bias[o] : 0.f;
            for (int c = 0; c < cin; c++) acc += xr[c] * wr[c];
            yr[o] = acc;
        }
    }
}

void tf_linear_backward(float *dx, float *dw, float *db,
                        const float *dout, const float *x, const float *w,
                        int n, int cin, int cout) {
    for (int i = 0; i < n; i++) {
        const float *xr = x + (size_t)i * cin;
        const float *dr = dout + (size_t)i * cout;
        float *dxr = dx + (size_t)i * cin;
        for (int c = 0; c < cin; c++) dxr[c] = 0.f;
        for (int o = 0; o < cout; o++) {
            const float *wr = w + (size_t)o * cin;
            float d = dr[o];
            float *dwr = dw + (size_t)o * cin;
            for (int c = 0; c < cin; c++) {
                dxr[c] += d * wr[c];
                dwr[c] += d * xr[c];
            }
            if (db) db[o] += d;
        }
    }
}

void tf_attn_forward(float *out, float *att, const float *q, const float *k, const float *v,
                     int B, int T, int NH, int HS) {
    float scale = 1.f / sqrtf((float)HS);
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            float *att_h = att + ((size_t)b * NH + h) * T * T;
            for (int t = 0; t < T; t++) {
                const float *qr = q + (((size_t)b * T + t) * NH + h) * HS;
                float m = -INFINITY;
                for (int j = 0; j <= t; j++) {
                    const float *kr = k + (((size_t)b * T + j) * NH + h) * HS;
                    float dot = 0.f;
                    for (int d = 0; d < HS; d++) dot += qr[d] * kr[d];
                    float score = dot * scale;
                    att_h[(size_t)t * T + j] = score;
                    if (score > m) m = score;
                }
                for (int j = t + 1; j < T; j++) att_h[(size_t)t * T + j] = 0.f;
                float sum = 0.f;
                for (int j = 0; j <= t; j++) {
                    float e = expf(att_h[(size_t)t * T + j] - m);
                    att_h[(size_t)t * T + j] = e;
                    sum += e;
                }
                float inv = 1.f / sum;
                for (int j = 0; j <= t; j++) att_h[(size_t)t * T + j] *= inv;
                float *orow = out + (((size_t)b * T + t) * NH + h) * HS;
                for (int d = 0; d < HS; d++) orow[d] = 0.f;
                for (int j = 0; j <= t; j++) {
                    const float *vr = v + (((size_t)b * T + j) * NH + h) * HS;
                    float a = att_h[(size_t)t * T + j];
                    for (int d = 0; d < HS; d++) orow[d] += a * vr[d];
                }
            }
        }
    }
}

void tf_attn_backward(float *dq, float *dk, float *dv,
                      const float *dout, const float *q, const float *k, const float *v,
                      const float *att, int B, int T, int NH, int HS) {
    float scale = 1.f / sqrtf((float)HS);
    size_t qn = (size_t)B * T * NH * HS;
    memset(dq, 0, qn * sizeof(float));
    memset(dk, 0, qn * sizeof(float));
    memset(dv, 0, qn * sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            const float *att_h = att + ((size_t)b * NH + h) * T * T;
            for (int t = 0; t < T; t++) {
                const float *drow = dout + (((size_t)b * T + t) * NH + h) * HS;
                /* datt = dout @ v^T, then softmax backward, then scale. */
                float *dscore = tf_alloc(T);
                float sum_datt_att = 0.f;
                for (int j = 0; j <= t; j++) {
                    const float *vr = v + (((size_t)b * T + j) * NH + h) * HS;
                    float datt = 0.f;
                    for (int d = 0; d < HS; d++) datt += drow[d] * vr[d];
                    float a = att_h[(size_t)t * T + j];
                    dscore[j] = datt;
                    sum_datt_att += datt * a;
                    float *dvr = dv + (((size_t)b * T + j) * NH + h) * HS;
                    for (int d = 0; d < HS; d++) dvr[d] += a * drow[d];
                }
                for (int j = 0; j <= t; j++) {
                    float a = att_h[(size_t)t * T + j];
                    dscore[j] = a * (dscore[j] - sum_datt_att) * scale;
                }
                const float *qr = q + (((size_t)b * T + t) * NH + h) * HS;
                float *dqr = dq + (((size_t)b * T + t) * NH + h) * HS;
                for (int j = 0; j <= t; j++) {
                    const float *kr = k + (((size_t)b * T + j) * NH + h) * HS;
                    float *dkr = dk + (((size_t)b * T + j) * NH + h) * HS;
                    float ds = dscore[j];
                    for (int d = 0; d < HS; d++) {
                        dqr[d] += ds * kr[d];
                        dkr[d] += ds * qr[d];
                    }
                }
                free(dscore);
            }
        }
    }
}

float tf_loss(float *losses, float *dlogits, const float *logits, const int *targets,
              int n, int V, int stride, float dloss_scale, float softcap) {
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        const float *row = logits + (size_t)i * stride;
        int tgt = targets[i];
        float m = -INFINITY;
        for (int j = 0; j < V; j++) {
            float z = row[j];
            if (softcap > 0.f) z = softcap * tanhf(z / softcap);
            if (z > m) m = z;
        }
        float sum = 0.f;
        float tgt_logit = 0.f;
        for (int j = 0; j < V; j++) {
            float z = row[j];
            if (softcap > 0.f) z = softcap * tanhf(z / softcap);
            sum += expf(z - m);
            if (j == tgt) tgt_logit = z;
        }
        float loss = (logf(sum) + m) - tgt_logit;
        if (losses) losses[i] = loss;
        total += loss;
        if (dlogits) {
            float *drow = dlogits + (size_t)i * stride;
            float inv = 1.f / sum;
            for (int j = 0; j < stride; j++) drow[j] = 0.f;
            for (int j = 0; j < V; j++) {
                float z = row[j];
                float th = 0.f;
                if (softcap > 0.f) {
                    th = tanhf(z / softcap);
                    z = softcap * th;
                }
                float p = expf(z - m) * inv;
                float dz = (p - (j == tgt ? 1.f : 0.f)) * dloss_scale;
                if (softcap > 0.f) dz *= (1.f - th * th);
                drow[j] = dz;
            }
        }
    }
    return (float)(total / (double)n);
}

void tf_muon_ns5(float *out, const float *g, int rows, int cols) {
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    int transposed = rows > cols;
    int m = transposed ? cols : rows;
    int n = transposed ? rows : cols;
    float *X = tf_alloc((size_t)m * n);
    float *A = tf_alloc((size_t)m * m);
    float *AA = tf_alloc((size_t)m * m);
    float *B = tf_alloc((size_t)m * m);
    float *Y = tf_alloc((size_t)m * n);
    double ss = 0.0;
    for (int i = 0; i < rows * cols; i++) ss += (double)g[i] * (double)g[i];
    float inv = 1.f / ((float)sqrt(ss) + 1e-7f);
    for (int r = 0; r < rows; r++) {
        for (int col = 0; col < cols; col++) {
            float v = g[(size_t)r * cols + col] * inv;
            if (transposed) X[(size_t)col * rows + r] = v;
            else X[(size_t)r * cols + col] = v;
        }
    }
    for (int step = 0; step < 5; step++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                float acc = 0.f;
                for (int k = 0; k < n; k++) acc += X[(size_t)i * n + k] * X[(size_t)j * n + k];
                A[(size_t)i * m + j] = acc;
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                float acc = 0.f;
                for (int k = 0; k < m; k++) acc += A[(size_t)i * m + k] * A[(size_t)k * m + j];
                AA[(size_t)i * m + j] = acc;
            }
        }
        for (int i = 0; i < m * m; i++) B[i] = b * A[i] + c * AA[i];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float acc = 0.f;
                for (int k = 0; k < m; k++) acc += B[(size_t)i * m + k] * X[(size_t)k * n + j];
                Y[(size_t)i * n + j] = a * X[(size_t)i * n + j] + acc;
            }
        }
        memcpy(X, Y, (size_t)m * n * sizeof(float));
    }
    for (int r = 0; r < rows; r++) {
        for (int col = 0; col < cols; col++) {
            if (transposed) out[(size_t)r * cols + col] = X[(size_t)col * rows + r];
            else out[(size_t)r * cols + col] = X[(size_t)r * cols + col];
        }
    }
    free(X); free(A); free(AA); free(B); free(Y);
}

static void tf_norm_forward(const TFConfig *cfg, float *out, float *mean, float *rstd,
                            const float *x, const float *w, const float *b, int n, int c) {
    if (cfg->rmsnorm) tf_rmsnorm_forward(out, rstd, x, n, c, 1e-6f);
    else tf_layernorm_forward(out, mean, rstd, x, w, b, n, c, 1e-5f);
}

static void tf_norm_backward(const TFConfig *cfg, float *dx, float *dw, float *db,
                             const float *dout, const float *x, const float *w,
                             const float *mean, const float *rstd, int n, int c) {
    if (cfg->rmsnorm) tf_rmsnorm_backward(dx, dout, x, rstd, n, c);
    else tf_layernorm_backward(dx, dw, db, dout, x, w, mean, rstd, n, c);
}

static void tf_zero_params(const TFConfig *cfg, TFParams *g) {
    int L = cfg->n_layer, C = cfg->n_embd, Vp = cfg->padded_vocab, T = cfg->max_seq;
    int H = 4 * C;
    tf_zero(g->wte, (size_t)Vp * C);
    tf_zero(g->wpe, g->wpe ? (size_t)T * C : 0);
    tf_zero(g->lm_head, g->lm_head ? (size_t)Vp * C : 0);
    tf_zero(g->lnfw, g->lnfw ? (size_t)C : 0);
    tf_zero(g->lnfb, g->lnfb ? (size_t)C : 0);
    tf_zero(g->ln1w, g->ln1w ? (size_t)L * C : 0);
    tf_zero(g->ln1b, g->ln1b ? (size_t)L * C : 0);
    tf_zero(g->qkvw, (size_t)L * 3 * C * C);
    tf_zero(g->qkvb, g->qkvb ? (size_t)L * 3 * C : 0);
    tf_zero(g->attprojw, (size_t)L * C * C);
    tf_zero(g->attprojb, g->attprojb ? (size_t)L * C : 0);
    tf_zero(g->ln2w, g->ln2w ? (size_t)L * C : 0);
    tf_zero(g->ln2b, g->ln2b ? (size_t)L * C : 0);
    tf_zero(g->fcw, (size_t)L * H * C);
    tf_zero(g->fcb, g->fcb ? (size_t)L * H : 0);
    tf_zero(g->fcprojw, (size_t)L * C * H);
    tf_zero(g->fcprojb, g->fcprojb ? (size_t)L * C : 0);
}

/* Apply RoPE (optional) then QK-norm (optional) and qk_scale. q is (B, T, NH, HS). */
static void tf_qk_prepare(const TFConfig *cfg, float *dst, float *saved_prenorm, float *rstd,
                          const float *src, int B, int T, int NH, int HS) {
    int n = B * T * NH;
    if (cfg->rope) tf_rope_forward(saved_prenorm, src, B, T, NH, HS, cfg->rope_base);
    else memcpy(saved_prenorm, src, (size_t)n * HS * sizeof(float));
    if (cfg->qknorm) {
        tf_rmsnorm_forward(dst, rstd, saved_prenorm, n, HS, 1e-6f);
        float s = cfg->qk_scale;
        int N = n * HS;
        for (int i = 0; i < N; i++) dst[i] *= s;
    } else {
        memcpy(dst, saved_prenorm, (size_t)n * HS * sizeof(float));
    }
}

static void tf_qk_prepare_backward(const TFConfig *cfg, float *dsrc,
                                   const float *ddst, const float *prenorm, const float *rstd,
                                   int B, int T, int NH, int HS) {
    int n = B * T * NH;
    int N = n * HS;
    if (cfg->qknorm) {
        float *dnorm = tf_alloc(N);
        float s = cfg->qk_scale;
        for (int i = 0; i < N; i++) dnorm[i] = ddst[i] * s;
        float *dpre = tf_alloc(N);
        tf_rmsnorm_backward(dpre, dnorm, prenorm, rstd, n, HS);
        if (cfg->rope) tf_rope_backward(dsrc, dpre, B, T, NH, HS, cfg->rope_base);
        else memcpy(dsrc, dpre, (size_t)N * sizeof(float));
        free(dnorm);
        free(dpre);
    } else if (cfg->rope) {
        tf_rope_backward(dsrc, ddst, B, T, NH, HS, cfg->rope_base);
    } else {
        memcpy(dsrc, ddst, (size_t)N * sizeof(float));
    }
}

float tf_loss_and_grads(const TFConfig *cfg, const TFParams *p, TFParams *g,
                        const int *idx, const int *targets, int B, int T) {
    const int L = cfg->n_layer;
    const int C = cfg->n_embd;
    const int NH = cfg->n_head;
    const int HS = C / NH;
    const int Hid = 4 * C;
    const int BT = B * T;
    const int Vp = cfg->padded_vocab;
    const int V = cfg->vocab;
    if (g) tf_zero_params(cfg, g);

    float *embed = tf_alloc((size_t)BT * C);
    float *x0 = tf_alloc((size_t)BT * C);
    float *embed_rstd = tf_alloc(BT);
    float *resid_in = tf_alloc((size_t)L * BT * C);
    float *ln1 = tf_alloc((size_t)L * BT * C);
    float *ln1_mean = tf_alloc((size_t)L * BT);
    float *ln1_rstd = tf_alloc((size_t)L * BT);
    float *qkv = tf_alloc((size_t)BT * 3 * C);
    float *q_raw = tf_alloc((size_t)BT * C);
    float *k_raw = tf_alloc((size_t)BT * C);
    float *v = tf_alloc((size_t)BT * C);
    float *q = tf_alloc((size_t)L * BT * C);
    float *k = tf_alloc((size_t)L * BT * C);
    float *v_saved = tf_alloc((size_t)L * BT * C);
    float *q_pre = tf_alloc((size_t)L * BT * C);
    float *k_pre = tf_alloc((size_t)L * BT * C);
    float *q_rstd = tf_alloc((size_t)L * BT * NH);
    float *k_rstd = tf_alloc((size_t)L * BT * NH);
    float *att = tf_alloc((size_t)L * B * NH * T * T);
    float *atty = tf_alloc((size_t)L * BT * C);
    float *resid2 = tf_alloc((size_t)L * BT * C);
    float *ln2 = tf_alloc((size_t)L * BT * C);
    float *ln2_mean = tf_alloc((size_t)L * BT);
    float *ln2_rstd = tf_alloc((size_t)L * BT);
    float *preact = tf_alloc((size_t)L * BT * Hid);
    float *acted = tf_alloc((size_t)L * BT * Hid);
    float *stream = tf_alloc((size_t)(L + 1) * BT * C); /* stream[0]=x0, stream[l+1]=block output */
    float *lnf = tf_alloc((size_t)BT * C);
    float *lnf_mean = tf_alloc(BT);
    float *lnf_rstd = tf_alloc(BT);
    float *logits = tf_alloc((size_t)BT * Vp);

    for (int i = 0; i < BT; i++) {
        const float *row = p->wte + (size_t)idx[i] * C;
        float *dst = embed + (size_t)i * C;
        int t = i % T;
        for (int c = 0; c < C; c++) {
            float e = row[c];
            if (!cfg->rope) e += p->wpe[(size_t)t * C + c];
            dst[c] = e;
        }
    }
    /* Embedding norm is always parameter-free RMSNorm (nanochat). It does not
       share the final LayerNorm affine of the GPT-2 configuration. */
    if (cfg->embed_norm) tf_rmsnorm_forward(x0, embed_rstd, embed, BT, C, 1e-6f);
    else memcpy(x0, embed, (size_t)BT * C * sizeof(float));
    memcpy(stream, x0, (size_t)BT * C * sizeof(float));

    for (int layer = 0; layer < L; layer++) {
        float *rin = resid_in + (size_t)layer * BT * C;
        memcpy(rin, stream + (size_t)layer * BT * C, (size_t)BT * C * sizeof(float));
        const float *l_ln1w = p->ln1w ? p->ln1w + (size_t)layer * C : NULL;
        const float *l_ln1b = p->ln1b ? p->ln1b + (size_t)layer * C : NULL;
        tf_norm_forward(cfg, ln1 + (size_t)layer * BT * C,
                        ln1_mean + (size_t)layer * BT, ln1_rstd + (size_t)layer * BT,
                        rin, l_ln1w, l_ln1b, BT, C);
        tf_linear_forward(qkv, ln1 + (size_t)layer * BT * C,
                          p->qkvw + (size_t)layer * 3 * C * C,
                          p->qkvb ? p->qkvb + (size_t)layer * 3 * C : NULL,
                          BT, C, 3 * C);
        for (int i = 0; i < BT; i++) {
            memcpy(q_raw + (size_t)i * C, qkv + (size_t)i * 3 * C, (size_t)C * sizeof(float));
            memcpy(k_raw + (size_t)i * C, qkv + (size_t)i * 3 * C + C, (size_t)C * sizeof(float));
            memcpy(v + (size_t)i * C, qkv + (size_t)i * 3 * C + 2 * C, (size_t)C * sizeof(float));
        }
        memcpy(v_saved + (size_t)layer * BT * C, v, (size_t)BT * C * sizeof(float));
        tf_qk_prepare(cfg, q + (size_t)layer * BT * C, q_pre + (size_t)layer * BT * C,
                      q_rstd + (size_t)layer * BT * NH, q_raw, B, T, NH, HS);
        tf_qk_prepare(cfg, k + (size_t)layer * BT * C, k_pre + (size_t)layer * BT * C,
                      k_rstd + (size_t)layer * BT * NH, k_raw, B, T, NH, HS);
        tf_attn_forward(atty + (size_t)layer * BT * C, att + (size_t)layer * B * NH * T * T,
                        q + (size_t)layer * BT * C, k + (size_t)layer * BT * C, v,
                        B, T, NH, HS);
        float *r2 = resid2 + (size_t)layer * BT * C;
        tf_linear_forward(r2, atty + (size_t)layer * BT * C,
                          p->attprojw + (size_t)layer * C * C,
                          p->attprojb ? p->attprojb + (size_t)layer * C : NULL,
                          BT, C, C);
        for (int i = 0; i < BT * C; i++) r2[i] += rin[i];

        const float *l_ln2w = p->ln2w ? p->ln2w + (size_t)layer * C : NULL;
        const float *l_ln2b = p->ln2b ? p->ln2b + (size_t)layer * C : NULL;
        tf_norm_forward(cfg, ln2 + (size_t)layer * BT * C,
                        ln2_mean + (size_t)layer * BT, ln2_rstd + (size_t)layer * BT,
                        r2, l_ln2w, l_ln2b, BT, C);
        tf_linear_forward(preact + (size_t)layer * BT * Hid, ln2 + (size_t)layer * BT * C,
                          p->fcw + (size_t)layer * Hid * C,
                          p->fcb ? p->fcb + (size_t)layer * Hid : NULL,
                          BT, C, Hid);
        if (cfg->relu2) tf_relu2_forward(acted + (size_t)layer * BT * Hid, preact + (size_t)layer * BT * Hid, BT * Hid);
        else tf_gelu_forward(acted + (size_t)layer * BT * Hid, preact + (size_t)layer * BT * Hid, BT * Hid);
        float *nxt = stream + (size_t)(layer + 1) * BT * C;
        tf_linear_forward(nxt, acted + (size_t)layer * BT * Hid,
                          p->fcprojw + (size_t)layer * C * Hid,
                          p->fcprojb ? p->fcprojb + (size_t)layer * C : NULL,
                          BT, Hid, C);
        for (int i = 0; i < BT * C; i++) nxt[i] += r2[i];
    }

    float *final_in = stream + (size_t)L * BT * C;
    tf_norm_forward(cfg, lnf, lnf_mean, lnf_rstd, final_in, p->lnfw, p->lnfb, BT, C);
    const float *head = cfg->untie ? p->lm_head : p->wte;
    tf_linear_forward(logits, lnf, head, NULL, BT, C, Vp);
    float *dlogits = g ? tf_alloc((size_t)BT * Vp) : NULL;
    float loss = tf_loss(NULL, dlogits, logits, targets, BT, V, Vp, g ? 1.f / (float)BT : 0.f, cfg->softcap);

    if (g) {
        float *dlnf = tf_alloc((size_t)BT * C);
        float *dhead = cfg->untie ? g->lm_head : g->wte;
        tf_linear_backward(dlnf, dhead, NULL, dlogits, lnf, head, BT, C, Vp);
        float *dfinal = tf_alloc((size_t)BT * C);
        tf_norm_backward(cfg, dfinal, g->lnfw, g->lnfb, dlnf, final_in, p->lnfw, lnf_mean, lnf_rstd, BT, C);

        float *dx = dfinal;
        float *dq = tf_alloc((size_t)BT * C);
        float *dk = tf_alloc((size_t)BT * C);
        float *dv = tf_alloc((size_t)BT * C);
        float *dqkv = tf_alloc((size_t)BT * 3 * C);
        float *dln = tf_alloc((size_t)BT * C);
        float *dpre = tf_alloc((size_t)BT * Hid);
        float *dact = tf_alloc((size_t)BT * Hid);
        for (int layer = L - 1; layer >= 0; layer--) {
            float *r2 = resid2 + (size_t)layer * BT * C;
            float *rin = resid_in + (size_t)layer * BT * C;
            /* MLP residual: nxt = r2 + proj(act). dx arrives as d(nxt). */
            tf_linear_backward(dact, g->fcprojw + (size_t)layer * C * Hid,
                               g->fcprojb ? g->fcprojb + (size_t)layer * C : NULL,
                               dx, acted + (size_t)layer * BT * Hid,
                               p->fcprojw + (size_t)layer * C * Hid, BT, Hid, C);
            if (cfg->relu2) tf_relu2_backward(dpre, dact, preact + (size_t)layer * BT * Hid, BT * Hid);
            else tf_gelu_backward(dpre, dact, preact + (size_t)layer * BT * Hid, BT * Hid);
            tf_linear_backward(dln, g->fcw + (size_t)layer * Hid * C,
                               g->fcb ? g->fcb + (size_t)layer * Hid : NULL,
                               dpre, ln2 + (size_t)layer * BT * C,
                               p->fcw + (size_t)layer * Hid * C, BT, C, Hid);
            float *dr2 = tf_alloc((size_t)BT * C);
            tf_norm_backward(cfg, dr2,
                             g->ln2w ? g->ln2w + (size_t)layer * C : NULL,
                             g->ln2b ? g->ln2b + (size_t)layer * C : NULL,
                             dln, r2,
                             p->ln2w ? p->ln2w + (size_t)layer * C : NULL,
                             ln2_mean + (size_t)layer * BT, ln2_rstd + (size_t)layer * BT, BT, C);
            for (int i = 0; i < BT * C; i++) dr2[i] += dx[i];

            float *datty = tf_alloc((size_t)BT * C);
            tf_linear_backward(datty, g->attprojw + (size_t)layer * C * C,
                               g->attprojb ? g->attprojb + (size_t)layer * C : NULL,
                               dr2, atty + (size_t)layer * BT * C,
                               p->attprojw + (size_t)layer * C * C, BT, C, C);
            tf_attn_backward(dq, dk, dv, datty,
                             q + (size_t)layer * BT * C, k + (size_t)layer * BT * C,
                             v_saved + (size_t)layer * BT * C,
                             att + (size_t)layer * B * NH * T * T, B, T, NH, HS);
            float *dq_raw = tf_alloc((size_t)BT * C);
            float *dk_raw = tf_alloc((size_t)BT * C);
            tf_qk_prepare_backward(cfg, dq_raw, dq, q_pre + (size_t)layer * BT * C,
                                   q_rstd + (size_t)layer * BT * NH, B, T, NH, HS);
            tf_qk_prepare_backward(cfg, dk_raw, dk, k_pre + (size_t)layer * BT * C,
                                   k_rstd + (size_t)layer * BT * NH, B, T, NH, HS);
            for (int i = 0; i < BT; i++) {
                memcpy(dqkv + (size_t)i * 3 * C, dq_raw + (size_t)i * C, (size_t)C * sizeof(float));
                memcpy(dqkv + (size_t)i * 3 * C + C, dk_raw + (size_t)i * C, (size_t)C * sizeof(float));
                memcpy(dqkv + (size_t)i * 3 * C + 2 * C, dv + (size_t)i * C, (size_t)C * sizeof(float));
            }
            tf_linear_backward(dln, g->qkvw + (size_t)layer * 3 * C * C,
                               g->qkvb ? g->qkvb + (size_t)layer * 3 * C : NULL,
                               dqkv, ln1 + (size_t)layer * BT * C,
                               p->qkvw + (size_t)layer * 3 * C * C, BT, C, 3 * C);
            float *drin = tf_alloc((size_t)BT * C);
            tf_norm_backward(cfg, drin,
                             g->ln1w ? g->ln1w + (size_t)layer * C : NULL,
                             g->ln1b ? g->ln1b + (size_t)layer * C : NULL,
                             dln, rin,
                             p->ln1w ? p->ln1w + (size_t)layer * C : NULL,
                             ln1_mean + (size_t)layer * BT, ln1_rstd + (size_t)layer * BT, BT, C);
            for (int i = 0; i < BT * C; i++) drin[i] += dr2[i];
            memcpy(dx, drin, (size_t)BT * C * sizeof(float));
            free(dr2); free(datty); free(dq_raw); free(dk_raw); free(drin);
        }

        /* dx is now d(x0), the gradient at the embedding-norm output (or the raw embed). */
        float *dembed = tf_alloc((size_t)BT * C);
        if (cfg->embed_norm) {
            tf_rmsnorm_backward(dembed, dx, embed, embed_rstd, BT, C);
        } else {
            memcpy(dembed, dx, (size_t)BT * C * sizeof(float));
        }
        for (int i = 0; i < BT; i++) {
            int tok = idx[i];
            int t = i % T;
            float *dw = g->wte + (size_t)tok * C;
            const float *de = dembed + (size_t)i * C;
            for (int c = 0; c < C; c++) {
                dw[c] += de[c];
                if (!cfg->rope) g->wpe[(size_t)t * C + c] += de[c];
            }
        }
        free(dembed); free(dlnf); free(dfinal);
        free(dq); free(dk); free(dv); free(dqkv); free(dln); free(dpre); free(dact);
        free(dlogits);
    }

    free(embed); free(x0); free(embed_rstd);
    free(resid_in); free(ln1); free(ln1_mean); free(ln1_rstd);
    free(qkv); free(q_raw); free(k_raw); free(v);
    free(q); free(k); free(v_saved); free(q_pre); free(k_pre); free(q_rstd); free(k_rstd);
    free(att); free(atty); free(resid2); free(ln2); free(ln2_mean); free(ln2_rstd);
    free(preact); free(acted); free(stream); free(lnf); free(lnf_mean); free(lnf_rstd); free(logits);
    return loss;
}
