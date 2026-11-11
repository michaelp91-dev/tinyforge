#include "tf_ops.hpp"
#include "tf_util.hpp"

#include <cmath>
#include <cstring>

namespace tf {
namespace {

inline int idx4(int n1, int n2, int n3, int i, int j, int k, int l) {
    return ((i * n1 + j) * n2 + k) * n3 + l;
}

}  // namespace

void gemm_nt(const float* X, const float* W, float* Y, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const float* x = X + static_cast<int64_t>(m) * K;
            const float* w = W + static_cast<int64_t>(n) * K;
            float s = 0.f;
            int k = 0;
            for (; k + 3 < K; k += 4) s += x[k] * w[k] + x[k + 1] * w[k + 1] + x[k + 2] * w[k + 2] + x[k + 3] * w[k + 3];
            for (; k < K; ++k) s += x[k] * w[k];
            Y[static_cast<int64_t>(m) * N + n] = s;
        }
    }
}

void gemm_nn(const float* A, const float* B, float* C, int M, int N, int K) {
    std::memset(C, 0, sizeof(float) * static_cast<size_t>(M) * N);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            float a = A[static_cast<int64_t>(m) * K + k];
            const float* b = B + static_cast<int64_t>(k) * N;
            float* c = C + static_cast<int64_t>(m) * N;
            for (int n = 0; n < N; ++n) c[n] += a * b[n];
        }
    }
}

void gemm_tn(const float* A, const float* B, float* C, int M, int N, int K) {
    // A is (K, M), B is (K, N), C is (M, N) = A^T @ B.
    std::memset(C, 0, sizeof(float) * static_cast<size_t>(M) * N);
    for (int k = 0; k < K; ++k) {
        const float* a = A + static_cast<int64_t>(k) * M;
        const float* b = B + static_cast<int64_t>(k) * N;
        for (int m = 0; m < M; ++m) {
            float av = a[m];
            float* c = C + static_cast<int64_t>(m) * N;
            for (int n = 0; n < N; ++n) c[n] += av * b[n];
        }
    }
}

void rmsnorm_fwd(const float* x, float* y, float* rstd, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + static_cast<int64_t>(r) * cols;
        float* yr = y + static_cast<int64_t>(r) * cols;
        float ss = 0.f;
        for (int c = 0; c < cols; ++c) ss += xr[c] * xr[c];
        float inv = 1.f / std::sqrt(ss / static_cast<float>(cols) + eps);
        rstd[r] = inv;
        for (int c = 0; c < cols; ++c) yr[c] = xr[c] * inv;
    }
}

void rmsnorm_bwd(const float* x, const float* rstd, const float* dy, float* dx, int rows, int cols) {
    float inv_n = 1.f / static_cast<float>(cols);
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + static_cast<int64_t>(r) * cols;
        const float* dyr = dy + static_cast<int64_t>(r) * cols;
        float* dxr = dx + static_cast<int64_t>(r) * cols;
        float inv = rstd[r];
        float inv3 = inv * inv * inv;
        float dot = 0.f;
        for (int c = 0; c < cols; ++c) dot += dyr[c] * xr[c];
        float scale = dot * inv3 * inv_n;
        for (int c = 0; c < cols; ++c) dxr[c] = dyr[c] * inv - xr[c] * scale;
    }
}

void rope_fwd(float* x, const float* cos, const float* sin, int B, int H, int T, int D) {
    int half = D / 2;
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                float* row = x + idx4(H, T, D, b, h, t, 0);
                const float* c = cos + static_cast<int64_t>(t) * half;
                const float* s = sin + static_cast<int64_t>(t) * half;
                for (int i = 0; i < half; ++i) {
                    float x1 = row[i];
                    float x2 = row[half + i];
                    row[i] = x1 * c[i] + x2 * s[i];
                    row[half + i] = -x1 * s[i] + x2 * c[i];
                }
            }
        }
    }
}

void rope_bwd(float* dx, const float* cos, const float* sin, int B, int H, int T, int D) {
    int half = D / 2;
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                float* row = dx + idx4(H, T, D, b, h, t, 0);
                const float* c = cos + static_cast<int64_t>(t) * half;
                const float* s = sin + static_cast<int64_t>(t) * half;
                for (int i = 0; i < half; ++i) {
                    float y1 = row[i];
                    float y2 = row[half + i];
                    row[i] = y1 * c[i] - y2 * s[i];
                    row[half + i] = y1 * s[i] + y2 * c[i];
                }
            }
        }
    }
}

void relu2_fwd(const float* x, float* y, int n) {
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        y[i] = v > 0.f ? v * v : 0.f;
    }
}

void relu2_bwd(const float* x, const float* dy, float* dx, int n) {
    for (int i = 0; i < n; ++i) dx[i] = x[i] > 0.f ? dy[i] * (2.f * x[i]) : 0.f;
}

void attn_fwd(const float* q, const float* k, const float* v, float* out, float* probs, int B, int H, int Hk, int T,
              int D, int window) {
    int rep = H / Hk;
    float scale = 1.f / std::sqrt(static_cast<float>(D));
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            int hk = h / rep;
            for (int i = 0; i < T; ++i) {
                const float* qi = q + idx4(H, T, D, b, h, i, 0);
                float* pi = probs + idx4(H, T, T, b, h, i, 0);
                float m = -1e30f;
                for (int j = 0; j < T; ++j) {
                    bool allow = j <= i && (window < 0 || i - j <= window);
                    float s = -1e30f;
                    if (allow) {
                        const float* kj = k + idx4(Hk, T, D, b, hk, j, 0);
                        s = 0.f;
                        for (int d = 0; d < D; ++d) s += qi[d] * kj[d];
                        s *= scale;
                    }
                    pi[j] = s;
                    if (s > m) m = s;
                }
                float sum = 0.f;
                for (int j = 0; j < T; ++j) {
                    float e = pi[j] > -1e20f ? std::exp(pi[j] - m) : 0.f;
                    pi[j] = e;
                    sum += e;
                }
                float inv = sum > 0.f ? 1.f / sum : 0.f;
                for (int j = 0; j < T; ++j) pi[j] *= inv;
                float* oi = out + idx4(H, T, D, b, h, i, 0);
                for (int d = 0; d < D; ++d) oi[d] = 0.f;
                for (int j = 0; j < T; ++j) {
                    const float* vj = v + idx4(Hk, T, D, b, hk, j, 0);
                    float p = pi[j];
                    for (int d = 0; d < D; ++d) oi[d] += p * vj[d];
                }
            }
        }
    }
}

void attn_bwd(const float* q, const float* k, const float* v, const float* probs, const float* dout, float* dq, float* dk,
              float* dv, int B, int H, int Hk, int T, int D, int window) {
    int rep = H / Hk;
    float scale = 1.f / std::sqrt(static_cast<float>(D));
    int64_t qn = static_cast<int64_t>(B) * H * T * D;
    int64_t kn = static_cast<int64_t>(B) * Hk * T * D;
    std::memset(dq, 0, sizeof(float) * qn);
    std::memset(dk, 0, sizeof(float) * kn);
    std::memset(dv, 0, sizeof(float) * kn);
    std::vector<float> ds(static_cast<size_t>(T));
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            int hk = h / rep;
            for (int i = 0; i < T; ++i) {
                const float* pi = probs + idx4(H, T, T, b, h, i, 0);
                const float* doi = dout + idx4(H, T, D, b, h, i, 0);
                float row = 0.f;
                for (int j = 0; j < T; ++j) {
                    bool allow = j <= i && (window < 0 || i - j <= window);
                    float dp = 0.f;
                    if (allow) {
                        const float* vj = v + idx4(Hk, T, D, b, hk, j, 0);
                        for (int d = 0; d < D; ++d) dp += doi[d] * vj[d];
                    }
                    ds[j] = dp;
                    row += dp * pi[j];
                }
                for (int j = 0; j < T; ++j) ds[j] = (ds[j] - row) * pi[j] * scale;
                float* dqi = dq + idx4(H, T, D, b, h, i, 0);
                const float* qi = q + idx4(H, T, D, b, h, i, 0);
                for (int j = 0; j < T; ++j) {
                    bool allow = j <= i && (window < 0 || i - j <= window);
                    if (!allow) continue;
                    const float* kj = k + idx4(Hk, T, D, b, hk, j, 0);
                    float* dkj = dk + idx4(Hk, T, D, b, hk, j, 0);
                    float* dvj = dv + idx4(Hk, T, D, b, hk, j, 0);
                    float p = pi[j];
                    float s = ds[j];
                    for (int d = 0; d < D; ++d) {
                        dqi[d] += s * kj[d];
                        dkj[d] += s * qi[d];
                        dvj[d] += p * doi[d];
                    }
                }
            }
        }
    }
}

float softcap_ce(const float* logits, const int32_t* target, float* dlogits, int rows, int vocab, float softcap, bool backward) {
    int valid = 0;
    double loss_sum = 0.0;
    if (backward && dlogits) std::memset(dlogits, 0, sizeof(float) * static_cast<size_t>(rows) * vocab);
    std::vector<float> z(static_cast<size_t>(vocab));
    for (int r = 0; r < rows; ++r) {
        int tgt = target[r];
        if (tgt < 0) continue;
        check(tgt < vocab, "target id out of range");
        const float* row = logits + static_cast<int64_t>(r) * vocab;
        float m = -1e30f;
        for (int v = 0; v < vocab; ++v) {
            float th = std::tanh(row[v] / softcap);
            z[v] = softcap * th;
            if (z[v] > m) m = z[v];
        }
        float sum = 0.f;
        for (int v = 0; v < vocab; ++v) {
            z[v] = std::exp(z[v] - m);
            sum += z[v];
        }
        float p_t = z[tgt] / sum;
        loss_sum += -std::log(std::max(p_t, 1e-20f));
        ++valid;
        if (backward && dlogits) {
            float* drow = dlogits + static_cast<int64_t>(r) * vocab;
            for (int v = 0; v < vocab; ++v) {
                float p = z[v] / sum;
                float dz = p - (v == tgt ? 1.f : 0.f);
                float th = std::tanh(row[v] / softcap);
                float du = dz * (1.f - th * th);
                drow[v] = du;  // scaled by 1/valid below
            }
        }
    }
    if (valid == 0) return 0.f;
    if (backward && dlogits) {
        float inv = 1.f / static_cast<float>(valid);
        for (int r = 0; r < rows; ++r) {
            if (target[r] < 0) continue;
            float* drow = dlogits + static_cast<int64_t>(r) * vocab;
            for (int v = 0; v < vocab; ++v) drow[v] *= inv;
        }
    }
    return static_cast<float>(loss_sum / valid);
}

}  // namespace tf
