#pragma once

#include <cstdint>

// Reference math used by the CPU engine and the gradient tests.
namespace tf {

void gemm_nt(const float* X, const float* W, float* Y, int M, int N, int K);  // Y = X @ W^T
void gemm_nn(const float* A, const float* B, float* C, int M, int N, int K);  // C = A @ B
void gemm_tn(const float* A, const float* B, float* C, int M, int N, int K);  // C = A^T @ B, A is (K,M)

void rmsnorm_fwd(const float* x, float* y, float* rstd, int rows, int cols, float eps);
void rmsnorm_bwd(const float* x, const float* rstd, const float* dy, float* dx, int rows, int cols);

// x is (B,H,T,D). cos/sin are (T, D/2). In place.
void rope_fwd(float* x, const float* cos, const float* sin, int B, int H, int T, int D);
void rope_bwd(float* dx, const float* cos, const float* sin, int B, int H, int T, int D);

void relu2_fwd(const float* x, float* y, int n);
void relu2_bwd(const float* x, const float* dy, float* dx, int n);

// Causal attention. window < 0 means full causal; otherwise attend at most `window` tokens back.
// q (B,H,T,D), k/v (B,Hk,T,D), out (B,H,T,D), probs (B,H,T,T) required.
void attn_fwd(const float* q, const float* k, const float* v, float* out, float* probs, int B, int H, int Hk, int T,
              int D, int window);
void attn_bwd(const float* q, const float* k, const float* v, const float* probs, const float* dout, float* dq, float* dk,
              float* dv, int B, int H, int Hk, int T, int D, int window);

// Mean cross-entropy over rows with target >= 0, after logit softcap.
// logits/dlogits are (rows, vocab). Returns the mean loss (0 if nothing is valid).
float softcap_ce(const float* logits, const int32_t* target, float* dlogits, int rows, int vocab, float softcap, bool backward);

}  // namespace tf
