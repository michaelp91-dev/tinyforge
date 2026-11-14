/*
CPU reference for TinyForge layers and the full decoder.
The CUDA trainer implements the same math. Python tests check these
functions against a PyTorch autograd reference.
*/
#ifndef TF_REF_H
#define TF_REF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int vocab;         /* real vocab V (loss is over these rows) */
    int padded_vocab;  /* Vp, row stride of wte / lm_head / logits */
    int max_seq;
    int n_layer;
    int n_head;
    int n_embd;        /* C, must be divisible by n_head */
    int rope;          /* 1 = rotary, no wpe */
    int rmsnorm;       /* 1 = parameter-free RMSNorm, 0 = LayerNorm with affine */
    int qknorm;        /* 1 = RMSNorm on Q and K over the head dim */
    int relu2;         /* 1 = ReLU^2 MLP, 0 = tanh-approximate GELU */
    int untie;         /* 1 = separate lm_head, 0 = tied to wte */
    int bias;          /* 1 = linear biases */
    int embed_norm;    /* 1 = norm the token embedding before the blocks */
    float softcap;     /* 0 disables; else cap * tanh(x / cap) */
    float qk_scale;    /* applied after QK-norm (nanochat uses 1.2) */
    float rope_base;   /* nanochat uses 100000 */
} TFConfig;

/* Every pointer is row-major and owned by the caller.
   Disabled tensors are NULL (wpe when rope=1, lm_head when untie=0,
   norm weights when rmsnorm=1, biases when bias=0). */
typedef struct {
    float *wte;       /* (Vp, C) */
    float *wpe;       /* (max_seq, C) */
    float *lm_head;   /* (Vp, C) */
    float *lnfw;      /* (C,) */
    float *lnfb;      /* (C,) */
    float *ln1w;      /* (L, C) */
    float *ln1b;      /* (L, C) */
    float *qkvw;      /* (L, 3C, C) */
    float *qkvb;      /* (L, 3C) */
    float *attprojw;  /* (L, C, C) */
    float *attprojb;  /* (L, C) */
    float *ln2w;      /* (L, C) */
    float *ln2b;      /* (L, C) */
    float *fcw;       /* (L, 4C, C) */
    float *fcb;       /* (L, 4C) */
    float *fcprojw;   /* (L, C, 4C) */
    float *fcprojb;   /* (L, C) */
} TFParams;

/* Parameter-free RMSNorm. eps is 1e-6 to match torch.nn.functional.rms_norm.
   rstd has one value per row. out and x are (n, c). */
void tf_rmsnorm_forward(float *out, float *rstd, const float *x, int n, int c, float eps);
void tf_rmsnorm_backward(float *dx, const float *dout, const float *x, const float *rstd, int n, int c);

/* LayerNorm with affine weight and bias. eps is 1e-5 (GPT-2 / llm.c). */
void tf_layernorm_forward(float *out, float *mean, float *rstd, const float *x,
                          const float *weight, const float *bias, int n, int c, float eps);
void tf_layernorm_backward(float *dx, float *dweight, float *dbias,
                           const float *dout, const float *x, const float *weight,
                           const float *mean, const float *rstd, int n, int c);

/* Rotate-half RoPE, nanochat sign convention.
   x is (B, T, NH, HS). base is the theta base (100000). */
void tf_rope_forward(float *out, const float *x, int B, int T, int NH, int HS, float base);
void tf_rope_backward(float *dx, const float *dout, int B, int T, int NH, int HS, float base);

void tf_relu2_forward(float *out, const float *x, int n);
void tf_relu2_backward(float *dx, const float *dout, const float *x, int n);

/* llm.c / GPT-2 tanh approximation, not the erf GELU. */
void tf_gelu_forward(float *out, const float *x, int n);
void tf_gelu_backward(float *dx, const float *dout, const float *x, int n);

void tf_softcap_forward(float *out, const float *x, int n, float cap);
void tf_softcap_backward(float *dx, const float *dout, const float *x, int n, float cap);

/* y = x @ W^T [+ bias]. x (n, cin), W (cout, cin), y (n, cout).
   Backward overwrites dx and accumulates dw / db. Caller zeros dw / db. */
void tf_linear_forward(float *y, const float *x, const float *w, const float *bias,
                       int n, int cin, int cout);
void tf_linear_backward(float *dx, float *dw, float *db,
                        const float *dout, const float *x, const float *w,
                        int n, int cin, int cout);

/* Causal attention. q/k/v/out are (B, T, NH, HS). att is (B, NH, T, T).
   Scale 1/sqrt(HS) is applied inside, matching llm.c and flash-attention. */
void tf_attn_forward(float *out, float *att, const float *q, const float *k, const float *v,
                     int B, int T, int NH, int HS);
void tf_attn_backward(float *dq, float *dk, float *dv,
                      const float *dout, const float *q, const float *k, const float *v,
                      const float *att, int B, int T, int NH, int HS);

/* Mean cross-entropy over n rows. logits has row stride `stride` (>= V);
   only the first V columns participate. dloss_scale multiplies every
   dlogits (pass 1/n for a mean loss). softcap 0 disables the cap.
   dlogits may be NULL. losses may be NULL. Returns the mean loss. */
float tf_loss(float *losses, float *dlogits, const float *logits, const int *targets,
              int n, int V, int stride, float dloss_scale, float softcap);

/* Classic Muon Newton-Schulz quintic (5 steps), coefficients (3.4445, -4.7750, 2.0315).
   Writes the orthogonalized matrix, same shape as g (rows, cols). Does not
   apply the aspect-ratio scale; the trainer multiplies by sqrt(max(1, rows/cols)). */
void tf_muon_ns5(float *out, const float *g, int rows, int cols);

/* Forward + backward of the full decoder. Zeros every non-NULL grad tensor,
   then accumulates. Returns the mean token loss. idx and targets are length B*T. */
float tf_loss_and_grads(const TFConfig *cfg, const TFParams *p, TFParams *g,
                        const int *idx, const int *targets, int B, int T);

#ifdef __cplusplus
}
#endif

#endif
