// Elementwise and normalization kernels for the TinyForge trainer.
// Formulas match src/tf_ref.c. Included after the llm.c CUDA headers.

static __device__ float tf_warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

static __device__ float tf_block_sum(float v, float* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = tf_warp_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    int nwarps = blockDim.x >> 5;
    v = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.f;
    if (warp == 0) v = tf_warp_sum(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    return smem[0];
}

__global__ void tf_rmsnorm_forward_kernel(floatX* out, float* rstd, const floatX* x,
                                          int n, int C, float eps) {
    int row = blockIdx.x;
    if (row >= n) return;
    extern __shared__ float smem[];
    const floatX* xr = x + (size_t)row * C;
    float ss = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = (float)xr[i];
        ss += v * v;
    }
    ss = tf_block_sum(ss, smem);
    float s = rsqrtf(ss / (float)C + eps);
    if (threadIdx.x == 0) rstd[row] = s;
    floatX* yr = out + (size_t)row * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        yr[i] = (floatX)((float)xr[i] * s);
    }
}

__global__ void tf_rmsnorm_backward_kernel(floatX* dx, const floatX* dout, const floatX* x,
                                           const float* rstd, int n, int C) {
    int row = blockIdx.x;
    if (row >= n) return;
    extern __shared__ float smem[];
    const floatX* xr = x + (size_t)row * C;
    const floatX* dr = dout + (size_t)row * C;
    float s = rstd[row];
    float mean_dy = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        mean_dy += (float)dr[i] * ((float)xr[i] * s);
    }
    mean_dy = tf_block_sum(mean_dy, smem) / (float)C;
    floatX* dxr = dx + (size_t)row * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float y = (float)xr[i] * s;
        dxr[i] = (floatX)(s * ((float)dr[i] - y * mean_dy));
    }
}

__global__ void tf_residual_kernel(floatX* out, const floatX* a, const floatX* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (floatX)((float)a[i] + (float)b[i]);
}

__global__ void tf_relu2_forward_kernel(floatX* out, const floatX* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = (float)x[i];
        v = v > 0.f ? v : 0.f;
        out[i] = (floatX)(v * v);
    }
}

__global__ void tf_relu2_backward_kernel(floatX* dx, const floatX* dout, const floatX* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = (float)x[i];
        dx[i] = (floatX)(v > 0.f ? (float)dout[i] * 2.f * v : 0.f);
    }
}

__global__ void tf_embed_forward_kernel(floatX* out, const int* tokens, const floatX* wte,
                                        const floatX* wpe, int B, int T, int C, int use_wpe) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * T * C) return;
    int bt = i / C;
    int c = i - bt * C;
    int tok = tokens[bt];
    int t = bt % T;
    float v = (float)wte[(size_t)tok * C + c];
    if (use_wpe) v += (float)wpe[(size_t)t * C + c];
    out[i] = (floatX)v;
}

__global__ void tf_embed_backward_kernel(float* dwte, float* dwpe, const floatX* dout,
                                         const int* tokens, int BT, int T, int C, int use_wpe) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BT * C) return;
    int bt = i / C;
    int c = i - bt * C;
    int tok = tokens[bt];
    atomicAdd(dwte + (size_t)tok * C + c, (float)dout[i]);
    if (use_wpe) atomicAdd(dwpe + (size_t)(bt % T) * C + c, (float)dout[i]);
}

__global__ void tf_add_fp32_kernel(floatX* dst, const float* src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (floatX)((float)dst[i] + src[i]);
}

__global__ void tf_cast_to_half_kernel(floatX* dst, const float* src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (floatX)src[i];
}

__global__ void tf_zero_float_kernel(float* x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.f;
}

// Mean cross-entropy. When write_grad is set, logits is overwritten with
// d(loss)/d(pre-softcap logits), scaled by dloss. Only the first V columns
// of each Vp-wide row enter the softmax.
__global__ void tf_loss_kernel(float* losses, floatX* logits, const int* targets,
                               int n, int V, int Vp, float dloss, float softcap, int write_grad) {
    int row = blockIdx.x;
    if (row >= n) return;
    extern __shared__ float smem[];
    floatX* z = logits + (size_t)row * Vp;
    float m = -1e30f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        float v = (float)z[j];
        if (softcap > 0.f) v = softcap * tanhf(v / softcap);
        m = fmaxf(m, v);
    }
    // max-reduce
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        m = fmaxf(m, __shfl_down_sync(0xffffffff, m, offset));
    }
    if (lane == 0) smem[warp] = m;
    __syncthreads();
    m = (threadIdx.x < (blockDim.x >> 5)) ? smem[threadIdx.x] : -1e30f;
    if (warp == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            m = fmaxf(m, __shfl_down_sync(0xffffffff, m, offset));
        }
        if (lane == 0) smem[0] = m;
    }
    __syncthreads();
    m = smem[0];

    float sum = 0.f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        float v = (float)z[j];
        if (softcap > 0.f) v = softcap * tanhf(v / softcap);
        sum += expf(v - m);
    }
    sum = tf_block_sum(sum, smem);
    if (threadIdx.x == 0) {
        int tgt = targets[row];
        float vt = (float)z[tgt];
        if (softcap > 0.f) vt = softcap * tanhf(vt / softcap);
        losses[row] = (logf(sum) + m) - vt;
    }
    if (!write_grad) return;
    float inv = 1.f / sum;
    int tgt = targets[row];
    for (int j = threadIdx.x; j < Vp; j += blockDim.x) {
        float dz = 0.f;
        if (j < V) {
            float raw = (float)z[j];
            float th = 0.f;
            float v = raw;
            if (softcap > 0.f) {
                th = tanhf(raw / softcap);
                v = softcap * th;
            }
            float p = expf(v - m) * inv;
            dz = (p - (j == tgt ? 1.f : 0.f)) * dloss;
            if (softcap > 0.f) dz *= (1.f - th * th);
        }
        z[j] = (floatX)dz;
    }
}

// In-place RoPE + optional QK-norm on the Q and K slices of a packed
// (B, T, 3, NH, HS) tensor. One block per (b, t, head); blockDim == HS.
__global__ void tf_qk_forward_kernel(floatX* qkv, float* q_rstd, float* k_rstd,
                                     int B, int T, int NH, int HS, int C,
                                     float base, int do_rope, int do_qknorm, float qk_scale) {
    int idx = blockIdx.x;
    int nh = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (NH * T);
    if (b >= B) return;
    extern __shared__ float sh[];
    int half = HS / 2;
    for (int which = 0; which < 2; which++) {
        floatX* vec = qkv + ((((size_t)b * T + t) * 3 + which) * C + (size_t)nh * HS);
        float* rstd = (which == 0 ? q_rstd : k_rstd) + ((size_t)b * T + t) * NH + nh;
        sh[threadIdx.x] = (float)vec[threadIdx.x];
        __syncthreads();
        if (do_rope && threadIdx.x < half) {
            float freq = (float)t / powf(base, (2.f * (float)threadIdx.x) / (float)HS);
            float c = cosf(freq);
            float s = sinf(freq);
            float x1 = sh[threadIdx.x];
            float x2 = sh[half + threadIdx.x];
            sh[threadIdx.x] = x1 * c + x2 * s;
            sh[half + threadIdx.x] = -x1 * s + x2 * c;
        }
        __syncthreads();
        float v = sh[threadIdx.x];
        if (do_qknorm) {
            float ss = tf_block_sum(v * v, sh + HS);
            float r = rsqrtf(ss / (float)HS + 1e-6f);
            if (threadIdx.x == 0) *rstd = r;
            v = v * r * qk_scale;
        }
        vec[threadIdx.x] = (floatX)v;
        __syncthreads();
    }
}

// dqkv holds grads of the post-RoPE/post-norm Q,K (V is left alone).
// q_post/k_post are the forward outputs in (B, NH, T, HS) layout (llm.c qkvr).
__global__ void tf_qk_backward_kernel(floatX* dqkv, const floatX* q_post, const floatX* k_post,
                                      const float* q_rstd, const float* k_rstd,
                                      int B, int T, int NH, int HS, int C,
                                      float base, int do_rope, int do_qknorm, float qk_scale) {
    int idx = blockIdx.x;
    int nh = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (NH * T);
    if (b >= B) return;
    extern __shared__ float sh[];
    int half = HS / 2;
    const floatX* posts[2] = {q_post, k_post};
    for (int which = 0; which < 2; which++) {
        size_t packed = ((((size_t)b * T + t) * 3 + which) * C + (size_t)nh * HS);
        size_t perm = (((size_t)b * NH + nh) * T + t) * HS;
        float rstd = ((which == 0 ? q_rstd : k_rstd) + ((size_t)b * T + t) * NH + nh)[0];
        sh[threadIdx.x] = (float)dqkv[packed + threadIdx.x];
        __syncthreads();
        if (do_qknorm) {
            float post = (float)posts[which][perm + threadIdx.x];
            float dpost = sh[threadIdx.x];
            float y = post / qk_scale;
            float dnorm = dpost * qk_scale;
            float mean = tf_block_sum(dnorm * y, sh + HS) / (float)HS;
            sh[threadIdx.x] = rstd * (dnorm - y * mean);
            __syncthreads();
        }
        if (do_rope && threadIdx.x < half) {
            float freq = (float)t / powf(base, (2.f * (float)threadIdx.x) / (float)HS);
            float c = cosf(freq);
            float s = sinf(freq);
            float y1 = sh[threadIdx.x];
            float y2 = sh[half + threadIdx.x];
            sh[threadIdx.x] = y1 * c - y2 * s;
            sh[half + threadIdx.x] = y1 * s + y2 * c;
        }
        __syncthreads();
        dqkv[packed + threadIdx.x] = (floatX)sh[threadIdx.x];
        __syncthreads();
    }
}

__global__ void tf_adamw_kernel(float* master, floatX* param, const floatX* grad,
                                float* m, float* v, size_t n,
                                float lr, float beta1, float beta2, float b1c, float b2c,
                                float eps, float wd) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = (float)grad[i];
    float mi = beta1 * m[i] + (1.f - beta1) * g;
    float vi = beta2 * v[i] + (1.f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    float mhat = mi / b1c;
    float vhat = vi / b2c;
    float p = master[i];
    p = p - lr * (mhat / (sqrtf(vhat) + eps) + wd * p);
    master[i] = p;
    param[i] = (floatX)p;
}

// buf = momentum * buf + (1 - momentum) * grad
// out = (1 - momentum) * grad + momentum * buf   (Nesterov)
__global__ void tf_muon_nesterov_kernel(float* out, float* buf, const floatX* grad,
                                        size_t n, float momentum) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = (float)grad[i];
    float b = momentum * buf[i] + (1.f - momentum) * g;
    buf[i] = b;
    out[i] = (1.f - momentum) * g + momentum * b;
}

__global__ void tf_scale_kernel(float* x, size_t n, float s) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * s;
}

__global__ void tf_muon_apply_kernel(float* master, floatX* param, const float* update,
                                     size_t n, float lr, float wd) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float p = master[i] * (1.f - lr * wd) - lr * update[i];
    master[i] = p;
    param[i] = (floatX)p;
}

__global__ void tf_transpose_kernel(float* dst, const float* src, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) dst[(size_t)j * rows + i] = src[(size_t)i * cols + j];
}

__global__ void tf_axpy_kernel(float* y, const float* x, size_t n, float a, float b) {
    // y = a * y + b * x
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * y[i] + b * x[i];
}

__global__ void tf_frobenius_kernel(float* partial, const float* x, size_t n) {
    float ss = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float v = x[i];
        ss += v * v;
    }
    __shared__ float smem[32];
    ss = tf_block_sum(ss, smem);
    if (threadIdx.x == 0) partial[blockIdx.x] = ss;
}
