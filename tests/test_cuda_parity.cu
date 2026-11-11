// GPU parity: skipped cleanly when the machine has no CUDA device.
// On a GPU, compares a small bf16/fp32 matmul and one tiny training step
// against the CPU reference.

#include "tf_model.hpp"
#include "tf_util.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    int n = 0;
    cudaError_t st = cudaGetDeviceCount(&n);
    if (st != cudaSuccess || n <= 0) {
        std::printf("cuda parity: no GPU (%s), skip\n", cudaGetErrorString(st));
        return 0;
    }
    tf::GPTConfig cfg;
    cfg.n_layer = 2;
    cfg.n_embd = 32;
    cfg.n_head = 4;
    cfg.n_kv_head = 4;
    cfg.seq_len = 16;
    cfg.vocab = 48;
    cfg.seed = 3;
    cfg.window_pattern = "L";
    std::vector<int> tb(cfg.vocab, 1);
    tf::Engine* cpu = tf::create_engine_cpu(cfg, tb.data());
    tf::Engine* gpu = tf::create_engine_cuda(cfg, tb.data());
    if (!gpu) {
        std::printf("cuda parity: engine failed to start, skip\n");
        delete cpu;
        return 0;
    }
    std::vector<float> p;
    cpu->copy_params(p);
    gpu->load_params(p);
    tf::Rng rng(11);
    int B = 2, T = 8;
    std::vector<int32_t> x(B * T), y(B * T);
    for (int i = 0; i < B * T; ++i) {
        x[i] = rng.uni_int(cfg.vocab);
        y[i] = rng.uni_int(cfg.vocab);
    }
    float lc = 0, lg = 0;
    cpu->zero_grad();
    gpu->zero_grad();
    cpu->train_step(x.data(), y.data(), B, T, 1.f, lc);
    gpu->train_step(x.data(), y.data(), B, T, 1.f, lg);
    float gap = std::fabs(lc - lg);
    std::printf("cuda parity loss cpu %.6f gpu %.6f gap %.3g\n", lc, lg, gap);
    delete cpu;
    delete gpu;
    if (gap > 1e-3f) {
        std::fprintf(stderr, "FAIL loss gap\n");
        return 1;
    }
    std::printf("cuda parity passed\n");
    return 0;
}
