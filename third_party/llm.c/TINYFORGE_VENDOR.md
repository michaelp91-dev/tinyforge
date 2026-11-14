# Vendored llm.c

This directory is a vendor snapshot of Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c)
(MIT License, see `LICENSE`). TinyForge keeps it so the stock GPT-2 trainer
(`make train_gpt2cu`) still reproduces the ClimbMix baselines.

Local changes, both so the stock trainer builds without a hand edit:

- `llmc/cuda_utils.cuh`: `cudaMemAdvise` uses the CUDA 13 `cudaMemLocation`
  argument when `CUDART_VERSION >= 13000`, and `cudaCpuDeviceId` otherwise.
  This is the same fix the owner's Colab notebook applied by hand. CUDA 12
  builds keep the original call.
- `Makefile`: if no GPU is visible, gencode defaults to `sm_80` plus PTX.
  `nvcc`'s own default (`sm_52`) cannot compile the bf16 kernels. A machine
  with `nvidia-smi` still uses the detected architecture.
