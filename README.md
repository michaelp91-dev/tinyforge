# tinyforge

A from-scratch trainer for small GPTs. C/C++ and CUDA only on the training and data path: our own matmul, our own BPE, our own Parquet/ZSTD reader. The CPU backend is the correctness oracle. The CUDA backend targets an A100 (sm_80) and is what the d6 Colab notebook runs.

The reference configuration is nanochat d6 on ClimbMix: depth 6, width 384, 3 heads, vocab 32,768, context 2,048, 262,144 tokens/step, 1,770 steps. The owner's run of that recipe scored **0.9938 val bpb** in about 50 minutes on an A100 40GB. Why the architecture looks the way it does is in [docs/DESIGN.md](docs/DESIGN.md).

## Status

| Piece | State |
| --- | --- |
| Parquet + ZSTD reader for ClimbMix shards | Reads `shard_00000` (86,016 docs, 84 row groups). Checked against `zstd -d` and pyarrow. |
| Byte-level BPE, GPT-4/nanochat pre-split, 9 special tokens | Round-trip tests pass. |
| CPU model, Muon + AdamW, bpb, checkpoint, sample | Finite-difference gradient checks pass, including a 2-layer model. |
| CUDA (sm_80) | Compiles with nvcc 12. WMMA bf16 matmul, online-softmax attention, RMSNorm, fused softcap cross-entropy. **Not executed on a GPU in this environment** (no device). The attention backward and value-embedding backward are behind the CPU oracle; read the design note before trusting a long run. |
| Colab launcher | [colab/tinyforge_d6.ipynb](colab/tinyforge_d6.ipynb). |

## Quick start (CPU)

```bash
make
make test
# synthetic smoke, no data
./build/tf_train --synthetic --depth 2 --dim 64 --heads 2 --seq 64 \
  --device-batch 4 --total-batch 256 --steps 20 --vocab 512 --out runs/smoke
```

A real run needs a tokenizer and token files:

```bash
./build/tf_tok_train --vocab 32768 --max-bytes 200000000 --out tokenizer.bin shard_00000.parquet
./build/tf_tok_encode --tokenizer tokenizer.bin --out data/train.bin shard_00000.parquet
./build/tf_train --backend cpu --tokenizer tokenizer.bin --data data/train.bin \
  --depth 2 --dim 64 --seq 128 --steps 50 --out runs/cpu
./build/tf_sample --ckpt runs/cpu/ckpt.bin --tokenizer tokenizer.bin --prompt "The capital of France is"
```

`make cuda` builds `build/tf_train_cuda` and `build/tf_sample_cuda` (requires `nvcc`, arch sm_80).

## Layout

```
include/          public headers
src/              zstd, parquet, bpe, model (cpu), model_cuda.cu, train/sample/tok tools
src/ops_cpu.cpp   reference math (the finite-difference target)
tests/            grad checks, tokenizer, parquet, zstd, cuda parity (skips without a GPU)
colab/            A100 launcher
docs/DESIGN.md    choices, alternatives, what will not match 0.9938
```

Checkpoints are `runs/<name>/ckpt.bin` (fp32 weights) plus `ckpt.bin.optim`. Logs are `train.log`.

## d6 on an A100

Use the notebook. It clones this private repo, builds, downloads ClimbMix shards `00000`–`00040` for train and `06542` for val, trains the tokenizer, tokenizes, runs a short smoke test, then the 1,770-step d6 run, then writes a sample. Checkpoints and the log are copied to `MyDrive/tinyforge/`. Raw Parquet and the token files stay on the Colab disk.

Defaults that reproduce the step budget: `--depth 6 --ratio 20 --total-batch 262144 --seq 2048`. The notebook uses microbatch 8 (16 accumulation steps) so the logit tile fits. Expect several hours rather than 50 minutes until the attention kernel is tightened; the smoke test prints tokens/sec and MFU before the long run. Details and the memory table are in the design note.
