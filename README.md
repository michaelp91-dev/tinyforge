# TinyForge

A C/C++/CUDA LLM trainer. The stock GPT-2 path is [Karpathy's llm.c](https://github.com/karpathy/llm.c), vendored under `third_party/llm.c` (MIT, see `LICENSE` and `TINYFORGE_VENDOR.md`). The training stack is C, C++, and CUDA. Python is only for tokenizer training, ClimbMix packing, Hugging Face export, and bits-per-byte eval.

`train_tinyforge` adds a flag-gated decoder aimed at the gap between stock llm.c and [nanochat](https://github.com/karpathy/nanochat) on the same ClimbMix val shard (`shard_06542`).

## What changed vs llm.c

Default `train_tinyforge` (`--arch modern`) is not GPT-2. Each piece can be turned off.

| Piece | modern default | Why |
| --- | --- | --- |
| RoPE, no learned positions | `--rope 1`, base 100000, nanochat rotate-half sign | relative position, no `wpe` |
| Parameter-free RMSNorm | `--rmsnorm 1` (eps 1e-6) | no affine weight or bias |
| RMSNorm on the token embedding | `--embed-norm 1` | same as nanochat, always parameter-free |
| QK-norm + 1.2 scale | `--qknorm 1 --qk-scale 1.2` | per-head RMSNorm after RoPE |
| ReLU² MLP | `--relu2 1` | replaces tanh-GELU |
| Untied `lm_head` | `--untie 1` | separate input and output embeddings |
| No linear biases | `--bias 0` | |
| Logit softcap 15 | `--softcap 15` | `cap * tanh(x / cap)` before the loss |
| Muon for matrices, AdamW for embeddings | `--optim muon` | Newton–Schulz quintic, then `sqrt(max(1, rows/cols))` |
| Warmup, constant, linear warmdown | `-k warmdown` | nanochat schedule |

Shapes follow nanochat's depth rule, not llm.c's GPT-2 table. `dN` means depth `N`, width `align(N * 64, 128)`, head dim 128.

| name | layers | width | heads | params | matrix params |
| --- | --- | --- | --- | --- | --- |
| d6 | 6 | 384 | 3 | 35.8M | 10.6M |
| d12 | 12 | 768 | 6 | 135.3M | 84.9M |
| d16 | 16 | 1024 | 8 | 268.4M | 201.3M |
| d20 | 20 | 1280 | 10 | 477.1M | 393.2M |

Params are `12·C²·L` for the blocks plus `2·32768·C` for the untied input and output embeddings.

`--arch gpt2` switches to the llm.c GPT-2 table (d6 = 384/6 heads, d12 = 768/12 heads), learned positions, affine LayerNorm, tanh-GELU, biases, tied embeddings, and one AdamW. That mode is the same **architecture** as llm.c. The bit-level baseline is still the vendored binary, which is the code that produced the numbers below.

Not in this cut, and the main remaining gap to current nanochat: value embeddings, smear, per-layer residual/`x0` lambdas, backout, sliding-window attention, and Muon's Polar Express / NorMuon / cautious weight decay. The tokenizer is byte-level BPE trained on ClimbMix, not rustbpe's GPT-4 regex.

## Measured baselines (same eval)

Bits per byte on ClimbMix `shard_06542`, from the owner's runs. These are stock llm.c GPT-2, not TinyForge modern.

| run | bpb | tokens |
| --- | --- | --- |
| llm.c d6 | 1.2110 | ~470M |
| llm.c d6 | 1.1037 | 1.2B |
| llm.c d12 | 1.0115 | 4254 steps × 262144 |
| nanochat D6 | 0.9938 | (owner's run) |

The modern d6/d12 recipe below has **not** been trained here. There is no GPU in the build environment, so no new bpb is claimed.

## Build

```bash
make test          # CPU reference vs PyTorch autograd
make train_tinyforge
make -C third_party/llm.c train_gpt2      # stock CPU trainer
make -C third_party/llm.c train_gpt2cu    # stock CUDA trainer
```

`nvcc` with no visible GPU defaults the vendored Makefile to `sm_80` plus PTX, so an A100 runs it natively and an L4 can JIT. Colab's `nvidia-smi` still selects the real architecture.

Python helpers need `torch`, `numpy`, `tokenizers`, `transformers`, `datasets`, `pyarrow`, `requests`, and `tiktoken` (GPT-2 eval only).

## Train

Micro-batch is `-b 16` (32 OOMs on the owner's GPU). Total batch is `-d`, filled in with gradient accumulation. Data is llm.c `uint16` shards (`python/prepare_climbmix.py`).

Recommended modern run, one GPU. Checkpoints stay on local disk. Only the Hugging Face folders are copied to Drive (see `TinyForge-Train.ipynb`).

```bash
python python/train_tokenizer.py --out data/tokenizer --vocab-size 32768
python python/prepare_climbmix.py --tokenizer data/tokenizer \
    --out data/climbmix --target-tokens 1400000000

# d6: depth*64 width, ~23.2M scaling params, 12x tokens → 1062 steps at batch 262144
./train_tinyforge -i "data/climbmix/train_*.bin" -j data/climbmix/val.bin \
    -e d6 -o runs/d6 -b 16 -t 1024 -d 262144 -x 1062 \
    -v 250 -n 250 --vocab 32768 \
    --token-bytes data/tokenizer/token_bytes.bin

# d12: ~110.1M scaling params → 5040 steps
./train_tinyforge -i "data/climbmix/train_*.bin" -j data/climbmix/val.bin \
    -e d12 -o runs/d12 -b 16 -t 1024 -d 262144 -x 5040 \
    -v 250 -n 250 --vocab 32768 \
    --token-bytes data/tokenizer/token_bytes.bin
```

Those steps are `12 * (matrix params + lm_head) / 262144`, the same param-to-data ratio nanochat uses for its reference run. Scaling params here are the four square-ish matrices per layer plus the untied head, not the token embedding.

Effective learning rates are the flags below, times `sqrt(batch / 524288)` and, for the two AdamW groups only, times `sqrt(768 / width)`.

| flag | default (modern) |
| --- | --- |
| `--matrix-lr` | 0.02 (Muon) |
| `--embed-lr` | 0.3 (AdamW, betas 0.8 / 0.995, eps 1e-10) |
| `--unembed-lr` | 0.008 (AdamW, betas 0.8 / 0.96) |
| `--embed-wd` / `--head-wd` / `--muon-wd` | 0.001 / 0.01 / 0 |
| `--warmup` | 40 |
| `--warmdown` | 0.65 of the run, linear to `--final-lr-frac` 0.05 |
| `--momentum` | 0.95 |

`-y 1` resumes weights, Adam/Muon state, step, and the dataloader cursor from `runs/<name>/`. Shuffle is off because ClimbMix is already shuffled; that keeps resume aligned with the file. `-l` sets the GPT-2 AdamW learning rate and, on the modern path, the Muon matrix learning rate.

Stock baseline (the run that scored 1.2110 / 1.1037 / 1.0115), GPT-2 tokenizer, from the vendored trainer:

```bash
make -C third_party/llm.c train_gpt2cu
./third_party/llm.c/train_gpt2cu -i "data/climbmix_gpt2/train_*.bin" -j data/climbmix_gpt2/val.bin \
    -e d6 -b 16 -t 1024 -o runs/llm.c-d6
```

Prepare that data with `python python/prepare_climbmix.py --tokenizer gpt2`.

## T4 and Kaggle (fp32, multi-GPU, resume)

A T4 (sm_75) has no bf16, so build it in fp32:

```bash
make train_tinyforge GPU=t4                       # one GPU
make train_tinyforge GPU=t4 USE_NCCL=1 NCCL_HOME=/path/to/nvidia/nccl   # data parallel
```

d12 on a 16 GB T4 uses `-b 16 -t 512` (about 9 GiB). `-b 16 -t 1024` needs about 17 GiB and does not fit. Keep the same `-d 262144` total batch; accumulation fills the rest.

Flags added for long runs on session-limited machines:

- `--time-limit <seconds>`: write a checkpoint and exit cleanly before the session ends. Rerun with `-y 1` to continue.
- `--keep-ckpts N`: keep only the newest N checkpoints (a d12 checkpoint is about 1.8 GB including optimizer state).
- `--nproc N --rank R --sync-dir DIR`: one process per GPU, gradients averaged with NCCL. Start one process per rank with the same arguments. `-d` is the global batch. Checkpoints written by one GPU count resume on two GPUs and back.
- Checkpoints are written under a temp name and renamed, so a killed session never leaves a half-written file.

`TinyForge.ipynb` (Colab) is a short T4 smoke test: about 60 small steps, a time-limited stop, and a resume. `TinyForge-Kaggle.ipynb` is the full d12 run on Kaggle's 2×T4 with resume across sessions.

## Evaluate

```bash
python python/export_hf.py --input runs/d6/model_00001062.bin \
    --output d6-hf --tokenizer data/tokenizer
MODEL_DIR=d6-hf python python/eval_bpb.py
```

`export_hf.py` writes `d6-hf/` (or `d12-hf/`) with `config.json`, `pytorch_model.bin`, and `modeling_tinyforge.py`. Classic GPT-2 flags are exported as `GPT2LMHeadModel` instead, which the original tiktoken eval can score. `eval_bpb.py` handles both: tiktoken GPT-2, or the saved tokenizer when `model_type` is `tinyforge`. Document separators count as 0 bytes. The val shard is `shard_06542` of `karpathy/climbmix-400b-shuffle`.

The trainer's own val loss is on `val.bin` from the prepare script. That is a monitor, not the published bpb.

## GGUF (llama.cpp)

Stock llama.cpp has no architecture for this block. nanochat uses the same block, and the `nanochat` branch of [ulanch/llama.cpp](https://github.com/ulanch/llama.cpp) adds it as arch `nanochat`. `python/convert_tinyforge_to_gguf.py` writes that arch. The QK-norm scale goes into the q/k norm weights, and the parameter-free norms are written as ones.

```bash
python python/convert_tinyforge_to_gguf.py --model d12-hf --out d12-f32.gguf
llama-quantize d12-f32.gguf d12-q8_0.gguf Q8_0
llama-server -m d12-q8_0.gguf --port 8099 &
python python/check_gguf.py --model d12-hf --url http://127.0.0.1:8099
```

`check_gguf.py` compares tokenization and next-token probabilities against the PyTorch module. Quantize from f32 or bf16, not f16, because ReLU² activations can overflow fp16.

## Tests

`src/tf_ref.c` is the fp32 reference for RMSNorm, LayerNorm, RoPE, ReLU², tanh-GELU, softcap, linear, causal attention, the Newton–Schulz iteration, and a full decoder forward/backward. `tests/test_layers.py` checks those gradients against PyTorch. `tests/test_export.py` round-trips a checkpoint through the exporter.

## What was not run

This environment has `nvcc` and no GPU.

- `train_tinyforge` and `third_party/llm.c/train_gpt2cu` both compile (sm_80 + PTX). Neither was executed.
- CUDA matmul, attention, RoPE, RMSNorm, Muon, and the loss kernel therefore have no runtime numerical check. They are written to the same formulas as `tf_ref.c`, and that CPU reference matches PyTorch, but the device path can still be wrong.
- No modern d6 or d12 bpb has been measured. Do not treat the recipe as having closed the 0.9938 gap until `eval_bpb.py` says so.

## Next steps

1. Run the notebook's d6 config and read `eval_bpb.py`. If the loss is unstable, lower `--matrix-lr` before changing the architecture.
2. Value embeddings, smear, residual/`x0` lambdas, and backout, in that order. They are the rest of the current nanochat block.
3. Replace Newton–Schulz with Polar Express plus NorMuon variance reduction if the orthogonalized update is the remaining optimizer gap.
4. Train the tokenizer with nanochat's regex (`\p{N}{1,2}` split) instead of byte-level BPE.
5. Flash attention. The stored scores are the large activation (`B * heads * T * T` per layer) and the reason `-b 32` OOMs.
