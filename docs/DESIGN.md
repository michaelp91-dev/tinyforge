# tinyforge design review

This is the note to defend the d6 run. It records what was chosen, why, and what was turned down. The reference point is the owner's nanochat d6 result on ClimbMix: depth 6, width 384, 3 heads, vocab 32,768, context 2,048, 262,144 tokens/step, 1,770 steps, val bpb 0.9938 in about 50 minutes on an A100 40GB.

## What "20:1" counts

nanochat's scaling-parameter count is **block matrices + the untied lm_head**. Token embeddings, value embeddings, and scalars are not in that denominator. For this shape that count is

```
lm_head            32768 * 384           = 12,582,912
6 * (Q,K,V,O)      6 * 4 * 384 * 384     =  3,538,944
6 * MLP (4x, two)  6 * 2 * 384 * 1536    =  7,077,888
3 value gates      3 * 3 * 12            =        108
                                            -----------
                                            23,199,852
```

`20 * 23,199,852 / 262,144 = 1,770` steps exactly. The default batch is the same power-of-two the nanochat batch rule produces at this depth when the reference batch at d12 is 524,288 (`B ~ D^0.383`). Total parameters, including the token embedding and three value embeddings, are about 73.5M. The 20:1 ratio is not that number.

## Architecture

The block is the current nanochat GPT, not the 2022 GPT-2 block.

| Piece | Choice | Why | Turned down |
| --- | --- | --- | --- |
| Position | RoPE, base 100,000, half-split (rotate the two halves, not interleaved pairs), nanochat's sign | Matches the checkpoint convention and is the 2024–2026 default for short context. At 2,048 the base is not the binding constraint. | Learned absolute positions; interleaved RoPE; base 10,000. |
| Norm | RMSNorm, no scale and no bias, eps 1e-6, including after the token embedding | nanochat dropped the affine parameters. One less tensor for Muon to mis-handle, and it is stable at this width. | LayerNorm; RMSNorm with a learned scale (Llama). |
| Attention | Causal, 3 heads of dim 128, optional sliding window `SSSL` (short = 512, last layer forced full) | Head dim 128 is the tensor-core friendly default. `SSSL` is nanochat's default pattern; the short window is a quarter of the context, floored to a multiple of 128. | GQA with fewer KV heads (pointless at 3 heads); full attention on every layer (simpler, slightly more FLOPs, not what produced 0.9938). |
| QK norm | RMSNorm over the head, then multiply Q and K by 1.2 | Stops logit blow-ups when Muon takes large steps. The 1.2 is nanochat's fixed sharpening. | No QK norm; learned per-head scales. |
| MLP | ReLU², expansion 4, no bias | `relu(x)^2` is what nanochat ships. It is cheaper than SwiGLU and has been the better small-model activation in the modded-nanogpt line. | SwiGLU (more matmuls, similar parameter count); GEGLU; squared-ReLU with a bias. |
| Embeddings | Untied. Value embeddings on alternating layers, last layer included, mixed in with a per-head gate `3 * sigmoid(W x[:12])` | Untied lm_head is in the scaling count above. Value embeddings are the ResFormer-style skip nanochat added; they are Adam parameters, not part of the 20:1 denominator. | Tied embeddings (would change the 1,770-step budget); no value embeddings. |
| Residuals | Per-layer `resid_lambda` (init ~1.15→1.05) and `x0_lambda` (init ~0.20→0.05) mixing the block input with the post-embedding stream. A smear gate mixes the previous token into the current one. A backout subtracts `0.2 *` the mid-network residual before the final norm. | These are the modded-nanogpt / nanochat scalars. They are cheap and they move loss at small depth. Smear starts at lambda 0, so it is a no-op until it earns its way in. | Plain pre-norm residual only. |
| Logits | Softcap 15: `15 * tanh(u / 15)`, then fp32 cross-entropy | Keeps the loss in a range Adam and Muon agree on. | No cap; a learned cap. |
| Init | Normal embeddings std 0.8, lm_head std 0.001, uniform hidden weights matched to `1/sqrt(d)`, zero output projections, MLP up-projection at 0.4× that scale | Zero output projections make the initial residual stream exactly the embedding. The small lm_head keeps the initial loss near `log(vocab)`. | Global normal init; muP fan-in on every matrix. |

Biases are absent everywhere.

## Optimizer

Matrices (Q, K, V, O, the two MLP matrices, and the value-embedding gates) use Muon. Embeddings, the lm_head, and the scalars use AdamW. That split is the one nanochat converged on: Muon wants a matrix with a meaningful spectrum, and an embedding row does not have one.

Muon here is the 2025 nanochat step, not the original Newton–Schulz blog post:

1. Nesterov momentum (buffer EMA, then a mix of the buffer and the new grad). Momentum warms from 0.85 to 0.97 over 400 steps and falls to 0.90 in the decay.
2. Row equilibration (MuonEq) before the orthogonalization.
3. Five Polar Express iterations (the quintic coefficients from Amsel et al.), fp32.
4. Renormalize the update to Frobenius norm `sqrt(min(rows, cols))`.
5. Factored second-moment rescaling (NorMuon), beta2 0.9.
6. Cautious weight decay: decay a weight only when the update already points the same way.

AdamW groups match nanochat's betas and decoupled decays: lm_head `(0.8, 0.96)` decay 0.01, token embedding `(0.8, 0.995)` decay 0.001, value embeddings the same betas with decay 0.01 and half the embedding learning rate, residual lambdas `(0.8, 0.95)` decay 0.05 at `0.01×` the scalar lr, x0 lambdas `(0.96, 0.95)` with no decay, smear/backout at a fixed base lr 0.2 with no decay.

Learning rates are transferred from the d12 reference the way nanochat does it. Adam rates scale by `1/sqrt(d/768)` and by `sqrt(batch / 524288)`. Muon's matrix rate scales only with the batch term, then again by `sqrt(max(1, rows/cols))` inside the update so a tall MLP up-projection takes a larger step. The schedule is linear warmup (40 steps), a flat middle, and a linear decay over the last 65% of steps down to 5% of the peak. Weight decay on matrices cosine-decays to zero.

Turned down: AdamW on every parameter (Muon is the whole point of this generation of small-model runs); Lion; a single global learning rate.

## Data, tokenizer, metric

ClimbMix shards are Parquet written by parquet-cpp-arrow 21: one optional `text` column, PLAIN byte arrays, ZSTD frames, one data page per row group of 1,024 documents. The reader speaks exactly that. The ZSTD decoder covers the frames those files actually contain (no dictionaries). It was checked against `zstd -d` on the first page of `shard_00000` and on a spread of compressor levels.

The tokenizer is a byte-level BPE. Pre-tokenization is the nanochat pattern, which is the GPT-4 split with numbers capped at two digits:

```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

Implemented by hand, checked against the `regex` module on the cases that pattern cares about. Vocab 32,768 includes 256 bytes, the merges, and nine special tokens appended after training. `<|bos|>` is the first of those and is prepended to every document. The other eight exist so a later SFT run can use the same vocab; they are not in the pretraining loss as text.

Documents are packed with nanochat's BOS-aligned best fit: every row starts with BOS, the largest buffered document that fits is placed, and a document that does not fit is cropped and dropped. There is no padding. About a third of tokens are thrown away at context 2,048. That is deliberate. It is the distribution the 0.9938 number was measured on.

Bits per byte is the nanochat definition. Sum the nats of target tokens whose byte length is positive, divide by `ln(2) * total bytes`. Specials contribute 0 bytes and are left out of both the numerator and the denominator. Ignore-index targets (`-1`) are left out of the training loss and of bpb. bpb does not depend on the vocab size, which is why it is the right number to compare against 0.9938 even though this BPE will not be byte-identical to rustbpe.

## What will not match 0.9938 on the first run

- The BPE is trained on a sample (the notebook default is a few hundred MB), not on the corpus nanochat's rustbpe saw. bpb is still the right metric, but the token distribution differs.
- Master weights stay fp32, including embeddings. nanochat stores the token and value embeddings in bf16.
- Muon's orthogonalization runs in fp32. nanochat runs Polar Express in bf16.
- The GPU backward is not yet a line-by-line port of the CPU backward. See the status section.

## Backend status

**CPU.** This is the oracle. fp32, finite-difference checks on RMSNorm, RoPE, causal attention (including a window), and on a 2-layer model covering the lm_head, embeddings, residual scalars, backout, smear, MLP, Q, and the value-embedding gate. A fixed batch's loss falls under the Muon+AdamW step. No third-party math library.

**CUDA (sm_80).** The file compiles with nvcc 12 targeting sm_80. It contains a bf16 WMMA matmul (fp32 accumulate) for aligned NT gemms, fp32 tiled gemms for the other two layouts and for odd shapes, an online-softmax attention kernel that does not store the score matrix, RMSNorm, RoPE, ReLU², and a fused softcap cross-entropy. The forward uses those kernels. The backward computes lm_head, MLP, and output-projection gradients on the device and then a partial Q path; the attention Jacobian, smear, and value-embedding backward are not finished to the standard of the CPU oracle. The parity test skips cleanly when the machine has no GPU, which is this VM (`no CUDA-capable device is detected`). Nothing in this repository has been executed on an A100.

That gap is the thing most likely to disappoint on first GPU contact. The smoke-test cell in the notebook exists so a 20-step run reports tokens/sec before anyone spends the full 1,770 steps. If that number is far below ~1e5 tokens/sec, the attention kernel is the first place to look: it is one block per query, which is correct and bandwidth-heavy, not a fully specialized flash kernel.

## Memory and time, d6, A100 40GB

Default microbatch in the notebook is 8 sequences (16,384 tokens) so the fused loss never holds the full `[tokens, vocab]` logit tensor. 262,144 / 16,384 = 16 gradient-accumulation steps.

Rough resident memory at that microbatch:

| Resident | Size |
| --- | --- |
| fp32 master weights + grads | ~0.6 GB |
| Adam state (embeddings, lm_head, value embeddings, scalars) | ~0.5 GB |
| Muon momentum | ~40 MB |
| Activations for one microbatch, scores not stored | ~2–4 GB |
| Logit tile / attention workspace | ~1–2 GB |

Call it under 10 GB. A 40GB A100 has room to raise the microbatch to 16 or 32 after the smoke test looks stable.

Wall time is the open question. nanochat's 50 minutes at ~7% of an A100's 312 TFLOP/s bf16 peak is the reference (the model is small, so MFU stays low). A WMMA matmul can live in that band. The attention kernel as written is unlikely to. Plan on a few hours, and read the smoke-test tokens/sec before launching all 1,770 steps. MFU is printed every step against 312 TFLOP/s.

## Known gaps vs nanochat

- No FP8, no multi-GPU, no ZeRO.
- No core-metric (multiple-choice) eval. bpb only.
- Generation recomputes the full prompt each token. Fine for a 64-token sample, not a chat server.
- Tokenizer sample size and tie-break (higher count, then smaller id) differ from rustbpe.
- GPU backward is behind the CPU oracle, as above.
- Embeddings are fp32 rather than bf16.
