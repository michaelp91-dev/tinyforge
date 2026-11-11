# Status (paused)

Work stopped at the owner's request. Branch `cursor/tinyforge-d6-03b9`, commit history on that branch. Nothing below was changed after the pause except this note.

## Done

- CPU model matching the nanochat d6 shape: RoPE, RMSNorm (no affine), QK-norm, ReLU², value embeddings, smear, backout, logit softcap, Muon for matrices and AdamW for embeddings/scalars. Finite-difference checks pass, including the d6 scaling count `23,199,852` and the `1,770`-step budget at batch `262,144`.
- Checkpoint, resume, val bpb, and a sampling tool. Synthetic CPU train writes `ckpt.bin` and `ckpt.bin.optim`.
- Byte-level BPE (nanochat/GPT-4 pre-split, 9 specials). Round-trip tests pass. `shard_00000` (86,016 docs, ~183M tokens) trains a tiny vocab and encodes in about 15 seconds.
- Parquet reader for ClimbMix (one optional `text` column, PLAIN byte arrays, ZSTD, data page v1). On `shard_00000`, every document's byte length matches pyarrow across all 84 row groups (0 mismatches). Row-group 0 document 51 is 253 bytes on both sides.
- ZSTD decoder checked against `zstd -d` on compressor levels 1/3/6/12/19 and on the first page of `shard_00000`.
- CUDA `sm_80` translation unit compiles with nvcc 12 (`make cuda`). This VM has no GPU. `test_cuda_parity` skips with `no CUDA-capable device is detected`.
- `docs/DESIGN.md`, `README.md`, and `colab/tinyforge_d6.ipynb` are in the tree. PR: https://github.com/michaelp91-dev/tinyforge/pull/1

## In progress

- **Doc-51 length drift (parquet/zstd).** Flagged by the owner and not fixed in this pause. A full length compare of `shard_00000` against pyarrow does not reproduce it (document 51 of row group 0 is 253 bytes both ways, and the other 86,015 docs match too). If the drift is on another shard, another row group index, or a character-length vs byte-length comparison, it is still open. Do not treat the decoder as closed until that case is pinned down.
- **CUDA backward is not the CPU oracle.** Forward omits value embeddings. Backward covers lm_head, MLP, and the output projection, then a partial Q path. Attention Jacobian, K/V, smear, value-embedding, and token-embedding gradients are not finished. A CUDA `train_step` will update from those incomplete gradients. `test_cuda_parity` is expected to fail the `1e-3` loss gap on a real GPU until this is done.
- **Throughput kernels.** Only the NT gemm uses WMMA bf16. NN and TN (the backward bulk) are naive fp32. Attention is one block per query. Cross-entropy is one thread per row and the forward still materializes the full logit matrix. None of this has been timed on an A100.

## Next steps

1. Reproduce the doc-51 length drift on the shard and row the owner saw. Fix the decoder only after that diff is in hand. Re-check every document length against pyarrow, not just page size.
2. Finish the CUDA forward (value embeddings) and backward so one step matches the CPU loss and gradients. Then run `test_cuda_parity` on an A100.
3. Move NN/TN gemms to WMMA and tile the softcap cross-entropy so a d6 microbatch does not store `[tokens, vocab]` logits.
4. Run the notebook smoke test (20 steps) and read tokens/sec before the 1,770-step d6 job. Expected memory at microbatch 8 is under 10 GB. Wall time is the open question: several hours is more likely than nanochat's ~50 minutes with the current attention kernel.
