# TinyForge autoresearch: researcher guide (for Jarvis)

Adapted from `program.md` in Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) (MIT). Same idea: short fixed-budget experiments, one score, keep what helps and drop what doesn't. What changed for TinyForge: the "LLM agent" is Jarvis, working asynchronously, not a model inside the notebook. The Colab notebook `TinyForge-AutoResearch.ipynb` is only the worker (see `worker.py`). Jarvis is the researcher and talks to it through this branch (GitHub) and the private HF repo `michaelp91-dev/tinyforge-autoresearch`.

## Goal

Get the lowest **val_bpb** after **12 minutes of training on one Colab T4**, starting from the d12 config of the 8-hour run (tag `tinyforge-8hr-v1`). Later the winners get confirmed in longer runs and then the official Kaggle pre-training.

## Fixed (never change these in an experiment)

- **Data:** `tinyforge-data/climbmix_v32768_610M`, first train shard. Every experiment starts at the same place in the same shard.
- **Tokenizer and vocab:** 32,768.
- **Eval:** the trainer's own bpb on 229,376 tokens of `val.bin`, sequence length 512, run once at the end. The worker sets `-j -m -v`, so experiments can't touch them. **`-t` is fixed at 512**, because a longer eval context would lower bpb by itself.
- **Budget:** 12 minutes of training time (`--time-budget 720`). The first 3 steps (warm-up) and the final eval don't count. The LR schedule runs on time progress, so every experiment gets a full warmdown.
- **Worker-owned flags (rejected if present):** `-i -j -o -x -v -m -n -y --time-limit --time-budget --token-bytes --vocab --keep-ckpts --nproc --rank --sync-dir --profile`.
- **The HF repos, the notebook's caps, and the 8-hour notebook.** Changes to the notebook itself need Michael.
- **Session caps (trial defaults Michael chose):** 8 experiments, 2.2 h session cap (no experiment starts unless it can finish before it), 20 min idle timeout, hard watchdog at 2.45 h. That is about 2 compute units. Raising them needs Michael.
- **Comparisons:** these bpb numbers are not comparable with the 8-hour run's 1.0861. That run trained 40x longer and evaluated on a slightly different token count.

## What you may change

- **Hyperparameters, as trainer flags in an experiment's `args`:**
  - learning rates: `--matrix-lr`, `--embed-lr`, `--unembed-lr`
  - schedule: `--warmup` (in steps), `--warmdown` (fraction of the budget), `--final-lr-frac`, `-k`
  - batch: `-b` (micro batch), `-d` (tokens per optimizer step, a multiple of `-b`*512). `-b`*512 must divide 229,376, so `-b` is one of 1, 2, 4, 7, 8, 14, 16, 28, 32.
  - optimizer: `--optim muon|adamw`, `--momentum`, `--muon-wd`, `--embed-wd`, `--head-wd`
  - depth: `-e dN`. Width is `N*64` rounded up to a multiple of 128, with head size 128, so d8 gives C=512 and 4 heads, d10 gives 640/5, d12 gives 768/6, d14 gives 896/7.
  - architecture switches: `--rope --rmsnorm --qknorm --relu2 --untie --bias --embed-norm` (0/1), `--softcap`, `--qk-scale`, `--rope-base`
  - speed: `--ns-fp16`, `--ns-batch`, `--fused-loss`, `--fuse`
- **Model and trainer code:** `src/train_tinyforge.cu` and `src/tf_kernels.cuh`. Examples are a width/aspect flag independent of depth, head size 64, or a different MLP. Commit the change to this branch, then point the experiment's `code_ref` at that commit. The worker rebuilds only when `src/`, `third_party/llm.c/llmc` or the Makefile changed, and caches builds. A build failure is logged as `crash` and the worker moves on. Don't break `--time-budget` or the `---` summary block. Don't change the eval code in the val block or the bpb formula.
- **VRAM is a soft limit (T4 has 15 GB).** An OOM just logs as crash.
- **Simplicity rule (from Karpathy):** a tiny gain that adds a lot of hacky code isn't worth keeping. Equal bpb with simpler code is a win.

## How a round works (each wake)

1. **Read results.** Run this on clarke-server (its HF token needs read access to the repo):
   `hf download michaelp91-dev/tinyforge-autoresearch results.tsv results.jsonl status.json --local-dir /tmp/ar`
   Use `python3 autoresearch/ar.py show /tmp/ar/results.tsv` for a sorted view. `status.json` has `phase`, `current`, `updated`, `done` and `stop_reason`, and gets a heartbeat at least every 5 minutes.
2. **Decide keep or revert.** The worker marks a run `keep` if it beat the best so far and `discard` otherwise. Keeping means `python3 autoresearch/ar.py promote <id> --results /tmp/ar/results.jsonl`, which updates `best.json`, the base for new experiments. Reverting means doing nothing: new experiments start from `best.json`, so a code change that lost is simply never referenced again. **Never rewrite or force-push this branch.** The worker checks out old commits by hash. Expect noise of about ±0.003 bpb between identical runs. Rerun a surprising win, or a gain under 0.003, with a new id before building on it.
3. **Queue the next experiments.** For example: `python3 autoresearch/ar.py new 012-mlr03 "matrix lr 0.02 -> 0.03" --set matrix-lr=0.03`. It writes the full args from `best.json` plus the change. Ids are 3 digits plus a short name, run in sorted order, and are never reused. A rejected or crashed id needs a new number. Run `python3 autoresearch/ar.py check`, then commit and push. **Keep 2 to 3 experiments queued** so the GPU never waits for a wake. Each one takes about 13.5 minutes.
4. **Stop early if needed.** Set `"stop": true` in `control.json` and push. The worker finishes the current experiment, uploads, and disconnects. `max_experiments` there can only lower the notebook's cap.
5. **Tell Michael only what matters:** the session started, a new best, a crash streak, or the session ended with a summary. Use plain language, no tables.

## Suggested experiments, ordered by expected value

The first round is already queued: `000-baseline` (always first), `001-microbatch32` (speed), `002-batch32k` and `003-batch16k`. At about 200 optimizer steps in 12 minutes, the baseline is step-starved, so smaller batches and faster steps should pay off most.

Next, each against the current best:
1. `--warmup 10`, since 40 warmup steps is about 20% of a 200-step run
2. `--matrix-lr 0.03` and `0.014`
3. `--warmdown 0.4` and `0.85`
4. `--embed-lr 0.45 --unembed-lr 0.012` (both x1.5), then x0.67
5. `-e d10` and `-e d8`: faster per token, so more tokens in 12 minutes
6. `--final-lr-frac 0.0` and `0.1`
7. `--momentum 0.9`, and `--muon-wd 0.01`
8. `--softcap 30`, or `--softcap 0`
9. Code ideas once the flags settle:
   - a width flag (aspect ratio) so depth and width can be tuned separately
   - head size 64
   - mixing tied and untied embeddings (`--untie 0` is already a flag)
   - value-embedding or residual-scalar tricks from nanochat
10. Combine the best near-misses. Before calling it done, rerun the top 2 configs once to check the noise.

Short budgets favor configs that learn fast early. Before the Kaggle pre-training, the top 1 or 2 configs should be confirmed against the baseline in a longer run of about 1 hour, and Michael decides on that run.

## Logging format (unchanged from autoresearch)

`results.tsv` is tab-separated, with the columns `commit  val_bpb  memory_gb  status  description`:
- `commit` is the code commit the experiment ran.
- `description` starts with the experiment id.
- Crashes show `0.000000` and `0.0`, with the reason in brackets.

`results.jsonl` holds the full record for each run: args, wall time, tok/s, steps, tokens and the log tail on a crash. Per-run logs are in `runs/<id>/run.log`.
