# TinyForge autoresearch

Fixed-budget hyperparameter and model research for TinyForge on one Colab T4. Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (MIT License, Copyright (c) Andrej Karpathy): the fixed time budget, val bpb as the only metric, the `results.tsv` format, keep/discard, and the `program.md` researcher guide all come from there. In this version the researcher is Jarvis (asynchronous, through git) and the Colab notebook is a dumb worker.

- `../TinyForge-AutoResearch.ipynb`: the Colab worker. Run all on a T4; it disconnects itself. Trial defaults: 8 experiments, 2.2 h session cap, 20 min idle timeout, watchdog at 2.45 h (about 2 compute units).
- `worker.py`: poll queue, build per commit (cached), train with `--time-budget`, score, upload.
- `program.md`: rules and procedure for the researcher.
- `ar.py`: researcher helper (`new`, `promote`, `show`, `check`).
- `best.json`: current base config. `control.json`: stop switch. `queue/`: experiments.
- `../tests/autoresearch/`: box dry runs (fake trainer, mock HF, local git):
  `python3 tests/autoresearch/dry_run.py main|control_stop|session_cap|idle_empty|hang|relaunch|nodata`
- `../tests/autoresearch/gen_nb.py`: regenerates the notebook. Edit that file, not the .ipynb.

Trainer change: `src/train_tinyforge.cu --time-budget S` runs the LR schedule on training-time progress, stops at S seconds of training, runs one final val, and prints a `---` summary (`val_bpb`, `peak_vram_mb`, `tok_per_sec`, ...). Without the flag, the trainer behaves exactly as before.
