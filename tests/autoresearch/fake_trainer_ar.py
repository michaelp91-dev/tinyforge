#!/usr/bin/env python3
"""CPU stand-in for train_tinyforge --time-budget (box tests of the autoresearch worker).
Same flags and the same '---' summary format. Score is a made-up function of the args.
-b >= 64 fakes an OOM; --matrix-lr >= 1 fakes a non-finite loss; FAKE_HANG=1 never exits."""
import math, os, sys, time
a = sys.argv[1:]
def opt(k, d=None):
    return a[a.index(k) + 1] if k in a else d
budget = float(opt("--time-budget", "0")); B = int(opt("-b", "16")); T = int(opt("-t", "512")); D = int(opt("-d", str(B * T)))
m = int(opt("-m", "20")); mlr = float(opt("--matrix-lr", "0.02")); step_s = float(os.environ.get("FAKE_STEP_S", "0.05"))
assert budget > 0 and opt("-v") == "0" and opt("-n") == "0", "worker must pass budget mode flags"
assert os.path.exists(opt("--token-bytes")) and os.path.exists(opt("-j")), "data paths"
print(f"device: FAKE\narch modern  {opt('-e', 'd12')}\nB={B} T={T} total_batch={D}", flush=True)
if B >= 64:
    print("CUDA error: out of memory (fake)", flush=True); sys.exit(1)
if os.environ.get("FAKE_HANG") == "1":
    time.sleep(10 ** 6)
t0 = time.time(); step = 0
while time.time() - t0 < budget:
    time.sleep(step_s); step += 1
    if mlr >= 1:
        print(f"step {step}: loss nan", flush=True); sys.stderr.write("loss is not finite at step 3; stopping\n"); sys.exit(2)
    print(f"step {step}/1000000  loss {5 - math.log1p(step):.4f}  lr_mult 1.000  {step_s:.2f}s  18000 tok/s  eta 0.0h", flush=True)
bpb = 1.40 - 0.03 * math.log2(65536 / D) + 2.0 * abs(mlr - 0.03)
print(f"  val loss {bpb * 3.2:.4f}  bpb {bpb:.4f}  (4.6 bytes/token)")
print("---")
print(f"val_bpb:          {bpb:.6f}\nval_loss:         {bpb * 3.2:.6f}\nval_tokens:       {m * B * T}")
print(f"training_seconds: {budget:.1f}\ntotal_seconds:    {time.time() - t0:.1f}\npeak_vram_mb:     {B * 600:.1f}")
print(f"total_tokens_M:   {step * D / 1e6:.2f}\ntok_per_sec:      18000\nnum_steps:        {step}\nnum_params_M:     135.00\ndepth:            12\ndone", flush=True)
