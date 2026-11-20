#!/usr/bin/env python3
"""CPU stand-in for train_tinyforge (box tests of TinyForge-Kaggle-Pretrain.ipynb).
Same CLI, checkpoint file names/headers, llm.c loader cursor logic (incl. --nproc: a sample is B*T*nproc tokens),
same stdout format. Behaves like the real binary where it matters for the notebook:
  * output dir: created with mkdir -p (what utils.h create_dir_if_not_exists does since bdc405d);
  * --sync-dir must already exist (rank 0 writes nccl_id.bin there; the real one exits "cannot write");
  * resuming at a shard index past the existing shards crashes (llm.c indexes glob results);
  * --token-bytes and -j files must exist.
Env: FAKE_STEP_S, FAKE_TRACE (log consumed micro-batches), FAKE_CRASH_AT (rank 0 crash once),
FAKE_RANK1_CRASH_AT (rank 1 crash once), FAKE_NCCL_HANG=1 (multi-rank launch never starts)."""
import glob, math, os, struct, sys, time
a = sys.argv[1:]
def opt(k, d=None):
    return a[a.index(k) + 1] if k in a else d
pat, out = opt("-i"), opt("-o")
B, T, total = int(opt("-b")), int(opt("-t")), int(opt("-d"))
max_steps, val_every, ckpt_every = int(opt("-x")), int(opt("-v")), int(opt("-n"))
resume, keep, tlimit = int(opt("-y", "0")), int(opt("--keep-ckpts", "0")), float(opt("--time-limit", "0"))
world, rank, sync = int(opt("--nproc", "1")), int(opt("--rank", "0")), opt("--sync-dir")
if total % (B * T * world):
    print(f"total batch {total} is not a multiple of B*T*nproc = {B*T*world}", file=sys.stderr); sys.exit(1)
accum = total // (B * T * world)
step_s = float(os.environ.get("FAKE_STEP_S", "0.2"))
for f in (opt("-j"), opt("--token-bytes")):
    if f and not os.path.exists(f):
        print(f"Error: cannot open {f}", file=sys.stderr); sys.exit(1)
shards = sorted(glob.glob(pat))
if not shards: print("Error: no files found matching the pattern"); sys.exit(1)
if rank != 0:
    sys.stdout = open(os.devnull, "w")
if world > 1:
    if not sync or not os.path.isdir(sync):
        print(f"cannot write {sync}/nccl_id.bin.tmp", file=sys.stderr); sys.exit(1)
    idp = os.path.join(sync, "nccl_id.bin")
    if os.environ.get("FAKE_NCCL_HANG") == "1":
        time.sleep(10 ** 6)
    if rank == 0:
        open(idp + ".tmp", "w").write(str(os.getpid())); os.replace(idp + ".tmp", idp)
    else:
        for _ in range(600):
            if os.path.exists(idp): break
            time.sleep(0.1)
        else:
            print(f"rank {rank}: no {idp}", file=sys.stderr); sys.exit(1)
    print(f"rank {rank}/{world}: NCCL ready")
def ntok(p): return struct.unpack("<3i", open(p, "rb").read(12))[2]
sample = B * T * world
nsamp = [(ntok(p) - 1) // sample for p in shards]
print("device: FAKE (box test)\ngemm check: max abs error 0.00e+00\nfused loss check: max |dloss| 0 (loss ~13.68), max |dgrad| 0 of 1")
print(f"B={B} T={T} nproc={world} total_batch={total} accum={accum}")
if out:   # create_dir_if_not_exists with the mkdir -p fix
    os.makedirs(out, exist_ok=True)
start, sh, si = 0, 0, 0
if resume:
    dones = sorted(int(os.path.basename(p)[5:]) for p in glob.glob(os.path.join(out, "DONE_*")))
    if dones:
        h = struct.unpack("<256i", open(os.path.join(out, f"state_{dones[-1]:08d}.bin"), "rb").read(1024))
        if not (h[0] == 20241115 and h[5] == B and h[6] == T):
            print("bad state file", file=sys.stderr); sys.exit(1)
        start, sh, si = h[2], h[3], h[4]
        rw = h[7] or 1
        if rw != world:
            si = si * rw // world
            print(f"checkpoint was nproc={rw}, now nproc={world}: data cursor rescaled")
        if sh >= len(shards):
            print(f"Segmentation fault (resume shard {sh} but only {len(shards)} shards)", file=sys.stderr); sys.exit(139)
        print(f"resumed from step {start}")
print(f"steps={max_steps} warmup=40 scheduler=warmdown shards={len(shards)}")
sys.stdout.flush()
trace = open(os.environ["FAKE_TRACE"], "a") if os.environ.get("FAKE_TRACE") and rank == 0 else None
def save(done):
    if rank != 0: return
    print(f"writing {out}/model_{done:08d}.bin")
    for kind, hdr in (("model", [20241114, 1]), ("state", [20241115, 1, done, sh, si, B, T, world])):
        pth = f"{out}/{kind}_{done:08d}.bin"
        with open(pth + ".tmp", "wb") as f:
            f.write(struct.pack("<256i", *(hdr + [0] * (256 - len(hdr))))); f.write(os.urandom(4096))
        os.replace(pth + ".tmp", pth)
    open(f"{out}/DONE_{done:08d}", "w").close()
    ds = sorted(int(os.path.basename(p)[5:]) for p in glob.glob(os.path.join(out, "DONE_*")))
    for old in ds[:-keep] if keep > 0 else []:
        for k in ("DONE_%08d", "model_%08d.bin", "state_%08d.bin"):
            try: os.remove(os.path.join(out, k % old))
            except FileNotFoundError: pass
    sys.stdout.flush()
crash = os.environ.get("FAKE_CRASH_AT")
crash1 = os.environ.get("FAKE_RANK1_CRASH_AT")
t0 = time.time(); last_saved = start if resume and start else -1
for step in range(start, max_steps):
    for _ in range(accum):
        if si >= nsamp[sh]:
            sh, si = (0, 0) if sh == len(shards) - 1 else (sh + 1, 0)
        if trace: trace.write(f"{step} {os.path.basename(shards[sh])} {si} {world}\n")
        si += 1
    if trace: trace.flush()
    time.sleep(step_s)
    if rank == 0 and crash and step + 1 == int(crash) and not os.path.exists(out + "/.crashed"):
        open(out + "/.crashed", "w").close(); print("CUDA error: fake crash"); sys.stdout.flush(); sys.exit(1)
    if rank == 1 and crash1 and step + 1 == int(crash1) and not os.path.exists(out + "/.crashed1"):
        open(out + "/.crashed1", "w").close(); print("rank 1: CUDA error: fake crash", file=sys.stderr); sys.exit(1)
    loss = 3 + 7 * math.exp(-step / 20)
    print(f"step {step + 1}/{max_steps}  loss {loss:.4f}  lr_mult 1.000  {step_s:.2f}s  {int(total / step_s)} tok/s  eta 0.1h"); sys.stdout.flush()
    if val_every > 0 and ((step + 1) % val_every == 0 or step + 1 == max_steps):
        print(f"  val loss {loss + 0.01:.4f}  bpb {loss / 3.2:.4f}  (4.615 bytes/token)")
    if ckpt_every > 0 and (step + 1) % ckpt_every == 0: save(step + 1); last_saved = step + 1
    if tlimit > 0 and step + 1 < max_steps and time.time() - t0 >= tlimit:
        if last_saved != step + 1: save(step + 1)
        print(f"time limit {tlimit:.0f}s reached after step {step + 1}; rerun with -y 1 to continue"); sys.exit(0)
if last_saved != max_steps: save(max_steps)
print("done")
