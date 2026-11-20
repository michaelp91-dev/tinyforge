"""Fake prepare_climbmix.py (box test): deterministic shards in llm.c format, same 'wrote ...' lines."""
import argparse, os, struct, time
import numpy as np
p = argparse.ArgumentParser()
p.add_argument("--tokenizer"); p.add_argument("--out"); p.add_argument("--target-tokens", type=int)
p.add_argument("--val-tokens", type=int); p.add_argument("--shard-tokens", type=int); p.add_argument("--overwrite", action="store_true")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
delay = float(os.environ.get("FAKE_SHARD_S", "0.5"))
def write(path, n, seed):
    toks = (np.arange(n, dtype=np.int64) * 7 + seed) % 32768
    with open(path, "wb") as f:
        f.write(struct.pack("<256i", *([20240520, 1, n] + [0] * 253))); f.write(toks.astype(np.uint16).tobytes())
time.sleep(delay)
write(os.path.join(a.out, "val.bin"), a.val_tokens, 1)
print(f"wrote val.bin ({a.val_tokens:,} tokens)", flush=True)
total, i = 0, 0
while total < a.target_tokens:
    n = min(a.shard_tokens, a.target_tokens - total)
    time.sleep(delay)
    pth = os.path.join(a.out, f"train_{i:05d}.bin")
    write(pth, n, 100 + i)
    total += n
    print(f"wrote {pth} ({n:,} tokens, {total:,} total)", flush=True)
    i += 1
print(f"train tokens {total:,}")
