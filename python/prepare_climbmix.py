"""Tokenize ClimbMix into llm.c shard files (uint16 tokens, 256-int header).

Each document is prefixed with the BOS id, matching nanochat and eval_bpb.py.
Shards stay under 100M tokens so the loader's 32-bit file offset does not overflow.
"""

import argparse
import struct
from array import array
from pathlib import Path

import numpy as np


def write_shard(path, tokens):
    tokens = np.frombuffer(tokens, dtype=np.uint16) if isinstance(tokens, array) else np.asarray(tokens, dtype=np.uint16)
    header = [0] * 256
    header[0] = 20240520
    header[1] = 1
    header[2] = int(tokens.size)
    with open(path, "wb") as f:
        f.write(struct.pack("<256I", *header))
        f.write(tokens.tobytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, help="directory from train_tokenizer.py, or 'gpt2'")
    p.add_argument("--out", default="data/climbmix")
    p.add_argument("--target-tokens", type=int, default=1_400_000_000)
    p.add_argument("--val-tokens", type=int, default=262144)
    p.add_argument("--shard-tokens", type=int, default=100_000_000)
    p.add_argument("--dataset", default="karpathy/climbmix-400b-shuffle")
    p.add_argument("--batch-docs", type=int, default=512)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    from datasets import load_dataset

    if args.tokenizer == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        bos = enc.eot_token

        def encode_batch(texts):
            return enc.encode_ordinary_batch(texts)
    else:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
        bos = tok.bos_token_id

        backend = tok.backend_tokenizer
        if max(tok.get_vocab().values()) > 65535:
            raise SystemExit("token id does not fit in uint16; keep vocab <= 65535")

        def encode_batch(texts):
            # Rust tokenizer, all cores, no BOS template (we add BOS ourselves).
            return [e.ids for e in backend.encode_batch(texts, add_special_tokens=False)]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("train_*.bin"))
    if existing and (out / "val.bin").exists() and not args.overwrite:
        have = sum((p.stat().st_size - 1024) // 2 for p in existing)
        if have >= args.target_tokens * 0.99:
            print(f"{out} already has {have:,} train tokens; skipping (--overwrite to redo)")
            return
    ds = load_dataset(args.dataset, split="train", streaming=True)
    total = 0
    val_buf = array("H")
    val_left = args.val_tokens
    shard_i = 0
    # array('H') is 2 bytes per token. A Python list of ints is ~36 bytes per
    # token, which is what used to push a 100M-token shard past 3 GB of RAM.
    shard_buf = array("H")

    def flush_train():
        nonlocal shard_i, shard_buf
        if not shard_buf:
            return
        path = out / f"train_{shard_i:05d}.bin"
        write_shard(path, shard_buf)
        print(f"wrote {path} ({len(shard_buf):,} tokens, {total:,} total)", flush=True)
        shard_i += 1
        shard_buf = array("H")

    def batches():
        batch = []
        for row in ds:
            text = row.get("text", "")
            if text:
                batch.append(text)
            if len(batch) >= args.batch_docs:
                yield batch
                batch = []
        if batch:
            yield batch

    done = False
    for texts in batches():
        for ids in encode_batch(texts):
            ids = [bos] + ids
            if val_left > 0:
                take = ids[:val_left]
                val_buf.extend(take)
                val_left -= len(take)
                ids = ids[len(take):]
                if val_left <= 0:
                    write_shard(out / "val.bin", val_buf)
                    print(f"wrote val.bin ({len(val_buf):,} tokens)", flush=True)
            if ids:
                shard_buf.extend(ids)
                total += len(ids)
                if len(shard_buf) >= args.shard_tokens:
                    flush_train()
            if total >= args.target_tokens:
                done = True
                break
        if done:
            break
    flush_train()
    print(f"train tokens {total:,}")


if __name__ == "__main__":
    import os
    import sys
    main()
    # The streaming reader leaves threads behind; skip interpreter teardown so
    # Python 3.12+ does not abort after the shards are already written.
    sys.stdout.flush()
    os._exit(0)
