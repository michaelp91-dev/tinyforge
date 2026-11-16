"""Tokenize ClimbMix into llm.c shard files (uint16 tokens, 256-int header).

Each document is prefixed with the BOS id, matching nanochat and eval_bpb.py.
Shards stay under 100M tokens so the loader's 32-bit file offset does not overflow.
"""

import argparse
import struct
from pathlib import Path

import numpy as np


def write_shard(path, tokens):
    tokens = np.asarray(tokens, dtype=np.uint16)
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
    args = p.parse_args()
    from datasets import load_dataset

    if args.tokenizer == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        bos = enc.eot_token

        def encode(text):
            return [bos] + enc.encode(text, disallowed_special=())
    else:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
        bos = tok.bos_token_id

        def encode(text):
            ids = tok.encode(text, add_special_tokens=False)
            return [bos] + ids

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.dataset, split="train", streaming=True)
    buf = []
    total = 0
    val_left = args.val_tokens
    shard_i = 0
    shard_buf = []

    def flush_train():
        nonlocal shard_i, shard_buf
        if not shard_buf:
            return
        path = out / f"train_{shard_i:05d}.bin"
        write_shard(path, shard_buf)
        print(f"wrote {path} ({len(shard_buf):,} tokens)")
        shard_i += 1
        shard_buf = []

    for row in ds:
        text = row.get("text", "")
        if not text:
            continue
        ids = encode(text)
        # uint16 cannot represent ids >= 65536
        if ids and max(ids) > 65535:
            raise SystemExit("token id does not fit in uint16; keep vocab <= 65535")
        if val_left > 0:
            take = ids[:val_left]
            buf.extend(take)
            val_left -= len(take)
            ids = ids[len(take):]
            if val_left <= 0 and buf:
                write_shard(out / "val.bin", buf)
                print(f"wrote val.bin ({len(buf):,} tokens)")
                buf = []
        if ids:
            shard_buf.extend(ids)
            total += len(ids)
            if len(shard_buf) >= args.shard_tokens:
                flush_train()
        if total >= args.target_tokens:
            break
    flush_train()
    print(f"train tokens {total:,}")


if __name__ == "__main__":
    main()
