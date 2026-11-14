"""Train a byte-level BPE tokenizer on ClimbMix text.

This is the data-prep half of a nanochat-style tokenizer: a vocabulary trained
on the same distribution as the model, with a document-separator token whose
bytes count as zero in bits-per-byte. The split is GPT-2 byte-level BPE rather
than nanochat's rustbpe regex; matching that regex is listed as a next step.

Writes tokenizer.json (Hugging Face) plus token_bytes.bin / token_bytes.npy.
"""

import argparse
import os
from pathlib import Path

import numpy as np


BOS = "<|bos|>"
# GPT-2 bytes <-> unicode, so a token string can be turned back into raw bytes
# without decoding it as UTF-8 (which corrupts tokens that are not valid text).
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(cs, bs))


UNICODE_TO_BYTE = _bytes_to_unicode()


def token_nbytes(token, specials):
    if token in specials:
        return 0
    try:
        return len(bytes(UNICODE_TO_BYTE[ord(ch)] for ch in token))
    except KeyError:
        return len(token.encode("utf-8"))


def train(texts, vocab_size, out_dir):
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import TemplateProcessing
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size - 1,  # reserve one id for BOS
        special_tokens=[BOS],
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    bos_id = tokenizer.token_to_id(BOS)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS} $A",
        special_tokens=[(BOS, bos_id)],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS,
        unk_token=None,
        model_max_length=8192,
    )
    fast.save_pretrained(out_dir)

    n = tokenizer.get_vocab_size()
    counts = np.zeros(n, dtype=np.int32)
    for i in range(n):
        tok = tokenizer.id_to_token(i)
        counts[i] = 0 if tok == BOS else token_nbytes(tok, {BOS})
    np.save(out_dir / "token_bytes.npy", counts)
    with open(out_dir / "token_bytes.bin", "wb") as f:
        np.array([n], dtype=np.int32).tofile(f)
        counts.tofile(f)
    print(f"vocab {n}  bos_id {bos_id}  mean bytes/token {counts[counts > 0].mean():.2f}")
    print(f"wrote {out_dir}")
    return tokenizer


def _stream_texts(dataset, max_chars, doc_cap):
    n = 0
    for row in dataset:
        text = row.get("text", "") if isinstance(row, dict) else str(row)
        if len(text) > doc_cap:
            text = text[:doc_cap]
        if not text:
            continue
        n += len(text)
        yield text
        if n >= max_chars:
            return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/tokenizer")
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--max-chars", type=int, default=200_000_000)
    p.add_argument("--doc-cap", type=int, default=10_000)
    p.add_argument("--dataset", default="karpathy/climbmix-400b-shuffle")
    args = p.parse_args()
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train", streaming=True)
    train(_stream_texts(ds, args.max_chars, args.doc_cap), args.vocab_size, args.out)


if __name__ == "__main__":
    main()
