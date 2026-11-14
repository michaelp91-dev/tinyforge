"""ClimbMix val bits-per-byte.

Same shard and reduction as the uploaded eval_bpb.py / nanochat:
nats on target tokens, divided by ln(2) * target-token bytes.
Document separators (GPT-2 EOT, or TinyForge <|bos|>) count as 0 bytes.

GPT-2 folders use tiktoken. TinyForge folders (model_type tinyforge) load the
exported tokenizer and modeling_tinyforge.py via trust_remote_code.
"""

import math
import os
import time

import numpy as np
import pyarrow.parquet as pq
import requests
import torch
import torch.nn.functional as F

MODEL_DIR = os.environ.get("MODEL_DIR", "/content/drive/MyDrive/llm.c/models/d6-hf")
EVAL_TOKENS = int(os.environ.get("EVAL_TOKENS", 80 * 524288))
B = int(os.environ.get("B", 16))
T = int(os.environ.get("T", 1024))
VAL_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main/shard_06542.parquet"
VAL_PATH = os.environ.get("VAL_PATH", "/content/climbmix_val_shard_06542.parquet")


def download_val():
    if os.path.exists(VAL_PATH):
        return
    print("Downloading NanoChat val shard...")
    with requests.get(VAL_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(VAL_PATH + ".tmp", "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(VAL_PATH + ".tmp", VAL_PATH)


def load_gpt2():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    bos = enc.eot_token
    token_bytes = torch.tensor(
        [len(enc.decode_single_token_bytes(i)) for i in range(enc.n_vocab)], dtype=torch.int64)
    token_bytes[bos] = 0

    def encode_doc(text):
        return [bos] + enc.encode_ordinary(text)

    return encode_doc, token_bytes, enc.n_vocab


def load_tinyforge(model_dir):
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(model_dir)
    npy = os.path.join(model_dir, "token_bytes.npy")
    if os.path.exists(npy):
        counts = np.load(npy)
    else:
        from train_tokenizer import BOS, token_nbytes
        raw = tok.backend_tokenizer
        counts = np.array([
            0 if raw.id_to_token(i) == BOS else token_nbytes(raw.id_to_token(i), {BOS})
            for i in range(tok.vocab_size)
        ], dtype=np.int64)
    token_bytes = torch.tensor(counts, dtype=torch.int64)
    bos = tok.bos_token_id

    def encode_doc(text):
        return [bos] + tok.encode(text, add_special_tokens=False)

    return encode_doc, token_bytes, int(token_bytes.numel())


def main():
    download_val()
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tiny = getattr(cfg, "model_type", "") == "tinyforge"
    if tiny:
        encode_doc, token_bytes, nvocab = load_tinyforge(MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)
    else:
        encode_doc, token_bytes, nvocab = load_gpt2()
        kwargs = {}
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, attn_implementation="sdpa")
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    need = B * T * math.ceil(EVAL_TOKENS / (B * T)) + 1
    toks = []
    pf = pq.ParquetFile(VAL_PATH)
    for rg in range(pf.num_row_groups):
        for text in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
            toks.extend(encode_doc(text))
        if len(toks) >= need:
            break
    toks = torch.tensor(toks[:need], dtype=torch.long)
    steps = (len(toks) - 1) // (B * T)
    print(f"Eval tokens: {steps * B * T:,} ({steps} batches of {B}x{T})")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = model.to(device=dev, dtype=dtype).eval()
    tb = token_bytes.to(dev)
    nats, nbytes, t0 = 0.0, 0, time.time()
    with torch.no_grad():
        for s in range(steps):
            chunk = toks[s * B * T:(s + 1) * B * T + 1].to(dev)
            x, y = chunk[:-1].view(B, T), chunk[1:].view(B, T)
            logits = model(x).logits.float()
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
            nb = tb[y.reshape(-1)]
            nats += (loss * (nb > 0)).sum().item()
            nbytes += nb.sum().item()
            if (s + 1) % 50 == 0:
                print(f"  {s + 1}/{steps}  bpb so far {nats / (math.log(2) * nbytes):.4f}")
    bpb = nats / (math.log(2) * nbytes)
    name = os.path.basename(MODEL_DIR.rstrip("/")).replace("-hf", "")
    print(f"\n{name} ClimbMix val bpb: {bpb:.4f}")
    print(f"bytes evaluated: {nbytes:,}   time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
