"""Write a TinyForge model as a GGUF for llama.cpp.

Stock llama.cpp has no architecture with this exact block (parameter-free
RMSNorm, QK-norm after RoPE, non-gated ReLU^2 MLP, embedding RMSNorm, untied
head, logit softcap). nanochat has the same block, and the `nanochat` branch of
https://github.com/ulanch/llama.cpp adds it as arch "nanochat". This script
targets that arch:

- parameter-free norms are written as all-ones weights
- the QK-norm scale (1.2) is written as the q/k norm weight, which is exact
  because RMSNorm multiplies by its weight
- RoPE uses nanochat's sign, which that branch already handles
- the tokenizer is GPT-2 style byte-level BPE (pre-tokenizer "gpt-2")

Input is either an HF folder from export_hf.py / the SFT script, or a raw
trainer checkpoint plus --tokenizer.

    python python/convert_tinyforge_to_gguf.py --model d12-hf --out d12-f32.gguf
    llama-quantize d12-f32.gguf d12-q8_0.gguf Q8_0

Use f32 or bf16 as the source for quantizing, not f16: ReLU^2 activations can
overflow fp16.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_hf(model_dir):
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    if (model_dir / "model.safetensors").exists():
        from safetensors.torch import load_file
        sd = load_file(str(model_dir / "model.safetensors"))
    else:
        sd = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    sd = {k[len("model."):] if k.startswith("model.") else k: v.float() for k, v in sd.items()}
    c = dict(
        vocab=cfg["vocab_size"], padded_vocab=cfg.get("padded_vocab_size") or cfg["vocab_size"],
        max_seq=cfg.get("n_positions", 1024), n_embd=cfg["n_embd"], n_layer=cfg["n_layer"],
        n_head=cfg["n_head"], rope=cfg.get("rope", True), rmsnorm=cfg.get("rmsnorm", True),
        qknorm=cfg.get("qknorm", True), relu2=cfg.get("relu2", True), untie=cfg.get("untie", True),
        bias=cfg.get("bias", False), embed_norm=cfg.get("embed_norm", True),
        softcap=cfg.get("softcap", 15.0), qk_scale=cfg.get("qk_scale", 1.2),
        rope_base=cfg.get("rope_base", 100000.0),
    )
    return c, sd


def load_ckpt(path):
    from export_hf import _load, _state_dict
    cfg, w = _load(path)
    sd = {k[len("model."):]: v.float() for k, v in _state_dict(cfg, w).items()}
    return cfg, sd


def vocab_from_tokenizer(tok_dir, n_rows):
    import gguf
    tj = json.loads((Path(tok_dir) / "tokenizer.json").read_text())
    vocab = dict(tj["model"]["vocab"])
    for a in tj.get("added_tokens", []):
        vocab[a["content"]] = a["id"]
    special = {a["content"] for a in tj.get("added_tokens", []) if a.get("special")}
    n = max(vocab.values()) + 1
    if n > n_rows:
        raise SystemExit(f"tokenizer has {n} ids but the model has {n_rows} embedding rows")
    tokens = [f"<|pad_{i}|>" for i in range(n_rows)]
    types = [int(gguf.TokenType.UNUSED)] * n_rows
    for t, i in vocab.items():
        tokens[i] = t
        types[i] = int(gguf.TokenType.CONTROL if t in special else gguf.TokenType.NORMAL)
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"].get("merges", [])]
    return tokens, types, merges, vocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="HF folder (config.json + weights [+ tokenizer.json])")
    p.add_argument("--checkpoint", help="raw model_*.bin from train_tinyforge")
    p.add_argument("--tokenizer", help="tokenizer folder (default: --model)")
    p.add_argument("--out", required=True)
    p.add_argument("--outtype", default="f32", choices=["f32", "bf16"])
    p.add_argument("--name", default="tinyforge")
    p.add_argument("--ctx", type=int, default=0, help="context length to record (default: training seq len)")
    args = p.parse_args()
    import gguf

    if args.model:
        cfg, sd = load_hf(args.model)
    elif args.checkpoint:
        cfg, sd = load_ckpt(args.checkpoint)
    else:
        raise SystemExit("pass --model or --checkpoint")
    tok_dir = args.tokenizer or args.model
    if not tok_dir or not (Path(tok_dir) / "tokenizer.json").exists():
        raise SystemExit("need a tokenizer.json (--tokenizer)")

    need = dict(rope=True, rmsnorm=True, qknorm=True, relu2=True, untie=True, bias=False, embed_norm=True)
    bad = [k for k, v in need.items() if bool(cfg[k]) != v]
    if bad:
        raise SystemExit(f"arch 'nanochat' cannot express this model (flags {bad}); export modern-preset models only")

    C, L, NH = cfg["n_embd"], cfg["n_layer"], cfg["n_head"]
    HS = C // NH
    Vp = cfg["padded_vocab"]
    tokens, types, merges, vocab = vocab_from_tokenizer(tok_dir, Vp)
    bos = vocab.get("<|bos|>", 0)
    eos = vocab.get("<|assistant_end|>", bos)

    w = gguf.GGUFWriter(args.out, "nanochat")
    w.add_name(args.name)
    w.add_context_length(args.ctx or cfg["max_seq"])
    w.add_embedding_length(C)
    w.add_block_count(L)
    w.add_feed_forward_length(4 * C)
    w.add_head_count(NH)
    w.add_head_count_kv(NH)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_key_length(HS)
    w.add_value_length(HS)
    w.add_rope_dimension_count(HS)
    w.add_rope_freq_base(float(cfg["rope_base"]))
    if cfg["softcap"] and cfg["softcap"] > 0:
        w.add_final_logit_softcapping(float(cfg["softcap"]))
    w.add_file_type(gguf.LlamaFileType.ALL_F32 if args.outtype == "f32" else gguf.LlamaFileType.MOSTLY_BF16)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_bos_token_id(bos)
    w.add_eos_token_id(eos)
    w.add_add_bos_token(True)
    w.add_add_eos_token(False)

    def add(name, t, keep_f32=False):
        t = t.detach().float().contiguous()
        if keep_f32 or args.outtype == "f32" or t.ndim == 1:
            w.add_tensor(name, t.numpy().astype(np.float32))
        else:
            w.add_tensor(name, t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16),
                         raw_dtype=gguf.GGMLQuantizationType.BF16)

    ones = torch.ones(C)
    add("token_embd.weight", sd["wte.weight"][:Vp])
    add("token_embd_norm.weight", ones, True)
    add("output_norm.weight", ones, True)
    add("output.weight", sd["lm_head.weight"][:Vp])
    qk = torch.full((HS,), float(cfg["qk_scale"]))
    for i in range(L):
        b = f"blk.{i}"
        qkv = sd[f"qkv.{i}.weight"]
        add(f"{b}.attn_norm.weight", ones, True)
        add(f"{b}.attn_q.weight", qkv[0:C])
        add(f"{b}.attn_k.weight", qkv[C:2 * C])
        add(f"{b}.attn_v.weight", qkv[2 * C:3 * C])
        add(f"{b}.attn_q_norm.weight", qk, True)
        add(f"{b}.attn_k_norm.weight", qk, True)
        add(f"{b}.attn_output.weight", sd[f"att_proj.{i}.weight"])
        add(f"{b}.ffn_norm.weight", ones, True)
        add(f"{b}.ffn_up.weight", sd[f"fc.{i}.weight"])
        add(f"{b}.ffn_down.weight", sd[f"fc_proj.{i}.weight"])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}  (arch nanochat, L={L} C={C} heads={NH} vocab rows={Vp})")


if __name__ == "__main__":
    main()
