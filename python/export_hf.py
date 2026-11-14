"""Export a TinyForge checkpoint to a Hugging Face folder.

Stock GPT-2 flags (no RoPE, affine LayerNorm, tanh-GELU, biases, tied
embeddings, no softcap) are written as a normal GPT-2 model so the original
bits-per-byte script can score them. Every other configuration is written as
TinyForgeForCausalLM (trust_remote_code) with this repo's modeling file copied
into the folder.

Checkpoint layout matches src/train_tinyforge.cu register_model().
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling_tinyforge import TinyForge

MAGIC = 20241114


def _f32(i):
    return np.array([i], dtype=np.int32).view(np.float32)[0].item()


def _load(path):
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32).copy()
        if int(header[0]) != MAGIC:
            raise SystemExit(f"{path} is not a TinyForge checkpoint (magic {int(header[0])})")
        flags = int(header[8])
        cfg = dict(
            max_seq=int(header[2]), vocab=int(header[3]), n_layer=int(header[4]),
            n_head=int(header[5]), n_embd=int(header[6]), padded_vocab=int(header[7]),
            rope=bool(flags & 1), rmsnorm=bool(flags & 2), qknorm=bool(flags & 4),
            relu2=bool(flags & 8), untie=bool(flags & 16), bias=bool(flags & 32),
            embed_norm=bool(flags & 64), muon=bool(flags & 128),
            softcap=float(_f32(int(header[9]))), qk_scale=float(_f32(int(header[10]))),
            rope_base=float(_f32(int(header[11]))), prec=int(header[12]),
            n_tensors=int(header[13]),
        )
        def take(shape):
            n = int(np.prod(shape))
            if cfg["prec"] == 2:
                raw = np.frombuffer(f.read(n * 2), dtype=np.int16).copy()
                t = torch.from_numpy(raw).view(torch.bfloat16).to(torch.float32)
            elif cfg["prec"] == 0:
                raw = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
                t = torch.from_numpy(raw)
            else:
                raise SystemExit(f"unsupported checkpoint precision code {cfg['prec']}")
            return t.reshape(shape).contiguous()

        C, L, Vp, T = cfg["n_embd"], cfg["n_layer"], cfg["padded_vocab"], cfg["max_seq"]
        weights = {"wte": take((Vp, C))}
        if not cfg["rope"]:
            weights["wpe"] = take((T, C))
        if cfg["untie"]:
            weights["lm_head"] = take((Vp, C))
        if not cfg["rmsnorm"]:
            weights["lnfw"] = take((C,))
            weights["lnfb"] = take((C,))
        for i in range(L):
            if not cfg["rmsnorm"]:
                weights[f"ln1w.{i}"] = take((C,))
                weights[f"ln1b.{i}"] = take((C,))
            weights[f"qkvw.{i}"] = take((3 * C, C))
            if cfg["bias"]:
                weights[f"qkvb.{i}"] = take((3 * C,))
            weights[f"attw.{i}"] = take((C, C))
            if cfg["bias"]:
                weights[f"attb.{i}"] = take((C,))
            if not cfg["rmsnorm"]:
                weights[f"ln2w.{i}"] = take((C,))
                weights[f"ln2b.{i}"] = take((C,))
            weights[f"fcw.{i}"] = take((4 * C, C))
            if cfg["bias"]:
                weights[f"fcb.{i}"] = take((4 * C,))
            weights[f"fpw.{i}"] = take((C, 4 * C))
            if cfg["bias"]:
                weights[f"fpb.{i}"] = take((C,))
        rest = f.read()
        if rest:
            raise SystemExit(f"checkpoint has {len(rest)} trailing bytes; flag/layout mismatch")
    return cfg, weights


def _is_gpt2(cfg):
    return (not cfg["rope"] and not cfg["rmsnorm"] and not cfg["qknorm"] and not cfg["relu2"]
            and not cfg["untie"] and cfg["bias"] and not cfg["embed_norm"] and cfg["softcap"] == 0)


def _state_dict(cfg, w):
    L = cfg["n_layer"]
    sd = {"model.wte.weight": w["wte"]}
    if "wpe" in w:
        sd["model.wpe.weight"] = w["wpe"]
    if "lm_head" in w:
        sd["model.lm_head.weight"] = w["lm_head"]
    if "lnfw" in w:
        sd["model.ln_f.weight"] = w["lnfw"]
        sd["model.ln_f.bias"] = w["lnfb"]
    for i in range(L):
        if f"ln1w.{i}" in w:
            sd[f"model.ln1.{i}.weight"] = w[f"ln1w.{i}"]
            sd[f"model.ln1.{i}.bias"] = w[f"ln1b.{i}"]
            sd[f"model.ln2.{i}.weight"] = w[f"ln2w.{i}"]
            sd[f"model.ln2.{i}.bias"] = w[f"ln2b.{i}"]
        sd[f"model.qkv.{i}.weight"] = w[f"qkvw.{i}"]
        sd[f"model.att_proj.{i}.weight"] = w[f"attw.{i}"]
        sd[f"model.fc.{i}.weight"] = w[f"fcw.{i}"]
        sd[f"model.fc_proj.{i}.weight"] = w[f"fpw.{i}"]
        if cfg["bias"]:
            sd[f"model.qkv.{i}.bias"] = w[f"qkvb.{i}"]
            sd[f"model.att_proj.{i}.bias"] = w[f"attb.{i}"]
            sd[f"model.fc.{i}.bias"] = w[f"fcb.{i}"]
            sd[f"model.fc_proj.{i}.bias"] = w[f"fpb.{i}"]
    return sd


def _gpt2_state_dict(cfg, w):
    """Map our (out, in) matrices onto GPT-2 Conv1D (in, out) weights."""
    V, C, L = cfg["vocab"], cfg["n_embd"], cfg["n_layer"]
    sd = {
        "transformer.wte.weight": w["wte"][:V].contiguous(),
        "transformer.wpe.weight": w["wpe"],
        "lm_head.weight": w["wte"][:V].contiguous(),
        "transformer.ln_f.weight": w["lnfw"],
        "transformer.ln_f.bias": w["lnfb"],
    }
    for i in range(L):
        p = f"transformer.h.{i}"
        sd[f"{p}.ln_1.weight"] = w[f"ln1w.{i}"]
        sd[f"{p}.ln_1.bias"] = w[f"ln1b.{i}"]
        sd[f"{p}.attn.c_attn.weight"] = w[f"qkvw.{i}"].T.contiguous()
        sd[f"{p}.attn.c_attn.bias"] = w[f"qkvb.{i}"]
        sd[f"{p}.attn.c_proj.weight"] = w[f"attw.{i}"].T.contiguous()
        sd[f"{p}.attn.c_proj.bias"] = w[f"attb.{i}"]
        sd[f"{p}.ln_2.weight"] = w[f"ln2w.{i}"]
        sd[f"{p}.ln_2.bias"] = w[f"ln2b.{i}"]
        sd[f"{p}.mlp.c_fc.weight"] = w[f"fcw.{i}"].T.contiguous()
        sd[f"{p}.mlp.c_fc.bias"] = w[f"fcb.{i}"]
        sd[f"{p}.mlp.c_proj.weight"] = w[f"fpw.{i}"].T.contiguous()
        sd[f"{p}.mlp.c_proj.bias"] = w[f"fpb.{i}"]
    return sd


def export(checkpoint, output, tokenizer_dir=None):
    cfg, weights = _load(checkpoint)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    gpt2 = _is_gpt2(cfg)
    if gpt2:
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
            config = GPT2Config(
                vocab_size=cfg["vocab"], n_positions=cfg["max_seq"], n_ctx=cfg["max_seq"],
                n_embd=cfg["n_embd"], n_layer=cfg["n_layer"], n_head=cfg["n_head"],
            )
            model = GPT2LMHeadModel(config)
            model.load_state_dict(_gpt2_state_dict(cfg, weights))
            model.save_pretrained(output, safe_serialization=False)
        except Exception as exc:
            print(f"transformers GPT-2 save failed ({exc}); writing a raw state dict")
            (output / "config.json").write_text(json.dumps({
                "model_type": "gpt2", "architectures": ["GPT2LMHeadModel"],
                "vocab_size": cfg["vocab"], "n_positions": cfg["max_seq"], "n_ctx": cfg["max_seq"],
                "n_embd": cfg["n_embd"], "n_layer": cfg["n_layer"], "n_head": cfg["n_head"],
            }, indent=2))
            torch.save(_gpt2_state_dict(cfg, weights), output / "pytorch_model.bin")
    else:
        sd = _state_dict(cfg, weights)
        # Drop the padded rows from the loss vocab but keep them in the matrices:
        # TinyForge.vocab crops logits, while the embedding is padded_vocab.
        config = {
            "model_type": "tinyforge",
            "architectures": ["TinyForgeForCausalLM"],
            "auto_map": {
                "AutoConfig": "modeling_tinyforge.TinyForgeConfig",
                "AutoModelForCausalLM": "modeling_tinyforge.TinyForgeForCausalLM",
            },
            "vocab_size": cfg["vocab"],
            "padded_vocab_size": cfg["padded_vocab"],
            "n_positions": cfg["max_seq"],
            "n_embd": cfg["n_embd"],
            "n_layer": cfg["n_layer"],
            "n_head": cfg["n_head"],
            "rope": cfg["rope"],
            "rmsnorm": cfg["rmsnorm"],
            "qknorm": cfg["qknorm"],
            "relu2": cfg["relu2"],
            "untie": cfg["untie"],
            "bias": cfg["bias"],
            "embed_norm": cfg["embed_norm"],
            "softcap": cfg["softcap"],
            "qk_scale": cfg["qk_scale"],
            "rope_base": cfg["rope_base"],
            "torch_dtype": "float32",
        }
        (output / "config.json").write_text(json.dumps(config, indent=2))
        torch.save(sd, output / "pytorch_model.bin")
        shutil.copy(Path(__file__).resolve().parent / "modeling_tinyforge.py", output / "modeling_tinyforge.py")
        # Sanity: the reference module must accept these weights.
        ref = TinyForge(
            vocab=cfg["vocab"], padded_vocab=cfg["padded_vocab"], n_layer=cfg["n_layer"],
            n_head=cfg["n_head"], n_embd=cfg["n_embd"], max_seq=cfg["max_seq"],
            rope=cfg["rope"], rmsnorm=cfg["rmsnorm"], qknorm=cfg["qknorm"], relu2=cfg["relu2"],
            untie=cfg["untie"], bias=cfg["bias"], embed_norm=cfg["embed_norm"],
            softcap=cfg["softcap"], qk_scale=cfg["qk_scale"], rope_base=cfg["rope_base"],
        )
        ref.load_state_dict({k[len("model."):]: v for k, v in sd.items()})
    if tokenizer_dir:
        src = Path(tokenizer_dir)
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "token_bytes.npy", "token_bytes.bin"):
            if (src / name).exists():
                shutil.copy(src / name, output / name)
    kind = "gpt2" if gpt2 else "tinyforge"
    print(f"wrote {output} ({kind}, L={cfg['n_layer']} C={cfg['n_embd']} V={cfg['vocab']})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--tokenizer", "-t", default=None, help="directory with tokenizer.json and token_bytes.npy")
    args = p.parse_args()
    export(args.input, args.output, args.tokenizer)
