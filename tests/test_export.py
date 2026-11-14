"""Round-trip a tiny fp32 checkpoint through export_hf and the reference module."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from export_hf import export
from modeling_tinyforge import TinyForge


def _bits(x):
    return np.array([np.float32(x)]).view(np.int32)[0]


def write_ckpt(path, cfg, arrays):
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20241114
    header[1] = 1
    header[2] = cfg["max_seq"]
    header[3] = cfg["vocab"]
    header[4] = cfg["n_layer"]
    header[5] = cfg["n_head"]
    header[6] = cfg["n_embd"]
    header[7] = cfg["padded_vocab"]
    flags = 0
    for bit, key in enumerate(["rope", "rmsnorm", "qknorm", "relu2", "untie", "bias", "embed_norm"]):
        if cfg[key]:
            flags |= 1 << bit
    header[8] = flags
    header[9] = _bits(cfg["softcap"])
    header[10] = _bits(cfg["qk_scale"])
    header[11] = _bits(cfg["rope_base"])
    header[12] = 0
    header[13] = len(arrays)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        for a in arrays:
            f.write(np.asarray(a, dtype=np.float32).tobytes())


def modern_arrays(cfg, pattern):
    C, L, Vp, T = cfg["n_embd"], cfg["n_layer"], cfg["padded_vocab"], cfg["max_seq"]
    arrays = [pattern((Vp, C))]
    arrays.append(pattern((Vp, C)))  # lm head
    for _ in range(L):
        arrays.append(pattern((3 * C, C)))
        arrays.append(pattern((C, C)))
        arrays.append(pattern((4 * C, C)))
        arrays.append(pattern((C, 4 * C)))
    return arrays


def test_modern_roundtrip():
    cfg = dict(max_seq=8, vocab=20, n_layer=2, n_head=2, n_embd=64, padded_vocab=64,
               rope=True, rmsnorm=True, qknorm=True, relu2=True, untie=True, bias=False,
               embed_norm=True, softcap=15.0, qk_scale=1.2, rope_base=100000.0)
    rng = np.random.default_rng(0)
    arrays = modern_arrays(cfg, lambda shape: rng.standard_normal(shape).astype(np.float32))
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "model.bin")
        out = os.path.join(d, "hf")
        write_ckpt(ckpt, cfg, arrays)
        export(ckpt, out)
        sd = torch.load(os.path.join(out, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        got = sd["model.wte.weight"].numpy()
        if not np.allclose(got, arrays[0]):
            raise AssertionError("wte mismatch after export")
        model = TinyForge(
            vocab=20, padded_vocab=64, n_layer=2, n_head=2, n_embd=64, max_seq=8,
            rope=True, rmsnorm=True, qknorm=True, relu2=True, untie=True, bias=False,
            embed_norm=True, softcap=15.0, qk_scale=1.2, rope_base=100000.0,
        )
        model.load_state_dict({k[len("model."):]: v for k, v in sd.items()})
        idx = torch.randint(0, 20, (2, 8))
        logits = model(idx)
        if logits.shape != (2, 8, 20):
            raise AssertionError(f"logits shape {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise AssertionError("non-finite logits")
    print("export roundtrip ok")


def test_gpt2_export_keys():
    cfg = dict(max_seq=8, vocab=20, n_layer=1, n_head=2, n_embd=64, padded_vocab=64,
               rope=False, rmsnorm=False, qknorm=False, relu2=False, untie=False, bias=True,
               embed_norm=False, softcap=0.0, qk_scale=1.2, rope_base=100000.0)
    rng = np.random.default_rng(1)
    C = 64
    arrays = [
        rng.standard_normal((64, C)).astype(np.float32),  # wte
        rng.standard_normal((8, C)).astype(np.float32),   # wpe
        np.ones(C, dtype=np.float32), np.zeros(C, dtype=np.float32),  # ln_f
        np.ones(C, dtype=np.float32), np.zeros(C, dtype=np.float32),  # ln1
        rng.standard_normal((3 * C, C)).astype(np.float32),
        np.zeros(3 * C, dtype=np.float32),
        rng.standard_normal((C, C)).astype(np.float32),
        np.zeros(C, dtype=np.float32),
        np.ones(C, dtype=np.float32), np.zeros(C, dtype=np.float32),  # ln2
        rng.standard_normal((4 * C, C)).astype(np.float32),
        np.zeros(4 * C, dtype=np.float32),
        rng.standard_normal((C, 4 * C)).astype(np.float32),
        np.zeros(C, dtype=np.float32),
    ]
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "model.bin")
        out = os.path.join(d, "hf")
        write_ckpt(ckpt, cfg, arrays)
        export(ckpt, out)
        weight_files = list(Path(out).glob("*.safetensors"))
        if weight_files:
            from safetensors.torch import load_file
            sd = load_file(str(weight_files[0]))
        else:
            sd = torch.load(os.path.join(out, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        w = sd["transformer.h.0.attn.c_attn.weight"]
        # Conv1D stores (in, out); we stored (out, in).
        if w.shape != (C, 3 * C):
            raise AssertionError(f"c_attn shape {tuple(w.shape)}")
        if not np.allclose(w.numpy(), arrays[6].T):
            raise AssertionError("c_attn transpose mismatch")
    print("gpt2 export ok")


def test_tokenizer_bytes():
    sys.path.insert(0, str(ROOT / "python"))
    from train_tokenizer import train
    texts = ["hello world " * 20, "the cat sat on the mat " * 20, "12345 67890 " * 10] * 30
    with tempfile.TemporaryDirectory() as d:
        train(texts, vocab_size=300, out_dir=d)
        counts = np.load(os.path.join(d, "token_bytes.npy"))
        if counts.min() < 0:
            raise AssertionError("negative byte counts")
        if not os.path.exists(os.path.join(d, "tokenizer.json")):
            raise AssertionError("tokenizer.json missing")
    print("tokenizer smoke ok")


if __name__ == "__main__":
    test_modern_roundtrip()
    test_gpt2_export_keys()
    test_tokenizer_bytes()
