"""Gradient checks: src/tf_ref.c against PyTorch autograd.

Run from the repo root after `make libtf_ref.so` (the Makefile target `test`
does both). A mismatch means the CPU reference and modeling_tinyforge.py
disagree; the CUDA trainer is written to the same formulas.
"""

import ctypes
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from modeling_tinyforge import TinyForge, apply_rope, gelu_tanh, rmsnorm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_PATH = os.path.join(ROOT, "libtf_ref.so")


def load_lib():
    lib = ctypes.CDLL(LIB_PATH)
    lib.tf_rmsnorm_forward.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.tf_rmsnorm_backward.argtypes = [ctypes.POINTER(ctypes.c_float)] * 4 + [ctypes.c_int, ctypes.c_int]
    lib.tf_layernorm_forward.argtypes = [ctypes.POINTER(ctypes.c_float)] * 6 + [ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.tf_layernorm_backward.argtypes = [ctypes.POINTER(ctypes.c_float)] * 8 + [ctypes.c_int, ctypes.c_int]
    lib.tf_rope_forward.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.tf_rope_backward.argtypes = lib.tf_rope_forward.argtypes
    lib.tf_relu2_forward.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.tf_relu2_backward.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int]
    lib.tf_gelu_forward.argtypes = lib.tf_relu2_forward.argtypes
    lib.tf_gelu_backward.argtypes = lib.tf_relu2_backward.argtypes
    lib.tf_softcap_forward.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float]
    lib.tf_softcap_backward.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int, ctypes.c_float]
    lib.tf_linear_forward.argtypes = [ctypes.POINTER(ctypes.c_float)] * 4 + [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.tf_linear_backward.argtypes = [ctypes.POINTER(ctypes.c_float)] * 6 + [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.tf_attn_forward.argtypes = [ctypes.POINTER(ctypes.c_float)] * 5 + [ctypes.c_int] * 4
    lib.tf_attn_backward.argtypes = [ctypes.POINTER(ctypes.c_float)] * 8 + [ctypes.c_int] * 4
    lib.tf_muon_ns5.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    lib.tf_loss.restype = ctypes.c_float
    lib.tf_loss_and_grads.restype = ctypes.c_float
    return lib


def p(t):
    assert t.dtype == torch.float32 and t.is_contiguous()
    return ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_float))


def check(name, a, b, atol=1e-4, rtol=1e-4):
    d = (a - b).abs()
    denom = b.abs().clamp(min=1e-8)
    rel = d / denom
    if d.max().item() > atol and rel.max().item() > rtol:
        raise AssertionError(f"{name}: max abs {d.max().item():.3e} max rel {rel.max().item():.3e}")
    print(f"  ok {name:20s}  max abs {d.max().item():.3e}")


def test_layers(lib):
    torch.manual_seed(0)
    print("layers")
    x = torch.randn(4, 8)
    xr = x.detach().clone().requires_grad_(True)
    y = rmsnorm(xr)
    y.sum().backward()
    out = torch.empty_like(x)
    rstd = torch.empty(4)
    lib.tf_rmsnorm_forward(p(out), p(rstd), p(x), 4, 8, 1e-6)
    check("rmsnorm fwd", out, y.detach())
    dout = torch.randn_like(x)
    dx = torch.empty_like(x)
    lib.tf_rmsnorm_backward(p(dx), p(dout), p(x), p(rstd), 4, 8)
    xr.grad = None
    (rmsnorm(xr) * dout).sum().backward()
    check("rmsnorm bwd", dx, xr.grad)

    w = torch.randn(8)
    b = torch.randn(8)
    xr = x.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    br = b.detach().clone().requires_grad_(True)
    y = torch.nn.functional.layer_norm(xr, (8,), wr, br, 1e-5)
    dout = torch.randn_like(x)
    y.backward(dout)
    out = torch.empty_like(x)
    mean = torch.empty(4)
    rstd = torch.empty(4)
    lib.tf_layernorm_forward(p(out), p(mean), p(rstd), p(x), p(w), p(b), 4, 8, 1e-5)
    check("layernorm fwd", out, y.detach())
    dx = torch.empty_like(x)
    dw = torch.zeros_like(w)
    db = torch.zeros_like(b)
    lib.tf_layernorm_backward(p(dx), p(dw), p(db), p(dout), p(x), p(w), p(mean), p(rstd), 4, 8)
    check("layernorm dx", dx, xr.grad)
    check("layernorm dw", dw, wr.grad)
    check("layernorm db", db, br.grad)

    q = torch.randn(2, 5, 2, 8)
    qr = q.detach().clone().requires_grad_(True)
    y = apply_rope(qr, 100000.0)
    out = torch.empty_like(q)
    lib.tf_rope_forward(p(out), p(q), 2, 5, 2, 8, 100000.0)
    check("rope fwd", out, y.detach())
    dout = torch.randn_like(q)
    dx = torch.empty_like(q)
    lib.tf_rope_backward(p(dx), p(dout), 2, 5, 2, 8, 100000.0)
    qr.grad = None
    (apply_rope(qr, 100000.0) * dout).sum().backward()
    check("rope bwd", dx, qr.grad)

    z = torch.randn(20)
    zr = z.detach().clone().requires_grad_(True)
    y = torch.relu(zr).square()
    out = torch.empty_like(z)
    lib.tf_relu2_forward(p(out), p(z), 20)
    check("relu2 fwd", out, y.detach())
    dout = torch.randn_like(z)
    dx = torch.empty_like(z)
    lib.tf_relu2_backward(p(dx), p(dout), p(z), 20)
    zr.grad = None
    torch.relu(zr).square().backward(dout)
    check("relu2 bwd", dx, zr.grad)

    zr = z.detach().clone().requires_grad_(True)
    y = gelu_tanh(zr)
    lib.tf_gelu_forward(p(out), p(z), 20)
    check("gelu fwd", out, y.detach())
    lib.tf_gelu_backward(p(dx), p(dout), p(z), 20)
    zr.grad = None
    gelu_tanh(zr).backward(dout)
    check("gelu bwd", dx, zr.grad)

    zr = z.detach().clone().requires_grad_(True)
    cap = 15.0
    y = cap * torch.tanh(zr / cap)
    lib.tf_softcap_forward(p(out), p(z), 20, cap)
    check("softcap fwd", out, y.detach())
    lib.tf_softcap_backward(p(dx), p(dout), p(z), 20, cap)
    zr.grad = None
    (cap * torch.tanh(zr / cap)).backward(dout)
    check("softcap bwd", dx, zr.grad)

    xin = torch.randn(3, 5)
    W = torch.randn(4, 5)
    bias = torch.randn(4)
    xr = xin.detach().clone().requires_grad_(True)
    wr = W.detach().clone().requires_grad_(True)
    br = bias.detach().clone().requires_grad_(True)
    y = torch.nn.functional.linear(xr, wr, br)
    out = torch.empty(3, 4)
    lib.tf_linear_forward(p(out), p(xin), p(W), p(bias), 3, 5, 4)
    check("linear fwd", out, y.detach())
    dout = torch.randn(3, 4)
    dx = torch.empty_like(xin)
    dw = torch.zeros_like(W)
    db = torch.zeros_like(bias)
    lib.tf_linear_backward(p(dx), p(dw), p(db), p(dout), p(xin), p(W), 3, 5, 4)
    xr.grad = None
    torch.nn.functional.linear(xr, wr, br).backward(dout)
    check("linear dx", dx, xr.grad)
    check("linear dw", dw, wr.grad)
    check("linear db", db, br.grad)

    B, T, NH, HS = 2, 4, 2, 4
    q = torch.randn(B, T, NH, HS)
    k = torch.randn(B, T, NH, HS)
    v = torch.randn(B, T, NH, HS)
    qr, kr, vr = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    scale = 1.0 / math.sqrt(HS)
    scores = torch.matmul(qr.permute(0, 2, 1, 3), kr.permute(0, 2, 1, 3).transpose(-2, -1)) * scale
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    att_t = torch.softmax(scores.masked_fill(causal, float("-inf")), dim=-1)
    y = torch.matmul(att_t, vr.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    out = torch.empty_like(q)
    att = torch.empty(B, NH, T, T)
    lib.tf_attn_forward(p(out), p(att), p(q), p(k), p(v), B, T, NH, HS)
    check("attn fwd", out, y.detach())
    dout = torch.randn_like(q)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    lib.tf_attn_backward(p(dq), p(dk), p(dv), p(dout), p(q), p(k), p(v), p(att), B, T, NH, HS)
    y.backward(dout)
    check("attn dq", dq, qr.grad)
    check("attn dk", dk, kr.grad)
    check("attn dv", dv, vr.grad)

    g = torch.randn(6, 4)
    ref = muon_ns5_torch(g)
    out = torch.empty_like(g)
    lib.tf_muon_ns5(p(out), p(g), 6, 4)
    check("muon ns tall", out, ref, atol=1e-4, rtol=1e-4)
    g = torch.randn(3, 7)
    ref = muon_ns5_torch(g)
    out = torch.empty_like(g)
    lib.tf_muon_ns5(p(out), p(g), 3, 7)
    check("muon ns wide", out, ref, atol=1e-4, rtol=1e-4)


def muon_ns5_torch(g):
    a, b, c = 3.4445, -4.7750, 2.0315
    transposed = g.size(0) > g.size(1)
    X = g.T.contiguous() if transposed else g.clone()
    X = X / (X.norm() + 1e-7)
    for _ in range(5):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T.contiguous() if transposed else X


class CConfig(ctypes.Structure):
    _fields_ = [
        ("vocab", ctypes.c_int), ("padded_vocab", ctypes.c_int), ("max_seq", ctypes.c_int),
        ("n_layer", ctypes.c_int), ("n_head", ctypes.c_int), ("n_embd", ctypes.c_int),
        ("rope", ctypes.c_int), ("rmsnorm", ctypes.c_int), ("qknorm", ctypes.c_int),
        ("relu2", ctypes.c_int), ("untie", ctypes.c_int), ("bias", ctypes.c_int),
        ("embed_norm", ctypes.c_int),
        ("softcap", ctypes.c_float), ("qk_scale", ctypes.c_float), ("rope_base", ctypes.c_float),
    ]


class CParams(ctypes.Structure):
    _fields_ = [(name, ctypes.POINTER(ctypes.c_float)) for name in [
        "wte", "wpe", "lm_head", "lnfw", "lnfb", "ln1w", "ln1b", "qkvw", "qkvb",
        "attprojw", "attprojb", "ln2w", "ln2b", "fcw", "fcb", "fcprojw", "fcprojb",
    ]]


def _fill(module, gen):
    with torch.no_grad():
        for t in module.parameters():
            t.copy_(torch.empty_like(t).uniform_(-0.1, 0.1, generator=gen))


def main():
    lib = load_lib()
    lib.tf_loss_and_grads.argtypes = [
        ctypes.POINTER(CConfig), ctypes.POINTER(CParams), ctypes.POINTER(CParams),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
    ]
    test_layers(lib)
    for modern in (True, False):
        run_model(lib, modern)
    print("all gradient checks passed")


def run_model(lib, modern):
    label = "modern" if modern else "gpt2"
    print(f"model {label}")
    B, T, L, NH, C, V = 2, 4, 2, 2, 8, 11
    Vp = 16
    gen = torch.Generator().manual_seed(2 if modern else 3)
    m = TinyForge(
        vocab=V, padded_vocab=Vp, n_layer=L, n_head=NH, n_embd=C, max_seq=T,
        rope=modern, rmsnorm=modern, qknorm=modern, relu2=modern, untie=modern,
        bias=not modern, embed_norm=modern, softcap=15.0 if modern else 0.0,
        qk_scale=1.2, rope_base=100000.0,
    )
    _fill(m, gen)
    idx = torch.randint(0, V, (B, T), generator=gen)
    targets = torch.randint(0, V, (B, T), generator=gen)
    loss_t = m(idx, targets)
    loss_t.backward()

    store = {}

    def take(name, tensor):
        if tensor is None:
            store[name] = None
            store["g_" + name] = None
        else:
            store[name] = tensor.detach().float().contiguous().clone()
            store["g_" + name] = torch.zeros_like(store[name])

    take("wte", m.wte.weight)
    take("wpe", None if m.wpe is None else m.wpe.weight)
    take("lm_head", None if m.lm_head is None else m.lm_head.weight)
    if m.ln1 is None:
        for n in ("lnfw", "lnfb", "ln1w", "ln1b", "ln2w", "ln2b"):
            take(n, None)
    else:
        take("lnfw", m.ln_f.weight)
        take("lnfb", m.ln_f.bias)
        take("ln1w", torch.stack([m.ln1[i].weight.detach() for i in range(L)]).contiguous())
        take("ln1b", torch.stack([m.ln1[i].bias.detach() for i in range(L)]).contiguous())
        take("ln2w", torch.stack([m.ln2[i].weight.detach() for i in range(L)]).contiguous())
        take("ln2b", torch.stack([m.ln2[i].bias.detach() for i in range(L)]).contiguous())
    take("qkvw", torch.stack([m.qkv[i].weight.detach() for i in range(L)]).contiguous())
    take("qkvb", torch.stack([m.qkv[i].bias.detach() for i in range(L)]).contiguous() if m.use_bias else None)
    take("attprojw", torch.stack([m.att_proj[i].weight.detach() for i in range(L)]).contiguous())
    take("attprojb", torch.stack([m.att_proj[i].bias.detach() for i in range(L)]).contiguous() if m.use_bias else None)
    take("fcw", torch.stack([m.fc[i].weight.detach() for i in range(L)]).contiguous())
    take("fcb", torch.stack([m.fc[i].bias.detach() for i in range(L)]).contiguous() if m.use_bias else None)
    take("fcprojw", torch.stack([m.fc_proj[i].weight.detach() for i in range(L)]).contiguous())
    take("fcprojb", torch.stack([m.fc_proj[i].bias.detach() for i in range(L)]).contiguous() if m.use_bias else None)

    names = ["wte", "wpe", "lm_head", "lnfw", "lnfb", "ln1w", "ln1b", "qkvw", "qkvb",
             "attprojw", "attprojb", "ln2w", "ln2b", "fcw", "fcb", "fcprojw", "fcprojb"]

    def ptr(name):
        t = store[name]
        return None if t is None else p(t)

    weights = CParams(**{n: ptr(n) for n in names})
    gparams = CParams(**{n: ptr("g_" + n) for n in names})
    cfg = CConfig(
        vocab=V, padded_vocab=Vp, max_seq=T, n_layer=L, n_head=NH, n_embd=C,
        rope=int(modern), rmsnorm=int(modern), qknorm=int(modern), relu2=int(modern),
        untie=int(modern), bias=int(not modern), embed_norm=int(modern),
        softcap=15.0 if modern else 0.0, qk_scale=1.2, rope_base=100000.0,
    )
    idx_i = idx.to(torch.int32).contiguous()
    tgt_i = targets.to(torch.int32).contiguous()
    loss_c = lib.tf_loss_and_grads(
        ctypes.byref(cfg), ctypes.byref(weights), ctypes.byref(gparams),
        ctypes.cast(idx_i.data_ptr(), ctypes.POINTER(ctypes.c_int)),
        ctypes.cast(tgt_i.data_ptr(), ctypes.POINTER(ctypes.c_int)), B, T)
    if abs(loss_c - loss_t.item()) > 1e-4:
        raise AssertionError(f"{label} loss c={loss_c} torch={loss_t.item()}")
    print(f"  ok loss {loss_c:.6f}")

    torch_grads = {
        "wte": m.wte.weight.grad,
        "wpe": None if m.wpe is None else m.wpe.weight.grad,
        "lm_head": None if m.lm_head is None else m.lm_head.weight.grad,
        "qkvw": torch.stack([m.qkv[i].weight.grad for i in range(L)]),
        "attprojw": torch.stack([m.att_proj[i].weight.grad for i in range(L)]),
        "fcw": torch.stack([m.fc[i].weight.grad for i in range(L)]),
        "fcprojw": torch.stack([m.fc_proj[i].weight.grad for i in range(L)]),
    }
    if m.use_bias:
        torch_grads["qkvb"] = torch.stack([m.qkv[i].bias.grad for i in range(L)])
        torch_grads["attprojb"] = torch.stack([m.att_proj[i].bias.grad for i in range(L)])
        torch_grads["fcb"] = torch.stack([m.fc[i].bias.grad for i in range(L)])
        torch_grads["fcprojb"] = torch.stack([m.fc_proj[i].bias.grad for i in range(L)])
    if m.ln1 is not None:
        torch_grads["lnfw"] = m.ln_f.weight.grad
        torch_grads["lnfb"] = m.ln_f.bias.grad
        torch_grads["ln1w"] = torch.stack([m.ln1[i].weight.grad for i in range(L)])
        torch_grads["ln1b"] = torch.stack([m.ln1[i].bias.grad for i in range(L)])
        torch_grads["ln2w"] = torch.stack([m.ln2[i].weight.grad for i in range(L)])
        torch_grads["ln2b"] = torch.stack([m.ln2[i].bias.grad for i in range(L)])
    for name, tg in torch_grads.items():
        if tg is None:
            continue
        check(f"{label} {name}", store["g_" + name], tg.detach().float())


if __name__ == "__main__":
    main()
