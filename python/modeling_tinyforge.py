"""TinyForge decoder, matching src/tf_ref.c and the CUDA trainer.

The class is a plain nn.Module so tests do not need Hugging Face. The
PretrainedConfig subclass at the bottom is what export writes into an HF
folder; AutoModelForCausalLM loads it with trust_remote_code=True.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu_tanh(x):
    """GPT-2 / llm.c tanh approximation, not the erf GELU."""
    k = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(k * (x + 0.044715 * x * x * x)))


def rmsnorm(x, eps=1e-6):
    ms = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(ms + eps)).to(dtype=x.dtype)


def apply_rope(x, base):
    """Rotate-half RoPE. x is (B, T, NH, HS). nanochat sign convention."""
    B, T, NH, HS = x.shape
    half = HS // 2
    i = torch.arange(half, device=x.device, dtype=torch.float32)
    t = torch.arange(T, device=x.device, dtype=torch.float32)
    freq = t[:, None] / (base ** ((2.0 * i) / HS))[None, :]
    cos = torch.cos(freq).to(dtype=x.dtype)[None, :, None, :]
    sin = torch.sin(freq).to(dtype=x.dtype)[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1)


class TinyForge(nn.Module):
    """Flag-gated decoder.

    modern (the recommended preset): RoPE, parameter-free RMSNorm, QK-norm,
    ReLU^2, untied lm_head, no biases, RMSNorm on the token embedding, logit
    softcap. gpt2: learned positions, affine LayerNorm, tanh-GELU, biases,
    tied embeddings.
    """

    def __init__(self, vocab, n_layer, n_head, n_embd, max_seq=1024,
                 padded_vocab=None, rope=True, rmsnorm=True, qknorm=True,
                 relu2=True, untie=True, bias=False, embed_norm=True,
                 softcap=15.0, qk_scale=1.2, rope_base=100000.0):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.vocab = vocab
        self.padded_vocab = padded_vocab or vocab
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.max_seq = max_seq
        self.rope = rope
        self.use_rmsnorm = rmsnorm
        self.qknorm = qknorm
        self.relu2 = relu2
        self.untie = untie
        self.use_bias = bias
        self.embed_norm = embed_norm
        self.softcap = softcap
        self.qk_scale = qk_scale
        self.rope_base = rope_base
        Vp, C, L = self.padded_vocab, n_embd, n_layer
        self.wte = nn.Embedding(Vp, C)
        self.wpe = None if rope else nn.Embedding(max_seq, C)
        self.lm_head = nn.Linear(C, Vp, bias=False) if untie else None
        if not rmsnorm:
            self.ln_f = nn.LayerNorm(C, eps=1e-5)
            self.ln1 = nn.ModuleList([nn.LayerNorm(C, eps=1e-5) for _ in range(L)])
            self.ln2 = nn.ModuleList([nn.LayerNorm(C, eps=1e-5) for _ in range(L)])
        else:
            self.ln_f = None
            self.ln1 = None
            self.ln2 = None
        self.qkv = nn.ModuleList([nn.Linear(C, 3 * C, bias=bias) for _ in range(L)])
        self.att_proj = nn.ModuleList([nn.Linear(C, C, bias=bias) for _ in range(L)])
        self.fc = nn.ModuleList([nn.Linear(C, 4 * C, bias=bias) for _ in range(L)])
        self.fc_proj = nn.ModuleList([nn.Linear(4 * C, C, bias=bias) for _ in range(L)])

    def _norm(self, x, ln):
        if self.use_rmsnorm:
            return rmsnorm(x)
        return ln(x)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx)
        if not self.rope:
            pos = torch.arange(T, device=idx.device)
            x = x + self.wpe(pos)[None, :, :]
        if self.embed_norm:
            x = rmsnorm(x)
        HS = self.n_embd // self.n_head
        for layer in range(self.n_layer):
            h = self._norm(x, None if self.ln1 is None else self.ln1[layer])
            qkv = self.qkv[layer](h).view(B, T, 3, self.n_head, HS)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            if self.rope:
                q = apply_rope(q, self.rope_base)
                k = apply_rope(k, self.rope_base)
            if self.qknorm:
                q = rmsnorm(q) * self.qk_scale
                k = rmsnorm(k) * self.qk_scale
            y = self._attn(q, k, v)
            x = x + self.att_proj[layer](y)
            h = self._norm(x, None if self.ln2 is None else self.ln2[layer])
            h = self.fc[layer](h)
            h = torch.relu(h).square() if self.relu2 else gelu_tanh(h)
            x = x + self.fc_proj[layer](h)
        x = self._norm(x, self.ln_f)
        head_w = self.lm_head.weight if self.untie else self.wte.weight
        logits = F.linear(x, head_w)
        return self._loss_or_logits(logits, targets)

    def _attn(self, q, k, v):
        # q/k/v: (B, T, NH, HS)
        B, T, NH, HS = q.shape
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(HS)
        causal = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        att = torch.softmax(scores, dim=-1)
        y = torch.matmul(att, v.float())
        return y.permute(0, 2, 1, 3).contiguous().view(B, T, NH * HS).to(dtype=q.dtype)

    def _loss_or_logits(self, logits, targets):
        z = logits[..., :self.vocab]
        if self.softcap and self.softcap > 0:
            z = self.softcap * torch.tanh(z / self.softcap)
        if targets is None:
            return z
        return F.cross_entropy(z.reshape(-1, z.size(-1)), targets.reshape(-1))


try:
    from transformers import PretrainedConfig
    from transformers.modeling_utils import PreTrainedModel

    class TinyForgeConfig(PretrainedConfig):
        model_type = "tinyforge"

        def __init__(self, vocab_size=32768, padded_vocab_size=None, n_positions=1024,
                     n_embd=768, n_layer=12, n_head=6, rope=True, rmsnorm=True,
                     qknorm=True, relu2=True, untie=True, bias=False, embed_norm=True,
                     softcap=15.0, qk_scale=1.2, rope_base=100000.0, **kwargs):
            super().__init__(**kwargs)
            self.vocab_size = vocab_size
            self.padded_vocab_size = padded_vocab_size or vocab_size
            self.n_positions = n_positions
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.n_head = n_head
            self.rope = rope
            self.rmsnorm = rmsnorm
            self.qknorm = qknorm
            self.relu2 = relu2
            self.untie = untie
            self.bias = bias
            self.embed_norm = embed_norm
            self.softcap = softcap
            self.qk_scale = qk_scale
            self.rope_base = rope_base

    class _CausalOut:
        def __init__(self, loss, logits):
            self.loss = loss
            self.logits = logits

    class TinyForgeForCausalLM(PreTrainedModel):
        config_class = TinyForgeConfig
        base_model_prefix = "model"

        def __init__(self, config):
            super().__init__(config)
            self.model = TinyForge(
                vocab=config.vocab_size,
                padded_vocab=config.padded_vocab_size,
                n_layer=config.n_layer,
                n_head=config.n_head,
                n_embd=config.n_embd,
                max_seq=config.n_positions,
                rope=bool(config.rope),
                rmsnorm=bool(config.rmsnorm),
                qknorm=bool(config.qknorm),
                relu2=bool(config.relu2),
                untie=bool(config.untie),
                bias=bool(config.bias),
                embed_norm=bool(config.embed_norm),
                softcap=config.softcap,
                qk_scale=config.qk_scale,
                rope_base=config.rope_base,
            )

        def forward(self, input_ids=None, labels=None, attention_mask=None, **kwargs):
            # logits are already cropped to vocab_size and softcapped, which is
            # the tensor the trainer's cross-entropy sees.
            logits = self.model(input_ids, targets=None)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            return _CausalOut(loss, logits)

except ImportError:
    TinyForgeConfig = None
    TinyForgeForCausalLM = None
