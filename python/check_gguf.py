"""Compare a TinyForge HF folder against the same model in a running llama-server.

    llama-server -m model.gguf --port 8099 &
    python python/check_gguf.py --model d12-hf --url http://127.0.0.1:8099

Checks that both sides tokenize the prompts the same way and that the
next-token top-5 probabilities agree.
"""

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_tinyforge_to_gguf import load_hf
from modeling_tinyforge import TinyForge

PROMPTS = [
    "The capital of France is",
    "turn the case lights off",
    "what's the weather tomorrow in Stillwater?",
]


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8099")
    p.add_argument("--tol", type=float, default=2e-3)
    args = p.parse_args()
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(args.model)
    cfg, sd = load_hf(args.model)
    m = TinyForge(vocab=cfg["vocab"], n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
                  max_seq=cfg["max_seq"], padded_vocab=cfg["padded_vocab"], softcap=cfg["softcap"],
                  qk_scale=cfg["qk_scale"], rope_base=cfg["rope_base"])
    m.load_state_dict(sd)
    m.eval()
    worst = 0.0
    for text in PROMPTS:
        ids = [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)
        srv = post(args.url + "/tokenize", {"content": text, "add_special": True})["tokens"]
        if srv != ids:
            print(f"TOKENIZE MISMATCH for {text!r}\n  hf  {ids}\n  cpp {srv}")
            sys.exit(1)
        with torch.no_grad():
            logits = m(torch.tensor([ids]))[0, -1, :cfg["vocab"]]
        probs = torch.softmax(logits.double(), -1)
        r = post(args.url + "/completion", {"prompt": ids, "n_predict": 1, "n_probs": 5,
                                             "temperature": 0, "post_sampling_probs": False})
        top = r["completion_probabilities"][0]["top_logprobs"]
        for e in top:
            hf_p = probs[e["id"]].item()
            cpp_p = math.exp(e["logprob"])
            worst = max(worst, abs(hf_p - cpp_p))
        hf_top = torch.topk(probs, 5).indices.tolist()
        cpp_top = [e["id"] for e in top]
        print(f"{text!r}: hf top5 {hf_top}  cpp top5 {cpp_top}")
    print(f"max |p_hf - p_cpp| over top-5: {worst:.2e}")
    sys.exit(0 if worst < args.tol else 1)


if __name__ == "__main__":
    main()
