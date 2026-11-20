#!/usr/bin/env python3
"""Researcher-side helper for TinyForge autoresearch (used by Jarvis, not by the notebook).

  ar.py new  ID "description" [--from best|<exp id>] [--code-ref SHA] [--set FLAG=VALUE ...] [--unset FLAG ...]
        FLAG without dashes: matrix-lr -> --matrix-lr, d -> -d
        writes autoresearch/queue/ID.json with the FULL trainer args (base args + changes)
  ar.py promote ID --results results.jsonl
        makes that experiment the new base in autoresearch/best.json (the 'keep' decision)
  ar.py show results.tsv     sorted table of results (best first) plus crashes
  ar.py check                validates every queue file the same way the worker will
Commit and push the branch afterwards; the worker picks queue files up within about a minute.
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.path.join(HERE, "queue")
BEST = os.path.join(HERE, "best.json")


def load_best():
    return json.load(open(BEST))


def head():
    return subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def cmd_new(a):
    if a.base == "best":
        base = load_best()
    else:
        base = json.load(open(os.path.join(QUEUE, a.base + ".json")))
    args = dict(base["args"])
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        if not k.startswith("-"):
            k = ("-" if len(k) == 1 else "--") + k   # --set matrix-lr=0.03 or --set d=32768
        args[k] = v if not v.lstrip("-").replace(".", "", 1).isdigit() else (float(v) if "." in v else int(v))
    for k in a.unset or []:
        args.pop(k, None)
    exp = {"id": a.id, "description": a.description, "code_ref": a.code_ref or base["code_ref"],
           "base": base.get("id"), "args": args}
    p = os.path.join(QUEUE, a.id + ".json")
    if os.path.exists(p) and not a.force:
        sys.exit(f"{p} exists; ids are never reused (pick a new number)")
    os.makedirs(QUEUE, exist_ok=True)
    json.dump(exp, open(p, "w"), indent=1)
    open(p, "a").write("\n")
    print("wrote", p)


def cmd_promote(a):
    recs = [json.loads(l) for l in open(a.results) if l.strip()]
    r = [x for x in recs if x["id"] == a.id]
    if not r:
        sys.exit(f"{a.id} not in {a.results}")
    r = r[-1]
    if r["status"] == "crash":
        sys.exit(f"{a.id} crashed; can't promote it")
    best = {"id": r["id"], "description": r.get("description", ""), "code_ref": r["commit"], "args": r["args"],
            "val_bpb": r["val_bpb"], "memory_gb": r.get("memory_gb")}
    json.dump(best, open(BEST, "w"), indent=1)
    open(BEST, "a").write("\n")
    print(f"best.json -> {r['id']} val_bpb {r['val_bpb']}")


def cmd_show(a):
    rows = [l.rstrip("\n").split("\t") for l in open(a.tsv)][1:]
    ok = sorted((r for r in rows if r[3] != "crash"), key=lambda r: float(r[1]))
    for r in ok:
        print(f"{r[1]}  {r[2]:>5} GB  {r[3]:8s} {r[0]}  {r[4]}")
    for r in rows:
        if r[3] == "crash":
            print(f"   crash            {r[0]}  {r[4]}")


def cmd_check(a):
    sys.path.insert(0, HERE)
    import worker
    w = worker.Worker.__new__(worker.Worker)
    w.c = {"ctrl": os.path.dirname(HERE), "fixed_T": 512, "eval_tokens": 229376, "branch": "autoresearch"}
    bad = 0
    for f in sorted(os.listdir(QUEUE)):
        if f.endswith(".json"):
            e, err = w.validate(os.path.join(QUEUE, f))
            print(f"{f}: {'OK' if not err else 'REJECT: ' + err}")
            bad += bool(err)
    sys.exit(1 if bad else 0)


p = argparse.ArgumentParser()
sp = p.add_subparsers(dest="cmd", required=True)
n = sp.add_parser("new"); n.add_argument("id"); n.add_argument("description")
n.add_argument("--from", dest="base", default="best"); n.add_argument("--code-ref")
n.add_argument("--set", action="append"); n.add_argument("--unset", action="append"); n.add_argument("--force", action="store_true")
pr = sp.add_parser("promote"); pr.add_argument("id"); pr.add_argument("--results", required=True)
s = sp.add_parser("show"); s.add_argument("tsv")
sp.add_parser("check")
a = p.parse_args()
{"new": cmd_new, "promote": cmd_promote, "show": cmd_show, "check": cmd_check}[a.cmd](a)
