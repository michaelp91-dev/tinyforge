"""Box dry run of TinyForge-AutoResearch.ipynb: no GPU, no Colab, no Hugging Face, no GitHub.
A local bare git repo stands in for GitHub, TF_HF_MOCK_DIR for the Hub, fake_trainer_ar.py for the CUDA trainer.
A 'researcher' thread pushes experiments while the notebook runs, like Jarvis would.
Usage (repo root): python3 tests/autoresearch/dry_run.py [scenario]   scenarios: main, control_stop, session_cap, idle_empty"""
import json, os, shutil, struct, subprocess, sys, tempfile, threading, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SC = sys.argv[1] if len(sys.argv) > 1 else "main"
tmp = tempfile.mkdtemp(prefix=f"ar_dry_{SC}_")
def sh(c, cwd=None):
    r = subprocess.run(c, shell=True, cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, (c, r.stdout, r.stderr)
    return r.stdout.strip()

# fake GitHub: bare repo seeded from the working tree (including uncommitted files)
bare, dev = os.path.join(tmp, "gh.git"), os.path.join(tmp, "dev")
sh(f"git init -q --bare {bare}")
sh(f"git clone -q {ROOT} {dev}")
sh(f"tar -C {ROOT} --exclude=.git -cf - . | tar -C {dev} -xf -")
sh("git -c user.name=t -c user.email=t@t checkout -q -B autoresearch && git add -A && git -c user.name=t -c user.email=t@t commit -qm seed --allow-empty", cwd=dev)
sh(f"rm -f autoresearch/queue/*.json", cwd=dev)
base = sh("git rev-parse HEAD", cwd=dev)
def write_exp(i, desc, args, ref=None):
    e = {"id": i, "description": desc, "code_ref": ref or base, "args": args}
    json.dump(e, open(os.path.join(dev, "autoresearch", "queue", i + ".json"), "w"))
def push(msg):
    sh(f"git add -A && git -c user.name=t -c user.email=t@t commit -qm '{msg}' --allow-empty && git push -q -f {bare} HEAD:autoresearch", cwd=dev)
    return sh("git rev-parse HEAD", cwd=dev)
json.dump({"stop": False}, open(os.path.join(dev, "autoresearch", "control.json"), "w"))
base_args = {"-e": "d12", "-b": 16, "-t": 512, "-d": 65536}
if SC != "idle_empty":
    write_exp("000-baseline", "baseline", base_args)
    write_exp("001-d32k", "total batch 32768", dict(base_args, **{"-d": 32768}))
push("queue 000-001")

# fake HF: data repo like the 8-hour notebook leaves it
mock = os.path.join(tmp, "hf")
tag = "climbmix_v32768_610M"
dd = os.path.join(mock, "dataset", "mockuser__tinyforge-data", tag)
os.makedirs(os.path.join(dd, "tokenizer")); os.makedirs(os.path.join(dd, "climbmix"))
def tokfile(p, n):
    with open(p, "wb") as f:
        f.write(struct.pack("<256i", *([20240520, 1, n] + [0] * 253))); f.write(b"\0" * 64)
tokfile(os.path.join(dd, "climbmix", "val.bin"), 262144)
tokfile(os.path.join(dd, "climbmix", "train_000000.bin"), 100_000_000)
tokfile(os.path.join(dd, "climbmix", "train_000001.bin"), 100_000_000)
open(os.path.join(dd, "tokenizer", "tokenizer.json"), "w").write("{}")
open(os.path.join(dd, "tokenizer", "token_bytes.bin"), "wb").write(b"\0" * 64)
json.dump({"tag": tag, "shards": {"train_000000.bin": 100_000_000, "train_000001.bin": 100_000_000}, "tok_sha": "abc"},
          open(os.path.join(dd, "meta.json"), "w"))

if SC == "nodata":
    os.remove(os.path.join(dd, "meta.json"))
content = os.path.join(tmp, "content")
env = {"TF_LOCAL_TEST": "1", "TF_REPO_URL": bare, "TF_HF_MOCK_DIR": mock, "TF_CONTENT": content,
       "TF_FAKE_TRAINER": os.path.join(ROOT, "tests", "autoresearch", "fake_trainer_ar.py"),
       "TF_TIME_BUDGET_MIN": "0.05", "TF_IDLE_TIMEOUT_MIN": "0.25", "TF_POLL_S": "1", "TF_EXP_OVERHEAD_S": "8",
       "TF_MAX_EXPERIMENTS": "12", "TF_MAX_SESSION_H": "1.0", "TF_TEST_TOKEN": "hf_SECRET_test_value", "FAKE_STEP_S": "0.05"}
if SC == "hang":
    env["FAKE_HANG"] = "1"
if SC == "session_cap":
    env["TF_MAX_SESSION_H"] = str(25 / 3600)   # 25 s session: room for about 2 experiments of 3 s + 8 s allowance
os.environ.update(env)

def researcher():
    """Pushes more work while the worker runs, like Jarvis between wakes."""
    if SC == "main":
        time.sleep(4)
        write_exp("002-t1024", "seq len 1024 (must be rejected: eval is fixed)", dict(base_args, **{"-t": 1024}))
        write_exp("003-oom", "micro batch 64 (fake OOM)", dict(base_args, **{"-b": 64}))
        write_exp("004-nan", "matrix lr 5 (fake divergence)", dict(base_args, **{"--matrix-lr": 5}))
        write_exp("005-mlr03", "matrix lr 0.03", dict(base_args, **{"--matrix-lr": 0.03}))
        push("queue 002-005")
        time.sleep(10)
        # a model-code change: the worker must rebuild for it, then reuse the cache
        open(os.path.join(dev, "src", "tf_kernels.cuh"), "a").write("\n// autoresearch test edit\n")
        ref = push("model code edit")
        write_exp("006-codeedit", "code edit + d32k", dict(base_args, **{"-d": 32768}), ref)
        write_exp("007-codeedit-mlr", "same code, matrix lr 0.03", dict(base_args, **{"-d": 32768, "--matrix-lr": 0.03}), ref)
        open(os.path.join(dev, "src", "BREAK_BUILD"), "w").write("x")
        bad = push("broken build")
        write_exp("008-broken", "broken build", base_args, bad)
        write_exp("009-afterbreak", "base code still runs after a broken build", dict(base_args, **{"-d": 16384}))
        push("queue 006-009")
    elif SC == "control_stop":
        time.sleep(2)
        write_exp("002-x", "should not run", base_args)
        json.dump({"stop": True}, open(os.path.join(dev, "autoresearch", "control.json"), "w"))
        push("stop")
threading.Thread(target=researcher, daemon=True).start()

nb = json.load(open(os.path.join(ROOT, "TinyForge-AutoResearch.ipynb")))
import io, contextlib
def run_nb():
    ns = {"__name__": "__main__"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                exec(compile("".join(c["source"]), "<cell>", "exec"), ns)
    return buf.getvalue()
t0 = time.time()
out = run_nb()
if SC == "relaunch":   # second session on the same results repo: must not repeat finished experiments
    out = run_nb()
open(os.path.join(tmp, "notebook_output.txt"), "w").write(out)
print(out[-3500:])
print(f"--- scenario {SC}: notebook finished in {time.time() - t0:.0f}s; files in {tmp}")
res = os.path.join(mock, "model", "mockuser__tinyforge-autoresearch")
if SC == "nodata":
    assert os.path.exists(os.path.join(content, "unassign.log")) and "[data] FAILED" in out and "work   FAILED" in out, out[-1500:]
    print("PASS nodata"); sys.exit(0)
tsv = open(os.path.join(res, "results.tsv")).read()
print(tsv)
status = json.load(open(os.path.join(res, "status.json")))
print("stop_reason:", status["stop_reason"], "| phase:", status["phase"])
assert os.path.exists(os.path.join(content, "unassign.log")), "runtime was not released"
assert "hf_SECRET_test_value" not in out and "hf_SECRET_test_value" not in tsv, "token leaked"
rows = [l.split("\t") for l in tsv.strip().splitlines()[1:]]
st = {r[4].split(":")[0]: r[3] for r in rows}
if SC == "main":
    assert st["000-baseline"] == "keep" and st["001-d32k"] == "keep", st
    assert st["002-t1024"] == "crash" and "rejected" in tsv
    assert st["003-oom"] == "crash" and "OOM" in tsv
    assert st["004-nan"] == "crash" and "loss not finite" in tsv
    assert st["005-mlr03"] in ("keep", "discard")
    assert st["006-codeedit"] != "crash" and st["007-codeedit-mlr"] != "crash"
    assert st["008-broken"] == "crash" and "build failed" in tsv
    assert st["009-afterbreak"] != "crash"
    assert status["stop_reason"].startswith("idle timeout"), status["stop_reason"]
    builds = [l for l in out.splitlines() if "building trainer" in l]
    assert len(builds) == 3, builds   # base, code edit, broken; 007 and 009 reuse the cache
    for r in ("runs/000-baseline/run.log", "runs/003-oom/result.json", "results.jsonl"):
        assert os.path.exists(os.path.join(res, r)), r
elif SC == "control_stop":
    assert "002-x" not in st and status["stop_reason"].startswith("control.json"), (st, status["stop_reason"])
elif SC == "hang":
    env["FAKE_HANG"] = "1"
if SC == "session_cap":
    assert status["stop_reason"].startswith("session time cap"), status["stop_reason"]
elif SC == "idle_empty":
    assert not rows and status["stop_reason"].startswith("idle timeout")
elif SC == "hang":
    assert st["000-baseline"] == "crash" and "timeout" in tsv, tsv
elif SC == "relaunch":
    assert len(rows) == 2 and "previous results: 2 experiment(s); best so far 001-d32k" in out, (rows, out[-2000:])
print(f"PASS {SC}")
