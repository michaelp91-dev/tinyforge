"""Generates TinyForge-AutoResearch.ipynb (run from the repo root: python3 tests/autoresearch/gen_nb.py)."""
import json

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r'''
# TinyForge AutoResearch (Colab T4 worker)

Short fixed-time training experiments on the d12 TinyForge model, scored only by **val bpb**. Adapted from Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) (MIT): fixed time budget per experiment, one score, a `results.tsv` log, keep the change if bpb improves and drop it if not, crashes logged and skipped.

**The difference from Karpathy's setup:** no model runs inside this notebook. The researcher is Jarvis. This notebook is only the **worker**:
1. It polls the `autoresearch` branch of the tinyforge repo about once a minute for experiment files in `autoresearch/queue/`.
2. It builds the trainer for that experiment's commit (cached; it rebuilds only when the C/CUDA source changed).
3. It trains for a fixed **12 minutes** of training time and runs the same fixed eval: 229,376 val tokens at sequence length 512.
4. It appends one row to `results.tsv` and uploads the log to the **private** HF repo `<you>/tinyforge-autoresearch`. Jarvis reads that repo, decides what to keep, and pushes the next experiments.

**Safety caps:** at most `MAX_EXPERIMENTS` per session and `MAX_SESSION_H` hours. If the queue stays empty for `IDLE_TIMEOUT_MIN` minutes, it saves and disconnects. `control.json` with `stop: true` ends the session after the current experiment. Crashes and OOMs are logged as `crash` and the worker moves on. Trial defaults: 8 experiments, 2.2 h session cap. Two watchdogs disconnect at `MAX_SESSION_H + 0.25` hours (2.45 h; the process watchdog 10 min later) no matter what, and the disconnect sits in a `finally`. Nothing touches Google Drive.

**Before the first run:** the Colab secret **`HF_TOKEN`** (a Hugging Face token with write access, Notebook access on) is the same one the 8-hour notebook uses. The token is never printed.

**How to run:** Runtime > Change runtime type > **T4 GPU**, then **Runtime > Run all**. Then tell Jarvis it's running.
''')

code(r'''
# ---- 1. Setup: knobs, HF token + results repo, watchdogs, GPU check, clone ----
import os, sys, re, glob, json, time, shutil, signal, subprocess, threading, traceback, importlib

LOCAL_TEST = os.environ.get("TF_LOCAL_TEST") == "1"   # box-side dry run only; ignore in Colab
def _knob(name, default):
    v = os.environ.get("TF_" + name) if LOCAL_TEST else None
    return type(default)(v) if v is not None else default

# Knobs (Michael: these are the caps that control compute-unit spend)
MAX_EXPERIMENTS  = _knob("MAX_EXPERIMENTS", 8)      # per session (trial: 8 x ~13.5 min, about 2 compute units)
MAX_SESSION_H    = _knob("MAX_SESSION_H", 2.2)      # 8 x 13.5 min + ~6 min setup + slack; no experiment starts that could pass it
IDLE_TIMEOUT_MIN = _knob("IDLE_TIMEOUT_MIN", 20.0)  # empty queue this long -> save and disconnect
TIME_BUDGET_MIN  = _knob("TIME_BUDGET_MIN", 12.0)   # training time per experiment (fixed; changing it resets comparability)
EXP_OVERHEAD_S   = _knob("EXP_OVERHEAD_S", 300)     # startup + final eval allowance; hard kill at budget + this
POLL_S           = _knob("POLL_S", 60)
AUTO_DISCONNECT  = True

REPO_URL      = os.environ.get("TF_REPO_URL") if LOCAL_TEST and os.environ.get("TF_REPO_URL") else "https://github.com/michaelp91-dev/tinyforge"
BRANCH        = "autoresearch"
HF_SECRET     = "HF_TOKEN"
RESULTS_REPO  = "tinyforge-autoresearch"   # private model repo for results
DATA_REPO     = "tinyforge-data"           # private dataset repo written by the 8-hour notebook
DATA_TAG      = "climbmix_v32768_610M"
VOCAB         = 32768
FIXED_T       = 512
EVAL_TOKENS   = 229_376                    # 28 x 8192; divisible by B*512 for B = 1, 2, 4, 7, 8, 14, 16, 28, 32
TZ_NAME       = "America/Chicago"

CONTENT = os.environ.get("TF_CONTENT", "/content")
T0 = time.time()
WATCHDOG_H = MAX_SESSION_H + 0.25   # hard stop just above the session cap
STATUS = {}
HF_ENV = None
HF_TOKEN_STR = None
WORKER = None
os.makedirs(CONTENT, exist_ok=True)

try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo(TZ_NAME)
except Exception:
    _TZ = None
from datetime import datetime
def clock(ts=None):
    d = datetime.fromtimestamp(ts if ts is not None else time.time(), _TZ)
    return d.strftime("%I:%M %p").lstrip("0") + (" CT" if _TZ else "")

def banner(msg):
    print("\n" + "!" * 78 + "\n" + msg + "\n" + "!" * 78, flush=True)

def run_stage(name, fn, needs=()):
    """Run one stage; never raise, so Run all always reaches the last cell (which disconnects)."""
    bad = [n for n in needs if STATUS.get(n, {}).get("ok") is not True]
    if bad:
        STATUS[name] = {"ok": False, "error": f"skipped: {', '.join(bad)} did not succeed"}
        banner(f"[{name}] SKIPPED because {', '.join(bad)} failed (see above).")
        return None
    t = time.time()
    try:
        out = fn()
        STATUS[name] = {"ok": True, "secs": round(time.time() - t, 1)}
        print(f"[{name}] ok in {time.time() - t:.0f}s", flush=True)
        return out
    except BaseException as e:
        STATUS[name] = {"ok": False, "secs": round(time.time() - t, 1), "error": f"{type(e).__name__}: {_scrub(str(e))[:500]}"}
        banner(f"[{name}] FAILED: {type(e).__name__}: {_scrub(str(e))[:2000]}")
        return None

def sh(cmd, check=True, env=None, cwd=None):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, env=env, cwd=cwd)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n{(r.stdout + r.stderr)[-3000:]}")
    return r

def _scrub(text):
    return text.replace(HF_TOKEN_STR, "***") if HF_TOKEN_STR and text else text

def hf(args, timeout):
    cmd = [sys.executable, os.path.join(CTRL, "python", "hf_sync.py")] + [str(a) for a in args]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=HF_ENV, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout); rc = p.returncode
    except subprocess.TimeoutExpired:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
        out, rc = "", "timeout"
    return rc == 0, rc, _scrub(out or "").strip()

def _unassign(reason):
    print(f"Disconnecting the runtime now ({reason}) to stop compute-unit use.", flush=True)
    if LOCAL_TEST:
        open(os.path.join(CONTENT, "unassign.log"), "a").write(f"{time.time():.0f} {reason}\n")
        return
    try:
        from google.colab import runtime
        runtime.unassign()
        return
    except Exception as e:
        print("runtime.unassign() failed:", type(e).__name__, e, flush=True)
    try:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request(f"http://{os.environ['TBE_RUNTIME_ADDR']}/unassign", method="POST"), timeout=30)
    except Exception as e:
        print("direct unassign failed:", type(e).__name__, e, flush=True)

def _watchdog():
    deadline = T0 + WATCHDOG_H * 3600
    while time.time() < deadline:
        time.sleep(min(30, max(1, deadline - time.time())))
    banner(f"WATCHDOG: {WATCHDOG_H} h reached; stopping the experiment, saving results, disconnecting.")
    try:
        if WORKER is not None:
            WORKER.stop_flag.set()
            p = WORKER._proc
            if p is not None and p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            WORKER.state["stop_reason"] = "watchdog"
            WORKER.final_upload(timeout=90)
    except Exception as e:
        print("watchdog upload failed:", type(e).__name__, flush=True)
    _unassign("watchdog")

def _process_watchdog():
    """A separate OS process that releases the VM 10 min after the thread watchdog, even if the kernel hangs."""
    addr = os.environ.get("TF_TEST_TBE_ADDR") if LOCAL_TEST else os.environ.get("TBE_RUNTIME_ADDR")
    if not addr:
        print("note: runtime service address not found; the process watchdog is off (the thread watchdog is still armed)")
        return
    deadline = T0 + WATCHDOG_H * 3600 + 600
    prog = ("import time,urllib.request\n"
            f"time.sleep(max(0, {deadline} - time.time()))\n"
            f"urllib.request.urlopen(urllib.request.Request('http://{addr}/unassign', method='POST'), timeout=30)\n")
    subprocess.Popen([sys.executable, "-c", prog], start_new_session=True,
                     stdout=open(os.path.join(CONTENT, "ar_watchdog.log"), "a"), stderr=subprocess.STDOUT)

def _setup():
    global CTRL, GPU_CC, GPU_NAME, HF_ENV, HF_TOKEN_STR, RESULTS_ID, DATA_ID
    if LOCAL_TEST:
        GPU_NAME, GPU_CC = "Tesla T4 (box test)", "7.5"
    else:
        r = sh("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader", check=False)
        if r.returncode != 0 or not r.stdout.strip():
            raise RuntimeError("No GPU visible. Runtime > Change runtime type > T4 GPU, then Run all again.")
        GPU_NAME, GPU_CC, mem = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        print(f"GPU: {GPU_NAME}, compute capability {GPU_CC}, {mem}")
        if "T4" not in GPU_NAME:
            banner(f"NOTE: results are only comparable on a T4 (fixed time budget). This is a {GPU_NAME}.")
    CTRL = os.path.join(CONTENT, "tinyforge-ar")
    if os.path.isdir(os.path.join(CTRL, ".git")):
        sh(["git", "-C", CTRL, "fetch", "-q", "origin", BRANCH]); sh(["git", "-C", CTRL, "reset", "-q", "--hard", "FETCH_HEAD"])
    else:
        sh(["git", "clone", "-q", "--single-branch", "-b", BRANCH, REPO_URL, CTRL])   # full history: experiments name older commits
    print("repo:", CTRL, "@", sh(["git", "-C", CTRL, "log", "-1", "--format=%h %s"]).stdout.strip())
    if LOCAL_TEST:
        tok = os.environ.get("TF_TEST_TOKEN", "hf_test_token_not_real")
    else:
        try:
            from google.colab import userdata
            tok = userdata.get(HF_SECRET)
        except Exception as e:
            raise RuntimeError(f"Could not read the Colab secret '{HF_SECRET}' ({type(e).__name__}). Add it with the key icon "
                               "on the left (a Hugging Face token with WRITE access) and turn on Notebook access.") from None
    if not tok:
        raise RuntimeError(f"The Colab secret '{HF_SECRET}' is empty.")
    HF_TOKEN_STR = tok
    HF_ENV = dict(os.environ, HF_TOKEN=tok, HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_TELEMETRY="1")
    if not LOCAL_TEST:
        try:
            import huggingface_hub
            ver = tuple(int(x) for x in re.findall(r"\d+", huggingface_hub.__version__)[:2])
        except Exception:
            ver = (0, 0)
        if ver < (0, 30):
            sh([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub>=0.30"])
    ok, rc, out = hf(["whoami", "--repo", "-"], 60)
    if not ok:
        raise RuntimeError(f"Hugging Face login failed ({rc}): {out[-300:]}")
    user = out.strip().splitlines()[-1]
    RESULTS_ID, DATA_ID = f"{user}/{RESULTS_REPO}", f"{user}/{DATA_REPO}"
    ok, rc, out = hf(["ensure", "--repo", RESULTS_ID, "--type", "model"], 120)
    if not ok:
        raise RuntimeError(f"could not open or create {RESULTS_ID} ({rc}): {out[-300:]}. Does the token have WRITE access?")
    info = json.loads(out.strip().splitlines()[-1])
    if not info["private"]:
        raise RuntimeError(f"{RESULTS_ID} is PUBLIC. Make it private on huggingface.co, then Run all again.")
    print(f"results repo {RESULTS_ID}: {'created (private)' if info['created'] else 'found (private)'}")
    print(f"session started {clock(T0)}; no new experiment starts after {clock(T0 + MAX_SESSION_H * 3600 - TIME_BUDGET_MIN * 60 - EXP_OVERHEAD_S)}; "
          f"watchdog disconnects at {clock(T0 + WATCHDOG_H * 3600)} no matter what")

run_stage("setup", _setup)
threading.Thread(target=_watchdog, daemon=True).start()
try:
    _process_watchdog()
except Exception as e:
    print("process watchdog not started:", e)
''')

code(r'''
# ---- 2. Build function (nvcc of this runtime), then build the branch head once to fail fast ----
TRAIN_ENV = dict(os.environ)

def _find_nvcc():
    for c in [shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"] + sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"), reverse=True):
        if c and os.path.exists(c):
            return c
    raise RuntimeError("nvcc not found; switch to a T4 GPU runtime and Run all again.")

def _pip_cublas(major):
    pkg = "nvidia-cublas-cu12" if major == 12 else f"nvidia-cublas>={major},<{major + 1}"
    sh([sys.executable, "-m", "pip", "install", "-q", pkg])
    import site
    for root in site.getsitepackages() + [site.getusersitepackages()]:
        for sub in (f"nvidia/cu{major}", "nvidia/cublas"):
            d = os.path.join(root, sub)
            if glob.glob(os.path.join(d, "lib", "libcublas.so*")):
                lib = os.path.join(d, "lib")
                for name in ("libcublas", "libcublasLt"):
                    real = sorted(glob.glob(os.path.join(lib, name + ".so.*")))
                    if real and not os.path.exists(os.path.join(lib, name + ".so")):
                        os.symlink(os.path.basename(real[0]), os.path.join(lib, name + ".so"))
                return os.path.join(d, "include"), lib
    raise RuntimeError(f"pip installed {pkg} but no libcublas.so was found")

def _missing_libs(binary, env):
    r = sh(["ldd", binary], check=False, env=env)
    return [l.split()[0] for l in r.stdout.splitlines() if "not found" in l]

def build_fn(src_dir, log):
    """Build src_dir/train_tinyforge for this GPU (fp16 T4 path). Returns the binary path."""
    global TRAIN_ENV
    if LOCAL_TEST:   # box test: the fake trainer stands in for the CUDA binary
        b = os.path.join(src_dir, "train_tinyforge")
        shutil.copy(os.environ["TF_FAKE_TRAINER"], b); os.chmod(b, 0o755)
        open(log, "w").write("fake build\n")
        if os.path.exists(os.path.join(src_dir, "src", "BREAK_BUILD")):
            raise RuntimeError("compile error (test)")
        return b
    nvcc = _find_nvcc()
    m = re.search(r"release (\d+)\.(\d+)", sh([nvcc, "--version"]).stdout)
    major = int(m.group(1)) if m else 0
    cuda_home = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    env = dict(os.environ); env["PATH"] = os.path.dirname(nvcc) + ":" + env.get("PATH", "")
    make = ["make", "-C", src_dir, "train_tinyforge", "GPU=t4"]
    cc = GPU_CC.replace(".", "")
    if cc != "75":
        make.append(f"NVCC_ARCH=-gencode arch=compute_{cc},code=sm_{cc} -DENABLE_FP16 -DTF_FP16_GEMMEX")
    lib_dirs = [d for d in (os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "lib"),
                            os.path.join(cuda_home, "targets/x86_64-linux/lib")) if os.path.isdir(d)]
    extra = ""
    for attempt in (1, 2):
        r = subprocess.run(make + [f"NVCC={nvcc} {extra}".strip()], capture_output=True, text=True, env=env)
        open(log, "a").write(f"=== attempt {attempt}\n{r.stdout}{r.stderr}\n")
        if r.returncode == 0:
            break
        out = r.stdout + r.stderr
        if attempt == 1 and re.search(r"cublas|cannot find -l", out, re.I):
            inc, lib = _pip_cublas(major); extra = f"-I{inc} -L{lib}"; lib_dirs.insert(0, lib)
            continue
        raise RuntimeError("build failed:\n" + out[-1500:])
    b = os.path.join(src_dir, "train_tinyforge")
    TRAIN_ENV = dict(os.environ)
    TRAIN_ENV["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")]).strip(":")
    if _missing_libs(b, TRAIN_ENV):
        extra_dirs = sorted({os.path.dirname(p) for p in glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/*/lib/*.so*")})
        TRAIN_ENV["LD_LIBRARY_PATH"] += ":" + ":".join(extra_dirs)
    miss = _missing_libs(b, TRAIN_ENV)
    if miss:
        raise RuntimeError(f"built, but these libraries can't be found at run time: {miss}")
    return b

def _make_worker():
    global WORKER
    sys.path.insert(0, os.path.join(CTRL, "autoresearch"))
    import worker as _w
    importlib.reload(_w)
    WORKER = _w.Worker({
        "work": os.path.join(CONTENT, "ar_work"), "ctrl": CTRL, "branch": BRANCH,
        "hf_sync": os.path.join(CTRL, "python", "hf_sync.py"), "hf_env": HF_ENV, "results_repo": RESULTS_ID,
        "train_glob": os.path.join(CONTENT, "data", "climbmix", "train_*.bin"),
        "val_bin": os.path.join(CONTENT, "data", "climbmix", "val.bin"),
        "token_bytes": os.path.join(CONTENT, "data", "tokenizer", "token_bytes.bin"), "vocab": VOCAB,
        "time_budget_s": int(TIME_BUDGET_MIN * 60), "eval_tokens": EVAL_TOKENS, "fixed_T": FIXED_T,
        "max_experiments": MAX_EXPERIMENTS, "session_deadline": T0 + MAX_SESSION_H * 3600,
        "idle_timeout_s": IDLE_TIMEOUT_MIN * 60, "poll_s": POLL_S, "exp_overhead_s": EXP_OVERHEAD_S,
        "build_fn": build_fn, "train_env": None,
    })

def _build():
    _make_worker()
    head = sh(["git", "-C", CTRL, "rev-parse", "HEAD"]).stdout.strip()
    b, key = WORKER.binary_for(head)
    WORKER.c["train_env"] = TRAIN_ENV
    print("trainer ready:", b)

run_stage("build", _build, needs=["setup"])
''')

code(r'''
# ---- 3. Data: tokenizer, val set, and the first 100M-token train shard from the private HF data repo ----
# One shard is plenty: a 12-minute experiment reads about 13M tokens, and every experiment starts from the
# same place in the same shard, so all experiments see identical data.
DATA = os.path.join(CONTENT, "data")

def _data():
    os.makedirs(DATA, exist_ok=True)
    mp = os.path.join(DATA, "meta.json")
    ok, rc, out = hf(["pull", "--repo", DATA_ID, "--type", "dataset", "--file", f"{DATA_TAG}/meta.json={mp}"], 180)
    if not ok:
        raise RuntimeError(f"{DATA_ID}/{DATA_TAG} is not on Hugging Face ({rc}). The 8-hour notebook uploads it; "
                           "this notebook does not rebuild data, so the tokenizer and val set stay the same. " + out[-200:])
    meta = json.load(open(mp))
    first = sorted(meta["shards"])[0]
    want = [f"{DATA_TAG}/tokenizer/tokenizer.json", f"{DATA_TAG}/tokenizer/token_bytes.bin",
            f"{DATA_TAG}/climbmix/val.bin", f"{DATA_TAG}/climbmix/{first}"]
    t = time.time()
    for r in want:
        local = os.path.join(DATA, r[len(DATA_TAG) + 1:])
        if os.path.exists(local) and os.path.getsize(local) > 0:
            continue
        ok, rc, out = hf(["pull", "--repo", DATA_ID, "--type", "dataset", "--file", f"{r}={local}", "--tries", "4"], 1200)
        if not ok:
            raise RuntimeError(f"download of {r} failed ({rc}): {out[-200:]}")
    import numpy as np
    nval = int(np.fromfile(os.path.join(DATA, "climbmix", "val.bin"), dtype=np.int32, count=3)[2])
    if nval - 1 < EVAL_TOKENS:
        raise RuntimeError(f"val.bin has {nval} tokens; the fixed eval needs {EVAL_TOKENS}")
    print(f"data ready in {time.time() - t:.0f}s: shard {first} ({meta['shards'][first]:,} tokens), val {nval:,} tokens "
          f"(eval uses {EVAL_TOKENS:,}), tokenizer {meta.get('tok_sha')}")

run_stage("data", _data, needs=["setup"])
''')

code(r'''
# ---- 4. Worker loop: poll the branch, run experiments, upload results. Always disconnects at the end. ----
def _work():
    print(f"worker: {TIME_BUDGET_MIN:.0f} min per experiment, at most {MAX_EXPERIMENTS} experiments, "
          f"idle timeout {IDLE_TIMEOUT_MIN:.0f} min, session cap {MAX_SESSION_H} h. Results: {RESULTS_ID}", flush=True)
    WORKER.run()

try:
    run_stage("work", _work, needs=["setup", "build", "data"])
finally:
    try:
        if WORKER is not None and HF_ENV:
            if WORKER.state.get("stop_reason") is None:
                WORKER.state["stop_reason"] = "stage failed: " + "; ".join(f"{k}: {v.get('error')}" for k, v in STATUS.items() if not v.get("ok"))
            print("final upload:", "ok" if WORKER.final_upload(timeout=300) else "FAILED")
    except BaseException as e:
        print("final upload error:", type(e).__name__, _scrub(str(e))[:300])
    print("\nstage status:")
    for k, v in STATUS.items():
        print(f"  {k:6s} {'ok' if v.get('ok') else 'FAILED'} {v.get('secs', '')}s {v.get('error', '')}")
    print(f"notebook wall time {(time.time() - T0) / 3600:.2f} h (started {clock(T0)}, now {clock()})")
    if AUTO_DISCONNECT:
        _unassign("end of notebook")
''')

nb = {"cells": [], "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for kind, src in cells:
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    nb["cells"].append(c)
json.dump(nb, open("TinyForge-AutoResearch.ipynb", "w"), indent=1, ensure_ascii=False)
open("TinyForge-AutoResearch.ipynb", "a").write("\n")
print("wrote TinyForge-AutoResearch.ipynb with", len(cells), "cells")
