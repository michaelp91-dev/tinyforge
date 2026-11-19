import json, sys

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r'''
# TinyForge d12: overnight 8-hour T4 run (Hugging Face storage, no Drive)

One Colab T4, about 8 hours of training, then the runtime disconnects itself. Nothing touches Google Drive.

**Before the first run:** add a Colab secret (key icon on the left) named **`HF_TOKEN`** holding a Hugging Face token with **write** access, and switch on **Notebook access** for it. The token is never printed.

**How to run:** Runtime > Change runtime type > **T4 GPU**, then **Runtime > Run all**. Nothing else needs a click.

What it does:
1. Clones `t4-kaggle-d12`, reads `HF_TOKEN`, and makes two **private** Hugging Face repos if they are missing: `<you>/tinyforge-data` (dataset: tokenizer + tokenized ClimbMix) and `<you>/tinyforge-runs` (logs, summary, one resume checkpoint). It arms two watchdogs that disconnect the runtime at `WATCHDOG_H` hours no matter what.
2. Builds the fp16 trainer with the runtime's own nvcc (same as the 1-hour test).
3. Data: 610M ClimbMix train tokens (7 shards) + 262,144 val tokens. **First run:** trains the tokenizer, then tokenizes in the background. Training starts as soon as the first 100M-token shard is ready (about 5 min), and the trainer picks up the other shards at its next checkpoint once they are done. The finished data is uploaded to `tinyforge-data`. **Later runs:** the data is downloaded from there in a couple of minutes.
4. Trains d12 with `-b 16 -t 512 -d 65536 -x 9200`: 602.9M tokens, one pass with no repeated data, about 8.0 h at 21k tok/s. The LR schedule is set for 9,200 steps. Val loss and bpb every 400 steps. A trainer time limit stops it in time for the whole notebook to finish within `BUDGET_H` (8.5 h).
5. One resume checkpoint (model + optimizer state) is uploaded to `tinyforge-runs/<RUN_NAME>/ckpt/` about every 40 min, overwriting the last one, and the history is squashed so the repo stays one checkpoint big. `train.log` and `status.json` go up every 10 min. Every upload runs in its own process with a timeout and retries, so a stuck upload can't block training or shutdown. **Re-running resumes** from that checkpoint.
6. At the end: final val, `summary.json` / `summary.txt`, 3 samples (from a temporary local export that is not kept), final upload, a file listing of the repo to verify it, then `runtime.unassign()`. The disconnect is in a `finally`, so it happens even if the upload fails or times out.
''')

code(r'''
# ---- 1. Setup: config, HF token + repos, watchdogs, GPU check, clone ----
import os, sys, re, glob, json, time, shutil, signal, subprocess, threading, traceback, statistics, importlib.util, hashlib
from datetime import datetime, timedelta

LOCAL_TEST = os.environ.get("TF_LOCAL_TEST") == "1"   # box-side dry run only; ignore in Colab
def _knob(name, default):
    """Box tests may override a knob with TF_<NAME>; in Colab the value below is used."""
    v = os.environ.get("TF_" + name) if LOCAL_TEST else None
    return type(default)(v) if v is not None else default

# Knobs
REPO_URL      = "https://github.com/michaelp91-dev/tinyforge"
BRANCH        = "t4-kaggle-d12"
HF_SECRET     = "HF_TOKEN"     # Colab secret with a WRITE token
HF_USER       = ""             # blank = the account that owns the token
DATA_REPO     = "tinyforge-data"   # private dataset repo: tokenizer + shards
RUNS_REPO     = "tinyforge-runs"   # private model repo: logs, summary, one checkpoint
RUN_NAME      = _knob("RUN_NAME", "d12_8hr")   # folder in the runs repo; re-running with the same name resumes
FRESH_RUN     = False          # True = new run folder (RUN_NAME + timestamp)
MAX_STEPS     = _knob("MAX_STEPS", 9200)      # 9200 x 65,536 = 602.9M tokens, about 8.0 h at 3.12 s/step
TRAIN_TOKENS  = _knob("TRAIN_TOKENS", 610_000_000)  # one pass covers 9,307 steps, so nothing repeats
SHARD_TOKENS  = _knob("SHARD_TOKENS", 100_000_000)
VAL_TOKENS    = 262_144
VOCAB         = 32768
TOK_CHARS     = _knob("TOK_CHARS", 50_000_000)
VAL_EVERY     = _knob("VAL_EVERY", 400)
VAL_BATCHES   = 31             # 31 x 8192 = the whole val set
CKPT_EVERY    = _knob("CKPT_EVERY", 400)       # local checkpoint, about every 21 min
BUDGET_H      = _knob("BUDGET_H", 8.5)         # whole notebook, setup to disconnect
FINAL_RESERVE_MIN = _knob("FINAL_RESERVE_MIN", 12.0)  # kept free for summary, samples, final upload
CKPT_PUSH_MIN = _knob("CKPT_PUSH_MIN", 40.0)   # upload the newest checkpoint this often
LOG_PUSH_MIN  = _knob("LOG_PUSH_MIN", 10.0)    # upload train.log + status.json this often
WATCHDOG_H    = _knob("WATCHDOG_H", 9.0)       # hard stop: disconnect after this many hours no matter what
DO_SAMPLE     = True
AUTO_DISCONNECT = True
TZ_NAME       = "America/Chicago"
TOKENS_PER_STEP = 65536

CONTENT = os.environ.get("TF_CONTENT", "/content")
T0 = time.time()
STATUS = {}
HF_ENV = None
HF_TOKEN_STR = None
TRAINER_PROC = None
WATCHDOG_FIRED = False
os.makedirs(CONTENT, exist_ok=True)

try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo(TZ_NAME)
except Exception:
    _TZ = None
def clock(ts=None):
    d = datetime.fromtimestamp(ts if ts is not None else time.time(), _TZ)
    return d.strftime("%I:%M %p").lstrip("0") + (" CT" if _TZ else "")

def banner(msg):
    print("\n" + "!" * 78 + "\n" + msg + "\n" + "!" * 78, flush=True)

def run_stage(name, fn, needs=()):
    """Run one notebook stage; never raise, so Run all reaches the last cell."""
    bad = [n for n in needs if STATUS.get(n, {}).get("ok") is not True]
    if bad:
        STATUS[name] = {"ok": False, "skipped": True, "error": f"skipped: {', '.join(bad)} did not succeed"}
        banner(f"[{name}] SKIPPED because {', '.join(bad)} failed (see above).")
        return None
    t = time.time()
    try:
        out = fn()
        STATUS[name] = {"ok": True, "secs": round(time.time() - t, 1)}
        print(f"[{name}] ok in {time.time() - t:.0f}s", flush=True)
        return out
    except BaseException as e:   # KeyboardInterrupt too: still let the last cell run
        STATUS[name] = {"ok": False, "secs": round(time.time() - t, 1), "error": f"{type(e).__name__}: {e}"}
        banner(f"[{name}] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

def sh(cmd, check=True, env=None, cwd=None):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, env=env, cwd=cwd)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n{(r.stdout + r.stderr)[-3000:]}")
    return r

def _scrub(text):
    return text.replace(HF_TOKEN_STR, "***") if HF_TOKEN_STR and text else text

_HF_LOCK = threading.Lock()
_HF_CHILD = {"p": None}
def hf(args, timeout):
    """Run python/hf_sync.py in its own process group; kill it after `timeout` seconds.
    Returns (ok, returncode or 'timeout', output). Never raises, never hangs."""
    cmd = [sys.executable, os.path.join(REPO, "python", "hf_sync.py")] + [str(a) for a in args]
    with _HF_LOCK:
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 env=HF_ENV, start_new_session=True)
        except Exception as e:
            return False, "spawn", f"{type(e).__name__}: {e}"
        _HF_CHILD["p"] = p
        try:
            out, _ = p.communicate(timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            try: os.killpg(p.pid, signal.SIGKILL)
            except Exception: pass
            try: out, _ = p.communicate(timeout=15)
            except Exception: out = ""
            rc = "timeout"
        finally:
            _HF_CHILD["p"] = None
    return rc == 0, rc, _scrub(out or "").strip()

def _unassign(reason):
    """Release the Colab VM. Tries google.colab, then the runtime service directly."""
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
    global WATCHDOG_FIRED
    WATCHDOG_FIRED = True
    banner(f"WATCHDOG: {WATCHDOG_H} h reached, stopping the trainer, saving logs, disconnecting.")
    try:
        if TRAINER_PROC is not None and TRAINER_PROC.poll() is None:
            TRAINER_PROC.kill()
    except Exception:
        pass
    try:
        p = _HF_CHILD.get("p")
        if p is not None and p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        files = [f"{f}={RUN_NAME}/{os.path.basename(f)}" for f in (globals().get("TRAIN_LOG"), globals().get("STATUS_PATH")) if f and os.path.exists(f)]
        if files and HF_ENV and globals().get("RUNS_ID"):
            args = ["push", "--repo", globals()["RUNS_ID"], "--type", "model", "--tries", "1", "-m", "watchdog logs"]
            for f in files: args += ["--file", f]
            # bypass the lock: the watchdog must not wait on a stuck upload
            subprocess.run([sys.executable, os.path.join(REPO, "python", "hf_sync.py")] + args,
                           env=HF_ENV, timeout=90, capture_output=True, start_new_session=True)
    except Exception as e:
        print("watchdog log upload failed:", type(e).__name__, flush=True)
    _unassign("watchdog")

def _process_watchdog():
    """A separate OS process that releases the VM 10 min after the thread watchdog, even if the kernel hangs."""
    pidf = os.path.join(CONTENT, "tf_watchdog.pid")
    try:   # a re-run replaces the old one
        os.kill(int(open(pidf).read()), signal.SIGKILL)
    except Exception:
        pass
    addr = os.environ.get("TF_TEST_TBE_ADDR") if LOCAL_TEST else os.environ.get("TBE_RUNTIME_ADDR")
    if not addr:
        print("note: runtime service address not found; the process watchdog is off (the thread watchdog is still armed)")
        return
    deadline = T0 + WATCHDOG_H * 3600 + _knob("PROC_WATCHDOG_EXTRA_S", 600)
    prog = ("import time,urllib.request\n"
            f"time.sleep(max(0, {deadline} - time.time()))\n"
            f"urllib.request.urlopen(urllib.request.Request('http://{addr}/unassign', method='POST'), timeout=30)\n")
    p = subprocess.Popen([sys.executable, "-c", prog], start_new_session=True,
                         stdout=open(os.path.join(CONTENT, "tf_watchdog.log"), "a"), stderr=subprocess.STDOUT)
    open(pidf, "w").write(str(p.pid))

def _setup():
    global REPO, RUN_NAME, GPU_CC, GPU_NAME, HF_ENV, HF_TOKEN_STR, HF_USER, DATA_ID, RUNS_ID
    if FRESH_RUN:
        RUN_NAME = RUN_NAME + time.strftime("_%Y%m%d_%H%M")
    if LOCAL_TEST:
        GPU_NAME, GPU_CC = os.environ.get("TF_GPU_NAME", "Tesla T4"), os.environ.get("TF_GPU_CC", "7.5")
    else:
        r = sh("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader", check=False)
        if r.returncode != 0 or not r.stdout.strip():
            raise RuntimeError("No GPU visible. Runtime > Change runtime type > T4 GPU, then Run all again.")
        GPU_NAME, GPU_CC, mem = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        print(f"GPU: {GPU_NAME}, compute capability {GPU_CC}, {mem}")
        if "T4" not in GPU_NAME:
            print(f"NOTE: this notebook is tuned for a T4; on {GPU_NAME} step times will differ.")

    REPO = os.environ.get("TF_REPO") if LOCAL_TEST and os.environ.get("TF_REPO") else os.path.join(CONTENT, "tinyforge")
    if not (LOCAL_TEST and os.environ.get("TF_REPO")):
        if os.path.isdir(os.path.join(REPO, ".git")):
            sh(["git", "-C", REPO, "fetch", "--depth", "1", "origin", BRANCH])
            sh(["git", "-C", REPO, "reset", "--hard", "FETCH_HEAD"])
        else:
            sh(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO])
    head = sh(["git", "-C", REPO, "log", "-1", "--format=%h %s"]).stdout.strip()
    STATUS["commit"] = head
    print("repo:", REPO, "@", head)

    # Hugging Face token from Colab secrets (never printed).
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
    try:
        import huggingface_hub
        ver = tuple(int(x) for x in re.findall(r"\d+", huggingface_hub.__version__)[:2])
    except Exception:
        ver = (0, 0)
    if ver < (0, 30):
        print("upgrading huggingface_hub ...")
        sh([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub>=0.30"])
    ok, rc, out = hf(["whoami", "--repo", "-"], 60)
    if not ok:
        raise RuntimeError(f"Hugging Face login failed ({rc}): {out[-400:]}. Check that {HF_SECRET} is a valid token.")
    user = out.strip().splitlines()[-1]
    HF_USER = HF_USER or user
    DATA_ID, RUNS_ID = f"{HF_USER}/{DATA_REPO}", f"{HF_USER}/{RUNS_REPO}"
    for rid, typ in ((DATA_ID, "dataset"), (RUNS_ID, "model")):
        ok, rc, out = hf(["ensure", "--repo", rid, "--type", typ], 120)
        if not ok:
            raise RuntimeError(f"could not open or create {typ} repo {rid} ({rc}): {out[-400:]}. Does the token have WRITE access?")
        info = json.loads(out.strip().splitlines()[-1])
        if not info["private"]:
            raise RuntimeError(f"{rid} exists and is PUBLIC. Make it private on huggingface.co (or rename DATA_REPO/RUNS_REPO), then Run all again.")
        print(f"HF {typ} repo {rid}: {'created (private)' if info['created'] else 'found (private)'}")
    print("run:", RUN_NAME, "| notebook started", clock(T0), "| budget", BUDGET_H, "h -> done by about", clock(T0 + BUDGET_H * 3600))

run_stage("setup", _setup)
threading.Thread(target=_watchdog, daemon=True).start()
try:
    _process_watchdog()
except Exception as e:
    print("process watchdog not started:", e)
print(f"watchdog armed: runtime disconnects at {WATCHDOG_H} h ({clock(T0 + WATCHDOG_H * 3600)}) no matter what")
''')

code(r'''
# ---- 2. Build the fp16 T4 trainer with this runtime's CUDA ----
TRAIN_ENV = dict(os.environ)
BIN = None

def _find_nvcc():
    cands = [shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"] + sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"), reverse=True)
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise RuntimeError("nvcc not found (looked on PATH, /usr/local/cuda/bin, /usr/local/cuda-*/bin). "
                       "This runtime has no CUDA toolkit; switch to a GPU runtime (T4) and Run all again.")

def _pip_cublas(major):
    """Install the cuBLAS wheel matching the toolkit's major version; return its (include, lib) dirs."""
    pkg = "nvidia-cublas-cu12" if major == 12 else f"nvidia-cublas>={major},<{major + 1}"
    print("installing", pkg, "...")
    sh([sys.executable, "-m", "pip", "install", "-q", pkg])
    import site
    roots = site.getsitepackages() + [site.getusersitepackages()]
    for root in roots:
        for sub in (f"nvidia/cu{major}", "nvidia/cublas"):
            d = os.path.join(root, sub)
            libs = glob.glob(os.path.join(d, "lib", "libcublas.so*"))
            if libs:
                lib = os.path.join(d, "lib")
                for name in ("libcublas", "libcublasLt"):   # make -lcublas resolve
                    real = sorted(glob.glob(os.path.join(lib, name + ".so.*")))
                    if real and not os.path.exists(os.path.join(lib, name + ".so")):
                        os.symlink(os.path.basename(real[0]), os.path.join(lib, name + ".so"))
                return os.path.join(d, "include"), lib
    raise RuntimeError(f"pip installed {pkg} but no libcublas.so was found under {roots}")

def _missing_libs(binary, env):
    r = sh(["ldd", binary], check=False, env=env)
    return [l.split()[0] for l in r.stdout.splitlines() if "not found" in l]

def _build():
    global BIN, TRAIN_ENV
    if LOCAL_TEST and os.environ.get("TF_FAKE_BIN"):   # box test: no CUDA here
        BIN = os.environ["TF_FAKE_BIN"]
        print("box test: using", BIN)
        return
    nvcc = _find_nvcc()
    ver = sh([nvcc, "--version"]).stdout
    m = re.search(r"release (\d+)\.(\d+)", ver)
    major = int(m.group(1)) if m else 0
    print(f"nvcc: {nvcc} (CUDA {m.group(1)}.{m.group(2)})" if m else f"nvcc: {nvcc}\n{ver}")
    if major and major < 11:
        raise RuntimeError(f"CUDA {major} is too old for this trainer (needs 11.8+).")
    cuda_home = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(nvcc) + ":" + env.get("PATH", "")
    make = ["make", "-C", REPO, "train_tinyforge", "GPU=t4"]
    cc = GPU_CC.replace(".", "")
    if cc != "75":   # not a T4: same fp16 code path, compiled for this GPU
        make.append(f"NVCC_ARCH=-gencode arch=compute_{cc},code=sm_{cc} -DENABLE_FP16 -DTF_FP16_GEMMEX")
    extra = os.environ.get("TF_NVCC_EXTRA", "")
    lib_dirs = [d for d in (os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "lib"),
                            os.path.join(cuda_home, "targets/x86_64-linux/lib")) if os.path.isdir(d)]
    log = os.path.join(CONTENT, "build.log")
    for attempt in (1, 2):
        nv = f"{nvcc} {extra}".strip()
        sh(["rm", "-f", os.path.join(REPO, "train_tinyforge")])
        r = subprocess.run(make + [f"NVCC={nv}"], capture_output=True, text=True, env=env)
        open(log, "a").write(f"=== attempt {attempt}: {' '.join(make)} NVCC={nv}\n{r.stdout}{r.stderr}\n")
        if r.returncode == 0:
            break
        out = r.stdout + r.stderr
        print(out[-2500:])
        if attempt == 1 and re.search(r"cublas|cannot find -l", out, re.I):
            inc, lib = _pip_cublas(major)
            extra = f"{extra} -I{inc} -L{lib}".strip()
            lib_dirs.insert(0, lib)
            continue
        raise RuntimeError(f"BUILD FAILED with CUDA {major} (log: {log}). The compiler error is printed above.")
    BIN = os.path.join(REPO, "train_tinyforge")
    TRAIN_ENV = dict(env)
    TRAIN_ENV["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")]).strip(":")
    miss = _missing_libs(BIN, TRAIN_ENV)
    if miss:
        extra_dirs = sorted({os.path.dirname(p) for p in glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/*/lib/*.so*")})
        TRAIN_ENV["LD_LIBRARY_PATH"] += ":" + ":".join(extra_dirs)
        miss = _missing_libs(BIN, TRAIN_ENV)
    if miss and any("cublas" in x for x in miss):
        _, lib = _pip_cublas(major)
        TRAIN_ENV["LD_LIBRARY_PATH"] = lib + ":" + TRAIN_ENV["LD_LIBRARY_PATH"]
        miss = _missing_libs(BIN, TRAIN_ENV)
    if miss:
        raise RuntimeError(f"built OK but these libraries can't be found at run time: {miss}")
    for l in sh(["ldd", BIN], env=TRAIN_ENV).stdout.splitlines():
        if "cublas" in l or "cudart" in l:
            print("  " + l.strip())
    print("built", BIN)

run_stage("build", _build, needs=["setup"])
''')

code(r'''
# ---- 3. Data: from the HF data repo, or built here (in the background) and uploaded there ----
DATA = os.path.join(CONTENT, "data")
TOK_DIR   = os.path.join(DATA, "tokenizer")
BUILD_DIR = os.path.join(DATA, "climbmix_build")   # the builder writes here
MIX_DIR   = os.path.join(DATA, "climbmix")         # finished shards only (hard links); the trainer reads this
DATA_TAG  = f"climbmix_v{VOCAB}_{TRAIN_TOKENS // 1_000_000}M"
DATA_STATE = {"tag": DATA_TAG, "source": None, "builder_done": False, "builder_rc": None,
              "upload": "not needed", "upload_tries": 0, "tok_sha": None}

def _ntok(path):
    import numpy as np
    return int(np.fromfile(path, dtype=np.int32, count=3)[2])

def _shards():
    return sorted(glob.glob(os.path.join(MIX_DIR, "train_*.bin")))

def _capacity(shards):
    """Optimizer steps one pass over these shards gives (llm.c loader: (n-1)//(B*T) micro-batches per shard, 8 per step)."""
    return sum((_ntok(s) - 1) // 8192 for s in shards) // (TOKENS_PER_STEP // 8192)

def _tok_sha():
    return hashlib.sha256(open(os.path.join(TOK_DIR, "tokenizer.json"), "rb").read()).hexdigest()[:16]

def _data_ok():
    need = [os.path.join(TOK_DIR, "tokenizer.json"), os.path.join(TOK_DIR, "token_bytes.bin"), os.path.join(MIX_DIR, "val.bin")]
    return all(os.path.exists(p) for p in need) and sum(_ntok(s) for s in _shards()) >= int(TRAIN_TOKENS * 0.99)

def _write_meta():
    meta = {"tag": DATA_TAG, "vocab": VOCAB, "train_tokens": sum(_ntok(s) for s in _shards()),
            "shards": {os.path.basename(s): _ntok(s) for s in _shards()}, "val_tokens": _ntok(os.path.join(MIX_DIR, "val.bin")),
            "tok_sha": DATA_STATE["tok_sha"], "built": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": STATUS.get("commit")}
    json.dump(meta, open(os.path.join(DATA, "meta.json"), "w"), indent=1)
    return meta

def _builder_watch(p, log):
    """Hard-link each shard into MIX_DIR once the builder reports it written; mark the data done at the end."""
    with open(log, "a") as lf:
        for line in p.stdout:
            lf.write(line); lf.flush()
            m = re.match(r"wrote (\S+)", line)
            if m:
                name = os.path.basename(m.group(1))   # train lines carry a full path, the val line only "val.bin"
                src, dst = os.path.join(BUILD_DIR, name), os.path.join(MIX_DIR, name)
                try:
                    if os.path.exists(dst): os.remove(dst)
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy(src, dst + ".tmp"); os.replace(dst + ".tmp", dst)
                    if name.startswith("train_"):
                        print(f"  [data] shard {name} ready ({len(_shards())} so far, {clock()})", flush=True)
                except Exception as e:
                    print(f"  [data] could not publish {name}: {type(e).__name__}: {e}", flush=True)
    rc = p.wait()
    DATA_STATE["builder_rc"] = rc
    if rc == 0 and _data_ok():
        meta = _write_meta()
        DATA_STATE["builder_done"] = True
        DATA_STATE["upload"] = "pending"
        print(f"  [data] build finished at {clock()}: {meta['train_tokens']:,} train tokens in {len(meta['shards'])} shards; "
              f"uploading to {DATA_ID} in the background", flush=True)
    else:
        banner(f"[data] builder exited with code {rc}; training continues on the {len(_shards())} shard(s) that finished "
               f"(data will repeat). Log: {log}")

def _data():
    os.makedirs(MIX_DIR, exist_ok=True)
    meta_p = os.path.join(DATA, "meta.json")
    if os.path.exists(meta_p) and json.load(open(meta_p)).get("tag") == DATA_TAG and _data_ok():
        print("local data already present:", DATA)
        DATA_STATE.update(source="local", builder_done=True)
    else:
        ok, rc, out = hf(["ls", "--repo", DATA_ID, "--type", "dataset", "--prefix", DATA_TAG + "/"], 180)
        have = json.loads(out.strip().splitlines()[-1]) if ok else {}
        if not ok:
            print(f"could not list {DATA_ID} ({rc}); building the data here instead")
        if f"{DATA_TAG}/meta.json" in have:
            gb = sum(have.values()) / 1e9
            print(f"downloading {DATA_TAG} ({gb:.2f} GB) from {DATA_ID} ...", flush=True)
            t = time.time()
            ok, rc, out = hf(["pulldir", "--repo", DATA_ID, "--type", "dataset", "--prefix", DATA_TAG, "--dest", DATA, "--tries", "4"], 2400)
            if ok and _data_ok():
                print(f"  downloaded in {time.time() - t:.0f}s")
                DATA_STATE.update(source="hf", builder_done=True)
            else:
                print(f"download failed ({rc}): {out[-300:]}\nbuilding the data here instead")
        if not DATA_STATE["source"]:
            _start_build()
    DATA_STATE["tok_sha"] = _tok_sha()
    shards = _shards()
    ntrain = sum(_ntok(s) for s in shards)
    nval = _ntok(os.path.join(MIX_DIR, "val.bin"))
    print(f"train tokens ready now: {ntrain:,} in {len(shards)} shard(s) | val tokens: {nval:,} | tokenizer {DATA_STATE['tok_sha']}")
    if nval < VAL_TOKENS * 0.99:
        raise RuntimeError(f"val.bin has only {nval} tokens")
    if DATA_STATE["builder_done"]:
        cap = _capacity(shards)
        print(f"one pass = {cap:,} steps; this run needs {MAX_STEPS:,}" + ("" if cap >= MAX_STEPS else "  NOTE: the loader will wrap and repeat data"))

def _start_build():
    need = [m for m in ("datasets", "transformers", "tokenizers") if importlib.util.find_spec(m) is None]
    if need:
        sh([sys.executable, "-m", "pip", "install", "-q"] + need)
    DATA_STATE["source"] = "built"
    for f in glob.glob(os.path.join(MIX_DIR, "*")) + [os.path.join(DATA, "meta.json")]:
        if os.path.exists(f): os.remove(f)
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    os.makedirs(BUILD_DIR, exist_ok=True)
    t = time.time()
    if not os.path.exists(os.path.join(TOK_DIR, "token_bytes.bin")):
        print("training tokenizer ...", flush=True)
        subprocess.run([sys.executable, "python/train_tokenizer.py", "--out", TOK_DIR, "--vocab-size", str(VOCAB),
                        "--max-chars", str(TOK_CHARS)], cwd=REPO, check=True, env=HF_ENV)
        print(f"  tokenizer done in {time.time() - t:.0f}s")
    print(f"tokenizing ClimbMix in the background ({TRAIN_TOKENS:,} tokens, {SHARD_TOKENS:,} per shard) ...", flush=True)
    log = os.path.join(CONTENT, "data_build.log")
    p = subprocess.Popen(["nice", "-n", "10", sys.executable, "-u", "python/prepare_climbmix.py", "--tokenizer", TOK_DIR,
                          "--out", BUILD_DIR, "--target-tokens", str(TRAIN_TOKENS), "--val-tokens", str(VAL_TOKENS),
                          "--shard-tokens", str(SHARD_TOKENS), "--overwrite"],
                         cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=HF_ENV)
    DATA_STATE["builder_pid"] = p.pid
    threading.Thread(target=_builder_watch, args=(p, log), daemon=True).start()
    while not (_shards() and os.path.exists(os.path.join(MIX_DIR, "val.bin"))):   # wait for val + the first shard
        if p.poll() is not None and not _shards():
            time.sleep(2)
            raise RuntimeError(f"data builder exited ({p.returncode}) before the first shard. Last lines:\n" +
                               "".join(open(log).readlines()[-20:]))
        if time.time() - t > 3600:
            raise RuntimeError("first shard not ready after 60 min")
        time.sleep(5)
    print(f"first shard ready after {time.time() - t:.0f}s; training can start while the rest is built")

run_stage("data", _data, needs=["setup"])
''')

code(r'''
# ---- 4. Train. Uploads run in the background; re-running resumes from the HF checkpoint ----
RUN_LOCAL = os.path.join(CONTENT, "runs", RUN_NAME)
STAGE = os.path.join(CONTENT, "hf_stage")   # hard-linked snapshots for upload (the trainer may prune its own copies)
TRAIN_LOG = os.path.join(RUN_LOCAL, "train.log")
STATUS_PATH = os.path.join(RUN_LOCAL, "status.json")
TRAIN_WALL = 0.0
TRAIN_DEADLINE = T0 + BUDGET_H * 3600 - FINAL_RESERVE_MIN * 60
PROGRESS = {"step": 0, "max_steps": MAX_STEPS, "loss": None, "tok_s": None, "val": None, "bpb": None,
            "eta_h": None, "phase": "starting", "launches": 0, "restarts_for_data": 0, "crash_restarts": 0}
SYNC = {"ckpt_pushed_step": 0, "ckpt_pushed_at": None, "ckpt_push_fail": 0, "logs_pushed_at": None, "last_error": None}
SYNC_STOP = threading.Event()

def _done_steps(d):
    return sorted(int(os.path.basename(p)[5:]) for p in glob.glob(os.path.join(d, "DONE_*")))

def _write_status():
    s = {"run": RUN_NAME, "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "updated_ct": clock(),
         "notebook_h": round((time.time() - T0) / 3600, 2), "progress": PROGRESS, "data": DATA_STATE,
         "sync": SYNC, "stages": STATUS}
    tmp = STATUS_PATH + ".tmp"
    json.dump(s, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, STATUS_PATH)

def _push(files, msg, timeout, squash=False, repo=None, typ="model", tries=3):
    args = ["push", "--repo", repo or RUNS_ID, "--type", typ, "-m", msg, "--tries", tries]
    for local, remote in files:
        args += ["--file", f"{local}={remote}"]
    if squash:
        args.append("--squash")
    return hf(args, timeout)

def _push_ckpt(step, timeout=1200, extra=()):
    """Upload one resume checkpoint to <RUN_NAME>/ckpt/ (same path every time) and squash the repo history."""
    os.makedirs(STAGE, exist_ok=True)
    snap = os.path.join(STAGE, "state.bin")
    src = os.path.join(RUN_LOCAL, f"state_{step:08d}.bin")
    if os.path.exists(snap): os.remove(snap)
    try:
        os.link(src, snap)
    except OSError:
        shutil.copy(src, snap)
    meta = {"step": step, "max_steps": MAX_STEPS, "tok_sha": DATA_STATE["tok_sha"], "data_tag": DATA_TAG,
            "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": STATUS.get("commit"), "bytes": os.path.getsize(snap)}
    json.dump(meta, open(os.path.join(STAGE, "ckpt_meta.json"), "w"), indent=1)
    _write_status()
    files = [(snap, f"{RUN_NAME}/ckpt/state.bin"), (os.path.join(STAGE, "ckpt_meta.json"), f"{RUN_NAME}/ckpt/meta.json"),
             (TRAIN_LOG, f"{RUN_NAME}/train.log"), (STATUS_PATH, f"{RUN_NAME}/status.json")] + list(extra)
    t = time.time()
    ok, rc, out = _push(files, f"{RUN_NAME} checkpoint step {step}", timeout, squash=True)
    if ok:
        SYNC.update(ckpt_pushed_step=step, ckpt_pushed_at=time.time(), last_error=None)
        print(f"  [hf] checkpoint step {step} uploaded in {time.time() - t:.0f}s ({meta['bytes'] / 1e9:.2f} GB, {clock()})", flush=True)
    else:
        SYNC["ckpt_push_fail"] += 1
        SYNC["last_error"] = f"ckpt {step}: {rc}: {out[-200:]}"
        print(f"  [hf] checkpoint upload failed ({rc}); will retry. {out[-200:]}", flush=True)
    return ok

def _push_data(timeout=2700):
    DATA_STATE["upload_tries"] += 1
    files = [(os.path.join(DATA, "meta.json"), f"{DATA_TAG}/meta.json")]
    for f in sorted(glob.glob(os.path.join(TOK_DIR, "*"))):
        if os.path.isfile(f):
            files.append((f, f"{DATA_TAG}/tokenizer/{os.path.basename(f)}"))
    for f in _shards() + [os.path.join(MIX_DIR, "val.bin")]:
        files.append((f, f"{DATA_TAG}/climbmix/{os.path.basename(f)}"))
    t = time.time()
    ok, rc, out = _push(files, f"{DATA_TAG} (tokenizer + shards)", timeout, repo=DATA_ID, typ="dataset")
    DATA_STATE["upload"] = "ok" if ok else ("failed" if DATA_STATE["upload_tries"] >= 3 else "pending")
    print(f"  [hf] data upload to {DATA_ID}: {'ok' if ok else 'FAILED (' + str(rc) + ') ' + out[-200:]} in {time.time() - t:.0f}s", flush=True)
    return ok

def _syncer():
    """The only thread that uploads during training: data once, a checkpoint every CKPT_PUSH_MIN, logs every LOG_PUSH_MIN."""
    while not SYNC_STOP.wait(20):
        try:
            if DATA_STATE["upload"] == "pending" and HF_ENV:
                _push_data()
            if SYNC_STOP.is_set(): break
            steps = _done_steps(RUN_LOCAL)
            last = SYNC["ckpt_pushed_at"] or T0
            if steps and steps[-1] > SYNC["ckpt_pushed_step"] and time.time() - last >= CKPT_PUSH_MIN * 60:
                if _push_ckpt(steps[-1]):
                    SYNC["logs_pushed_at"] = time.time()
                continue
            if SYNC_STOP.is_set(): break
            if os.path.exists(TRAIN_LOG) and time.time() - (SYNC["logs_pushed_at"] or T0) >= LOG_PUSH_MIN * 60:
                _write_status()
                ok, rc, out = _push([(TRAIN_LOG, f"{RUN_NAME}/train.log"), (STATUS_PATH, f"{RUN_NAME}/status.json")],
                                    f"{RUN_NAME} logs step {PROGRESS['step']}", 180, tries=2)
                SYNC["logs_pushed_at"] = time.time()   # on failure, just try again next interval
                if not ok:
                    SYNC["last_error"] = f"logs: {rc}: {out[-200:]}"
                    print(f"  [hf] log upload failed ({rc}); training is unaffected", flush=True)
        except Exception as e:
            SYNC["last_error"] = f"syncer: {type(e).__name__}: {e}"
            print("  [hf] syncer error (training is unaffected):", type(e).__name__, e, flush=True)

def _resume_from_hf():
    """Fetch <RUN_NAME>/ckpt from the runs repo if it is newer than anything local."""
    ok, rc, out = hf(["ls", "--repo", RUNS_ID, "--type", "model", "--prefix", RUN_NAME + "/"], 180)
    if not ok:
        banner(f"could not list {RUNS_ID} ({rc}); not resuming from HF. {out[-200:]}")
        return
    files = json.loads(out.strip().splitlines()[-1])
    if f"{RUN_NAME}/train.log" in files and not os.path.exists(TRAIN_LOG):   # one log across sessions
        hf(["pull", "--repo", RUNS_ID, "--file", f"{RUN_NAME}/train.log={TRAIN_LOG}"], 300)
    if f"{RUN_NAME}/ckpt/meta.json" not in files or f"{RUN_NAME}/ckpt/state.bin" not in files:
        print("no checkpoint in", RUNS_ID, "for", RUN_NAME, "-> fresh run")
        return
    mp = os.path.join(CONTENT, "hf_ckpt_meta.json")
    ok, rc, out = hf(["pull", "--repo", RUNS_ID, "--file", f"{RUN_NAME}/ckpt/meta.json={mp}"], 120)
    if not ok:
        banner(f"checkpoint meta download failed ({rc}); starting fresh. {out[-200:]}")
        return
    meta = json.load(open(mp))
    step = int(meta["step"])
    if meta.get("tok_sha") != DATA_STATE["tok_sha"]:
        banner(f"The HF checkpoint (step {step}) used tokenizer {meta.get('tok_sha')}, but this data has {DATA_STATE['tok_sha']}. "
               "It can't be resumed on this data, so this is a FRESH run; the old checkpoint will be replaced.")
        return
    if meta.get("max_steps") != MAX_STEPS:
        print(f"NOTE: checkpoint was made with MAX_STEPS={meta.get('max_steps')}; continuing with {MAX_STEPS} (LR schedule follows {MAX_STEPS})")
    if step <= max(_done_steps(RUN_LOCAL) or [0]):
        print(f"local checkpoint is at least as new as HF step {step}; using local")
        return
    print(f"downloading HF checkpoint step {step} ({files[f'{RUN_NAME}/ckpt/state.bin'] / 1e9:.2f} GB) ...", flush=True)
    t = time.time()
    dst = os.path.join(RUN_LOCAL, f"state_{step:08d}.bin")
    ok, rc, out = hf(["pull", "--repo", RUNS_ID, "--file", f"{RUN_NAME}/ckpt/state.bin={dst}", "--tries", "4"], 1800)
    if not ok or os.path.getsize(dst) != files[f"{RUN_NAME}/ckpt/state.bin"]:
        banner(f"checkpoint download failed ({rc}); starting fresh. {out[-200:]}")
        if os.path.exists(dst): os.remove(dst)
        return
    open(os.path.join(RUN_LOCAL, f"DONE_{step:08d}"), "w").close()
    SYNC.update(ckpt_pushed_step=step, ckpt_pushed_at=time.time())
    print(f"  resuming from HF checkpoint step {step} (downloaded in {time.time() - t:.0f}s)")

_STEP_RE = re.compile(r"step (\d+)/(\d+)\s+loss ([\d.naninf]+)\s+lr_mult ([\d.]+)\s+([\d.]+)s\s+(\d+) tok/s")

def _glance(step, recent):
    left = MAX_STEPS - step
    sps = statistics.median(recent) if recent else 3.12
    eta = left * sps
    PROGRESS["eta_h"] = round(eta / 3600, 2)
    tps = PROGRESS["tok_s"] or 0
    vb = f"  val {PROGRESS['val']:.3f} bpb {PROGRESS['bpb']:.3f}" if PROGRESS["val"] is not None else ""
    print(f"[{clock()}] step {step:,}/{MAX_STEPS:,} ({100 * step / MAX_STEPS:4.1f}%)  loss {PROGRESS['loss']:.4f}  "
          f"{tps:,} tok/s  {sps:.2f} s/step{vb}  | {(time.time() - T0) / 3600:.2f} h in, ETA {eta / 3600:.2f} h (~{clock(time.time() + eta)})",
          flush=True)

def _train():
    global TRAIN_WALL, TRAINER_PROC
    os.makedirs(RUN_LOCAL, exist_ok=True)
    _resume_from_hf()
    local = max(_done_steps(RUN_LOCAL) or [0])
    print(f"starting at step {local}" if local else "fresh run", "| training must stop by", clock(TRAIN_DEADLINE))
    th = threading.Thread(target=_syncer, daemon=True); th.start()
    finished = False
    with open(TRAIN_LOG, "a") as logf:
        while True:
            remaining = TRAIN_DEADLINE - time.time()
            if WATCHDOG_FIRED:
                print("watchdog fired; not restarting the trainer"); break
            if remaining < 120:
                print("training time budget used up"); break
            shards = _shards()
            n_launch, cap = len(shards), _capacity(shards)
            start = max(_done_steps(RUN_LOCAL) or [0])
            cmd = [BIN, "-i", os.path.join(MIX_DIR, "train_*.bin"), "-j", os.path.join(MIX_DIR, "val.bin"),
                   "-e", "d12", "-o", RUN_LOCAL, "-b", "16", "-t", "512", "-d", str(TOKENS_PER_STEP),
                   "-x", str(MAX_STEPS), "-v", str(VAL_EVERY), "-m", str(VAL_BATCHES),
                   "-n", str(CKPT_EVERY), "-y", "1", "--keep-ckpts", "2", "--time-limit", str(int(remaining)),
                   "--vocab", str(VOCAB), "--token-bytes", os.path.join(TOK_DIR, "token_bytes.bin")]
            PROGRESS["launches"] += 1
            PROGRESS["phase"] = "training"
            print(f"launch {PROGRESS['launches']} at step {start}: {n_launch} shard(s) = {cap:,} steps of data, "
                  f"time limit {remaining / 3600:.2f} h\n" + " ".join(cmd), flush=True)
            logf.write(f"\n=== session {time.strftime('%Y-%m-%d %H:%M:%S')}  commit {STATUS.get('commit')}  gpu {GPU_NAME}  "
                       f"launch {PROGRESS['launches']}  shards {n_launch}\n" + " ".join(cmd) + "\n"); logf.flush()
            t = time.time()
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=TRAIN_ENV)
            TRAINER_PROC = p
            restart, done_seen, limit_seen, recent = False, False, False, []
            try:
                for line in p.stdout:
                    logf.write(line); logf.flush()
                    m = _STEP_RE.match(line)
                    if m:
                        step = int(m.group(1))
                        PROGRESS.update(step=step, loss=float(m.group(3)), tok_s=int(m.group(6)))
                        recent = (recent + [float(m.group(5))])[-50:]
                        if step % 50 == 0 or step == start + 1:
                            _glance(step, recent)
                        # More shards are ready than this launch can see: switch at the next checkpoint.
                        if len(_shards()) > n_launch and (DATA_STATE["builder_done"] or step + CKPT_EVERY >= cap):
                            newest = max(_done_steps(RUN_LOCAL) or [0])
                            if newest > start and step == newest + 1:
                                print(f"  [data] {len(_shards())} shards ready now; restarting the trainer from checkpoint "
                                      f"{newest} to use them", flush=True)
                                restart = True
                                p.send_signal(signal.SIGTERM)
                                break
                        continue
                    v = re.search(r"val loss ([\d.]+)\s+bpb ([\d.]+)", line)
                    if v:
                        PROGRESS.update(val=float(v.group(1)), bpb=float(v.group(2)))
                    done_seen |= line.strip() == "done"
                    limit_seen |= "time limit" in line and "reached" in line
                    print(line, end="", flush=True)
                try:
                    rc = p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    p.kill(); rc = p.wait()
            finally:
                if p.poll() is None:            # cell interrupted: stop the trainer
                    p.send_signal(signal.SIGINT); time.sleep(5); p.kill()
                TRAIN_WALL += time.time() - t
                logf.write(f"=== trainer exit {p.returncode} after {time.time() - t:.0f}s\n"); logf.flush()
            if restart:
                PROGRESS["restarts_for_data"] += 1
                continue
            if rc == 0 and done_seen:
                finished = True; break
            if rc == 0 and limit_seen:
                banner(f"trainer hit its time limit at step {PROGRESS['step']} of {MAX_STEPS}; the LR schedule did not quite finish")
                break
            if WATCHDOG_FIRED:
                print("watchdog fired; not restarting the trainer"); break
            tail = "".join(open(TRAIN_LOG).readlines()[-25:])
            if PROGRESS["crash_restarts"] < 2 and TRAIN_DEADLINE - time.time() > 900 and _done_steps(RUN_LOCAL):
                PROGRESS["crash_restarts"] += 1
                banner(f"trainer exited with code {rc}; restarting from the last checkpoint "
                       f"(attempt {PROGRESS['crash_restarts']}/2). Last lines:\n{tail}")
                time.sleep(10)
                continue
            raise RuntimeError(f"trainer exited with code {rc}. Last lines:\n{tail}")
    PROGRESS["phase"] = "finished" if finished else "stopped"
    print(f"training wall time this session: {TRAIN_WALL / 3600:.2f} h, reached step {PROGRESS['step']}/{MAX_STEPS}")

run_stage("train", _train, needs=["build", "data"])
''')

code(r'''
# ---- 5. Summary + 3 samples (from a temporary local export that is deleted afterwards) ----
SUMMARY = {}

def _summary():
    global SUMMARY
    if not os.path.exists(TRAIN_LOG):
        raise RuntimeError("no training log (training did not start)")
    lines = open(TRAIN_LOG).read().splitlines()
    steps, sess_first = [], 0
    for l in lines:
        if l.startswith("=== session"):
            sess_first = 0
        m = _STEP_RE.match(l)
        if m:
            sess_first += 1
            steps.append(dict(step=int(m.group(1)), loss=float(m.group(3)), secs=float(m.group(5)),
                              tps=int(m.group(6)), warm=sess_first > 5))
    vals = [(float(a), float(b)) for a, b in re.findall(r"val loss ([\d.]+)\s+bpb ([\d.]+)", "\n".join(lines))]
    if not steps:
        raise RuntimeError("no step lines in the log")
    last = steps[-1]["step"]
    warm = [s for s in steps if s["warm"]] or steps
    tail = [s["loss"] for s in steps[-20:]]
    SUMMARY = {
        "run": RUN_NAME, "commit": STATUS.get("commit"), "gpu": GPU_NAME,
        "steps_done": last, "max_steps": MAX_STEPS, "tokens_processed": last * TOKENS_PER_STEP,
        "steady_tok_per_s_median": int(statistics.median([s["tps"] for s in warm])),
        "steady_s_per_step_median": round(statistics.median([s["secs"] for s in warm]), 3),
        "final_train_loss": steps[-1]["loss"],
        "final_train_loss_avg20": round(sum(tail) / len(tail), 4),
        "final_val_loss": vals[-1][0] if vals else None,
        "final_val_bpb": vals[-1][1] if vals else None,
        "val_curve": vals,
        "fp16_overflow_skips": sum("gradient overflow" in l for l in lines),
        "self_checks": [l.strip() for l in lines if "check" in l.lower() and ("gemm" in l.lower() or "fused" in l.lower())][-2:],
        "finished_lr_schedule": last >= MAX_STEPS,
        "train_wall_h_this_session": round(TRAIN_WALL / 3600, 2),
        "notebook_wall_h": round((time.time() - T0) / 3600, 2),
        "launches": PROGRESS["launches"], "restarts_for_data": PROGRESS["restarts_for_data"], "crash_restarts": PROGRESS["crash_restarts"],
        "data": {k: DATA_STATE.get(k) for k in ("tag", "source", "upload", "tok_sha")},
        "unique_train_steps_available": _capacity(_shards()) if _shards() else None,
        "stage_secs": {k: v.get("secs") for k, v in STATUS.items() if isinstance(v, dict)},
    }
    txt = "\n".join(f"{k}: {v}" for k, v in SUMMARY.items())
    print(txt)
    open(os.path.join(RUN_LOCAL, "summary.txt"), "w").write(txt + "\n")
    json.dump(SUMMARY, open(os.path.join(RUN_LOCAL, "summary.json"), "w"), indent=1)

SAMPLE_PY = r"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
out, dst = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True).to(dev).eval()
g = torch.Generator(device=dev).manual_seed(0)
samples = []
for prompt in ("The capital of France is", "Once upon a time,", "In this article we explain how"):
    ids = torch.tensor([[tok.bos_token_id] + tok.encode(prompt, add_special_tokens=False)], device=dev)
    with torch.no_grad():
        for _ in range(64):
            logits = model(input_ids=ids[:, -512:]).logits[:, -1, :].float() / 0.8
            v, _ = torch.topk(logits, 40)
            logits[logits < v[:, -1:]] = -float("inf")
            ids = torch.cat([ids, torch.multinomial(torch.softmax(logits, -1), 1, generator=g)], 1)
    samples.append(tok.decode(ids[0, 1:].tolist()))
    print("---", samples[-1].replace("\n", " "), flush=True)
open(dst, "w").write("\n---\n".join(samples) + "\n")
"""

def _samples():
    if not DO_SAMPLE:
        return
    s = max(_done_steps(RUN_LOCAL) or [0])
    ckpt = os.path.join(RUN_LOCAL, f"model_{s:08d}.bin")
    if not s or not os.path.exists(ckpt):
        raise RuntimeError("no local model file for the last checkpoint (nothing was trained this session)")
    out = os.path.join(CONTENT, "tmp_export")
    shutil.rmtree(out, ignore_errors=True)
    try:
        r = subprocess.run([sys.executable, "python/export_hf.py", "--input", ckpt, "--output", out, "--tokenizer", TOK_DIR],
                           cwd=REPO, timeout=600, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"temporary export failed: {(r.stdout + r.stderr)[-1500:]}")
        open(os.path.join(CONTENT, "sample.py"), "w").write(SAMPLE_PY)
        r = subprocess.run([sys.executable, os.path.join(CONTENT, "sample.py"), out, os.path.join(RUN_LOCAL, "samples.txt")],
                           cwd=REPO, timeout=900, capture_output=True, text=True)
        print(r.stdout[-3000:])
        if r.returncode != 0:
            raise RuntimeError(f"sampling failed: {r.stderr[-1500:]}")
        print(f"samples from step {s} (the export was temporary and is deleted now)")
    finally:
        shutil.rmtree(out, ignore_errors=True)   # no model is kept

run_stage("summary", _summary, needs=["setup"])
run_stage("samples", _samples, needs=["setup"])
''')

code(r'''
# ---- 6. Final upload to HF, verify it, then ALWAYS disconnect ----
FINAL_HARD_TIMEOUT_S = _knob("FINAL_HARD_TIMEOUT_S", 1800)   # the disconnect happens after this even if uploads hang

def _final():
    SYNC_STOP.set()
    p = _HF_CHILD.get("p")   # don't wait on a background upload; the final upload below covers it
    if p is not None and p.poll() is None:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
    PROGRESS["phase"] = "final upload"
    STATUS["notebook_wall_h"] = round((time.time() - T0) / 3600, 2)
    _write_status()
    extra = []
    for f in ("summary.json", "summary.txt", "samples.txt"):
        if os.path.exists(os.path.join(RUN_LOCAL, f)):
            extra.append((os.path.join(RUN_LOCAL, f), f"{RUN_NAME}/{f}"))
    for f in ("build.log", "data_build.log"):
        if os.path.exists(os.path.join(CONTENT, f)):
            extra.append((os.path.join(CONTENT, f), f"{RUN_NAME}/{f}"))
    steps = _done_steps(RUN_LOCAL)
    expect = [r for _, r in extra] + [f"{RUN_NAME}/train.log", f"{RUN_NAME}/status.json"]
    if steps and steps[-1] > SYNC["ckpt_pushed_step"]:
        ok = _push_ckpt(steps[-1], timeout=1200, extra=extra)
        expect += [f"{RUN_NAME}/ckpt/state.bin", f"{RUN_NAME}/ckpt/meta.json"]
        if not ok:   # at least get the small files up
            _push(extra + [(TRAIN_LOG, f"{RUN_NAME}/train.log"), (STATUS_PATH, f"{RUN_NAME}/status.json")], f"{RUN_NAME} final logs", 300)
    else:
        ok, rc, out = _push(extra + [(TRAIN_LOG, f"{RUN_NAME}/train.log"), (STATUS_PATH, f"{RUN_NAME}/status.json")],
                            f"{RUN_NAME} final logs", 300)
        print(f"  [hf] final logs/summary upload: {'ok' if ok else 'FAILED (' + str(rc) + ') ' + out[-200:]}")
    if DATA_STATE.get("upload") == "pending" and time.time() < T0 + WATCHDOG_H * 3600 - 1800:
        _push_data(timeout=1200)
    # Verify: list what is really in the repos.
    ok, rc, out = hf(["ls", "--repo", RUNS_ID, "--type", "model", "--prefix", RUN_NAME + "/"], 120)
    if ok:
        have = json.loads(out.strip().splitlines()[-1])
        print(f"\n{RUNS_ID} / {RUN_NAME}:")
        for k in sorted(have):
            print(f"  {k}  {have[k] / 1e6:.1f} MB")
        missing = [e for e in expect if e not in have]
        print("verified: all final files are on HF" if not missing else f"MISSING on HF: {missing}")
        STATUS["hf_verified"] = not missing
    else:
        print(f"could not list {RUNS_ID} to verify ({rc})")
        STATUS["hf_verified"] = False
    ok, rc, out = hf(["ls", "--repo", DATA_ID, "--type", "dataset", "--prefix", DATA_TAG + "/"], 120)
    if ok:
        have = json.loads(out.strip().splitlines()[-1])
        print(f"{DATA_ID} / {DATA_TAG}: {len(have)} files, {sum(have.values()) / 1e9:.2f} GB"
              + ("" if f"{DATA_TAG}/meta.json" in have else "  (data NOT uploaded; the next run rebuilds it)"))

try:
    if HF_ENV:
        th = threading.Thread(target=_final, daemon=True); th.start()
        th.join(FINAL_HARD_TIMEOUT_S)
        if th.is_alive():
            banner(f"final upload still running after {FINAL_HARD_TIMEOUT_S}s; disconnecting anyway")
    else:
        print("no HF access (setup failed), nothing to upload")
except BaseException as e:
    banner(f"final stage error: {type(e).__name__}: {e}")
finally:
    print("\nstage status:")
    for k, v in STATUS.items():
        if isinstance(v, dict):
            print(f"  {k:8s} {'ok' if v.get('ok') else 'FAILED'} {v.get('secs', '')}s {v.get('error', '')}")
    print(f"notebook wall time: {(time.time() - T0) / 3600:.2f} h (started {clock(T0)}, now {clock()})")
    if AUTO_DISCONNECT:
        _unassign("end of notebook")
''')

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "name": "TinyForge-8hr.ipynb"},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
    "cells": [],
}
for kind, src in cells:
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    nb["cells"].append(c)
out = sys.argv[1] if len(sys.argv) > 1 else "TinyForge-8hr.ipynb"
json.dump(nb, open(out, "w"), indent=1)
# Also write the code cells as one script for box tests.
open(out.replace(".ipynb", "_script.py"), "w").write("\n\n".join(s for k, s in cells if k == "code") + "\n")
print("wrote", out)
