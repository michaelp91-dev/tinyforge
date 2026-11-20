import json, sys

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r'''
# TinyForge d12: official pre-training on Kaggle (2x T4, multi-session, Hugging Face storage)

d12 (135M params) with the autoresearch config: `-b 16 -t 512 -d 32768` (32,768 tokens per optimizer step), all other trainer defaults (warmup 40, warmdown 0.65, matrix lr 0.02, embed lr 0.3, unembed lr 0.008). On 2 GPUs that is 16 x 512 x 2 = 16,384 tokens per micro-step and 2 accumulation steps, so the total stays 32,768. The LR schedule is set once for `MAX_STEPS` = 72,000 steps (2.36B tokens) and spans several Kaggle sessions. Nothing touches Google Drive.

**Setup (once):**
1. Settings (right panel) > **Accelerator: GPU T4 x2**, **Internet: on** (needs a phone-verified Kaggle account).
2. Add-ons > **Secrets** > add **`HF_TOKEN`** (a Hugging Face token with **write** access) and tick it for this notebook. The token is never printed.

**Each session:** click **Save Version > Save & Run All (Commit)**. It runs headless for up to about 11.5 h, saves to Hugging Face, and ends by itself (a commit stops using GPU quota when it finishes). **Run the same notebook again for the next session**: it resumes from the checkpoint on Hugging Face. When the run reaches `MAX_STEPS`, the last session exports the model to HF format + GGUF and uploads them. Running it again after that only re-checks the export.

What each session does:
1. Clones branch `kaggle-pretrain`, reads `HF_TOKEN`, checks that `<you>/tinyforge-data` (dataset) and `<you>/tinyforge-runs` (model) are **private**, and arms a watchdog that stops everything at `WATCHDOG_H` (11.75 h), before Kaggle's 12 h limit.
2. Builds the fp16 T4 trainer with NCCL for the two GPUs (falls back to one GPU if NCCL can't build or start).
3. Data `climbmix_v32768_2400M` (2.4B tokens, 24 shards, same tokenizer as the 8-hour run): downloaded from `tinyforge-data` if it is there. **First session:** the tokenizer is copied from `climbmix_v32768_610M`, ClimbMix is tokenized in the background, training starts once the first shard is ready, and each finished shard is uploaded as soon as it is written.
4. Trains until the session's time budget. One resume checkpoint is uploaded to `tinyforge-runs/d12_kaggle/ckpt/` about every 40 min (overwritten, history squashed) and again at the time-limit stop. `train.log`, `status.json` and `sessions.json` go up every 10 min.
5. End of session: summary, 3 samples, final upload and a file listing to verify it.

`INIT_FROM`: `"fresh"` (default) starts from random weights. `"d12_8hr"` starts from the 8-hour run's checkpoint (step 7,976, trained at 65,536 tokens/step) instead; it only matters for the very first session of a run name.
''')

code(r'''
# ---- 1. Setup: config, HF token + repos, watchdog, GPU check, clone ----
import os, sys, re, glob, json, time, shutil, signal, subprocess, threading, traceback, statistics, importlib.util, hashlib
from datetime import datetime

LOCAL_TEST = os.environ.get("TF_LOCAL_TEST") == "1"   # box-side dry run only; ignore on Kaggle
def _knob(name, default):
    """Box tests may override a knob with TF_<NAME>; on Kaggle the value below is used."""
    v = os.environ.get("TF_" + name) if LOCAL_TEST else None
    if v is None:
        return default
    if isinstance(default, bool):
        return v.lower() in ("1", "true", "yes")
    return type(default)(v)

# Knobs
REPO_URL      = "https://github.com/michaelp91-dev/tinyforge"
BRANCH        = "kaggle-pretrain"
HF_SECRET     = "HF_TOKEN"            # Kaggle secret with a WRITE token
HF_USER       = ""                    # blank = the account that owns the token
DATA_REPO     = "tinyforge-data"      # private dataset repo: tokenizer + shards
RUNS_REPO     = "tinyforge-runs"      # private model repo: logs, summary, one checkpoint, final export
RUN_NAME      = _knob("RUN_NAME", "d12_kaggle")   # folder in the runs repo; every session with this name resumes it
INIT_FROM     = _knob("INIT_FROM", "fresh")       # "fresh" or "d12_8hr" (start from that run's checkpoint); first session only
MAX_STEPS     = _knob("MAX_STEPS", 72000)         # 72,000 x 32,768 = 2.36B tokens; LR schedule spans all sessions
TOTAL_BATCH   = 32768                             # tokens per optimizer step (autoresearch best)
B, T          = 16, 512                           # micro-batch per GPU
MAX_GPUS      = _knob("MAX_GPUS", 2)
TRAIN_TOKENS  = _knob("TRAIN_TOKENS", 2_400_000_000)   # one pass = ~73,200 steps >= MAX_STEPS, so nothing repeats
SHARD_TOKENS  = _knob("SHARD_TOKENS", 100_000_000)
TOKENIZER_FROM = _knob("TOKENIZER_FROM", "climbmix_v32768_610M")   # reuse the 8-hour run's tokenizer
VAL_TOKENS    = _knob("VAL_TOKENS", 262_144)
VOCAB         = 32768
TOK_CHARS     = _knob("TOK_CHARS", 50_000_000)    # only if no tokenizer can be downloaded
VAL_EVERY     = _knob("VAL_EVERY", 2000)          # about every 35 min
VAL_BATCHES   = 31                                # 31 x 8192 = the whole val set
CKPT_EVERY    = _knob("CKPT_EVERY", 1000)         # local checkpoint, about every 17 min
BUDGET_H      = _knob("BUDGET_H", 11.5)           # whole session, setup to last upload (Kaggle limit is 12 h)
FINAL_RESERVE_MIN = _knob("FINAL_RESERVE_MIN", 25.0)  # kept free for summary, samples, export, final upload
CKPT_PUSH_MIN = _knob("CKPT_PUSH_MIN", 40.0)      # upload the newest checkpoint this often
LOG_PUSH_MIN  = _knob("LOG_PUSH_MIN", 10.0)       # upload train.log + status.json this often
WATCHDOG_H    = _knob("WATCHDOG_H", 11.75)        # hard stop: kill everything after this many hours
STARTUP_TIMEOUT_S = _knob("STARTUP_TIMEOUT_S", 900)   # 2-GPU launch with no step line by then -> one GPU
MIN_LAUNCH_S  = _knob("MIN_LAUNCH_S", 300)        # don't (re)launch the trainer with less time than this left
MIN_RESTART_S = _knob("MIN_RESTART_S", 900)       # crash restarts only with at least this much time left
NCCL_P2P_OFF  = True                              # T4 pairs on Kaggle: shared-memory transport, no P2P
DO_SAMPLE     = _knob("DO_SAMPLE", True)
TZ_NAME       = "America/Chicago"
TOKENS_PER_STEP = TOTAL_BATCH

CONTENT = os.environ.get("TF_CONTENT", "/tmp/tf")          # scratch (wiped after the session)
OUT_DIR = os.environ.get("TF_OUT", "/kaggle/working")       # Kaggle "Output": small files + final export only
T0 = time.time()
STATUS = {}
HF_ENV = None
HF_TOKEN_STR = None
TRAINER_PROCS = []
WATCHDOG_FIRED = False
NGPU = 0
os.makedirs(CONTENT, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

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
    """Run one notebook stage; never raise, so Run All reaches the last cell."""
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

def _kill_trainers(sig=signal.SIGKILL):
    for p in list(TRAINER_PROCS):
        try:
            if p.poll() is None:
                p.send_signal(sig)
        except Exception:
            pass

def _end_session(reason):
    """Kaggle has no 'disconnect' call. A committed run ends when the notebook ends; only the watchdog
    needs to force it, by killing the kernel process (the version is then marked as failed, but
    everything that matters is already on Hugging Face)."""
    print(f"Ending the session now ({reason}).", flush=True)
    if LOCAL_TEST:
        open(os.path.join(CONTENT, "end_session.log"), "a").write(f"{time.time():.0f} {reason}\n")
        return
    os.kill(os.getpid(), signal.SIGKILL)

def _watchdog():
    deadline = T0 + WATCHDOG_H * 3600
    while time.time() < deadline:
        time.sleep(min(30, max(1, deadline - time.time())))
    global WATCHDOG_FIRED
    WATCHDOG_FIRED = True
    banner(f"WATCHDOG: {WATCHDOG_H} h reached, stopping the trainer, saving logs, ending the session.")
    _kill_trainers()
    try:
        p = _HF_CHILD.get("p")
        if p is not None and p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        files = [f"{f}={RUN_NAME}/{os.path.basename(f)}" for f in (globals().get("TRAIN_LOG"), globals().get("STATUS_PATH"))
                 if f and os.path.exists(f)]
        if files and HF_ENV and globals().get("RUNS_ID"):
            args = ["push", "--repo", globals()["RUNS_ID"], "--type", "model", "--tries", "1", "-m", "watchdog logs"]
            for f in files: args += ["--file", f]
            # bypass the lock: the watchdog must not wait on a stuck upload
            subprocess.run([sys.executable, os.path.join(REPO, "python", "hf_sync.py")] + args,
                           env=HF_ENV, timeout=90, capture_output=True, start_new_session=True)
    except Exception as e:
        print("watchdog log upload failed:", type(e).__name__, flush=True)
    _end_session("watchdog")

def _process_watchdog():
    """A separate OS process that kills the kernel 5 min after the thread watchdog, even if the kernel hangs."""
    pidf = os.path.join(CONTENT, "tf_watchdog.pid")
    try:   # a re-run replaces the old one
        os.kill(int(open(pidf).read()), signal.SIGKILL)
    except Exception:
        pass
    deadline = T0 + WATCHDOG_H * 3600 + _knob("PROC_WATCHDOG_EXTRA_S", 300)
    kernel = os.getpid()
    if LOCAL_TEST:
        action = f"open({os.path.join(CONTENT, 'proc_watchdog.log')!r}, 'a').write('fired\\n')"
    else:
        action = f"os.kill({kernel}, signal.SIGKILL)"
    prog = ("import os, signal, time\n"
            f"time.sleep(max(0, {deadline} - time.time()))\n"
            "try:\n"
            f"    os.kill({kernel}, 0)\n"
            "except OSError:\n"
            "    raise SystemExit\n"
            f"{action}\n")
    p = subprocess.Popen([sys.executable, "-c", prog], start_new_session=True,
                         stdout=open(os.path.join(CONTENT, "tf_watchdog.log"), "a"), stderr=subprocess.STDOUT)
    open(pidf, "w").write(str(p.pid))

def _read_secret():
    if LOCAL_TEST:
        return os.environ.get("TF_TEST_TOKEN", "hf_test_token_not_real")
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(HF_SECRET)
    except Exception as e:
        err = type(e).__name__
    try:   # also works on Colab
        from google.colab import userdata
        return userdata.get(HF_SECRET)
    except Exception:
        pass
    if os.environ.get(HF_SECRET):
        return os.environ[HF_SECRET]
    raise RuntimeError(f"Could not read the Kaggle secret '{HF_SECRET}' ({err}). Add-ons > Secrets: add it (a Hugging Face "
                       "token with WRITE access) and tick it for this notebook, then run again.") from None

def _setup():
    global REPO, GPU_CC, GPU_NAME, HF_ENV, HF_TOKEN_STR, HF_USER, DATA_ID, RUNS_ID, NGPU
    if INIT_FROM not in ("fresh", "d12_8hr") and not INIT_FROM.startswith("d12"):
        raise RuntimeError(f"INIT_FROM must be 'fresh' or a run name like 'd12_8hr', not {INIT_FROM!r}")
    if TOTAL_BATCH % (B * T) != 0:
        raise RuntimeError("TOTAL_BATCH must be a multiple of B*T")
    if LOCAL_TEST:
        GPU_NAME, GPU_CC = os.environ.get("TF_GPU_NAME", "Tesla T4"), os.environ.get("TF_GPU_CC", "7.5")
        found = int(os.environ.get("TF_NGPU", "2"))
    else:
        r = sh("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader", check=False)
        rows = [l for l in r.stdout.strip().splitlines() if l.strip()] if r.returncode == 0 else []
        if not rows:
            raise RuntimeError("No GPU visible. Settings > Accelerator > GPU T4 x2, then run again.")
        GPU_NAME, GPU_CC, mem = [x.strip() for x in rows[0].split(",")]
        found = len(rows)
        print(f"GPUs: {found} x {GPU_NAME}, compute capability {GPU_CC}, {mem}")
        if "T4" not in GPU_NAME:
            print(f"NOTE: this notebook is tuned for T4s; on {GPU_NAME} step times will differ.")
    NGPU = max(1, min(found, MAX_GPUS))
    if TOTAL_BATCH % (B * T * NGPU) != 0:
        NGPU = 1
    print(f"using {NGPU} GPU(s): {B}x{T}x{NGPU} = {B * T * NGPU:,} tokens per micro-step, "
          f"{TOTAL_BATCH // (B * T * NGPU)} accumulation step(s), {TOTAL_BATCH:,} tokens per optimizer step")
    if found < 2:
        banner("Only one GPU is visible. Settings > Accelerator should be 'GPU T4 x2' (one GPU runs at about half speed).")

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

    tok = _read_secret()
    if not tok:
        raise RuntimeError(f"The secret '{HF_SECRET}' is empty.")
    HF_TOKEN_STR = tok
    HF_ENV = dict(os.environ, HF_TOKEN=tok, HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_TELEMETRY="1")
    try:
        import huggingface_hub
        ver = tuple(int(x) for x in re.findall(r"\d+", huggingface_hub.__version__)[:2])
    except Exception:
        ver = (0, 0)
    if ver < (0, 30) and not LOCAL_TEST:
        print("upgrading huggingface_hub ...")
        sh([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub>=0.30"])
    ok, rc, out = hf(["whoami", "--repo", "-"], 60)
    if not ok:
        raise RuntimeError(f"Hugging Face login failed ({rc}): {out[-400:]}. Check that {HF_SECRET} is a valid token "
                           "and that Internet is on in the notebook settings.")
    user = out.strip().splitlines()[-1]
    HF_USER = HF_USER or user
    DATA_ID, RUNS_ID = f"{HF_USER}/{DATA_REPO}", f"{HF_USER}/{RUNS_REPO}"
    for rid, typ in ((DATA_ID, "dataset"), (RUNS_ID, "model")):
        ok, rc, out = hf(["ensure", "--repo", rid, "--type", typ], 120)
        if not ok:
            raise RuntimeError(f"could not open or create {typ} repo {rid} ({rc}): {out[-400:]}. Does the token have WRITE access?")
        info = json.loads(out.strip().splitlines()[-1])
        if not info["private"]:
            raise RuntimeError(f"{rid} exists and is PUBLIC. Make it private on huggingface.co, then run again.")
        print(f"HF {typ} repo {rid}: {'created (private)' if info['created'] else 'found (private)'}")
    print("run:", RUN_NAME, "| session started", clock(T0), "| budget", BUDGET_H, "h -> done by about", clock(T0 + BUDGET_H * 3600))

run_stage("setup", _setup)
threading.Thread(target=_watchdog, daemon=True).start()
try:
    _process_watchdog()
except Exception as e:
    print("process watchdog not started:", e)
print(f"watchdog armed: everything stops at {WATCHDOG_H} h ({clock(T0 + WATCHDOG_H * 3600)}) no matter what")
''')
code(r'''
# ---- 2. Build the fp16 T4 trainer (with NCCL for 2 GPUs) using this session's CUDA ----
TRAIN_ENV = dict(os.environ)
BIN = None

def _find_nvcc():
    cands = [shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"] + sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"), reverse=True)
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise RuntimeError("nvcc not found (looked on PATH, /usr/local/cuda/bin, /usr/local/cuda-*/bin). "
                       "Settings > Accelerator must be a GPU (T4 x2).")

def _site_roots():
    import site
    return site.getsitepackages() + [site.getusersitepackages()]

def _pip_cublas(major):
    """Install the cuBLAS wheel matching the toolkit's major version; return its (include, lib) dirs."""
    pkg = "nvidia-cublas-cu12" if major == 12 else f"nvidia-cublas>={major},<{major + 1}"
    print("installing", pkg, "...")
    sh([sys.executable, "-m", "pip", "install", "-q", pkg])
    roots = _site_roots()
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

def _find_nccl(major):
    """NCCL headers + lib: the pip wheel (Kaggle's torch ships one) or the system. Returns make flags."""
    def scan():
        for root in _site_roots():
            for sub in ("nvidia/nccl", f"nvidia/cu{major}"):
                d = os.path.join(root, sub)
                if os.path.exists(os.path.join(d, "include", "nccl.h")) and glob.glob(os.path.join(d, "lib", "libnccl.so.2*")):
                    return d
        return None
    d = scan()
    if d is None and os.path.exists("/usr/include/nccl.h"):
        return ["USE_NCCL=1"], None
    if d is None:
        pkg = "nvidia-nccl-cu12" if major == 12 else f"nvidia-nccl-cu{major}"
        print("installing", pkg, "...")
        sh([sys.executable, "-m", "pip", "install", "-q", pkg])
        d = scan()
    if d is None:
        raise RuntimeError("NCCL not found (no nccl.h in the pip wheels or /usr/include)")
    return ["USE_NCCL=1", f"NCCL_HOME={d}"], os.path.join(d, "lib")

def _missing_libs(binary, env):
    r = sh(["ldd", binary], check=False, env=env)
    return [l.split()[0] for l in r.stdout.splitlines() if "not found" in l]

def _make(make, nvcc, extra, env, log, label):
    """One build attempt. Returns None on success, else the compiler output."""
    nv = f"{nvcc} {extra}".strip()
    sh(["rm", "-f", os.path.join(REPO, "train_tinyforge")])
    r = subprocess.run(make + [f"NVCC={nv}"], capture_output=True, text=True, env=env)
    open(log, "a").write(f"=== {label}: {' '.join(make)} NVCC={nv}\n{r.stdout}{r.stderr}\n")
    return None if r.returncode == 0 else (r.stdout + r.stderr)

def _build():
    global BIN, TRAIN_ENV, NGPU
    if LOCAL_TEST and os.environ.get("TF_FAKE_BIN"):   # box test: no CUDA here
        BIN = os.environ["TF_FAKE_BIN"]
        if os.environ.get("TF_FAKE_NCCL_BUILD_FAIL") == "1" and NGPU > 1:
            banner("box test: NCCL build 'failed'; falling back to ONE GPU")
            NGPU = 1
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
    base = ["make", "-C", REPO, "train_tinyforge", "GPU=t4"]
    cc = GPU_CC.replace(".", "")
    if cc != "75":   # not a T4: same fp16 code path, compiled for this GPU
        base.append(f"NVCC_ARCH=-gencode arch=compute_{cc},code=sm_{cc} -DENABLE_FP16 -DTF_FP16_GEMMEX")
    extra = os.environ.get("TF_NVCC_EXTRA", "")
    lib_dirs = [d for d in (os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "lib"),
                            os.path.join(cuda_home, "targets/x86_64-linux/lib")) if os.path.isdir(d)]
    log = os.path.join(CONTENT, "build.log")
    nccl_flags, nccl_lib = [], None
    if NGPU > 1:
        try:
            nccl_flags, nccl_lib = _find_nccl(major)
            print("NCCL:", " ".join(nccl_flags))
        except Exception as e:
            banner(f"NCCL setup failed ({e}); training on ONE GPU")
            NGPU = 1
    make = base + nccl_flags
    err = _make(make, nvcc, extra, env, log, "build")
    if err and re.search(r"cublas_v2\.h|cannot find -lcublas", err, re.I):
        inc, lib = _pip_cublas(major)
        extra = f"{extra} -I{inc} -L{lib}".strip()
        lib_dirs.insert(0, lib)
        err = _make(make, nvcc, extra, env, log, "build + pip cuBLAS")
    if err and nccl_flags:
        print(err[-2500:])
        banner("the NCCL (2-GPU) build failed (error above); building the ONE-GPU trainer instead")
        NGPU, nccl_flags, nccl_lib = 1, [], None
        err = _make(base, nvcc, extra, env, log, "build 1-GPU")
    if err:
        print(err[-2500:])
        raise RuntimeError(f"BUILD FAILED with CUDA {major} (log: {log}). The compiler error is printed above.")
    BIN = os.path.join(REPO, "train_tinyforge")
    TRAIN_ENV = dict(env)
    TRAIN_ENV["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([nccl_lib] if nccl_lib else []) + [env.get("LD_LIBRARY_PATH", "")]).strip(":")
    miss = _missing_libs(BIN, TRAIN_ENV)
    if miss:
        extra_dirs = sorted({os.path.dirname(p) for p in glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/*/lib/*.so*")
                             + glob.glob("/usr/local/lib/python3*/site-packages/nvidia/*/lib/*.so*")
                             + glob.glob("/opt/conda/lib/python3*/site-packages/nvidia/*/lib/*.so*")})
        TRAIN_ENV["LD_LIBRARY_PATH"] += ":" + ":".join(extra_dirs)
        miss = _missing_libs(BIN, TRAIN_ENV)
    if miss and any("cublas" in x for x in miss):
        _, lib = _pip_cublas(major)
        TRAIN_ENV["LD_LIBRARY_PATH"] = lib + ":" + TRAIN_ENV["LD_LIBRARY_PATH"]
        miss = _missing_libs(BIN, TRAIN_ENV)
    if miss:
        raise RuntimeError(f"built OK but these libraries can't be found at run time: {miss}")
    for l in sh(["ldd", BIN], env=TRAIN_ENV).stdout.splitlines():
        if "cublas" in l or "cudart" in l or "nccl" in l:
            print("  " + l.strip())
    if NGPU > 1:
        try:
            syms = sh(["nm", "-D", BIN], check=False).stdout
        except Exception:   # no binutils: rely on the launch check instead
            syms = ""
        if syms and "ncclAllReduce" not in syms:
            banner("the trainer was built without NCCL; training on ONE GPU")
            NGPU = 1
    print("built", BIN, "| GPUs for training:", NGPU)

run_stage("build", _build, needs=["setup"])
''')
code(r'''
# ---- 3. Data: from the HF data repo, or built here (in the background) and uploaded shard by shard ----
DATA = os.path.join(CONTENT, "data")
TOK_DIR   = os.path.join(DATA, "tokenizer")
BUILD_DIR = os.path.join(DATA, "climbmix_build")   # the builder writes here
MIX_DIR   = os.path.join(DATA, "climbmix")         # finished shards only (hard links); the trainer reads this
DATA_TAG  = f"climbmix_v{VOCAB}_{TRAIN_TOKENS // 1_000_000}M"
DATA_STATE = {"tag": DATA_TAG, "source": None, "builder_done": False, "builder_rc": None,
              "upload": "not needed", "upload_queue": [], "uploaded": [], "upload_fail": 0, "tok_sha": None,
              "tokenizer_from": None}

def _ntok(path):
    import numpy as np
    return int(np.fromfile(path, dtype=np.int32, count=3)[2])

def _shards():
    return sorted(glob.glob(os.path.join(MIX_DIR, "train_*.bin")))

def _capacity(shards, ngpu=None):
    """Optimizer steps one pass over these shards gives. llm.c loader: a sample is B*T*ngpu tokens,
    (ntok-1)//sample samples per shard, and one optimizer step uses TOTAL_BATCH//sample of them."""
    sample = B * T * (ngpu or NGPU or 1)
    return sum((_ntok(s) - 1) // sample for s in shards) // (TOTAL_BATCH // sample)

def _tok_sha():
    return hashlib.sha256(open(os.path.join(TOK_DIR, "tokenizer.json"), "rb").read()).hexdigest()[:16]

def _data_ok():
    need = [os.path.join(TOK_DIR, "tokenizer.json"), os.path.join(TOK_DIR, "token_bytes.bin"), os.path.join(MIX_DIR, "val.bin")]
    return all(os.path.exists(p) for p in need) and sum(_ntok(s) for s in _shards()) >= int(TRAIN_TOKENS * 0.99)

def _write_meta():
    meta = {"tag": DATA_TAG, "vocab": VOCAB, "train_tokens": sum(_ntok(s) for s in _shards()),
            "shards": {os.path.basename(s): _ntok(s) for s in _shards()}, "val_tokens": _ntok(os.path.join(MIX_DIR, "val.bin")),
            "tok_sha": DATA_STATE["tok_sha"], "tokenizer_from": DATA_STATE["tokenizer_from"],
            "built": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": STATUS.get("commit")}
    json.dump(meta, open(os.path.join(DATA, "meta.json"), "w"), indent=1)
    return meta

def _queue_upload(local, remote):
    if remote not in DATA_STATE["uploaded"] and remote not in [r for _, r in DATA_STATE["upload_queue"]]:
        DATA_STATE["upload_queue"].append((local, remote))
        if DATA_STATE["upload"] == "not needed":
            DATA_STATE["upload"] = "pending"

def _builder_watch(p, log):
    """Hard-link each shard into MIX_DIR once the builder reports it written, queue it for upload,
    and mark the data done at the end."""
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
                    _queue_upload(dst, f"{DATA_TAG}/climbmix/{name}")
                    if name.startswith("train_"):
                        print(f"  [data] shard {name} ready ({len(_shards())} so far, {clock()})", flush=True)
                except Exception as e:
                    print(f"  [data] could not publish {name}: {type(e).__name__}: {e}", flush=True)
    rc = p.wait()
    DATA_STATE["builder_rc"] = rc
    if rc == 0 and _data_ok():
        meta = _write_meta()
        DATA_STATE["builder_done"] = True
        print(f"  [data] build finished at {clock()}: {meta['train_tokens']:,} train tokens in {len(meta['shards'])} shards", flush=True)
    else:
        banner(f"[data] builder exited with code {rc}; training continues on the {len(_shards())} shard(s) that finished "
               f"(data will repeat). Log: {log}")

def _data():
    os.makedirs(MIX_DIR, exist_ok=True)
    os.makedirs(TOK_DIR, exist_ok=True)
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
            ok, rc, out = hf(["pulldir", "--repo", DATA_ID, "--type", "dataset", "--prefix", DATA_TAG, "--dest", DATA, "--tries", "4"], 3000)
            if ok and _data_ok():
                print(f"  downloaded in {time.time() - t:.0f}s")
                DATA_STATE.update(source="hf", builder_done=True)
            else:
                print(f"download failed ({rc}): {out[-300:]}\nbuilding the data here instead")
        elif have:
            print(f"{DATA_TAG} on HF is incomplete ({len(have)} files, no meta.json); rebuilding it here (same order, same tokenizer)")
            # every rebuilt file is uploaded again (overwriting), so the set on HF always comes from one build
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
        print(f"one pass = {cap:,} steps on {NGPU} GPU(s); this run needs {MAX_STEPS:,}"
              + ("" if cap >= MAX_STEPS else "  NOTE: the loader will wrap and repeat data"))

def _get_tokenizer():
    """Same tokenizer as the 8-hour run (so its checkpoint and this data agree): download it from
    TOKENIZER_FROM in the data repo; train a new one only if that fails."""
    if os.path.exists(os.path.join(TOK_DIR, "token_bytes.bin")) and os.path.exists(os.path.join(TOK_DIR, "tokenizer.json")):
        return
    if TOKENIZER_FROM:
        ok, rc, out = hf(["ls", "--repo", DATA_ID, "--type", "dataset", "--prefix", f"{TOKENIZER_FROM}/tokenizer/"], 120)
        files = json.loads(out.strip().splitlines()[-1]) if ok else {}
        if f"{TOKENIZER_FROM}/tokenizer/tokenizer.json" in files:
            args = ["pull", "--repo", DATA_ID, "--type", "dataset", "--tries", "3"]
            for k in files:
                args += ["--file", f"{k}={os.path.join(TOK_DIR, os.path.basename(k))}"]
            ok, rc, out = hf(args, 600)
            if ok and os.path.exists(os.path.join(TOK_DIR, "token_bytes.bin")):
                DATA_STATE["tokenizer_from"] = TOKENIZER_FROM
                print(f"tokenizer copied from {DATA_ID}/{TOKENIZER_FROM} ({_tok_sha()})")
                return
            print(f"tokenizer download failed ({rc}): {out[-200:]}")
        else:
            print(f"no tokenizer at {DATA_ID}/{TOKENIZER_FROM}/tokenizer/")
    banner("training a NEW tokenizer: the d12_8hr checkpoint can't be continued on this data (INIT_FROM must stay 'fresh')")
    t = time.time()
    subprocess.run([sys.executable, "python/train_tokenizer.py", "--out", TOK_DIR, "--vocab-size", str(VOCAB),
                    "--max-chars", str(TOK_CHARS)], cwd=REPO, check=True, env=HF_ENV)
    DATA_STATE["tokenizer_from"] = "trained here"
    print(f"  tokenizer done in {time.time() - t:.0f}s")

def _start_build():
    need = [m for m in ("datasets", "transformers", "tokenizers") if importlib.util.find_spec(m) is None]
    if need and not LOCAL_TEST:
        sh([sys.executable, "-m", "pip", "install", "-q"] + need)
    DATA_STATE["source"] = "built"
    for f in glob.glob(os.path.join(MIX_DIR, "*")) + [os.path.join(DATA, "meta.json")]:
        if os.path.exists(f): os.remove(f)
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    os.makedirs(BUILD_DIR, exist_ok=True)
    os.makedirs(MIX_DIR, exist_ok=True)
    _get_tokenizer()
    for f in sorted(glob.glob(os.path.join(TOK_DIR, "*"))):
        if os.path.isfile(f):
            _queue_upload(f, f"{DATA_TAG}/tokenizer/{os.path.basename(f)}")
    t = time.time()
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
# ---- 4. Train. Uploads run in the background; every session resumes from the HF checkpoint ----
RUN_LOCAL = os.path.join(CONTENT, "runs", RUN_NAME)
STAGE = os.path.join(CONTENT, "hf_stage")     # hard-linked snapshots for upload (the trainer prunes its own copies)
SYNC_DIR = os.path.join(CONTENT, "nccl_sync")  # rank 0 writes the NCCL id here for the other rank
TRAIN_LOG = os.path.join(RUN_LOCAL, "train.log")
STATUS_PATH = os.path.join(RUN_LOCAL, "status.json")
SESSIONS_PATH = os.path.join(RUN_LOCAL, "sessions.json")
for _d in (RUN_LOCAL, STAGE, SYNC_DIR):
    os.makedirs(_d, exist_ok=True)
TRAIN_WALL = 0.0
TRAIN_DEADLINE = T0 + BUDGET_H * 3600 - FINAL_RESERVE_MIN * 60
PROGRESS = {"step": 0, "max_steps": MAX_STEPS, "loss": None, "tok_s": None, "val": None, "bpb": None,
            "eta_h": None, "phase": "starting", "launches": 0, "restarts_for_data": 0, "crash_restarts": 0,
            "ngpu": NGPU, "session_start_step": None, "init_from": None}
SYNC = {"ckpt_pushed_step": 0, "ckpt_pushed_at": None, "ckpt_push_fail": 0, "logs_pushed_at": None, "last_error": None}
SYNC_STOP = threading.Event()
SESSIONS = []          # one entry per Kaggle session, kept in <RUN_NAME>/sessions.json
TOKS = []              # tok/s of steady steps this session

def _done_steps(d):
    return sorted(int(os.path.basename(p)[5:]) for p in glob.glob(os.path.join(d, "DONE_*")))

def _session_entry():
    return {"started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(T0)), "started_ct": clock(T0),
            "hours": round((time.time() - T0) / 3600, 2), "gpus": NGPU, "gpu": GPU_NAME,
            "start_step": PROGRESS["session_start_step"], "end_step": PROGRESS["step"],
            "tok_s_median": int(statistics.median(TOKS)) if TOKS else None, "commit": STATUS.get("commit"),
            "phase": PROGRESS["phase"], "init_from": PROGRESS.get("init_from")}

def _write_status():
    s = {"run": RUN_NAME, "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "updated_ct": clock(),
         "session_h": round((time.time() - T0) / 3600, 2), "progress": PROGRESS,
         "data": {k: v for k, v in DATA_STATE.items() if k != "upload_queue"} | {"upload_queue_len": len(DATA_STATE["upload_queue"])},
         "sync": SYNC, "stages": STATUS}
    tmp = STATUS_PATH + ".tmp"
    json.dump(s, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, STATUS_PATH)
    sess = SESSIONS + [_session_entry()]
    json.dump({"sessions": sess, "total_hours": round(sum(x["hours"] for x in sess), 2)},
              open(SESSIONS_PATH + ".tmp", "w"), indent=1)
    os.replace(SESSIONS_PATH + ".tmp", SESSIONS_PATH)

def _small_files():
    _write_status()
    return [(TRAIN_LOG, f"{RUN_NAME}/train.log"), (STATUS_PATH, f"{RUN_NAME}/status.json"),
            (SESSIONS_PATH, f"{RUN_NAME}/sessions.json")]

def _push(files, msg, timeout, squash=False, repo=None, typ="model", tries=3):
    args = ["push", "--repo", repo or RUNS_ID, "--type", typ, "-m", msg, "--tries", tries]
    for local, remote in files:
        args += ["--file", f"{local}={remote}"]
    if squash:
        args.append("--squash")
    return hf(args, timeout)

def _stage(src, name):
    dst = os.path.join(STAGE, name)
    if os.path.exists(dst): os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)
    return dst

def _push_ckpt(step, timeout=1500, extra=()):
    """Upload one resume checkpoint to <RUN_NAME>/ckpt/ (same paths every time) and squash the repo history."""
    os.makedirs(STAGE, exist_ok=True)
    snap = _stage(os.path.join(RUN_LOCAL, f"state_{step:08d}.bin"), "state.bin")
    files = [(snap, f"{RUN_NAME}/ckpt/state.bin")]
    model = os.path.join(RUN_LOCAL, f"model_{step:08d}.bin")
    if os.path.exists(model):   # the weights alone (what export reads), so any checkpoint can be exported later
        files.append((_stage(model, "model.bin"), f"{RUN_NAME}/ckpt/model.bin"))
    meta = {"step": step, "max_steps": MAX_STEPS, "total_batch": TOTAL_BATCH, "ngpu": NGPU, "tokens": step * TOTAL_BATCH,
            "tok_sha": DATA_STATE["tok_sha"], "data_tag": DATA_TAG, "init_from": PROGRESS.get("init_from"),
            "has_model": os.path.exists(model), "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            "commit": STATUS.get("commit"), "bytes": os.path.getsize(snap)}
    json.dump(meta, open(os.path.join(STAGE, "ckpt_meta.json"), "w"), indent=1)
    files += [(os.path.join(STAGE, "ckpt_meta.json"), f"{RUN_NAME}/ckpt/meta.json")] + _small_files() + list(extra)
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

def _push_data_step(timeout=900):
    """Upload ONE queued data file (or meta.json once everything else is up). Returns True if it did something."""
    if DATA_STATE["upload"] in ("ok", "failed", "not needed"):
        return False
    q = DATA_STATE["upload_queue"]
    if q:
        local, remote = q[0]
        ok, rc, out = _push([(local, remote)], f"{DATA_TAG}: {os.path.basename(remote)}", timeout, repo=DATA_ID, typ="dataset", tries=2)
        q.pop(0)
        if ok:
            DATA_STATE["uploaded"].append(remote)
        else:
            DATA_STATE["upload_fail"] += 1
            q.append((local, remote))   # retry later
            print(f"  [hf] data upload of {remote} failed ({rc}); will retry. {out[-200:]}", flush=True)
            if DATA_STATE["upload_fail"] >= 12:
                DATA_STATE["upload"] = "failed"
                banner(f"[data] giving up on uploading {DATA_TAG} after 12 failures; the next session rebuilds it")
        return True
    if DATA_STATE["builder_done"] and os.path.exists(os.path.join(DATA, "meta.json")):
        ok, rc, out = _push([(os.path.join(DATA, "meta.json"), f"{DATA_TAG}/meta.json")], f"{DATA_TAG}: complete", 300,
                            repo=DATA_ID, typ="dataset")
        if ok:
            DATA_STATE["upload"] = "ok"
            print(f"  [hf] data {DATA_TAG} complete on {DATA_ID} ({len(DATA_STATE['uploaded'])} files, {clock()})", flush=True)
        else:
            DATA_STATE["upload_fail"] += 1
        return True
    return False

def _syncer():
    """The only thread that uploads during training: data files as they appear, a checkpoint every
    CKPT_PUSH_MIN, logs every LOG_PUSH_MIN."""
    pause = 10
    while not SYNC_STOP.wait(pause):
        pause = 10
        try:
            if HF_ENV and _push_data_step():
                pause = 1   # more data files may be waiting: come back soon
                if SYNC_STOP.is_set(): break
            steps = _done_steps(RUN_LOCAL)
            last = SYNC["ckpt_pushed_at"] or T0
            if steps and steps[-1] > SYNC["ckpt_pushed_step"] and time.time() - last >= CKPT_PUSH_MIN * 60:
                if _push_ckpt(steps[-1]):
                    SYNC["logs_pushed_at"] = time.time()
                continue
            if SYNC_STOP.is_set(): break
            if os.path.exists(TRAIN_LOG) and time.time() - (SYNC["logs_pushed_at"] or T0) >= LOG_PUSH_MIN * 60:
                ok, rc, out = _push(_small_files(), f"{RUN_NAME} logs step {PROGRESS['step']}", 180, tries=2)
                SYNC["logs_pushed_at"] = time.time()   # on failure, just try again next interval
                if not ok:
                    SYNC["last_error"] = f"logs: {rc}: {out[-200:]}"
                    print(f"  [hf] log upload failed ({rc}); training is unaffected", flush=True)
        except Exception as e:
            SYNC["last_error"] = f"syncer: {type(e).__name__}: {e}"
            print("  [hf] syncer error (training is unaffected):", type(e).__name__, e, flush=True)

def _pull_ckpt(run, files, why):
    """Download <run>/ckpt/state.bin as this run's local checkpoint. Returns the step or 0."""
    mp = os.path.join(CONTENT, f"hf_ckpt_meta_{run}.json")
    ok, rc, out = hf(["pull", "--repo", RUNS_ID, "--file", f"{run}/ckpt/meta.json={mp}"], 120)
    if not ok:
        banner(f"checkpoint meta download for {run} failed ({rc}). {out[-200:]}")
        return 0
    meta = json.load(open(mp))
    step = int(meta["step"])
    if meta.get("tok_sha") != DATA_STATE["tok_sha"]:
        banner(f"The {run} checkpoint (step {step}) used tokenizer {meta.get('tok_sha')}, but this data has {DATA_STATE['tok_sha']}. "
               "It can't be used on this data.")
        return 0
    if run == RUN_NAME and meta.get("max_steps") != MAX_STEPS:
        print(f"NOTE: checkpoint was made with MAX_STEPS={meta.get('max_steps')}; continuing with {MAX_STEPS} (LR schedule follows {MAX_STEPS})")
    if meta.get("total_batch", 65536) != TOTAL_BATCH:
        print(f"NOTE: {run} was trained at {meta.get('total_batch', 65536):,} tokens/step; this run uses {TOTAL_BATCH:,}")
    if step <= max(_done_steps(RUN_LOCAL) or [0]):
        print(f"local checkpoint is at least as new as HF step {step}; using local")
        return step
    size = files[f"{run}/ckpt/state.bin"]
    print(f"downloading {why} step {step} ({size / 1e9:.2f} GB) ...", flush=True)
    t = time.time()
    dst = os.path.join(RUN_LOCAL, f"state_{step:08d}.bin")
    ok, rc, out = hf(["pull", "--repo", RUNS_ID, "--file", f"{run}/ckpt/state.bin={dst}", "--tries", "4"], 1800)
    if not ok or not os.path.exists(dst) or os.path.getsize(dst) != size:
        banner(f"checkpoint download failed ({rc}). {out[-200:]}")
        if os.path.exists(dst): os.remove(dst)
        return 0
    open(os.path.join(RUN_LOCAL, f"DONE_{step:08d}"), "w").close()
    print(f"  got {why} step {step} in {time.time() - t:.0f}s")
    return step

def _resume_from_hf():
    """Fetch <RUN_NAME>/ckpt from the runs repo if it is newer than anything local. On the very first
    session of a run, INIT_FROM != 'fresh' starts from that run's checkpoint instead of random weights."""
    global SESSIONS
    ok, rc, out = hf(["ls", "--repo", RUNS_ID, "--type", "model"], 180)
    if not ok:
        raise RuntimeError(f"could not list {RUNS_ID} ({rc}); refusing to start (a fresh start could overwrite the "
                           f"run's checkpoint). Run the session again. {out[-200:]}")
    files = json.loads(out.strip().splitlines()[-1])
    if f"{RUN_NAME}/train.log" in files and not os.path.exists(TRAIN_LOG):   # one log across sessions
        hf(["pull", "--repo", RUNS_ID, "--file", f"{RUN_NAME}/train.log={TRAIN_LOG}"], 300)
    if f"{RUN_NAME}/sessions.json" in files:
        sp = os.path.join(CONTENT, "hf_sessions.json")
        if hf(["pull", "--repo", RUNS_ID, "--file", f"{RUN_NAME}/sessions.json={sp}"], 120)[0]:
            SESSIONS = json.load(open(sp)).get("sessions", [])
    have_own = f"{RUN_NAME}/ckpt/meta.json" in files and f"{RUN_NAME}/ckpt/state.bin" in files
    if have_own:
        step = _pull_ckpt(RUN_NAME, files, "HF checkpoint")
        if not step:
            raise RuntimeError(f"{RUN_NAME} has a checkpoint on HF but it could not be used (see above). Not starting "
                               "fresh, because that would overwrite it. Run the session again, or change RUN_NAME.")
        SYNC.update(ckpt_pushed_step=step, ckpt_pushed_at=time.time())
        prev = [s.get("init_from") for s in SESSIONS if s.get("init_from")]
        PROGRESS["init_from"] = prev[0] if prev else None
        print(f"  resuming {RUN_NAME} from step {step}")
        return
    if SESSIONS:
        print(f"note: {len(SESSIONS)} earlier session(s) of {RUN_NAME} left no checkpoint; starting it over")
    if INIT_FROM == "fresh":
        print("no checkpoint in", RUNS_ID, "for", RUN_NAME, "-> fresh run (random init)")
        PROGRESS["init_from"] = "fresh"
        return
    if f"{INIT_FROM}/ckpt/state.bin" not in files:
        raise RuntimeError(f"INIT_FROM={INIT_FROM!r} but {RUNS_ID} has no {INIT_FROM}/ckpt/state.bin")
    step = _pull_ckpt(INIT_FROM, files, f"{INIT_FROM} checkpoint (INIT_FROM)")
    if not step:
        raise RuntimeError(f"could not start from {INIT_FROM} (see above); set INIT_FROM = 'fresh' to start from scratch")
    PROGRESS["init_from"] = f"{INIT_FROM}@{step}"
    banner(f"continuing from {INIT_FROM} step {step}: the step counter starts at {step} of {MAX_STEPS}, "
           f"the LR jumps back to the plateau value (no re-warmup), and the data cursor carries over")

_STEP_RE = re.compile(r"step (\d+)/(\d+)\s+loss ([\d.naninf]+)\s+lr_mult ([\d.]+)\s+([\d.]+)s\s+(\d+) tok/s")

def _glance(step, recent):
    left = MAX_STEPS - step
    sps = statistics.median(recent) if recent else TOTAL_BATCH / 32000
    eta = left * sps
    PROGRESS["eta_h"] = round(eta / 3600, 2)
    tps = PROGRESS["tok_s"] or 0
    vb = f"  val {PROGRESS['val']:.3f} bpb {PROGRESS['bpb']:.3f}" if PROGRESS["val"] is not None else ""
    print(f"[{clock()}] step {step:,}/{MAX_STEPS:,} ({100 * step / MAX_STEPS:4.1f}%)  loss {PROGRESS['loss']:.4f}  "
          f"{tps:,} tok/s  {sps:.2f} s/step{vb}  | {(time.time() - T0) / 3600:.2f} h in, training left {eta / 3600:.1f} h",
          flush=True)

def _wait_for_data(start):
    """Before a launch: the trainer must not resume at a data position past the shards that exist
    (the loader would index a missing shard). Wait for the background build to catch up."""
    want = min(start + CKPT_EVERY + 1, MAX_STEPS)
    t = time.time()
    while not DATA_STATE["builder_done"] and _capacity(_shards()) < want:
        if DATA_STATE["builder_rc"] not in (None, 0):
            raise RuntimeError(f"data builder failed and the shards cover only {_capacity(_shards())} steps; resume needs {start}")
        if time.time() > TRAIN_DEADLINE - MIN_LAUNCH_S or WATCHDOG_FIRED:
            raise RuntimeError("time budget used up while waiting for data")
        if time.time() - t < 25 or int(time.time() - t) % 300 < 20:
            print(f"  [data] waiting for shards to reach step {want:,} (now {_capacity(_shards()):,} steps of data, {clock()})", flush=True)
        time.sleep(_knob("DATA_WAIT_POLL_S", 20))

def _launch(cmd, ngpu):
    """Start one trainer process per GPU. Rank 0's stdout is returned for parsing; the other ranks
    print errors only (to rank<N>.err)."""
    env = dict(TRAIN_ENV)
    if ngpu > 1:
        if NCCL_P2P_OFF:
            env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_IB_DISABLE"] = "1"
        shutil.rmtree(SYNC_DIR, ignore_errors=True)   # a stale nccl_id.bin would pair the ranks wrongly
        os.makedirs(SYNC_DIR, exist_ok=True)
    for d in (RUN_LOCAL, SYNC_DIR):   # the trainer writes here: they must exist before launch
        if not os.path.isdir(d):
            raise RuntimeError(f"directory missing before launch: {d}")
    procs = []
    for r in range(1, ngpu):
        procs.append(subprocess.Popen(cmd + ["--nproc", str(ngpu), "--rank", str(r), "--sync-dir", SYNC_DIR], env=env,
                                      stdout=subprocess.DEVNULL, stderr=open(os.path.join(CONTENT, f"rank{r}.err"), "a")))
    c0 = cmd + (["--nproc", str(ngpu), "--rank", "0", "--sync-dir", SYNC_DIR] if ngpu > 1 else [])
    p0 = subprocess.Popen(c0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    return p0, procs, c0

def _train():
    global TRAIN_WALL, TRAINER_PROCS, NGPU
    for d in (RUN_LOCAL, STAGE, SYNC_DIR):
        os.makedirs(d, exist_ok=True)
    _resume_from_hf()
    local = max(_done_steps(RUN_LOCAL) or [0])
    PROGRESS["session_start_step"] = local
    PROGRESS["step"] = local
    print(f"starting at step {local:,} of {MAX_STEPS:,}" if local else "fresh run", "| training must stop by", clock(TRAIN_DEADLINE))
    th = threading.Thread(target=_syncer, daemon=True); th.start()
    if local >= MAX_STEPS:
        PROGRESS["phase"] = "finished"
        print("this run already reached MAX_STEPS; nothing to train (the export stage runs next)")
        return
    finished, fallback_used = False, False
    with open(TRAIN_LOG, "a") as logf:
        while True:
            if WATCHDOG_FIRED:
                print("watchdog fired; not restarting the trainer"); break
            if TRAIN_DEADLINE - time.time() < MIN_LAUNCH_S:
                print("training time budget used up"); break
            start = max(_done_steps(RUN_LOCAL) or [0])
            _wait_for_data(start)
            remaining = TRAIN_DEADLINE - time.time()
            shards = _shards()
            n_launch, cap = len(shards), _capacity(shards)
            cmd = [BIN, "-i", os.path.join(MIX_DIR, "train_*.bin"), "-j", os.path.join(MIX_DIR, "val.bin"),
                   "-e", "d12", "-o", RUN_LOCAL, "-b", str(B), "-t", str(T), "-d", str(TOTAL_BATCH),
                   "-x", str(MAX_STEPS), "-v", str(VAL_EVERY), "-m", str(VAL_BATCHES),
                   "-n", str(CKPT_EVERY), "-y", "1", "--keep-ckpts", "2", "--time-limit", str(int(remaining)),
                   "--vocab", str(VOCAB), "--token-bytes", os.path.join(TOK_DIR, "token_bytes.bin")]
            PROGRESS["launches"] += 1
            PROGRESS["phase"] = "training"
            PROGRESS["ngpu"] = NGPU
            p, others, c0 = _launch(cmd, NGPU)
            TRAINER_PROCS = [p] + others
            print(f"launch {PROGRESS['launches']} at step {start:,} on {NGPU} GPU(s): {n_launch} shard(s) = {cap:,} steps of data, "
                  f"time limit {remaining / 3600:.2f} h\n" + " ".join(c0), flush=True)
            logf.write(f"\n=== session {time.strftime('%Y-%m-%d %H:%M:%S')}  commit {STATUS.get('commit')}  gpu {NGPU}x {GPU_NAME}  "
                       f"launch {PROGRESS['launches']}  shards {n_launch}\n" + " ".join(c0) + "\n"); logf.flush()
            t = time.time()
            mon = {"stall": False, "rank_died": None, "first_step": False}
            def _monitor(p=p, others=others, t=t, mon=mon):
                while p.poll() is None:
                    for i, o in enumerate(others, 1):
                        if o.poll() not in (None, 0) and mon["rank_died"] is None:
                            mon["rank_died"] = (i, o.returncode)
                            print(f"rank {i} exited with code {o.returncode}; stopping rank 0", flush=True)
                            p.kill()
                    if not mon["first_step"] and len(others) and time.time() - t > STARTUP_TIMEOUT_S:
                        mon["stall"] = True
                        print(f"no training step after {STARTUP_TIMEOUT_S}s on {NGPU} GPUs; stopping the launch", flush=True)
                        p.kill()
                    time.sleep(2)
            threading.Thread(target=_monitor, daemon=True).start()
            restart, done_seen, limit_seen, recent = False, False, False, []
            try:
                for line in p.stdout:
                    logf.write(line); logf.flush()
                    m = _STEP_RE.match(line)
                    if m:
                        step = int(m.group(1))
                        mon["first_step"] = True
                        PROGRESS.update(step=step, loss=float(m.group(3)), tok_s=int(m.group(6)))
                        recent = (recent + [float(m.group(5))])[-200:]
                        if step - start > 20:
                            TOKS.append(int(m.group(6)))
                            if len(TOKS) > 20000: del TOKS[:10000]
                        if step % 500 == 0 or step == start + 1:
                            _glance(step, recent)
                        # More shards are ready than this launch can see: switch at the next checkpoint.
                        if len(_shards()) > n_launch and (DATA_STATE["builder_done"] or step + CKPT_EVERY >= cap):
                            newest = max(_done_steps(RUN_LOCAL) or [0])
                            if newest > start and step == newest + 1:
                                print(f"  [data] {len(_shards())} shards ready now; restarting the trainer from checkpoint "
                                      f"{newest} to use them", flush=True)
                                restart = True
                                _kill_trainers(signal.SIGTERM)
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
                for o in others:                # the other ranks follow rank 0 (or are stopped)
                    try:
                        o.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        o.kill(); o.wait()
                TRAINER_PROCS = []
                TRAIN_WALL += time.time() - t
                logf.write(f"=== trainer exit {p.returncode} after {time.time() - t:.0f}s\n"); logf.flush()
            for r in range(1, len(others) + 1):
                ep = os.path.join(CONTENT, f"rank{r}.err")
                e = open(ep).read().strip() if os.path.exists(ep) else ""
                if e and (rc != 0 or mon["rank_died"]):
                    print(f"rank {r} stderr (last lines):\n" + "\n".join(e.splitlines()[-15:]))
            if restart:
                PROGRESS["restarts_for_data"] += 1
                continue
            if rc == 0 and done_seen:
                finished = True; break
            if rc == 0 and limit_seen:
                print(f"trainer stopped at its time limit at step {PROGRESS['step']:,} of {MAX_STEPS:,}; the next session continues")
                break
            if WATCHDOG_FIRED:
                print("watchdog fired; not restarting the trainer"); break
            if NGPU > 1 and not mon["first_step"] and not fallback_used:
                fallback_used = True
                NGPU = 1
                banner(f"the 2-GPU launch never reached step 1 (exit {rc}, stalled={mon['stall']}, rank died={mon['rank_died']}); "
                       "continuing on ONE GPU (about half speed). Last lines:\n" + "".join(open(TRAIN_LOG).readlines()[-15:]))
                continue
            tail = "".join(open(TRAIN_LOG).readlines()[-25:])
            if PROGRESS["crash_restarts"] < 3 and TRAIN_DEADLINE - time.time() > MIN_RESTART_S and _done_steps(RUN_LOCAL):
                PROGRESS["crash_restarts"] += 1
                banner(f"trainer exited with code {rc}; restarting from the last checkpoint "
                       f"(attempt {PROGRESS['crash_restarts']}/3). Last lines:\n{tail}")
                time.sleep(10)
                continue
            raise RuntimeError(f"trainer exited with code {rc}. Last lines:\n{tail}")
    PROGRESS["phase"] = "finished" if finished else "stopped"
    print(f"training wall time this session: {TRAIN_WALL / 3600:.2f} h, reached step {PROGRESS['step']:,}/{MAX_STEPS:,}")
    if not finished and TOKS:
        tps = statistics.median(TOKS[-2000:])
        left_h = (MAX_STEPS - PROGRESS["step"]) * TOTAL_BATCH / tps / 3600
        per = BUDGET_H - FINAL_RESERVE_MIN / 60 - 0.25
        print(f"at {tps:,.0f} tok/s: {left_h:.1f} h of training left = about {left_h / per:.2f} more session(s) of {BUDGET_H} h")

run_stage("train", _train, needs=["build", "data"])
''')
code(r'''
# ---- 5. Summary, final export (HF format + GGUF, last session only), 3 samples ----
SUMMARY = {}
FINAL = {"exported": False, "files": [], "step": None}
HF_DIR = os.path.join(CONTENT, "d12-hf")
GGUF = os.path.join(CONTENT, "tinyforge-d12-f32.gguf")

def _summary():
    global SUMMARY
    if not os.path.exists(TRAIN_LOG):
        raise RuntimeError("no training log (training did not start)")
    lines = open(TRAIN_LOG).read().splitlines()
    steps = []
    for l in lines:
        m = _STEP_RE.match(l)
        if m:
            steps.append(dict(step=int(m.group(1)), loss=float(m.group(3)), secs=float(m.group(5)), tps=int(m.group(6))))
    vals = [(float(a), float(b)) for a, b in re.findall(r"val loss ([\d.]+)\s+bpb ([\d.]+)", "\n".join(lines))]
    if not steps:
        raise RuntimeError("no step lines in the log")
    last = max(PROGRESS["step"], steps[-1]["step"])
    tail = [s["loss"] for s in steps[-50:]]
    sess = SESSIONS + [_session_entry()]
    SUMMARY = {
        "run": RUN_NAME, "commit": STATUS.get("commit"), "gpus": f"{NGPU} x {GPU_NAME}", "init_from": PROGRESS.get("init_from"),
        "steps_done": last, "max_steps": MAX_STEPS, "tokens_per_step": TOTAL_BATCH,
        "tokens_processed_by_this_run": last * TOTAL_BATCH,
        "this_session_steps": last - (PROGRESS["session_start_step"] or 0),
        "this_session_tok_per_s_median": int(statistics.median(TOKS)) if TOKS else None,
        "final_train_loss_avg50": round(sum(tail) / len(tail), 4),
        "final_val_loss": vals[-1][0] if vals else None,
        "final_val_bpb": vals[-1][1] if vals else None,
        "val_curve_last10": vals[-10:],
        "fp16_overflow_skips": sum("gradient overflow" in l for l in lines),
        "finished_lr_schedule": last >= MAX_STEPS,
        "train_wall_h_this_session": round(TRAIN_WALL / 3600, 2),
        "session_wall_h": round((time.time() - T0) / 3600, 2),
        "sessions_so_far": len(sess), "gpu_hours_so_far": round(sum(x["hours"] for x in sess), 2),
        "launches": PROGRESS["launches"], "restarts_for_data": PROGRESS["restarts_for_data"], "crash_restarts": PROGRESS["crash_restarts"],
        "data": {k: DATA_STATE.get(k) for k in ("tag", "source", "upload", "tok_sha", "tokenizer_from")},
        "unique_train_steps_available": _capacity(_shards()) if _shards() else None,
        "stage_secs": {k: v.get("secs") for k, v in STATUS.items() if isinstance(v, dict)},
    }
    txt = "\n".join(f"{k}: {v}" for k, v in SUMMARY.items())
    print(txt)
    open(os.path.join(RUN_LOCAL, "summary.txt"), "w").write(txt + "\n")
    json.dump(SUMMARY, open(os.path.join(RUN_LOCAL, "summary.json"), "w"), indent=1)

def _export():
    """Only once the run reached MAX_STEPS: HF folder + f32 GGUF, uploaded to <RUN_NAME>/final/ in the last cell."""
    done = max(_done_steps(RUN_LOCAL) or [0])
    if done < MAX_STEPS:
        print(f"not finished yet (step {done:,} of {MAX_STEPS:,}); export happens in the session that reaches {MAX_STEPS:,}")
        return
    ok, rc, out = hf(["ls", "--repo", RUNS_ID, "--type", "model", "--prefix", f"{RUN_NAME}/final/"], 120)
    have = json.loads(out.strip().splitlines()[-1]) if ok else {}
    if f"{RUN_NAME}/final/tinyforge-d12-f32.gguf" in have and f"{RUN_NAME}/final/hf/config.json" in have:
        print(f"the final export is already on HF ({RUN_NAME}/final/); nothing to do")
        FINAL["step"] = done
        return
    model = os.path.join(RUN_LOCAL, f"model_{done:08d}.bin")
    if not os.path.exists(model):
        print("no local model file for the final step; downloading ckpt/model.bin from HF ...")
        mp = os.path.join(CONTENT, "hf_final_meta.json")
        ok, rc, out = hf(["pull", "--repo", RUNS_ID, "--file", f"{RUN_NAME}/ckpt/meta.json={mp}"], 120)
        meta = json.load(open(mp)) if ok else {}
        if int(meta.get("step", 0)) != done or not meta.get("has_model"):
            raise RuntimeError(f"no model file for step {done} locally or on HF (HF meta: {meta})")
        ok, rc, out = hf(["pull", "--repo", RUNS_ID, "--file", f"{RUN_NAME}/ckpt/model.bin={model}", "--tries", "3"], 900)
        if not ok:
            raise RuntimeError(f"model download failed ({rc}): {out[-300:]}")
    if not LOCAL_TEST and (importlib.util.find_spec("gguf") is None or importlib.util.find_spec("safetensors") is None):
        sh([sys.executable, "-m", "pip", "install", "-q", "gguf", "safetensors"])
    shutil.rmtree(HF_DIR, ignore_errors=True)
    r = subprocess.run([sys.executable, "python/export_hf.py", "--input", model, "--output", HF_DIR, "--tokenizer", TOK_DIR],
                       cwd=REPO, timeout=900, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"HF export failed: {(r.stdout + r.stderr)[-1500:]}")
    r = subprocess.run([sys.executable, "python/convert_tinyforge_to_gguf.py", "--model", HF_DIR, "--out", GGUF,
                        "--outtype", "f32", "--name", "tinyforge-d12"], cwd=REPO, timeout=900, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed: {(r.stdout + r.stderr)[-1500:]}")
    files = [(GGUF, f"{RUN_NAME}/final/tinyforge-d12-f32.gguf")]
    for f in sorted(glob.glob(os.path.join(HF_DIR, "*"))):
        if os.path.isfile(f):
            files.append((f, f"{RUN_NAME}/final/hf/{os.path.basename(f)}"))
    info = {"step": done, "tokens": done * TOTAL_BATCH, "init_from": PROGRESS.get("init_from"),
            "final_val_bpb": SUMMARY.get("final_val_bpb"), "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
            "commit": STATUS.get("commit")}
    json.dump(info, open(os.path.join(CONTENT, "export.json"), "w"), indent=1)
    files.append((os.path.join(CONTENT, "export.json"), f"{RUN_NAME}/final/export.json"))
    FINAL.update(exported=True, files=files, step=done)
    kout = os.path.join(OUT_DIR, "final")   # also in this version's Kaggle Output
    try:
        shutil.rmtree(kout, ignore_errors=True)
        shutil.copytree(HF_DIR, os.path.join(kout, "d12-hf"))
        shutil.copy(GGUF, kout)
    except Exception as e:
        print("could not copy the export to Kaggle Output (HF upload is unaffected):", type(e).__name__, e)
    print(f"exported step {done:,}: {HF_DIR} ({len(files) - 2} files) and {GGUF} ({os.path.getsize(GGUF) / 1e6:.0f} MB)")

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
        print("samples off")
        return
    s = max(_done_steps(RUN_LOCAL) or [0])
    src, tmp = HF_DIR, None
    if not os.path.exists(os.path.join(HF_DIR, "config.json")):
        ckpt = os.path.join(RUN_LOCAL, f"model_{s:08d}.bin")
        if not s or not os.path.exists(ckpt):
            raise RuntimeError("no local model file for the last checkpoint (nothing was trained this session)")
        tmp = src = os.path.join(CONTENT, "tmp_export")
        shutil.rmtree(tmp, ignore_errors=True)
        r = subprocess.run([sys.executable, "python/export_hf.py", "--input", ckpt, "--output", tmp, "--tokenizer", TOK_DIR],
                           cwd=REPO, timeout=600, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"temporary export failed: {(r.stdout + r.stderr)[-1500:]}")
    try:
        open(os.path.join(CONTENT, "sample.py"), "w").write(SAMPLE_PY)
        r = subprocess.run([sys.executable, os.path.join(CONTENT, "sample.py"), src, os.path.join(RUN_LOCAL, "samples.txt")],
                           cwd=REPO, timeout=900, capture_output=True, text=True)
        print(r.stdout[-3000:])
        if r.returncode != 0:
            raise RuntimeError(f"sampling failed: {r.stderr[-1500:]}")
        print(f"samples from step {s:,}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)   # the temporary export is not kept

run_stage("summary", _summary, needs=["setup"])
run_stage("export", _export, needs=["train"])
run_stage("samples", _samples, needs=["setup"])
''')

code(r'''
# ---- 6. Final upload to HF and verify it. The session then ends by itself (committed run). ----
FINAL_HARD_TIMEOUT_S = _knob("FINAL_HARD_TIMEOUT_S", 1800)   # the notebook ends after this even if uploads hang

def _final():
    SYNC_STOP.set()
    p = _HF_CHILD.get("p")   # don't wait on a background upload; the final upload below covers it
    if p is not None and p.poll() is None:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
    PROGRESS["phase"] = PROGRESS["phase"] if PROGRESS["phase"] in ("finished", "stopped") else "final upload"
    STATUS["session_wall_h"] = round((time.time() - T0) / 3600, 2)
    extra = []
    for f in ("summary.json", "summary.txt", "samples.txt"):
        if os.path.exists(os.path.join(RUN_LOCAL, f)):
            extra.append((os.path.join(RUN_LOCAL, f), f"{RUN_NAME}/{f}"))
    for f in ("build.log", "data_build.log"):
        if os.path.exists(os.path.join(CONTENT, f)):
            extra.append((os.path.join(CONTENT, f), f"{RUN_NAME}/logs/{f}"))
    steps = _done_steps(RUN_LOCAL)
    expect = [r for _, r in extra] + [f"{RUN_NAME}/train.log", f"{RUN_NAME}/status.json", f"{RUN_NAME}/sessions.json"]
    if steps and steps[-1] > SYNC["ckpt_pushed_step"]:
        ok = _push_ckpt(steps[-1], timeout=1500, extra=extra)
        expect += [f"{RUN_NAME}/ckpt/state.bin", f"{RUN_NAME}/ckpt/meta.json"]
        if not ok:   # at least get the small files up
            _push(extra + _small_files(), f"{RUN_NAME} final logs", 300)
    else:
        ok, rc, out = _push(extra + _small_files(), f"{RUN_NAME} final logs", 300)
        print(f"  [hf] final logs/summary upload: {'ok' if ok else 'FAILED (' + str(rc) + ') ' + out[-200:]}")
    if FINAL["exported"]:
        t = time.time()
        ok, rc, out = _push(FINAL["files"], f"{RUN_NAME} final export (step {FINAL['step']})", 1500)
        print(f"  [hf] final export upload: {'ok' if ok else 'FAILED (' + str(rc) + ') ' + out[-300:]} in {time.time() - t:.0f}s")
        expect += [r for _, r in FINAL["files"]]
    while DATA_STATE.get("upload") == "pending" and time.time() < T0 + WATCHDOG_H * 3600 - 600:
        if not _push_data_step():
            break
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
              + ("" if f"{DATA_TAG}/meta.json" in have else "  (data NOT complete on HF; the next session rebuilds it)"))
    for f in (STATUS_PATH, SESSIONS_PATH, os.path.join(RUN_LOCAL, "summary.txt")):   # small copies in Kaggle Output
        try:
            if os.path.exists(f): shutil.copy(f, OUT_DIR)
        except Exception:
            pass

try:
    if HF_ENV and globals().get("RUNS_ID") and "RUN_LOCAL" in globals():
        th = threading.Thread(target=_final, daemon=True); th.start()
        th.join(FINAL_HARD_TIMEOUT_S)
        if th.is_alive():
            banner(f"final upload still running after {FINAL_HARD_TIMEOUT_S}s; ending anyway")
    else:
        print("no HF access (setup failed), nothing to upload")
except BaseException as e:
    banner(f"final stage error: {type(e).__name__}: {e}")
finally:
    print("\nstage status:")
    for k, v in STATUS.items():
        if isinstance(v, dict):
            print(f"  {k:8s} {'ok' if v.get('ok') else 'FAILED'} {v.get('secs', '')}s {v.get('error', '')}")
    print(f"session wall time: {(time.time() - T0) / 3600:.2f} h (started {clock(T0)}, now {clock()})")
    done = max(_done_steps(globals().get("RUN_LOCAL", "/nonexistent")) or [0]) if "RUN_LOCAL" in globals() else 0
    if done >= MAX_STEPS:
        print(f"RUN COMPLETE: {RUN_NAME} reached step {done:,}. Final model: {globals().get('RUNS_ID')}/{RUN_NAME}/final/")
    else:
        print(f"{RUN_NAME} is at step {done:,} of {MAX_STEPS:,}. Next session: Save Version > Save & Run All (Commit) again.")
    if LOCAL_TEST:
        open(os.path.join(CONTENT, "notebook_end.log"), "a").write(f"{time.time():.0f} end step {done}\n")
    print("(In an interactive session, stop it with the power button so it stops using GPU quota.)")
''')

nb = {"cells": [], "metadata": {"accelerator": "nvidiaTeslaT4", "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
       "isGpuEnabled": True, "language": "python", "sourceType": "notebook"},
       "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
for kind, src in cells:
    lines = src.split("\n")
    c = {"cell_type": kind, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    nb["cells"].append(c)
out = sys.argv[1] if len(sys.argv) > 1 else "notebooks/TinyForge-Kaggle-Pretrain.ipynb"
json.dump(nb, open(out, "w"), indent=1)
open(out, "a").write("\n")
print("wrote", out, len(nb["cells"]), "cells")
