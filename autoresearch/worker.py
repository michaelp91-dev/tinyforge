"""TinyForge autoresearch worker: the GPU side of the loop.

Adapted from the ideas in karpathy/autoresearch (https://github.com/karpathy/autoresearch, MIT):
a fixed training-time budget per experiment, val bpb as the only score, one results.tsv row per
experiment (commit, val_bpb, memory_gb, status, description), keep/discard against the best so
far, crashes logged and skipped. The difference: the researcher is not a model running inside
this notebook. The researcher (Jarvis) pushes experiments to the `autoresearch` git branch and
reads results from a private Hugging Face repo; this worker only polls, builds, trains, scores,
and uploads. It never decides what to try next.

Queue protocol (branch `autoresearch`, folder autoresearch/):
  queue/<id>.json   one experiment: {"id", "description", "code_ref", "args": {flag: value}}
                    run in sorted filename order; an id that already has a result is skipped
  control.json      {"stop": false, "max_experiments": null}; stop=true ends the session after
                    the current experiment (results saved, runtime released)
Results (HF model repo, private): results.tsv, results.jsonl, status.json, runs/<id>/run.log,
runs/<id>/result.json.

The caller (the notebook) owns the HF token, the watchdogs, and the final disconnect.
"""

import glob, hashlib, json, os, re, shutil, signal, subprocess, sys, threading, time

TSV_HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
# Flags the worker sets itself. Experiments may not touch them: data, eval, and checkpointing are fixed.
FIXED_FLAGS = {"-i", "-j", "-o", "-x", "-v", "-m", "-n", "-y", "--time-limit", "--time-budget",
               "--token-bytes", "--vocab", "--keep-ckpts", "--nproc", "--rank", "--sync-dir", "--profile"}
ID_RE = re.compile(r"^[0-9]{3}[A-Za-z0-9._-]{0,60}$")


class Worker:
    def __init__(self, cfg):
        """cfg keys (all required unless noted):
        work          scratch dir (local disk)
        ctrl          git clone of the autoresearch branch (full history, single branch)
        branch        "autoresearch"
        hf_sync       path to python/hf_sync.py
        hf_env        env dict holding HF_TOKEN (never printed)
        results_repo  "<user>/tinyforge-autoresearch" (private model repo)
        train_glob, val_bin, token_bytes, vocab
        time_budget_s, eval_tokens, fixed_T
        max_experiments, session_deadline (epoch s), idle_timeout_s, poll_s
        exp_overhead_s   startup + final eval allowance; also used to refuse an experiment that can't finish
        build_fn(src_dir, log_path) -> binary path      (notebook supplies the nvcc build)
        log(msg)         optional printer
        """
        self.c = cfg
        self.log = cfg.get("log") or (lambda m: print(m, flush=True))
        self.work = cfg["work"]
        os.makedirs(self.work, exist_ok=True)
        self.res_dir = os.path.join(self.work, "results")
        os.makedirs(os.path.join(self.res_dir, "runs"), exist_ok=True)
        self.tsv = os.path.join(self.res_dir, "results.tsv")
        self.jsonl = os.path.join(self.res_dir, "results.jsonl")
        self.status_path = os.path.join(self.res_dir, "status.json")
        self.bin_cache = {}
        self.records = []
        self.state = {"phase": "starting", "session_started": time.time(), "experiments_this_session": 0,
                      "current": None, "branch_head": None, "idle_since": None, "stop_reason": None,
                      "last_upload_error": None}
        self.stop_flag = threading.Event()   # the notebook watchdog may set it
        self._proc = None
        self._last_hb = 0.0

    # ---------- helpers ----------
    def _scrub(self, text):
        tok = (self.c.get("hf_env") or {}).get("HF_TOKEN")
        return text.replace(tok, "***") if tok and text else text

    def hf(self, args, timeout):
        cmd = [sys.executable, self.c["hf_sync"]] + [str(a) for a in args]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 env=self.c["hf_env"], start_new_session=True)
        except Exception as e:
            return False, "spawn", f"{type(e).__name__}: {e}"
        try:
            out, _ = p.communicate(timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            try: os.killpg(p.pid, signal.SIGKILL)
            except Exception: pass
            try: out, _ = p.communicate(timeout=15)
            except Exception: out = ""
            rc = "timeout"
        return rc == 0, rc, self._scrub(out or "").strip()

    def git(self, *args, check=True):
        r = subprocess.run(["git", "-C", self.c["ctrl"]] + list(args), capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {(r.stdout + r.stderr)[-500:]}")
        return r.stdout.strip()

    def _write_status(self):
        s = dict(self.state)
        s.update(updated=time.strftime("%Y-%m-%d %H:%M:%S %Z"), now=time.time(),
                 done=len(self.records), best=self.best(),
                 caps={"max_experiments": self.c["max_experiments"], "session_deadline": self.c["session_deadline"],
                       "idle_timeout_s": self.c["idle_timeout_s"], "time_budget_s": self.c["time_budget_s"]})
        tmp = self.status_path + ".tmp"
        json.dump(s, open(tmp, "w"), indent=1, default=str)
        os.replace(tmp, self.status_path)

    def upload(self, extra=(), what="results", timeout=180):
        self._write_status()
        files = [(self.status_path, "status.json")]
        for p, r in ((self.tsv, "results.tsv"), (self.jsonl, "results.jsonl")):
            if os.path.exists(p): files.append((p, r))
        files += list(extra)
        args = ["push", "--repo", self.c["results_repo"], "--type", "model", "-m", what, "--tries", "2"]
        for l, r in files:
            args += ["--file", f"{l}={r}"]
        ok, rc, out = self.hf(args, timeout)
        self.state["last_upload_error"] = None if ok else f"{rc}: {out[-200:]}"
        if not ok:
            self.log(f"  [hf] upload of {what} failed ({rc}); will retry with the next upload")
        self._last_hb = time.time()
        return ok

    def heartbeat(self, every_s=300):
        if time.time() - self._last_hb >= every_s:
            self.upload(what=f"heartbeat {self.state['phase']}", timeout=90)

    # ---------- results ----------
    def load_previous(self):
        """Pull earlier results from HF so a relaunch continues the same research (no repeats, same best)."""
        ok, rc, out = self.hf(["ls", "--repo", self.c["results_repo"], "--type", "model"], 120)
        have = json.loads(out.strip().splitlines()[-1]) if ok and out.strip() else {}
        if not ok:
            self.log(f"could not list {self.c['results_repo']} ({rc}); starting a fresh results file")
        for name, local in (("results.jsonl", self.jsonl), ("results.tsv", self.tsv)):
            if name in have:
                self.hf(["pull", "--repo", self.c["results_repo"], "--file", f"{name}={local}"], 120)
        if os.path.exists(self.jsonl):
            for line in open(self.jsonl):
                line = line.strip()
                if line:
                    try: self.records.append(json.loads(line))
                    except Exception: pass
        if not os.path.exists(self.tsv):
            open(self.tsv, "w").write(TSV_HEADER)
        b = self.best()
        self.log(f"previous results: {len(self.records)} experiment(s)"
                 + (f"; best so far {b['id']} val_bpb {b['val_bpb']:.6f}" if b else ""))

    def best(self):
        ok = [r for r in self.records if r.get("status") in ("keep", "discard") and r.get("val_bpb")]
        return min(ok, key=lambda r: r["val_bpb"]) if ok else None

    def record(self, rec):
        b = self.best()
        if rec["status"] != "crash":
            rec["status"] = "keep" if (b is None or rec["val_bpb"] < b["val_bpb"]) else "discard"
            rec["best_before"] = b["val_bpb"] if b else None
        self.records.append(rec)
        with open(self.jsonl, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        desc = f"{rec['id']}: {rec.get('description', '')}".replace("\t", " ").replace("\n", " ")
        if rec["status"] == "crash":
            desc += f" [{rec.get('error', 'crash')}]"
        with open(self.tsv, "a") as f:
            f.write(f"{rec['commit'][:7]}\t{rec.get('val_bpb') or 0:.6f}\t{rec.get('memory_gb') or 0:.1f}\t"
                    f"{rec['status']}\t{desc}\n")

    # ---------- queue ----------
    def refresh(self):
        self.git("fetch", "-q", "origin", self.c["branch"])
        self.git("reset", "-q", "--hard", "FETCH_HEAD")
        head = self.git("rev-parse", "HEAD")
        self.state["branch_head"] = head[:7]
        return head

    def control(self):
        p = os.path.join(self.c["ctrl"], "autoresearch", "control.json")
        try:
            return json.load(open(p))
        except Exception:
            return {}

    def pending(self):
        done = {r["id"] for r in self.records}
        out = []
        for p in sorted(glob.glob(os.path.join(self.c["ctrl"], "autoresearch", "queue", "*.json"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem in done:
                continue
            out.append(p)
        return out

    def validate(self, path):
        """Return (exp, error). The worker refuses anything that would change the data, the eval, or checkpointing."""
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            e = json.load(open(path))
        except Exception as ex:
            return {"id": stem}, f"bad json: {type(ex).__name__}: {ex}"
        e.setdefault("id", stem)
        if e["id"] != stem:
            return e, f"id {e['id']} does not match file name {stem}"
        if not ID_RE.match(stem):
            return e, "id must start with 3 digits and use only letters, digits, . _ -"
        args = e.get("args")
        if not isinstance(args, dict):
            return e, "args must be an object of flag: value"
        bad = sorted(k for k in args if k in FIXED_FLAGS or not k.startswith("-"))
        if bad:
            return e, f"fixed or invalid flags: {bad}"
        for k, v in args.items():
            if not re.match(r"^-{1,2}[A-Za-z][A-Za-z0-9-]*$", k) or not re.match(r"^[A-Za-z0-9._+-]+$", str(v)):
                return e, f"bad flag or value: {k} {v}"
        try:
            B = int(args.get("-b", 16)); T = int(args.get("-t", self.c["fixed_T"])); D = int(args.get("-d", B * T))
        except Exception:
            return e, "-b/-t/-d must be integers"
        if T != self.c["fixed_T"]:
            return e, f"-t is fixed at {self.c['fixed_T']} so val bpb stays comparable"
        if self.c["eval_tokens"] % (B * T):
            return e, f"B*T={B * T} must divide the fixed eval size {self.c['eval_tokens']}"
        if D % (B * T):
            return e, f"-d {D} must be a multiple of B*T={B * T}"
        ref = str(e.get("code_ref") or "")
        if not re.match(r"^[0-9a-f]{7,40}$", ref):
            return e, "code_ref must be a commit hash on the autoresearch branch"
        r = subprocess.run(["git", "-C", self.c["ctrl"], "rev-parse", "--verify", "-q", ref + "^{commit}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return e, f"code_ref {ref} not found on branch {self.c['branch']}"
        e["code_ref"] = r.stdout.strip()
        e["B"], e["T"], e["D"] = B, T, D
        return e, None

    # ---------- build ----------
    def code_key(self, sha):
        """Rebuild only when trainer sources change: key = git tree ids of src/ and third_party/llm.c + Makefile."""
        parts = []
        for p in ("src", "third_party/llm.c/llmc", "Makefile"):
            r = subprocess.run(["git", "-C", self.c["ctrl"], "rev-parse", f"{sha}:{p}"], capture_output=True, text=True)
            parts.append(r.stdout.strip())
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]

    def binary_for(self, sha):
        key = self.code_key(sha)
        if key in self.bin_cache:
            return self.bin_cache[key], key
        d = os.path.join(self.work, "build", key)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        a = subprocess.run(f"git -C {self.c['ctrl']} archive {sha} | tar -x -C {d}", shell=True, capture_output=True, text=True)
        if a.returncode != 0:
            raise RuntimeError(f"git archive failed: {a.stderr[-300:]}")
        self.log(f"  building trainer for code {sha[:7]} (source key {key}) ...")
        t = time.time()
        b = self.c["build_fn"](d, os.path.join(d, "build.log"))
        self.log(f"  built in {time.time() - t:.0f}s")
        self.bin_cache[key] = b
        return b, key

    # ---------- one experiment ----------
    def run_one(self, e):
        rid = e["id"]
        rdir = os.path.join(self.res_dir, "runs", rid)
        shutil.rmtree(rdir, ignore_errors=True)
        os.makedirs(rdir)
        logp = os.path.join(rdir, "run.log")
        rec = {"id": rid, "description": e.get("description", ""), "commit": e["code_ref"], "args": e["args"],
               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "time_budget_s": self.c["time_budget_s"],
               "eval_tokens": self.c["eval_tokens"], "val_bpb": 0.0, "memory_gb": 0.0, "status": "crash"}
        self.state.update(phase="building", current=rid)
        try:
            binary, key = self.binary_for(e["code_ref"])
            rec["source_key"] = key
        except Exception as ex:
            rec["error"] = f"build failed: {str(ex)[:300]}"
            open(logp, "w").write(rec["error"] + "\n")
            return rec, logp
        os.makedirs(os.path.join(self.work, "scratch"), exist_ok=True)
        cmd = [binary, "-i", self.c["train_glob"], "-j", self.c["val_bin"], "-o", os.path.join(self.work, "scratch", rid),
               "-x", "1000000", "-v", "0", "-m", str(self.c["eval_tokens"] // (e["B"] * e["T"])), "-n", "0", "-y", "0",
               "--time-budget", str(self.c["time_budget_s"]), "--vocab", str(self.c["vocab"]),
               "--token-bytes", self.c["token_bytes"]]
        args = dict(e["args"])
        args.setdefault("-e", "d12"); args.setdefault("-b", e["B"]); args.setdefault("-t", e["T"]); args.setdefault("-d", e["D"])
        for k, v in args.items():
            cmd += [k, str(v)]
        rec["cmd"] = " ".join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd))
        hard = self.c["time_budget_s"] + self.c["exp_overhead_s"]
        self.state.update(phase="training", current=rid, current_started=time.time())
        self.log(f"  run {rid}: {e.get('description', '')}\n  {rec['cmd']}")
        t0 = time.time()
        timed_out = False
        with open(logp, "w") as lf:
            lf.write(f"# {rid}: {e.get('description', '')}\n# {rec['cmd']}\n")
            lf.flush()
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True,
                                 env=self.c.get("train_env"))
            self._proc = p
            while p.poll() is None:
                if time.time() - t0 > hard:
                    timed_out = True
                if timed_out or self.stop_flag.is_set():
                    try: os.killpg(p.pid, signal.SIGKILL)
                    except Exception: pass
                    p.wait()
                    break
                time.sleep(2)
                self.heartbeat()
            self._proc = None
        rec["wall_s"] = round(time.time() - t0, 1)
        txt = open(logp, errors="replace").read()
        summ = {}
        if "\n---\n" in txt:
            for line in txt.split("\n---\n")[-1].splitlines():
                m = re.match(r"^([a-z_]+):\s+([-\d.]+)\s*$", line)
                if m: summ[m.group(1)] = float(m.group(2))
        rec["summary"] = summ
        if p.returncode == 0 and summ.get("val_bpb", 0) > 0:
            rec.update(status="ok", val_bpb=round(summ["val_bpb"], 6), memory_gb=round(summ.get("peak_vram_mb", 0) / 1024, 1))
        else:
            low = txt.lower()
            why = ("timeout" if timed_out else "stopped by watchdog" if self.stop_flag.is_set()
                   else "OOM" if "out of memory" in low else "loss not finite" if "not finite" in low
                   else f"exit {p.returncode}")
            rec["error"] = why
            rec["log_tail"] = txt[-1500:]
        shutil.rmtree(os.path.join(self.work, "scratch", rid), ignore_errors=True)
        return rec, logp

    # ---------- main loop ----------
    def run(self):
        c = self.c
        self.load_previous()
        self.upload(what="session start")
        idle_since = None
        last_poll = 0.0
        while True:
            if self.stop_flag.is_set():
                self.state["stop_reason"] = "watchdog"; break
            if time.time() - last_poll >= c["poll_s"] or idle_since is None:
                try:
                    self.refresh()
                except Exception as ex:
                    self.log(f"  [git] fetch failed: {str(ex)[-200:]}")
                last_poll = time.time()
            ctl = self.control()
            cap = min(c["max_experiments"], int(ctl.get("max_experiments") or c["max_experiments"]))
            if ctl.get("stop"):
                self.state["stop_reason"] = "control.json stop=true"; break
            if self.state["experiments_this_session"] >= cap:
                self.state["stop_reason"] = f"max experiments ({cap}) reached"; break
            left = c["session_deadline"] - time.time()
            if left < c["time_budget_s"] + c["exp_overhead_s"]:
                self.state["stop_reason"] = "session time cap: not enough time left for another experiment"; break
            todo = self.pending()
            if not todo:
                if idle_since is None:
                    idle_since = time.time()
                    self.state.update(phase="idle", current=None, idle_since=idle_since)
                    self.log(f"queue empty; waiting up to {c['idle_timeout_s'] / 60:g} min for a new experiment")
                    self.upload(what="idle")
                if time.time() - idle_since >= c["idle_timeout_s"]:
                    self.state["stop_reason"] = f"idle timeout ({c['idle_timeout_s'] / 60:g} min with an empty queue)"; break
                self.heartbeat()
                time.sleep(min(c["poll_s"], 15))
                continue
            idle_since = None
            self.state["idle_since"] = None
            e, err = self.validate(todo[0])
            if err:
                rec = {"id": e["id"], "description": e.get("description", ""), "commit": str(e.get("code_ref") or "0000000"),
                       "val_bpb": 0.0, "memory_gb": 0.0, "status": "crash", "error": f"rejected: {err}",
                       "started": time.strftime("%Y-%m-%d %H:%M:%S")}
                self.log(f"  {e['id']} rejected: {err}")
                self.record(rec)
                self.upload(what=f"{e['id']} rejected")
                continue
            rec, logp = self.run_one(e)
            self.state["experiments_this_session"] += 1
            self.record(rec)
            b = self.best()
            self.log(f"  -> {rec['id']}: {rec['status']}"
                     + (f" val_bpb {rec['val_bpb']:.6f}, {rec['memory_gb']} GB, {rec.get('summary', {}).get('num_steps', 0):.0f} steps"
                        if rec["status"] != "crash" else f" ({rec.get('error')})")
                     + (f" | best {b['id']} {b['val_bpb']:.6f}" if b else ""))
            rdir = os.path.dirname(logp)
            json.dump(rec, open(os.path.join(rdir, "result.json"), "w"), indent=1)
            self.state.update(phase="uploading", current=None)
            self.upload(extra=[(logp, f"runs/{rec['id']}/run.log"), (os.path.join(rdir, "result.json"), f"runs/{rec['id']}/result.json")],
                        what=f"{rec['id']} {rec['status']}")
            last_poll = 0.0   # look for new work right away
        self.state.update(phase="finished", current=None)
        self.log(f"worker stopping: {self.state['stop_reason']}")
        return self.state["stop_reason"]

    def final_upload(self, timeout=300):
        self.state["phase"] = "finished"
        return self.upload(what=f"session end: {self.state.get('stop_reason')}", timeout=timeout)
