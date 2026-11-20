#!/usr/bin/env python3
"""Box dry run of notebooks/TinyForge-Kaggle-Pretrain.ipynb: runs every code cell (like Run All) with a fake
multi-rank trainer, fake data builder/export, and hf_sync.py's local mock Hub. No GPU, no network.
  python3 tests/kaggle/dryrun.py [scenario ...]"""
import glob, hashlib, json, os, shutil, struct, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NB = os.path.join(ROOT, "notebooks", "TinyForge-Kaggle-Pretrain.ipynb")
W = os.environ.get("DRYRUN_DIR", "/tmp/tf_kaggle_dryrun")

def make_repo():
    repo = os.path.join(W, "repo")
    shutil.rmtree(repo, ignore_errors=True)
    os.makedirs(os.path.join(repo, "python"))
    shutil.copy(os.path.join(ROOT, "python", "hf_sync.py"), os.path.join(repo, "python"))
    for f in glob.glob(os.path.join(HERE, "fakes", "*.py")):
        shutil.copy(f, os.path.join(repo, "python"))
    subprocess.run("git init -q && git add -A && git -c user.name=t -c user.email=t@t commit -qm 'fake repo'", shell=True, cwd=repo, check=True)
    return repo

TOK_JSON = '{"fake": "tokenizer of the 8-hour run"}'
TOK_SHA = hashlib.sha256(TOK_JSON.encode()).hexdigest()[:16]

def seed_hub(hub, with_8hr=True):
    d = os.path.join(hub, "dataset", "mockuser__tinyforge-data", "climbmix_v32768_610M", "tokenizer")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "tokenizer.json"), "w").write(TOK_JSON)
    open(os.path.join(d, "token_bytes.bin"), "wb").write(b"\1" * 1024)
    os.makedirs(os.path.join(hub, "model", "mockuser__tinyforge-runs"), exist_ok=True)
    if with_8hr:   # the 8-hour run: world 1, step 10, cursor shard 1 sample 4 (8192-token samples)
        c = os.path.join(hub, "model", "mockuser__tinyforge-runs", "d12_8hr", "ckpt")
        os.makedirs(c, exist_ok=True)
        with open(os.path.join(c, "state.bin"), "wb") as f:
            f.write(struct.pack("<256i", *([20241115, 1, 10, 1, 4, 16, 512, 1] + [0] * 248))); f.write(os.urandom(4096))
        json.dump({"step": 10, "max_steps": 9200, "tok_sha": TOK_SHA, "data_tag": "climbmix_v32768_610M",
                   "bytes": os.path.getsize(os.path.join(c, "state.bin"))}, open(os.path.join(c, "meta.json"), "w"))

BASE = dict(TF_LOCAL_TEST="1", TF_FAKE_BIN=os.path.join(HERE, "fake_trainer.py"), TF_HF_MOCK_USER="mockuser",
            TF_MAX_STEPS="54", TF_TRAIN_TOKENS="2000000", TF_SHARD_TOKENS="200000", TF_VAL_TOKENS="8192",
            TF_CKPT_EVERY="4", TF_VAL_EVERY="10", TF_BUDGET_H="0.006", TF_FINAL_RESERVE_MIN="0.15", TF_CKPT_PUSH_MIN="0.1",
            TF_LOG_PUSH_MIN="0.1", TF_WATCHDOG_H="0.2", TF_STARTUP_TIMEOUT_S="8", TF_MIN_LAUNCH_S="3", TF_MIN_RESTART_S="5",
            TF_DATA_WAIT_POLL_S="1", TF_DO_SAMPLE="0", TF_PROC_WATCHDOG_EXTRA_S="5", TF_FINAL_HARD_TIMEOUT_S="120",
            FAKE_STEP_S="0.3", FAKE_SHARD_S="0.5", TF_TEST_TOKEN="hf_FAKE_SECRET_never_printed_42")

RUNNER = r'''
import json, sys, os
nb = json.load(open(sys.argv[1]))
g = {"__name__": "__main__"}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        print(f"\n######## cell {i} ########", flush=True)
        exec(compile("".join(c["source"]), f"<cell {i}>", "exec"), g)
json.dump({"STATUS": g.get("STATUS"), "PROGRESS": g.get("PROGRESS"), "DATA_STATE": {k: v for k, v in g.get("DATA_STATE", {}).items() if k != "upload_queue"},
           "SYNC": g.get("SYNC"), "NGPU": g.get("NGPU"), "FINAL": {k: v for k, v in g.get("FINAL", {}).items() if k != "files"},
           "SUMMARY": g.get("SUMMARY")}, open(os.path.join(os.environ["TF_CONTENT"], "result.json"), "w"), default=str, indent=1)
'''

def run(name, hub, repo, trace, **env):
    content = os.path.join(W, name, "content")
    out = os.path.join(W, name, "kaggle_working")
    shutil.rmtree(os.path.join(W, name), ignore_errors=True)
    os.makedirs(content)
    e = dict(os.environ, **BASE)
    e.update(TF_CONTENT=content, TF_OUT=out, TF_REPO=repo, TF_HF_MOCK_DIR=hub, FAKE_TRACE=trace)
    e.update({k: str(v) for k, v in env.items()})
    open(os.path.join(W, "runner.py"), "w").write(RUNNER)
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(W, "runner.py"), NB], env=e, capture_output=True, text=True, timeout=600)
    log = r.stdout + r.stderr
    open(os.path.join(W, name, "log.txt"), "w").write(log)
    try:   # stop the process watchdog of this "kernel"
        os.kill(int(open(os.path.join(content, "tf_watchdog.pid")).read()), 9)
    except Exception:
        pass
    res = json.load(open(os.path.join(content, "result.json"))) if os.path.exists(os.path.join(content, "result.json")) else {}
    print(f"--- {name}: exit {r.returncode} in {time.time() - t:.0f}s; stages: " +
          ", ".join(f"{k}={'ok' if v.get('ok') else 'FAIL'}" for k, v in (res.get("STATUS") or {}).items() if isinstance(v, dict)))
    return res, log, content

def hub_files(hub, typ, repo):
    d = os.path.join(hub, typ, repo)
    return {os.path.relpath(os.path.join(r, f), d): os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs}

FAILS = []
def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond: FAILS.append(msg)

def effective(trace):
    """The trainer rewinds to its last checkpoint on a restart, so steps after it are run again.
    Replay the trace the same way: a step number <= the last one seen drops everything from that step on."""
    eff = []
    for st, sh, si, w in (l.split() for l in open(trace)):
        st = int(st)
        if eff and st < eff[-1][0]:
            eff = [r for r in eff if r[0] < st]
        elif eff and st == eff[-1][0] and eff[-1][3] != w:
            eff = [r for r in eff if r[0] < st]
        eff.append((st, sh, int(si), w))
    return eff

def trace_ok(trace, label, max_steps, accum_by_world={"2": 2, "1": 4}):
    eff = effective(trace)
    steps = sorted({r[0] for r in eff})
    keys = [(r[1], r[2] * int(r[3])) for r in eff]   # sample index in units of 8192-token micro-batches
    per = {}
    for r in eff: per[r[0]] = per.get(r[0], 0) + 1
    check(steps == list(range(steps[0] if steps else 0, max_steps)), f"{label}: every step run, none missing ({len(steps)} steps)")
    check(all(per[s] == accum_by_world[[r for r in eff if r[0] == s][0][3]] for s in per), f"{label}: each step used its accumulation count of micro-batches")
    check(len(keys) == len(set(keys)), f"{label}: no data consumed twice in the effective run ({len(keys)} micro-batches)")
    return eff

def main(which):
    shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
    repo = make_repo()
    S = set(which) if which else None
    want = lambda n: S is None or n in S

    # S1 + S2 + S3: fresh run over two sessions, then a re-run after it finished
    hub = os.path.join(W, "hub_main"); seed_hub(hub); tr = os.path.join(W, "trace_main.txt")
    if want("main"):
        res, log, c1 = run("s1_fresh_session1", hub, repo, tr)
        P = res.get("PROGRESS", {})
        check(res["STATUS"]["setup"]["ok"] and res["STATUS"]["build"]["ok"] and res["STATUS"]["data"]["ok"], "s1: setup/build/data ok")
        check(res["STATUS"]["train"]["ok"], "s1: train stage ok")
        check(P.get("phase") == "stopped" and 0 < P.get("step", 0) < 54, f"s1: stopped at the time limit mid-run (step {P.get('step')})")
        check("directory missing" not in log, "s1: run/sync dirs existed at every launch")
        check("--nproc 2 --rank 0 --sync-dir" in log, "s1: launched 2 ranks")
        check("B=16 T=512 nproc=2 total_batch=32768 accum=2" in log, "s1: 2 GPUs x 16 x 512 = 16,384 per micro-step, accum 2 -> 32,768 per step")
        check(res["DATA_STATE"]["tokenizer_from"] == "climbmix_v32768_610M" and res["DATA_STATE"]["tok_sha"] == TOK_SHA,
              "s1: tokenizer copied from the 610M set (same sha as the 8-hour run)")
        hf = hub_files(hub, "model", "mockuser__tinyforge-runs")
        for f in ("d12_kaggle/ckpt/state.bin", "d12_kaggle/ckpt/meta.json", "d12_kaggle/ckpt/model.bin", "d12_kaggle/train.log",
                  "d12_kaggle/status.json", "d12_kaggle/sessions.json", "d12_kaggle/summary.json", "d12_kaggle/logs/data_build.log"):
            check(f in hf, f"s1: HF runs has {f}")
        meta = json.load(open(os.path.join(hub, "model", "mockuser__tinyforge-runs", "d12_kaggle/ckpt/meta.json")))
        check(meta["step"] == P["step"] and meta["total_batch"] == 32768 and meta["ngpu"] == 2, f"s1: ckpt meta step {meta['step']} = last step, 32k, 2 GPUs")
        hd = hub_files(hub, "dataset", "mockuser__tinyforge-data")
        check("climbmix_v32768_2M/meta.json" in hd and "climbmix_v32768_2M/tokenizer/tokenizer.json" in hd and "climbmix_v32768_2M/climbmix/val.bin" in hd,
              "s1: data set complete on HF (tokenizer, val, meta.json)")
        check(sum(1 for k in hd if "/climbmix/train_" in k) == 10, "s1: all 10 shards uploaded")
        check(res["PROGRESS"]["restarts_for_data"] >= 1, f"s1: trainer picked up new shards ({res['PROGRESS']['restarts_for_data']} restart(s))")
        check("verified: all final files are on HF" in log, "s1: final verify passed")
        check(os.path.exists(os.path.join(W, "s1_fresh_session1", "kaggle_working", "status.json")), "s1: status.json copied to Kaggle Output")
        s1_step = P["step"]

        res, log, c2 = run("s2_resume_session2", hub, repo, tr, TF_BUDGET_H="0.2")
        P = res["PROGRESS"]
        check(res["DATA_STATE"]["source"] == "hf", "s2: data downloaded from HF (no rebuild)")
        check(f"resuming d12_kaggle from step {s1_step}" in log, f"s2: resumed from HF checkpoint step {s1_step}")
        check(P["phase"] == "finished" and P["step"] == 54, "s2: reached MAX_STEPS")
        check(res["STATUS"]["export"]["ok"] and res["FINAL"]["exported"], "s2: export stage ran")
        hf = hub_files(hub, "model", "mockuser__tinyforge-runs")
        for f in ("d12_kaggle/final/tinyforge-d12-f32.gguf", "d12_kaggle/final/hf/config.json", "d12_kaggle/final/hf/model.safetensors",
                  "d12_kaggle/final/export.json"):
            check(f in hf, f"s2: HF has {f}")
        sess = json.load(open(os.path.join(hub, "model", "mockuser__tinyforge-runs", "d12_kaggle/sessions.json")))
        check(len(sess["sessions"]) == 2 and sess["sessions"][0]["end_step"] == s1_step, "s2: sessions.json has both sessions")
        check(os.path.exists(os.path.join(W, "s2_resume_session2", "kaggle_working", "final", "tinyforge-d12-f32.gguf")), "s2: export copied to Kaggle Output")
        eff = trace_ok(tr, "trace s1+s2", 54)
        check(all(r[3] == "2" for r in eff), "trace: all on 2 ranks")

        res, log, c3 = run("s3_rerun_after_finish", hub, repo, tr)
        check(res["STATUS"]["train"]["ok"] and "nothing to train" in log, "s3: re-run after finish trains nothing")
        check("already on HF" in log, "s3: export not redone")

    if want("continue"):   # S4: INIT_FROM = d12_8hr
        hub4 = os.path.join(W, "hub_cont"); seed_hub(hub4); tr4 = os.path.join(W, "trace_cont.txt")
        res, log, _ = run("s4_continue_8hr", hub4, repo, tr4, TF_INIT_FROM="d12_8hr", TF_BUDGET_H="0.2")
        check("continuing from d12_8hr step 10" in log, "s4: started from the 8-hour checkpoint at step 10")
        check("checkpoint was nproc=1, now nproc=2: data cursor rescaled" in log, "s4: data cursor rescaled 1 -> 2 GPUs")
        rows = [l.split() for l in open(tr4)]
        check(rows and rows[0][0] == "10" and rows[0][1] == "train_00001.bin" and rows[0][2] == "2", f"s4: first micro-batch = step 10, shard 1, sample 4/2=2 ({rows[0] if rows else None})")
        check(res["PROGRESS"]["init_from"] == "d12_8hr@10" and res["PROGRESS"]["phase"] == "finished", "s4: init_from recorded, run finished")
        meta = json.load(open(os.path.join(hub4, "model", "mockuser__tinyforge-runs", "d12_kaggle/ckpt/meta.json")))
        check(meta["init_from"] == "d12_8hr@10", "s4: ckpt meta records init_from")
        h8 = json.load(open(os.path.join(hub4, "model", "mockuser__tinyforge-runs", "d12_8hr/ckpt/meta.json")))
        check(h8["step"] == 10, "s4: the d12_8hr checkpoint itself is untouched")

    if want("rank1crash"):
        hub5 = os.path.join(W, "hub_r1"); seed_hub(hub5)
        res, log, _ = run("s5_rank1_crash", hub5, repo, os.path.join(W, "trace_r1.txt"), FAKE_RANK1_CRASH_AT="14", TF_BUDGET_H="0.2")
        check("rank 1 exited with code 1; stopping rank 0" in log, "s5: rank-1 death detected, rank 0 stopped")
        check(res["PROGRESS"]["crash_restarts"] == 1 and res["PROGRESS"]["phase"] == "finished", "s5: restarted from checkpoint and finished")

    if want("ncclhang"):
        hub6 = os.path.join(W, "hub_hang"); seed_hub(hub6)
        res, log, _ = run("s6_nccl_hang", hub6, repo, os.path.join(W, "trace_hang.txt"), FAKE_NCCL_HANG="1", TF_BUDGET_H="0.2")
        check("no training step after 8s on 2 GPUs" in log and "continuing on ONE GPU" in log, "s6: 2-GPU stall -> fallback to one GPU")
        check(res["NGPU"] == 1 and res["PROGRESS"]["phase"] == "finished", "s6: finished on one GPU")
        check("nproc=1 total_batch=32768 accum=4" in log, "s6: one GPU uses 4 accumulation steps for 32k")

    if want("ncclbuild"):
        hub9 = os.path.join(W, "hub_nb"); seed_hub(hub9)
        res, log, _ = run("s9_nccl_build_fail", hub9, repo, os.path.join(W, "trace_nb.txt"), TF_FAKE_NCCL_BUILD_FAIL="1", TF_BUDGET_H="0.2")
        check(res["NGPU"] == 1 and "--nproc" not in log.split("launch 1")[1].split("\n")[1] and res["PROGRESS"]["phase"] == "finished",
              "s9: NCCL build failure -> one-GPU run finished")

    if want("partialdata"):   # S7: session 2 finds incomplete data on HF and a cursor beyond the first shard
        hub7 = os.path.join(W, "hub_part"); seed_hub(hub7); tr7 = os.path.join(W, "trace_part.txt")
        res, log, _ = run("s7a_session1", hub7, repo, tr7)
        step1 = res["PROGRESS"]["step"]
        dd = os.path.join(hub7, "dataset", "mockuser__tinyforge-data", "climbmix_v32768_2M")
        os.remove(os.path.join(dd, "meta.json"))
        for f in glob.glob(os.path.join(dd, "climbmix", "train_0000[5-9].bin")): os.remove(f)
        res, log, _ = run("s7b_session2_rebuild", hub7, repo, tr7, TF_BUDGET_H="0.2", FAKE_SHARD_S="1.5")
        check("is incomplete" in log and res["DATA_STATE"]["source"] == "built", "s7: incomplete HF data detected and rebuilt")
        check("[data] waiting for shards to reach step" in log, f"s7: waited for the build to cover the resume point (step {step1})")
        check("Segmentation fault" not in log and res["PROGRESS"]["phase"] == "finished", "s7: no out-of-range shard crash; finished")
        trace_ok(tr7, "s7 trace", 54)

    if want("watchdog"):
        hub8 = os.path.join(W, "hub_wd"); seed_hub(hub8)
        res, log, c8 = run("s8_watchdog", hub8, repo, os.path.join(W, "trace_wd.txt"), TF_BUDGET_H="1", TF_WATCHDOG_H="0.006",
                           TF_MAX_STEPS="5000", TF_TRAIN_TOKENS="2000000")
        check("WATCHDOG" in log and os.path.exists(os.path.join(c8, "end_session.log")), "s8: watchdog fired and ended the session")
        hf = hub_files(hub8, "model", "mockuser__tinyforge-runs")
        check("d12_kaggle/train.log" in hf, "s8: logs reached HF")

    if want("lsfail"):   # HF listing fails at resume: must refuse rather than start fresh over an existing checkpoint
        hub10 = os.path.join(W, "hub_ls"); seed_hub(hub10)
        if not os.path.exists(os.path.join(hub, "model", "mockuser__tinyforge-runs", "d12_kaggle")):
            check(False, "s10 needs the main scenario first"); return
        shutil.copytree(os.path.join(hub, "model"), os.path.join(hub10, "model"), dirs_exist_ok=True)
        before = open(os.path.join(hub10, "model", "mockuser__tinyforge-runs", "d12_kaggle/ckpt/meta.json")).read()
        res, log, _ = run("s10_ls_fail", hub10, repo, os.path.join(W, "trace_ls.txt"), TF_HF_MOCK_FAIL="ls")
        check(not res["STATUS"]["train"]["ok"] and "refusing to start" in log, "s10: refuses to train when the runs repo can't be listed")
        after = open(os.path.join(hub10, "model", "mockuser__tinyforge-runs", "d12_kaggle/ckpt/meta.json")).read()
        check(before == after, "s10: existing checkpoint untouched")

    leaks = subprocess.run(["grep", "-rl", "hf_FAKE_SECRET_never_printed_42", W, "--exclude=runner.py"], capture_output=True, text=True).stdout.split()
    check(not leaks, f"token never written to any log, status, or uploaded file ({leaks[:3]})")
    print("\nRESULT:", "ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)

if __name__ == "__main__":
    main(sys.argv[1:])
