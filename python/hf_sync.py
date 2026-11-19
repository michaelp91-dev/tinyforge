"""Small Hugging Face Hub helper for the TinyForge Colab notebooks.

Every call runs as its own process so the notebook can kill it on a timeout;
a stuck upload can never block training or the final disconnect.

The token is read from the HF_TOKEN environment variable and is never printed.

  hf_sync.py whoami --repo -                                 -> account name
  hf_sync.py ensure --repo user/name --type model|dataset
  hf_sync.py ls     --repo R --type T [--prefix P]           -> JSON {path: size}
  hf_sync.py push   --repo R --type T --file LOCAL=REMOTE ... [--squash] [-m MSG]
  hf_sync.py pull   --repo R --type T --file REMOTE=LOCAL ...
  hf_sync.py pulldir --repo R --type T --prefix P --dest DIR

Exit code 0 on success, 2 when a pulled file does not exist, 1 on any other error.
Box-side tests set TF_HF_MOCK_DIR to use a local folder instead of the Hub.
"""

import argparse
import json
import os
import shutil
import sys
import time


def _pairs(items):
    out = []
    for it in items or []:
        a, b = it.split("=", 1)
        out.append((a, b))
    return out


class MockApi:
    """Local-folder stand-in for HfApi (tests only). TF_HF_MOCK_HANG=<op> sleeps forever;
    TF_HF_MOCK_FAIL=<op> raises, to exercise the notebook's timeouts and retries."""

    def __init__(self, root):
        self.root = root

    def _chk(self, op):
        if os.environ.get("TF_HF_MOCK_HANG") == op:
            time.sleep(10 ** 6)
        if os.environ.get("TF_HF_MOCK_FAIL") == op:
            raise RuntimeError(f"mock failure in {op}")

    def _dir(self, repo, typ):
        return os.path.join(self.root, typ, repo.replace("/", "__"))

    def whoami(self):
        self._chk("whoami")
        return os.environ.get("TF_HF_MOCK_USER", "mockuser")

    def ensure(self, repo, typ):
        self._chk("ensure")
        d = self._dir(repo, typ)
        created = not os.path.isdir(d)
        os.makedirs(d, exist_ok=True)
        return created, True

    def ls(self, repo, typ):
        self._chk("ls")
        d = self._dir(repo, typ)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"repo {repo} not found")
        out = {}
        for r, _, fs in os.walk(d):
            for f in fs:
                p = os.path.join(r, f)
                out[os.path.relpath(p, d)] = os.path.getsize(p)
        return out

    def push(self, repo, typ, files, msg, squash):
        self._chk("push")
        d = self._dir(repo, typ)
        for local, remote in files:
            dst = os.path.join(d, remote)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(local, dst + ".part")
            os.replace(dst + ".part", dst)
        if squash:
            open(os.path.join(self.root, "squash.log"), "a").write(f"{repo} {time.time()}\n")

    def pull(self, repo, typ, remote, local):
        self._chk("pull")
        src = os.path.join(self._dir(repo, typ), remote)
        if not os.path.exists(src):
            raise FileNotFoundError(remote)
        os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
        shutil.copy(src, local + ".part")
        os.replace(local + ".part", local)


class HubApi:
    def __init__(self, token):
        from huggingface_hub import HfApi
        self.api = HfApi(token=token)
        self.token = token

    def whoami(self):
        return self.api.whoami()["name"]

    def ensure(self, repo, typ):
        created = False
        try:
            info = self.api.repo_info(repo, repo_type=typ)
        except Exception:
            self.api.create_repo(repo, repo_type=typ, private=True, exist_ok=True)
            info = self.api.repo_info(repo, repo_type=typ)
            created = True
        return created, bool(getattr(info, "private", False))

    def ls(self, repo, typ):
        out = {}
        for f in self.api.list_repo_tree(repo, repo_type=typ, recursive=True):
            size = getattr(f, "size", None)
            if size is not None and not hasattr(f, "tree_id"):   # files only
                out[f.path] = size
        return out

    def push(self, repo, typ, files, msg, squash):
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=l) for l, r in files]
        self.api.create_commit(repo, operations=ops, commit_message=msg, repo_type=typ)
        if squash:
            # One overwritten checkpoint path plus a squashed history keeps the repo size bounded:
            # old checkpoint blobs are no longer referenced and the Hub frees them.
            try:
                self.api.super_squash_history(repo, repo_type=typ, commit_message=msg)
            except Exception as e:
                print(f"squash failed (upload itself is fine): {type(e).__name__}: {e}", flush=True)

    def pull(self, repo, typ, remote, local):
        from huggingface_hub import hf_hub_download
        try:
            from huggingface_hub.utils import EntryNotFoundError
        except Exception:
            EntryNotFoundError = FileNotFoundError
        tmp = local + ".dl"
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            p = hf_hub_download(repo, remote, repo_type=typ, local_dir=tmp, token=self.token)
        except EntryNotFoundError as e:
            raise FileNotFoundError(remote) from e
        os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
        os.replace(p, local)
        shutil.rmtree(tmp, ignore_errors=True)


def _api():
    mock = os.environ.get("TF_HF_MOCK_DIR")
    if mock:
        return MockApi(mock)
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise SystemExit("HF_TOKEN is not set")
    return HubApi(tok)


def _retry(fn, tries, what):
    for i in range(tries):
        try:
            return fn()
        except FileNotFoundError:
            raise
        except Exception as e:
            msg = str(e).splitlines()[0][:300] if str(e) else ""
            print(f"{what} attempt {i + 1}/{tries} failed: {type(e).__name__}: {msg}", flush=True)
            if i + 1 == tries:
                raise
            time.sleep(min(60, 5 * 2 ** i))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["whoami", "ensure", "ls", "push", "pull", "pulldir"])
    p.add_argument("--repo", required=True)
    p.add_argument("--type", default="model", choices=["model", "dataset"])
    p.add_argument("--file", action="append")
    p.add_argument("--prefix", default="")
    p.add_argument("--dest")
    p.add_argument("--squash", action="store_true")
    p.add_argument("-m", "--message", default="tinyforge sync")
    p.add_argument("--tries", type=int, default=3)
    a = p.parse_args()
    api = _api()
    try:
        if a.cmd == "whoami":
            print(_retry(api.whoami, a.tries, "whoami"))
        elif a.cmd == "ensure":
            created, private = _retry(lambda: api.ensure(a.repo, a.type), a.tries, "ensure")
            print(json.dumps({"created": created, "private": private}))
        elif a.cmd == "ls":
            files = _retry(lambda: api.ls(a.repo, a.type), a.tries, "ls")
            files = {k: v for k, v in files.items() if k.startswith(a.prefix)}
            print(json.dumps(files))
        elif a.cmd == "push":
            files = _pairs(a.file)
            for l, _ in files:
                if not os.path.exists(l):
                    raise SystemExit(f"missing local file {l}")
            _retry(lambda: api.push(a.repo, a.type, files, a.message, a.squash), a.tries, "push")
            print(f"pushed {len(files)} file(s) to {a.repo}")
        elif a.cmd == "pull":
            for remote, local in _pairs(a.file):
                _retry(lambda: api.pull(a.repo, a.type, remote, local), a.tries, f"pull {remote}")
            print("pulled")
        elif a.cmd == "pulldir":
            files = _retry(lambda: api.ls(a.repo, a.type), a.tries, "ls")
            want = sorted(k for k in files if k.startswith(a.prefix.rstrip("/") + "/"))
            if not want:
                raise FileNotFoundError(a.prefix)
            for k in want:
                local = os.path.join(a.dest, k[len(a.prefix.rstrip("/")) + 1:])
                if os.path.exists(local) and os.path.getsize(local) == files[k]:
                    continue
                _retry(lambda: api.pull(a.repo, a.type, k, local), a.tries, f"pull {k}")
            print(f"pulled {len(want)} file(s)")
    except FileNotFoundError as e:
        print(f"not found: {e}", flush=True)
        sys.exit(2)
    except SystemExit:
        raise
    except Exception as e:
        msg = str(e).splitlines()[0][:300] if str(e) else ""
        print(f"error: {type(e).__name__}: {msg}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
