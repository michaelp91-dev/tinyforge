import argparse, os
p = argparse.ArgumentParser(); p.add_argument("--model"); p.add_argument("--out"); p.add_argument("--outtype"); p.add_argument("--name")
a = p.parse_args(); assert os.path.exists(os.path.join(a.model, "config.json"))
open(a.out, "wb").write(b"GGUF" + os.urandom(1024)); print("fake gguf ok")
