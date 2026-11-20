import argparse, os
p = argparse.ArgumentParser(); p.add_argument("--out"); p.add_argument("--vocab-size"); p.add_argument("--max-chars")
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
open(os.path.join(a.out, "tokenizer.json"), "w").write('{"fake": "trained-here"}')
open(os.path.join(a.out, "token_bytes.bin"), "wb").write(b"\0" * 1024)
print("fake tokenizer trained")
