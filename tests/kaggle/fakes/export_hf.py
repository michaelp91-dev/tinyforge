import argparse, os, json, struct
p = argparse.ArgumentParser(); p.add_argument("--input", "-i"); p.add_argument("--output", "-o"); p.add_argument("--tokenizer", "-t")
a = p.parse_args()
h = struct.unpack("<2i", open(a.input, "rb").read(8)); assert h[0] == 20241114, "not a model file"
os.makedirs(a.output, exist_ok=True)
json.dump({"model_type": "tinyforge", "fake": True}, open(os.path.join(a.output, "config.json"), "w"))
open(os.path.join(a.output, "model.safetensors"), "wb").write(os.urandom(2048))
open(os.path.join(a.output, "tokenizer.json"), "w").write(open(os.path.join(a.tokenizer, "tokenizer.json")).read())
print("fake export ok")
