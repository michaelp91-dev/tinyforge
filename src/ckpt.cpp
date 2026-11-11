#include "tf_ckpt.hpp"
#include "tf_util.hpp"

#include <cstring>
#include <fstream>

namespace tf {

static void write_str(std::ofstream& o, const std::string& s) {
    uint32_t n = (uint32_t)s.size();
    o.write((char*)&n, 4);
    o.write(s.data(), n);
}
static std::string read_str(std::ifstream& in) {
    uint32_t n = 0;
    in.read((char*)&n, 4);
    std::string s(n, '\0');
    if (n) in.read(s.data(), n);
    return s;
}

void save_checkpoint(const std::string& path, const Checkpoint& ckpt) {
    std::ofstream o(path, std::ios::binary);
    check(o.good(), "cannot write checkpoint");
    o.write("TFCK0001", 8);
    const GPTConfig& c = ckpt.cfg;
    int32_t fields[] = {c.n_layer, c.n_embd, c.n_head, c.n_kv_head, c.seq_len, c.vocab, ckpt.step};
    o.write((char*)fields, sizeof(fields));
    float ff[] = {c.rope_base, c.softcap, c.qk_scale, c.rms_eps, c.matrix_lr, c.embedding_lr, c.unembedding_lr,
                  c.scalar_lr, c.weight_decay, c.smear_lr, c.warmdown_ratio, c.final_lr_frac, c.data_ratio};
    o.write((char*)ff, sizeof(ff));
    int32_t warmup = c.warmup_steps;
    o.write((char*)&warmup, 4);
    uint64_t seed = c.seed;
    o.write((char*)&seed, 8);
    o.write((char*)&ckpt.doc_cursor, 8);
    write_str(o, c.window_pattern);
    uint64_t n = ckpt.params.size();
    o.write((char*)&n, 8);
    o.write((char*)ckpt.params.data(), sizeof(float) * n);
    check(o.good(), "checkpoint write failed");
}

Checkpoint load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    check(in.good(), "cannot read checkpoint " + path);
    char magic[8];
    in.read(magic, 8);
    check(std::memcmp(magic, "TFCK0001", 8) == 0, "bad checkpoint magic");
    Checkpoint ckpt;
    int32_t fields[7];
    in.read((char*)fields, sizeof(fields));
    ckpt.cfg.n_layer = fields[0];
    ckpt.cfg.n_embd = fields[1];
    ckpt.cfg.n_head = fields[2];
    ckpt.cfg.n_kv_head = fields[3];
    ckpt.cfg.seq_len = fields[4];
    ckpt.cfg.vocab = fields[5];
    ckpt.step = fields[6];
    float ff[13];
    in.read((char*)ff, sizeof(ff));
    ckpt.cfg.rope_base = ff[0];
    ckpt.cfg.softcap = ff[1];
    ckpt.cfg.qk_scale = ff[2];
    ckpt.cfg.rms_eps = ff[3];
    ckpt.cfg.matrix_lr = ff[4];
    ckpt.cfg.embedding_lr = ff[5];
    ckpt.cfg.unembedding_lr = ff[6];
    ckpt.cfg.scalar_lr = ff[7];
    ckpt.cfg.weight_decay = ff[8];
    ckpt.cfg.smear_lr = ff[9];
    ckpt.cfg.warmdown_ratio = ff[10];
    ckpt.cfg.final_lr_frac = ff[11];
    ckpt.cfg.data_ratio = ff[12];
    int32_t warmup = 0;
    in.read((char*)&warmup, 4);
    ckpt.cfg.warmup_steps = warmup;
    in.read((char*)&ckpt.cfg.seed, 8);
    in.read((char*)&ckpt.doc_cursor, 8);
    ckpt.cfg.window_pattern = read_str(in);
    uint64_t n = 0;
    in.read((char*)&n, 8);
    ckpt.params.resize(n);
    in.read((char*)ckpt.params.data(), sizeof(float) * n);
    check(in.good(), "truncated checkpoint");
    return ckpt;
}

}  // namespace tf
