// Zstandard decoder (RFC 8878) for the frames parquet-cpp writes.
// No dictionaries. Checksums are skipped (the parquet layer checks sizes).

#include "tf_zstd.hpp"
#include "tf_util.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace tf {
namespace {

constexpr uint32_t kMagic = 0xFD2FB528u;

static std::vector<uint8_t> g_ref;
static bool g_ref_on = false;
static int g_last_ll = -1, g_last_ml = -1, g_last_of = -1, g_last_seq = -1;
static void check_out(const std::vector<uint8_t>& out, size_t from);

int hibit(uint32_t v) { return v ? 31 - __builtin_clz(v) : 0; }

struct FseEntry {
    uint8_t symbol;
    uint8_t nbits;
    int32_t base;
};

void fse_build(std::vector<FseEntry>& table, int accuracy, const int16_t* norm, int nsym) {
    const unsigned size = 1u << accuracy;
    const unsigned mask = size - 1;
    const unsigned step = (size >> 1) + (size >> 3) + 3;
    table.assign(size, FseEntry{0, 0, 0});
    std::vector<uint16_t> next(nsym);
    unsigned high = size - 1;
    for (int s = 0; s < nsym; ++s) {
        if (norm[s] == -1) {
            table[high].symbol = static_cast<uint8_t>(s);
            next[s] = 1;
            check(high > 0, "zstd: FSE low-probability overflow");
            --high;
        } else {
            check(norm[s] >= 0, "zstd: negative FSE count");
            next[s] = static_cast<uint16_t>(norm[s]);
        }
    }
    unsigned pos = 0;
    for (int s = 0; s < nsym; ++s) {
        for (int i = 0; i < norm[s]; ++i) {
            table[pos].symbol = static_cast<uint8_t>(s);
            pos = (pos + step) & mask;
            while (pos > high) pos = (pos + step) & mask;
        }
    }
    for (unsigned u = 0; u < size; ++u) {
        unsigned ns = next[table[u].symbol]++;
        unsigned nb = static_cast<unsigned>(accuracy) - static_cast<unsigned>(hibit(ns));
        table[u].nbits = static_cast<uint8_t>(nb);
        table[u].base = static_cast<int32_t>((ns << nb) - size);
    }
}

// Backward bit reader. Bit 0 is the LSB of the first byte. `pos` is the
// index of the next bit to consume (the bit just below the end-of-stream 1).
struct BitBack {
    const uint8_t* data = nullptr;
    int64_t pos = 0;

    void init(const uint8_t* s, size_t n) {
        check(n > 0, "zstd: empty bitstream");
        data = s;
        uint8_t last = s[n - 1];
        check(last != 0, "zstd: missing bitstream end mark");
        pos = static_cast<int64_t>(n - 1) * 8 + hibit(last);
    }
    int bit_at(int64_t i) const {
        if (i < 0) return 0;
        return (data[static_cast<size_t>(i) >> 3] >> (i & 7)) & 1;
    }
    uint32_t read(int nb) {
        uint32_t v = 0;
        for (int i = 0; i < nb; ++i) {
            --pos;
            v = (v << 1) | static_cast<uint32_t>(bit_at(pos));
        }
        return v;
    }
    uint32_t peek(int nb) const {
        uint32_t v = 0;
        int64_t q = pos;
        for (int i = 0; i < nb; ++i) {
            --q;
            v = (v << 1) | static_cast<uint32_t>(bit_at(q));
        }
        return v;
    }
    void skip(int nb) { pos -= nb; }
    int64_t remaining() const { return pos; }
};

// Forward bit reader for FSE normalized-count headers.
struct BitFwd {
    const uint8_t* begin = nullptr;
    const uint8_t* p = nullptr;
    const uint8_t* end = nullptr;
    uint32_t acc = 0;
    int bits = 0;

    void init(const uint8_t* s, size_t n) {
        begin = s;
        p = s;
        end = s + n;
    }
    void pull() {
        if (p < end) {
            acc |= static_cast<uint32_t>(*p++) << bits;
            bits += 8;
        }
    }
    uint32_t peek_mask(uint32_t mask) {
        while (bits < 24 && p < end) pull();
        return acc & mask;
    }
    void consume(int n) {
        acc >>= n;
        bits -= n;
    }
    size_t bytes_used() const {
        // Bytes actually consumed, rounded up so the unused low bits of the last
        // header byte count, but speculative pulls past that byte do not.
        size_t pulled_bits = static_cast<size_t>(p - begin) * 8;
        size_t consumed = pulled_bits - static_cast<size_t>(bits);
        return (consumed + 7) / 8;
    }
};

// Port of FSE_readNCount. Counts are written for symbols 0..nsym-1.
// Trailing symbols the caller expects can be zero.
void fse_read_ncount(const uint8_t* src, size_t len, int max_symbol, int& accuracy, std::vector<int16_t>& norm,
                     size_t& used) {
    check(len > 0, "zstd: truncated FSE header");
    BitFwd b;
    b.init(src, len);
    b.pull();
    int nb = static_cast<int>(b.acc & 0xF) + 5;
    b.consume(4);
    check(nb <= 9, "zstd: FSE accuracy too large");
    accuracy = nb;
    int remaining = (1 << nb) + 1;
    int threshold = 1 << nb;
    ++nb;
    norm.assign(max_symbol + 1, 0);
    int sym = 0;
    bool prev0 = false;
    while (remaining > 1 && sym <= max_symbol) {
        while (b.bits < 24 && b.p < b.end) b.pull();
        check(b.bits > 0 || b.p < b.end || remaining == 1, "zstd: truncated FSE counts");
        if (prev0) {
            int n0 = sym;
            while ((b.acc & 0xFFFF) == 0xFFFF) {
                n0 += 24;
                if (b.p + 2 <= b.end) {
                    // Drop 16 bits and refill, matching the reference's 2-byte skip
                    // only when a full reload is safe. Fall back to a plain shift.
                    b.consume(16);
                } else {
                    b.consume(16);
                }
                while (b.bits < 16 && b.p < b.end) b.pull();
            }
            while ((b.acc & 3) == 3) {
                n0 += 3;
                b.consume(2);
            }
            n0 += static_cast<int>(b.acc & 3);
            b.consume(2);
            check(n0 <= max_symbol + 1, "zstd: FSE zero-run past alphabet");
            while (sym < n0) norm[sym++] = 0;
            while (b.bits < 24 && b.p < b.end) b.pull();
        }
        int maxv = (2 * threshold - 1) - remaining;
        int count;
        if (static_cast<int>(b.acc & static_cast<uint32_t>(threshold - 1)) < maxv) {
            count = static_cast<int>(b.acc & static_cast<uint32_t>(threshold - 1));
            b.consume(nb - 1);
        } else {
            count = static_cast<int>(b.acc & static_cast<uint32_t>(2 * threshold - 1));
            if (count >= threshold) count -= maxv;
            b.consume(nb);
        }
        --count;
        int spend = count < 0 ? -count : count;
        check(spend < remaining, "zstd: FSE count exceeds remaining");
        remaining -= spend;
        check(sym <= max_symbol, "zstd: FSE symbol overflow");
        norm[sym++] = static_cast<int16_t>(count);
        prev0 = (count == 0);
        while (remaining < threshold) {
            --nb;
            threshold >>= 1;
        }
    }
    check(remaining == 1, "zstd: FSE counts are corrupt");
    // Trim to the last listed symbol. Zeros already fill the tail.
    while (sym > 0 && norm[sym - 1] == 0) --sym;
    norm.resize(std::max(sym, 1));
    used = b.bytes_used();
    if (b.bits < 0) die("zstd: FSE count reader consumed bits it did not have");
    if (std::getenv("TF_ZSTD_DEBUG")) {
        std::fprintf(stderr, "ncount acc=%d nsym=%zu used=%zu bits_left=%d:", accuracy, norm.size(), used, b.bits);
        for (size_t i = 0; i < norm.size(); ++i)
            if (norm[i]) std::fprintf(stderr, " %zu:%d", i, norm[i]);
        std::fprintf(stderr, "\n");
    }
}

constexpr int16_t kLlDef[36] = {4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};
constexpr int16_t kMlDef[53] = {1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1};
constexpr int16_t kOfDef[29] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1};

constexpr uint32_t kLlBase[36] = {0,  1,  2,  3,  4,  5,  6,  7,  8,   9,   10,   11,   12,   13,   14,    15,   16,  18,
                                  20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
constexpr uint8_t kLlExtra[36] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2,
                                  3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
constexpr uint32_t kMlBase[53] = {3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,  16,  17,  18,  19, 20,
                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,  34,  35,  37,  39, 41,
                                  43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539};
constexpr uint8_t kMlExtra[53] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

struct Huf {
    int max_bits = 0;
    std::vector<uint8_t> sym;
    std::vector<uint8_t> nbits;
    bool ok = false;
};

void huf_from_weights(Huf& h, const uint8_t* w, int nw) {
    uint32_t sum = 0;
    for (int i = 0; i < nw; ++i)
        if (w[i]) sum += 1u << (w[i] - 1);
    check(sum > 0, "zstd: empty huffman tree");
    // tableLog = highbit(sum) + 1, so a sum that is already a power of two
    // still grows to the next power (the implied symbol takes the rest).
    int table_log = hibit(sum) + 1;
    uint32_t rest_pow = 1u << table_log;
    uint32_t rest = rest_pow - sum;
    if (!(rest && (rest & (rest - 1)) == 0)) {
        std::fprintf(stderr, "huff weights n=%d sum=%u rest=%u:", nw, sum, rest);
        for (int i = 0; i < nw; ++i) std::fprintf(stderr, " %u", w[i]);
        std::fprintf(stderr, "\n");
        die("zstd: huffman weight remainder is not a power of two");
    }
    int last_w = hibit(rest) + 1;
    // Table log is log2 of the power of two the weights complete to.
    int max_bits = hibit(rest_pow);
    check(max_bits >= 1 && max_bits <= 11, "zstd: huffman depth out of range");
    int nsym = nw + 1;
    std::vector<int> nb(nsym);
    for (int i = 0; i < nw; ++i) nb[i] = w[i] ? max_bits + 1 - w[i] : 0;
    nb[nw] = max_bits + 1 - last_w;
    int rank[12] = {};
    for (int i = 0; i < nsym; ++i)
        if (nb[i]) rank[nb[i]]++;
    int next_code[12] = {};
    int code = 0;
    for (int bits = max_bits; bits >= 1; --bits) {
        next_code[bits] = code;
        code = (code + rank[bits]) >> 1;
    }
    h.max_bits = max_bits;
    int tsize = 1 << max_bits;
    h.sym.assign(tsize, 0);
    h.nbits.assign(tsize, 0);
    for (int s = 0; s < nsym; ++s) {
        if (!nb[s]) continue;
        int c = next_code[nb[s]]++;
        int shift = max_bits - nb[s];
        int base = c << shift;
        int stride = 1 << shift;
        if (base < 0 || stride < 0 || base + stride > tsize) {
            std::fprintf(stderr, "huff overflow sym=%d nb=%d c=%d shift=%d base=%d stride=%d tsize=%d max_bits=%d last_w=%d nw=%d\n",
                         s, nb[s], c, shift, base, stride, tsize, max_bits, last_w, nw);
            for (int i = 0; i < nw; ++i)
                if (w[i]) std::fprintf(stderr, "  w[%d]=%u\n", i, w[i]);
            std::fprintf(stderr, "  implied w=%d\n", last_w);
            die("zstd: huffman code does not fit the table");
        }
        for (int k = 0; k < stride; ++k) {
            h.sym[base + k] = static_cast<uint8_t>(s);
            h.nbits[base + k] = static_cast<uint8_t>(nb[s]);
        }
    }
    h.ok = true;
}

// Decode a 2-state FSE stream of huffman weights. Returns the weight bytes
// NOT including the implied final weight.
std::vector<uint8_t> fse_decode_weights(const uint8_t* src, size_t len) {
    int accuracy = 0;
    std::vector<int16_t> norm;
    size_t used = 0;
    // Weight alphabet is small (0..12). Allow up to 255.
    fse_read_ncount(src, len, 255, accuracy, norm, used);
    check(accuracy <= 6, "zstd: huffman FSE accuracy > 6");
    check(used < len, "zstd: huffman FSE header ate the bitstream");
    std::vector<FseEntry> table;
    fse_build(table, accuracy, norm.data(), static_cast<int>(norm.size()));
    BitBack bits;
    bits.init(src + used, len - used);
    auto init_state = [&]() {
        if (accuracy == 0) return 0;
        return static_cast<int>(bits.read(accuracy));
    };
    int s1 = init_state();
    int s2 = init_state();
    std::vector<uint8_t> w;
    auto emit_update = [&](int& st) -> uint8_t {
        const FseEntry& e = table[st];
        uint8_t sym = e.symbol;
        uint32_t add = e.nbits ? bits.read(e.nbits) : 0;
        st = e.base + static_cast<int>(add);
        return sym;
    };
    // FSE_decompress tail: once a read passes the start of the backward stream,
    // flush one more symbol from the other state. Those symbols were fixed by
    // bits already consumed.
    while (w.size() < 255) {
        w.push_back(emit_update(s1));
        if (bits.pos < 0) {
            if (w.size() < 255) w.push_back(emit_update(s2));
            break;
        }
        if (w.size() >= 255) break;
        w.push_back(emit_update(s2));
        if (bits.pos < 0) {
            if (w.size() < 255) w.push_back(emit_update(s1));
            break;
        }
    }
    check(!w.empty(), "zstd: no huffman weights");
    return w;
}

void huf_read(const uint8_t* src, size_t len, Huf& h, size_t& used) {
    check(len > 0, "zstd: missing huffman header");
    uint8_t header = src[0];
    if (header >= 128) {
        int n = header - 127;
        int nbytes = (n + 1) / 2;
        check(1 + nbytes <= static_cast<int>(len), "zstd: truncated direct huffman weights");
        std::vector<uint8_t> w(n);
        for (int i = 0; i < n; ++i) {
            uint8_t b = src[1 + i / 2];
            w[i] = (i & 1) ? (b & 15) : (b >> 4);
        }
        huf_from_weights(h, w.data(), n);
        used = static_cast<size_t>(1 + nbytes);
        return;
    }
    check(header > 0 && 1u + header <= len, "zstd: truncated FSE huffman header");
    if (std::getenv("TF_ZSTD_DEBUG")) {
        std::fprintf(stderr, "huff fse header len=%u first:", header);
        for (size_t i = 0; i < header && i < 24; ++i) std::fprintf(stderr, " %02x", src[1 + i]);
        std::fprintf(stderr, "\n");
    }
    auto w = fse_decode_weights(src + 1, header);
    huf_from_weights(h, w.data(), static_cast<int>(w.size()));
    used = 1u + header;
}

void huf_decode(const uint8_t* src, size_t len, uint8_t* dst, size_t n, const Huf& h) {
    if (n == 0) return;
    check(len > 0 && h.ok, "zstd: bad huffman stream");
    BitBack bits;
    bits.init(src, len);
    for (size_t i = 0; i < n; ++i) {
        int64_t before = bits.pos;
        uint32_t peeked = bits.peek(h.max_bits);
        uint8_t nb = h.nbits[peeked];
        check(nb > 0, "zstd: invalid huffman code");
        dst[i] = h.sym[peeked];
        bits.skip(nb);
        if (std::getenv("TF_ZSTD_DEBUG") && before < static_cast<int64_t>(nb)) {
            std::fprintf(stderr, "huff overread i=%zu/%zu before=%lld nb=%u sym=%02x\n", i, n, (long long)before, nb,
                         dst[i]);
        }
    }
}

struct SeqTabs {
    std::vector<FseEntry> ll, of, ml;
    int ll_a = 6, of_a = 5, ml_a = 6;
    bool have = false;
};

void load_defaults(SeqTabs& t) {
    fse_build(t.ll, 6, kLlDef, 36);
    fse_build(t.of, 5, kOfDef, 29);
    fse_build(t.ml, 6, kMlDef, 53);
    t.ll_a = 6;
    t.of_a = 5;
    t.ml_a = 6;
    t.have = true;
}

size_t read_mode_table(const uint8_t* src, size_t len, int mode, int which, SeqTabs& cur, const SeqTabs& prev) {
    auto set_default = [&]() {
        if (which == 0) {
            fse_build(cur.ll, 6, kLlDef, 36);
            cur.ll_a = 6;
        } else if (which == 1) {
            fse_build(cur.of, 5, kOfDef, 29);
            cur.of_a = 5;
        } else {
            fse_build(cur.ml, 6, kMlDef, 53);
            cur.ml_a = 6;
        }
    };
    if (mode == 0) {
        set_default();
        return 0;
    }
    if (mode == 3) {
        check(prev.have, "zstd: repeat mode with no previous table");
        if (which == 0) {
            cur.ll = prev.ll;
            cur.ll_a = prev.ll_a;
        } else if (which == 1) {
            cur.of = prev.of;
            cur.of_a = prev.of_a;
        } else {
            cur.ml = prev.ml;
            cur.ml_a = prev.ml_a;
        }
        return 0;
    }
    if (mode == 1) {
        check(len >= 1, "zstd: truncated RLE table");
        FseEntry e{src[0], 0, 0};
        if (which == 0) {
            cur.ll.assign(1, e);
            cur.ll_a = 0;
        } else if (which == 1) {
            cur.of.assign(1, e);
            cur.of_a = 0;
        } else {
            cur.ml.assign(1, e);
            cur.ml_a = 0;
        }
        return 1;
    }
    int max_sym = which == 0 ? 35 : which == 1 ? 31 : 52;
    int accuracy = 0;
    std::vector<int16_t> norm;
    size_t used = 0;
    fse_read_ncount(src, len, max_sym, accuracy, norm, used);
    int cap = which == 1 ? 8 : 9;
    check(accuracy <= cap, "zstd: sequence accuracy too large");
    std::vector<FseEntry> table;
    fse_build(table, accuracy, norm.data(), static_cast<int>(norm.size()));
    if (which == 0) {
        cur.ll = std::move(table);
        cur.ll_a = accuracy;
    } else if (which == 1) {
        cur.of = std::move(table);
        cur.of_a = accuracy;
    } else {
        cur.ml = std::move(table);
        cur.ml_a = accuracy;
    }
    return used;
}

void copy_match(std::vector<uint8_t>& out, size_t offset, size_t len) {
    check(offset >= 1 && offset <= out.size(), "zstd: bad match offset");
    size_t at = out.size();
    out.resize(at + len);
    for (size_t i = 0; i < len; ++i) out[at + i] = out[at + i - offset];
}

void decode_sequences(std::vector<uint8_t>& out, const std::vector<uint8_t>& lits, const uint8_t* src, size_t len,
                      SeqTabs& tabs, uint32_t rep[3], size_t window) {
    check(len >= 1, "zstd: missing sequences section");
    const uint8_t* p = src;
    const uint8_t* end = src + len;
    uint8_t b0 = *p++;
    if (b0 == 0) {
        out.insert(out.end(), lits.begin(), lits.end());
        return;
    }
    uint32_t nseq;
    if (b0 < 128) nseq = b0;
    else if (b0 < 255) {
        check(p < end, "zstd: truncated sequence count");
        nseq = (static_cast<uint32_t>(b0 - 128) << 8) + *p++;
    } else {
        check(p + 1 < end, "zstd: truncated sequence count");
        nseq = p[0] + (static_cast<uint32_t>(p[1]) << 8) + 0x7F00u;
        p += 2;
    }
    check(p < end, "zstd: truncated sequence modes");
    uint8_t modes = *p++;
    check((modes & 3) == 0, "zstd: reserved sequence mode bits");
    SeqTabs prev = tabs;
    size_t remain = static_cast<size_t>(end - p);
    size_t u = 0;
    u += read_mode_table(p + u, remain - u, (modes >> 6) & 3, 0, tabs, prev);
    u += read_mode_table(p + u, remain - u, (modes >> 4) & 3, 1, tabs, prev);
    u += read_mode_table(p + u, remain - u, (modes >> 2) & 3, 2, tabs, prev);
    tabs.have = true;
    check(u < remain, "zstd: sequence bitstream missing");
    BitBack bits;
    bits.init(p + u, remain - u);
    auto istate = [&](int acc) -> int {
        if (acc == 0) return 0;
        return static_cast<int>(bits.read(acc));
    };
    int st_ll = istate(tabs.ll_a);
    int st_of = istate(tabs.of_a);
    int st_ml = istate(tabs.ml_a);
    size_t lp = 0;
    for (uint32_t s = 0; s < nseq; ++s) {
        int ll_code = tabs.ll[st_ll].symbol;
        int of_code = tabs.of[st_of].symbol;
        int ml_code = tabs.ml[st_ml].symbol;
        check(ll_code < 36 && of_code < 31 && ml_code < 53, "zstd: sequence code out of range");
        uint32_t of_extra = of_code ? bits.read(of_code) : 0;
        uint32_t ml_extra = kMlExtra[ml_code] ? bits.read(kMlExtra[ml_code]) : 0;
        uint32_t ll_extra = kLlExtra[ll_code] ? bits.read(kLlExtra[ll_code]) : 0;
        uint32_t lit = kLlBase[ll_code] + ll_extra;
        uint32_t match = kMlBase[ml_code] + ml_extra;
        uint32_t oval = (1u << of_code) + of_extra;
        g_last_seq = static_cast<int>(s);
        g_last_ll = static_cast<int>(lit);
        g_last_ml = static_cast<int>(match);
        g_last_of = static_cast<int>(oval);
        check(lp + lit <= lits.size(), "zstd: literal overrun");
        if (lit) {
            size_t before_lit = out.size();
            out.insert(out.end(), lits.data() + lp, lits.data() + lp + lit);
            lp += lit;
            if (g_ref_on) check_out(out, before_lit);
        }
        uint32_t offset;
        if (oval > 3) {
            offset = oval - 3;
            rep[2] = rep[1];
            rep[1] = rep[0];
            rep[0] = offset;
        } else if (lit == 0) {
            if (oval == 3) {
                check(rep[0] > 1, "zstd: repeat offset underflow");
                offset = rep[0] - 1;
                rep[2] = rep[1];
                rep[1] = rep[0];
                rep[0] = offset;
            } else {
                int idx = static_cast<int>(oval);  // 1 -> rep[1], 2 -> rep[2]
                offset = rep[idx];
                if (idx == 1) std::swap(rep[0], rep[1]);
                else {
                    uint32_t tmp = rep[2];
                    rep[2] = rep[1];
                    rep[1] = rep[0];
                    rep[0] = tmp;
                }
            }
        } else {
            int idx = static_cast<int>(oval) - 1;
            offset = rep[idx];
            if (idx == 1) std::swap(rep[0], rep[1]);
            else if (idx == 2) {
                uint32_t tmp = rep[2];
                rep[2] = rep[1];
                rep[1] = rep[0];
                rep[0] = tmp;
            }
        }
        check(offset <= window, "zstd: offset exceeds window");
        size_t before_match = out.size();
        copy_match(out, offset, match);
        if (g_ref_on) check_out(out, before_match > 0 ? before_match - std::min<size_t>(before_match, 1) : 0);
        if (s + 1 != nseq) {
            auto upd = [&](std::vector<FseEntry>& table, int& st) {
                FseEntry e = table[st];
                uint32_t add = e.nbits ? bits.read(e.nbits) : 0;
                st = e.base + static_cast<int>(add);
            };
            upd(tabs.ll, st_ll);
            upd(tabs.ml, st_ml);
            upd(tabs.of, st_of);
        }
    }
    if (lp < lits.size()) out.insert(out.end(), lits.begin() + static_cast<std::ptrdiff_t>(lp), lits.end());
}

static void load_ref() {
    if (g_ref_on) return;
    const char* p = std::getenv("TF_ZSTD_REF");
    if (!p) return;
    FILE* f = std::fopen(p, "rb");
    if (!f) return;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::rewind(f);
    g_ref.resize(static_cast<size_t>(n));
    if (n > 0 && std::fread(g_ref.data(), 1, static_cast<size_t>(n), f) != static_cast<size_t>(n)) die("zstd ref read");
    std::fclose(f);
    g_ref_on = true;
    std::fprintf(stderr, "zstd ref %ld bytes\n", n);
}
static void check_out(const std::vector<uint8_t>& out, size_t from) {
    if (!g_ref_on) return;
    for (size_t i = from; i < out.size(); ++i) {
        if (i >= g_ref.size() || out[i] != g_ref[i]) {
            std::fprintf(stderr, "first mismatch at %zu mine %02x ref %02x outsize %zu seq=%d ll=%d ml=%d of=%d\n", i,
                         out[i], i < g_ref.size() ? g_ref[i] : 0, out.size(), g_last_seq, g_last_ll, g_last_ml, g_last_of);
            size_t a = i > 16 ? i - 16 : 0;
            std::fprintf(stderr, " mine:");
            for (size_t k = a; k < out.size() && k < i + 8; ++k) std::fprintf(stderr, " %02x", out[k]);
            std::fprintf(stderr, "\n ref: ");
            for (size_t k = a; k < g_ref.size() && k < i + 8; ++k) std::fprintf(stderr, " %02x", g_ref[k]);
            std::fprintf(stderr, "\n");
            die("zstd mismatch");
        }
    }
}

}  // namespace

void zstd_decompress(const uint8_t* src, size_t src_len, std::vector<uint8_t>& out, size_t expected_size) {
    load_ref();
    out.clear();
    check(src_len >= 4, "zstd: input too small");
    if (expected_size) out.reserve(expected_size);
    size_t pos = 0;
    while (pos + 4 <= src_len) {
        uint32_t magic = rd_u32(src + pos);
        if (magic >= 0x184D2A50u && magic <= 0x184D2A5Fu) {
            check(pos + 8 <= src_len, "zstd: truncated skippable frame");
            uint32_t sz = rd_u32(src + pos + 4);
            check(pos + 8 + sz <= src_len, "zstd: truncated skippable payload");
            pos += 8 + sz;
            continue;
        }
        check(magic == kMagic, "zstd: bad frame magic");
        pos += 4;
        check(pos < src_len, "zstd: truncated frame header");
        uint8_t desc = src[pos++];
        int fcs_flag = desc >> 6;
        int single = (desc >> 5) & 1;
        int checksum = (desc >> 2) & 1;
        int dict_flag = desc & 3;
        check(((desc >> 3) & 1) == 0, "zstd: reserved frame bit is set");
        size_t window = 0;
        if (!single) {
            check(pos < src_len, "zstd: truncated window descriptor");
            uint8_t wd = src[pos++];
            int exp = wd >> 3;
            int mant = wd & 7;
            int wlog = 10 + exp;
            size_t base = 1ull << wlog;
            window = base + (base / 8) * static_cast<size_t>(mant);
        }
        if (dict_flag) {
            int nb = dict_flag == 3 ? 4 : dict_flag;
            check(pos + static_cast<size_t>(nb) <= src_len, "zstd: truncated dictionary id");
            uint32_t did = 0;
            for (int i = 0; i < nb; ++i) did |= static_cast<uint32_t>(src[pos + i]) << (8 * i);
            pos += nb;
            check(did == 0, "zstd: dictionaries are not supported");
        }
        uint64_t content = 0;
        bool have_size = false;
        if (fcs_flag == 0 && single) {
            check(pos < src_len, "zstd: truncated content size");
            content = src[pos++];
            have_size = true;
        } else if (fcs_flag == 1) {
            check(pos + 2 <= src_len, "zstd: truncated content size");
            content = static_cast<uint64_t>(rd_u16(src + pos)) + 256ull;
            pos += 2;
            have_size = true;
        } else if (fcs_flag == 2) {
            check(pos + 4 <= src_len, "zstd: truncated content size");
            content = rd_u32(src + pos);
            pos += 4;
            have_size = true;
        } else if (fcs_flag == 3) {
            check(pos + 8 <= src_len, "zstd: truncated content size");
            content = rd_u64(src + pos);
            pos += 8;
            have_size = true;
        }
        if (single) {
            check(have_size, "zstd: single-segment frame without a content size");
            window = static_cast<size_t>(content ? content : 1);
        }
        check(window > 0, "zstd: zero window");
        size_t frame_begin = out.size();
        Huf huf;
        SeqTabs tabs;
        load_defaults(tabs);
        uint32_t rep[3] = {1, 4, 8};
        bool last = false;
        int block_index = 0;
        while (!last) {
            size_t block_out_begin = out.size();
            check(pos + 3 <= src_len, "zstd: truncated block header");
            uint32_t bh = src[pos] | (uint32_t(src[pos + 1]) << 8) | (uint32_t(src[pos + 2]) << 16);
            pos += 3;
            last = bh & 1;
            int btype = (bh >> 1) & 3;
            uint32_t bsize = bh >> 3;
            size_t block_max = std::min(window, static_cast<size_t>(128 * 1024));
            check(btype != 3, "zstd: reserved block");
            if (btype == 0) {
                check(bsize <= block_max && pos + bsize <= src_len, "zstd: bad raw block");
                out.insert(out.end(), src + pos, src + pos + bsize);
                pos += bsize;
            } else if (btype == 1) {
                check(bsize <= block_max && pos < src_len, "zstd: bad RLE block");
                out.insert(out.end(), bsize, src[pos]);
                pos += 1;
            } else {
                check(bsize <= block_max && pos + bsize <= src_len, "zstd: bad compressed block");
                const uint8_t* bp = src + pos;
                const uint8_t* be = bp + bsize;
                check(bp < be, "zstd: empty compressed block");
                uint8_t lb0 = bp[0];
                int lit_type = lb0 & 3;
                int size_fmt = (lb0 >> 2) & 3;
                uint32_t regen = 0, comp = 0;
                int four = 0;
                const uint8_t* ldata = nullptr;
                if (lit_type <= 1) {
                    if ((lb0 & 4) == 0) {
                        regen = lb0 >> 3;
                        ldata = bp + 1;
                    } else if (size_fmt == 1) {
                        check(bp + 2 <= be, "zstd: truncated literals header");
                        regen = (lb0 >> 4) + (uint32_t(bp[1]) << 4);
                        ldata = bp + 2;
                    } else {
                        check(bp + 3 <= be, "zstd: truncated literals header");
                        regen = (lb0 >> 4) + (uint32_t(bp[1]) << 4) + (uint32_t(bp[2]) << 12);
                        ldata = bp + 3;
                    }
                    comp = lit_type == 0 ? regen : 1;
                } else {
                    if (size_fmt == 0 || size_fmt == 1) {
                        check(bp + 3 <= be, "zstd: truncated literals header");
                        uint32_t hw = bp[0] | (bp[1] << 8) | (bp[2] << 16);
                        regen = (hw >> 4) & 0x3FF;
                        comp = (hw >> 14) & 0x3FF;
                        ldata = bp + 3;
                        four = size_fmt == 1;
                    } else if (size_fmt == 2) {
                        check(bp + 4 <= be, "zstd: truncated literals header");
                        uint32_t hw = bp[0] | (bp[1] << 8) | (bp[2] << 16) | (uint32_t(bp[3]) << 24);
                        regen = (hw >> 4) & 0x3FFF;
                        comp = (hw >> 18) & 0x3FFF;
                        ldata = bp + 4;
                        four = 1;
                    } else {
                        check(bp + 5 <= be, "zstd: truncated literals header");
                        uint64_t hw = uint64_t(bp[0]) | (uint64_t(bp[1]) << 8) | (uint64_t(bp[2]) << 16) |
                                      (uint64_t(bp[3]) << 24) | (uint64_t(bp[4]) << 32);
                        regen = static_cast<uint32_t>((hw >> 4) & 0x3FFFF);
                        comp = static_cast<uint32_t>((hw >> 22) & 0x3FFFF);
                        ldata = bp + 5;
                        four = 1;
                    }
                }
                check(ldata && ldata + comp <= be, "zstd: literals exceed the block");
                std::vector<uint8_t> lits(regen);
                if (lit_type == 0) {
                    if (regen) std::memcpy(lits.data(), ldata, regen);
                } else if (lit_type == 1) {
                    if (regen) std::memset(lits.data(), ldata[0], regen);
                } else {
                    const uint8_t* stream = ldata;
                    size_t slen = comp;
                    if (lit_type == 2) {
                        size_t hu = 0;
                        huf_read(ldata, comp, huf, hu);
                        stream += hu;
                        slen -= hu;
                    } else {
                        check(huf.ok, "zstd: treeless literals without a previous tree");
                    }
                    if (!four) {
                        huf_decode(stream, slen, lits.data(), regen, huf);
                    } else {
                        check(slen >= 6, "zstd: truncated huffman jump table");
                        uint32_t s1 = rd_u16(stream), s2 = rd_u16(stream + 2), s3 = rd_u16(stream + 4);
                        check(6ull + s1 + s2 + s3 <= slen, "zstd: huffman streams exceed the block");
                        uint32_t s4 = static_cast<uint32_t>(slen - 6 - s1 - s2 - s3);
                        uint32_t sz[4] = {s1, s2, s3, s4};
                        size_t each = (static_cast<size_t>(regen) + 3) / 4;
                        const uint8_t* sp = stream + 6;
                        size_t produced = 0;
                        for (int i = 0; i < 4; ++i) {
                            size_t want = (i == 3) ? regen - produced : each;
                            huf_decode(sp, sz[i], lits.data() + produced, want, huf);
                            sp += sz[i];
                            produced += want;
                        }
                    }
                }
                decode_sequences(out, lits, ldata + comp, static_cast<size_t>(be - (ldata + comp)), tabs, rep, window);
                pos += bsize;
                if (std::getenv("TF_ZSTD_DEBUG")) {
                    uint32_t h = 2166136261u;
                    for (size_t i = block_out_begin; i < out.size(); ++i) {
                        h ^= out[i];
                        h *= 16777619u;
                    }
                    std::fprintf(stderr, "blk %d out=%zu..%zu hash=%08x lit=%d four=%d\n", block_index, block_out_begin, out.size(), h,
                                 lit_type, four);
                }
                ++block_index;
            }
        }
        if (have_size) check(out.size() - frame_begin == content, "zstd: content size mismatch");
        if (checksum) {
            check(pos + 4 <= src_len, "zstd: truncated checksum");
            pos += 4;
        }
        if (expected_size && out.size() >= expected_size) break;
        if (!expected_size) break;
    }
    if (expected_size) check(out.size() == expected_size, "zstd: decompressed size mismatch");
}

}  // namespace tf
