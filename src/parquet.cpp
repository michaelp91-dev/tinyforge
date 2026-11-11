// Parquet reader for ClimbMix shards: one optional UTF8 column named "text",
// PLAIN byte arrays, ZSTD or uncompressed data pages, Thrift compact footer.

#include "tf_parquet.hpp"
#include "tf_util.hpp"
#include "tf_zstd.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>

namespace tf {
namespace {

struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    int last_fid = 0;
    uint8_t u8() {
        check(p < end, "parquet: truncated thrift");
        return *p++;
    }
    uint32_t varint() {
        uint32_t n = 0;
        int shift = 0;
        for (;;) {
            uint8_t b = u8();
            n |= uint32_t(b & 0x7f) << shift;
            if ((b & 0x80) == 0) return n;
            shift += 7;
            check(shift <= 28, "parquet: varint too long");
        }
    }
    int32_t zigzag() {
        uint32_t n = varint();
        return static_cast<int32_t>((n >> 1) ^ -(n & 1));
    }
    // Returns false on STOP.
    bool field(int& fid, int& typ) {
        uint8_t b = u8();
        if (b == 0) return false;
        typ = b & 0x0f;
        int delta = b >> 4;
        fid = delta == 0 ? static_cast<int>(varint()) : last_fid + delta;
        last_fid = fid;
        return true;
    }
    void skip(int typ);
    void skip_list() {
        uint8_t h = u8();
        int et = h & 0x0f;
        uint32_t n = h >> 4;
        if (n == 15) n = varint();
        for (uint32_t i = 0; i < n; ++i) {
            if (et == 1 || et == 2) u8();
            else skip(et);
        }
    }
    std::string read_bin() {
        uint32_t n = varint();
        check(p + n <= end, "parquet: truncated binary");
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }
};

void Cursor::skip(int typ) {
    if (typ == 1 || typ == 2) return;
    if (typ == 3) {
        u8();
        return;
    }
    if (typ == 4 || typ == 5 || typ == 6) {
        zigzag();
        return;
    }
    if (typ == 7) {
        check(p + 8 <= end, "parquet: truncated double");
        p += 8;
        return;
    }
    if (typ == 8) {
        uint32_t n = varint();
        check(p + n <= end, "parquet: truncated binary");
        p += n;
        return;
    }
    if (typ == 9 || typ == 10) {
        skip_list();
        return;
    }
    if (typ == 11) {
        uint32_t n = varint();
        if (!n) return;
        uint8_t kv = u8();
        int kt = kv >> 4, vt = kv & 0x0f;
        for (uint32_t i = 0; i < n; ++i) {
            if (kt == 1 || kt == 2) u8();
            else skip(kt);
            if (vt == 1 || vt == 2) u8();
            else skip(vt);
        }
        return;
    }
    if (typ == 12) {
        int saved = last_fid;
        last_fid = 0;
        int fid, t;
        while (field(fid, t)) skip(t);
        last_fid = saved;
        return;
    }
    die("parquet: unknown thrift type");
}

std::vector<uint8_t> read_levels(const uint8_t* p, size_t n, int bit_width, int64_t nvalues) {
    std::vector<uint8_t> levels;
    levels.reserve(static_cast<size_t>(nvalues));
    const uint8_t* end = p + n;
    auto varint = [&]() {
        uint32_t v = 0;
        int shift = 0;
        while (p < end) {
            uint8_t b = *p++;
            v |= uint32_t(b & 0x7f) << shift;
            if ((b & 0x80) == 0) return v;
            shift += 7;
        }
        die("parquet: truncated level run");
    };
    int nbytes_val = (bit_width + 7) / 8;
    while (p < end && static_cast<int64_t>(levels.size()) < nvalues) {
        uint32_t header = varint();
        if ((header & 1) == 0) {
            uint32_t count = header >> 1;
            check(p + nbytes_val <= end, "parquet: truncated RLE value");
            uint32_t value = 0;
            for (int i = 0; i < nbytes_val; ++i) value |= uint32_t(p[i]) << (8 * i);
            p += nbytes_val;
            for (uint32_t i = 0; i < count; ++i) levels.push_back(static_cast<uint8_t>(value));
        } else {
            uint32_t groups = header >> 1;
            size_t nbytes = static_cast<size_t>(groups) * static_cast<size_t>(bit_width);
            check(p + nbytes <= end, "parquet: truncated bit-packed levels");
            // Little-endian bit packing, 8 values per group, bit_width bits each.
            for (uint32_t g = 0; g < groups; ++g) {
                // Unpack 8 values from `bit_width` bytes.
                uint32_t acc = 0;
                int acc_bits = 0;
                const uint8_t* gp = p + static_cast<size_t>(g) * bit_width;
                int bi = 0;
                for (int v = 0; v < 8; ++v) {
                    while (acc_bits < bit_width) {
                        check(bi < bit_width, "parquet: bit-pack underflow");
                        acc |= uint32_t(gp[bi++]) << acc_bits;
                        acc_bits += 8;
                    }
                    levels.push_back(static_cast<uint8_t>(acc & ((1u << bit_width) - 1u)));
                    acc >>= bit_width;
                    acc_bits -= bit_width;
                }
            }
            p += nbytes;
        }
    }
    check(static_cast<int64_t>(levels.size()) >= nvalues, "parquet: not enough definition levels");
    levels.resize(static_cast<size_t>(nvalues));
    return levels;
}

std::vector<std::string> decode_plain(const uint8_t* p, size_t n, const std::vector<uint8_t>& defs) {
    std::vector<std::string> out;
    out.reserve(defs.size());
    size_t i = 0;
    for (uint8_t d : defs) {
        if (d == 0) {
            out.emplace_back();
            continue;
        }
        check(i + 4 <= n, "parquet: truncated plain value");
        uint32_t len = rd_u32(p + i);
        i += 4;
        check(i + len <= n, "parquet: plain value exceeds page");
        out.emplace_back(reinterpret_cast<const char*>(p + i), len);
        i += len;
    }
    return out;
}

}  // namespace

ParquetFile::ParquetFile(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    check(fd_ >= 0, "parquet: cannot open " + path);
    struct stat st;
    check(::fstat(fd_, &st) == 0, "parquet: stat failed");
    size_ = static_cast<size_t>(st.st_size);
    check(size_ >= 12, "parquet: file too small");
    map_ = static_cast<const uint8_t*>(::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    check(map_ != MAP_FAILED, "parquet: mmap failed");
    check(std::memcmp(map_, "PAR1", 4) == 0, "parquet: bad magic");
    check(std::memcmp(map_ + size_ - 4, "PAR1", 4) == 0, "parquet: bad footer magic");
    uint32_t footer_len = rd_u32(map_ + size_ - 8);
    check(footer_len + 8 <= size_, "parquet: footer length out of range");
    Cursor c{map_ + size_ - 8 - footer_len, map_ + size_ - 8, 0};
    int fid, typ;
    // version
    check(c.field(fid, typ) && fid == 1 && typ == 5, "parquet: missing version");
    int version = c.zigzag();
    check(version == 1 || version == 2, "parquet: unsupported version");
    // schema
    check(c.field(fid, typ) && fid == 2 && typ == 9, "parquet: missing schema");
    uint8_t h = c.u8();
    check((h & 0x0f) == 12, "parquet: schema element type");
    uint32_t nse = h >> 4;
    if (nse == 15) nse = c.varint();
    check(nse == 2, "parquet: expected a single data column");
    for (uint32_t si = 0; si < nse; ++si) {
        int saved = c.last_fid;
        c.last_fid = 0;
        int f2, t2;
        int col_type = -1, rep = -1;
        std::string name;
        while (c.field(f2, t2)) {
            if (f2 == 1 && t2 == 5) col_type = c.zigzag();
            else if (f2 == 3 && t2 == 5) rep = c.zigzag();
            else if (f2 == 4 && t2 == 8) name = c.read_bin();
            else c.skip(t2);
        }
        c.last_fid = saved;
        if (si == 0) check(name == "schema", "parquet: unexpected root name");
        if (si == 1) {
            check(name == "text", "parquet: expected column 'text', got " + name);
            check(col_type == 6, "parquet: text column is not BYTE_ARRAY");
            check(rep == 1, "parquet: text column is not optional");
        }
    }
    check(c.field(fid, typ) && fid == 3 && typ == 6, "parquet: missing num_rows");
    num_rows_ = c.zigzag();
    check(c.field(fid, typ) && fid == 4 && typ == 9, "parquet: missing row groups");
    h = c.u8();
    check((h & 0x0f) == 12, "parquet: row group type");
    uint32_t ng = h >> 4;
    if (ng == 15) ng = c.varint();
    groups_.resize(ng);
    for (uint32_t g = 0; g < ng; ++g) {
        int saved = c.last_fid;
        c.last_fid = 0;
        int f2, t2;
        while (c.field(f2, t2)) {
            if (f2 == 1 && t2 == 9) {
                uint8_t ch = c.u8();
                check((ch & 0x0f) == 12, "parquet: column chunk type");
                uint32_t nc = ch >> 4;
                if (nc == 15) nc = c.varint();
                check(nc == 1, "parquet: expected one column");
                int cs = c.last_fid;
                c.last_fid = 0;
                int f3, t3;
                while (c.field(f3, t3)) {
                    if (f3 == 3 && t3 == 12) {
                        int ms = c.last_fid;
                        c.last_fid = 0;
                        int f4, t4;
                        while (c.field(f4, t4)) {
                            if (t4 == 5 || t4 == 6) {
                                int64_t v = c.zigzag();
                                if (f4 == 4) groups_[g].codec = static_cast<int>(v);
                                else if (f4 == 5) groups_[g].num_rows = v;
                                else if (f4 == 6) groups_[g].uncompressed = v;
                                else if (f4 == 7) groups_[g].compressed = v;
                                else if (f4 == 9) groups_[g].data_offset = v;
                            } else c.skip(t4);
                        }
                        c.last_fid = ms;
                    } else c.skip(t3);
                }
                c.last_fid = cs;
            } else if ((t2 == 5 || t2 == 6) && f2 == 3) {
                groups_[g].num_rows = c.zigzag();
            } else c.skip(t2);
        }
        c.last_fid = saved;
        check(groups_[g].codec == 0 || groups_[g].codec == 6, "parquet: only zstd or uncompressed pages are supported");
        check(groups_[g].data_offset > 0, "parquet: missing data page offset");
    }
}

ParquetFile::~ParquetFile() {
    if (map_ && map_ != MAP_FAILED) ::munmap(const_cast<uint8_t*>(map_), size_);
    if (fd_ >= 0) ::close(fd_);
}

std::vector<std::string> ParquetFile::read_row_group(int index) const {
    check(index >= 0 && index < static_cast<int>(groups_.size()), "parquet: row group out of range");
    const Group& g = groups_[index];
    check(static_cast<size_t>(g.data_offset) < size_, "parquet: page offset past EOF");
    Cursor c{map_ + g.data_offset, map_ + size_, 0};
    int fid, typ;
    int page_type = -1, uncomp = 0, comp = 0, nvalues = 0, encoding = -1;
    check(c.field(fid, typ) && fid == 1, "parquet: bad page header");
    page_type = c.zigzag();
    check(c.field(fid, typ) && fid == 2, "parquet: missing uncompressed size");
    uncomp = c.zigzag();
    check(c.field(fid, typ) && fid == 3, "parquet: missing compressed size");
    comp = c.zigzag();
    bool saw = false;
    while (c.field(fid, typ)) {
        if (fid == 5 && typ == 12) {
            saw = true;
            int saved = c.last_fid;
            c.last_fid = 0;
            int f2, t2;
            while (c.field(f2, t2)) {
                if (t2 == 5) {
                    int v = c.zigzag();
                    if (f2 == 1) nvalues = v;
                    else if (f2 == 2) encoding = v;
                } else c.skip(t2);
            }
            c.last_fid = saved;
        } else c.skip(typ);
    }
    check(page_type == 0, "parquet: only data page v1 is supported");
    check(saw && encoding == 0, "parquet: only PLAIN encoding is supported");
    check(nvalues == g.num_rows, "parquet: page value count does not match the row group");
    const uint8_t* payload = c.p;
    check(payload + comp <= map_ + size_, "parquet: compressed page exceeds file");
    std::vector<uint8_t> raw;
    if (g.codec == 6) {
        zstd_decompress(payload, static_cast<size_t>(comp), raw, static_cast<size_t>(uncomp));
    } else {
        check(comp == uncomp, "parquet: uncompressed size mismatch");
        raw.assign(payload, payload + comp);
    }
    check(raw.size() >= 4, "parquet: page too small for level length");
    uint32_t level_bytes = rd_u32(raw.data());
    if (std::getenv("TF_PQ_DEBUG")) {
        std::fprintf(stderr, "page uncomp %zu comp_field nvalues=%d level_bytes=%u raw0", raw.size(), nvalues, level_bytes);
        for (size_t i = 0; i < 24 && i < raw.size(); ++i) std::fprintf(stderr, " %02x", raw[i]);
        std::fprintf(stderr, "\n");
    }
    check(4u + level_bytes <= raw.size(), "parquet: level block exceeds page");
    auto defs = read_levels(raw.data() + 4, level_bytes, /*bit_width=*/1, nvalues);
    return decode_plain(raw.data() + 4 + level_bytes, raw.size() - 4 - level_bytes, defs);
}

}  // namespace tf
