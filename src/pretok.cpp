#include "tf_pretok.hpp"
#include "tf_unicode_tables.hpp"

namespace tf {
namespace {

bool in_ranges(uint32_t cp, const uint32_t ranges[][2], int n) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (cp < ranges[mid][0]) hi = mid - 1;
        else if (cp > ranges[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

bool is_L(uint32_t cp) { return in_ranges(cp, UNI_L, UNI_L_N); }
bool is_N(uint32_t cp) { return in_ranges(cp, UNI_N, UNI_N_N); }
bool is_WS(uint32_t cp) { return in_ranges(cp, UNI_WS, UNI_WS_N); }

// Returns the byte length of one code point, or 1 for an invalid leading byte.
// `ok` is false when the byte is not well-formed UTF-8 (the caller treats it
// as a raw byte that is neither a letter, a number, nor whitespace).
int decode(const uint8_t* s, size_t n, size_t i, uint32_t& cp, bool& ok) {
    ok = false;
    if (i >= n) return 0;
    uint8_t c = s[i];
    if (c < 0x80) {
        cp = c;
        ok = true;
        return 1;
    }
    auto cont = [&](size_t k) { return (s[i + k] & 0xC0) == 0x80; };
    if ((c & 0xE0) == 0xC0 && i + 1 < n && cont(1)) {
        cp = (uint32_t(c & 0x1F) << 6) | (s[i + 1] & 0x3F);
        ok = cp >= 0x80;
        return ok ? 2 : 1;
    }
    if ((c & 0xF0) == 0xE0 && i + 2 < n && cont(1) && cont(2)) {
        cp = (uint32_t(c & 0x0F) << 12) | (uint32_t(s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
        ok = cp >= 0x800 && !(cp >= 0xD800 && cp <= 0xDFFF);
        return ok ? 3 : 1;
    }
    if ((c & 0xF8) == 0xF0 && i + 3 < n && cont(1) && cont(2) && cont(3)) {
        cp = (uint32_t(c & 0x07) << 18) | (uint32_t(s[i + 1] & 0x3F) << 12) | (uint32_t(s[i + 2] & 0x3F) << 6) |
             (s[i + 3] & 0x3F);
        ok = cp >= 0x10000 && cp <= 0x10FFFF;
        return ok ? 4 : 1;
    }
    cp = c;
    return 1;
}

char lower_ascii(uint8_t c) {
    if (c >= 'A' && c <= 'Z') return static_cast<char>(c - 'A' + 'a');
    return static_cast<char>(c);
}

size_t match_contraction(const uint8_t* s, size_t n, size_t i) {
    if (i >= n || s[i] != '\'') return 0;
    if (i + 1 < n) {
        char c = lower_ascii(s[i + 1]);
        if (c == 's' || c == 'd' || c == 'm' || c == 't') return 2;
    }
    if (i + 2 < n) {
        char a = lower_ascii(s[i + 1]), b = lower_ascii(s[i + 2]);
        if ((a == 'l' && b == 'l') || (a == 'v' && b == 'e') || (a == 'r' && b == 'e')) return 3;
    }
    return 0;
}

size_t match_letters(const uint8_t* s, size_t n, size_t i) {
    uint32_t cp;
    bool ok;
    int k = decode(s, n, i, cp, ok);
    if (!k) return 0;
    bool prefix = !(cp == '\r' || cp == '\n' || (ok && (is_L(cp) || is_N(cp))));
    size_t pos = prefix ? i + static_cast<size_t>(k) : i;
    size_t start = pos;
    while (pos < n) {
        int k2 = decode(s, n, pos, cp, ok);
        if (!ok || !is_L(cp)) break;
        pos += static_cast<size_t>(k2);
    }
    if (pos == start) return 0;
    return pos - i;
}

size_t match_numbers(const uint8_t* s, size_t n, size_t i) {
    size_t pos = i;
    int count = 0;
    while (count < 2 && pos < n) {
        uint32_t cp;
        bool ok;
        int k = decode(s, n, pos, cp, ok);
        if (!ok || !is_N(cp)) break;
        pos += static_cast<size_t>(k);
        ++count;
    }
    return count ? pos - i : 0;
}

size_t match_punct(const uint8_t* s, size_t n, size_t i) {
    auto from = [&](size_t pos) -> size_t {
        size_t p = pos;
        int count = 0;
        while (p < n) {
            uint32_t cp;
            bool ok;
            int k = decode(s, n, p, cp, ok);
            if (!ok) {
                ++p;
                ++count;
                continue;
            }
            if (is_WS(cp) || is_L(cp) || is_N(cp)) break;
            p += static_cast<size_t>(k);
            ++count;
        }
        if (!count) return pos;
        while (p < n && (s[p] == '\r' || s[p] == '\n')) ++p;
        return p;
    };
    if (i < n && s[i] == ' ') {
        size_t e = from(i + 1);
        if (e > i + 1) return e - i;
    }
    size_t e = from(i);
    return e > i ? e - i : 0;
}

size_t match_newline(const uint8_t* s, size_t n, size_t i) {
    size_t pos = i;
    size_t end = i;
    while (pos < n) {
        uint32_t cp;
        bool ok;
        int k = decode(s, n, pos, cp, ok);
        if (!ok || !is_WS(cp)) break;
        pos += static_cast<size_t>(k);
        if (cp == '\n' || cp == '\r') end = pos;
    }
    return end > i ? end - i : 0;
}

size_t match_ws_run(const uint8_t* s, size_t n, size_t i, size_t& last_cp) {
    size_t pos = i;
    last_cp = i;
    while (pos < n) {
        uint32_t cp;
        bool ok;
        int k = decode(s, n, pos, cp, ok);
        if (!ok || !is_WS(cp)) break;
        last_cp = pos;
        pos += static_cast<size_t>(k);
    }
    return pos;
}

size_t match_trailing_ws(const uint8_t* s, size_t n, size_t i) {
    size_t last = i;
    size_t end = match_ws_run(s, n, i, last);
    if (end == i) return 0;
    if (end == n) return end - i;
    // Followed by a non-space: the lookahead only holds after giving back one
    // whitespace code point, so a one-character run fails.
    if (last == i) return 0;
    return last - i;
}

size_t match_ws(const uint8_t* s, size_t n, size_t i) {
    size_t last = i;
    size_t end = match_ws_run(s, n, i, last);
    return end > i ? end - i : 0;
}

}  // namespace

std::vector<std::pair<size_t, size_t>> pretok_split(const uint8_t* s, size_t n) {
    std::vector<std::pair<size_t, size_t>> spans;
    size_t i = 0;
    while (i < n) {
        size_t m = match_contraction(s, n, i);
        if (!m) m = match_letters(s, n, i);
        if (!m) m = match_numbers(s, n, i);
        if (!m) m = match_punct(s, n, i);
        if (!m) m = match_newline(s, n, i);
        if (!m) m = match_trailing_ws(s, n, i);
        if (!m) m = match_ws(s, n, i);
        if (!m) m = 1;  // unreachable for well-classified bytes; keeps the cursor moving
        spans.emplace_back(i, i + m);
        i += m;
    }
    return spans;
}

}  // namespace tf
