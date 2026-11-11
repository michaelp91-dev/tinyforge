#include "tf_bpe.hpp"
#include "tf_pretok.hpp"
#include "tf_util.hpp"

#include <cstdio>
#include <string>
#include <vector>

static int fails = 0;
static void expect(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL %s\n", msg);
        ++fails;
    }
}

static std::vector<std::string> pieces(const std::string& s) {
    auto sp = tf::pretok_split(s);
    std::vector<std::string> out;
    for (auto [a, b] : sp) out.emplace_back(s.substr(a, b - a));
    std::string joined;
    for (auto& p : out) joined += p;
    expect(joined == s, "pretok does not cover the input");
    return out;
}

static void expect_pieces(const std::string& s, const std::vector<std::string>& want, const char* name) {
    auto got = pieces(s);
    if (got != want) {
        std::fprintf(stderr, "FAIL pretok %s\n  got:", name);
        for (auto& g : got) std::fprintf(stderr, " [%s]", g.c_str());
        std::fprintf(stderr, "\n");
        ++fails;
    }
}

int main() {
    expect_pieces("Hello world", {"Hello", " world"}, "hello");
    expect_pieces("It's a test", {"It", "'s", " a", " test"}, "its");
    expect_pieces("Hello, world!", {"Hello", ",", " world", "!"}, "comma");
    expect_pieces("12345 67", {"12", "34", "5", " ", "67"}, "nums");
    expect_pieces("end   ", {"end", "   "}, "trail");
    expect_pieces("a\n\nb", {"a", "\n\n", "b"}, "newlines");
    expect_pieces("foo\tbar", {"foo", "\tbar"}, "tab");
    expect_pieces("price $12.50", {"price", " $", "12", ".", "50"}, "price");
    expect_pieces("don't stop", {"don", "'t", " stop"}, "dont");
    expect_pieces("  leading", {" ", " leading"}, "lead2");
    expect_pieces("   leading", {"  ", " leading"}, "lead3");
    expect_pieces("a  b", {"a", " ", " b"}, "dbl");
    expect_pieces("a \n b", {"a", " \n", " b"}, "nlmix");
    expect_pieces("...", {"..."}, "dots");
    expect_pieces("(hello)", {"(hello", ")"}, "paren");
    expect_pieces("C++", {"C", "++"}, "cpp");
    expect_pieces("não", {"não"}, "accent");
    expect_pieces("a\r\nb", {"a", "\r\n", "b"}, "crlf");
    expect_pieces("  \n  x", {"  \n", " ", " x"}, "nlws");
    expect_pieces("...\n", {"...\n"}, "dotsnl");
    expect_pieces("I'M", {"I", "'M"}, "IM");
    expect_pieces("'re", {"'re"}, "re");
    expect_pieces("rock'n'roll", {"rock", "'n", "'roll"}, "rock");
    expect_pieces("The capital of France is", {"The", " capital", " of", " France", " is"}, "cap");
    expect_pieces("", {}, "empty");
    expect_pieces(" ", {" "}, "sp");

    std::string corpus;
    for (int i = 0; i < 50; ++i)
        corpus += "The capital of France is Paris. Hello world, it's a test of byte pair encoding.\n";
    corpus += "don't stop 12345...\n";
    tf::Tokenizer tok;
    tok.train(corpus, 400, 2);
    expect(tok.vocab_size <= 400 && tok.vocab_size >= 256 + tf::Tokenizer::kNumSpecial, "vocab size");
    std::string sample = "The capital of France is Paris. <|bos|> hello";
    auto ids = tok.encode(sample, true);
    expect(!ids.empty() && ids[0] == tok.bos_id, "prepend bos");
    std::string back = tok.decode(ids);
    // prepended bos decodes to the spelling, then the original text.
    expect(back == std::string("<|bos|>") + sample, "roundtrip with bos and a special");
    auto again = tok.encode(sample, true);
    expect(again == ids, "encode is deterministic");
    // A string with no specials round-trips without a forced bos.
    std::string plain = "Hello, world! 123";
    auto pids = tok.encode(plain, false);
    expect(tok.decode(pids) == plain, "plain roundtrip");
    std::string path = "/tmp/tinyforge_tok.bin";
    tok.save(path);
    tf::Tokenizer tok2;
    tok2.load(path);
    expect(tok2.encode(plain, false) == pids, "save/load encode");
    expect(tok2.bos_id == tok.bos_id, "bos id");
    bool saw_special = false;
    for (int id : tok.encode("x<|bos|>y", false))
        if (id == tok.bos_id) saw_special = true;
    expect(saw_special, "special token is atomic");

    if (fails) {
        std::fprintf(stderr, "%d failure(s)\n", fails);
        return 1;
    }
    std::printf("text tests passed\n");
    return 0;
}
