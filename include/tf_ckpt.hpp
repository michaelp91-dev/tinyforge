#pragma once

#include "tf_model.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace tf {

struct Checkpoint {
    GPTConfig cfg;
    int step = 0;
    uint64_t doc_cursor = 0;
    std::vector<float> params;
};

void save_checkpoint(const std::string& path, const Checkpoint& ckpt);
Checkpoint load_checkpoint(const std::string& path);

}  // namespace tf
