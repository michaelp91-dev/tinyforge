#include "tf_model.hpp"

namespace tf {
Engine* create_engine_cuda(const GPTConfig&, const int*) { return nullptr; }
}
