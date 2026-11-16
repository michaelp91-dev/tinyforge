# TinyForge: stock llm.c trainer plus the modern decoder.
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fPIC
NVCC ?= nvcc
# PTX for sm_80 so an A100 runs it natively and an L4 (sm_89) can JIT.
NVCC_ARCH ?= -gencode arch=compute_80,code=compute_80 -gencode arch=compute_80,code=sm_80
NVCC_FLAGS ?= -std=c++17 -O3 --use_fast_math $(NVCC_ARCH) -Ithird_party/llm.c -Isrc
NVCC_LIBS ?= -lcublas -lcublasLt -lnvToolsExt

.PHONY: all test cpu cuda stock clean

all: test cuda

test: libtf_ref.so
	python3 tests/test_layers.py

libtf_ref.so: src/tf_ref.c src/tf_ref.h
	$(CC) $(CFLAGS) -shared -o $@ src/tf_ref.c -lm

cpu: third_party/llm.c/train_gpt2

third_party/llm.c/train_gpt2: third_party/llm.c/train_gpt2.c
	$(MAKE) -C third_party/llm.c train_gpt2 CC=$(CC)

cuda: train_tinyforge

train_tinyforge: src/train_tinyforge.cu src/tf_kernels.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ src/train_tinyforge.cu $(NVCC_LIBS)

stock: third_party/llm.c/train_gpt2cu

third_party/llm.c/train_gpt2cu: third_party/llm.c/train_gpt2.cu
	$(MAKE) -C third_party/llm.c train_gpt2cu GPU_COMPUTE_CAPABILITY=80

clean:
	rm -f libtf_ref.so train_tinyforge
	$(MAKE) -C third_party/llm.c clean
