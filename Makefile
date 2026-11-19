# TinyForge: stock llm.c trainer plus the modern decoder.
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fPIC
NVCC ?= nvcc
# PTX for sm_80 so an A100 runs it natively and an L4 (sm_89) can JIT.
NVCC_ARCH ?= -gencode arch=compute_80,code=compute_80 -gencode arch=compute_80,code=sm_80
NVCC_FLAGS ?= -std=c++17 -O3 --use_fast_math $(NVCC_ARCH) -Ithird_party/llm.c -Isrc
NVCC_LIBS ?= -lcublas -lcublasLt -lnvToolsExt

# make GPU=t4: Turing (Colab/Kaggle T4) has no bf16, so build sm_75 in fp32.
ifeq ($(GPU),t4)
NVCC_ARCH := -gencode arch=compute_75,code=sm_75 -DENABLE_FP32
NVCC_LIBS := -lcublas -lcublasLt
endif

# make USE_NCCL=1 [NCCL_HOME=/path]: multi-GPU data parallel (--nproc/--rank).
# NCCL_HOME can be the pip wheel dir (python -c "import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__))").
ifeq ($(USE_NCCL),1)
ifneq ($(NCCL_HOME),)
NVCC_DEFS += -DTF_NCCL -I$(NCCL_HOME)/include
NVCC_LIBS += -L$(NCCL_HOME)/lib -Xlinker -rpath=$(NCCL_HOME)/lib -l:libnccl.so.2
else
NVCC_DEFS += -DTF_NCCL
NVCC_LIBS += -lnccl
endif
endif

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
	$(NVCC) $(NVCC_FLAGS) $(NVCC_DEFS) -o $@ src/train_tinyforge.cu $(NVCC_LIBS)

stock: third_party/llm.c/train_gpt2cu

third_party/llm.c/train_gpt2cu: third_party/llm.c/train_gpt2.cu
	$(MAKE) -C third_party/llm.c train_gpt2cu GPU_COMPUTE_CAPABILITY=80

clean:
	rm -f libtf_ref.so train_tinyforge
	$(MAKE) -C third_party/llm.c clean
