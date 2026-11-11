CXX ?= g++
NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wno-unused-function -Iinclude -pthread
NVCCFLAGS ?= -O3 -std=c++17 -Iinclude -arch=sm_80 --compiler-options -Wno-unused-function

CPU_SRC = src/zstd.cpp src/parquet.cpp src/pretok.cpp src/bpe.cpp src/ops_cpu.cpp src/model_cpu.cpp \
          src/data.cpp src/ckpt.cpp src/cuda_stub.cpp
CPU_OBJ = $(CPU_SRC:src/%.cpp=build/%.o)

.PHONY: all cpu cuda test clean

all: cpu

cpu: build/tf_train build/tf_sample build/tf_tok_train build/tf_tok_encode \
     build/test_grad build/test_text build/test_parquet build/test_zstd

cuda: build/tf_train_cuda build/tf_sample_cuda build/test_cuda_parity

build:
	mkdir -p build

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) -c -o $@ $<

build/tf_train: src/train.cpp $(CPU_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ src/train.cpp $(CPU_OBJ)

build/tf_sample: src/sample.cpp $(CPU_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ src/sample.cpp $(CPU_OBJ)

build/tf_tok_train: src/tok_train.cpp $(CPU_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ src/tok_train.cpp $(CPU_OBJ)

build/tf_tok_encode: src/tok_encode.cpp $(CPU_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ src/tok_encode.cpp $(CPU_OBJ)

build/test_grad: tests/test_grad.cpp src/ops_cpu.cpp src/model_cpu.cpp
	$(CXX) $(CXXFLAGS) -o $@ tests/test_grad.cpp src/ops_cpu.cpp src/model_cpu.cpp

build/test_text: tests/test_text.cpp src/pretok.cpp src/bpe.cpp
	$(CXX) $(CXXFLAGS) -o $@ tests/test_text.cpp src/pretok.cpp src/bpe.cpp

build/test_parquet: tests/test_parquet.cpp src/parquet.cpp src/zstd.cpp
	$(CXX) $(CXXFLAGS) -o $@ tests/test_parquet.cpp src/parquet.cpp src/zstd.cpp

build/test_zstd: tests/test_zstd.cpp src/zstd.cpp
	$(CXX) $(CXXFLAGS) -o $@ tests/test_zstd.cpp src/zstd.cpp

build/cuda_model.o: src/model_cuda.cu | build
	$(NVCC) $(NVCCFLAGS) -c -o $@ src/model_cuda.cu

CUDA_CPU_OBJ = $(filter-out build/cuda_stub.o,$(CPU_OBJ))

build/tf_train_cuda: src/train.cpp $(CUDA_CPU_OBJ) build/cuda_model.o
	$(NVCC) $(NVCCFLAGS) -o $@ src/train.cpp $(CUDA_CPU_OBJ) build/cuda_model.o

build/tf_sample_cuda: src/sample.cpp $(CUDA_CPU_OBJ) build/cuda_model.o
	$(NVCC) $(NVCCFLAGS) -o $@ src/sample.cpp $(CUDA_CPU_OBJ) build/cuda_model.o

build/test_cuda_parity: tests/test_cuda_parity.cu build/cuda_model.o src/ops_cpu.cpp src/model_cpu.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ tests/test_cuda_parity.cu build/cuda_model.o src/ops_cpu.cpp src/model_cpu.cpp

test: cpu
	./build/test_grad
	./build/test_text
	@echo CPU tests passed

clean:
	rm -rf build
