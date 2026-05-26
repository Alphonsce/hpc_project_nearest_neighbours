CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-array-bounds
LDFLAGS  ?= -fopenmp
NVCC     ?= nvcc
# Auto-detect GPU arch; fall back to sm_75 (T4/V100) if nvcc unavailable
CUDA_ARCH?= $(shell $(NVCC) --run "int main(){}" 2>/dev/null && \
              nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | \
              head -1 | tr -d '.' | awk '{print "sm_"$$1}' || echo sm_75)
NVCCFLAGS?= -std=c++17 -O3 --extended-lambda -arch=$(CUDA_ARCH)

all: lsh hnsw lsh_cuda hnsw_cuda kd_tree kd_tree_cuda

lsh: lsh.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

hnsw: hnsw.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

lsh_cuda: lsh_cuda.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcublas

hnsw_cuda: hnsw_cuda.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp $< -o $@ -lgomp

debug: CXXFLAGS = -std=c++17 -O0 -g -fopenmp -Wall -Wextra
debug: lsh hnsw kd_tree

kd_tree: kd_tree.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

kd_tree_cuda: kd_tree_cuda.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f lsh hnsw lsh_cuda hnsw_cuda kd_tree kd_tree_cuda

.PHONY: all clean debug
