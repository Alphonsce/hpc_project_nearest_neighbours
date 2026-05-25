CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -Wno-unused-parameter
LDFLAGS  ?= -fopenmp
NVCC     ?= nvcc
NVCCFLAGS?= -std=c++17 -O3 --extended-lambda -arch=sm_89

all: lsh hnsw lsh_cuda hnsw_cuda

lsh: lsh.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

hnsw: hnsw.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

lsh_cuda: lsh_cuda.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcublas

hnsw_cuda: hnsw_cuda.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp $< -o $@ -lgomp

debug: CXXFLAGS = -std=c++17 -O0 -g -fopenmp -Wall -Wextra
debug: lsh hnsw

clean:
	rm -f lsh hnsw lsh_cuda hnsw_cuda

.PHONY: all clean debug
