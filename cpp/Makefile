CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -Wno-unused-parameter
LDFLAGS  ?= -fopenmp
NVCC     ?= nvcc
NVCCFLAGS?= -std=c++17 -O3 --extended-lambda -arch=sm_89

all: lsh lsh_cuda

lsh: lsh.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

lsh_cuda: lsh_cuda.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcublas

debug: CXXFLAGS = -std=c++17 -O0 -g -fopenmp -Wall -Wextra
debug: lsh

clean:
	rm -f lsh lsh_cuda

.PHONY: all clean debug
