CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -Wno-unused-parameter
LDFLAGS  ?= -fopenmp

all: lsh

lsh: lsh.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

debug: CXXFLAGS = -std=c++17 -O0 -g -fopenmp -Wall -Wextra
debug: lsh

clean:
	rm -f lsh

.PHONY: all clean debug
