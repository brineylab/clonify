# Makefile for C++ oracle (test reference implementation)

CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

.PHONY: all clean test

all: cluster

cluster: cluster.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test: cluster
	./cluster ../data/test_HIV_bnAb_heavies.tsv /tmp/oracle_test.tsv

clean:
	rm -f cluster /tmp/oracle_test.tsv
