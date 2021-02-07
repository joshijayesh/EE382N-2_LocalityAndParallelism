#!/bin/bash

BASICPERF='cycles:u,instructions:u,cache-references:u,cache-misses:u'
L1PERF='L1-dcache-loads:u,L1-dcache-load-misses:u,L1-dcache-stores:u,L1-dcache-store-misses:u'
LLCPERF='LLC-loads:u,LLC-load-misses:u,LLC-stores:u,LLC-store-misses:u'

perf stat -e $BASICPERF \
    ./basicmatmul 512 512 512

perf stat -e $L1PERF \
    ./basicmatmul 512 512 512

perf stat -e $LLCPERF \
    ./basicmatmul 512 512 512
