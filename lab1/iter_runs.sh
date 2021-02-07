#!/bin/bash

NUM_RUNS=5
TARGET=('basicmatmul')
N=(32 512 4096)  # Assuming square matrices

BASICPERF='cycles:u,instructions:u,cache-references:u,cache-misses:u'
L1PERF='L1-dcache-load:u,L1-dcache-load-misses:u,L1-dcache-stores:u,L1-dcache-store-misses:u'
LLCPERF='LLC-loads:u,LLC-load-misses:u,LLC-stores:u,LLC-store-misses:u'

for target in "${TARGET[@]}"; do
    for side in "${N[@]}"; do
        TARGET_DIR="results/${target}_$side-$side-$side/"
        mkdir -p $TARGET_DIR
        for ((run = 1; run <= $NUM_RUNS; run++))
        do
            echo "perf stat -o $TARGET_DIR/basic_perf_run${run}.out -e $BASICPERF ./$target $side $side $side" 
            echo "perf stat -o $TARGET_DIR/l1_perf_run${run}.out -e $L1PERF ./$target $side $side $side"
            echo "perf stat -o $TARGET_DIR/llc_perf_run${run}.out -e $LLCPERF ./$target $side $side $side" 
        done
    done
done

