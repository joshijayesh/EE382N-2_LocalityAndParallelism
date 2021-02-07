#!/bin/bash

if test "$#" -ne 4; then
    echo "Usage: {BUILD} {N} {M} {P}. I.e.: ./iter_runs.sh basicmatmul 512 512 512"
    exit 1
fi

NUM_RUNS=20
TARGET=$1
N=$2
M=$3
P=$4

TARGET_DIR="results/${TARGET}_$N-$M-$P/"
mkdir -p $TARGET_DIR

BASICPERF='cycles:u,instructions:u,cache-references:u,cache-misses:u'
L1PERF='L1-dcache-load:u,L1-dcache-load-misses:u,L1-dcache-stores:u,L1-dcache-store-misses:u'
LLCPERF='LLC-loads:u,LLC-load-misses:u,LLC-stores:u,LLC-store-misses:u'

for ((run = 1; run <= $NUM_RUNS; run++))
do
    perf stat -o $TARGET_DIR/basic_perf_run${run}.out -e $BASICPERF ./$TARGET $N $M $P 
    perf stat -o $TARGET_DIR/l1_perf_run${run}.out -e $L1PERF ./$TARGET $N $M $P 
    perf stat -o $TARGET_DIR/llc_perf_run${run}.out -e $LLCPERF ./$TARGET $N $M $P 
done

