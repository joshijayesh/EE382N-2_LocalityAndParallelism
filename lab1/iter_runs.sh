#!/bin/bash

NUM_RUNS=5
TARGET=('basicmatmul' 'cacheaware')
CACHEAWARE='cacheaware'
N=(32 512 4096)  # Assuming square matrices

B1=32     # 104, round down to 64
B2=256    # 591, round down to 512
B3=1024   # 3344, round down to 2048

BASICPERF='cycles:u,instructions:u,cache-references:u,cache-misses:u'
L1PERF='L1-dcache-load:u,L1-dcache-load-misses:u,L1-dcache-stores:u,L1-dcache-store-misses:u'
LLCPERF='LLC-loads:u,LLC-load-misses:u,LLC-stores:u,LLC-store-misses:u'

for target in "${TARGET[@]}"; do
    for side in "${N[@]}"; do
        b1_tgt=$((B1 < side ? B1 : side))
        b2_tgt=$((B2 < side ? B2 : side))
        b3_tgt=$((B3 < side ? B3 : side))

        if [[ "$target" == "$CACHEAWARE" ]]; then
            b1_arg="$b1_tgt"
            b2_arg="$b2_tgt"
            b3_arg="$b3_tgt"
        else
            b1_arg=""
            b2_arg=""
            b3_arg=""
        fi

        TARGET_DIR="results/${target}_$side-$side-$side/"
        mkdir -p $TARGET_DIR
        for ((run = 1; run <= $NUM_RUNS; run++))
        do
            echo "perf stat -o $TARGET_DIR/basic_perf_run${run}.out -e $BASICPERF ./$target $side $side $side $b1_arg $b2_arg $b3_arg" 
            echo "perf stat -o $TARGET_DIR/l1_perf_run${run}.out -e $L1PERF ./$target $side $side $side $b1_arg $b2_arg $b3_arg"
            echo "perf stat -o $TARGET_DIR/llc_perf_run${run}.out -e $LLCPERF ./$target $side $side $side $b1_arg $b2_arg $b3_arg" 
        done
    done
done

