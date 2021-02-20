#!/bin/bash

if [[ -z "${PIN_HOME}" ]]; then
    echo "PIN_HOME has not been set up!"
    exit 1
fi

if [[ $1 == "push" ]]; then
    cp $PIN_HOME/source/tools/SimpleExamples/dcache.H pin/
    cp $PIN_HOME/source/tools/SimpleExamples/dcache.cpp pin/
elif [[ $1 == "pull" ]]; then
    cp pin/* $PIN_HOME/source/tools/SimpleExamples/
    cd $PIN_HOME/source/tools/SimpleExamples/
    make
else
    echo "Unknown command: ${@:1}; Use push or pull"
    exit 1
fi

