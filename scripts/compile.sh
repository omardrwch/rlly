#!/bin/bash
TARGET=$1
mkdir -p build
cd build 
cmake ..
make $TARGET
cd ..
