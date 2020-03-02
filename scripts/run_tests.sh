#!/bin/bash
mkdir -p build
cd build 
cmake ..
make unit_tests
cd ..
./build/test/unit_tests
