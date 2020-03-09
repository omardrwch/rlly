#!/bin/bash
cd test
mkdir -p build
cd build 
cmake ..
make unit_tests
cd ..
cd ..
./test/build/unit_tests
