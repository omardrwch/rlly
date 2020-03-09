#!/bin/bash
cd test
mkdir -p build
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Debug
make unit_tests
cd ..
cd ..
./test/build/unit_tests
