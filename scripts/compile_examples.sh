#!/bin/bash
bash scripts/compile.sh
bash generate_header/run.sh -rendering
cd examples
mkdir -p build
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
cd ..
cd ..