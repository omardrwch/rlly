#!/bin/bash
cd test
mkdir -p build
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Coverage
make unit_tests
lcov --zerocounters  --directory .
cd ../..
./test/build/unit_tests
lcov --directory test/build --capture --output-file test/build/unit_tests.info
lcov --remove test/build/unit_tests.info "/usr/include/*" \
                                         "$(pwd)/test/*"  \
                                         -o test/build/unit_tests_filtered.info
genhtml test/build/unit_tests_filtered.info -o test/build/out


