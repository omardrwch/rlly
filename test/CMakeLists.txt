cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(RLLY_TEST)

link_directories(../rlly/src/env
                 ../rlly/src/spaces
                 ../rlly/src/utils)

include_directories(../rlly/include/env
                    ../rlly/include/spaces
                    ../rlly/include/utils)


# build library
# Addint all .cpp files as source
file(GLOB SOURCES "../rlly/src/*.cpp")  # if there are no subfolders
file(GLOB SOURCES "../rlly/src/*/*.cpp")

# Generate shared library from the code
add_library(rlly SHARED ${SOURCES})

# testing files
add_executable(unit_tests main.cpp
                          space_test.cpp
                          random_test.cpp
                          vector_op_test.cpp
                          env_test.cpp)
target_link_libraries(unit_tests rlly)


# - add_test() is not necessary when running ./unit_tests
# - add_test() is needed to use the command "make test". In this case, include enable_testing()
#   in source_dir/CMakeLists.txt

# # adding tests
# add_test(NAME tests COMMAND unit_tests)
