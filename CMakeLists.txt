cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(RLLY)

# enable warnings
# (see https://cmake.org/cmake/help/latest/command/add_compile_options.html)
add_compile_options(-Wall)

# enable_testing()
add_subdirectory(rlly)
add_subdirectory(rlly_rendering)
add_subdirectory(examples)

# Useful link: https://stackoverflow.com/questions/8304190/cmake-with-include-and-source-paths-basic-setup
