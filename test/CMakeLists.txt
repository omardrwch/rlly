cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} --coverage")
SET(CMAKE_EXE_LINKER_FLAGS_COVERAGE "${CMAKE_EXE_LINKER_FLAGS_DEBUG} --coverage")
SET(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} --coverage")

project(RLLY_TEST)


include_directories(../rlly/include/env
                    ../rlly/include/spaces
                    ../rlly/include/utils
                    ../rlly/include/wrappers)


#
# build library
# 

# Adding all .cpp files as source
file(GLOB SOURCES "../rlly/src/*.cpp")  # if there are no subfolders
file(GLOB SOURCES "../rlly/src/*/*.cpp")

# Generate shared library from the code
add_library(rlly SHARED ${SOURCES})

#
# test targets
#
add_executable(unit_tests main.cpp
                          utils_test.cpp
                          env_test.cpp
                          wrapper_test.cpp)
target_link_libraries(unit_tests rlly)


# - add_test() is not necessary when running ./unit_tests
# - add_test() is needed to use the command "make test". In this case, include enable_testing()
#   in source_dir/CMakeLists.txt

# # adding tests
# add_test(NAME tests COMMAND unit_tests)
