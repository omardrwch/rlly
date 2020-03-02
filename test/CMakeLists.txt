link_directories(${RLLY_SOURCE_DIR}/rlly/src/env
                 ${RLLY_SOURCE_DIR}/rlly/src/spaces
                 ${RLLY_SOURCE_DIR}/rlly/src/utils)

include_directories(${RLLY_SOURCE_DIR}/rlly/include/env
                    ${RLLY_SOURCE_DIR}/rlly/include/spaces
                    ${RLLY_SOURCE_DIR}/rlly/include/utils)
# testing files
add_executable(unit_tests main.cpp
                          space_test.cpp
                          random_test.cpp
                          vector_op_test.cpp
                          chain_test.cpp)
target_link_libraries(unit_tests rlly)


# - add_test() is not necessary when running ./unit_tests
# - add_test() is needed to use the command "make test". In this case, include enable_testing()
#   in source_dir/CMakeLists.txt

# # adding tests
# add_test(NAME tests COMMAND unit_tests)
