include_directories(${RLLY_SOURCE_DIR})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(debug debug.cpp)

add_executable(mdp_example mdp_example.cpp)

add_executable(gridworld_example gridworld_example.cpp)

