set(DEFAULT_BUILD_TYPE Debug)  # Debug or Release. Used for libraries

# Adding sources manually
# set(SOURCES src/mainapp.cpp src/student.cpp)

# Addint all .cpp files as source
file(GLOB SOURCES "src/*.cpp")  # if there are no subfolders

# Generate shared library from the code
add_library(rlly_rendering SHARED ${SOURCES})
target_include_directories(rlly_rendering PUBLIC "${RLLY_SOURCE_DIR}/rlly/include/env"
                                                 "${RLLY_SOURCE_DIR}/rlly/include/utils" 
                                                 "${RLLY_SOURCE_DIR}/rlly_rendering/include")
