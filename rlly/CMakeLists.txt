set(DEFAULT_BUILD_TYPE Debug)  # Debug or Release. Used for libraries

# Include headers
include_directories("include"
                    "include/env"
                    "include/spaces"
                    "include/utils"
                    "include/wrappers")

# Adding sources manually
# set(SOURCES src/mainapp.cpp src/student.cpp)

# Adding all .cpp files as source
file(GLOB SOURCES "src/*.cpp")  # if there are no subfolders
file(GLOB SOURCES "src/*/*.cpp")

# Generate shared library from the code
add_library(rlly SHARED ${SOURCES})
