set(DEFAULT_BUILD_TYPE Debug)  # Debug or Release. Used for libraries

# Include headers
include_directories("include")

# Adding sources manually
# set(SOURCES src/mainapp.cpp src/student.cpp)

# Addint all .cpp files as source
file(GLOB SOURCES "src/*.cpp")  # if there are no subfolders

# Generate shared library from the code
add_library(rlly_rendering SHARED ${SOURCES})
