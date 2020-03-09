set(DEFAULT_BUILD_TYPE Debug)  # Debug or Release. Used for libraries

# Adding sources manually
# set(SOURCES src/mainapp.cpp src/student.cpp)

# GLUT and OpenGL
find_package(GLUT REQUIRED)
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

# Addint all .cpp files as source
file(GLOB SOURCES "src/*.cpp")  # if there are no subfolders

# Generate shared library from the code
add_library(rlly_rendering SHARED ${SOURCES})
target_link_libraries(rlly_rendering ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
target_include_directories(rlly_rendering PUBLIC "${RLLY_SOURCE_DIR}/rlly/include/env"
                                                 "${RLLY_SOURCE_DIR}/rlly/include/utils" 
                                                 "${RLLY_SOURCE_DIR}/rlly_rendering/include")
