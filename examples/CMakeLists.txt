include_directories(${RLLY_SOURCE_DIR}
                    ${RLLY_SOURCE_DIR}/rlly_rendering/include)
include_directories(${RLLY_SOURCE_DIR})


set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED) 


#
# Boost packages
#
set(Boost_USE_STATIC_LIBS        ON)  # only find static libs
set(Boost_USE_DEBUG_LIBS         OFF) # ignore debug libs and 
set(Boost_USE_RELEASE_LIBS       ON)  # only find release libs 
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     OFF) 
find_package(Boost COMPONENTS filesystem date_time)


# GLUT and opengl
find_package(GLUT REQUIRED)
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)


#
add_executable(debug debug.cpp)

#
add_executable(gridworld_example gridworld_example.cpp)

#
add_executable(render_example render_example.cpp)
target_link_libraries(render_example rlly_rendering ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})

