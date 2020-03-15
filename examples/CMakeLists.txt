cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED) 


# GLUT and OpenGL
find_package(GLUT REQUIRED)
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

#
add_executable(debug debug.cpp)
target_link_libraries(debug ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
target_include_directories(debug PUBLIC "..")

#
add_executable(gridworld_example gridworld_example.cpp)
target_link_libraries(gridworld_example ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
target_include_directories(gridworld_example PUBLIC "..")


#
add_executable(render_example render_example.cpp)
target_link_libraries(render_example ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
target_include_directories(render_example PUBLIC "..")

#
add_executable(mountaincar_example mountaincar_example.cpp)
target_link_libraries(mountaincar_example ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
target_include_directories(mountaincar_example PUBLIC "..")


#
add_executable(cartpole_example cartpole_example.cpp)
target_link_libraries(cartpole_example ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
target_include_directories(cartpole_example PUBLIC "..")

#
add_executable(squareworld_example squareworld_example.cpp)
target_link_libraries(squareworld_example ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
target_include_directories(squareworld_example PUBLIC "..")
                                                
        
#
add_executable(wrapper_example wrapper_example.cpp)
target_link_libraries(wrapper_example ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
target_include_directories(wrapper_example PUBLIC  "..")

#
add_executable(discretize_state_example discretize_state_example.cpp)
target_link_libraries(discretize_state_example ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
target_include_directories(discretize_state_example PUBLIC  "..")
