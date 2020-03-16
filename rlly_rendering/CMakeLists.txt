set(DEFAULT_BUILD_TYPE Debug)  # Debug or Release. Used for libraries

# GLUT and OpenGL
find_package(GLUT)
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL)

if(GLUT_FOUND AND OpenGL_FOUND)
    message(" ")
    message("GLUT and OpenGL found: rlly_rendering can be installed.")
    message(" ")

    # Include files
    file(GLOB_RECURSE INCLUDE_FILES "include/*.h")

    # Addint all .cpp files as source
    file(GLOB SOURCES "src/*.cpp")  # if there are no subfolders

    # Useful defs for install
    include(GNUInstallDirs)

    # Generate shared library from the code
    add_library(rlly_rendering SHARED ${SOURCES})
    target_link_libraries(rlly_rendering ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})
    target_include_directories(rlly_rendering PUBLIC $<BUILD_INTERFACE:${RLLY_SOURCE_DIR}/rlly/include/env>
                                                    $<BUILD_INTERFACE:${RLLY_SOURCE_DIR}/rlly/include/utils>
                                                    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                                    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)


    set_target_properties(rlly_rendering PROPERTIES 
                                        VERSION ${PROJECT_VERSION}
                                        PUBLIC_HEADER "${INCLUDE_FILES}")

    # Installation parameters
    install(TARGETS rlly_rendering EXPORT rlly_rendering-config
                        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}/rlly
                        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}/rlly
                        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}/rlly
                        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rlly)

    install(EXPORT rlly_rendering-config 
            FILE   rlly_rendering-config.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rlly_rendering)

else()
    message(" ")
    message("GLUT and/or OpenGL not found: rlly_rendering cannot be installed.")
    message(" ")
endif()




