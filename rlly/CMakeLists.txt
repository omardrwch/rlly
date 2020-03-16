set(DEFAULT_BUILD_TYPE Debug)  # Debug or Release. Used for libraries

# Adding all .cpp files as source
file(GLOB SOURCES "src/*.cpp" "src/*/*.cpp") 

# Include files
file(GLOB_RECURSE INCLUDE_FILES "include/*/*.h")

# Useful defs for install
include(GNUInstallDirs)

# Generate shared library from the code
add_library(rlly SHARED ${SOURCES})

target_include_directories(rlly PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/env>
                                       $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/spaces>
                                       $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/utils>
                                       $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/wrappers>
                                       $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

set_target_properties(rlly  PROPERTIES 
                            VERSION ${PROJECT_VERSION}
                            PUBLIC_HEADER "${INCLUDE_FILES}")

# Installation parameters
install(TARGETS rlly EXPORT   rlly-config
                     ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}/rlly
                     LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}/rlly
                     RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}/rlly
                     PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rlly)

install(EXPORT rlly-config 
        FILE   rlly-config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rlly)
