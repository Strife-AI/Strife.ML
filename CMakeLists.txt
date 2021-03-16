cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

include (ExternalProject)

project(ML VERSION 1 LANGUAGES C CXX)

ExternalProject_Add(tcc
        PREFIX tinycc
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/tinycc
        CONFIGURE_COMMAND "./configure"
        BUILD_IN_SOURCE 1
        BUILD_COMMAND ${MAKE}
        INSTALL_COMMAND "")

add_subdirectory(Strife.Common)
add_subdirectory(src)