cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

include (ExternalProject)

project(ML VERSION 1 LANGUAGES C CXX)

if (MSVC)
    ExternalProject_Add(tcc
            PREFIX tinycc
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tinycc/win32
            CONFIGURE_COMMAND ""
            BUILD_IN_SOURCE 1
            BUILD_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tinycc/win32/build-tcc.bat -c cl
            INSTALL_COMMAND "")
else()
        ExternalProject_Add(tcc
            PREFIX tinycc
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tinycc
            CONFIGURE_COMMAND "./configure"
            BUILD_IN_SOURCE 1
            BUILD_COMMAND ${MAKE}
            INSTALL_COMMAND "")
endif()


add_subdirectory(Strife.Common)
add_subdirectory(src)