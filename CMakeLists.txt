cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

include (ExternalProject)
include(FetchContent)

project(ML VERSION 1 LANGUAGES C CXX)

if (MSVC)
    ExternalProject_Add(tcc
            PREFIX tinycc
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tinycc/win32
            CONFIGURE_COMMAND ""
            BUILD_IN_SOURCE 1
            BUILD_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tinycc/win32/build-tcc.bat -c cl
            INSTALL_COMMAND COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/tcc
                && ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tinycc/win32/include $<TARGET_FILE_DIR:Strife.ML>/tcc/include
                && ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/src/TorchApi.h $<TARGET_FILE_DIR:Strife.ML>/tcc/include
                && ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:Strife.ML>/tcc/lib
                && ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/tinycc/win32/lib/libtcc1-64.a $<TARGET_FILE_DIR:Strife.ML>/tcc/lib
                && ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/tinycc/win32/lib/libtcc1-32.a $<TARGET_FILE_DIR:Strife.ML>/tcc/lib)
else()
    ExternalProject_Add(tcc
        PREFIX tinycc
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tinycc
        CONFIGURE_COMMAND "./configure"
        BUILD_IN_SOURCE 1
        BUILD_COMMAND ${MAKE}
        INSTALL_COMMAND mkdir -p ${CMAKE_BINARY_DIR}/tcc/lib
            && cp ${CMAKE_CURRENT_SOURCE_DIR}/tinycc/libtcc1.a ${CMAKE_BINARY_DIR}/tcc/lib/libtcc1.a
            && cp -r ${CMAKE_CURRENT_SOURCE_DIR}/tinycc/include ${CMAKE_BINARY_DIR}/tcc)
endif()

FetchContent_Declare(
        torch-test
        PREFIX "torch-test"
        URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
)

FetchContent_MakeAvailable(torch-test)
set(CMAKE_PREFIX_PATH "${torch-test_SOURCE_DIR}/share/cmake/Torch/")
find_package(Torch)
add_subdirectory(Strife.Common)
add_subdirectory(src)