cmake_minimum_required(VERSION 3.2.02 FATAL_ERROR)

include (ExternalProject)
include(FetchContent)

project(ML VERSION 1 LANGUAGES C CXX)

if (MSVC)
    message("${CMAKE_CURRENT_SOURCE_DIR}/tinycc/win32")
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
        INSTALL_COMMAND mkdir -p ${CMAKE_BINARY_DIR}/tcc/lib
            && cp ${CMAKE_CURRENT_SOURCE_DIR}/tinycc/libtcc1.a ${CMAKE_BINARY_DIR}/tcc/lib/libtcc1.a
            && cp -r ${CMAKE_CURRENT_SOURCE_DIR}/tinycc/include ${CMAKE_BINARY_DIR}/tcc)
endif()


if (MSVC)
	if (DEFINED CMAKE_BUILD_TYPE)
		message("============================ CMake_Build_Type is ${CMAKE_BUILD_TYPE}")
		set(BUILD_TYPE ${CMAKE_BUILD_TYPE})
	else()
		set(BUILD_TYPE ${CMAKE_CONFIGURATION_TYPES})
	endif()

	if ("${BUILD_TYPE}" STREQUAL "Debug")
        set(TorchCpuCache ${CMAKE_SOURCE_DIR}/.torch/cpu/debug)
        set(TorchCpuUrl https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-debug-1.8.1%2Bcu102.zip)

        set(TorchCudaCache ${CMAKE_SOURCE_DIR}/.torch/cuda/debug)
        set(TorchCudaUrl https://download.pytorch.org/libtorch/cu111/libtorch-win-shared-with-deps-debug-1.8.1%2Bcu111.zip)
	else()
        set(TorchCpuCache ${CMAKE_SOURCE_DIR}/.torch/cpu/release)
        set(TorchCpuUrl https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.8.1%2Bcpu.zip)

        set(TorchCudaCache ${CMAKE_SOURCE_DIR}/.torch/cuda/release)
        set(TorchCudaUrl https://download.pytorch.org/libtorch/cu111/libtorch-win-shared-with-deps-1.8.1%2Bcu111.zip)
	endif()
else()
    set(TorchCpuCache ${CMAKE_SOURCE_DIR}/.torch/cpu)
    set(TorchCpuUrl https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip)
endif()

set(FETCHCONTENT_BASE_DIR ${TorchCpuCache})

FetchContent_Declare(
        torch-test
        URL ${TorchCpuUrl}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND "")

FetchContent_MakeAvailable(torch-test)

set(TORCH_DIR ${TorchCpuCache} CACHE STRING "Torch directory")

list(APPEND CMAKE_PREFIX_PATH "${torch-test_SOURCE_DIR}/share/cmake/Torch/")
find_package(Torch)
add_subdirectory(Strife.Common)
add_subdirectory(src)

