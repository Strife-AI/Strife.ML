cmake_minimum_required(VERSION 3.2.02 FATAL_ERROR)

include (ExternalProject)
include(FetchContent)

project(ML VERSION 1 LANGUAGES C CXX)

if (MSVC)
	if (defined ${CMAKE_BUILD_TYPE})
		message("============================ CMake_Build_Type is ${CMAKE_BUILD_TYPE}")
		set(BUILD_TYPE ${CMAKE_BUILD_TYPE})
	else()
		set(BUILD_TYPE ${CMAKE_CONFIGURATION_TYPES})
	endif()

	if (${BUILD_TYPE} STREQUAL "Debug")
        set(TorchCpuCache ${CMAKE_SOURCE_DIR}/.torch/cpu/debug)
        set(TorchCpuUrl https://download.pytorch.org/libtorch/cu102/libtorch-win-shared-with-deps-debug-1.9.0%2Bcu102.zip)

        set(TorchCudaCache ${CMAKE_SOURCE_DIR}/.torch/cuda/debug)
        set(TorchCudaUrl https://download.pytorch.org/libtorch/cu111/libtorch-win-shared-with-deps-debug-1.9.0%2Bcu111.zip)
	else()
        set(TorchCpuCache ${CMAKE_SOURCE_DIR}/.torch/cpu/release)
        set(TorchCpuUrl https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.9.0%2Bcpu.zip)

        set(TorchCudaCache ${CMAKE_SOURCE_DIR}/.torch/cuda/release)
        set(TorchCudaUrl https://download.pytorch.org/libtorch/cu111/libtorch-win-shared-with-deps-1.9.0%2Bcu111.zip)
	endif()
elseif(APPLE)
    set(TorchCpuCache ${CMAKE_SOURCE_DIR}/.torch/cpu)
    set(TorchCpuUrl https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.9.0.zip)
else()
    set(TorchCpuCache ${CMAKE_SOURCE_DIR}/.torch/cpu)
    set(TorchCpuUrl https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip)
endif()

set(FETCHCONTENT_BASE_DIR ${TorchCudaCache})

message("============================ Downloading Torch ============================")
FetchContent_Declare(
        torch-test
        URL ${TorchCudaUrl}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND "")

FetchContent_MakeAvailable(torch-test)

set(TORCH_DIR ${TorchCudaCache} CACHE STRING "Torch directory")

list(APPEND CMAKE_PREFIX_PATH "${torch-test_SOURCE_DIR}/share/cmake/Torch/")
find_package(Torch)
add_subdirectory(src)

