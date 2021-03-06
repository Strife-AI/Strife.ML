cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Torch REQUIRED)

add_library(Strife.ML STATIC
        StrifeML.hpp
        StrifeML.cpp
        TensorPacking.hpp
        Trainer.hpp
        Serialization.hpp
        Decider.hpp
        NeuralNetwork.hpp
        SampleRepository.hpp
        NetworkContext.hpp
        MlUtil.hpp
        Sample.hpp)

set_property(TARGET Strife.ML PROPERTY CXX_STANDARD 17)

find_package(Microsoft.GSL CONFIG REQUIRED)

# Copy torch dlls
if (MSVC)
  message("============================ Copying Torch to Output ============================")
  message("From Torch Dir: ${TORCH_DIR}")
  message("To Output Dir: ${CMAKE_BINARY_DIR}/$<CONFIG>/")
  file(GLOB TORCH_DLLS "${TORCH_DIR}/torch-test-src/lib/*.dll")

  add_custom_command(TARGET Strife.ML
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} "${CMAKE_BINARY_DIR}/$<CONFIG>/")
endif()

target_link_libraries(Strife.ML PUBLIC Microsoft.GSL::GSL Strife.Common)
target_include_directories(Strife.ML PUBLIC .)

target_include_directories(Strife.ML PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(Strife.ML PUBLIC ${TORCH_LIBRARIES})

message("Include dirs ${TORCH_INCLUDE_DIRS}")