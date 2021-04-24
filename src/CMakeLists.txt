cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Torch REQUIRED)

add_library(Strife.ML STATIC
        StrifeML.hpp
        StrifeML.cpp
        TensorPacking.hpp
        TorchApi.h
        TorchApi.cpp
        TorchApiInternal.hpp
        Scripting/ScriptCompiler.cpp
        Scripting/Scripting.cpp
        ScriptNetwork.cpp Trainer.hpp Serialization.hpp Decider.hpp NeuralNetwork.hpp SampleRepository.hpp NetworkContext.hpp MlUtil.hpp Sample.hpp)

set_property(TARGET Strife.ML PROPERTY CXX_STANDARD 17)


find_package(Microsoft.GSL CONFIG REQUIRED)


if (MSVC)
    target_link_directories(Strife.ML PUBLIC ../tinycc/win32)
    target_link_libraries(Strife.ML PUBLIC libtcc)

    add_custom_command(TARGET Strife.ML
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/../tinycc/win32/libtcc.dll $<TARGET_FILE_DIR:Strife.ML>)
else()
    target_link_directories(Strife.ML PUBLIC ../tinycc)
    target_link_libraries(Strife.ML PUBLIC libtcc.a)
endif()

add_custom_command(TARGET Strife.ML
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/../src/TorchApi.h ${CMAKE_BINARY_DIR}/tcc/include)

target_link_libraries(Strife.ML PUBLIC Microsoft.GSL::GSL Strife.Common)
target_include_directories(Strife.ML PUBLIC ../tinycc)
target_include_directories(Strife.ML PUBLIC .)

target_include_directories(Strife.ML PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(Strife.ML PUBLIC ${TORCH_LIBRARIES})

message("Include dirs ${TORCH_INCLUDE_DIRS}")