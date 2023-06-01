# histogram equalization python module
cmake_minimum_required(VERSION 3.18)
project(mean_shift_module CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE) # default built type to Release
	set(CMAKE_BUILD_TYPE Release)
endif()
message( STATUS "Build type: ${CMAKE_BUILD_TYPE}" )

message(STATUS "Configured CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")

# Setup CUDA
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 11)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Setup python and numpy
find_package(Python REQUIRED Development NumPy)

# Module target
add_subdirectory( ../lib/histogram_equalization ./lib )
set( SOURCE_DIR "./histogram_equalization" )
add_library( histogram_equalization_module SHARED
	${SOURCE_DIR}/timer.hpp
	${SOURCE_DIR}/color_conversion.hpp 
	${SOURCE_DIR}/histogram_equalization_py_module.h
	${SOURCE_DIR}/histogram_equalization_py_module.cpp
	${SOURCE_DIR}/histogram_equalization_py_wrappers.h
	${SOURCE_DIR}/histogram_equalization_py_wrappers.cpp )
target_link_libraries( histogram_equalization_module
	PRIVATE histogram_equalization 
	PRIVATE Python::NumPy )