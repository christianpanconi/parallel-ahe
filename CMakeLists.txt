cmake_minimum_required(VERSION 3.18)
project(histogram_equalization CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory( lib/histogram_equalization ./lib )

option( SKIP_TESTS "SKIP_TESTS" OFF )
if( NOT SKIP_TESTS )
	add_subdirectory( tests ./test )
endif()
