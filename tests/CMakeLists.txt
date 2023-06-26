# histogram_equalization_tests
cmake_minimum_required(VERSION 3.18)
project(mean_shift_tests CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE) # default built type to Release
	set(CMAKE_BUILD_TYPE Release)
endif()
message( STATUS "Build type: ${CMAKE_BUILD_TYPE}" )

message(STATUS "Configured CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Setup CUDA
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 11)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Google benchmark
option( DOWNLOAD_DEPENDENCIES "DOWNALOD_DEPENDENCIES" OFF )

if( NOT DOWNLOAD_DEPENDENCIES )
	# libjpeg-turbo
	find_path(JPEGTURBO_INCLUDE_DIR turbojpeg.h)
	find_library(JPEGTURBO_LIB turbojpeg)
	if( JPEGTURBO_INCLUDE_DIR STREQUAL "JPEGTURBO_INCLUDE_DIR-NOTFOUND" OR JPEGTURBO_LIB STREQUAL "JPEGTURBO_LIB-NOTFOUND" )
		message( FATAL_ERROR "libjpeg-turbo not found. Install it or enable automatic dependencies download with '-DDOWNLOAD_DEPENDENCIES'." )
	endif()
	message( STATUS "JPEGTURBO_INCLUDE_DIR: ${JPEGTURBO_INCLUDE_DIR}" )
	message( STATUS "JPEGTURBO_LIB: ${JPEGTURBO_LIB}" )

	# Google benchmark
	find_package( benchmark )
	if( NOT benchmark_FOUND )
		message( FATAL_ERROR "Google benchmark library not found. Supply an installation path using '-Dbenchmark_DIR' or enable automatic dependencies download with '-DDOWNLOAD_DEPENDENCIES=ON'." )
	endif()	
else() # DOWNLOAD_DEPENDENCIES
	include(FetchContent)
	include(ExternalProject)
	# libjpeg-turbo
	message( STATUS "Downloading libjpeg-turbo" )
	FetchContent_Declare( 
		libjpeg_turbo
		GIT_REPOSITORY https://github.com/libjpeg-turbo/libjpeg-turbo.git
		GIT_TAG main
	)
	FetchContent_GetProperties(libjpeg_turbo)
	if( NOT libjpeg_turbo_POPULATED )
		FetchContent_Populate(libjpeg_turbo)	
		ExternalProject_Add( 
			libjpeg_turbo
			PREFIX "../_deps/libjpeg_turbo-build"
			SOURCE_DIR "../_deps/libjpeg_turbo-src"
			BINARY_DIR "../_deps/libjpeg_turbo-build"
#			STAMP_DIR "../_deps/libjpeg_turbo-stamp"
			INSTALL_DIR "../_deps/libjpeg_turbo-install"
			CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=../libjpeg_turbo-install
		)
		set( JPEGTURBO_INCLUDE_DIR ${CMAKE_BINARY_DIR}/_deps/libjpeg_turbo-install/include )
		set( JPEGTURBO_LIB ${CMAKE_BINARY_DIR}/_deps/libjpeg_turbo-install/lib/libturbojpeg.so )
		message( STATUS "JPEGTURBO_INCLUDE_DIR: ${JPEGTURBO_INCLUDE_DIR}" )
		message( STATUS "JPEGTURBO_LIB: ${JPEGTURBO_LIB}" )  
	endif()	
	
	# Google benchmark
	message( STATUS "Downloading Google Benchmark" )
	set( BENCHMARK_ENABLE_GTEST_TESTS OFF )
	set( BENCHMARK_ENABLE_TESTING OFF )
	FetchContent_Declare(
		google_benchmark
		GIT_REPOSITORY https://github.com/google/benchmark.git
		GIT_TAG 	   main
	)
	FetchContent_MakeAvailable(google_benchmark)
	message( STATUS "benchmark_SOURCE_DIR=${google_benchmark_SOURCE_DIR}" )
	message( STATUS "benchmark_BINARY_DIR=${google_benchmark_BINARY_DIR}" )
endif()

# Utils
set( timer_LIBS timer/timer.hpp )
set( imutils_LIBS img_utils/color_conversion.hpp
				  img_utils/ImageProcessor.hpp )

# TARGETS 
# hist-equalization
add_executable( hist-equalization hist-equalization.cpp
	${imutils_LIBS}
	${timer_LIBS} )
if( DOWNLOAD_DEPENDENCIES )
	add_dependencies( hist-equalization libjpeg_turbo )
endif()
target_include_directories(hist-equalization PRIVATE ${JPEGTURBO_INCLUDE_DIR} )
target_link_libraries(hist-equalization
	PRIVATE histogram_equalization
	PRIVATE ${JPEGTURBO_LIB} )
#set_property( TARGET hist-equalization PROPERTY CUDA_SEPARABLE_COMPILATION ON )
target_compile_options( hist-equalization PRIVATE $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CUDA>:
	--generate-line-info
>>)

# benchmark-hist-equalization
add_executable(benchmark-hist-equalization benchmark-hist-equalization.cpp
	${imutils_LIBS} 
	${timer_LIBS} )
if( DOWNLOAD_DEPENDENCIES )
	add_dependencies( benchmark-hist-equalization libjpeg_turbo )
endif()
target_include_directories(benchmark-hist-equalization PRIVATE ${JPEGTURBO_INCLUDE_DIR} )
target_link_libraries( benchmark-hist-equalization
	PRIVATE histogram_equalization
	PRIVATE ${JPEGTURBO_LIB}
	PUBLIC benchmark::benchmark )
#set_property( TARGET benchmark-hist-equalization PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options( benchmark-hist-equalization PRIVATE $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CUDA>:
	--generate-line-info
>>)
