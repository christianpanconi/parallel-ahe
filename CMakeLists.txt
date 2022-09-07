cmake_minimum_required(VERSION 3.18)
project(histogram_equalization CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(CheckLanguage)
check_language(CUDA)

# Output detected CUDA compiler
message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID: ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")

# Enable CUDA language
enable_language(CUDA)

# Include Cudatoolkit directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTIORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 11)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Specify the CUDA architecture / compute capabilities
set(CMAKE_CUDA_ARCHITECTURES 61)
message(STATUS "target CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_SOURCE_DIR src)
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

# libjpeg-turbo
find_path(JPEGTURBO_INCLUDE_DIR turbojpeg.h)
find_library(JPEGTURBO_LIB turbojpeg)
message(STATUS "JPEGTURBO_LIB: ${JPEGTURBO_LIB}")

# OpenMP
find_package( OpenMP REQUIRED )

# Google Benchmark
find_package( benchmark REQUIRED )

# Custom libraries
set(timer_LIBS ${CMAKE_SOURCE_DIR}/timer/timer.hpp)
set(imutils_LIBS ${CMAKE_SOURCE_DIR}/img_utils/color_conversion.hpp
				 ${CMAKE_SOURCE_DIR}/img_utils/ImageProcessor.hpp )
set(equalization_CPU 
	${CMAKE_SOURCE_DIR}/equalization_cpu/equalization_common.hpp
	${CMAKE_SOURCE_DIR}/equalization_cpu/equalization.hpp
	${CMAKE_SOURCE_DIR}/equalization_cpu/equalization_parallel.hpp )
set(equalization_GPU 
	${CMAKE_SOURCE_DIR}/equalization_cuda/helper_cuda.h 
	${CMAKE_SOURCE_DIR}/equalization_cuda/equalization_gpu.cuh 
	${CMAKE_SOURCE_DIR}/equalization_cuda/equalization_gpu_common.cu 
	${CMAKE_SOURCE_DIR}/equalization_cuda/equalization_gpu.cu )
 
# Targets 
	
# hist-equalization
add_executable(hist-equalization ${CMAKE_SOURCE_DIR}/hist-equalization.cpp
	${imutils_LIBS}
	${timer_LIBS}
	${equalization_CPU}
	${equalization_GPU} )
target_include_directories(hist-equalization PRIVATE ${JPEGTURBO_INCLUDE_DIR} )
target_link_libraries(hist-equalization PRIVATE ${JPEGTURBO_LIB} )
set_property( TARGET hist-equalization PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# target_compile_options(hist-equalization PRIVATE -g) 
#target_compile_options(hist-equalization PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#	--generate-line-info
#>)

# benchmark-hist-equalization
add_executable(benchmark-hist-equalization ${CMAKE_SOURCE_DIR}/benchmark-hist-equalization.cpp
	${imutils_LIBS}
	${equalization_CPU}
	${equalization_GPU} )
target_include_directories(benchmark-hist-equalization PRIVATE ${JPEGTURBO_INCLUDE_DIR} )
target_link_libraries(benchmark-hist-equalization 
	PRIVATE ${JPEGTURBO_LIB} 
	PUBLIC benchmark::benchmark)
set_property( TARGET benchmark-hist-equalization PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#target_compile_options(benchmark-hist-equalization PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#	--generate-line-info
#>)