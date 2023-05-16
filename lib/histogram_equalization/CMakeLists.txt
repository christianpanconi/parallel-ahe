# histogram equalization lib
cmake_minimum_required(VERSION 3.18)
project(histogram_equalization_lib CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# Setup CUDA
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 11)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Target cuda architecture
message(STATUS "Configured target CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
option( USE_CUDA_LAUNCH_BOUNDS "USE_CUDA_LAUNCH_BOUNDS" ON )

if( NOT CMAKE_CUDA_ARCHITECTURES )
	message( FATAL_ERROR "No CUDA architectures configured. Rerun cmake using the '-DCMAKE_CUDA_ARCHITECTURES' option." )
else()
	if( USE_CUDA_LAUNCH_BOUNDS )
		if( CMAKE_CUDA_ARCHITECTURES MATCHES ";" )
			# multiple CUDA architectures specified,
			# pick first one for launch bounds
			# string( REPLACE "," ";" CUDA_ARCHS_LIST ${CMAKE_CUDA_ARCHITECTURES} )
			list( GET CMAKE_CUDA_ARCHITECTURES 0 LB_CUDA_ARCHITECTURE )		
			message( STATUS "Multiple CUDA architectures specified. The kernels '__launch_bounds__' will be optimized for the first one only: ${LB_CUDA_ARCHITECTURE}" )
		else()
			set( LB_CUDA_ARCHITECTURE ${CMAKE_CUDA_ARCHITECTURES} )
		endif() 
		
		if( NOT LB_CUDA_ARCHITECTURE MATCHES "^[0-9]+$" )
			if( LB_CUDA_ARCHITECTURE MATCHES "^[0-9]+-.*$" )
				string( REGEX REPLACE "-.*$" "" LB_CUDA_ARCHITECTURE ${LB_CUDA_ARCHITECTURE} )
			else()
				message( FATAL_ERROR "Cannot optimize kernels '__launch_bounds__' for architecture: ${LB_CUDA_ARCHITECTURE}.\nSpecify one or more valid CUDA architectures using '-DCMAKE_CUDA_ARCHITECTURES' or disable the launch bounds optimization with '-DUSE_CUDA_LAUNCH_BOUNDS=OFF'." )
			endif()
		endif()
	
		message( STATUS "Kernels '__launch_bounds__' will be optimized for CUDA architecture: ${LB_CUDA_ARCHITECTURE}" )
	else()
		message( STATUS "Kernels '__launch_bounds__' optimization disabled." )
	endif()
endif()

# OpenMP
find_package( OpenMP REQUIRED )

# Library target
add_library(histogram_equalization SHARED 
	helper_cuda.h 
	include/equalization_common.hpp 
	include/equalization.hpp 
	include/equalization_parallel.hpp 
	include/equalization_gpu.cuh 
	equalization_gpu_common.cu 
	equalization_gpu.cu )
if( USE_CUDA_LAUNCH_BOUNDS AND LB_CUDA_ARCHITECTURE )
	target_compile_definitions( histogram_equalization 
		PUBLIC LB_CUDA_ARCHITECTURE=${LB_CUDA_ARCHITECTURE} )
endif()
target_include_directories(histogram_equalization PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")
set_property(TARGET histogram_equalization PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(histogram_equalization PROPERTIES PUBLIC_HEADER "include/equalization.hpp;include/equalization_parallel.hpp;include/equalization_gpu.cuh" )
target_compile_options(histogram_equalization PRIVATE $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CUDA>:
	--generate-line-info
>>)
target_link_libraries(histogram_equalization PUBLIC OpenMP::OpenMP_CXX)

# Support installation of the library
include(GNUInstallDirs)
install( TARGETS histogram_equalization 
         RUNTIME DESTINATION bin/histogram-equalization
         LIBRARY DESTINATION lib/histogram-equalization
         PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/histogram-equalization )
        

