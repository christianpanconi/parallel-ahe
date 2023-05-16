#ifndef EQUALIZATION_GPU_CUH_
#define EQUALIZATION_GPU_CUH_

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

/**
 * Common
 */
extern __device__ unsigned char clamp8bit_d( unsigned int x );

extern __device__ unsigned char clamp8bit_d( int x );

/**
 * Augment the channel by mirroring the image borders.
 * The size of the produced channel is:
 * (width+window_size/2)*(height+window_size/2)
 */
unsigned char* augment_channel( unsigned char* channel , unsigned int width , unsigned int height , unsigned int window_size ,
								unsigned int *aug_ch_width , unsigned int *aug_ch_height );


__host__ unsigned int determine_pbs( unsigned int width , unsigned int height );

/**
 * Equalization functions
 */
__host__ unsigned char* equalize_hist_SWAHE_gpu_bi(
	unsigned char* channel , unsigned int width , unsigned int height ,
	unsigned int window_size ,
	unsigned int pb_width , unsigned int pb_height );

__host__ unsigned char* equalize_hist_SWAHE_gpu_mono(
	unsigned char* channel , unsigned int width , unsigned int height ,
	unsigned int window_size ,
	unsigned int pb_width , unsigned int pb_height );


/**
 * Benchmarking purposes
 */
__host__ void init_cuda_context();

#endif /* EQUALIZATION_GPU_CUH_ */
