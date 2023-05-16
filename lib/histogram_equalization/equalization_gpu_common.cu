#include "include/equalization_gpu.cuh"

/**
 * Clamp the argument on 8-bit range
 */
__device__ unsigned char clamp8bit_d( unsigned int x ){
	return (unsigned char)( x > 255 ? 255 : x );
}

__device__ unsigned char clamp8bit_d( int x ){
	return (unsigned char)( x > 255 ? 255 : (x < 0 ? 0 : x) );
}

/**
 * Augment the channel by mirroring the image borders.
 * The size of the produced channel is:
 * (width+window_size/2)*(height+window_size/2)
 */
unsigned char* augment_channel( unsigned char* channel , unsigned int width , unsigned int height , unsigned int window_size ,
								unsigned int *aug_ch_width , unsigned int *aug_ch_height ){

	const unsigned int aug_width = width + 2 * (window_size/2);
	const unsigned int aug_height = height + 2 * (window_size/2);
	unsigned char* aug_channel = new unsigned char[aug_width*aug_height];

	unsigned int n_mir = window_size / 2;

	for( int r = 0 ; r < height ; r++ ){
		for( int c = 0 ; c < n_mir ; c++ ){
			// left border
			aug_channel[ (r+n_mir)*aug_width + c ] = channel[ r*width + n_mir - c ];
			// right border
			aug_channel[ (r+n_mir)*aug_width + n_mir + width + c ] =
							channel[ r*width + width - 2 - c ];
		}

		// row center
		for( int c = 0 ; c < width ; c++ ){
			aug_channel[ (r+n_mir)*aug_width + n_mir + c ] = channel[ r*width + c ];
		}
	}

	// top - bottom
	for( int r=0 ; r < n_mir ; r++ ){
		for( int c=0 ; c < aug_width ; c++ ){
			// top
			aug_channel[ r*aug_width + c ] = aug_channel[ (2*n_mir-r)*aug_width + c ];
			// bottom
			aug_channel[ (height+n_mir+r)*aug_width + c] = aug_channel[ (height + n_mir - 2 - r)*aug_width + c];
		}
	}

	*aug_ch_width = aug_width;
	*aug_ch_height = aug_height;
	return aug_channel;
}
