#ifndef EQUALIZATION_HPP_
#define EQUALIZATION_HPP_

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <cmath>
#include <cstring>

#include "equalization_common.hpp"

/**
 * Simple histogram equalization.
 * Equalizes a given channel with n_pixels.
 * The values in the input channel are assumed to be in
 * the range [0,255].
 */
unsigned char* equalize_hist( unsigned char* channel ,
							  const unsigned int n_pixels ){

	const unsigned int N_VALUES = 256;
	unsigned int hist[N_VALUES];
	memset( hist , 0 , sizeof hist );

	// Compute channel histogram
	for( int i=0 ; i < n_pixels ; i++ )
		hist[channel[i]]++;

	// Compute equalized channel values
	unsigned char lut[N_VALUES];
	int sum = 0;
	for( int i=0; i < N_VALUES ; i++ ){
		sum += hist[i];
		lut[i] = clamp8bit( (unsigned int)((float)sum * (N_VALUES-1) / n_pixels) + 0.5 );
	}

	unsigned char* eq_channel = new unsigned char[n_pixels];
	for( int i=0 ; i < n_pixels ; i++ )
		eq_channel[i] = lut[channel[i]];

	return eq_channel;
}

/**
 * Augment the channel by mirroring the image borders.
 * The size of the produced channel is:
 * (width+window_size/2)*(height+window_size/2)
 */
unsigned char* augmented_channel( unsigned char* channel , unsigned int width , unsigned int height , unsigned int window_size ,
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


typedef struct aligned_hist_value{
	alignas(64) unsigned int value;
} aligned_hist_value;

/**
 * AHE implementation with sliding window.
 * At each step the window histogram is computed from
 * the previous one by:
 * - adding the histogram of the last window row in the current iteration
 * - subtracting the histogram of the first window row in the previous iteration
 */
unsigned char* equalize_hist_SWAHE_mono( unsigned char *channel ,
										 unsigned int width , unsigned int height ,
										 unsigned int window_size ,
										 unsigned int n_values=256 ){
	// window_size to nearest odd integer
	window_size += window_size % 2 == 0 ? 1 : 0;

	// augmented channel with mirrored borders
	unsigned int aug_width , aug_height;
	unsigned char* aug_channel = augmented_channel(
		channel , width, height, window_size, &aug_width, &aug_height );
	// equalized values channel
	unsigned char* eq_channel = new unsigned char[width*height];

	// histograms
	unsigned int hist[n_values];

	// indices
	unsigned int c_start;
	unsigned int r_start;

	// cdf
	unsigned int cdf_v;

	for( int i=0 ; i < width ; i++ ){ // i -> channel col index
		c_start = i;
		r_start = 0;

		// calculate the histogram on the whole window only for
		// the first pixel of the column (leave out last row)
		memset( hist , 0 , sizeof hist );
		for( int r=0 ; r < window_size-1 ; r++ ){
			for( int c=0 ; c < window_size ; c++ ){
				// XXX : here always r_start = 0
				hist[aug_channel[(r+r_start)*aug_width + c_start + c]]++;
			}
		}

		// iterate on the column (downward)
		for( int j=0 ; j < height ; j++ ){ // j -> channel row index

			// add the histogram of the last window row
			for( int c=0 ; c < window_size ; c++ )
				hist[aug_channel[(r_start + window_size - 1) * aug_width + c_start + c]]++;

			// CDF
			// (vectorizable)
			cdf_v = 0;
			for( unsigned int l=0 ; l <= channel[j*width+i] ; l++ )
				cdf_v += hist[l];

			// Compute equalized value
			eq_channel[j*width+i] = clamp8bit(
				(unsigned int)((float)cdf_v*(n_values-1)/(window_size*window_size) + 0.5)
			);

			// subtract the top row histogram for the next iteration
			for( int c = 0 ; c < window_size ; c++ )
				hist[aug_channel[r_start*aug_width + c_start + c]]--;

			r_start++;
		}
	}

	delete aug_channel;
	return eq_channel;
}

/**
 * AHE implementation with sliding window in both directions
 */
unsigned char* equalize_hist_SWAHE_bi( unsigned char* channel ,
									   unsigned int width , unsigned int height ,
									   unsigned int window_size ,
									   unsigned int n_values=256 ){

	// window_size to nearest odd integer
	window_size += window_size % 2 == 0 ? 1 : 0;

	// augmented channel with mirrored borders
	unsigned int aug_width , aug_height;
	unsigned char* aug_channel = augmented_channel(
		channel , width, height, window_size, &aug_width, &aug_height );
	// equalized values channel
	unsigned char* eq_channel = new unsigned char[width*height];

	unsigned int hist[n_values];
	unsigned int c_start , r_start;
	unsigned int cdf_v;
	bool hist_initialized = false;

	int dir = 0; // 0=DOWN , 1=UP

	for( int i=0 ; i < width ; i++ ){ // i <- channel col index
		c_start = i;
		r_start = dir*(height-1);

		if( !hist_initialized ){ // compute the histogram
			memset( hist , 0 , sizeof hist );
			for( int r=0 ; r < window_size ; r++ )
				for( int c=0 ; c < window_size ; c++ )
					hist[aug_channel[(r+r_start)*aug_width + c_start + c]]++;
			hist_initialized = true;
		}else{ 					// add the histogram of the last window_column
			for( int r=0 ; r<window_size ; r++ )
				hist[aug_channel[(r_start+r)*aug_width + c_start + window_size-1]]++;
		}


//		for( int j=0 ; j < height ; j++ ){ // j <- channel row index
		for( int j=dir*(height-1) ; (dir==0 && j < height) || (dir==1 && j >= 0) ; j += (-1+2*((dir+1)%2)) ){
			// add the histogram of the last/first window row
			if( (dir==0 && j>0) || (dir==1 && j < height-1) ){
				for( int c=0 ; c < window_size ; c++ )
					hist[aug_channel[(j + (1-dir)*(window_size-1))*aug_width + c_start + c]]++;
			}


			// compute equalized value
			cdf_v = 0;
			for( int l=0 ; l <= channel[j*width+i] ; l++ )
				cdf_v += hist[l];

			eq_channel[j*width+i] = clamp8bit(
				(unsigned int)((float)cdf_v*(n_values-1)/(window_size*window_size) + 0.5)
			);

			// subtract the histogram of the first/last window row
			if( (dir==0 && j < height-1) || (dir==1 && j > 0 ) ){
				for( int c=0 ; c < window_size ; c++ )
					hist[aug_channel[(j + dir*(window_size-1))*aug_width + c_start + c ]]--;
				r_start += (-1+2*((dir+1)%2));
			}
		}

		// subtract the histogram of the first window col
		for( int r=0 ; r < window_size ; r++ ){
			hist[aug_channel[(r_start+r)*aug_width + c_start]]--;
		}

		dir=1-dir;
	}

	delete aug_channel;
	return eq_channel;
}

/**
 * Alternative implementation with the window sliding along the rows.
 * Only for testing purposes.
 */
unsigned char* equalize_hist_AHE_alt( unsigned char *channel , unsigned int width , unsigned int height ,
									  unsigned int window_size , unsigned int n_values=256){

	// window_size to nearest odd integer
	window_size += window_size % 2 == 0 ? 1 : 0;

	// augmented channel with mirrored borders
	unsigned int aug_width , aug_height;
	unsigned char* aug_channel = augmented_channel(
		channel , width, height, window_size, &aug_width, &aug_height );
	// equalized values channel
	unsigned char* eq_channel = new unsigned char[width*height];

	// histograms
	unsigned int hist[n_values];

	// indices
	unsigned int c_start;
	unsigned int r_start;

	// cdf
	unsigned int cdf_v;

	for( int i=0 ; i < height ; i++ ){ // i -> channel row index
//		c_start = i;
//		r_start = 0;
		c_start = 0;
		r_start = i;

		// calculate the histogram on the whole window only for
		// the first pixel of the row (leave out last column)
		memset( hist , 0 , sizeof hist );
		for( int r=0 ; r < window_size ; r++ ){
			for( int c=0 ; c < window_size-1 ; c++ ){
				// XXX : here always r_start = 0
				hist[aug_channel[(r+r_start)*aug_width + c_start + c]]++;
			}
		}

		// iterate on the column (downward)
		for( int j=0 ; j < width ; j++ ){ // j -> channel column index

			// add the histogram of the last window column
			for( int r=0 ; r < window_size ; r++ )
				hist[aug_channel[(r_start + r) * aug_width + c_start + window_size-1 ]]++;

			// CDF
			// (vectorizable)
			cdf_v = 0;
			for( unsigned int l=0; l <= channel[i*width+j] ; l++ )
				cdf_v += hist[l];

			// Compute equalized value
			eq_channel[i*width+j] = clamp8bit(
//				(unsigned int)((float)cdf_v*(max-min)/(window_size*window_size) + min + 0.5)
				(unsigned int)((float)cdf_v*(n_values-1)/(window_size*window_size) + 0.5)
			);

			// subtract the left column histogram for the next iteration
			for( int r = 0 ; r < window_size ; r++ )
				hist[aug_channel[(r_start+r)*aug_width + c_start ]]--;
//			r_start++;
			c_start++;

		}
	}

	delete aug_channel;
	return eq_channel;
}

#endif /* EQUALIZATION_HPP_ */
