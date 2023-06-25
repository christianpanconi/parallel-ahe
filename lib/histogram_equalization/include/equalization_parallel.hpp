#ifndef EQUALIZATION_PARALLEL_HPP_
#define EQUALIZATION_PARALLEL_HPP_

#include <iostream>
#include <cstring>

#include <omp.h>

#include "equalization.hpp"
#include "equalization_common.hpp"

/**
 * Parallel AHE implementation using OpenMP
 * (monodirectional window sliding)
 */
unsigned char* equalize_hist_SWAHE_omp_mono( unsigned char* channel ,
									  	     unsigned int width , unsigned int height ,
											 unsigned int window_size ,
											 unsigned int n_values=256 ,
											 unsigned int n_threads=4 ){

	// window_size to nearest odd integer
	window_size += window_size % 2 == 0 ? 1 : 0;

	// augmented channel with mirrored borders
	unsigned int aug_width , aug_height;
	unsigned char* aug_channel = augmented_channel(
		channel , width, height, window_size, &aug_width, &aug_height );
	// equalized values channel
	unsigned char* eq_channel = new unsigned char[width*height];

#pragma omp parallel default(none) shared( channel , width , height , \
										   aug_channel , aug_width , aug_height , \
										   window_size , eq_channel , n_values ) \
								   num_threads(n_threads)
	{
		unsigned int hist[n_values];
		unsigned int c_start , r_start;
		unsigned char min, max;
		unsigned int cdf_v;

#pragma omp for
		for( int i=0 ; i < width ; i++ ){ // i <- channel col index
			c_start = i;
			r_start = 0;

			// histogram on the full window for the first pixel
			memset( hist , 0 , sizeof hist );
			for( int r=0 ; r < window_size-1 ; r++ )
				for( int c=0 ; c < window_size ; c++ )
					hist[aug_channel[(r+r_start)*aug_width + c_start + c]]++;

			// iterate downward on the column
			for( int j=0 ; j<height ; j++ ){ // j <- channel row index

				// add the histogram of the last window row
				for( int c=0 ; c < window_size ; c++ )
					hist[aug_channel[(r_start + window_size - 1) * aug_width + c_start + c]]++;

				// CDF
				// (vectorizable)
				cdf_v = 0;
				for( unsigned int l = 0 ; l <= channel[j*width+i] ; l++ )
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

	} // ends parallel

	delete aug_channel;
	return eq_channel;
}

/**
 * Parallel AHE implementation using OpenMP
 * (bidirectional window sliding)
 */
unsigned char* equalize_hist_SWAHE_omp_bi( unsigned char* channel ,
										unsigned int width , unsigned int height ,
										unsigned int window_size ,
										unsigned int n_values=256 ,
										unsigned int n_threads=4 ){

	// window_size to nearest odd integer
	window_size += window_size % 2 == 0 ? 1 : 0;

	// augmented channel with mirrored borders
	unsigned int aug_width , aug_height;
	unsigned char* aug_channel = augmented_channel(
		channel , width, height, window_size, &aug_width, &aug_height );
	// equalized values channel
	unsigned char* eq_channel = new unsigned char[width*height];

#pragma omp parallel default(none) shared( channel , width , height , \
										   aug_channel , aug_width , aug_height , \
										   window_size , eq_channel , n_values ) \
								   num_threads(n_threads)
	{
		unsigned int hist[n_values];
		unsigned int c_start , r_start;
		unsigned int cdf_v;
		bool hist_initialized = false;

		int dir = 0; // 0=DOWN , 1=UP

#pragma omp for
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
	}

	delete aug_channel;
	return eq_channel;
}

#endif /* EQUALIZATION_PARALLEL_HPP_ */
