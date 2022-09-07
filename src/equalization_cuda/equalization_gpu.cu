#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <cmath>

#include "equalization_gpu.cuh"
#include "helper_cuda.h"

#define N_VALUES 256

/**
 * Mono-Directional SWAHE
 * Each block operates on a pb_width*pb_height image tile.
 * CDF with parallel scan.
 */
__global__ void
__launch_bounds__( 64 , 32 ) // CC 6.1 : 65536 registers/SM, 32 blocks/SM, 64 threads/block
equalize_hist_SWAHE_kernel_mono(
	unsigned char* aug_channel , unsigned int aug_width , unsigned int aug_height ,
	unsigned char* dst_channel , unsigned int width , unsigned int height ,
	unsigned int window_size , unsigned int pb_width , unsigned int pb_height ){

	unsigned int window_half = window_size /2;

	unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int n_threads = blockDim.x * blockDim.y;
	unsigned int acc_steps = N_VALUES / (blockDim.x*blockDim.y); // 256 multiple of n_threads

	// how many steps per window the current block should perform
	// in both directions x and y
	unsigned int x_steps = ceil( window_size/(float)blockDim.x );
	unsigned int y_steps = ceil( window_size/(float)blockDim.y );
	__shared__ unsigned int hist[3*N_VALUES]; // Double buffer + tmp

	// target pixel coordinates
	unsigned int px = blockIdx.x * pb_width;
	unsigned int py = blockIdx.y * pb_height;
	unsigned char target_value;
	unsigned int in = 0, out = 1 , tmp = 2 , last_acc = 0;

	for( int c=0 ; c < pb_width && blockIdx.x*pb_width+c < width; c++ ){ 	// col index
		py = blockIdx.y * pb_height;
		px = blockIdx.x * pb_width + c;

		// reset the histogram buffers
		for( int reset_step=0 ; reset_step < acc_steps ; reset_step++ ) // acc_steps = 256 bins / 64 threads
			hist[ tmp*N_VALUES + reset_step*n_threads + tid ] = 0;
		__syncthreads();

		// Build full window histogram (leave out last row)
		for( int i=0 ; i < y_steps ; i++ ){
			for( int j=0 ; j < x_steps ; j++ ){
				if( i*blockDim.y+threadIdx.y < window_size-1 && j*blockDim.x + threadIdx.x < window_size )
					atomicAdd(
						&(hist[ tmp*N_VALUES + aug_channel[(py+threadIdx.y+i*blockDim.y)*aug_width + px+j*blockDim.x+threadIdx.x ]]),
						1 );
			}
		}
		__syncthreads();

		for( int r=0 ; r < pb_height && blockIdx.y*pb_height+r < height; r++){  // row index
			py = blockIdx.y * pb_height + r;

			// Add the histogram of the last window row
			for( int i=0 ; i < ceil(window_size/(float)n_threads) ; i++ ){
				if( i * n_threads + tid < window_size )
					atomicAdd(
						&(hist[tmp*N_VALUES + aug_channel[(py+window_size-1)*aug_width + px + i*n_threads + tid ]]) ,
						1 );
			}
			__syncthreads();

			target_value = aug_channel[(window_half+py)*aug_width + window_half+px ];

			// CDF ACCUMULATION
			last_acc = 0;
			for( int acc_step=0 ; acc_step < acc_steps ; acc_step++ ){ // 4 = 256 bins / 64 threads
//				in = 1;
				in = tmp;
				out = 0;
				for( int offset=1 ; offset < n_threads ; offset *= 2 ){
					// swap buffers
					out = 1 - out;
					in = ( offset == 1 ? in : 1-(in%2) );
					if( tid >= offset )
						hist[out*N_VALUES + acc_step*n_threads + tid ] =
							hist[in*N_VALUES + acc_step*n_threads + tid - offset ] + hist[in*N_VALUES + acc_step*n_threads + tid];
					else
						hist[out*N_VALUES + acc_step*n_threads + tid ] = hist[in*N_VALUES + acc_step*n_threads + tid];
					__syncthreads();
				}
				hist[out*N_VALUES + acc_step*n_threads + tid] += last_acc;
				__syncthreads();
				last_acc = hist[out*N_VALUES + acc_step*n_threads + n_threads-1];
				__syncthreads();
			}

			// Compute the equalized value using only the first thread
			if( threadIdx.x == 0 && threadIdx.y==0 ){
				dst_channel[py*width + px] = clamp8bit_d(
					(unsigned int)((float)hist[out*N_VALUES + target_value] * (N_VALUES-1) / (window_size*window_size) + 0.5 )
				);
			}
			__syncthreads();


			// Subtract the histogram of the first window row
			for( int i=0 ; i < ceil(window_size/(float)n_threads) ; i++ ){
				if( i * n_threads + tid < window_size )
					atomicSub( &(hist[tmp*N_VALUES + aug_channel[py*aug_width + px + i*n_threads + tid ]]) , 1 );
			}
			__syncthreads();

		} // batch rows loop
	} // batch cols loop

}

/**
 * Bi-Directional SWAHE
 * Each block operates on a pb_width*pb_height image tile alternating the direction.
 * CDF with parallel scan.
 */
__global__ void
__launch_bounds__( 64 , 32 )
equalize_hist_SWAHE_kernel_bi(
	unsigned char* aug_channel , unsigned int aug_width , unsigned int aug_height ,
	unsigned char* dst_channel , unsigned int width , unsigned int height ,
	unsigned int window_size , unsigned int pb_width , unsigned int pb_height ){

	unsigned int window_half = window_size /2;

	unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int n_threads = blockDim.x * blockDim.y;
	unsigned int acc_steps = N_VALUES / (blockDim.x*blockDim.y); // 256 multiple of n_threads

	// how many steps per window the current block should perform
	// in both directions x and y
	unsigned int x_steps = ceil( window_size/(float)blockDim.x );
	unsigned int y_steps = ceil( window_size/(float)blockDim.y );
	__shared__ unsigned int hist[3*N_VALUES]; // Double buffer + tmp

	// target pixel coordinates
	unsigned int px = blockIdx.x * pb_width;
	unsigned int py = blockIdx.y * pb_height;
	unsigned char target_value;
	unsigned int in = 0, out = 1 , tmp = 2 , last_acc = 0;

	// reset the histogram buffers
	for( int reset_step=0 ; reset_step < acc_steps ; reset_step++ ) // acc_steps = 256 bins / 64 threads
		hist[ tmp*N_VALUES + reset_step*n_threads + tid ] = 0;
	__syncthreads();

	// Build full window histogram (leave out last col)
	for( int i=0 ; i < y_steps ; i++ ){
		for( int j=0 ; j < x_steps ; j++ ){
			if( i*blockDim.y+threadIdx.y < window_size && j*blockDim.x + threadIdx.x < window_size-1 )
				atomicAdd(
					&(hist[ tmp*N_VALUES + aug_channel[(py+threadIdx.y+i*blockDim.y)*aug_width + px+j*blockDim.x+threadIdx.x ]]),
					1 );
		}
	}
	__syncthreads();

	int dir = 0; // slide direction on the column, 0 = DOWN / 1 = UP

	for( int c=0 ; c < pb_width && blockIdx.x*pb_width+c < width; c++ ){ 	// pixels batch col index
		py = blockIdx.y * pb_height + dir*(pb_height-1); // if dir=DOWN start from the first window row, if dir=LAST start from the last
		px = blockIdx.x * pb_width + c;

		// Add the histogram of the last window column
		for( int i=0 ; i < ceil(window_size/(float)n_threads) ; i++ ){
			if( i*n_threads + tid < window_size && py < height )
				atomicAdd( &(hist[tmp*N_VALUES + aug_channel[(py+i*n_threads+tid)*aug_width + px + window_size-1 ]])
				, 1 );
		}
		__syncthreads();

//										(dir==0 && r < pb_height) || (dir==1 && r >= 0 )
		for( int r = dir*(pb_height-1) ; r*(-1+2*((dir+1)%2)) <= (int)(pb_height-1)*(1-dir) ; r += (-1 + 2*((dir+1)%2)) ){ // pixels batch row index

			if( blockIdx.y * pb_height + r < height ){
				py = blockIdx.y * pb_height + r;

				// skip if FIRST iteration to receive the histogram from the previous column
				if( r*(-1+2*dir) < (int)(dir*(pb_height-1)) ){ // (dir==0 && r>0) || (dir==1 && r < (pb_height-1))
					// Add the histogram of the [last if down]/[first if up] window row
					for( int i=0 ; i < ceil(window_size/(float)n_threads) ; i++ ){
						if( i * n_threads + tid < window_size )
							atomicAdd( &(hist[tmp*N_VALUES + aug_channel[(py+(1-dir)*(window_size-1))*aug_width + px + i*n_threads + tid ]]) , 1 );
					}
				}
				__syncthreads();

				// CDF accumulation
				last_acc = 0;

				for( int acc_step=0 ; acc_step < acc_steps ; acc_step++ ){ // 4 = 256 bins / 64 threads
					in = tmp;
					out = 0;
					for( int offset=1 ; offset < n_threads ; offset *= 2 ){
						// swap buffers
						out = 1 - out;
						in = ( offset == 1 ? in : 1-(in%2) );
						if( tid >= offset )
							hist[out*N_VALUES + acc_step*n_threads + tid ] =
								hist[in*N_VALUES + acc_step*n_threads + tid - offset ] + hist[in*256 + acc_step*n_threads + tid];
						else
							hist[out*N_VALUES + acc_step*n_threads + tid ] = hist[in*256 + acc_step*n_threads + tid];
						__syncthreads();
					}
					hist[out*N_VALUES + acc_step*n_threads + tid] += last_acc;
					__syncthreads();
					last_acc = hist[out*N_VALUES + acc_step*n_threads + n_threads-1];
					__syncthreads();
				}

				// target (center window) pixel value
				target_value = aug_channel[(window_half+py)*aug_width + window_half+px ];

				// Compute the equalized value using only the first thread of the block
				if( threadIdx.x == 0 && threadIdx.y==0 )
					dst_channel[py*width + px] = clamp8bit_d(
						(unsigned int)((float)hist[out*N_VALUES + target_value] * (N_VALUES-1) / (window_size*window_size) + 0.5 )
					);
				__syncthreads();


				// skip if last iteration to preserve the histogram for next column
				if( r*(-1+2*((dir+1)%2)) < (int)(pb_height-1)*(1-dir) ){ // (dir==0 && r<pb_height-1) || (dir==1 && r>0)
					// Subtract the histogram of the [first if down]/[last if up] window row
					for( int i=0 ; i < ceil(window_size/(float)n_threads) ; i++ ){
						if( i * n_threads + tid < window_size )
							atomicSub( &(hist[tmp*N_VALUES + aug_channel[(py+dir*(window_size-1))*aug_width + px + i*n_threads + tid ]]) , 1 );
					}
				}
				__syncthreads();
			}

		} // closes batch ROWS loop

		// Subtract the histogram of the FIRST window column
		for( int i=0 ; i < ceil(window_size/(float)n_threads) ; i++ ){
			if( i*n_threads + tid < window_size && py < height )
				atomicSub( &(hist[tmp*N_VALUES + aug_channel[(py+i*n_threads+tid)*aug_width + px ]]) , 1 );
		}
		__syncthreads();

		dir = 1 - dir; // invert direction
	} // closes batch COLUMNS loop

}


__host__ void prepare(
	unsigned char *channel , unsigned int width , unsigned int height , unsigned int window_size ,
	unsigned char** aug_channel , unsigned int* aug_width , unsigned int* aug_height ,
	unsigned char** aug_channel_d , unsigned char** eq_channel_d ){

	unsigned int aw , ah;
	*aug_channel = augment_channel( channel , width , height , window_size , &aw , &ah );

	cudaMalloc( aug_channel_d , ah*aw*sizeof(unsigned char) );
	cudaMemcpy( *aug_channel_d , *aug_channel , aw*ah*sizeof(unsigned char) , cudaMemcpyKind::cudaMemcpyHostToDevice );

	cudaMalloc( eq_channel_d , ah*aw*sizeof(unsigned char) );
	*aug_width = aw;
	*aug_height = ah;
}

__host__ unsigned char *get_result_and_free(
	unsigned char *aug_channel_d , unsigned int width , unsigned int height ,
	unsigned char *eq_channel_d ){

	unsigned char* eq_channel = new unsigned char[width*height];
	cudaMemcpy( eq_channel , eq_channel_d , width*height*sizeof(unsigned char) , cudaMemcpyKind::cudaMemcpyDeviceToHost );

	cudaFree( eq_channel_d );
	return eq_channel;
}

__host__ unsigned char* equalize_hist_SWAHE_gpu_bi(
	unsigned char* channel , unsigned int width , unsigned int height ,
	unsigned int window_size ,
	unsigned int pb_width , unsigned int pb_height ){

	window_size += window_size % 2 == 0 ? 1 : 0;
	unsigned int aug_width , aug_height;
	unsigned char *aug_channel , *aug_channel_d, *eq_channel_d;
	prepare( channel , width , height , window_size ,
		&aug_channel , &aug_width , &aug_height ,
		&aug_channel_d , &eq_channel_d );

	// ---- Kernel launch
	dim3 blockDim( 8 , 8 , 1 );
	dim3 gridDim( (int)ceil((float)width/pb_width) ,
				  (int)ceil((float)height/pb_height) );

	equalize_hist_SWAHE_kernel_bi<<< gridDim , blockDim >>>(
		aug_channel_d , aug_width , aug_height ,
		eq_channel_d , width , height ,
		window_size , pb_width , pb_height
	);

	cudaDeviceSynchronize();
	printLastCudaError( "Kernel" );

	cudaFree(aug_channel_d);
	delete aug_channel;

	return get_result_and_free(aug_channel_d, width, height, eq_channel_d);
}

__host__ unsigned char* equalize_hist_SWAHE_gpu_mono(
	unsigned char* channel , unsigned int width , unsigned int height ,
	unsigned int window_size ,
	unsigned int pb_width , unsigned int pb_height ){

	window_size += window_size % 2 == 0 ? 1 : 0;
	unsigned int aug_width , aug_height;
	unsigned char *aug_channel , *aug_channel_d, *eq_channel_d;
	prepare( channel , width , height , window_size ,
		&aug_channel , &aug_width , &aug_height ,
		&aug_channel_d , &eq_channel_d );

	// ---- Kernel launch
	dim3 blockDim( 8 , 8 , 1 );
	dim3 gridDim( (int)ceil((float)width/pb_width) ,
				  (int)ceil((float)height/pb_height) );

	equalize_hist_SWAHE_kernel_mono<<< gridDim , blockDim >>>(
		aug_channel_d , aug_width , aug_height ,
		eq_channel_d , width , height ,
		window_size , pb_width , pb_height
	);

	cudaDeviceSynchronize();
	printLastCudaError( "Kernel" );

	cudaFree(aug_channel_d);
	delete aug_channel;

	return get_result_and_free(aug_channel_d, width, height, eq_channel_d);
}
