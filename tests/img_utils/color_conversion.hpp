/**
 * Color space conversion utilities
 *  RGB <===> YCbCr
 */

#ifndef COLOR_CONVERSION_CPP_
#define COLOR_CONVERSION_CPP_

#include <iostream>
#include <cmath>

// Clamp values between 0 and 255
#define CLAMP(x) ( (x) > 255 ? 255 : (x) < 0 ? 0 : (x) )

/** RGB to YCbCr (full range)
 *
 * R in [0,255] , G  in [0,255] , B  in [0,255]
 * Y in [0,255] , Cb in [0,255] , Cr in [0,255]
 *
 * Uses the following formulas implemented with integer calculations:
 *
 * Y  =  0.299 * R + 0.587 * G + 0.114 * B
 * Cb = -0.169 * R - 0.331 * G + 0.500 * B + 128
 * Cr =  0.500 * R - 0.419 * G - 0.081 * B + 128
 *
 */
#define RGB2Y( R , G , B )  CLAMP( (  77 * R + 150 * G +  29 * B + 128 ) >> 8 )
#define RGB2Cb( R , G , B ) CLAMP((( -43 * R -  85 * G + 128 * B + 128 ) >> 8 ) + 128 )
#define RGB2Cr( R , G , B ) CLAMP((( 128 * R - 107 * G -  21 * B + 128 ) >> 8 ) + 128 )

unsigned char ** RGB_to_YCbCr( unsigned char *rgb , unsigned int width , unsigned int height ){

	unsigned int n_pix = width*height;

	unsigned char *Y = new unsigned char [n_pix];
	unsigned char *Cb = new unsigned char [n_pix];
	unsigned char *Cr = new unsigned char [n_pix];

	for( int i=0 ; i < n_pix ; i++ ){
		Y[i]  = RGB2Y(  rgb[i*3] , rgb[i*3+1] , rgb[i*3+2] );
		Cb[i] = RGB2Cb( rgb[i*3] , rgb[i*3+1] , rgb[i*3+2] );
		Cr[i] = RGB2Cr( rgb[i*3] , rgb[i*3+1] , rgb[i*3+2] );
	}

	unsigned char** ycbcr = new unsigned char*[3];
	ycbcr[0] = Y; ycbcr[1] = Cb; ycbcr[2] = Cr;
	return ycbcr;
}

/**
 * Color conversion with padding.
 * (here the for loop cannot be vectorized)
 */
unsigned char ** RGB_to_YCbCr( unsigned char *rgb , unsigned int width , unsigned int height ,
							   unsigned int padding_row_start , unsigned int padding_row_end ,
							   unsigned int padding_col_start , unsigned int padding_col_end ){

	const unsigned int n_pix = width * height;
	const unsigned int tot_w = width + padding_row_start + padding_row_end;
	const unsigned int tot_h = height + padding_col_start + padding_col_end;

	unsigned char *Y = new unsigned char[tot_w*tot_h];
	unsigned char *Cb = new unsigned char[tot_w*tot_h];
	unsigned char *Cr = new unsigned char[tot_w*tot_h];

	int j;
	for( int i=0 ; i < n_pix ; i++ ){
		j = ((i/width) + padding_col_start) * tot_w +
				(i%width) + padding_row_start;
		Y[j]  = RGB2Y(  rgb[i*3] , rgb[i*3+1] , rgb[i*3+2] );
		Cb[j] = RGB2Cb( rgb[i*3] , rgb[i*3+1] , rgb[i*3+2] );
		Cr[j] = RGB2Cr( rgb[i*3] , rgb[i*3+1] , rgb[i*3+2] );
	}

	unsigned char **ycbcr = new unsigned char*[3];
	ycbcr[0] = Y; ycbcr[1] = Cb; ycbcr[2] = Cr;
	return ycbcr;
}

/** YCbCr (full range) to RGB
 *
 * Uses the following formulas implemented with integer calculations:
 *
 * R = 1.000 * Y + 1.400 * (Cr - 128)
 * G = 1.000 * Y - 0.343 * (Cb - 128) - 0.711 * (Cr - 128)
 * B = 1.000 * Y - 1.765 * (Cb - 128)
 *
 */
#define YCbCr2R( Y , Cb , Cr ) CLAMP(( 256 * Y + 358 * (Cr - 128) + 128 ) >> 8 )
#define YCbCr2G( Y , Cb , Cr ) CLAMP(( 256 * Y -  88 * (Cb - 128) - 182 * (Cr - 128) + 128 ) >> 8 )
#define YCbCr2B( Y , Cb , Cr ) CLAMP(( 256 * Y + 451 * (Cb - 128) + 128 ) >> 8 )

unsigned char* YCbCr_to_RGB( unsigned char** __restrict__ ycbcr , unsigned int width , unsigned int height ){
	unsigned int n_pix = width*height;

	unsigned char* __restrict__ rgb = new unsigned char[n_pix*3];
	for( int i=0 ; i < n_pix ; i++ ){
		rgb[i*3]   = YCbCr2R( ycbcr[0][i] , ycbcr[1][i] , ycbcr[2][i] );
		rgb[i*3+1] = YCbCr2G( ycbcr[0][i] , ycbcr[1][i] , ycbcr[2][i] );
		rgb[i*3+2] = YCbCr2B( ycbcr[0][i] , ycbcr[1][i] , ycbcr[2][i] );
	}

	return rgb;
}

unsigned char* YCbCr_to_BGR( unsigned char** __restrict__ ycbcr ,
							 unsigned int width , unsigned int height ){
	unsigned int n_pix = width * height;

	unsigned char * __restrict__ bgr = new unsigned char[n_pix*3];
	for( int i=0 ; i < n_pix ; i++ ){
		bgr[i*3]   = YCbCr2B( ycbcr[0][i] , ycbcr[1][i] , ycbcr[2][i] );
		bgr[i*3+1] = YCbCr2G( ycbcr[0][i] , ycbcr[1][i] , ycbcr[2][i] );
		bgr[i*3+2] = YCbCr2R( ycbcr[0][i] , ycbcr[1][i] , ycbcr[2][i] );
	}

	return bgr;
}


#endif /* COLOR_CONVERSION_CPP_ */
