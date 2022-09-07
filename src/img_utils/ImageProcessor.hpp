#ifndef IMAGEPROCESSOR_HPP_
#define IMAGEPROCESSOR_HPP_

#include <iostream>
#include <turbojpeg.h>

// Exceptions with conveniency macros
void throwException( std::string msg , const char* file , const int line ){
	std::string errMsg(msg);
	errMsg += "\n\tat " + std::string(file) + ": " + std::to_string(line);
	throw errMsg;
}
#define throw_exception( msg ) throwException( msg , __FILE__ , __LINE__ )

void throwTJpegException( std::string msg , tjhandle handle , const char* file , const int line ){
	std::string errMsg(msg);
	errMsg += ": " + std::string( tjGetErrorStr2(handle) ) + "\n\tat" + std::string(file) + ": " + std::to_string(line);
	throw errMsg;
}
#define throw_tj_exception( msg , handle ) throwTJpegException( msg , handle , __FILE__ , __LINE__ )

/**
 * Utility class to load and save images using libjpeg-turbo.
 */
class ImageProcessor {

public:
	tjhandle tjInstance;

	ImageProcessor() : tjInstance(0){ }

	void init_decompress(){
		this->clear_tjhandle();
		if( (this->tjInstance = tjInitDecompress()) == nullptr )
			throw_tj_exception( "Failed decompress initialization" , this->tjInstance );
	}

	void clear_tjhandle(){
		if( this->tjInstance != 0 ){
			tjDestroy(tjInstance);
			tjInstance = 0;
		}
	}

	~ImageProcessor(){
		this->clear_tjhandle();
	}

	unsigned char** read_jpeg( std::string infile , int *w , int *h , int *subs){
		int width, height;
		char *inFormat;
		FILE *jpegFile = nullptr;
		unsigned char* jpegBuf;
		int pixelFormat = TJPF_UNKNOWN;

		long size;
		int inSubsamp, inColorspace;
		unsigned long jpegSize;

		// Read the JPEG file into memory
		if( (jpegFile = fopen( infile.c_str() , "rb")) == nullptr ){
			std::string errMsg("Failed to open file "); errMsg += infile;
			throw_exception(errMsg);
		}

		// Determine file size
		if( fseek(jpegFile , 0 , SEEK_END) < 0 || ((size = ftell(jpegFile)) < 0 ) ||
			fseek(jpegFile , 0 , SEEK_SET) < 0 ){
			throw_exception( "Error determining input file size" );
		}
		if( size == 0 ) throw_exception( "Input file contains no data" );

		jpegSize = (unsigned int) size;
		// Allocate jpeg data buffer
		if( (jpegBuf = (unsigned char*)tjAlloc(jpegSize)) == nullptr )
			throw_exception( "Failed jpeg buffer allocation" );
		// Read file
		if( fread(jpegBuf , jpegSize, 1, jpegFile) < 1 )
			throw_exception( "Error reding input file" );
		fclose(jpegFile); jpegFile = nullptr;

		// Init decompressor
		this->init_decompress();

		// Decompress header
		if( tjDecompressHeader3(this->tjInstance, jpegBuf, jpegSize, &width, &height,
								&inSubsamp, &inColorspace) < 0 )
			throw_tj_exception( "Failed header decompression" , this->tjInstance );

		// Allocate YCbCr plane buffers
		unsigned char** yuvBuf = (unsigned char**) malloc(3*sizeof(unsigned char*) );
		int strides[3] = { 0 , 0 , 0 };
		if( (yuvBuf[0] = (unsigned char*)tjAlloc(tjPlaneSizeYUV( 0 , width, strides[0], height, inSubsamp ))) == nullptr )
			throw_exception( "Error allocating Y plane buffer" );
		if( (yuvBuf[1] = (unsigned char*)tjAlloc(tjPlaneSizeYUV( 1 , width, strides[1], height, inSubsamp ))) == nullptr )
			throw_exception( "Error allocating Cb plane buffer" );
		if( (yuvBuf[2] = (unsigned char*)tjAlloc(tjPlaneSizeYUV( 2 , width, strides[2], height, inSubsamp ))) == nullptr )
			throw_exception( "Error allocating Cr plane buffer" );

		// Decompress
		int flags = 0;
		if( tjDecompressToYUVPlanes( this->tjInstance , jpegBuf , jpegSize , yuvBuf ,
				          	  	  	 width , strides , height , flags ) < 0 )
			throw_tj_exception( "Failed JPEG decompression" , this->tjInstance );
		*w = width;
		*h = height;
		*subs = inSubsamp;

		tjFree(jpegBuf); jpegBuf = nullptr;
		return yuvBuf;
	}

	unsigned char* read_uncompressed( std::string infile , unsigned int *w , unsigned int *h , int* pixFmt){
		int width, height;
		unsigned char* imgBuf;
		int pixelFormat = TJPF_UNKNOWN;

		int flags = 0;
		imgBuf = tjLoadImage( infile.c_str() , &width , 1 , &height ,
						     	 	 	 	 &pixelFormat , flags );

		if( imgBuf == nullptr ){
			std::string err( "Failed to load image: " ); err += infile;
			throw_exception( err );
		}
		*w = width;
		*h = height;
		*pixFmt = pixelFormat;

		return imgBuf;
	}

	void save_uncompressed( std::string outfile , unsigned char* buffer , int width , int height , int pixFmt ){
		int res = tjSaveImage( outfile.c_str() , buffer , width , 0 , height , pixFmt , 0 );
		if( res != 0 ){
			std::cerr << "Failed to save image: " << outfile << std::endl;
		}
	}

};



#endif /* IMAGEPROCESSOR_HPP_ */
