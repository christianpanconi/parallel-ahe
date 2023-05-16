#include <iostream>
#include <map>

// custom utils
#include "img_utils/ImageProcessor.hpp"
#include "img_utils/color_conversion.hpp"
//#include "img_utils/image_utils.hpp"
#include "launch_utils.hpp"
#include "timer/timer.hpp"
// equalization implementations
//#include "equalization_cpu/equalization.hpp"			// Sequential
//#include "equalization_cpu/equalization_parallel.hpp"	// OpenMP
//#include "equalization_cuda/equalization_gpu.cuh"		// CUDA
#include "../lib/histogram_equalization/include/equalization.hpp"			// Sequential
#include "../lib/histogram_equalization/include/equalization_parallel.hpp"	// OpenMP
#include "../lib/histogram_equalization/include/equalization_gpu.cuh"		// CUDA

typedef struct LaunchArgs{
	EqArgs::Configs eq;

	// OpenMP args
	bool parallel=false;
	unsigned int n_threads=0;

	// CUDA args
	bool gpu=false;
//	unsigned int pbw=64;
//	unsigned int pbh=64;
	// automatically determined
	unsigned int pbh=0;
	unsigned int pbw=0;

	// Ordinary HE
	bool ordinary_he=false;

	// Output file
	std::string out_img="";

	// help
	bool help=false;

	std::string to_string(){
		std::string str("Launch args:");
		str += eq.to_string();
		if( parallel ){
		   str += std::string("\n\tOpenMP") +
			      "\n\tn threads:         " + std::to_string(n_threads);
		}else if( gpu ){
			str += std::string("\n\tCUDA") +
				   "\n\tpbw:              " + std::to_string(pbw) +
				   "\n\tpbh:              " + std::to_string(pbh);
		}
		return str;
	}
} LaunchArgs;

LaunchArgs parseCmdLineArgs( int argc , char **argv ){
	LaunchArgs largs;
	largs.eq = EqArgs::parseCmdEqualizationArgs( argc , argv );

	std::vector<char *> args;
	std::copy( &(argv[0]) , &(argv[argc]) , std::back_inserter(args) );

	int i=0;
	std::string arg;
	while( i < args.size() ){
		arg = args[i];
		if( arg == "-t" || arg == "--threads" ){
			largs.n_threads=std::stoi(argv[i+1]);
			if( largs.n_threads > 1 ){
				largs.parallel=true;
				if( largs.gpu )
					largs.gpu=false;
			} else {
				largs.parallel=false;
				largs.n_threads=0;
				std::cout << "Ignored '-t' argument. Requested: " << argv[i+1] << " threads?!." << std::endl;
			}
			i++;
		}
		if( arg == "--gpu" ){
			largs.gpu = true;
			if( largs.parallel ){
				largs.parallel=false;
				largs.n_threads=0;
			}
		}
		if( arg == "-pbw" ){
			largs.pbw = std::stoi(argv[i+1]);
			i++;
		}
		if( arg == "-pbh" ){
			largs.pbh = std::stoi(argv[i+1]);
			i++;
		}
		if( arg == "-o" || arg == "--output-image" ){
			largs.out_img = argv[i+1];
			i++;
		}
		if( arg == "--ordinary-he" ){
			largs.ordinary_he = true;
		}
		if( arg == "-h" || arg == "--help" )
			largs.help = true;
		i++;
	}

	return largs;
}

const char *usage = R"(
Output args:
-o <file path>          Save the output to the specified file.

OpenMP specific args:
-t | --threads <int>    Uses OpenMP implementation with the 
                        specified number of threads.

CUDA specific args:
--gpu        Uses CUDA implementation.
-pbw         Number of columns in the processed image
             region for each block.
-pbh         Number of rows in the processed image
             region for each block.

In case of conflicitng args (like -t and --gpu) the last one provided in
the command is used.
)";

int main(int argc, char **argv) {
	LaunchArgs args = parseCmdLineArgs( argc , argv );
	if( args.help ){ // print usage
		std::cout << R"(
Bsic usage:        hist-equalization -f <input_img> -ws <window_size> -o <output_image>
)" << std::endl << EqArgs::usage << usage << std::endl;//<< EqArgsC::usage << usage << std::endl;
		return EXIT_SUCCESS;
	}
	std::cout << args.to_string() << std::endl;

	if( args.eq.equalization_type != EqArgs::EQ_TYPE_BI && args.eq.equalization_type != EqArgs::EQ_TYPE_MONO ){
		std::cerr << "Unknown equalization type: '" << args.eq.equalization_type << "'" << std::endl;
		return EXIT_FAILURE;
	}

	unsigned char *img;
	unsigned int width , height;
	int pixfmt;
	ImageProcessor imp;

	try{ // Load image
		img = imp.read_uncompressed( args.eq.img_file , &width , &height , &pixfmt );
	}catch( std::string& ex ){
		std::cerr << ex << std::endl;
		return EXIT_FAILURE;
	}

	if( args.gpu && (args.pbw==0 || args.pbh==0) ){
		unsigned int pbs = determine_pbs(width, height);
		std::cout << "\tdetermined pbw: " << pbs << std::endl
				  << "\tdetermined pbh: " << pbs << std::endl;
	}

	unsigned char** ycbcr = RGB_to_YCbCr( img , width , height );
	unsigned char* yEq;

	// Initialize context before benchmark loop
	// to not account for context initialization overhead
	init_cuda_context();

	c8::Timer t;
	t.start();
	if( args.ordinary_he ){
		yEq = equalize_hist( ycbcr[0], width * height );
	}else{
		if( args.eq.equalization_type == EqArgs::EQ_TYPE_BI ){
			if( args.parallel ){
				yEq = equalize_hist_SWAHE_omp_bi( ycbcr[0] , width , height , args.eq.window_size , 256 , args.n_threads );
			}else if( args.gpu ) {
				yEq = equalize_hist_SWAHE_gpu_bi( ycbcr[0], width, height, args.eq.window_size ,
					args.pbw , args.pbh );
			}else
				yEq = equalize_hist_SWAHE_bi( ycbcr[0] , width , height , args.eq.window_size );
		}else if( args.eq.equalization_type == EqArgs::EQ_TYPE_MONO ){
			if( args.parallel ){
				yEq = equalize_hist_SWAHE_omp_mono( ycbcr[0], width, height, args.eq.window_size , 256 , args.n_threads );
			}else if( args.gpu ){
				yEq = equalize_hist_SWAHE_gpu_mono( ycbcr[0] , width , height , args.eq.window_size ,
					args.pbh , args.pbw);
			}else
				yEq = equalize_hist_SWAHE_mono( ycbcr[0], width, height, args.eq.window_size );
		}
	}
	t.stop();

	delete ycbcr[0];
	ycbcr[0] = yEq;

	std::cout << "Elapsed: " << t.elapsed<std::chrono::milliseconds>() << " ms" << std::endl;

	if( !args.out_img.empty() ){
		unsigned char* rgb = YCbCr_to_RGB(ycbcr, width, height);
		imp.save_uncompressed( args.out_img , rgb , width, height, pixfmt );
		std::cout << "Saved equalized image: " << args.out_img << std::endl;
		delete rgb;
	}

	tjFree( img ); img=nullptr;
	delete ycbcr[0]; delete ycbcr[1]; delete ycbcr[2];
	delete ycbcr;
	return EXIT_SUCCESS;
}







