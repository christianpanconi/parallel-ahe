#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <cstring>
#include <vector>
#include <map>
#include <algorithm>

#include <random>
#include <sched.h>
#include <benchmark/benchmark.h> // Google benchmark

// custom utils
#include "launch_utils.hpp"
#include "img_utils/ImageProcessor.hpp"
#include "img_utils/color_conversion.hpp"
// equalization implementations
//#include "equalization_cpu/equalization.hpp"			// Sequential
//#include "equalization_cpu/equalization_parallel.hpp"	// OpenMP
//#include "equalization_cuda/equalization_gpu.cuh"		// CUDA
#include "../lib/histogram_equalization/include/equalization.hpp" // Sequential
#include "../lib/histogram_equalization/include/equalization_parallel.hpp"	// OpenMP
#include "../lib/histogram_equalization/include/equalization_gpu.cuh"		// CUDA


// UTILS:
//
// Cache flushing
int *volatile cfb;
volatile int sumb;

void flush_the_cache(){
	const size_t flush_size = 16*1024*1024;
	int* flush_buffer = new int[flush_size]; // 64 Mb buffer
	memset( flush_buffer , 1 , flush_size );
	cfb = flush_buffer;

	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<unsigned long long> dis(0, flush_size-1);
	unsigned long int read_its = 30*(16*1024*1024/64);
	unsigned long long int sum=0;
	for( int i=0 ; i < read_its ; i++ ){ // random read loop
		sum += flush_buffer[dis(gen)];
	}
	sumb=sum;
	delete flush_buffer;
}
//
// CPU affinity
void setCPUAffinity( unsigned int cpu_affinity ){
	if( cpu_affinity < 0 ) return;
	cpu_set_t mask;
	int status;
	CPU_ZERO(&mask);
	CPU_SET( cpu_affinity , &mask );
	status = sched_setaffinity( 0 , sizeof(mask) , &mask );
	if( status != 0 )
		std::cerr << "Failed to set cpu affinity" << std::endl;
}
// --------

/**
 * Command line args parsing and launch configs.
 */
typedef struct LaunchArgs {
//	EqualizationArgs eq;
	EqArgs::Configs eq;

	bool parallel=false;
	unsigned int n_threads=0;
	unsigned int benchmark_its=0;

	bool gpu=false;
//	unsigned int pbh=64;
//	unsigned int pbw=64;
	// automatically determined
	unsigned int pbh=0;
	unsigned int pbw=0;

	bool help=false;

	std::string to_string(){
		std::string str("Launch args:");
		str += eq.to_string() +
				"\n\tn threads:         " + std::to_string(n_threads) +
				"\n\tbenchmark its:     " + std::to_string(benchmark_its);
		return str;
	}
} LaunchArgs;

LaunchArgs parseCmdLineArgs(int argc , char **argv, std::vector<int> *multiple_ws=nullptr ){
	LaunchArgs largs;
	largs.eq = EqArgs::parseCmdEqualizationArgs(argc, argv);

	std::vector<char *> args;
	std::copy( &(argv[0]) , &(argv[argc]) , std::back_inserter(args) );

	int i = 0;
	std::string arg;
	while( i < args.size() ){
		arg = args[i];
		if( arg == "-t" || arg == "--threads" ){
			largs.n_threads=std::stoi(argv[i+1]);
			if( largs.n_threads > 1 ){
				largs.parallel=true;
				largs.gpu = false;
			} else {
				largs.parallel=false;
				largs.n_threads=0;
				std::cout << "Ignored '-t' argument. Requested: " << argv[i+1] << " threads?!." << std::endl;
			}
			i++;
		}
		if( arg == "-pbw" ){
			largs.pbw = std::stoi(argv[i+1]);
			i++;
		}
		if( arg == "-pbh" ){
			largs.pbh = std::stoi(argv[i+1]);
			i++;
		}
		if( arg == "--gpu" ){
			largs.gpu=true;
			if( largs.parallel ){
				largs.parallel = false;
				largs.n_threads = 0;
			}
		}
		if( arg == "-bmits" || arg == "--benchmark-its" ){
			largs.benchmark_its = std::stoi(argv[i+1]);
			if( largs.benchmark_its < 1 ){
				largs.benchmark_its = 0;
				std::cout << "Ignored '-bmits' argument. Requested: " << argv[i+1] << " benchmark iterations?!" << std::endl;
			}
			i++;
		}
		if( arg == "-mws" || arg == "--multiple-window-size" ){
			std::string mws(argv[i+1]);
			std::stringstream mws_ss(mws);
			std::string token;

			std::vector<int> *mws_v;// = new std::vector<int>();
			if( multiple_ws != nullptr )
				mws_v = multiple_ws;
			else
				mws_v = new std::vector<int>();

			while( std::getline( mws_ss , token , ',') ){
				mws_v->push_back( std::stoi(token) );
			}
			largs.eq.window_size = (*mws_v)[0];

			if( multiple_ws == nullptr )
				delete mws_v;
			i++;
		}
		if( arg == "-h" || arg == "--help" )
			largs.help = true;
		i++;
	}

	return largs;
}

const char *usage = R"(
This executable evaluates Adaptive Histogram Equalization implementations by running
benchmarks using Google Benchmark (https://github.com/google/benchmark).
Google Benchmark command line arguments are supported, some useful ones:
--benchmark_repetitions=<int>                Repeat every benchmark the given number of times.
--benhcmark_out_format=<json|console|csv>    Specifies output type.
--benchmark_out=<filename>                   Specified the output file (for JSON and CSV only).

Basic usage:   
	benchmark-hist-equalization -f <input img> -ws <window_size>

Example usage with Google benchmark args:
	benchmark-hist-equalization -f <input img> -ws <window_size> \
                                --benchmark_repetitions=10 \
                                --benchmark_out_format=csv \
                                --benchmark_out=results.csv
(NOTE: Google Benchmarks args values are passed with "=")
)";
const char *args_usage = R"(
Benchmark args:
-bmits | --benchmark-its <int>                          The number of iterations per benchmark.
-mws   | --multiple-window-size <int[,int[,int]...]>    Performs benchmarks for every window size
                                                        specified in the list.

OpenMP specific args:
-t | --threads <int>        Uses OpenMP implementation with the 
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

/**
 * Single-core/sequential benchmark
 */
typedef unsigned char* (*EqFunc)(unsigned char* , unsigned int, unsigned int, unsigned int, unsigned int);
typedef std::map<std::string , EqFunc> EqFuncMap;
EqFuncMap eq_f_map;

static void BM_hist_equalization(benchmark::State& state, LaunchArgs args){
	auto fit = eq_f_map.find(args.eq.equalization_type);
	if( fit == eq_f_map.end() ){
		std::string err( "Unknown equalization type: \"");
		err += args.eq.equalization_type + "\"";
		state.SkipWithError( err.c_str() );
	}
	auto equalize_hist = *fit->second;

	unsigned int window_size;
	if( state.range(0) > 0 ){ // use ->Arg(0) to use window size from LaunchArgs
		window_size = state.range(0);
	}else{
		window_size = args.eq.window_size;
	}
	state.counters["ws"] = window_size;

	unsigned char* img;
	int pixfmt;
	unsigned int width, height;
	ImageProcessor imp;

	try{
		img = imp.read_uncompressed( args.eq.img_file , &width, &height, &pixfmt );
	}catch( std::string& ex ){
		std::cerr << ex << std::endl;
		return;
	}

	unsigned char** ycbcr;
	unsigned char* yEq;

	for( auto _ : state ){ // Benchmark loop
		state.PauseTiming();
		ycbcr = RGB_to_YCbCr( img , width, height );
		state.ResumeTiming();

		yEq = equalize_hist( ycbcr[0], width, height, window_size , 256 );

		state.PauseTiming();
		delete yEq;
		delete ycbcr[0]; delete ycbcr[1]; delete ycbcr[2];
		delete ycbcr;
		state.ResumeTiming();
	}

	tjFree( img ); img=nullptr;
}

/**
 * Multi-core/parallel benchmark
 */
typedef unsigned char* (*ParallelEqFunc)
	(unsigned char*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int );
typedef std::map<std::string , ParallelEqFunc> ParallelEqFuncMap;
ParallelEqFuncMap par_eq_f_map;

static void BM_hist_equalization_parallel(benchmark::State& state, LaunchArgs args){
	auto fit = par_eq_f_map.find(args.eq.equalization_type);
	if( fit == par_eq_f_map.end() ){
		std::string err( "Unknown equalization type: \"");
		err += args.eq.equalization_type + "\"";
		state.SkipWithError( err.c_str() );
	}
	auto equalize_hist = *fit->second;

	unsigned int window_size;
	if( state.range(0) > 0 ){ // use ->Arg(0) to use window size from LaunchArgs
		window_size = state.range(0);
	}else{
		window_size = args.eq.window_size;
	}
	state.counters["ws"] = window_size;
	state.counters["threads"] = args.n_threads;

	unsigned char* img;
	int pixfmt;
	unsigned int width, height;
	ImageProcessor imp;

	try{
		img = imp.read_uncompressed( args.eq.img_file , &width, &height, &pixfmt );
	}catch( std::string& ex ){
		std::cerr << ex << std::endl;
		return;
	}

	unsigned char** ycbcr;
	unsigned char* yEq;

	for( auto _ : state ){ // Benchmark loop
		state.PauseTiming();
		ycbcr = RGB_to_YCbCr( img , width, height );
		state.ResumeTiming();

		yEq = equalize_hist( ycbcr[0], width, height, window_size ,
				256 , args.n_threads );

		state.PauseTiming();
		delete yEq;
		delete ycbcr[0]; delete ycbcr[1]; delete ycbcr[2];
		delete ycbcr;
		state.ResumeTiming();
	}

	tjFree( img ); img=nullptr;
}

/**
 * GPU benchmark
 */
typedef unsigned char* (*GPUEqFunc)(unsigned char*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);
typedef std::map<std::string , GPUEqFunc> GPUEqFuncMap;
GPUEqFuncMap gpu_eq_f_map;

static void BM_hist_equalization_gpu(benchmark::State& state , LaunchArgs args ){
	auto fit = gpu_eq_f_map.find(args.eq.equalization_type);
	if( fit == gpu_eq_f_map.end() ){
		std::string err( "Unknown equalization type: \"");
		err += args.eq.equalization_type + "\"";
		state.SkipWithError( err.c_str() );
	}
	auto equalize_hist = *fit->second;

	unsigned int window_size;
	if( state.range(0) > 0 ){ // use ->Arg(0) to use window size from LaunchArgs
		window_size = state.range(0);
	}else{
		window_size = args.eq.window_size;
	}
	state.counters["ws"] = window_size;
	state.counters["pbw"] = args.pbw;
	state.counters["pbh"] = args.pbh;

	unsigned char* img;
	int pixfmt;
	unsigned int width, height;
	ImageProcessor imp;

	try{
		img = imp.read_uncompressed( args.eq.img_file , &width, &height, &pixfmt );
	}catch( std::string& ex ){
		std::cerr << ex << std::endl;
		return;
	}

	if( args.pbw == 0 || args.pbh == 0 ){
		unsigned int pbs = determine_pbs(width, height);
		state.counters["pbw"] = pbs;
		state.counters["pbh"] = pbs;
	}

	// Initialize context before benchmark loop
	// to not account for context initialization overhead
	init_cuda_context();

	unsigned char** ycbcr;
	unsigned char* yEq;

	for( auto _ : state ){ // Benchmark loop
		state.PauseTiming();
		ycbcr = RGB_to_YCbCr( img , width, height );
		state.ResumeTiming();

		yEq = equalize_hist( ycbcr[0], width, height, window_size ,
				args.pbw , args.pbh);

		state.PauseTiming();
		delete yEq;
		delete ycbcr[0]; delete ycbcr[1]; delete ycbcr[2];
		delete ycbcr;
		state.ResumeTiming();
	}

	tjFree( img ); img=nullptr;
}

// MAIN
int main(int argc, char **argv) {
//	setCPUAffinity( 0 );

	eq_f_map.emplace( EqArgs::EQ_TYPE_MONO , equalize_hist_SWAHE_mono );
	eq_f_map.emplace( EqArgs::EQ_TYPE_BI , equalize_hist_SWAHE_bi );
	par_eq_f_map.emplace( EqArgs::EQ_TYPE_MONO , equalize_hist_SWAHE_omp_mono );
	par_eq_f_map.emplace( EqArgs::EQ_TYPE_BI , equalize_hist_SWAHE_omp_bi );
	gpu_eq_f_map.emplace( EqArgs::EQ_TYPE_MONO , equalize_hist_SWAHE_gpu_mono );
	gpu_eq_f_map.emplace( EqArgs::EQ_TYPE_BI , equalize_hist_SWAHE_gpu_bi );

	std::vector<int> window_size_v;
	LaunchArgs args = parseCmdLineArgs(argc, argv , &window_size_v );
	if( args.help ){
		std::cout << usage << std::endl << EqArgs::usage << args_usage << std::endl;
		return EXIT_SUCCESS;
	}

	auto bm_lambda = [args](benchmark::State& st){
		if( args.parallel )
			BM_hist_equalization_parallel(st, args); // OpenMP
		else if( args.gpu )
			BM_hist_equalization_gpu(st, args);		 // CUDA
		else
			BM_hist_equalization(st, args);			 // Sequential
	};

	if( window_size_v.empty() )
		window_size_v.push_back( args.eq.window_size );

	// Register benchmarks
	for( int i=0 ; i < window_size_v.size() ; i++ ){
		std::string bm_name("equalize_hist/");
		bm_name += args.eq.equalization_type;
		benchmark::internal::Benchmark *b = benchmark::RegisterBenchmark( bm_name.c_str() , bm_lambda );
		b->Unit(benchmark::kMillisecond)->UseRealTime();
		if( args.benchmark_its > 0 )
			b->Iterations( args.benchmark_its );
		b->Arg(window_size_v[i]);
	}

	// Init and run benchmarks
	benchmark::Initialize( &argc , argv );

	benchmark::AddCustomContext( "image" , args.eq.img_file );
	benchmark::AddCustomContext( "equalization" , args.eq.equalization_type );
	benchmark::AddCustomContext( "parallel" , (args.parallel?"true":"false") );
	benchmark::AddCustomContext( "n threads" , std::to_string(args.n_threads) );

	benchmark::RunSpecifiedBenchmarks();
	benchmark::Shutdown();
}


