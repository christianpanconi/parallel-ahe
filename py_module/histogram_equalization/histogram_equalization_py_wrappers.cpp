/**
 * Python wrappers implementation.
 */
#include <cstdio>
#include <iostream>

#include <Python.h>
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL histogramequalization_ARRAY_API
#include <numpy/arrayobject.h>

#include <chrono>

#include "histogram_equalization_py_wrappers.h"
#include "color_conversion.hpp"
#include "timer.hpp"

#include "../../lib/histogram_equalization/include/equalization.hpp"
#include "../../lib/histogram_equalization/include/equalization_parallel.hpp"
#include "../../lib/histogram_equalization/include/equalization_gpu.cuh"

PyObject* init_cuda_context_py(PyObject* self, PyObject* args){
	init_cuda_context();
	return Py_None;
}

unsigned char** get_YCbCr_image( PyArrayObject* img_ndarray, unsigned int imgw, unsigned int imgh, int img_fmt ){
	// Parse/convert data
	// img_ndarray is assumed np.uint8
	unsigned char* img = static_cast<unsigned char *>(PyArray_DATA(img_ndarray));
	unsigned char** ycbcr;
	if( img_fmt == IMG_FMT_RGB ){
		ycbcr = RGB_to_YCbCr(img, imgw, imgh);
	}
	if( img_fmt == IMG_FMT_YCbCr ){ // TODO: add support
		std::cerr << "Missing support for direct YCbCr input image." << std::endl;
	}

	return ycbcr;
}

PyObject* build_result( unsigned char** ycbcr , unsigned int imgw , unsigned int imgh ,
		int img_fmt , unsigned long long eq_time ){
		unsigned char* out;
		if( img_fmt == IMG_FMT_RGB )
			out = YCbCr_to_RGB( ycbcr , imgw , imgh );
		if( img_fmt == IMG_FMT_YCbCr ){
			std::cerr << "Missing support for YCbCr output." << std::endl;
			return Py_None;
		}

		npy_intp* out_dims = new npy_intp[3];
		out_dims[0] = imgh;
		out_dims[1] = imgw;
		out_dims[2] = 3;
		PyObject* out_ndarray = PyArray_SimpleNewFromData(
			3 , out_dims , NPY_TYPES::NPY_UBYTE , out );

		PyObject *result = PyDict_New();
		PyDict_SetItemString( result , "equalized" , out_ndarray );
		PyDict_SetItemString( result , "equalization_time" , PyLong_FromUnsignedLongLong(eq_time) );

		return result;
}

// (img , window_size , n_values , eq_type , img_fmt)
PyObject* hist_equalization( PyObject* self, PyObject* args ){
	PyArrayObject* img_ndarray;
	unsigned int window_size, n_values;
	int eq_type, img_fmt;

	if( !PyArg_ParseTuple( args , "O!IIii" ,
			&PyArray_Type , &img_ndarray ,
			&window_size , &n_values ,
			&eq_type , &img_fmt ) )
		return nullptr;

	npy_intp* img_dims = PyArray_DIMS(img_ndarray);
	unsigned int imgh = img_dims[0];
	unsigned int imgw = img_dims[1];

	unsigned char** ycbcr = get_YCbCr_image(img_ndarray, imgw, imgh, img_fmt);

	// run histogram equalization
	unsigned char* yEq;
	c8::Timer timer;
	timer.start();
	if( eq_type == EQ_TYPE_MONO )
		yEq = equalize_hist_SWAHE_mono( ycbcr[0] , imgw , imgh , window_size , n_values );
	if( eq_type == EQ_TYPE_BI )
		yEq = equalize_hist_SWAHE_bi( ycbcr[0] , imgw , imgh , window_size , n_values );
	timer.stop();

	delete ycbcr[0];
	ycbcr[0] = yEq;

	PyObject* result = build_result( ycbcr , imgw , imgh , img_fmt ,
		timer.elapsed<std::chrono::milliseconds>() );
	delete ycbcr[0]; delete ycbcr[1]; delete ycbcr[2];
	delete ycbcr;
	return result;
}

// (img , window_size , n_values , eq_type , img_fmt , n_threads)
PyObject* hist_equalization_omp( PyObject* self , PyObject* args ){
	PyArrayObject* img_ndarray;
	unsigned int window_size, n_values, n_threads;
	int eq_type, img_fmt;

	if( !PyArg_ParseTuple( args , "O!IIiiI" ,
			&PyArray_Type , &img_ndarray ,
			&window_size , &n_values ,
			&eq_type , &img_fmt ,
			&n_threads) )
		return nullptr;

	npy_intp* img_dims = PyArray_DIMS(img_ndarray);
	unsigned int imgh = img_dims[0];
	unsigned int imgw = img_dims[1];

	unsigned char** ycbcr = get_YCbCr_image(img_ndarray, imgw, imgh, img_fmt);

	// run histogram equalization
	unsigned char* yEq;
	c8::Timer timer;
	timer.start();
	if( eq_type == EQ_TYPE_MONO )
		yEq = equalize_hist_SWAHE_omp_mono( ycbcr[0] , imgw , imgh , window_size , n_values , n_threads );
	if( eq_type == EQ_TYPE_BI )
		yEq = equalize_hist_SWAHE_omp_bi( ycbcr[0] , imgw , imgh , window_size , n_values , n_threads );
	timer.stop();

	delete ycbcr[0];
	ycbcr[0] = yEq;

	PyObject* result = build_result( ycbcr , imgw , imgh , img_fmt ,
		timer.elapsed<std::chrono::milliseconds>() );
	delete ycbcr[0]; delete ycbcr[1]; delete ycbcr[2];
	delete ycbcr;
	return result;

}

// (img , window_size , eq_type , img_fmt , pbw , pbh)
PyObject* hist_equalization_cuda( PyObject* self , PyObject* args ){
	PyArrayObject* img_ndarray;
	unsigned int window_size, pbw, pbh;
	int eq_type, img_fmt;

	if( !PyArg_ParseTuple( args , "O!IiiII" ,
			&PyArray_Type , &img_ndarray ,
			&window_size ,
			&eq_type , &img_fmt ,
			&pbw , &pbh ) )
		return nullptr;

	npy_intp* img_dims = PyArray_DIMS(img_ndarray);
	unsigned int imgh = img_dims[0];
	unsigned int imgw = img_dims[1];

	unsigned char** ycbcr = get_YCbCr_image(img_ndarray, imgw, imgh, img_fmt);

	// run histogram equalization
	unsigned char* yEq;
	c8::Timer timer;
	timer.start();
	if( eq_type == EQ_TYPE_MONO )
		yEq = equalize_hist_SWAHE_gpu_mono( ycbcr[0] , imgw , imgh , window_size , pbw , pbh );
	if( eq_type == EQ_TYPE_BI )
		yEq = equalize_hist_SWAHE_gpu_bi( ycbcr[0] , imgw , imgh , window_size , pbw , pbh );
	timer.stop();

	delete ycbcr[0];
	ycbcr[0] = yEq;

	PyObject* result = build_result( ycbcr , imgw , imgh , img_fmt ,
		timer.elapsed<std::chrono::milliseconds>() );
	delete ycbcr[0]; delete ycbcr[1]; delete ycbcr[2];
	delete ycbcr;
	return result;
}
