/**
 * Histogram equalization module definitions
 */
#ifndef PY_MODULE_HISTOGRAM_EQUALIZATION_HISTOGRAM_EQUALIZATION_PY_MODULE_H_
#define PY_MODULE_HISTOGRAM_EQUALIZATION_HISTOGRAM_EQUALIZATION_PY_MODULE_H_

#include <Python.h>

#include <stdio.h>

#include "histogram_equalization_py_wrappers.h"

// Module def
//	{ "method_name" , function_pointer , METH_VARARGS , "Docstring" }
static PyMethodDef methods[] = {
	{ "hist_equalization" , hist_equalization , METH_VARARGS , "Sequential implementation." },
	{ "hist_equalization_omp" , hist_equalization_omp , METH_VARARGS , "OpenMP implementation." },
	{ "hist_equalization_cuda" , hist_equalization_cuda , METH_VARARGS , "CUDA implementation." },
	{ "init_cuda_context" , init_cuda_context_py , METH_VARARGS , "" },
	{NULL, NULL, 0, NULL}
};

static PyModuleDef module = {
	PyModuleDef_HEAD_INIT ,
	"_histogram_equalization" ,
	"_histogram_equalization" ,
	-1 ,
	methods
};

PyMODINIT_FUNC PyInit__histogram_equalization(void);

#endif /* PY_MODULE_HISTOGRAM_EQUALIZATION_HISTOGRAM_EQUALIZATION_PY_MODULE_H_ */
