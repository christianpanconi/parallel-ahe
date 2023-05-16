/**
 * Histogram Equalization module implementation
 */
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL histogramequalization_ARRAY_API
#include <numpy/arrayobject.h>

#include "histogram_equalization_py_module.h"

PyMODINIT_FUNC PyInit__histogram_equalization(void){
	import_array();
	return PyModule_Create(&module);
}





