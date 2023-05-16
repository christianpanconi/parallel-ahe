/*
 * Python wrappers definitions
 */
#ifndef PY_MODULE_HISTOGRAM_EQUALIZATION_HISTOGRAM_EQUALIZATION_PY_WRAPPERS_H_
#define PY_MODULE_HISTOGRAM_EQUALIZATION_HISTOGRAM_EQUALIZATION_PY_WRAPPERS_H_

#include <Python.h>

const int IMG_FMT_RGB = 0;
const int IMG_FMT_YCbCr = 1;

const int EQ_TYPE_MONO = 0;
const int EQ_TYPE_BI = 1;

PyObject* hist_equalization( PyObject* self , PyObject* args );

PyObject* hist_equalization_omp( PyObject* self , PyObject* args );

PyObject* hist_equalization_cuda( PyObject* self , PyObject* args );

// Benchmarking purposes
PyObject* init_cuda_context_py(PyObject* self, PyObject* args);

#endif /* PY_MODULE_HISTOGRAM_EQUALIZATION_HISTOGRAM_EQUALIZATION_PY_WRAPPERS_H_ */
