# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
cnp.import_array()

cdef cnp.int32_t* data_int32 = NULL

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)

cdef extern from "c_arrays.h":
    void getHist(const vector[int]& shape, int*& data) except +

cdef data_to_numpy_array_int32(cnp.int32_t* ptr, cnp.npy_intp N):
    cdef cnp.ndarray[cnp.int32_t, ndim = 1] arr = cnp.PyArray_SimpleNewFromData(1, < cnp.npy_intp* > & N, cnp.NPY_INT32, < cnp.int32_t* > ptr)
    PyArray_ENABLEFLAGS(arr, cnp.NPY_OWNDATA)
    return arr


def get_hist(list shape):
    cdef vector[int] array_size = shape

    getHist( < const vector[int] & > array_size, < int*& > data_int32)

    if data_int32 != NULL:
        numpy_array_1d = data_to_numpy_array_int32(data_int32, np.prod(shape))
        return numpy_array_1d.reshape(shape)
