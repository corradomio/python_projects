import numpy as np
cimport numpy as cnp

def process_array(cnp.ndarray[cnp.float64_t, ndim=2] arr):
    # This provides fast, C-level access to array elements
    cdef int i, j
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] += 1.0
