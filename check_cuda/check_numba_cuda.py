#
# https://medium.com/data-science/cuda-by-numba-examples-1-4-e0d06651612f
#

import numpy as np
import numba
from numba import cuda

print(np.__version__)
print(numba.__version__)
print(cuda.detect())

# Example 1.1: Add scalars
# @cuda.jit
# def add_scalars(a, b, c):
#     c[0] = a + b
#
# dev_c = cuda.device_array((1,), np.float32)
#
# add_scalars[1, 1](2.0, 7.0, dev_c)
#
# c = dev_c.copy_to_host()
# print(f"2.0 + 7.0 = {c[0]}")
# #  2.0 + 7.0 = 9.0


# Example 1.2: Add arrays
# @cuda.jit
# def add_array(a, b, c):
#     i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
#     if i < a.size:
#         c[i] = a[i] + b[i]
#
# N = 20
# a = np.arange(N, dtype=np.float32)
# b = np.arange(N, dtype=np.float32)
# dev_c = cuda.device_array_like(a)
#
# add_array[4, 8](a, b, dev_c)
#
# c = dev_c.copy_to_host()
# print(c)


# Example 1.3: Add arrays with cuda.grid
# N = 20
# a = np.arange(N, dtype=np.float32)
# b = np.arange(N, dtype=np.float32)
# dev_c = cuda.device_array_like(a)
#
# dev_a = cuda.to_device(a)
# dev_b = cuda.to_device(b)
#
# @cuda.jit
# def add_array(a, b, c):
#     i = cuda.grid(1)
#     if i < a.size:
#         c[i] = a[i] + b[i]
#
# add_array[4, 8](dev_a, dev_b, dev_c)
#
# c = dev_c.copy_to_host()
# print(c)


# Example 1.4
N = 1_000_000
a = np.arange(N, dtype=np.float32)
b = np.arange(N, dtype=np.float32)

dev_a = cuda.to_device(a)
dev_b = cuda.to_device(b)
dev_c = cuda.device_array_like(a)

threads_per_block = 256
blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block
# Note that
#     blocks_per_grid == ceil(N / threads_per_block)
# ensures that blocks_per_grid * threads_per_block >= N

@cuda.jit
def add_array(a, b, c):
    i = cuda.grid(1)
    if i < a.size:
        c[i] = a[i] + b[i]

add_array[blocks_per_grid, threads_per_block](dev_a, dev_b, dev_c)

c = dev_c.copy_to_host()
print(np.allclose(a + b, c))

