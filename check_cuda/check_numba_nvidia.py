import math
import numpy as np
import numba
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray

# Just to remember if the data in host (computer) or device (graphics card)
# h_<var> d_<var>
# <var>_h, <var>_d
# <var>_host, <var>_device

print(np.__version__)
print(numba.__version__)
print(cuda.detect())

# cuda.gridDim   .x .y .z .w
# cuda.blockDim
# cuda.blockIdx
# cuda.threadIdx

@cuda.jit
def increment_by_one(an_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < an_array.size:  # Check array boundaries
        an_array[pos] += 1


@cuda.jit
def increment_a_1D_array(an_array):
    pos = cuda.grid(1)
    if pos < an_array.size:
        an_array[pos] += 1


@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1

# ---------------------------------------------------------------------------

an_array: np.ndarray = np.zeros(50, dtype=np.float16)
an_array: DeviceNDArray = cuda.to_device(an_array)

threadsperblock = 16
blockspergrid = math.ceil(an_array.shape[0] / threadsperblock)
increment_a_1D_array[blockspergrid, threadsperblock](an_array)

an_array: np.ndarray = an_array.copy_to_host()
print(an_array)

# ---------------------------------------------------------------------------

an_array: np.ndarray = np.zeros((50, 50), dtype=np.float16)
an_array: DeviceNDArray = cuda.to_device(an_array)

threadsperblock = (16, 16)
blockspergrid_x = math.ceil(an_array.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(an_array.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
increment_a_2D_array[blockspergrid, threadsperblock](an_array)


an_array: np.ndarray = an_array.copy_to_host()
print(an_array)