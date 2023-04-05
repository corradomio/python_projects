import cupy as cp
import numpy as np
from pprint import pprint


squared_diff = cp.ElementwiseKernel(
   'float32 x, float32 y',
   'float32 z',
   'z = (x - y) * (x - y)',
   'squared_diff')

x = cp.arange(10, dtype=np.float32).reshape(2, 5)
y = cp.arange(5, dtype=np.float32)
pprint(squared_diff(x, y))
pprint(squared_diff(x, 5))

z = cp.empty((2, 5), dtype=np.float32)
pprint(squared_diff(x, y, z))
