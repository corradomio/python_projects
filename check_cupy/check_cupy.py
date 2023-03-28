import numpy as np
import cupy as cp

print(1)
x_gpu = cp.array([1, 2, 3])

print(2)
x_cpu = np.array([1, 2, 3])
l2_cpu = np.linalg.norm(x_cpu)
print(l2_cpu)
print(3)
x_gpu = cp.array([1, 2, 3])
l2_gpu = cp.linalg.norm(x_gpu)
print(l2_gpu)

x_on_gpu0 = cp.array([1, 2, 3, 4, 5])
