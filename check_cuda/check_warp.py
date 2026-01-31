import numpy as  np
import warp as wp

v: np.float16 = 0

wp.init()

@wp.kernel
def simple_kernel(a: wp.array(dtype=wp.vec3),
                  b: wp.array(dtype=wp.vec3),
                  c: wp.array(dtype=float)):

    # get thread index
    tid = wp.tid()

    # load two vec3s
    x = a[tid]
    y = b[tid]

    # compute the dot product between vectors
    r = wp.dot(x, y)

    # write result back to memory
    c[tid] = r


n = 1000

# allocate an uninitialized array of vec3s
v = wp.empty(shape=n, dtype=wp.vec3, device="cuda")

# allocate a zero-initialized array of quaternions
q = wp.zeros(shape=n, dtype=wp.quat, device="cuda")

# allocate and initialize an array from a NumPy array
# will be automatically transferred to the specified device
a = wp.zeros((10, 3), dtype=wp.float32, device="cuda")
b = wp.ones((10, 3), dtype=wp.float32, device="cuda")
c = wp.ones((10, 3), dtype=wp.float32, device="cuda")
v = wp.from_numpy(a, dtype=wp.vec3, device="cuda")


wp.launch(kernel=simple_kernel, # kernel to launch
          dim=1024,             # number of threads
          inputs=[a, b, c],     # parameters
          device="cuda")        # execution device


a = a.to("cpu")
print(a)