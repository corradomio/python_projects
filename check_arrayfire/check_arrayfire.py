# ArrayFire per CUDA 11.8
# M

# https://arrayfire.com/
import arrayfire as af

# af.set_backend('cpu')
# A = af.randu(2**15, 2**15)

# max size with 16GB ram GPU
af.set_backend('cuda')
A = af.randu(2**14, 2**14)

A2 = af.matmul(A, A)
B = af.fft2(A2)

print(B)
