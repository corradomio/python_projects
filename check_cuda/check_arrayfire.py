import arrayfire as af

af.set_backend("cuda")

A:af.Array = af.randu(2**15, 2**15)
A2:af.Array = af.matmul(A, A)
B:af.Array = af.fft2(A2)
print(B[0,0].scalar())
