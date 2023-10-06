from random import shuffle
import numpy as np

N = 4

MA = np.zeros((N, N))

for r in range(N):
    for c in range(N):
        MA[r, c] = r*N + c

print(MA)
print()
# indices = list(range(N))
# shuffle(indices)
indices = [0, 2, 1, 3]

print("indices:", indices)

MA_shuffled = np.zeros((N, N))

for r in range(N):
    for c in range(N):
        i = indices[r]
        j = indices[c]

        MA_shuffled[r, c] = MA[i, j]
        pass
    pass


MA_ordered = np.zeros((N, N))

for r in range(N):
    for c in range(N):
        i = indices[r]
        j = indices[c]

        MA_ordered[r, c] = MA_shuffled[i, j]
        pass
    pass

pass
