import numpy as np


def zo_matrix(nrows, ncols):
    mat = np.zeros((nrows, ncols), dtype=int)
    c = 0
    for i in range(nrows):
        for j in range(ncols):
            mat[i, j] = c
            c += 1
    return mat


def ij_matrix(nrows, ncols):
    def _factor(n):
        f = 10
        while n >= f:
            f *= 10
        return f

    # mat = np.zeros((nrows, ncols), dtype=int)
    mat = np.zeros((nrows, ncols), dtype=float)
    f = _factor(ncols)
    if ncols > 1:
        for r in range(nrows):
            for c in range(ncols):
                # mat[r, c] = r * f + c
                # mat[r, c] = (r + 1) * f + (c + 1)
                mat[r, c] = (r + 1) + (c + 1)/f
    else:
        for r in range(nrows):
            mat[r, 0] = (r+1)

    return mat
# end