#
# Implementation of a projection of a dataset from a lower dimension
# to an  higher one using a random non linear transformation.
# The transformation used is a set of polynomials of maximum degree 5
# It is used all combination of variables, with related powers, equal
# to the term's power. For example, if the lower dimension is 2 (u,v are
# the dimensions), we have:
#
#   0 -> constant
#   1 -> u, v
#   2 -> u^2, uv, v^2
#   3 -> u^3, u^2v, uv^2, v^3
#   ...
#
from random import uniform

import numpy as np


def combinations(n, k):
    c = 1
    for i in range(1, k+1):
        c *= (n+i-1)
        c //= i
    return c


def random_coeffs(degree, nvars):
    w = 1/(degree+1)
    ncoeffs = combinations(nvars, degree)
    return [w*uniform(-1, 1) for i in range(ncoeffs)]


def poly_coeffs(degree, nvars):
    return [
        random_coeffs(deg, nvars)
        for deg in range(degree+1)
    ]


def dim_coeffs(tdim, degree, nvars):
    return [
        poly_coeffs(degree, nvars)
        for d in range(tdim)
    ]


def eval_coeffs(x, coeffs):
    if isinstance(x, np.ndarray):
        y = np.zeros(x.shape[1], dtype=x.dtype)
    else:
        y = 0
    d = len(coeffs)
    n = len(x)
    # y = 0
    if d > 0:
        c = 0
        dc = coeffs[0]
        y += dc[c]
    if d > 1:
        c = 0
        dc = coeffs[1]
        for i1 in range(0, n):
            y += dc[c] * x[i1]
            c += 1
    if d > 2:
        c = 0
        dc = coeffs[2]
        for i1 in range(0, n):
            for i2 in range(i1, n):
                y += dc[c] * x[i1] * x[i2]
                c += 1
    if d > 3:
        c = 0
        dc = coeffs[3]
        for i1 in range(0, n):
            for i2 in range(i1, n):
                for i3 in range(i2, n):
                    y += dc[c] * x[i1] * x[i2] * x[i3]
                    c += 1
    if d > 4:
        c = 0
        dc = coeffs[4]
        for i1 in range(0, n):
            for i2 in range(i1, n):
                for i3 in range(i2, n):
                    for i4 in range(i3, n):
                        y += dc[c] * x[i1] * x[i2] * x[i3] * x[i4]
                        c += 1
    if d > 5:
        c = 0
        dc = coeffs[5]
        for i1 in range(0, n):
            for i2 in range(i1, n):
                for i3 in range(i2, n):
                    for i4 in range(i3, n):
                        for i5 in range(i4, n):
                            y += dc[c] * x[i1] * x[i2] * x[i3] * x[i4] * x[i5]
                            c += 1
    return y
# end


class PolyCoeff:

    def __init__(self, tdim, degree, nvars):
        self.dim = tdim
        self.degree = degree
        self.nvars = nvars
        self.coeffs = dim_coeffs(tdim, degree, nvars)
        pass

    def __call__(self, x):
        y = [None]*self.dim
        for d in range(self.dim):
            coeffs = self.coeffs[d]
            y[d] = eval_coeffs(x, coeffs)
        return np.array(y, dtype=x.dtype)
# end


def project_data(X, tdim, degree=3):
    n, nvars = X.shape

    # generate a random projection
    pc = PolyCoeff(tdim, degree, nvars)

    # apply the random projection to the dataset.
    # Note: it is used X transposed because in this way
    #       X[i] is the i-th dimension of the complete dataset
    #       and the implementation works exactly aas for
    #       single floating points
    Pt = pc(X.T)

    return Pt.T
# end


