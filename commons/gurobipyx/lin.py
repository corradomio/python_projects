from .utils import *


# ---------------------------------------------------------------------------
# dot
# ---------------------------------------------------------------------------
# v.v
# M.M
# v.M
# M.v

def dot(A1, A2):
    r1 = rank(A1)
    r2 = rank(A2)
    r = None

    if r1 == 1 and r2 == 1:
        n = len(A1)
        r = 0
        for i in range(n):
            r += A1[i] * A2[i]

    elif r1 == 2 and r2 == 2:
        n, k = shape(A1)
        k, m = shape(A2)
        r = grb.tupledict()
        for i in range(n):
            for j in range(m):
                s = 0
                for t in range(k):
                    s += A1[i,k] * A2[k,j]
                r[(i,j)] = s

    elif r1 == 2 and r2 == 1:
        n, m = shape(A1)
        r = []
        for i in range(n):
            s = 0
            for j in range(m):
                s += A1[i,j] * A2[j]
            r.append(s)

    elif r1 == 1 and r2 == 2:
        n, m = shape(A2)
        r = []
        for i in range(n):
            s = 0
            for j in range(m):
                s += A1[i] * A2[i, j]
            r.append(s)

    else:
        raise ValueError()
    return r
# end


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def sum_all(M1):
    n, m = shape(M1)
    sum = 0
    for i in range(n):
        for j in range(m):
            sum += M1[i, j]
    return sum


# sum by rows (sum the columns)
def sum_rows(M1):
    n, m = shape(M1)
    sum = []
    for i in range(n):
        s = 0
        for j in range(m):
            s += M1[i, j]
        sum.append(s)
    return sum


# sum by cols (sum the rows)
def sum_cols(M1):
    n, m = shape(M1)
    sum = []
    for j in range(m):
        s = 0
        for i in range(n):
            s += M1[i, j]
        sum.append(s)
    return sum

# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

# product element wise
def hadamard(A1, A2):
    assert shape(A1) == shape(A2)

    r = rank(A1)
    if r == 1:
        res = []
        n = len(A1)
        for i in range(n):
            res.append(A1[i](A2[i]))
    elif r == 2:
        res = grb.tupledict()
        n, m = shape(A1)
        for i in range(n):
            for j in range(m):
                res[i,j] = A1[i,j]*A2[i,j]
    else:
        raise ValueError(f"Unsupported shape {shape(A1)}")
    return res
# end


# sum all elements
def norm1(A1):
    r = rank(A1)
    if r == 1:
        n = len(A1)
        res = 0
        for i in range(n):
            res += A1[i]
    elif r == 2:
        n, m = shape(A1)
        res = 0
        for i in range(n):
            for j in range(m):
                res += A1[i,j]
    else:
        raise ValueError(f"Unsupported shape {shape(A1)}")
    return res
# end


def sum_hadamard(M1, M2):
    return norm1(hadamard(M1, M2))
# end


# ---------------------------------------------------------------------------
# broadcast
# ---------------------------------------------------------------------------

def broadcast_col(M1, R):
    # broadcast along the columns
    n, m = shape(M1)
    assert n == len(R)

    res = grb.tupledict()
    for i in range(n):
        for j in range(m):
            res[i, j] = M1[i,j] * R[i]
    return res
# end


def broadcast_row(M1, C):
    # broadcast along the rows
    n, m = shape(M1)
    assert m == len(C)

    res = grb.tupledict()
    for i in range(n):
        for j in range(m):
            res[i, j] = M1[i,j] * C[j]
    return res
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------


