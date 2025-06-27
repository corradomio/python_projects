from .utils import *
from multimethod import multimethod, overload


def eq(A1, A2) -> list:
    r = rank(A1)
    constrs = []
    if r == 1:
        n = len(A1)
        for i in range(n):
            constrs.append(A1[i] == A2[i])
    elif r == 2:
        n, m = shape(A1)
        for i in range(n):
            for j in range(m):
                constrs.append(A1[i, j] == A2[i, j])
    else:
        raise ValueError(f"Unsupported shape {shape(A1)}")
    return constrs
# end


def leq(A1, A2) -> list:
    r = rank(A1)
    constrs = []
    if r == 1:
        n = len(A1)
        for i in range(n):
            constrs.append(A1[i] <= A2[i])
    elif r == 2:
        n, m = shape(A1)
        for i in range(n):
            for j in range(m):
                constrs.append(A1[i, j] <= A2[i, j])
    else:
        raise ValueError(f"Unsupported shape {shape(A1)}")
    return constrs
# end


def geq(A1, A2) -> list:
    r = rank(A1)
    constrs = []
    if r == 1:
        n = len(A1)
        for i in range(n):
            constrs.append(A1[i] <= A2[i])
    elif r == 2:
        n, m = shape(A1)
        for i in range(n):
            for j in range(m):
                constrs.append(A1[i, j] >= A2[i, j])
    else:
        raise ValueError(f"Unsupported shape {shape(A1)}")
    return constrs
# end


@overload
def sum_row_leq(M1, R):
    constrs = []
    n, m = shape(M1)
    for i in range(n):
        c = 0
        for j in range(m):
            c = c + M1[(i,j)]

        constrs.append(c <= R[i])
    return constrs


@overload
def sum_row_leq(M1, C, R):
    constrs = []
    n, m = shape(M1)
    for i in range(n):
        c = 0
        for j in range(m):
            c = c + M1[(i,j)]*C[j]

        constrs.append(c <= R[i])
    return constrs


@overload
def sum_col_leq(M1, R):
    constrs = []
    n, m = shape(M1)
    for j in range(m):
        c = 0
        for i in range(n):
            c = c + M1[(i,j)]

        constrs.append(c <= R[j])
    return constrs


@overload
def sum_col_leq(M1, C, R):
    constrs = []
    n, m = shape(M1)
    for j in range(m):
        c = 0
        for i in range(n):
            c = c + M1[(i,j)]*C[i]

        constrs.append(c <= R[j])
    return constrs

# ---------------------------------------------------------------------------

@overload
def sum_row_geq(M1, R):
    constrs = []
    n, m = shape(M1)
    for i in range(n):
        c = 0
        for j in range(m):
            c = c + M1[(i,j)]

        constrs.append(c >= R[i])
    return constrs


@overload
def sum_row_geq(M1, C, R):
    constrs = []
    n, m = shape(M1)
    for i in range(n):
        c = 0
        for j in range(m):
            c = c + M1[(i,j)]*C[j]

        constrs.append(c >= R[i])
    return constrs


@overload
def sum_col_geq(M1, R):
    constrs = []
    n, m = shape(M1)
    for j in range(m):
        c = 0
        for i in range(n):
            c = c + M1[(i,j)]

        constrs.append(c >= R[j])
    return constrs


@overload
def sum_col_geq(M1, C, R):
    constrs = []
    n, m = shape(M1)
    for j in range(m):
        c = 0
        for i in range(n):
            c = c + M1[(i,j)]*C[i]

        constrs.append(c >= R[j])
    return constrs

# ---------------------------------------------------------------------------

@overload
def sum_row_eq(M1, R):
    constrs = []
    n, m = shape(M1)
    for i in range(n):
        c = 0
        for j in range(m):
            c = c + M1[(i,j)]

        constrs.append(c == R[i])
    return constrs


@overload
def sum_row_eq(M1, C, R):
    constrs = []
    n, m = shape(M1)
    for i in range(n):
        c = 0
        for j in range(m):
            c = c + M1[(i,j)]*C[j]

        constrs.append(c == R[i])
    return constrs


@overload
def sum_col_eq(M1, R):
    constrs = []
    n, m = shape(M1)
    for j in range(m):
        c = 0
        for i in range(n):
            c = c + M1[(i,j)]

        constrs.append(c == R[j])
    return constrs


@overload
def sum_col_eq(M1, C, R):
    constrs = []
    n, m = shape(M1)
    for j in range(m):
        c = 0
        for i in range(n):
            c = c + M1[(i,j)]*C[i]

        constrs.append(c == R[j])
    return constrs

# ---------------------------------------------------------------------------

@overload
def broadcast_row_geq(M1, C, M2):
    constrs = []
    n, m = shape(M1)
    for i in range(n):
        for j in range(m):
            c = M1[i,j]*C[j] >= M2[i,j]
            constrs.append(c)
    return constrs


@overload
def broadcast_col_geq(M1, C, M2):
    constrs = []
    n, m = shape(M1)
    for i in range(n):
        for j in range(m):
            c = M1[i, j] * C[i] >= M2[i, j]
            constrs.append(c)
    return constrs


@overload
def broadcast_row_leq(M1, C, M2):
    constrs = []
    n, m = shape(M1)
    for i in range(n):
        for j in range(m):
            c = M1[i,j]*C[j] <= M2[i,j]
            constrs.append(c)
    return constrs

@overload
def broadcast_col_leq(M1, C, M2):
    constrs = []
    n, m = shape(M1)
    for i in range(n):
        for j in range(m):
            c = M1[i, j] * C[i] <= M2[i, j]
            constrs.append(c)
    return constrs
