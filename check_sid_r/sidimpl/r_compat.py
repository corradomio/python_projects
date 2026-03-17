from typing import Union, cast

import numpy as np

# R data types
#   logical (bool)
#   integer (int)
#   numeric/double (float)
#   character (str)
#   complex
#
# R data structures
#   vector, matrix, array
#   list | dict
#   data frame
#   .

# vector
#   v = c(1,2,3)
#
# matrix
#   m = matrix(1:6, nrow=2,ncol=3)
#
# array
#   a = array(1:12), dim=c(2,3,2))
#
# conversions
#   logical -> integer -> numeric -> character

R_DATA_TYPES = [None, bool, int, float, str]

# l = list(
#   name="R",
#   age=30,
#   scores=c(90,85,88)
# )
# l[1]      [name="R"]
# l[[1]]    "R"
# l$name    "R"
#

def dtype_of(arg):
    if isinstance(arg, (bool, int, float, str)):
        return type(arg)
    if isinstance(arg, np.ndarray):
        return arg.dtype

    ttype = 1
    for a in arg:
        t = R_DATA_TYPES.index(type(a))
        if t > ttype:
            ttype = t
    # end
    return R_DATA_TYPES[ttype]
#

def c(*args, dtype=None) -> np.ndarray:
    if len(args) == 0:
        return None

    c = []
    for a in args:
        if isinstance(a, (bool, int, float, str)):
            c.append(a)
        elif isinstance(a, list):
            c += a
        elif isinstance(a, tuple):
            c += list(a)
        elif isinstance(a, np.ndarray):
            c += cast(np.ndarray, a).reshape(-1).tolist()

    dtype = dtype if dtype is not None else dtype_of(c)
    return np.array(c, dtype=dtype)
# end

def rbind(*args) -> np.ndarray:
    if len(args) == 0:
        return None
    else:
        return np.array(args)


def cbind(*args) -> np.ndarray:
    if len(args) == 0:
        return None
    else:
        return np.array(args).T


def t(a: np.ndarray) -> np.ndarray:
    return a.T


def matrix(a, nrow, ncol=-1, dtype=None) -> np.ndarray:
    assert isinstance(nrow, (int, tuple))
    assert isinstance(ncol, int)
    assert isinstance(a, (int, float, np.ndarray))

    if isinstance(nrow, tuple):
        nrow, ncol = nrow
    if isinstance(a, int):
        dtype = int if dtype is None else dtype
        return np.zeros((nrow, ncol), dtype=dtype) + a
    if isinstance(a, float):
        dtype = float if dtype is None else dtype
        return np.zeros((nrow, ncol), dtype=dtype) + a
    else:
        return a.reshape((nrow, ncol))


def length(a: np.ndarray) -> int:
    assert isinstance(a, (list, np.ndarray))
    if isinstance(a, list): return len(a)
    if len(a.shape) == 0: return 0
    if len(a.shape) == 1: return a.shape[0]
    if len(a.shape) == 2: return a.shape[0]*a.shape[1]
    else: raise ValueError(f"Unsupported array with shape {a.shape}")


def seq(start: int, end: int, by: int=1, dtype=int):
    # start:end == seq(start, end, 1)
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(by, int)
    return np.arange(start, end+1, by, dtype=dtype)


def dim(a: np.ndarray) -> tuple:
    return a.shape


def rep(a, each: int=1, dtype=None) -> Union[list, np.ndarray]:
    assert isinstance(a, (int, float, list, np.ndarray))
    if isinstance(a, int):
        dtype = int if dtype is None else dtype
        return np.zeros(each, dtype=dtype) + a
    if isinstance(a, float):
        dtype = float if dtype is None else dtype
        return np.zeros(each, dtype=dtype) + a
    if isinstance(a, np.ndarray):
        a = a.tolist()
    r = []
    for e in a:
        r += [e]*each
    if isinstance(r[0], int) and dtype is None:
        dtype = int
    elif isinstance(r[0], float) and dtype is None:
        dtype = float
    elif isinstance(r[0], bool) and dtype is None:
        dtype = bool
    return np.array(r, dtype=dtype)

def which(p):
    return np.where(p)[0]


def expand_grid(m) -> np.ndarray:
    if isinstance(m, np.ndarray):
        m = m.tolist()

    grid = [[]]
    for prefix in m[::-1]:
        ric = []
        for p in prefix:
            for g in grid:
                ric.append(
                    [p] + g
                )
        grid = ric
    # end
    return np.array(grid)


def as_matrix(m):
    return np.array(m)


def list_(a: np.array):
    assert isinstance(a, np.ndarray)
    return a.tolist()


def duplicated(m: np.ndarray):
    n = m.shape[0]
    d = np.zeros(n, dtype=bool)
    for i in range(1, n):
        for j in range(i):
            if np.all(m[i, :] == m[j, :]):
                d[i] = True
                break
    return d
# end


def colSums(m: np.ndarray):
    return np.sum(m, axis=0)


def print_r_mat(var, A):
    n, m = A.shape
    print(f"{var} <- rbind(")
    for i in range(n):
        print(f"   c({A[i,0]}", end="")
        for j in range(1, m):
            print(f",{A[i,j]}", end="")
        if i < m-1:
            print("),")
        else:
            print(")")
    print(")")
