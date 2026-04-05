import cython

def cfib(n: cython.int) -> cython.float:
    i: cython.int = 0
    a: cython.float = 0.0
    b: cython.float = 1.0
    for i in range(n):
        a, b = a + b, a
    return a
