import numpy as np

from etime.lag import resolve_lag


def gen(size):
    if isinstance(size, int):
        size = (size,)
    if len(size) == 1:
        n = size[0]
        data = [100+i+1 for i in range(n)]
    else:
        n, m = size
        data = [[j*m+i+1 for i in range(m)] for j in range(n)]
    return np.array(data)


def main():
    slots = resolve_lag((3, 5))
    pass


if __name__ == "__main__":
    main()
