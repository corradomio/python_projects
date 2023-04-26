import numpy as np
from sklearn.linear_model import LinearRegression

from etime.lag import resolve_lag
from etime.linear_model import format_data, format_single


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
    islots, tslots = resolve_lag((3, 5))

    X = gen((100, 3))
    y = gen((100,))

    tx, ty = format_data(y=y, X=X, x_slots=islots, y_slots=tslots)
    sx = format_single(y=y, X=X, x_slots=islots, y_slots=tslots, start=5)
    pass


if __name__ == "__main__":
    main()
