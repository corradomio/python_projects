from random import randrange, randint, random
import numpy as np
import matplotlib.pyplot as plt


MAX_SEGMENTS = 5


def label_of(x, bounds) -> int:
    if isinstance(x, np.ndarray):
        d = x.shape[0]
        labels = [label_of(x[i], bounds[i]) for i in range(d)]
        return sum(labels) % 2
    else:
        l = len(bounds) - 1
        for i in range(l):
            if bounds[i] <= x < bounds[i+1]:
                return i
        return l
    # end
# end


def gen_data(n: int, d: int=1) -> tuple[np.ndarray, np.ndarray, list]:
    """

    :param n: of points
    :param d: data dimension
    :return: array of data
    """
    X = np.zeros((n, d), dtype=float)
    y = np.zeros(n, dtype=int)

    bounds = []
    for i in range(d):
        nsegs = 0
        while nsegs < 2:
            nsegs = randint(1, MAX_SEGMENTS)
        segs_bounds = [0.] + sorted(random() for s in range(nsegs-1)) + [1.]
        bounds.append(segs_bounds)
    # end

    for i in range(n):
        X[i] = np.random.rand(d)
        y[i] = label_of(X[i], bounds)
    return X, y, bounds


def plot_data(X, y, bounds):
    X_red = X[y == 1]
    X_blu = X[y == 0]

    plt.scatter(X_red[:, 0], X_red[:, 1], c='red', s=5)
    plt.scatter(X_blu[:, 0], X_blu[:, 1], c='blue', s=5)

    nx = len(bounds[0])
    for i in range(1, nx-1):
        bx = bounds[0][i]
        plt.plot((bx, bx), (0, 1), c='gray')

    ny = len(bounds[1])
    for i in range(1, ny-1):
        by = bounds[1][i]
        plt.plot((0, 1), (by, by), c='gray')

    plt.show()

    pass


def main():
    X, y, bounds = gen_data(1000, 2)

    plot_data(X, y, bounds)

    print(y.sum())
    pass


if __name__ == "__main__":
    main()

