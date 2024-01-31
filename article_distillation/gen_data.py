from random import randrange, randint, random
import pandas as pd
import numpy as np
import json
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


def gen_bounds(d) -> list:
    """
    :param d: data dimension
    :return:
    """
    bounds = []
    for i in range(d):
        nsegs = 0
        while nsegs < 2:
            nsegs = randint(1, MAX_SEGMENTS)
        segs_bounds = [0.] + sorted(random() for s in range(nsegs-1)) + [1.]
        bounds.append(segs_bounds)
    # end

    # save bounds:
    with open("Xy_bounds.json", mode='w') as fp:
        json.dump(bounds, fp, indent="  ")

    return bounds
# end


def gen_data(n: int, d: int, bounds: list) -> tuple[np.ndarray, np.ndarray]:
    """
    :param n: of points
    :param d: data dimension
    :return: array of data
    """
    X = np.zeros((n, d), dtype=float)
    y = np.zeros(n, dtype=int)

    for i in range(n):
        X[i] = np.random.rand(d)
        y[i] = label_of(X[i], bounds)

    # save data
    df = pd.DataFrame(data={"x1": X[:, 0], "x2": X[:, 1], "y": y})
    df.to_csv("Xy.csv", header=True, index=False)

    return X, y


def plot_data(X, y, bounds):
    plt.clf()

    X_blu = X[y == 0]
    X_red = X[y == 1]

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

    # plt.show()
    plt.savefig(f"Xy-{len(y)}.png", dpi=300)
    pass


def main():
    N = 1000
    M = 2
    bounds = gen_bounds(M)
    X, y = gen_data(N, M, bounds)

    plot_data(X, y, bounds)
    plot_data(X[:100], y[:100], bounds)
    plot_data(X[:10], y[:10], bounds)

    print(y.sum())
    pass


if __name__ == "__main__":
    main()

