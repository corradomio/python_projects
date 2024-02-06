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


def gen_bounds(n, d, k) -> list:
    """
    :param d: data dimension
    :return:
    """
    k = 0 if k is None else k
    bounds = []

    if k == 0:
        # random suddivisions
        for i in range(d):
            nsegs = 0
            while nsegs < 2:
                nsegs = randint(1, MAX_SEGMENTS)
            segs_bounds = [0.] + sorted(random() for s in range(nsegs-1)) + [1.]
            bounds.append(segs_bounds)
        # end
    else:
        for i in range(n):
            ds = 1./k
            bounds.append([i*ds for i in range(k+1)])
        # end
    # end

    if n < 1000:
        sn = str(n)
    elif n == 1000:
        sn = "1k"
    elif n < 1000000:
        sn = f"{n // 1000}k"
    else:
        sn = f"{n// 1000000}m"

    # save bounds:
    fname = f"Xy_bounds-{sn}x{d}x{k}.json"
    with open(fname, mode='w') as fp:
        json.dump(bounds, fp, indent="  ")

    return bounds


def gen_data(n: int, d: int, k, bounds: list) -> tuple[np.ndarray, np.ndarray]:
    """
    :param n: of points
    :param d: data dimension
    :return: array of data
    """
    k = 0 if k is None else k
    X = np.zeros((n, d), dtype=float)
    y = np.zeros(n, dtype=int)

    valid = False
    while not valid:
        for i in range(n):
            X[i] = np.random.rand(d)
            y[i] = label_of(X[i], bounds)

        c1 = y.sum()
        valid = min(c1, n-c1)/n > 0.40
        if not valid:
            print(f"Classes non balanced: c1={c1}. Retry")

    data = {"y": y}
    for i in range(d):
        if d <= 10:
            kd = f"x{i}"
        elif d <= 100:
            kd = f"x{i:02}"
        else:
            kd = f"x{i:03}"

        data[kd] = X[:, i]

    # save data
    # df = pd.DataFrame(data={"x0": X[:, 0], "x1": X[:, 1], "y": y})
    df = pd.DataFrame(data=data)

    if n < 1000:
        nk = str(n)
    elif n == 1000:
        nk = "1k"
    elif n < 1000000:
        nk = f"{n // 1000}k"
    else:
        nk = f"{n// 1000000}m"

    fname = f"Xy-{nk}x{d}x{k}.csv"
    df.to_csv(fname, header=True, index=False)

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


def gen_dataset(N, M, K):
    print(f"Generate X[{N}x{M}x{K}]")
    bounds = gen_bounds(N, M, K)
    X, y = gen_data(N, M, K, bounds)
    print(y.sum())


def main():
    # N = 1000
    # M = 2
    # bounds = gen_bounds(M)
    # X, y = gen_data(N, M, bounds)

    # plot_data(X, y, bounds)
    # plot_data(X[:100], y[:100], bounds)
    # plot_data(X[:10], y[:10], bounds)

    k = 3
    ns = [100, 1000, 10000]
    ds = [2, 3, 4, 5, 10, 25, 50, 100]
    # ns = [1000, 10000]
    # ds = [3, 5]
    # ns = [1000, 10000]
    # ds = [4]
    for n in ns:
        for d in ds:
            gen_dataset(n, d, k)

    pass


if __name__ == "__main__":
    main()

