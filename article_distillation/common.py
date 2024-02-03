import json

import numpy as np
import numpy.linalg as npl
import pandas as pd
import matplotlib.pyplot as plt


def plot_bounds(bounds):
    nx = len(bounds[0])
    for i in range(1, nx-1):
        bx = bounds[0][i]
        plt.plot((bx, bx), (0, 1), c='gray')

    ny = len(bounds[1])
    for i in range(1, ny-1):
        by = bounds[1][i]
        plt.plot((0, 1), (by, by), c='gray')


def plot_data(name, X, y, bounds):
    fname = name + ".png"
    plt.clf()
    plt.title(name)

    X_blu = X[y == 0]
    X_red = X[y == 1]

    plt.scatter(X_blu[:, 0], X_blu[:, 1], c='blue', s=5)
    plt.scatter(X_red[:, 0], X_red[:, 1], c='red', s=5)
    plot_bounds(bounds)

    plt.savefig(fname, dpi=300)
    plt.clf()

    # Xd_blu = Xd[yd == 0]
    # Xd_red = Xd[yd == 1]
    #
    # plt.scatter(Xd_blu[:, 0], Xd_blu[:, 1], c='blue', s=20)
    # plt.scatter(Xd_red[:, 0], Xd_red[:, 1], c='red', s=20)
    # plot_bounds(bounds)
    #
    # fname = f"Distilled{suffix}.png"
    # plt.savefig(fname, dpi=300)


# SUFFIX = "-1kx2"
# DATA = f"data/Xy{SUFFIX}.csv"
# BOUNDS = f"data/Xy_bounds{SUFFIX}.json"

DATA_DIR = "data_reg"


def load_data(SUFFIX):
    DATA = f"{DATA_DIR}/Xy{SUFFIX}.csv ..."
    print(f"Loading data {DATA}")
    df = pd.read_csv(DATA)
    X = df[df.columns.difference(["y"])].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    print("... done")
    return X, y


def load_bounds(SUFFIX):
    BOUNDS = f"{DATA_DIR}/Xy_bounds{SUFFIX}.json"
    print(f"Loading bounds {BOUNDS}")
    with open(BOUNDS) as fp:
        return json.load(fp)


class CoresetSelector:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def select(self, Xd):
        n = len(Xd)
        Xcs = np.zeros_like(Xd, dtype=self.X.dtype)
        ycs = np.zeros(n, dtype=self.y.dtype)
        for i in range(n):
            xcs, ccs = self._find_nearest(Xd[i])
            Xcs[i] = xcs
            ycs[i] = ccs
        return Xcs, ycs

    def _find_nearest(self, d):
        X = self.X
        y = self.y
        n = len(X)

        best_point = X[0]
        best_class = y[0]
        best_dist = npl.norm(d-best_point)
        for i in range(n):
            x = X[i]
            dist = npl.norm(d-x)
            if dist < best_dist:
                best_dist = dist
                best_point = x
                best_class = y[i]
        # end
        return best_point, best_class
    # end