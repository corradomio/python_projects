import numpy as np

import numpyx as npx
import sktimex as skx


def check_lin():
    Xa = +npx.ij_matrix(120, 9)
    ya = -npx.ij_matrix(120, 2)

    X, Xp, y, yp = npx.size_split(Xa, ya, split_size=100)

    ltt = skx.LinearTrainTransform(lags=[2, 3], tlags=4)
    Xt, yt = ltt.fit_transform(y, X)

    lpt = skx.LinearPredictTransform(lags=[2, 3], tlags=4)

    ypp = lpt.fit(y, X).transform(fh=0, X=Xp)
    yp0 = np.zeros_like(yt[0:1])

    i = 0
    for i in range(len(ypp)):
        Xpp = lpt.step(i)
        print("i:", i)
        for j in range(8):
            yp0[0, j] = -(101 + i + j//2 + (j%2+1)/10.)

        lpt.update(i, yp0)

    pass


def check_rnn():
    Xa = +npx.ij_matrix(120, 9)
    ya = -npx.ij_matrix(120, 2)

    X, Xp, y, yp = npx.size_split(Xa, ya, split_size=100)

    ltt = skx.RNNTrainTransform(lags=[2, 2], tlags=4, flatten=True)
    Xt, yt = ltt.fit_transform(y, X)

    lpt = skx.RNNPredictTransform(lags=[2, 2], tlags=4)

    ypp = lpt.fit(y, X).transform(fh=0, X=Xp)
    yp0 = np.zeros_like(yt[0:1])

    i = 0
    for i in range(len(ypp)):
        Xpp = lpt.step(i)
        print("i:", i)
        for j in range(8):
            yp0[0, j] = -(101 + i + j // 2 + (j % 2 + 1) / 10.)

        lpt.update(i, yp0)

    pass


def main():
    Xa = +npx.ij_matrix(120, 9)
    ya = -npx.ij_matrix(120, 2)

    X, Xp, y, yp = npx.size_split(Xa, ya, split_size=100)

    ltt = skx.CNNTrainTransform(lags=[2, 2], tlags=4, flatten=True)
    Xt, yt = ltt.fit_transform(y, X)

    lpt = skx.CNNPredictTransform(lags=[2, 2], tlags=4)

    ypp = lpt.fit(y, X).transform(fh=0, X=Xp)
    yp0 = np.zeros_like(yt[0:1])

    i = 0
    for i in range(len(ypp)):
        Xpp = lpt.step(i)
        print("i:", i)
        for j in range(8):
            yp0[0, j] = -(101 + i + j // 2 + (j % 2 + 1) / 10.)

        lpt.update(i, yp0)

    pass


if __name__ == "__main__":
    main()
