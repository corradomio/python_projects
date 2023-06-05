import numpy as np
import numpyx as npx


def main():
    X = np.array([[r*10+c for c in range(1, 5)] for r in range(1, 10)])
    y = np.array([100+i for i in range(1, 10)]).reshape((-1, 1))

    Xp = np.array([[-(r*10+c) for c in range(1, 5)] for r in range(1, 10)])
    yp = np.array([-(100+i) for i in range(1, 10)]).reshape((-1, 1))

    Xt, yt = npx.reshape(X, y, xlags=[0], ylags=[], tlags=[0, 1])

    lf = npx.LagFuture(xlags=[0], ylags=[1], tlags=[0, 1])
    yt = lf.fit(X, y).transform(Xp, 10)

    for i in range(10):
        Xt = lf.step(i)

    pass
# end


if __name__ == "__main__":
    main()
    pass
