import numpy as np
from sktimex.lag import resolve_lag, LagTrainTransform, LagPredictTransform, unroll_loop, back_step


def main():
    X = np.array([[r*10+c for c in range(1, 5)] for r in range(1, 10)])
    y = np.array([100+i for i in range(1, 10)]).reshape((-1, 1))

    Xp = np.array([[-(r*10+c) for c in range(1, 5)] for r in range(1, 10)])
    yp = np.array([-(100+i) for i in range(1, 10)]).reshape((-1, 1))

    lags = resolve_lag(lag=([1], [1]))
    # print(len(lags))
    # print(lags.input)
    # print(lags.target)
    # print(lags[0])
    # print(lags[1])

    ltt = LagTrainTransform(lags)
    Xt, yt = ltt.fit_transform(None, y)

    lpt = LagPredictTransform(lags)
    yp = lpt.fit(X, y).transform(Xp, fh=9)

    for i in range(9):
        T = lpt.prepare(i)
        print(T)
        pass

    print(Xt)
    print(yt)

    Xt = unroll_loop(X, 1)
    yt = unroll_loop(y, 1)

    Xt, yt = back_step(X, y, steps=1)

    pass


if __name__ == "__main__":
    main()
    pass
