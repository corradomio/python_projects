import numpy as np
import numpyx as npx

# ---------------------------------------------------------
# data

def prepare_data():
    X = np.array([[r*10+c for c in range(1, 5)] for r in range(1, 10)])
    y = np.array([100+i for i in range(1, 10)]).reshape((-1, 1))

    Xp = np.array([[-(r*10+c) for c in range(1, 5)] for r in range(1, 10)])
    yp = np.array([-(100+i) for i in range(1, 10)]).reshape((-1, 1))
    n = len(yp)

    return X, y, Xp, yp, n


# ---------------------------------------------------------
# LagReshaper
# LagPreparer
# OK

def check_lag(X, y, Xp, yp, n):

    lr = npx.LinearTrainTransform(xlags=[0], ylags=[1])
    Xt, yt = lr.fit_transform(X, y)


    lp = npx.LinearPredictTransform(xlags=[0], ylags=[1])
    ys = lp.fit(X, y).transform(Xp, n)

    for i in range(n):
        Xs = lp.step(i)
        ys[i] = i
        pass
# end


# ---------------------------------------------------------
# UnfoldLoop
# LagPreparer
# OK

def check_unfold(X, y, Xp, yp, n):
    ul = npx.RNNTrainTransform(steps=2, xlags=[0], ylags=[1])
    Xt, yt = ul.fit_transform(X, y)

    up = npx.RNNPredictTransform(steps=4, xlags=[1], ylags=[1])
    ys = up.fit(X, y).transform(Xp, n)

    for i in range(n):
        Xs = up.step(i)
        ys[i] = -(i+1)
        pass
    pass


def main():
    data = prepare_data()

    # check_lag(*data)
    check_unfold(*data)
# end


if __name__ == "__main__":
    main()
