from sktimex.transform.lags import yx_lags, tu_lags, matrix
from sktimex.transform.lagt import LagsTrainTransform


def main():
    X0 = matrix(9, 2)
    y0 = matrix(9, 0)
    X1 = matrix(11, 2, 10)
    y1 = matrix(10, 1, 10)

    ylags, xlags = yx_lags([2,2])
    tlags, ulags = tu_lags([[1,3],1])

    ltt = LagsTrainTransform(xlags=xlags, ylags=ylags, tlags=tlags, ulags=ulags)
    Xt, yt = ltt.fit_transform(y=y0, X=X0)

    lpt = ltt.predict_transform()
    Xt, yt = lpt.fit(y=y0, X=X0).transform(fh=10, X=X1)

    Xf, yf = lpt.predict_steps(y1)

    pass


if __name__ == "__main__":
    main()

