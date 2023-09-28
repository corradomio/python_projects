import logging.config
from numpy import arange
from numpyx.utils import zo_matrix, ij_matrix
from sktimex.transform import LinearTrainTransform, LinearPredictTransform
from sktimex import resolve_lags


def main():
    N = 100
    mx = 9
    my = 3
    X = ij_matrix(100, mx)
    y = ij_matrix(100, my)*(-1)
    p = arange(my)+1
    slots = resolve_lags([2, 2])
    tlags = [0, 2]

    ltt = LinearTrainTransform(slots=slots, tlags=tlags)
    Xt, yt = ltt.fit(X, y).transform(X, y)

    Xh, Xt = X[:80], X[80:]
    yh, yt = y[:80], y[80:]
    n = len(yt)

    lpt = LinearPredictTransform(slots=slots, tlags=tlags)
    yp = lpt.fit(Xh, yh).transform(Xt, 0)

    for i in range(n):
        Xp = lpt.step(i)
        c = 0
        for j in range(len(tlags)):
            yp[i, c:c+my] = -(p+i+j)
            c += my
        pass

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
