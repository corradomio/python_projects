import logging.config
from numpyx.utils import ij_matrix, zo_matrix
from sktimex.transform import CNNTrainTransform, CNNPredictTransform
from sktimex import resolve_lags


def main():
    X = zo_matrix(100, 9)
    y = zo_matrix(100, 2)+100

    slots = resolve_lags([2, 2])
    tlags = [0]

    ltt = CNNTrainTransform(slots=slots, tlags=tlags)
    Xt, yt = ltt.fit(X, y).transform(X, y)

    Xh, Xt = X[:80], X[80:]
    yh, yt = y[:80], y[80:]
    n = len(yt)

    lpt = CNNPredictTransform(slots=slots, tlags=tlags)
    yp = lpt.fit(Xh, yh).transform(Xt, 0)

    for i in range(20):
        Xp = lpt.step(i)
        pass

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
