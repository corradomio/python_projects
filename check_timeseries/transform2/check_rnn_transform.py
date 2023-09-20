import logging.config
from numpyx.utils import ij_matrix, zo_matrix
from sktimex.model_transform import RNNTrainTransform, RNNPredictTransform
from sktimex import resolve_lags


def main():
    X = ij_matrix(150, 10)
    y = ij_matrix(150, 3)+1000

    slots = resolve_lags([2, 2])
    tlags = [0]
    nt = len(tlags)

    ltt = RNNTrainTransform(slots=slots, tlags=tlags)
    Xt, yt = ltt.fit(X, y).transform(X, y)

    Xh, Xt = X[:100], X[100:]
    yh, yt = y[:100], y[100:]
    n, ny = yt.shape

    lpt = RNNPredictTransform(slots=slots, tlags=tlags)
    yp = lpt.fit(Xh, yh).transform(Xt, 0)

    for i in range(n):
        Xp = lpt.step(i)
        for j in range(nt*ny):
            yp[i, j] = -(i+j+1)
        pass

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
