from sktimex.transform.lags import yx_lags, t_lags, matrix
from sktimex.transform.lagt import LagsTrainTransform, LagsPredictTransform
import stdlib.assertionx


X0 = matrix(9, 9)
y0 = matrix(9, 0)
y1 = matrix(9, 1)


def test_lag_transform_1():

    ltt = LagsTrainTransform(xlags=[0], ylags=[], tlags=[])
    lpt = ltt.predict_transform()

    Xt, yt = ltt.fit_transform(y=y0, X=X0)

    Xp, yp = lpt.fit_transform(y=y0, X=X0)

