#
# Recursive prediction
# --------------------
#
#   Add support to recursive prediction to models don't supporting it
#
import pandas as pd
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


def _combine_pandas(P,F):
    if P is None or F is None:
        return None
    if isinstance(P, pd.Series) and isinstance(F, pd.DataFrame):
        F = F[P.name]
    if isinstance(P, pd.DataFrame) and isinstance(F, pd.Series):
        F = pd.DataFrame(data={F.name: F}, index=F.index)
    n = len(P)
    # 1) concatenate
    H = pd.concat([P, F], axis=0, ignore_index=True)
    # 2) truncate
    H = H.iloc[-n:]
    # 3) set index
    H.index = P.index
    # 4) done
    return H


def _extends_pandas(P, F):
    if P is None:
        return F
    if isinstance(P, pd.Series) and isinstance(F, pd.DataFrame):
        F = F[P.name]
    if isinstance(P, pd.DataFrame) and isinstance(F, pd.Series):
        F = pd.DataFrame(data={F.name: F}, index=F.index)
    E = pd.concat([P,F], axis=0, ignore_index=True)
    e = len(E)
    I = ForecastingHorizon(list(range(e)), is_relative=True).to_absolute(P.index[0:1])
    E.index = I.to_pandas()
    return E


def _split_pandas(D, d):
    if D is None:
        return None, None
    Dp = D.iloc[:d]
    Df = D.iloc[d:]
    return Dp, Df


def _reindex_pandas(Xp, Xf):
    if Xp is None:
        return None
    d = len(Xf)
    I = ForecastingHorizon(list(range(1, d+1))).to_absolute(Xp.index[-1:])
    Xf.index = I.to_pandas()
    return Xf


class RecursivePredict(BaseForecaster):

    def recursive_predict(self, fh: ForecastingHorizon, X):
        assert len(fh) % len(self._fh_in_fit) == 0
        if len(fh) == len(self._fh_in_fit):
            return self._single_step_predict(fh, X)
        else:
            return self._multi_step_predict(fh, X)

    def _single_step_predict(self, fh, X):
        return super().predict(self._fh_in_fit, X)

    def _multi_step_predict(self, fh, X):
        yo = self._y
        Xo = self._X
        Xf = X if Xo is not None else None
        n = len(fh)
        p = 0
        yp = None
        while p < n:
            y_pred = super().predict(self._fh_in_fit, Xf)
            yp = _extends_pandas(yp, y_pred)
            p = len(yp)
            yh = _combine_pandas(yo, yp)
            Xp, Xf = _split_pandas(X, p)
            Xh = _combine_pandas(Xo, Xp)
            Xf = _reindex_pandas(Xo, Xf)
            self.update(yh, Xh)
        else:
            self.update(yo, Xo)
        return yp

    def update(self, y, X=None, update_params=True):
        super().update(y, X, update_params=update_params)
        # self._y = y
        # self._X = X