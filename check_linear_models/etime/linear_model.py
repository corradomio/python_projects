from typing import Union, Optional, Sized

import numpy as np
import pandas as pd
import sktime.forecasting.base as skf
from .stdlib import NoneType, import_from

from .lag import resolve_lag, LagSlots


__all__ = [
    "LinearForecastRegressor"
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_TYPES = (NoneType, pd.DataFrame, np.ndarray)
TARGET_TYPES = (NoneType, pd.DataFrame, pd.Series, np.ndarray)
FH_TYPES = (NoneType, int, list, np.ndarray, pd.Index, pd.TimedeltaIndex, pd.Timedelta, skf.ForecastingHorizon)
X_EMPTY = np.zeros((0, 0))
Y_EMPTY = np.zeros(0)


# ---------------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------------
# this class is used to generate the predictions
# Note: Xh, yh, X, y can be collected from 2 different sources
#
#   1) Xh, yh are the X, y used during the training
#   2) Xh, yh are passed in input on 'predict'.
#       In this case, we must have |y| < |X|, y is used as yh
#       and X is split in 2 parts
#
#       Xh = X[:y]      such that |X[:y]| == |y|
#       Xp = X[y:]      used for the predictions
#
# Prediction cases
#
#   params      past        future      pred len
#   fh          yh          -           |fh|
#   X           Xh, yh      X           |X|
#   fh, X       Xh, yh      X           |fh|
#   fh, y       y           -           |fh|
#   y, X        X[:y], y    X[y:]       |X|-|y|
#   fh, y, X    X[:y], y    X[y:]       |fh|
#

class _PredHist:

    def __init__(self,
                 model,
                 y: Optional[np.ndarray],
                 X: Optional[np.ndarray],
                 yh: np.ndarray,
                 Xh: Optional[np.ndarray],
                 y_pred: np.ndarray,
                 slots: LagSlots):

        # if Xh is none, X is not necessary
        if Xh is None: X = None

        # Xh and X must BOTH None OR not None
        assert X is None and Xh is None or X is not None and Xh is not None
        assert yh is not None
        assert y_pred is not None
        assert slots is not None

        self.model = model
        self.slots: LagSlots = slots
        if y is None:
            self.yh = yh
            self.Xh = Xh
        else:
            yh, Xh, X = self._split_for_history(y, X)
            self.yh = yh
            self.Xh = Xh
        self.Xp = X
        self.yp = y_pred

        sx = len(slots.input_slots)
        sy = len(slots.target_slots)
        m = 0 if X is None else X.shape[1]

        nt = 1
        mt = sx * m + sy

        # mini cache used for the predictions
        # Note: tx MUST BE a 2-rank vector
        self.m = m
        self.tx = np.zeros((nt, mt))
    # end

    def predict(self, i: int) -> float:
        tx = self._prepare(i)
        y_pred = self.model.predict(tx)
        return y_pred[0]
    # end

    def _prepare(self, i: int):
        def at(past, future, index) -> np.ndarray:
            return past[index] if index < 0 else future[index]

        slots = self.slots
        Xh = self.Xh
        Xp = self.Xp
        yh = self.yh
        yp = self.yp
        tx = self.tx
        m = self.m

        if Xh is None:
            c = 0
            for j in reversed(slots.target_slots):
                tx[0, c] = at(yh, yp, i - j)
                c += 1
        else:
            c = 0
            for j in reversed(slots.input_slots):
                tx[0, c:c + m] = at(Xh, Xp, i - j)
                c += m
            for j in reversed(slots.target_slots):
                tx[0, c] = at(yh, yp, i - j)
                c += 1
        # end

        return self.tx
    # end

    # -----------------------------------------------------------------------
    # static functions
    # -----------------------------------------------------------------------

    def _split_for_history(self, y: np.ndarray, X: Optional[np.ndarray]) \
            -> (np.ndarray, Optional[np.ndarray], Optional[np.ndarray]):

        s = len(self.slots)
        # We have |y| < |X|
        # than:
        #   y is reduced to s element (at the end)
        #   X is splitted in
        #       Xh: used for history
        #       Xp: used in prediction

        yh = y[-s:]
        if X is None:
            return yh, None, None

        n = len(y)
        Xp = X[n:]
        Xh = X[n - s:n]
        return yh, Xh, Xp
    # end

    # -----------------------------------------------------------------------
    # end
    # -----------------------------------------------------------------------
# end


# ---------------------------------------------------------------------------
# LinearForecastRegressor
# ---------------------------------------------------------------------------

#
# We suppose that the dataset is ALREADY normalized.
# the ONLY information is to know the name of the target column'
#

class LinearForecastRegressor:

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self,
                 class_name: str,
                 lag: Union[int, list, tuple, dict],
                 **kwargs):
        self._slots = resolve_lag(lag)
        model_class = import_from(class_name)
        self._model = model_class(**kwargs)
        self._X_history: Optional[np.ndarray] = None
        self._y_history: Optional[np.ndarray] = None
    # end

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def fit(self, y: TARGET_TYPES, X: INPUT_TYPES = None, fh: FH_TYPES = None):
        y, X, fh = self._validate_data(y, X, fh)
        self._save_history(y, X, fh)

        Xt, yt = self._prepare_data(X=X, y=y)
        self._model.fit(Xt, yt)

        return self
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def predict(self, fh: FH_TYPES = None, X: INPUT_TYPES = None, y: TARGET_TYPES = None) -> pd.Series:
        # normalize fh, y, X
        yp, Xp, fh = self._validate_data(y, X, fh, predict=True)
        """:type: np.ndarray, np.ndarray, skf.ForecastingHorizon"""

        n = self._prediction_length(fh=fh, y=yp, X=Xp)
        y_pred = np.zeros(n)

        ph = _PredHist(
            self._model,
            yp, Xp,
            self._y_history, self._X_history,
            y_pred, self._slots)

        for i in range(n):
            y_pred[i] = ph.predict(i)

        return self._compose_predictions(y_pred, y, X, fh)
    # end

    @staticmethod
    def _compose_predictions(y_pred, y, X, fh):
        if X is None or not isinstance(X, (pd.Series, pd.DataFrame)):
            return pd.Series(y_pred, index=fh.to_pandas())

        if len(X) == len(y_pred):
            return pd.Series(y_pred, index=X.index)
        else:
            index = X.index
            s = len(y)
            n = len(y_pred)
            return pd.Series(y_pred, index=index[s:s+n])
    # end

    # -----------------------------------------------------------------------
    # score (not implemented yet)
    # -----------------------------------------------------------------------

    def score(self, fh: FH_TYPES, X: INPUT_TYPES, y: TARGET_TYPES) -> dict[str, float]:
        raise NotImplemented()
    # end

    # -----------------------------------------------------------------------
    # Implementation
    # -----------------------------------------------------------------------

    def _validate_data(self, y=None, X=None, fh=None, predict: bool = False):
        # validate the data and converts it:
        #
        #   y in a np.array with rank 1
        #   X in a np.array with rank 2 OR None
        #  fh in a ForecastingHorizon
        #
        # Prediction cases
        #
        #   params      past        future      pred len
        #   fh          yh          -           |fh|
        #   X           Xh, yh      X           |X|
        #   fh, X       Xh, yh      X           |fh|
        #   fh, y       y           -           |fh|
        #   y, X        X[:y], y    X[y:]       |X|-|y|
        #   fh, y, X    X[:y], y    X[y:]       |fh|
        #

        if predict and self._y_history is None:
            raise ValueError(f'{self.__class__.__name__} not fitted yet')

        # X to np.array
        if X is not None:
            assert isinstance(X, INPUT_TYPES)
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            assert isinstance(X, np.ndarray)
            assert len(X.shape) == 2

        # y to np.array
        if y is not None:
            assert isinstance(y, TARGET_TYPES)
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y = y.to_numpy()
            assert isinstance(y, np.ndarray)
            assert len(y.shape) == 1 or len(y.shape) == 2 and y.shape[1] == 1
            if len(y.shape) == 2:
                y = y.reshape(-1)
            assert len(y.shape) == 1

        # fh to ForecastingHorizon
        if fh is not None:
            assert isinstance(fh, FH_TYPES)
            if not isinstance(fh, skf.ForecastingHorizon):
                fh = skf.ForecastingHorizon(fh)

        if fh is not None or not predict:
            return y, X, fh

        if X is not None and y is not None:
            n = len(X) - len(y)
            fh = skf.ForecastingHorizon(list(range(n)))
        elif X is not None:
            n = len(X)
            fh = skf.ForecastingHorizon(list(range(n)))
        else:
            raise ValueError("Why 'fh' is not defined ?")

        if X is None and self._X_history is not None:
            raise ValueError(f"predict needs X")

        return y, X, fh
    # end

    def _save_history(self, y: np.ndarray, X: Optional[np.ndarray], fh: Optional[skf.ForecastingHorizon]):
        s = len(self._slots)

        # history length == 0: none is necessary
        if s == 0:
            self._X_history = X_EMPTY
            self._y_history = Y_EMPTY
            return

        # data used with
        #   predict(fh)
        #   predict(fh, X)
        if X is not None:
            self._X_history = X[-s:]
        if y is not None:
            self._y_history = y[-s:]
    # end

    def _prepare_data(self,
                      y: np.ndarray,
                      X: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:

        slots: LagSlots = self._slots
        s = len(slots)

        if X is not None:
            assert len(y) == len(X)

        sx = len(slots.input_slots)
        sy = len(slots.target_slots)
        n = len(y)
        m = 0 if X is None else X.shape[1]

        nt = n - s
        mt = sx * m + sy
        tx = np.zeros((nt, mt))
        ty = np.zeros(nt)

        if X is None:
            for i in range(nt):
                c = 0
                for j in reversed(slots.target_slots):
                    tx[i, c] = y[s + i - j]
                    c += 1
                ty[i] = y[s + i]
            # end
        else:
            for i in range(nt):
                c = 0
                for j in reversed(slots.input_slots):
                    tx[i, c:c + m] = X[s + i - j]
                    c += m
                for j in reversed(slots.target_slots):
                    tx[i, c] = y[s + i - j]
                    c += 1
                ty[i] = y[s + i]
            # end

        return tx, ty
    # end

    @staticmethod
    def _prediction_length(
            fh: Union[skf.ForecastingHorizon, Sized],
            y: Optional[np.ndarray],
            X: Optional[np.ndarray]) -> int:

        # Prediction cases
        #
        #   params      past        future      pred len
        #   fh          yh          -           |fh|
        #   X           Xh, yh      X           |X|
        #   fh, X       Xh, yh      X           |fh|
        #   fh, y       y           -           |fh|
        #   y, X        X[:y], y    X[y:]       |X|-|y|
        #   fh, y, X    X[:y], y    X[y:]       |fh|
        #

        # compute the 'prediction length', number of future slots to generate
        if fh is not None:
            return len(fh)
        if X is not None and y is not None:
            return len(X) - len(y)
        if X is not None:
            return len(X)
        else:
            raise ValueError(f"Unable to compute the prediction length: fh or X are not specified")
    # end

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------

# end
