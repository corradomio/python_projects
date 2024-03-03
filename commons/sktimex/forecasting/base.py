from sktime.forecasting.base import BaseForecaster

from numpyx.scalers import MinMaxScaler
from ..utils import to_matrix


class ExtendedBaseForecaster(BaseForecaster):
    """
    Base class for the new forecasters
    """

    def __init__(self):
        super().__init__()

    # def get_params(self, deep=True):
    #     return {} | super().get_params(deep=deep)

    def _fit(self, y, X, fh):
        ...

    def _predict(self, fh, X):
        ...

    def transform(self, y, X):
        X = to_matrix(X)
        y = to_matrix(y)
        return y, X

    def inverse_transform(self, y):
        return y
# end


class TransformForecaster(ExtendedBaseForecaster):
    """
    Apply automatically a (de)normalization to all data
    """

    def __init__(self, method=None):
        super().__init__()
        self.method = method
        self._X_scaler = None
        self._y_scaler = None

    def get_params(self, deep=True):
        params = super().get_params(deep=deep) | {
            "method": self.method
        }
        return params

    def create_scaler(self):
        if self.method == 'minmax':
            return MinMaxScaler()
        else:
            return MinMaxScaler()
    # end

    def transform(self, y, X):
        # the first time it is used to train the scalers
        # the second time the scalers are used as is
        y = to_matrix(y)
        X = to_matrix(X)

        if y is not None:
            if self._y_scaler is None:
                self._y_scaler = self.create_scaler()
                self._y_scaler.fit(y)
            y = self._y_scaler.transform(y)
        # end
        if X is not None:
            if self._X_scaler is None:
                self._X_scaler = self.create_scaler()
                self._X_scaler.fit(X)
            X = self._X_scaler.transform(X)
        return y, X
    # end

    def inverse_transform(self, y):
        if y is not None:
            y = to_matrix(y)
            y = self._y_scaler.inverse_transform(y)
        return y
    # end
# end
