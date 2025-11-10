from sktime.forecasting.base import BaseForecaster


class PredictFix(BaseForecaster):
    def _predic(self, fh, X):
        y_pred = super()._predict(fh, X)
        return y_pred