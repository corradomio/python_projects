import sktime.forecasting.exp_smoothing as sktf

class ExponentialSmoothing(sktf.ExponentialSmoothing):
    def _predict(self, fh, X):
        y_pred = super()._predict(fh, X)