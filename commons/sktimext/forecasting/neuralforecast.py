import sktime.forecasting.neuralforecast as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative

#
# fh_in_fit
#

class NeuralForecastRNN(sktf.NeuralForecastRNN):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred


class NeuralForecastLSTM(sktf.NeuralForecastLSTM):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred


class NeuralForecastGRU(sktf.NeuralForecastGRU):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred


class NeuralForecastDilatedRNN(sktf.NeuralForecastDilatedRNN):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred


class NeuralForecastTCN(sktf.NeuralForecastTCN):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred
