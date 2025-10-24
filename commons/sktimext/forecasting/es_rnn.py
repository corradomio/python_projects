import sktime.forecasting.es_rnn as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative

#
# fh_in_fit
#

class ESRNNForecaster(sktf.ESRNNForecaster):

    def __init__(
            self,
            prediction_length,
            hidden_size=10,
            num_layer=5,
            season1_length=12,
            season2_length=6,
            seasonality_type="single",
            window=10,
            stride=1,
            batch_size=32,
            num_epochs=1000,
            criterion=None,
            optimizer="Adam",
            lr=1e-1,
            optimizer_kwargs=None,
            criterion_kwargs=None,
            custom_dataset_train=None,
            custom_dataset_pred=None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_layer=num_layer,
            season1_length=season1_length,
            season2_length=season2_length,
            seasonality_type=seasonality_type,
            window=window,
            stride=stride,
            batch_size=batch_size,
            num_epochs=num_epochs,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            optimizer_kwargs=optimizer_kwargs,
            criterion_kwargs=criterion_kwargs,
            custom_dataset_train=custom_dataset_train,
            custom_dataset_pred=custom_dataset_pred,
        )
        self.prediction_length = prediction_length
        self._fh_in_fit = ForecastingHorizon(values=list(range(prediction_length)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def _predict(self, fh=None, X=None):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred
