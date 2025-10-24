import sktime.forecasting.patch_tst as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative

#
# fh_in_fit
#

class PatchTSTForecaster(sktf.PatchTSTForecaster):

    def __init__(
        self,
        prediction_length,
        model_path=None,
        fit_strategy="full",
        validation_split=0.2,
        config=None,
        training_args=None,
        compute_metrics=None,
        callbacks=None,
    ):
        super().__init__(
            model_path=model_path,
            fit_strategy=fit_strategy,
            validation_split=validation_split,
            config=config,
            training_args=training_args,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        self.prediction_length=prediction_length
        self._fh_in_fit = ForecastingHorizon(values=list(range(prediction_length)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred
