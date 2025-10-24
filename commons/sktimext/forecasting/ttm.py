import sktime.forecasting.ttm as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative

#
# Dependencies: accelerate
#
# fh_in_fit
#

class TinyTimeMixerForecaster(sktf.TinyTimeMixerForecaster):

    def __init__(
        self,
        prediction_length,
        model_path="ibm/TTM",
        revision="main",
        validation_split=0.2,
        config=None,
        training_args=None,
        compute_metrics=None,
        callbacks=None,
        broadcasting=False,
        use_source_package=False,
        fit_strategy="minimal",
    ):
        super().__init__(
            model_path=model_path,
            revision=revision,
            validation_split=validation_split,
            config=config,
            training_args=training_args,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            broadcasting=broadcasting,
            use_source_package=use_source_package,
            fit_strategy=fit_strategy,
        )
        self.prediction_length=prediction_length
        self._fh_in_fit = ForecastingHorizon(values=list(range(prediction_length)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred
