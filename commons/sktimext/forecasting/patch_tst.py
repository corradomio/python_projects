import sktime.forecasting.patch_tst as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative
from .recpred import RecursivePredict

#
# fh_in_fit
#

class PatchTSTForecaster(sktf.PatchTSTForecaster, RecursivePredict):

    def __init__(
        self,
        pred_len=1,
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
        self.pred_len=pred_len
        self._fh_in_fit = ForecastingHorizon(values=list(range(1, pred_len+1)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def predict(self, fh=None, X=None):
        fh = fix_fh_relative(fh)
        return self.recursive_predict(fh, X)
