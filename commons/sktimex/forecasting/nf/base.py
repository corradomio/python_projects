from typing import Optional, Any

import pandas as pd
import neuralforecast as nf

from .utils import to_nfdf, from_nfdf, extends_nfdf, name_of
from ...forecasting.base import BaseForecaster


# ---------------------------------------------------------------------------
# Single
# ---------------------------------------------------------------------------

class NeuralForecastSingle(nf.NeuralForecast):

    def __init__(self, model, freq):
        super().__init__(
            models=[model],
            freq=freq,
            # local_scaler_type=model.scaler_type
        )


# ---------------------------------------------------------------------------
# BaseNFForecaster
# ---------------------------------------------------------------------------

class BaseNFForecaster(BaseForecaster):
    # default tag values - these typically make the "safest" assumption
    # for more extensive documentation, see extension_templates/forecasting.py
    _tags = {
        # estimator type
        # --------------
        "object_type": "forecaster",  # type of object
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "capability:insample": False,  # can the estimator make in-sample predictions?
        "capability:pred_int": False,  # can the estimator produce prediction intervals?
        "capability:pred_int:insample": False,  # if yes, also for in-sample horizons?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit/_predict, support for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit/_predict, support for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": False,  # is fit empty and can be skipped?
    }

    def __init__(self):
        super().__init__()

        self._model = None
        self._freq = None
        self.data_kwargs = {}
        self.prediction_length = 1

        self.val_size = None

        self.id_col: str = "unique_id"
        self.time_col: str = "ds"
        self.target_col: str = "y"
        self.stat_exog_list = []
        self.hist_exog_list = []
        self.futr_exog_list = []

        self._nfdf = None
    # end

    @property
    def model(self):
        return self._model

    def _compose_df(self, y: pd.Series, X: Optional[pd.DataFrame]) -> pd.DataFrame:
        self.time_col = y.index.name
        self.target_col = y.name

        ynf = pd.DataFrame(data=y)
        ynf.reset_index(inplace=True)

        if X is not None:
            self.hist_exog_list = list(X.columns)

    def _compile_model(self, y, X=None) -> Any:
        ...

    def _fit(self, y, X=None, fh=None):

        # create the model
        model = self._compile_model(y, X)

        # create the NF wrapper
        self._nf = NeuralForecastSingle(
            model=model,
            freq=self._freq
        )

        # combine (y,X) in NF format
        nf_df = to_nfdf(y, X)
        self._nf.fit(df=nf_df, val_size=self.val_size)

        return self

    def _predict(self, fh, X=None):

        nfh = len(fh)
        nf_dfp = to_nfdf(self._y, self._X)

        plen = 0
        predictions = []
        while plen < nfh:
            y_pred = self._nf.predict(
                df=None,
                static_df=None,
                futr_df=nf_dfp,
                **(self.data_kwargs or {})
            )

            # dataframe contains columns: 'unique_id', 'y'
            # it must have the same length that the prediction_window
            assert self.prediction_length == y_pred.shape[0]

            predictions.append(y_pred)
            nf_dfp = extends_nfdf(nf_dfp, y_pred, X, plen, name_of(self._nf))

            plen += self.prediction_length
        # end

        return from_nfdf(predictions, self._y, nfh)
