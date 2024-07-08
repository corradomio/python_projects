from datetime import datetime
from typing import Optional, Union, Any

import numpy as np
import pandas as pd
# from ...forecasting.base import BaseForecaster
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

# from sktime.forecasting.base import ForecastingHorizon, BaseForecaster
from ..base import ForecastingHorizon, BaseForecaster
from ...utils import import_from


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

ACTIVATION_FUNCTIONS = {
    None: None,

    'relu': 'ReLU',
    'rrelu': 'RReLU',
    'prelu': 'PReLU',
    'softplus': 'Softplus',
    'tanh': 'Tanh',
    'selu': 'SELU',
    'leakyrelu': 'LeakyReLU',
    'sigmoid': 'Sigmoid',

    'ReLU': 'ReLU',
    'RReLU': 'RReLU',
    'PReLU': 'PReLU',
    'Softplus': 'Softplus',
    'Tanh': 'Tanh',
    'SELU': 'SELU',
    'LeakyReLU': 'LeakyReLU',
    'Sigmoid': 'Sigmoid',
}


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

NF_LOSSES = {
    None: None,

    # "mae": nflp.MAE,
    # "mse": nflp.MSE,
    # "rmse": nflp.RMSE,
    # "mape": nflp.MAPE,
    # "smape": nflp.SMAPE,
    # "mase": nflp.MASE,
    # "relmse": nflp.relMSE,
    # "quatileloss": nflp.QuantileLoss,
    # "mqloss": nflp.MQLoss,
    # "huberloss": nflp.HuberLoss,
    # "huberqloss": nflp.HuberQLoss,
    # "hubermqloss": nflp.HuberMQLoss,
    # "distributionloss": nflp.DistributionLoss
}


def loss_from(loss, kwars):
    def select_loss(loss):
        if loss in NF_LOSSES:
            return NF_LOSSES[loss]
        elif isinstance(loss, type):
            return loss
        elif isinstance(loss, str):
            return import_from(loss)
        else:
            raise ValueError(f"Loss type {loss} not supported")
    # end
    loss_class = select_loss(loss)
    return loss_class(**kwars)
# end


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def to_timeseries(x: Union[None, pd.Series, pd.DataFrame, np.ndarray],
                  freq: Optional[str] = None) \
        -> Optional[TimeSeries]:

    assert isinstance(x, (type(None), pd.Series, pd.DataFrame, np.ndarray))
    assert isinstance(freq, (type(None), str))

    def reindex(x: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        # create a copy!
        if isinstance(x.index, pd.PeriodIndex):
            if isinstance(x, pd.Series):
                x = x.set_axis(x.index.to_timestamp())
            else:
                x = x.set_index(x.index.to_timestamp())
        return x

    ts: Optional[TimeSeries] = None
    if x is None:
        pass
    elif isinstance(x, pd.Series):
        ts = TimeSeries.from_series(reindex(x), freq=x.index.freqstr)
    elif isinstance(x, pd.DataFrame):
        ts = TimeSeries.from_dataframe(reindex(x), freq=x.index.freqstr)
    elif isinstance(x, np.ndarray):
        ts = TimeSeries.from_values(x)
    else:
        raise ValueError(f"Unsupported data type {type(x)}")

    return ts


def from_timeseries(ts: TimeSeries, y_template, X, cutoff) -> Any:
    def reindex(y):
        if X is not None:
            y_index = X.index
        elif type(y_template.index) == pd.DatetimeIndex:
            y_index = pd.date_range(cutoff, periods=len(y))
        elif type(y_template.index) == pd.PeriodIndex:
            y_index = pd.period_range(cutoff, periods=len(y))
        else:
            raise ValueError(f"Unsupported index type {type(y_template.index)}")

        if isinstance(y, pd.Series):
            y = y.set_axis(y_index)
        else:
            y = y.set_index(y_index)
        return y
    # end

    if isinstance(y_template, pd.Series):
        data = reindex(ts.pd_series())
    elif isinstance(y_template, pd.DataFrame):
        data = reindex(ts.pd_dataframe())
    elif isinstance(y_template, np.ndarray):
        data = ts.values()
    else:
        raise ValueError(f"Unsupported type {type(y_template)}")

    return data


def fh_relative(fh: ForecastingHorizon, cutoff: datetime):
    if not fh.is_relative:
        fh = fh.to_relative(cutoff)
    return fh


# ---------------------------------------------------------------------------
# DartsBaseForecaster
# ---------------------------------------------------------------------------

class BaseDartsForecaster(BaseForecaster):

    # Each derived class must have a specifalized set of tags

    _tags = {
        # to list all valid tags with description, use sktime.registry.all_tags
        #   all_tags(estimator_types="forecaster", as_dataframe=True)
        #
        # behavioural tags: internal type
        # -------------------------------
        #
        # y_inner_mtype, X_inner_mtype control which format X/y appears in
        # in the inner functions _fit, _predict, etc
        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "X_inner_mtype": ["pd.Series", "pd.DataFrame"],
        # valid values: str and list of str
        # if str, must be a valid mtype str, in sktime.datatypes.MTYPE_REGISTER
        #   of scitype Series, Panel (panel data) or Hierarchical (hierarchical series)
        #   in that case, all inputs are converted to that one type
        # if list of str, must be a list of valid str specifiers
        #   in that case, X/y are passed through without conversion if on the list
        #   if not on the list, converted to the first entry of the same scitype
        #
        # scitype:y controls whether internal y can be univariate/multivariate
        # if multivariate is not valid, applies vectorization over variables
        "scitype:y": "both",
        # valid values: "univariate", "multivariate", "both"
        #   "univariate": inner _fit, _predict, etc, receive only univariate series
        #   "multivariate": inner methods receive only series with 2 or more variables
        #   "both": inner methods can see series with any number of variables
        #
        # capability tags: properties of the estimator
        # --------------------------------------------
        #
        # ignores-exogeneous-X = does estimator ignore the exogeneous X?
        "ignores-exogeneous-X": False,
        # valid values: boolean True (ignores X), False (uses X in non-trivial manner)
        # CAVEAT: if tag is set to True, inner methods always see X=None
        #
        # requires-fh-in-fit = is forecasting horizon always required in fit?
        "requires-fh-in-fit": False,
        # valid values: boolean True (yes), False (no)
        # if True, raises exception in fit if fh has not been passed
    }

    # -----------------------------------------------------------------------

    def __init__(self, darts_class, locals):
        super().__init__()

        self._darts_class = darts_class
        self._model = None
        self._init_kwargs = {}
        self._kwargs = {}

        self.lags = None
        self.lags_past_covariates = None

        self._ignores_exogeneous_X = self.get_tag("ignores-exogeneous-X", False)
        self._future_exogeneous_X = self.get_tag("future-exogeneous-X", False, False)
        self._analyze_locals(locals)
    # end

    def _analyze_locals(self, locals):
        for k in locals:
            if k in ['self', '__class__']:
                continue
            # elif k in ['input_chunk_length']:
            #     self._init_kwargs['input_chunk_length'] = locals[k]
            # elif k in ['output_chunk_length']:
            #     self._init_kwargs['output_chunk_length'] = locals[k]
            elif k == 'activation':
                self._init_kwargs[k] = ACTIVATION_FUNCTIONS[locals[k]]
            elif k == 'kwargs':
                self._kwargs |= locals[k]
                continue
            else:
                self._init_kwargs[k] = locals[k]
            setattr(self, k, locals[k])
        return
    #end

    def _compile_model(self, y, X=None):

        model: GlobalForecastingModel = self._darts_class(
            **(self._init_kwargs | self._kwargs)
        )

        return model
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    def _fit(self, y, X=None, fh=None):

        self._y_type = type(y)
        yts = to_timeseries(y)

        if X is not None and not self._ignores_exogeneous_X:
            covariates = to_timeseries(X)
        else:
            covariates = None

        # create the model
        self._model = self._compile_model(y, X)

        if covariates is None:
            self._model.fit(yts)
        elif not self._future_exogeneous_X:
            self._model.fit(yts, past_covariates=covariates)
        else:
            self._model.fit(yts, future_covariates=covariates)

        return self

    def _predict(self, fh: ForecastingHorizon, X=None):

        if self._X is not None and X is not None:
            X_all = pd.concat([self._X, X], axis='rows')
        elif self._X is not None:
            X_all = self._X
        else:
            X_all = None

        if X_all is not None and not self._ignores_exogeneous_X:
            covariates = to_timeseries(X_all)
        else:
            covariates = None

        nfh = len(fh)

        if covariates is None:
            ts_pred: TimeSeries = self._model.predict(nfh)
        elif not self._future_exogeneous_X:
            ts_pred: TimeSeries = self._model.predict(
                nfh,
                past_covariates=covariates,
            )
        else:
            ts_pred: TimeSeries = self._model.predict(
                nfh,
                future_covariates=covariates,
            )

        y_pred = from_timeseries(ts_pred, self._y, X, self._cutoff)
        return y_pred

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------
# end
