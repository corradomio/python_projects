from datetime import datetime
from typing import Optional, Union, Any

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode
from sktime.forecasting.base import ForecastingHorizon

from pandasx.base import PANDAS_TYPE
from sktimex.forecasting.base import ScaledForecaster

# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

TREND_MODE = {
    "linear": TrendMode.LINEAR,
    "exponential": TrendMode.EXPONENTIAL,
    None: None
}

SEASONALITY_MODE = {
    "additive": SeasonalityMode.ADDITIVE,
    "multiplicative": SeasonalityMode.MULTIPLICATIVE,
    None: None
}


MODEL_MODE = {
    "additive": ModelMode.ADDITIVE,
    "multiplicative": ModelMode.MULTIPLICATIVE,
    None: None
}

# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def to_series(S: TimeSeries) -> pd.Series:
    try:
        # Python 3.10 !!!
        return S.to_series()
    except:
        # Python 3.12 !!!
        return S.pd_series()


def to_dataframe(D: TimeSeries) -> pd.DataFrame:
    try:
        # Python 3.10 !!!
        return D.to_dataframe()
    except:
        # Python 3.12 !!!
        return D.pd_dataframe()


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

ACTIVATION_FUNCTIONS = {
    None: None,

    "relu": "ReLU",
    "rrelu": "RReLU",
    "prelu": "PReLU",
    "softplus": "Softplus",
    "tanh": "Tanh",
    "selu": "SELU",
    "leakyrelu": "LeakyReLU",
    "sigmoid": "Sigmoid",

    "ReLU": "ReLU",
    "RReLU": "RReLU",
    "PReLU": "PReLU",
    "Softplus": "Softplus",
    "Tanh": "Tanh",
    "SELU": "SELU",
    "LeakyReLU": "LeakyReLU",
    "Sigmoid": "Sigmoid",
}


PL_TRAINER_KEYS = [
    "accelerator",
    "strategy",
    "devices",
    "num_nodes",
    "precision",
    "logger",
    "callbacks",
    "fast_dev_run",
    "max_epochs",
    "min_epochs",
    "max_steps",
    "min_steps",
    "max_time",
    "limit_train_batches",
    "limit_val_batches",
    "limit_test_batches",
    "limit_predict_batches",
    "overfit_batches",
    "val_check_interval",
    "check_val_every_n_epoch",
    "num_sanity_val_steps",
    "log_every_n_steps",
    "enable_checkpointing",
    "enable_progress_bar",
    "enable_model_summary",
    "accumulate_grad_batches",
    "gradient_clip_val",
    "gradient_clip_algorithm",
    "deterministic",
    "benchmark",
    "inference_mode",
    "use_distributed_sampler",
    "profiler",
    "detect_anomaly",
    "barebones",
    "plugins",
    "sync_batchnorm",
    "reload_dataloaders_every_n_epochs",
    "default_root_dir",
    "enable_autolog_hparams",
    "model_registry"
]

# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def to_timeseries(x: Union[None, pd.Series, pd.DataFrame, np.ndarray],
                  freq: Optional[str] = None) \
        -> Optional[TimeSeries]:

    assert isinstance(x, (type(None), pd.Series, pd.DataFrame, np.ndarray))
    assert isinstance(freq, (type(None), str))

    def reindex(x: PANDAS_TYPE) -> PANDAS_TYPE:
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
        ts = TimeSeries.from_series(reindex(x), freq=_freqstr(x.index))
    elif isinstance(x, pd.DataFrame):
        ts = TimeSeries.from_dataframe(reindex(x), freq=_freqstr(x.index))
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
            y_index = pd.date_range(cutoff[0]+1, periods=len(y))
        elif type(y_template.index) == pd.PeriodIndex:
            y_index = pd.period_range(cutoff[0]+1, periods=len(y))
        elif type(y_template.index) == pd.Index or type(y_template.index) == pd.RangeIndex:
            assert (y_template.index[-1] - y_template.index[0] + 1) == len(y_template)
            # Note: "cutoff" is the LAST point in the past!!!
            start = cutoff[0]+1
            stop = start + len(y)
            y_index = pd.RangeIndex(start, stop)
        else:
            raise ValueError(f"Unsupported index type {type(y_template.index)}")

        if isinstance(y, pd.Series):
            y = y.set_axis(y_index)
        else:
            y = y.set_index(y_index)
        return y
    # end

    if isinstance(y_template, pd.Series):
        # data = reindex(ts.to_series())
        data = reindex(to_series(ts))
    elif isinstance(y_template, pd.DataFrame):
        # data = reindex(ts.to_dataframe())
        data = reindex(to_dataframe(ts))
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

def _freqstr(index: pd.Index) -> Optional[str]:
    if hasattr(index, "freqstr"):
        return index.freqstr
    else:
        return None


def _setattr(obj, a, v):
    if a not in ["scaler"]:
        assert not hasattr(obj, a)
    setattr(obj, a, v)


class _BaseDartsForecaster(ScaledForecaster):

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
        # "ignores-exogeneous-X": False,
        "capability:exogenous": True,
        # valid values: boolean True (ignores X), False (uses X in non-trivial manner)
        # CAVEAT: if tag is set to True, inner methods always see X=None
        #
        # requires-fh-in-fit = is forecasting horizon always required in fit?
        "requires-fh-in-fit": False,
        # valid values: boolean True (yes), False (no)
        # if True, raises exception in fit if fh has not been passed
    }

    # -----------------------------------------------------------------------

    def __init__(self, darts_class: type, locals: dict):
        """
        Darts uses 'pl_trainer_kwargs' to pass the parameters to 'lightning.Trainer'.
        Instead 'Neuralforecast' pass to

        :param darts_class:
        :param locals:
        """
        super().__init__(scaler=locals.get("scaler", None))

        self._darts_class = darts_class

        self._model = None
        self._init_kwargs = {}
        self._kwargs = {}

        self._ignores_exogenous_X = not self.get_tag("capability:exogenous", False)
        self._future_exogenous_X = self.get_tag("capability:future-exogenous", False, False)
        self._can_use_exogenous_X = (not self._ignores_exogenous_X) or self._future_exogenous_X
        self._analyze_locals(locals)
        pass
    # end

    def _analyze_locals(self, locals):
        self._kwargs["pl_trainer_kwargs"] = {}
        for k in locals:
            if k in ["self", "__class__", "scaler"]:
                continue
            elif k in ["pl_trainer_kwargs", "trainer_kwargs"]:
                self._kwargs["pl_trainer_kwargs"] |= locals[k]
                continue
            elif k in PL_TRAINER_KEYS:
                self._kwargs["pl_trainer_kwargs"][k] = locals[k]
                continue
            # elif k in ["input_chunk_length", "input_size", "input_length", "window_length"]:
            #     self._init_kwargs["input_chunk_length"] = locals[k]
            # elif k in ["output_chunk_length", "output_size", "output_length", "prediction_length"]:
            #     self._init_kwargs["output_chunk_length"] = locals[k]
            elif k in ["input_chunk_length"]:
                self._init_kwargs["input_chunk_length"] = locals[k]
            elif k in ["output_chunk_length"]:
                self._init_kwargs["output_chunk_length"] = locals[k]
            elif k == "activation":
                self._init_kwargs[k] = ACTIVATION_FUNCTIONS[locals[k]]
            elif k == "kwargs":
                kwargs = locals[k]
                if "fit_kwargs" not in locals:
                    self._kwargs |= kwargs
                    for h in kwargs:
                        _setattr(self, h, kwargs[h])
                    continue
                else:
                    self._init_kwargs[k] = kwargs
            elif k == "fit_kwargs":
                kwargs = locals[k]
                self._kwargs |= kwargs
                for h in kwargs:
                    _setattr(self, h, kwargs[h])
                continue
            else:
                self._init_kwargs[k] = locals[k]

            _setattr(self, k, locals[k])
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

    def _fit(self, y, X, fh):

        y, X = self.transform(y, X)

        self._y_type = type(y)

        yts = to_timeseries(y)
        Xts = to_timeseries(X) if X is not None and self._can_use_exogenous_X else None

        # create the model
        self._model = self._compile_model(y, X)

        # train the model
        if self._future_exogenous_X:
            self._model.fit(yts, future_covariates=Xts)
        elif not self._ignores_exogenous_X:
            self._model.fit(yts, past_covariates=Xts)
        else:
            self._model.fit(yts)

        return self

    def _predict(self, fh: ForecastingHorizon, X=None):

        nfh = len(fh)
        yts = to_timeseries(self._y)

        if self._ignores_exogenous_X:
            Xts = None
        elif self._X is not None and X is not None:
            X_all = pd.concat([self._X, X], axis="rows")
            Xts = to_timeseries(X_all)
        else:
            Xts = None

        if self._future_exogenous_X:
            ts_pred = self._model.predict(
                nfh,
                series=yts,
                future_covariates=Xts
            )
        elif not self._ignores_exogenous_X:
            ts_pred = self._model.predict(
                nfh,
                series=yts,
                past_covariates=Xts
            )
        else:
            ts_pred = self._model.predict(
                nfh,
                series=yts
            )

        y_pred = from_timeseries(ts_pred, self._y, X, self._cutoff)

        y_pred = self.inverse_transform(y_pred)
        return y_pred

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------
# end
