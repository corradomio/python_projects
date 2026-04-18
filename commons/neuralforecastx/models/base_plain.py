from typing import Optional, Union

import neuralforecast.losses.pytorch as nflp
import numpy as np
import pandas as pd
from neuralforecast.tsdataset import TimeSeriesDataset
from sktime.forecasting.base import ForecastingHorizon

from sktimex.forecasting.base import ScaledForecaster
from stdlib.qname import create_from

# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

ACTIVATION_FUNCTIONS = {
    None: None,

    "relu": "ReLU",
    "selu": "SELU",
    "leakyrelu": "LeakyReLU",
    "prelu": "PReLU",
    "gelu": "gelu",
    "tanh": "Tanh",
    "softplus": "Softplus",
    "sigmoid": "Sigmoid",

    "ReLU": "ReLU",
    "SELU": "SELU",
    "LeakyReLU": "LeakyReLU",
    "PReLU": "PReLU",
    "Tanh": "Tanh",
    "Softplus": "Softplus",
    "Sigmoid": "Sigmoid",
}


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

NF_LOSSES = {
    None: nflp.MSE,
    "mae": nflp.MAE,
    "mse": nflp.MSE,
    "rmse": nflp.RMSE,
    "mape": nflp.MAPE,
    "smape": nflp.SMAPE,
    "mase": nflp.MASE,
    "relmse": nflp.relMSE,
    "quatileloss": nflp.QuantileLoss,
    "mqloss": nflp.MQLoss,
    "huberloss": nflp.HuberLoss,
    "huberqloss": nflp.HuberQLoss,
    "hubermqloss": nflp.HuberMQLoss,
    "distributionloss": nflp.DistributionLoss
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
# unique_id, ds, y

def to_tsds(
        y: Union[pd.Series, list[pd.Series]],
        X: Union[None, pd.DataFrame, list[pd.DataFrame]],
        ignores_exogenous_X: bool
) -> Optional[TimeSeriesDataset]:
    assert isinstance(y, (pd.Series, list))
    assert isinstance(X, (type(None), pd.DataFrame, list))

    if ignores_exogenous_X:
        X = None

    if isinstance(y, list):
        y = _concat_ser(y)

    if isinstance(X, list):
        X = _concat_df(X)

    if X is None:
        index = y.index
        df = pd.DataFrame({
            "ds": index.to_series(),
            "y": y.values,
            "unique_id": 1
        })
    else:
        index = X.index
        df = pd.DataFrame({
            "ds": index.to_series(),
            "y": y.values,
            "unique_id": 1
        })
        df = pd.concat([df, X], axis=1, ignore_index=False)
    # end

    if isinstance(df["ds"].dtype, pd.PeriodDtype):
        freq = index.freq
        df["ds"] = df["ds"].map(lambda t: t.to_timestamp(freq=freq))
    # end

    df.reset_index(drop=True, inplace=True)
    dataset, indices, dates, ds = TimeSeriesDataset.from_df(df)
    return dataset


def _concat_ser(slist: list[pd.Series]) -> pd.Series:
    slist = [s for s in slist if s is not None]
    if len(slist) == 1:
        return slist[0]
    else:
        return pd.concat(slist, axis=0)


def _concat_df(dflist: list[pd.DataFrame]) -> Optional[pd.DataFrame]:
    dflist = [df for df in dflist if df is not None]
    if len(dflist) == 0:
        return None
    if len(dflist) == 1:
        return dflist[0]
    else:
        return pd.concat(dflist, axis=0)


def _compose_ser(y_next: np.ndarray, y_pred: pd.Series, y_past: pd.Series, fh: ForecastingHorizon) -> pd.Series:
    n_pred = 0 if y_pred is None else len(y_pred)
    n_next = len(y_next)
    index_next = fh.to_pandas()[n_pred:n_pred+n_next]
    y_next_ser = pd.Series(data=y_next.reshape(-1), index=index_next, name=y_past.name)

    return y_next_ser if y_pred is None else pd.concat([y_pred, y_next_ser], axis=0)
# end


# ---------------------------------------------------------------------------
# BaseNFForecaster
# ---------------------------------------------------------------------------

def _freqstr(index: pd.Index) -> Optional[str]:
    if hasattr(index, "freqstr"):
        return index.freqstr
    else:
        return 1

def _setattr(obj, a, v):
    if a not in ["scaler", "h"]:
        assert not hasattr(obj, a), f"Attribute {a} already present"
    setattr(obj, a, v)


def _parse_kwargs(kwargs: dict) -> dict:
    # nf_kwargs = to_py_types(kwargs)
    nf_kwargs = {}|kwargs
    nf_keys = list(nf_kwargs.keys())

    for k in nf_keys:
        if k == "scaler":
            del nf_kwargs["scaler"]
        elif k == "pl_trainer_kwargs":
            nf_kwargs |= nf_kwargs["pl_trainer_kwargs"]
            del nf_kwargs["pl_trainer_kwargs"]
        elif k in ["loss", "valid_loss"]:
            # nf_kwargs[k] = loss_from(nf_kwargs[k])
            nf_kwargs[k] = create_from(kwargs[k], aliases=NF_LOSSES)
            del kwargs[k]

    return nf_kwargs


# ---------------------------------------------------------------------------

class _BaseNFForecaster(ScaledForecaster):
    # default tag values - these typically make the "safest" assumption
    # for more extensive documentation, see extension_templates/forecasting.py
    _tags = {
        # estimator type
        # --------------
        "object_type": "forecaster",  # type of object
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        # "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "capability:exogenous": True,  # does estimator ignore the exogeneous X?
        "capability:insample": False,  # can the estimator make in-sample predictions?
        "capability:pred_int": False,  # can the estimator produce prediction intervals?
        "capability:pred_int:insample": False,  # if yes, also for in-sample horizons?
        # "handles-missing-data": False,  # can estimator handle missing data?
        "capability:missing_values": False,   # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit/_predict, support for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit/_predict, support for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": False,  # is fit empty and can be skipped?
    }

    def __init__(self, nf_class, locals: dict):
        super().__init__()

        self._nf_class = nf_class

        self.h = 1
        self._val_size = 0
        self._data_kwargs = {}

        self._model = None
        self._freq = None
        self._kwargs = {}

        self._ignores_exogenous_X = not self.get_tag("capability:exogenous", False)
        self._future_exogenous_X = self.get_tag("capability:future-exogenous", False, False)
        self._analyze_locals(locals)
    # end

    def _analyze_locals(self, locals: dict):

        assert "kwargs" not in locals

        if "trainer_kwargs" in locals.keys():
            locals = locals | locals["trainer_kwargs"]

        keys = list(locals.keys())
        for k in keys:
            if k in ["self", "__class__", "trainer_kwargs"]:
                del locals[k]
            else:
                _setattr(self, k, locals[k])

        self._kwargs = locals
    # end

    # -----------------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------------

    def get_param_names(self, sort=True):
        param_names = list(self._kwargs.keys())
        if sort:
            param_names = sorted(param_names)
        return param_names

    # def get_params(self, deep=True):
    #     return super().get_params(deep=deep)
    def get_params(self, deep=True):
        params = {}
        params |= super().get_params(deep=deep)
        params |= self._kwargs
        return params

    def set_params(self, **params):
        super_params = {}
        for k in params:
            if k == "scaler":
                super_params[k] = params[k]
            else:
                self._kwargs[k] = params[k]
        return super().set_params(**super_params)

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------
    # Compatibility with neuraforecast:
    # darts: the parameters for pl.Trainer are saved in 'pl_trainer_kwargs'
    #    nf: the parameters for pl.Trainer are flat

    def _compile_model(self, y, X):
        self._freq = _freqstr(y.index)

        # create the Neuralforecast parameters (cloned)
        nf_kwargs = _parse_kwargs(self._kwargs)

        nf_kwargs["class"] = self._nf_class

        # apply special transformations to the parameters
        nf_kwargs = self._validate_kwargs(nf_kwargs, y, X)

        model = create_from(nf_kwargs)

        return model
    # end

    def _validate_kwargs(self, nf_kwargs: dict, y, X) -> dict:

        hist_exog_list = None if X is None or self._ignores_exogenous_X else list(X.columns)

        nf_kwargs["hist_exog_list"] = hist_exog_list

        return nf_kwargs
    # end

    # -----------------------------------------------------------------------

    def _fit(self, y, X, fh):

        assert (self._X is None and X is None) or (self._X is not None and X is not None)

        # y, X = self.transform(y, X)

        # create the model
        self._model = self._compile_model(y, X)

        tsds = to_tsds(y, X, self._ignores_exogenous_X)

        self._model.fit(tsds, val_size=self._val_size)
        return self

    # -----------------------------------------------------------------------

    def _predict(self, fh, X):
        assert len(fh) % self.h == 0, \
            f"ForecastingHorizon length must be a multiple than prediction_length: fh={fh}, prediction_length={self.h}"

        assert (self._X is None and X is None) or (self._X is not None and X is not None)

        fha = fh.to_absolute(self._cutoff)
        nfh = len(fh)
        l = 0
        y_rec = None
        while l < nfh:
            tsds = self._to_tsds_ext(y_rec, X)

            y_next: np.ndarray = self._model.predict(
                tsds, **self._data_kwargs
            )
            y_rec = _compose_ser(y_next, y_rec, self._y, fha)
            l = len(y_rec)
        # end
        return y_rec.astype(self._y.dtype)
    # end

    def _to_tsds_ext(self, yext, X):
        if X is not None and yext is not None:
            ny = len(yext)
            Xext = X.iloc[0:ny]
        else:
            Xext = None

        return to_tsds([self._y, yext], [self._X, Xext], self._ignores_exogenous_X)

# end
