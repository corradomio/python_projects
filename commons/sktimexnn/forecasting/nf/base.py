from typing import Optional

import neuralforecast as nf
import neuralforecast.losses.pytorch as nflp
import pandas as pd
import numpy as np

from sktime.forecasting.base import ForecastingHorizon
from sktimex.forecasting.base import ScaledForecaster
from stdlib.qname import create_from


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------

#   "identity": identity_statistics,
#   "standard": std_statistics,
#   "revin": std_statistics,
#   "robust": robust_statistics,
#   "minmax": minmax_statistics,
#   "minmax1": minmax1_statistics,
#   "invariant": invariant_statistics,


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


# def loss_from(loss):
#     def select_loss(loss):
#         if loss in NF_LOSSES:
#             return NF_LOSSES[loss]
#         elif isinstance(loss, type):
#             return loss
#         elif isinstance(loss, str):
#             return import_from(loss)
#         else:
#             raise ValueError(f"Loss type {loss} not supported")
#     #
#
#     if loss is None:
#         return None
#     elif isinstance(loss, str):
#         loss_class = select_loss(loss)
#         return loss_class()
#     elif isinstance(loss, dict):
#         loss_class = select_loss(loss["method"])
#         loss_args = {} | loss
#         del loss_args["method"]
#         return loss_class(**loss_args)
#
# # end


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
# unique_id, ds, y

def to_nfdf(y: pd.Series, X: Optional[pd.DataFrame]) -> pd.DataFrame:
    assert isinstance(y, pd.Series)
    assert isinstance(X, (type(None), pd.DataFrame))

    ydf = pd.DataFrame({
        "ds": y.index.to_series(),
        "y": y.values,
        "unique_id": 1
    })

    if isinstance(ydf["ds"].dtype, pd.PeriodDtype):
        freq = y.index.freq
        ydf["ds"] = ydf["ds"].map(lambda t: t.to_timestamp(freq=freq))

    if X is not None:
        ydf = pd.concat([ydf, X], axis=1, ignore_index=False)

    ydf.reset_index(drop=True)
    return ydf


def from_nfdf(predictions: list[pd.DataFrame], fha: ForecastingHorizon, y_template: pd.Series, model_name) -> pd.DataFrame:
    index = fha.to_pandas() if isinstance(fha, ForecastingHorizon) else fha
    y_name = y_template.name
    y_pred = pd.concat(predictions)
    y_pred.rename(columns={model_name: y_name}, inplace=True)
    y_pred.drop(columns=["ds"], inplace=True)
    y_pred.set_index(index, inplace=True)

    # "unique_id" is as index OR column!
    if "unique_id" in y_pred.columns:
        y_pred.drop(columns=["unique_id"], inplace=True)

    return y_pred[y_name]


def to_pred_nfdf(fh: ForecastingHorizon, X: Optional[pd.DataFrame]):
    if X is None:
        return None

    freq = fh.freq
    if freq is None:
        freq = pd.infer_freq(fh)

    if X is not None:
        df = pd.DataFrame({"ds": X.index.to_series(), "unique_id": 1})
        df = pd.concat([df, X], axis=1)
    else:
        df = pd.DataFrame({"ds": fh.to_pandas(), "unique_id": 1})

    if isinstance(df["ds"].dtype, pd.PeriodDtype):
        df["ds"] = df["ds"].map(lambda t: t.to_timestamp(freq=freq))

    return df


def extends_nfdf(df: pd.DataFrame, y_pred: pd.DataFrame, X: Optional[pd.DataFrame], at: int, name: str):
    n_pred = len(y_pred)

    y_pred.rename(columns={name: "y"}, inplace=True)
    y_pred.reset_index(drop=False, inplace=True)
    # y_pred["unique_id"] = unique_id

    # 1) if X is available, expand y_pred with X
    if X is not None:
        X_pred = X.iloc[at: at+n_pred]
        # X_pred.reset_index(drop=False, inplace=True)
        y_pred.set_index(X_pred.index, inplace=True)
        y_pred = pd.concat([y_pred, X_pred], axis=1)
    else:
        pass
    # 2) concat df with y_pred
    df = pd.concat([df, y_pred], axis=0, ignore_index=True)
    return df


def name_of(model):
    return model.__class__.__name__ if model.alias is None else model.alias



def concat_ser(slist) -> pd.Series:
    slist = [s for s in slist if s is not None]
    if len(slist) == 1:
        return slist[0]
    else:
        return pd.concat(slist, axis=0)


def concat_df(slist) -> pd.Series:
    slist = [s for s in slist if s is not None]
    if len(slist) == 1:
        return slist[0]
    else:
        return pd.concat(slist, axis=0)


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

    def __init__(self, nf_class, locals):
        super().__init__()

        self._nf_class = nf_class

        self.h = 1
        self._val_size = 0
        self._data_kwargs = {}

        self._model = None  # TS model
        self._nf = None     # NeuralForecast wrapper
        self._freq = None
        self._kwargs = {}
        self._nfdf = None

        self._ignores_exogenous_X = not self.get_tag("capability:exogenous", False)
        self._future_exogenous_X = self.get_tag("capability:future-exogenous", False, False)
        self._analyze_locals(locals)
    # end

    def _analyze_locals(self, locals):

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

        hist_exog_list = None if X is None or self._ignores_exogenous_X else list(X.columns)

        # create the Neuralforecast parameters
        nf_kwargs = _parse_kwargs(self._kwargs)

        # model = self._nf_class(
        #     hist_exog_list=hist_exog_list,
        #     # stat_exog_list=None,
        #     # futr_exog_list=None,
        #     **nf_kwargs
        # )

        nf_kwargs["class"] = self._nf_class
        nf_kwargs["hist_exog_list"] = hist_exog_list

        model = create_from(nf_kwargs)

        return model
    # end

    def _fit(self, y, X, fh):

        # y, X = self.transform(y, X)

        # create the model
        self._model = self._compile_model(y, X)

        # create the NF wrapper
        self._nf = nf.NeuralForecast(
            models=[self._model],
            freq=self._freq
        )

        nf_df = self._to_nfdf(y, X)

        self._nf.fit(df=nf_df, val_size=self._val_size)
        return self

    def _to_nfdf(self, y, X):
        # combine (y,X) in NF format
        if X is None or self._ignores_exogenous_X:
            nf_df = to_nfdf(y, None)
        else:
            nf_df = to_nfdf(y, X)
        return nf_df

    def _to_nfdf_ext(self, yext, X):
        if yext is None:
            return self._to_nfdf(self._y, self._X)

        n = len(yext)
        Xext = X.iloc[:n] if X is not None else None
        yall = concat_ser([self._y, yext])
        Xall = concat_df([self._X, Xext]) if self._X is not None else None
        nfdf = self._to_nfdf(yall, Xall)
        return nfdf

    def _predict(self, fh, X):
        assert len(fh) % self.h == 0, \
            "ForecastingHorizon length must be a multiple than prediction_length"

        if len(fh) == self.h:
            y_pred = self._predict_same(fh, X)
        else:
            y_pred = self._predict_recursive(fh, X)

        # y_pred = self.inverse_transform(y_pred)
        return y_pred

    def _predict_same(self, fh, X):
        nf_df = self._to_nfdf(self._y, self._X)

        y_pred = self._nf.predict(
            df=nf_df,
            **self._data_kwargs
        )

        model_name = name_of(self._model)
        fha = fh.to_absolute(self._cutoff)
        return from_nfdf([y_pred], fha, self._y, model_name)

    def _predict_recursive(self, fh, X):
        n = len(fh)
        l = 0
        y_rec = None
        fha = fh.to_absolute(self._cutoff)
        while l<n:
            nf_df = self._to_nfdf_ext(y_rec, X)
            y_nf = self._nf.predict(
                df=nf_df,
                **self._data_kwargs
            )
            h = len(y_nf)
            model_name = name_of(self._model)
            y_pred = from_nfdf([y_nf], fha[l:l+h], self._y, model_name)
            y_rec = concat_ser([y_rec, y_pred])
            l = len(y_rec)
        # end
        return y_rec.astype(self._y.dtype)
    # end

# end
