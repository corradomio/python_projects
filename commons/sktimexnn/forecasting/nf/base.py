from typing import Optional

import neuralforecast as nf
import neuralforecast.losses.pytorch as nflp
import pandas as pd
import pandasx as pdx

from sktime.forecasting.base import ForecastingHorizon
from sktimex.forecasting.base import ScaledForecaster
from stdlib import import_from, create_from


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


def loss_from(loss):
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

    if isinstance(loss, str):
        loss_class = select_loss(loss)
        return loss_class()
    elif isinstance(loss, dict):
        loss_class = select_loss(loss["method"])
        loss_args = {} | loss
        del loss_args["method"]
        return loss_class(**loss_args)

# end


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
        self._trainer_kwargs = {}
        self._val_size = 0
        self._data_kwargs = {}

        self._model = None  # TS model
        self._nf = None     # NeuralForecast wrapper
        self._freq = None
        self._init_kwargs = {}
        self._kwargs = {}
        self._nfdf = None

        # self._ignores_exogenous_X = self.get_tag("ignores-exogenous-X", True)
        self._ignores_exogenous_X = not self.get_tag("capability:exogenous", False)
        # self._future_exogenous_X = self.get_tag("future-exogenous-X", False, False)
        self._future_exogenous_X = self.get_tag("capability:future-exogenous", False, False)
        self._analyze_locals(locals)
    # end

    def _analyze_locals(self, locals):

        fast_activation = self.get_tag("fast-activation", False, False)

        for k in locals:
            if k in ["self", "__class__", "scaler"]:
                continue
            elif k in ["input_size", "input_length", "window_length"]:
                self._init_kwargs["input_size"] = locals[k]
            elif k in ["h", "output_size", "output_length", "prediction_length"]:
                self._init_kwargs["h"] = locals[k]
            elif k in ["activation"]:
                self._init_kwargs[k] = ACTIVATION_FUNCTIONS[locals[k]]
            elif k in ["encoder_activation"]:
                if not fast_activation:
                    self._init_kwargs[k] = ACTIVATION_FUNCTIONS[locals[k]]
            elif k == "val_size":
                self._val_size = locals[k]
                continue
            elif k == "loss":
                loss_fun = locals[k]
                # loss = loss_from(loss_fun)
                loss = create_from(loss_fun, NF_LOSSES)
                self._init_kwargs[k] = loss
            elif k == "valid_loss":
                loss_fun = locals[k]
                loss = loss_from(loss_fun)
                self._init_kwargs[k] = loss
            elif k == "trainer_kwargs":
                trainer_kwargs = locals[k]
                self._trainer_kwargs |= trainer_kwargs
                for h in trainer_kwargs:
                    _setattr(self, h, trainer_kwargs[h])
                continue
            elif k == "data_kwargs":
                self._data_kwargs = locals[k] or {}
                continue
            elif k == "kwargs":
                kwargs = locals[k]
                self._kwargs |= kwargs
                for h in kwargs:
                    _setattr(self, h, kwargs[h])
                continue
            else:
                self._init_kwargs[k] = locals[k]
            _setattr(self, k, locals[k])
        return
    # end

    @property
    def model(self):
        return self._model

    def _compile_model(self, y, X):
        self._freq = _freqstr(y.index)

        hist_exog_list = None if X is None or self._ignores_exogenous_X else list(X.columns)

        self._model = self._nf_class(
            hist_exog_list=hist_exog_list,
            # stat_exog_list=None,
            # futr_exog_list=None,
            **(self._init_kwargs | self._trainer_kwargs)
        )

        return self._model
    # end

    def _fit(self, y, X, fh):

        y, X = self.transform(y, X)

        # create the model
        model = self._compile_model(y, X)

        # create the NF wrapper
        self._nf = nf.NeuralForecast(
            models=[model],
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
        yall = pdx.concat_series([self._y, yext])
        Xall = pdx.concat([self._X, Xext]) if self._X is not None else None
        nfdf = self._to_nfdf(yall, Xall)
        return nfdf

    def _predict(self, fh, X):
        assert len(fh) % self.h == 0, \
            "ForecastingHorizon length must be a multiple than prediction_length"

        if len(fh) == self.h:
            y_pred = self._predict_same(fh, X)
        else:
            y_pred = self._predict_recursive(fh, X)

        y_pred = self.inverse_transform(y_pred)
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
            y_rec = pdx.concat_series([y_rec, y_pred])
            l = len(y_rec)
        # end
        return y_rec.astype(self._y.dtype)
    # end

    # def _predict_long(self, fh, X):
    #     plen = self.h
    #     fha = fh.to_absolute(self.cutoff)
    #     nfh = len(fha)
    #     past_df = to_nfdf(self._y, self._X)
    #     futr_df_fh = to_pred_nfdf(fha, X)
    #     model_name = name_of(self._model)
    #
    #     predictions = []
    #     at = 0
    #     while (at+plen) <= nfh:
    #         futr_df = futr_df_fh.iloc[at:at+plen]
    #
    #         y_pred = self._nf.predict(
    #             df=past_df,
    #             futr_df=futr_df,
    #             **self._data_kwargs
    #         )
    #
    #         predictions.append(y_pred)
    #         past_df = extends_nfdf(past_df, y_pred, X, at, model_name)
    #
    #         at += plen
    #     # end
    #     return from_nfdf(predictions, fha, self._y, model_name)
# end
