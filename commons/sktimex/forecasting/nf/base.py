from typing import Optional

import neuralforecast as nf
import neuralforecast.losses.pytorch as nflp
import pandas as pd

# from sktime.forecasting.base import ForecastingHorizon, BaseForecaster
from ..base import ForecastingHorizon, BaseForecaster
from ...utils import import_from


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

    'relu': 'ReLU',
    'selu': 'SELU',
    'leakyrelu': 'LeakyReLU',
    'prelu': 'PReLU',
    'gelu': 'gelu',
    'tanh': 'Tanh',
    'softplus': 'Softplus',
    'sigmoid': 'Sigmoid',

    'ReLU': 'ReLU',
    'SELU': 'SELU',
    'LeakyReLU': 'LeakyReLU',
    'PReLU': 'PReLU',
    'Tanh': 'Tanh',
    'Softplus': 'Softplus',
    'Sigmoid': 'Sigmoid',
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
# Utilities
# ---------------------------------------------------------------------------
# unique_id, ds, y

def to_nfdf(y: pd.Series, X: Optional[pd.DataFrame]) -> pd.DataFrame:
    assert isinstance(y, pd.Series)
    assert isinstance(X, (type(None), pd.DataFrame))

    freq = y.index.freq
    if freq is None:
        freq = pd.infer_freq(y.index)

    # if isinstance(ds.dtype, pd.PeriodDtype):
    #     ds = ds.map(lambda t: t.to_timestamp(freq=freq))

    ydf = pd.DataFrame({
        "ds": y.index.to_series(),
        "y": y.values,
        "unique_id": 1
    })

    if isinstance(ydf['ds'].dtype, pd.PeriodDtype):
        ydf['ds'] = ydf['ds'].map(lambda t: t.to_timestamp(freq=freq))

    if X is not None:
        ydf = pd.concat([ydf, X], axis=1, ignore_index=False)

    ydf.reset_index(drop=True)
    return ydf


def from_nfdf(predictions: list[pd.DataFrame], fha: ForecastingHorizon, y_template: pd.Series, model_name) -> pd.DataFrame:
    y_name = y_template.name
    y_pred = pd.concat(predictions)
    y_pred.rename(columns={model_name: y_name}, inplace=True)
    y_pred.drop(columns=['ds'], inplace=True)
    y_pred.set_index(fha.to_pandas(), inplace=True)

    # 'unique_id' is as index OR column!
    if 'unique_id' in y_pred.columns:
        y_pred.drop(columns=['unique_id'], inplace=True)

    return y_pred


def to_futr_nfdf(fh: ForecastingHorizon, X: Optional[pd.DataFrame]):
    freq = fh.freq
    if freq is None:
        freq = pd.infer_freq(fh)

    if X is not None:
        df = pd.DataFrame({"ds": X.index.to_series(), "unique_id": 1})
        df = pd.concat([df, X], axis=1)
    else:
        df = pd.DataFrame({"ds": fh.to_pandas(), "unique_id": 1})

    if isinstance(df['ds'].dtype, pd.PeriodDtype):
        df['ds'] = df['ds'].map(lambda t: t.to_timestamp(freq=freq))

    return df


def extends_nfdf(df: pd.DataFrame, y_pred: pd.DataFrame, X: Optional[pd.DataFrame], at: int, name: str):
    n_pred = len(y_pred)

    y_pred.rename(columns={name: "y"}, inplace=True)
    y_pred.reset_index(drop=False, inplace=True)
    # y_pred['unique_id'] = unique_id

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

    def __init__(self, nf_class, locals):
        super().__init__()

        self.h = 1
        self.trainer_kwargs = {}
        self.data_kwargs = {}
        self.val_size = None

        self._nf_class = nf_class
        self._model = None
        self._freq = None
        self._init_kwargs = {}
        self._nfdf = None

        self._analyze_locals(locals)
        self._ignores_exogeneous_X = self.get_tag("ignores-exogeneous-X", False)
        pass
    # end

    def _analyze_locals(self, locals):

        fast_activation = self.get_tag('fast-activation', False, False)

        for k in locals:
            if k in ['self', '__class__']:
                continue
            elif k in ['input_size', 'input_length', 'window_length']:
                self._init_kwargs['input_size'] = locals[k]
            elif k in ['h', 'output_length', 'prediction_length']:
                self._init_kwargs['h'] = locals[k]
            elif k in ['activation']:
                self._init_kwargs[k] = ACTIVATION_FUNCTIONS[locals[k]]
            elif k in ['encoder_activation']:
                if not fast_activation:
                    self._init_kwargs[k] = ACTIVATION_FUNCTIONS[locals[k]]
            elif k == 'val_size':
                self.val_size = locals[k]
                continue
            elif k == 'loss':
                loss_fun = locals[k]
                loss_kwargs = locals['loss_kwargs']
                loss = loss_from(loss_fun, (loss_kwargs or {}))
                self._init_kwargs[k] = loss
            elif k == 'valid_loss':
                loss_fun = locals[k]
                loss_kwargs = locals['valid_loss_kwargs']
                loss = loss_from(loss_fun, (loss_kwargs or {}))
                self._init_kwargs[k] = loss
            elif k in ['loss_kwargs', 'valid_loss_kwargs']:
                pass
            elif k == 'trainer_kwargs':
                self.trainer_kwargs = locals[k] or {}
                continue
            elif k == 'data_kwargs':
                self.data_kwargs = locals[k] or {}
                continue
            else:
                self._init_kwargs[k] = locals[k]
            setattr(self, k, locals[k])
        return
    # end

    @property
    def model(self):
        return self._model

    def _compile_model(self, y, X=None):
        self._freq = y.index.freqstr

        hist_exog_list = None if X is None or self._ignores_exogeneous_X else list(X.columns)

        self._model = self._nf_class(
            stat_exog_list=None,
            hist_exog_list=hist_exog_list,
            futr_exog_list=None,
            **(self._init_kwargs | self.trainer_kwargs)
        )

        return self._model
    # end

    def _fit(self, y, X=None, fh=None):

        # create the model
        model = self._compile_model(y, X)

        # create the NF wrapper
        self._nf = nf.NeuralForecast(
            models=[model],
            freq=self._freq
        )

        # combine (y,X) in NF format
        if X is None or self._ignores_exogeneous_X:
            nf_df = to_nfdf(y, None)
        else:
            nf_df = to_nfdf(y, X)

        self._nf.fit(df=nf_df, val_size=self.val_size)
        return self

    def _predict(self, fh, X=None):
        assert len(fh) % self.h == 0, \
            "ForecastingHorizon length must be a multiple than predition_length"

        if len(fh) == self.h:
            return self._predict_same(fh, X)
        else:
            return self._predict_long(fh, X)

    def _predict_same(self, fh, X):
        fha = fh.to_absolute(self._cutoff)

        futr_df = to_futr_nfdf(fha, self._X)

        y_pred = self._nf.predict(
            futr_df=futr_df,
            **(self.data_kwargs or {})
        )

        model_name = name_of(self._model)
        return from_nfdf([y_pred], fha, self._y, model_name)

    def _predict_long(self, fh, X):
        plen = self.h
        fha = fh.to_absolute(self.cutoff)
        nfh = len(fha)
        past_df = to_nfdf(self._y, self._X)
        futr_df_fh = to_futr_nfdf(fha, X)
        model_name = name_of(self._model)

        predictions = []
        at = 0
        while (at+plen) <= nfh:
            futr_df = futr_df_fh.iloc[at:at+plen]

            y_pred = self._nf.predict(
                df=past_df,
                futr_df=futr_df,
                **(self.data_kwargs or {})
            )

            predictions.append(y_pred)
            past_df = extends_nfdf(past_df, y_pred, X, at, model_name)

            at += plen
        # end
        return from_nfdf(predictions, fha, self._y, model_name)
    # end
# end
