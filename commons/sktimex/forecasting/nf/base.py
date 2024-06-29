import neuralforecast as nf

from .losses import select_loss
from .utils import to_nfdf, from_nfdf, extends_nfdf, name_of, to_futr_nfdf
from ...forecasting.base import BaseForecaster

# Scalers
#   "identity": identity_statistics,
#   "standard": std_statistics,
#   "revin": std_statistics,
#   "robust": robust_statistics,
#   "minmax": minmax_statistics,
#   "minmax1": minmax1_statistics,
#   "invariant": invariant_statistics,


ACTIVATION_FUNCTIONS = {
    'relu': 'ReLU',
    'selu': 'SELU',
    'leakyrelu': 'LeakyReLU',
    'prelu': 'PReLU',
    'gelu': 'gelu',

    'tanh': 'Tanh',
    'softplus':'Softplus',
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

    def __init__(self, nf_class, locals):
        super().__init__()

        self.h = 1
        self.trainer_kwargs = {}
        self.data_kwargs = {}
        self.val_size = None

        self._model = None
        self._freq = None
        self._init_kwargs = {}
        self._nf_class = nf_class
        self._nfdf = None

        self._analyze_locals(locals)
    # end

    def _analyze_locals(self, locals):
        for k in locals:
            if k in ['self', '__class__']:
                continue
            elif k in ['encoder_activation', 'activation']:
                self._init_kwargs[k] = ACTIVATION_FUNCTIONS[locals[k]]
            elif k == 'val_size':
                self.val_size = locals[k]
                continue
            elif k == 'loss':
                loss_fun = locals[k]
                loss_kwargs = locals['loss_kwargs']
                loss = select_loss(loss_fun)(**(loss_kwargs or {}))
                self._init_kwargs[k] = loss
            elif k == 'valid_loss':
                loss_fun = locals[k]
                loss_kwargs = locals['valid_loss_kwargs']
                loss = select_loss(loss_fun)(**(loss_kwargs or {}))
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
        # end
    # end


    @property
    def model(self):
        return self._model

    def _compile_model(self, y, X=None):
        self._freq = y.index.freqstr

        self._model = self._nf_class(
            stat_exog_list=None,
            hist_exog_list=list(X.columns) if X is not None else None,
            futr_exog_list=None,
            **(self._init_kwargs | self.trainer_kwargs)
        )

        return self._model

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
        assert len(fh) % self.h == 0, \
            "ForecastingHorizon length must be a multiple than predition_length"

        if len(fh) == self.h:
            return self._predict_same(fh, X)
        # elif len(fh) < self.h:
        #     return self._predict_short(fh, X)
        else:
            return self._predict_long(fh, X)

    # def _predict_short(self, fh, X):
    #     fhr = fh.to_relative(self._cutoff)
    #     start = fhr[0]
    #     fhs = ForecastingHorizon(lrange(start, start+self.h))
    #
    #     y_pred = self._predict_same(fhs, X)
    #     return y_pred.iloc[:len(fh)]

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