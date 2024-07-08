import torch.nn
import torch.optim
import skorchx
from ...transform import NNTrainTransform, NNPredictTransform
from ...transform import yx_lags, t_lags, compute_input_output_shapes
from ..base import ForecastingHorizon, BaseForecaster

# ---------------------------------------------------------------------------
# DartsBaseForecaster
# ---------------------------------------------------------------------------

ENGINE_DEFAULTS = dict(
    optimizer=torch.optim.Adam,
    criterion=torch.nn.MSELoss,
    batch_size=12,
    max_epochs=250,
    lr=0.0001,
)


# ---------------------------------------------------------------------------
# DartsBaseForecaster
# ---------------------------------------------------------------------------

class BaseTTSForecaster(BaseForecaster):

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
        "y_inner_mtype": ["np.ndarray"],
        "X_inner_mtype": ["np.ndarray"],
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

    def __init__(self, tts_class, locals):
        super().__init__()

        self._tts_class = tts_class
        self._lags = []
        self._tlags = []
        self._model_kwargs = {}
        self._engine_kwargs = {}

        self._analyze_locals(locals)
    # end

    def _analyze_locals(self, locals):
        for k in locals:
            if k in ['self', '__class__']:
                continue
            elif k == 'lags':
                self._lags = locals[k]
            elif k == 'tlags':
                self._tlags = locals[k]
            elif k == 'engine_kwargs':
                self._engine_kwargs = {} if locals[k] is None else locals[k]
            else:
                self._model_kwargs[k] = locals[k]
            setattr(self, k, locals[k])
        # end
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    def _fit(self, y, X=None, fh=None):
        ylags, xlags = yx_lags(self._lags)
        tlags = t_lags(self._tlags)
        input_shape, output_shape = compute_input_output_shapes(X, y, xlags, ylags, tlags)

        # create the model
        self._model = self._compile_model(input_shape, output_shape)

        tt = NNTrainTransform(
            ylags=ylags,
            xlags=xlags,
            tlags=tlags,
        )

        Xt, yt = tt.fit_transform(y=y, X=X)
        self._model.fit(Xt, yt)

        return self

    def _predict(self, fh: ForecastingHorizon, X=None):
        ylags, xlags = yx_lags(self._lags)
        tlags = t_lags(self._tlags)
        nfh = len(fh)

        pt = NNPredictTransform(
            ylags=ylags,
            xlags=xlags,
            tlags=tlags,
        )

        y_pred = pt.fit(y=self._y, X=self._X).transform(fh=fh, X=X)

        i = 0
        while i < nfh:
            # [1,36,19]
            Xp = pt.step(i)
            y_pred = self._model.predict(Xp)
            i = pt.update(i, y_pred)
        # end

        return y_pred

    def _compile_model(self, input_shape, output_shape) -> skorchx.NeuralNetRegressor:

        module = self._tts_class(
            input_shape=input_shape,
            output_shape=output_shape,
            **self._model_kwargs
        )

        model = skorchx.NeuralNetRegressor(
            module=module,
            **(ENGINE_DEFAULTS | self._engine_kwargs)
        )

        return model

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------
# end
