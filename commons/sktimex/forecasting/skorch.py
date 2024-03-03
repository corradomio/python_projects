import logging

from .nn import *
from ..utils import FH_TYPES, PD_TYPES
from ..utils import as_dict, kwparams, qualified_name
from ..transform.rnn import RNNTrainTransform, RNNPredictTransform
import torchx.nn as nnx

__all__ = [
    'SkorchForecaster',
    # "ScikitForecastRegressor"
]


# skorch.NeuralNet()
# constructor parameters
#
#       module,
#       criterion,
#       optimizer=torch.optim.SGD,
#       lr=0.01,
#       max_epochs=10,
#       batch_size=128,
#       iterator_train=DataLoader,
#       iterator_valid=DataLoader,
#       dataset=Dataset,
#       train_split=ValidSplit(5),
#       callbacks=None,
#       predict_nonlinearity='auto',
#       warm_start=False,
#       verbose=1,
#       device='cpu',
#       compile=False,
#       use_caching='auto',
#       **kwargs
#
# skorch.RegressorMixin
#       no parameters
#
#
# skorch.NeuralNetRegressor(NeuralNet, RegressorMixin)
#       only override some default values
#
#
# skorchx.NeuralNetRegressor(skorch.NeuralNetRegressor)
#       only extends 'predict(X)'
#

# ---------------------------------------------------------------------------
# ScikitForecaster
# ---------------------------------------------------------------------------

class SkorchForecaster(BaseNNForecaster):

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------
    # can be passed:
    #
    #   1) a sktime  class name -> instantiate it
    #   2) a sklearn class name -> instantiate it and wrap it with make_reduction
    #   3) a sktime  instance   -> as is
    #   4) a sklearn instance   -> wrap it with make_reduction
    #
    # Note: it is possible to pass the parameters in 2 way:
    #
    # 1)    module=[module class]
    #       module__[param1]=[value1]
    #
    # or
    #
    # 2)    module=[module class]
    #       module_params=dict(
    #           [param1] = [value1]
    #       )
    #
    # 1) requires to know the parameter's names
    # 2) is more flexible
    #

    def __init__(self,
                 lags: Union[int, list, tuple, dict],
                 tlags: Union[int, list[int]],

                 flavour="default",
                 scale=True,

                 # -- model

                 module=None,
                 module_params=None,

                 # -- opt/loss

                 criterion=None,
                 optimizer=None,
                 lr=0.01,

                 batch_size=16,
                 max_epochs=300,
                 callbacks=None,

                 patience=0,
                 **kwargs):

        assert module is not None, "Parameter 'module' is mandatory"
        assert isinstance(module, str) or issubclass(module, nnx.Module)

        super().__init__(
            lags=lags,
            tlags=tlags,
            scale=scale,
            flavours=flavour,

            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            callbacks=callbacks,
            patience=patience,
            **kwargs
        )

        # convert the module class into string to permit the serialization
        # into a JSON file
        #
        # TODO: there are other configurations to make serializable
        #
        self.module_class = qualified_name(module)
        self.module_params = module_params

        self._module_params = as_dict(module_params) | kwparams(kwargs, 'module__')

        name = self.module_class
        self._log = logging.getLogger(f"ScikitForecaster.{name}")
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    def get_params(self, deep=True):
        params = super().get_params(deep=deep) | {
            'module':self.module_class,
            'module_params': self.module_params
        }
        return params

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------
    # fit(y)        fit(y, X)
    #
    # predict(fh)
    # predict(fh, X)        == predict(X)
    # predict(fh, X, y)     == predict(X, y)        <== piu' NO che SI
    #                       == fit(y, X[:y])
    #                          predict(fh, X[y:]
    #

    def _fit(self, y, X=None, fh: FH_TYPES = None):

        input_shape, output_shape = self._compute_input_output_shapes()

        self._model = self._create_skorch_model(input_shape, output_shape)

        yh, Xh = self.transform(y, X)

        tt = RNNTrainTransform(slots=self._slots, tlags=self._tlags, flatten=False)
        Xt, yt = tt.fit_transform(X=Xh, y=yh)

        self._model.fit(y=yt, X=Xt)
        return self
    # end

    def _create_skorch_model(self, input_shape, output_shape):
        module_class = import_from(self.module_class)

        # create the model
        try:
            module = module_class(
                input_shape=input_shape,
                output_shape=output_shape,
                **self.module_params
            )
        except Exception as e:
            self._log.fatal("Unable to create the NN module.")
            self._log.fatal(e)
            self._log.fatal("The model MUST RECEIVE the parameters 'input_shape' AND 'output_shape'")
            raise e

        # create the skorch model
        #   module: torch module
        #   criterion:  loss function
        #   optimizer: optimizer
        #   lr
        #
        model = skorch.NeuralNetRegressor(
            module=module,
            **self._skt_args
        )
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None) -> pd.DataFrame:
        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.
        yh, Xh = self.transform(self._y, self._X)
        _, Xs = self.transform(None, X)

        fh, fhp = self._make_fh_relative_absolute(fh)

        nfh = int(fh[-1])
        pt = RNNPredictTransform(slots=self._slots, tlags=self._tlags, flatten=True)
        ys = pt.fit(y=yh, X=Xh).transform(fh=nfh, X=Xs)

        i = 0
        while i < nfh:
            Xt = pt.step(i)

            y_pred = self._model.predict(Xt)

            i = pt.update(i, y_pred)
        # end

        ys = self.inverse_transform(ys)
        yp = self._from_numpy(ys, fhp)
        return yp
    # end

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def get_state(self) -> bytes:
        import pickle
        state: bytes = pickle.dumps(self)
        return state

    def __repr__(self):
        return f"SkorchForecaster[{self.estimator}]"

    # -----------------------------------------------------------------------
    # end
    # -----------------------------------------------------------------------
# end

