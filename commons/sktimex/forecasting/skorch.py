#
# __all__ = [
#     'SkorchForecaster',
# ]
#
# import logging
# from typing import Sized, cast, Optional
#
# import pandas as pd
# from sktime.forecasting.base import ForecastingHorizon
#
# import skorchx
# import torchx.nn as nnx
# from .nn import *
# from ..transform.nn import NNTrainTransform, NNPredictTransform
# from ..utils import PD_TYPES, qualified_name, import_from
#
#
# # skorch.NeuralNet()
# # constructor parameters
# #
# #       module,
# #       criterion,
# #       optimizer=torch.optim.SGD,
# #       lr=0.01,
# #       max_epochs=10,
# #       batch_size=128,
# #       iterator_train=DataLoader,
# #       iterator_valid=DataLoader,
# #       dataset=Dataset,
# #       train_split=ValidSplit(5),
# #       callbacks=None,
# #       predict_nonlinearity='auto',
# #       warm_start=False,
# #       verbose=1,
# #       device='cpu',
# #       compile=False,
# #       use_caching='auto',
# #       **kwargs
# #
# # skorch.RegressorMixin
# #       no parameters
# #
# #
# # skorch.NeuralNetRegressor(NeuralNet, RegressorMixin)
# #       only override some default values
# #
# #
# # skorchx.NeuralNetRegressor(skorch.NeuralNetRegressor)
# #       only extends 'predict(X)'
# #
#
# # ---------------------------------------------------------------------------
# # ScikitForecaster
# # ---------------------------------------------------------------------------
#
# class SkorchForecaster(_BaseNNForecaster):
#
#     # -----------------------------------------------------------------------
#     # Constructor
#     # -----------------------------------------------------------------------
#     # can be passed:
#     #
#     #   1) a sktime  class name -> instantiate it
#     #   2) a sklearn class name -> instantiate it and wrap it with make_reduction
#     #   3) a sktime  instance   -> as is
#     #   4) a sklearn instance   -> wrap it with make_reduction
#     #
#     # Note: it is possible to pass the parameters in 2 way:
#     #
#     # 1)    module=[module class]
#     #       module__[param1]=[value1]
#     #
#     # or
#     #
#     # 2)    module=[module class]
#     #       module_params=dict(
#     #           [param1] = [value1]
#     #       )
#     #
#     # 1) requires to know the parameter's names
#     # 2) is more flexible
#     #
#
#     def __init__(
#         self, *,
#
#         # -- torch model
#
#         module,
#         module_kwargs: Optional[dict] = None,
#
#         # -- skorch
#
#         engine: Optional[dict] = None
#
#         # criterion=None,
#         # optimizer=None,
#         # lr=0.01,
#         # batch_size=16,
#         # max_epochs=300,
#         # callbacks=None,
#         # patience=0,
#
#         # -- extra params
#     ):
#         assert module is not None, "Parameter 'module' is mandatory"
#         assert isinstance(module, str) or issubclass(module, nnx.Module)
#
#         super().__init__(
#             # criterion=criterion,
#             # optimizer=optimizer,
#             # lr=lr,
#             # batch_size=batch_size,
#             # max_epochs=max_epochs,
#             # callbacks=callbacks,
#             # patience=patience,
#         )
#
#         # Configuration parameters [read-only]
#         # convert the module class into string to permit the serialization
#         # into a JSON file
#         #
#         # TODO: there are other configurations to make serializable
#         #
#         self.module_class = qualified_name(module)
#         self.module_kwargs = module_kwargs
#
#         # Effective parameters
#         # Note: module_params could be None, 'as_dict' ensure an empty dictionary
#         # Note: extracts parameters defined as 'module__*'
#         self._module_kwargs = module_kwargs or {}
#
#         name = self.module_class
#         self._log = logging.getLogger(f"ScikitForecaster.{name}")
#     # end
#
#     # -----------------------------------------------------------------------
#     # Properties
#     # -----------------------------------------------------------------------
#
#     def get_params(self, deep=True):
#         params = super().get_params(deep=deep) | {
#             'module': self.module_class,
#             'module_kwargs': self.module_kwargs
#         }
#         return params
#
#     # -----------------------------------------------------------------------
#     # fit
#     # -----------------------------------------------------------------------
#     # fit(y)        fit(y, X)
#     #
#     # predict(fh)
#     # predict(fh, X)        == predict(X)
#     # predict(fh, X, y)     == predict(X, y)        <== piu' NO che SI
#     #                       == fit(y, X[:y])
#     #                          predict(fh, X[y:]
#     #
#
#     def _fit(self, y, X=None, fh=None):
#
#         input_shape, output_shape = self._compute_input_output_shapes()
#
#         self._model = self._create_skorch_model(input_shape, output_shape)
#
#         yh, Xh = self.transform(y, X)
#
#         tt = NNTrainTransform(xlags=self._xlags, ylags=self._ylags, tlags=self._tlags, flatten=False)
#         Xt, yt = tt.fit_transform(X=Xh, y=yh)
#
#         self._model.fit(y=yt, X=Xt)
#         return self
#     # end
#
#     def _create_skorch_model(self, input_shape, output_shape):
#         module_class = import_from(self.module_class)
#
#         # create the model
#         try:
#             module = module_class(
#                 input_shape=input_shape,
#                 output_shape=output_shape,
#                 **self.module_kwargs
#             )
#         except Exception as e:
#             self._log.fatal("Unable to create the NN module.")
#             self._log.fatal(e)
#             self._log.fatal("The model MUST RECEIVE the parameters 'input_shape' AND 'output_shape'")
#             raise e
#
#         # create the skorch model
#         #   module: torch module
#         #   criterion:  loss function
#         #   optimizer: optimizer
#         #   lr
#         #
#         model = skorchx.NeuralNetRegressor(
#             module=module,
#             **self._skt_args
#         )
#     # end
#
#     # -----------------------------------------------------------------------
#     # predict
#     # -----------------------------------------------------------------------
#
#     def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None) -> pd.DataFrame:
#         # [BUG]
#         # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.
#         yh, Xh = self.transform(self._y, self._X)
#         _, Xs = self.transform(None, X)
#
#         # fh, fhp = self._make_fh_relative_absolute(fh)
#         # nfh = int(fh[-1])
#         nfh = len(cast(Sized, fh))
#
#         pt = NNPredictTransform(xlags=self._xlags, ylags=self._ylags, tlags=self._tlags, flatten=True)
#         ys = pt.fit(y=yh, X=Xh).transform(fh=nfh, X=Xs)
#
#         i = 0
#         while i < nfh:
#             Xt = pt.step(i)
#
#             y_pred = self._model.predict(Xt)
#
#             i = pt.update(i, y_pred)
#         # end
#
#         yp = self.inverse_transform(ys)
#         return yp
#     # end
#
#     # -----------------------------------------------------------------------
#     # Support
#     # -----------------------------------------------------------------------
#
#     def __repr__(self):
#         return f"SkorchForecaster[{self.estimator}]"
#
#     # -----------------------------------------------------------------------
#     # end
#     # -----------------------------------------------------------------------
# # end
#
