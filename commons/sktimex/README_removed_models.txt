

# ---------------------------------------------------------------------------
# LinearForecaster
# ---------------------------------------------------------------------------
#
# We suppose that the dataset is ALREADY normalized.
# the ONLY information is to know the name of the target column'
#

# class LinearForecasterOld(ExtendedBaseForecaster):
#     _tags = {
#         # to list all valid tags with description, use sktime.registry.all_tags
#         #   all_tags(estimator_types="forecaster", as_dataframe=True)
#         #
#         # behavioural tags: internal type
#         # -------------------------------
#         #
#         # y_inner_mtype, X_inner_mtype control which format X/y appears in
#         # in the inner functions _fit, _predict, etc
#         "y_inner_mtype": "pd.Series",
#         "X_inner_mtype": "pd.DataFrame",
#         # valid values: str and list of str
#         # if str, must be a valid mtype str, in sktime.datatypes.MTYPE_REGISTER
#         #   of scitype Series, Panel (panel data) or Hierarchical (hierarchical series)
#         #   in that case, all inputs are converted to that one type
#         # if list of str, must be a list of valid str specifiers
#         #   in that case, X/y are passed through without conversion if on the list
#         #   if not on the list, converted to the first entry of the same scitype
#         #
#         # scitype:y controls whether internal y can be univariate/multivariate
#         # if multivariate is not valid, applies vectorization over variables
#         "scitype:y": "univariate",
#         # valid values: "univariate", "multivariate", "both"
#         #   "univariate": inner _fit, _predict, etc, receive only univariate series
#         #   "multivariate": inner methods receive only series with 2 or more variables
#         #   "both": inner methods can see series with any number of variables
#         #
#         # capability tags: properties of the estimator
#         # --------------------------------------------
#         #
#         # ignores-exogeneous-X = does estimator ignore the exogeneous X?
#         "ignores-exogeneous-X": False,
#         # valid values: boolean True (ignores X), False (uses X in non-trivial manner)
#         # CAVEAT: if tag is set to True, inner methods always see X=None
#         #
#         # requires-fh-in-fit = is forecasting horizon always required in fit?
#         "requires-fh-in-fit": False,
#         # valid values: boolean True (yes), False (no)
#         # if True, raises exception in fit if fh has not been passed
#     }
#
#     # -----------------------------------------------------------------------
#     # Constructor
#     # -----------------------------------------------------------------------
#
#     def __init__(self,
#                  lags: Union[int, list, tuple, dict],
#                  tlags: Union[int, list],
#                  estimator: Union[str, Any] = "sklearn.linear_model.LinearRegression",
#                  flatten=False,
#                  **kwargs):
#         """
#         Sktime compatible forecaster based on Scikit models.
#         It offers the same interface to other sktime forecasters, instead than to use
#         'make_reduction'.
#         It extends the flexibility of 'make_reduction' because it is possible to
#         specify past & future lags not only as simple integers but using specific
#         list of integers to use as offset respect the timeslot to predict.
#
#         TODO: there is AN INCOMPATIBILITY to resolve:
#
#             in sktime the FIRST timeslot to predict has t=1
#             here      the FIRST timeslot to predict has t=0
#
#             THIS MUST BE RESOLVED!
#
#         There are 2 lags specifications:
#
#             lags:   lags for the past
#             tlags:  lags for the future (target lags)
#
#         The lags values can be:
#
#             int         : represent the sequence [1,2,...,n] for past lags
#                           and   the sequence [0,1,2,...,n-1] for target lags
#             list/tuple  : specific from the FIRST day to predict
#
#         :param lags:
#                 int                 same for input and target
#                 (ilag, tlag)        input lags, target lags
#                 ([ilags], [tlags])  selected input/target lags
#                 {
#                     'period_type': <period_type>,
#                     'input': {
#                         <period_type_1>: <count_1>,
#                         <period_type_2>: <count_2>,
#                         ...
#                     },
#                     'target: {
#                         <period_type_1>: <count_1>,
#                         <period_type_2>: <count_2>,
#                         ...
#                     },
#                     'current': True
#                 }
#
#         :param estimator: estimator to use. It can be
#                 - a fully qualified class name (str). The parameters to use must be passed with 'kwargs'
#                 - a Python class (type). The parameters to use must be passed with 'kwargs'
#                 - a class instance. The parameters 'kwargs' are retrieved from the instance
#         :param kwargs: parameters to pass to the estimator constructor, if necessary, or retrieved from the
#                 estimator instance
#         :param flatten: if to use a single model to predict the forecast horizon or a model for each
#                 timeslot
#         """
#         super().__init__(**kwargs)
#
#         # Unmodified parameters [readonly]
#         self.lags = lags
#         self.tlags = tlags
#         self.estimator = estimator
#         self.flatten = flatten
#
#         # Effective parameters
#         self._tlags = resolve_tlags(tlags)
#         self._slots = resolve_lags(lags)
#
#         self._estimators = {}       # one model for each 'tlag'
#
#         if isinstance(estimator, str):
#             # self.estimator = estimator
#             self._create_estimators(import_from(self.estimator))
#         elif isinstance(estimator, type):
#             self.estimator = qualified_name(estimator)
#             self._create_estimators(estimator)
#         else:
#             self.estimator = qualified_name(type(estimator))
#             self._kwargs = estimator.get_params()
#             self._create_estimators(type(estimator))
#
#         self._X = None
#         self._y = None
#
#         name = self.estimator[self.estimator.rfind('.')+1:]
#         self._log = logging.getLogger(f"LinearForecaster.{name}")
#         # self._log.info(f"Created {self}")
#     # end
#
#     def _create_estimators(self, estimator=None):
#         if self.flatten:
#             self._estimators[0] = estimator(**self._kwargs)
#         else:
#             for t in self._tlags:
#                 self._estimators[t] = estimator(**self._kwargs)
#     # end
#
#     # -----------------------------------------------------------------------
#     # Properties
#     # -----------------------------------------------------------------------
#
#     def get_params(self, deep=True):
#         params = super().get_params(deep=deep) | {
#             'lags': self.lags,
#             'tlags': self.tlags,
#             'estimator': self.estimator,
#             'flatten': self.flatten,
#         }   # | self._kwargs
#         return params
#
#     # -----------------------------------------------------------------------
#     # Data transformation
#     # -----------------------------------------------------------------------
#
#     def transform(self, y, X):
#         X = to_matrix(X)
#         y = to_matrix(y)
#         return y, X
#
#     # -----------------------------------------------------------------------
#     # fit
#     # -----------------------------------------------------------------------
#
#     def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: Optional[ForecastingHorizon] = None):
#         self._X = X
#         self._y = y
#
#         yh, Xh = self.transform(y, X)
#
#         if self.flatten:
#             self._fit_flatten(Xh, yh)
#         else:
#             self._fit_tlags(Xh, yh)
#         return self
#
#     def _fit_flatten(self, Xh, yh):
#         tt = LinearTrainTransform(slots=self._slots, tlags=self._tlags)
#         Xt, yt = tt.fit_transform(X=Xh, y=yh)
#         self._estimators[0].fit(Xt, yt)
#
#     def _fit_tlags(self, Xh, yh):
#         tlags = self._tlags
#         tt = LinearTrainTransform(slots=self._slots, tlags=tlags)
#         Xt, ytt = tt.fit_transform(X=Xh, y=yh)
#         st = len(tlags)
#
#         for i in range(st):
#             t = tlags[i]
#             yt = ytt[:, i:i+1]
#             self._estimators[t].fit(Xt, yt)
#
#     # -----------------------------------------------------------------------
#     # predict
#     # -----------------------------------------------------------------------
#     # fit(y)        fit(y, X)
#     # predict(fh)   predict(fh, X)  predict(fh, X, y)
#     #               predict(    X)  predict(    X, y)
#     #
#
#     def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
#         # [BUG]
#         # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.
#
#         fhp = fh
#         if fhp.is_relative:
#             fh = fhp
#             fhp = fh.to_absolute(self.cutoff)
#         else:
#             fh = fhp.to_relative(self.cutoff)
#
#         nfh = int(fh[-1])
#
#         if self.flatten:
#             y_pred = self._predict_flatten(X, nfh, fhp)
#         else:
#             y_pred = self._predict_tlags(X, nfh, fhp)
#
#         return y_pred
#     # end
#
#     def _predict_flatten(self, X, nfh, fhp):
#         yh, Xh = self.transform(self._y, self._X)
#         _, Xs  = self.transform(None, X)
#
#         pt = LinearPredictTransform(slots=self._slots, tlags=self._tlags)
#         yp = pt.fit(y=yh, X=Xh).transform(fh=nfh, X=Xs)  # save X,y prediction
#
#         i = 0
#         while i < nfh:
#             Xt = pt.step(i)
#
#             y_pred: np.ndarray = self._estimators[0].predict(Xt)
#
#             i = pt.update(i, y_pred)
#         # end
#
#         # add the index
#         yp = self.inverse_transform(yp)
#         y_series: pd.Series = self._from_numpy(yp, fhp)
#         return y_series
#
#     def _predict_tlags(self, X, nfh, fhp):
#         yh, Xh = self.transform(self._y, self._X)
#         _, Xs = self.transform(None, X)
#
#         pt = LinearPredictTransform(slots=self._slots, tlags=self._tlags)
#         yp = pt.fit(y=yh, X=Xh).transform(fh=nfh, X=Xs)  # save X,y prediction
#         tlags = self._tlags
#
#         i = 0
#         while i < nfh:
#             it = i
#             for t in tlags:
#                 model = self._estimators[t]
#
#                 Xt = pt.step(i)
#
#                 y_pred: np.ndarray = model.predict(Xt)
#
#                 it = pt.update(i, y_pred, t)
#             i = it
#         # end
#
#         # add the index
#         yp = self.inverse_transform(yp)
#         y_series: pd.Series = self._from_numpy(yp, fhp)
#         return y_series
#
#     def _from_numpy(self, ys, fhp):
#         ys = ys.reshape(-1)
#
#         index = pd.period_range(self.cutoff[0] + 1, periods=len(ys))
#         yp = pd.Series(ys, index=index)
#         yp = yp.loc[fhp.to_pandas()]
#         return yp.astype(float)
#
#     # -----------------------------------------------------------------------
#     # update
#     # -----------------------------------------------------------------------
#
#     def _update(self, y, X=None, update_params=True):
#         for key in self._estimators:
#             self._update_estimator(self._estimators[key], y=y, X=X, update_params=False)
#         return super()._update(y=y, X=X, update_params=False)
#
#     def _update_estimator(self, estimator, y, X=None, update_params=True):
#         try:
#             estimator.update(y=y, X=X, update_params=update_params)
#         except:
#             pass
#     # end
#
#     # -----------------------------------------------------------------------
#     # Support
#     # -----------------------------------------------------------------------
#
#     def get_state(self) -> bytes:
#         import pickle
#         state: bytes = pickle.dumps(self)
#         return state
#
#     def __repr__(self, **kwargs):
#         return f"LinearForecaster[{self.estimator}]"
#
#     # -----------------------------------------------------------------------
#     # End
#     # -----------------------------------------------------------------------
# # end


# Compatibility
# LinearForecastRegressor = LinearForecaster


# ---------------------------------------------------------------------------
# MultiLagsCNNForecaster
# ---------------------------------------------------------------------------

# class MultiLagsCNNForecaster(BaseCNNForecaster):
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     # -----------------------------------------------------------------------
#     # fit
#     # -----------------------------------------------------------------------
#
#     def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: FH_TYPES = None):
#         # self._save_history(X, y)
#
#         yh, Xh = self.transform(y, X)
#
#         input_size, output_size = self._compute_input_output_sizes()
#
#         self._model = self._create_skorch_model(input_size, output_size)
#
#         tt = CNNSlotsTrainTransform(slots=self._slots, tlags=self._tlags)
#         Xt, yt = tt.fit_transform(Xh, yh)
#
#         self._model.fit(Xt, yt)
#         return self
#     # end
#
#     # -----------------------------------------------------------------------
#     # predict
#     # -----------------------------------------------------------------------
#
#     def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
#         # [BUG]
#         # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.
#
#         yh, Xh = self.transform(self._y, self._X)
#         _, Xs = self.transform(None, X)
#
#         fh, fhp = self._make_fh_relative_absolute(fh)
#
#         nfh = int(fh[-1])
#         pt = CNNSlotsPredictTransform(slots=self._slots, tlags=self._tlags)
#         ys = pt.fit(y=yh, X=Xh, ).transform(fh=nfh, X=Xs)
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
#         ys = self.inverse_transform(ys)
#         yp = self._from_numpy(ys, fhp)
#         return yp
#     # end
#
#     # -----------------------------------------------------------------------
#     # Support
#     # -----------------------------------------------------------------------
#
#     def _compute_input_output_sizes(self):
#         # (sx, mx+my), (st, my)
#         input_shape, ouput_shape = super()._compute_input_output_shapes()
#
#         my = ouput_shape[1]
#         mx = input_shape[1] - my
#
#         return (mx, my), my
#
#     def _create_skorch_model(self, input_size, output_size):
#         mx, my = input_size
#
#         # create the torch model
#         #   input_size      this depends on xlags, |X[0]| and |y[0]|
#         #   hidden_size     2*input_size
#         #   output_size=1
#         #   num_layers=1
#         #   bias=True
#         #   batch_first=True
#         #   dropout=0
#         #   bidirectional=False
#         #   proj_size =0
#         cnn_constructor = NNX_CNN_FLAVOURS[self.flavour]
#
#         #
#         # input models
#         #
#         input_models = []
#         inner_size = 0
#
#         xlags_lists = self._slots.xlags_lists
#         for xlags in xlags_lists:
#             cnn = cnn_constructor(
#                 steps=len(xlags),
#                 input_size=mx,
#                 output_size=-1,         # disable nn.Linear layer
#                 **self._cnn_args
#             )
#             inner_size += cnn.output_size
#
#             input_models.append(cnn)
#
#         ylags_lists = self._slots.ylags_lists
#         for ylags in ylags_lists:
#             cnn = cnn_constructor(
#                 steps=len(ylags),
#                 input_size=my,
#                 output_size=-1,         # disable nn.Linear layer
#                 **self._cnn_args
#             )
#             inner_size += cnn.output_size
#
#             input_models.append(cnn)
#
#         #
#         # output model
#         #
#         output_model = nn.Linear(in_features=inner_size, out_features=output_size)
#
#         #
#         # compose the list of input models with the output model
#         #
#         inner_model = nnx.MultiInputs(input_models, output_model)
#
#         # create the skorch model
#         #   module: torch module
#         #   criterion:  loss function
#         #   optimizer: optimizer
#         #   lr
#         #
#         model = skorch.NeuralNetRegressor(
#             module=inner_model,
#             warm_start=True,
#             **self._skt_args
#         )
#         model.set_params(callbacks__print_log=PrintLog(
#             sink=logging.getLogger(str(self)).info,
#             delay=3))
#
#         return model
#     # end
#
#     def __repr__(self):
#         return f"MultiLagsCNNForecaster[{self.flavour}]"
# # end




# ---------------------------------------------------------------------------
# MultiLagsRNNForecaster
# ---------------------------------------------------------------------------

# class MultiLagsRNNForecaster(BaseRNNForecaster):
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     # -----------------------------------------------------------------------
#     # fit
#     # -----------------------------------------------------------------------
#
#     def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: FH_TYPES = None):
#         # self._save_history(X, y)
#
#         yh, Xh = self.transform(y, X)
#
#         input_size, output_size = self._compute_input_output_sizes()
#
#         self._model = self._create_skorch_model(input_size, output_size)
#
#         tt = RNNSlotsTrainTransform(slots=self._slots, tlags=self._tlags)
#         Xt, yt = tt.fit_transform(X=Xh, y=yh)
#
#         self._model.fit(Xt, yt)
#         return self
#     # end
#
#     # -----------------------------------------------------------------------
#     # predict
#     # -----------------------------------------------------------------------
#
#     def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
#         # [BUG]
#         # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.
#
#         yh, Xh = self.transform(self._y, self._X)
#         _, Xs = self.transform(None, X)
#
#         fh, fhp = self._make_fh_relative_absolute(fh)
#
#         nfh = int(fh[-1])
#         pt = RNNSlotsPredictTransform(slots=self._slots, tlags=self._tlags)
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
#         ys = self.inverse_transform(ys)
#         yp = self._from_numpy(ys, fhp)
#         return yp
#     # end
#
#     # -----------------------------------------------------------------------
#     # Support
#     # -----------------------------------------------------------------------
#
#     def _compute_input_output_sizes(self):
#         # (sx, mx+my), (st, my)
#         input_shape, ouput_shape = super()._compute_input_output_shapes()
#
#         st, my = ouput_shape
#         mx = input_shape[1] - my
#
#         return (mx, my), my*st
#
#     def _create_skorch_model(self, input_size, output_size):
#         mx, my = input_size
#
#         # create the torch model
#         #   input_size      this depends on xlags, |X[0]| and |y[0]|
#         #   hidden_size     2*input_size
#         #   output_size=1
#         #   num_layers=1
#         #   bias=True
#         #   batch_first=True
#         #   dropout=0
#         #   bidirectional=False
#         #   proj_size =0
#         rnn_constructor = NNX_RNN_FLAVOURS[self.flavour]
#
#         #
#         # input models
#         #
#         input_models = []
#         inner_size = 0
#
#         xlags_lists = self._slots.xlags_lists
#         for xlags in xlags_lists:
#             rnn = rnn_constructor(
#                 steps=len(xlags),
#                 input_size=mx,
#                 output_size=-1,         # disable nn.Linear layer
#                 **self._rnn_args
#             )
#             inner_size += rnn.output_size
#
#             input_models.append(rnn)
#
#         ylags_lists = self._slots.ylags_lists
#         for ylags in ylags_lists:
#             rnn = rnn_constructor(
#                 steps=len(ylags),
#                 input_size=my,
#                 output_size=-1,         # disable nn.Linear layer
#                 **self._rnn_args
#             )
#             inner_size += rnn.output_size
#
#             input_models.append(rnn)
#
#         #
#         # output model
#         #
#         output_model = nn.Linear(in_features=inner_size, out_features=output_size)
#
#         #
#         # compose the list of input models with the output model
#         #
#         inner_model = nnx.MultiInputs(input_models, output_model)
#
#         # create the skorch model
#         #   module: torch module
#         #   criterion:  loss function
#         #   optimizer: optimizer
#         #   lr
#         #
#         model = skorch.NeuralNetRegressor(
#             module=inner_model,
#             warm_start=True,
#             **self._skt_args
#         )
#         model.set_params(callbacks__print_log=PrintLog(
#             sink=logging.getLogger(str(self)).info,
#             delay=3))
#
#         return model
#     # end
#
#     def __repr__(self):
#         return f"MultiLagsRNNForecaster[{self.flavour}]"
# # end


