
__all__ = [
    'ScikitLearnForecaster',
]

import logging
from collections import defaultdict
from typing import Union, Any

import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from stdlib import kwval, kwexclude
from stdlib.qname import name_of, class_of, create_from
from .base import BaseForecaster
from ..forecasting.compose import make_reduction
from ..utils import SKTIME_NAMESPACES, SKLEARN_NAMESPACES, PD_TYPES
from ..utils import starts_with


# ---------------------------------------------------------------------------
# ScikitLearnForecaster
# ---------------------------------------------------------------------------

class ScikitLearnForecaster(BaseForecaster):

    """
    sktime's forecaster equivalent to 'make_reduction(...)'

    For a more extended version of this class, see 'LinearForecaster'
    """

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------
    # can be passed:
    #
    #   1) a sktime  class name -> instantiate it
    #   2) a sklearn class name -> instantiate it and wrap it with make_reduction
    #   3) a sktime  instance   -> as is
    #   4) a sklearn instance   -> wrap it with make_reduction

    def __init__(
        self, *,
        estimator: Union[str, type, dict] = "sklearn.linear_model.LinearRegression",
        window_length=10,
        prediction_length=1,
        **kwargs
    ):

        super().__init__()

        assert isinstance(estimator, (str, type, dict))

        # Unmodified parameters [readonly]
        self.estimator: str|dict|type = estimator
        self.window_length = window_length
        self.prediction_length = prediction_length

        self._ekwargs_names = self._set_ekwargs(kwargs)
        self._estimator = None
        self._create_estimator()

        estimator_class = class_of(self.estimator)
        name = name_of(estimator_class)
        self._log = logging.getLogger(f"sktimex.ScikitLearnForecaster.{name}")
    # end

    def _set_ekwargs(self, kwargs: dict[str, Any]) -> list[str]:
        for k in kwargs:
            assert k.startswith("estimator__"), f"'{k}' is not a valid kwarg"
            setattr(self, k, kwargs[k])
        return list(kwargs.keys())
    # end

    def _get_ekwargs(self):
        ekwargs = {}
        for k in self._ekwargs_names:
            ekwargs[k[11:]] = getattr(self, k)
        return ekwargs
    # end

    def _create_estimator(self):
        estimator_class = class_of(self.estimator)
        ekwargs: dict = {"class": self.estimator} if isinstance(self.estimator, str) else self.estimator
        ekwargs |= self._get_ekwargs()

        if starts_with(estimator_class, SKLEARN_NAMESPACES):
            window_length = self.window_length
            # window_length = kwval(kwargs, "window_length", 5)
            strategy = kwval(ekwargs, "strategy", "recursive")
            ekwargs = kwexclude(ekwargs, ["window_length", "strategy"])

            # create the scikit-learn regressor
            regressor = create_from(ekwargs)
            # create the forecaster
            self._estimator = make_reduction(regressor, window_length=window_length, strategy=strategy)
        elif starts_with(estimator_class, SKTIME_NAMESPACES):
            # create a sktime forecaster
            self._estimator = create_from(ekwargs)
        else:
            raise ValueError(f"Unsupported estimator '{estimator_class}'")
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # params |=  {
        #     'estimator': self.estimator,
        #     'prediction_length': self.prediction_length,
        #     'window_length': self.window_length
        # }
        for k in self._ekwargs_names:
            params[k] = getattr(self, k)
        return params

    def set_params(self, **params):
        self._improved_set_params(**params)
        ekwargs = self._get_ekwargs()
        self._estimator.set_params(**ekwargs)
        return self

    def _improved_set_params(self, **params):
        # REIMPLEMENTATION of super().set_params(...) to support
        #
        #   key__subkey
        #
        # with  'params[key]' a dictionary and not an object with method 'object.set_params(...)'

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        unmatched_keys = []

        nested_params = defaultdict(dict)  # grouped by prefix
        for full_key, value in params.items():
            # split full_key by first occurrence of __, if contains __
            # "key_without_dblunderscore" -> "key_without_dbl_underscore", None, None
            # "key__with__dblunderscore" -> "key", "__", "with__dblunderscore"
            key, delim, sub_key = full_key.partition("__")
            # if key not recognized, remember for suffix matching
            if key not in valid_params:
                unmatched_keys += [key]
            # if full_key contained __, collect suffix for component set_params
            elif delim:
                nested_params[key][sub_key] = value
            # if key is found and did not contain __, set self.key to the value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        # all matched params have now been set
        # reset object to clean post-init state with those params
        self.reset()

        # recurse in components
        for key, sub_params in nested_params.items():
            # [IMPROVEMENT]check for 'dict'
            if isinstance(valid_params[key], dict):
                for sub_key, sub_value in sub_params.items():
                    valid_params[key][sub_key] = sub_value
            else:
                valid_params[key].set_params(**sub_params)

        # for unmatched keys, resolve by aliasing via available __ suffixes, recurse
        if len(unmatched_keys) > 0:
            valid_params = self.get_params(deep=True)
            unmatched_params = {key: params[key] for key in unmatched_keys}

            # aliasing, syntactic sugar to access uniquely named params more easily
            aliased_params = self._alias_params(unmatched_params, valid_params)

            # if none of the parameter names change through aliasing, raise error
            if set(aliased_params) == set(unmatched_params):
                raise ValueError(
                    f"Invalid parameter keys provided to set_params of object {self}. "
                    "Check the list of available parameters "
                    "with `object.get_params().keys()`. "
                    f"Invalid keys provided: {unmatched_keys}"
                )

            # recurse: repeat matching and aliasing until no further matches found
            #   termination condition is above, "no change in keys via aliasing"
            self.set_params(**aliased_params)

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------
    # fit(y)        fit(y, X)
    #
    # predict(fh)
    # predict(fh, X)        == predict(X)
    # predict(fh, X, y)     == predict(X, y)        <== piu' NO che SI
    #                       == fit(y, X[:y])
    #                          predict(fh, X[y:])
    #

    def _fit(self, y, X, fh):
        # if fh is None, it uses self.prediction_length
        # Note: prediction length

        # ensure fh relative AND NOT None for tabular models
        fh = self._compose_tabular_fh(fh)

        self._estimator.fit(y=y, X=X, fh=fh)

        return self
    # end

    def _compose_tabular_fh(self, fh):
        # ensure fh relative AND NOT None for tabular models
        if fh is None and self.prediction_length is None:
            fh = ForecastingHorizon([1], is_relative=True)
        elif isinstance(fh, int):
            pl = fh
            fh = ForecastingHorizon(list(range(1, pl + 1)), is_relative=True)
        elif isinstance(fh, ForecastingHorizon):
            fh = fh if fh.is_relative else fh.to_relative(self.cutoff)
        elif self.prediction_length >= 1:
            pl = self.prediction_length
            fh = ForecastingHorizon(list(range(1, pl + 1)), is_relative=True)
        else:
            raise ValueError(f"Unsupported fh {fh}")
        return fh
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None) -> Union[pd.DataFrame, pd.Series]:
        super()._predict(fh, X)

        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.
        # ensure fh relative
        # fh = fh.to_relative(self.cutoff)
        # nfh = len(fh)
        # efh = ForecastingHorizon(list(range(1, nfh+1)))
        efh = fh.to_relative(self.cutoff)

        # using 'sktimex.forecasting.compose.make_reduction'
        # it is resolved the problems with predict a horizon larger than the train horizon

        y_pred: pd.Series = self._estimator.predict(fh=efh, X=X)

        # assert isinstance(y_pred, (pd.DataFrame, pd.Series))
        index = fh.to_absolute(self.cutoff).to_pandas()
        y_pred.index = index
        # y_pred = pd.Series(data=y_pred.values, index=index)
        return y_pred
    # end

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------

    def _update(self, y, X=None, update_params=True):
        try:
            self._estimator.update(y=y, X=X, update_params=False)
        except:
            pass
        return super()._update(y=y, X=X, update_params=False)

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def get_state(self) -> bytes:
        import pickle
        state: bytes = pickle.dumps(self)
        return state

    def __repr__(self):
        estimator_class = class_of(self.estimator)
        name = name_of(estimator_class)
        return f"ScikitLearnForecaster[{name}]"

    # -----------------------------------------------------------------------
    # end
    # -----------------------------------------------------------------------
# end


# Compatibility
ScikitLearnForecastRegressor = ScikitLearnForecaster

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
