import numbers
import warnings
from typing import Optional, Iterable, cast
import skopt.space.space as skoss
from skopt.space.space import Dimension, Integer, Real, Categorical
from sktime.forecasting.model_selection import ForecastingSkoptSearchCV as Sktime_ForecastingSkoptSearchCV

from stdlib.tprint import tprint, tprint_exception
from stdlib.qname import create_from
from ._base import ModelSelection


# ---------------------------------------------------------------------------
# _check_dimension_ext
# ---------------------------------------------------------------------------
# Extends the support for an enumeration composed by list of values.
# Each list is handled as a single value

def _check_dimension_ext(dimension, transform=None):
    if isinstance(dimension, Dimension):
        return dimension
    if isinstance(dimension, tuple) and 2 <= len(dimension) <= 4:
        low, high, *args = dimension
        # Check that optional distribution and base have correct types
        if (not args or isinstance(args[0], str)) and (
            len(args) < 2 or isinstance(args[1], int)
        ):
            # Infer an Integer if both bounds are Integral
            if isinstance(low, numbers.Integral) and isinstance(high, numbers.Integral):
                return Integer(int(low), int(high), *args, transform=transform)
            # Infer a Real if both bounds are Real numbers
            elif isinstance(low, numbers.Real) and isinstance(high, numbers.Real):
                return Real(float(low), float(high), *args, transform=transform)
        # warn if falling back on Categorical for tuples that look like they
        # might be an error, because there is more than one type in them
        if len(set(map(type, dimension))) > 1:
            warnings.warn(
                f"{dimension!r} was inferred to a Categorical "
                "object, but looks like a tuple for an Integer or "
                "Real dimension that was miss-spelled. Pass a list "
                "or a Categorical object to suppress this warning.",
                UserWarning,
            )
    # [CM: 2026/05/12] Extension: support for: [[v11,v12,...],[v21,v22,...], ...]
    # list converted in tuple to have it 'hashable'
    if isinstance(dimension, list) and isinstance(dimension[0], list):
        dimension = list(map(tuple, dimension))
        return Categorical(dimension, transform=transform)
    elif isinstance(dimension, Iterable):
        return Categorical(dimension, transform=transform)
    # Unconditionned so handle all cases that make it here
    raise ValueError(
        f"Invalid dimension {dimension!r}. See the "
        "documentation of check_dimension for supported values."
    )

# OVVERRIDE the original implementation of 'skopt.space.space._check_dimension(..)'
# with the implementation available HERE
skoss._check_dimension = _check_dimension_ext

# ---------------------------------------------------------------------------
# ForecastingRandomizedSearchCV
# ---------------------------------------------------------------------------

class ForecastingSkoptSearchCV(Sktime_ForecastingSkoptSearchCV, ModelSelection):
    """
    Added support to create the class using a dict/JSON object
    """

    def __init__(
        self,
        forecaster: str | dict,
        cv: str | dict,
        param_grid: Optional[dict] = None,
        param_distributions: None | dict | list[dict] = None,

        scoring=None,
        strategy: str | None = "refit",
        refit: bool = True,
        update_behaviour: str = "full_refit",

        n_iter: int = 10,
        n_points: int | None = 1,
        optimizer_kwargs: dict | None = None,

        return_n_best_forecasters: int = 1,

        error_score="nan",
        tune_by_instance=False,
        tune_by_variable=False,
        random_state: int | None = None,

        backend: str = "loky",
        backend_params=None,
        verbose: int = 0,
    ):
        # TRICK: force [1,2] to be converted into Categorical([1,2])
        import skopt.space.space as skopt_space
        skopt_space._check_dimension_old = skopt_space._check_dimension

        assert param_grid is not None or param_distributions is not None, \
            "Only one of 'param_grid' or 'param_distributions' must be not None"

        forecaster_instance = create_from(forecaster)
        cv_instance = create_from(cv)
        super().__init__(
            forecaster=forecaster_instance,
            cv=cv_instance,
            param_distributions=cast(dict|list[dict], param_distributions if param_grid is None else param_grid),
            n_iter=n_iter,
            n_points=n_points,
            scoring=scoring,
            optimizer_kwargs=optimizer_kwargs,
            strategy=strategy,
            refit=refit,
            verbose=verbose,
            return_n_best_forecasters=return_n_best_forecasters,
            random_state=random_state,
            update_behaviour=update_behaviour,
            error_score=error_score,
            tune_by_instance=tune_by_instance,
            tune_by_variable=tune_by_variable,

            backend=backend,
            backend_params=backend_params,
        )
        assert return_n_best_forecasters > 0, "Unsupported 'return_n_best_forecasters' <= 0"
        self.param_grid = param_grid
        self._forecaster_override = forecaster
        self._cv_override = cv
        self._param_distributions_override = param_distributions
        self._error_score_override = error_score
        return

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # OVERRIDE the parameters passed as configuration and NOT as Python objects
        params['forecaster'] = self._forecaster_override
        params['cv'] = self._cv_override
        params['param_distributions'] = self._param_distributions_override
        params["error_score"] = self._error_score_override
        return params

    def set_params(self, **params):
        super().set_params(**params)
        return self

    def _evaluate_step(self, y, X, optimizer, n_points, mapping=None):
        tprint(f"... _evaluate_step on {n_points} points")
        try:
            return super()._evaluate_step(y, X, optimizer, n_points, mapping)
        except Exception as e:
            tprint_exception(e)
            return None

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
