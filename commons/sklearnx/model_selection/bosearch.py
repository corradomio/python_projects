from typing import Union
from random import choices

import numpy as np
import skopt
from skopt.searchcv import _get_score_names, point_asdict
from skopt.space.space import Dimension, Real, Integer, Categorical, rv_discrete
from skopt.space.transformers import Identity

from stdlib.is_instance import is_instance


# ---------------------------------------------------------------------------
# PartialOrder
# ---------------------------------------------------------------------------

class POValue:
    def __init__(self, value, distance=None):
        def hdist(x, y): return 0 if id(x) == id(y) else 1

        self.value = value
        self.distance = distance if distance is not None else hdist

    def __eq__(self, other):
        return id(self.value) == id(other.value)

    def __ne__(self, other):
        return id(self.value) == id(other.value)

    def distance(self, a, b):
        return self.distance(a, b)


class PartialOrder(Dimension):
    def __init__(self, values, distance=None):
        self.values = [POValue(v, distance) if not isinstance(v, POValue) else v for v in values]
        self.n = len(values)

    @property
    def bounds(self):
        return 0, self.n - 1

    @property
    def is_constant(self):
        return self.n == 1

    def rvs(self, n_samples=1, random_state=None):
        return choices(self.values, k=n_samples)

    def distance(self, a, b):
        return a.distance(b)


# ---------------------------------------------------------------------------
# Constant
# ---------------------------------------------------------------------------

class Const:
    def __init__(self, const):
        self.const = const

    @property
    def value(self):
        return self.const

    def __eq__(self, other):
        return id(self.const) == id(other.const)

    def __ne__(self, other):
        return id(self.const) != id(other.const)


class ConstantTransformer(skopt.space.transformers.Transformer):

    def __init__(self):
        self.value = None

    def fit(self, X):
        self.value = X[0]
        return self

    def transform(self, X):
        return [0]*len(X)

    def inverse_transform(self, Xt):
        return [self.value]*len(Xt)


class Constant(Categorical):

    def __init__(self, values, prior=None, transform=None, name=None,):
        if not isinstance(values, (tuple, list)) or len(values) > 1:
            raise ValueError(
                "values must be a list or tuple of a single element"
            )
        super().__init__(values, prior=prior, transform=transform, name=name)
        # value = self.categories[0]
        # value.__ne__ = lambda self, other: id(self) != id(other)
        # value.__eq__ = lambda self, other: id(self) == id(other)

    def __contains__(self, item):
        return id(item) == id(self.categories[0])

    def __iter__(self):
        return self.categories.__iter__()

    def set_transformer(self, transform="onehot"):
        if transform not in ["identity", "onehot"]:
            raise ValueError(f"Unsupported transformer {transform}")
        self.transform_ = transform
        if self.transform_ == 'identity':
            self.transformer = Identity()
            self.transformer.fit(self.categories)
        elif self.transform_ == 'onehot':
            self.transformer = ConstantTransformer()
            self.transformer.fit(self.categories)
        self._rvs = rv_discrete(values=(range(len(self.categories)), self.prior_))
        pass

    def rvs(self, n_samples=None, random_state=None):
        return super().rvs(n_samples=n_samples, random_state=random_state)

    def __repr__(self):
        return f"Constant({self.categories[0].__class__.__name__})"


# ---------------------------------------------------------------------------
# BayesOptSearchCV
# ---------------------------------------------------------------------------

def _search_spaces_of(param_distributions):
    def to_dimension(values):
        if is_instance(values, tuple[Union[int, np.int32, np.int64]]):
            return Integer(*values)
        elif is_instance(values, tuple[Union[float, np.float32, np.float64]]):
            return Real(*values)
        elif is_instance(values, list[str]):
            return Categorical(values)
        elif is_instance(values, (list, tuple)) and len(values) == 1:
            return Constant(values)
        elif is_instance(values, list[int]) and len(values) > 1:
            return Categorical(values)
        else:
            raise ValueError(f"Unsupported space type type {type(values)}")

    ssdict = None
    if isinstance(param_distributions, dict):
        ssdict = {}
        for pname in param_distributions.keys():
            ssdict[pname] = to_dimension(param_distributions[pname])
    elif isinstance(param_distributions, list):
        sslist = []
        for param in param_distributions:
            sslist.append(to_dimension(param))
    else:
        raise ValueError(f"Unsupported param_distributions type {type(param_distributions)}")
    return ssdict


class BayesSearchCV(skopt.searchcv.BayesSearchCV):

    def __init__(
        self,
        estimator,
        search_spaces=None,
        param_distributions=None,
        optimizer_kwargs=None,
        n_iter=50,
        scoring=None,
        fit_params=None,
        n_jobs=1,
        n_points=1,
        iid='deprecated',
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch='2*n_jobs',
        random_state=None,
        error_score='raise',
        return_train_score=False,
    ):
        if param_distributions is not None:
            search_spaces = _search_spaces_of(param_distributions)
        super().__init__(
            estimator=estimator,
            search_spaces=search_spaces,
            optimizer_kwargs=optimizer_kwargs,
            n_iter=n_iter,
            scoring=scoring,
            fit_params=fit_params,
            n_jobs=n_jobs,
            n_points=n_points,
            iid=iid,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            random_state=random_state,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_distributions = param_distributions

    def _step(
        self, search_space, optimizer, score_name, evaluate_candidates, n_points=1
    ):
        """Generate n_jobs parameters and evaluate them in parallel."""
        # get parameter values to evaluate
        params = optimizer.ask(n_points=n_points)

        # convert parameters to python native types
        # [CM/2024/03/13] fix to handle DataFrame, Series
        # params = [[np.array(v).item() for v in p] for p in params]

        # make lists into dictionaries
        params_dict = [point_asdict(search_space, p) for p in params]

        all_results = evaluate_candidates(params_dict)

        # if self.scoring is a callable, we have to wait until here
        # to get the score name
        if score_name is None:
            score_names = _get_score_names(all_results)
            if len(score_names) > 1:
                # multimetric case
                # early check to fail before lengthy computations, as
                # BaseSearchCV only performs this check *after* _run_search
                self._check_refit_for_multimetric(score_names)
                score_name = f"mean_test_{self.refit}"
            elif len(score_names) == 1:
                # single metric, or a callable self.scoring returning a dict
                # with a single value
                # In both case, we just use the score that is available
                score_name = f"mean_test_{score_names.pop()}"
            else:
                # failsafe, shouldn't happen
                raise ValueError(
                    "No score was detected after fitting. This is probably "
                    "due to a callable 'scoring' returning an empty dict."
                )

        # Feed the point and objective value back into optimizer
        # Optimizer minimizes objective, hence provide negative score
        local_results = all_results[score_name][-len(params) :]
        # return the score_name to cache it if callable refit
        # this avoids checking self.refit all the time
        return (optimizer.tell(params, [-score for score in local_results]), score_name)


BayesOptSearchCV = BayesSearchCV