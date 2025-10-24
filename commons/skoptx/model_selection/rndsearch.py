import numpy as np
import sklearn.model_selection as sklms
from sklearn.model_selection._search import (check_random_state)


class PartialParameterSampler(sklms.ParameterSampler):

    def __init__(self, param_distributions, n_iter, *, random_state=None):
        super().__init__(param_distributions, n_iter, random_state=random_state)

    def __len__(self):
        return self.n_iter*len(self.param_distributions)

    def __iter__(self):
        rng = check_random_state(self.random_state)

        if self._is_all_lists():
            candidates = []
            n_param_distributions = len(self.param_distributions)
            for i in range(self.n_iter):
                for p in range(n_param_distributions):
                    param_distributions = self.param_distributions[p]
                    param_grid = {}
                    for param in param_distributions:
                        try:
                            param_grid[param] = rng.choice(param_distributions[param])
                        except ValueError as e:
                            if '1-dimensional' not in str(e):
                                raise e
                            param_values = param_distributions[param]
                            sel = rng.randint(0, len(param_values))
                            param_grid[param] = param_values[sel]
                    # yield param_grid
                    candidates.append(param_grid)

            for candidate in candidates:
                yield candidate

        else:
            raise NotImplementedError()
# end


class RandomizedSearchCV(sklms.RandomizedSearchCV):

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            random_state=random_state,
            error_score=error_score,
            return_train_score=return_train_score
        )

    def _run_search(self, evaluate_candidates):
        """Search n_iter candidates from param_distributions"""
        evaluate_candidates(
            PartialParameterSampler(self.param_distributions, self.n_iter, random_state=self.random_state)
            # ParameterSampler(self.param_distributions, self.n_iter, random_state=self.random_state)
        )
# end
