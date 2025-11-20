import sklearn.linear_model as sklm


class GammaRegressor(sklm.GammaRegressor):
    """
    Resolve a problem with the values of 'y': they can be not <=0
    """

    EPS = 0.001

    def __init__(
            self,
            *,
            alpha=1.0,
            fit_intercept=True,
            solver="lbfgs",
            max_iter=100,
            tol=1e-4,
            warm_start=False,
            verbose=0,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose
        )
        self._X_offset = 0
        self._y_offset = 0

    def fit(self, X, y, sample_weight=None):
        if X.min(axis=None) <= 0:
            self._X_offset = -X.min() + self.EPS
            X = X + self._X_offset
        if y.min(axis=None) <= 0:
            self._y_offset = -y.min() + self.EPS
            y = y + self._y_offset
        return super().fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        if self._X_offset != 0:
            X = X + self._X_offset
        yp = super().predict(X)
        if self._y_offset != 0:
            yp = yp - self._y_offset
        return yp