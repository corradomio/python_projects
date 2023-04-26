from sktime.forecasting.compose import make_reduction

from .stdlib import import_from


class SklearnForecasterRegressor:

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self,
                 class_name: str,
                 window_length=5,
                 reduction_strategy='recursive',
                 **kwargs):

        model_class = import_from(class_name)
        regressor = model_class(**kwargs)
        self.forecaster = make_reduction(regressor, window_length=window_length, strategy=reduction_strategy)
    # end

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    def fit(self, y, X=None, fh=None):
        self.forecaster.fit(y=y, X=X, fh=fh)
        return self

    def predict(self, fh=None, X=None):
        return self.forecaster.predict(fh=fh, X=X)

    # def fit_predict(self, y, X=None, fh=None):
    #     return self.forecaster.fit_predict(y=y, X=X, fh=fh)
    #
    # def score(self, y, X=None, fh=None):
    #     return self.forecaster.score(y=y, X=X, fh=fh)

    # -----------------------------------------------------------------------

    # def predict_quantiles(self, fh=None, X=None, alpha=None):
    #     return self.forecaster.predict_quantiles(fh=fh, X=X, alpha=alpha)
    #
    # def predict_interval(self, fh=None, X=None, coverage=0.90):
    #     return self.forecaster.predict_interval(fh=fh, X=X, coverage=coverage)
    #
    # def predict_var(self, fh=None, X=None, cov=False):
    #     return self.forecaster.predict_var(fh=fh, X=X, cov=cov)
    #
    # def predict_proba(self, fh=None, X=None, marginal=True):
    #     return self.forecaster.predict_proba(fh=fh, X=X, marginal=marginal)
    #
    # def predict_residuals(self, y=None, X=None):
    #     return self.forecaster.predict_residuals(y=y, X=X)

    # -----------------------------------------------------------------------

    # def update(self, y, X=None, update_params=True):
    #     self.forecaster.update(y=y, X=X, update_params=update_params)
    #     return self
    #
    # def update_predict(self, y, cv=None, X=None, update_params=True, reset_forecaster=True):
    #     self.forecaster.update_predict(y=y, cv=cv, X=X, update_params=update_params,
    #                                    reset_forecaster=reset_forecaster)
    #     return self
    #
    # def update_predict_single(self, y=None, fh=None, X=None, update_params=True):
    #     self.forecaster.update_predict_single(y=y, fh=fh, X=X, update_params=update_params)
    #     return self

    # -----------------------------------------------------------------------
    # end
    # -----------------------------------------------------------------------
# end

