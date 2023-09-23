from sktime.forecasting.base import BaseForecaster


class ExtendedBaseForecaster(BaseForecaster):

    def __init__(self):
        super().__init__()

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------

    def predict_history(self, fh, X=None, yh=None, Xh=None):
        Xs = None
        ys = None
        if yh is not None:
            Xs = self._X
            ys = self._y
            self.update(yh, Xh)
        prediction = self.predict(fh=fh, X=X)
        if yh is not None:
            self.update(y=ys, X=Xs)
        return prediction
    # end

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------

    # def fit(self, y, X=None, fh=None):
    #     pass

    # def predict(self, fh, X=None):
    #     pass

    # def fit_predict(self, y, X=None, fh=None):
    #     ...

    # def score(self, y, X=None, fh=None):
    #     ...

    # def update(self, y, X):
    #     ...

    # def update_predict(
    #         self,
    #         y,
    #         cv=None,
    #         X=None,
    #         update_params=True,
    #         reset_forecaster=True,
    # ):
    #     ...

    # def update_predict_single(
    #     self,
    #     y=None,
    #     fh=None,
    #     X=None,
    #     update_params=True,
    # ):
    #     ...

    # -----------------------------------------------------------------------

    # def predict_quantiles(self, fh=None, X=None, alpha=None):
    #     ...

    # def predict_interval(self, fh=None, X=None, coverage=0.90):
    #     ...

    # def predict_var(self, fh=None, X=None, cov=False):
    #     ...

    # def predict_proba(self, fh=None, X=None, marginal=True):
    #     ...

    # def predict_residuals(self, y=None, X=None):
    #     ...

    # -----------------------------------------------------------------------

    # def update(self, y, X=None, update_params=True):
    #     ...

    # def update_predict(self, y, cv=None, X=None, update_params=True, reset_forecaster=True):
    #     ...

    # def update_predict_single(self, y=None, fh=None, X=None, update_params=True):
    #     ...

# end
