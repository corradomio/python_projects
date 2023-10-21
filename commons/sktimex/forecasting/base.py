from sktime.forecasting.base import BaseForecaster


class ExtendedBaseForecaster(BaseForecaster):

    def __init__(self):
        super().__init__()

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
# end
