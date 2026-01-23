
To use "sktime" BaseForecaster if is 'expensive': there is important overhead NOT NECESSARY
because the following features are NOT used:

    1) TS multivariate
    2) multi TS in the same DataFrame


    sktime.forecasting.base.BaseForecaster              "sktime"
        sktimex.forecasting.base.BaseForecaster         "sktimex"


In theory, it is enough to have a smarter implementation of 'sktimex.forecasting.base.BaseForecaster'
However, to "remove" interna; "y" and "X" must be done using 'clear_yX(model)' because "sktime"
doesn't support 'model.update(y=None, X=None, update_params=False)'



class BaseForecaster(Sktime_BaseForecaster):

    _tags = {
        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "X_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "scitype:y": "both",
        # "ignores-exogeneous-X": False,
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "capability:missing_values": False
    }

    """
    Base class for the new forecasters.
    """

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    # def __init__(self):
    #     super().__init__()

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------
    # Copy & paste of the parent implementation to change just
    # how y_pred is converted into y_out
    # -----------------------------------------------------------------------
    # Note: there is a problem converting a Pandas object to a numpy array and
    # back to a Pandas object.

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------

    # def fit(self, y, X=None, fh=None):
    #     ...

    # def predict(self, fh=None, X=None):
    #     ...

    # def fit_predict(self, y, X=None, fh=None):
    #     ...

    # def score(self, y, X=None, fh=None):
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

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    # def _fit(self, y, X, fh):
    #     raise NotImplementedError("abstract method")

    def _predict(self, fh, X):
        assert isinstance(fh, ForecastingHorizon), "'fh' must be a ForecastingHorizon"

    # def _update(self, y, X=None, update_params=True):
    #     return super()._update(y, X=X, update_params=update_params)

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------

    def _check_fh(self, fh, pred_int=False):
        """
        This fix an error in the fh conversion:
        IF fh is a pandas Index AND new_fh is "relative", change the
        flag into "absolute"
        """
        new_fh: ForecastingHorizon = super()._check_fh(fh, pred_int=pred_int)
        if isinstance(fh, pd.Index) and new_fh.is_relative:
            new_fh._is_relative = False
        return new_fh

# end
