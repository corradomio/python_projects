# https://www.kaggle.com/code/pateljay731/multivariate-time-series-forecasting


from sktime.datasets import load_longley
from sktime.forecasting.var import VAR
from sktime.utils.plotting import plot_series
import matplotlib.pyplot as plt

_, y = load_longley()

y = y.drop(columns=["UNEMP", "ARMED", "POP"])

forecaster = VAR()
forecaster.fit(y, fh=[1, 2, 3])


y_pred = forecaster.predict()


# GNP
# GNPDEFL
what = "GNP"
plot_series(y[what], y_pred[what], labels=["y", "y_pred"], title=what)
plt.show()

what = "GNPDEFL"
plot_series(y[what], y_pred[what], labels=["y", "y_pred"], title=what)
plt.show()


pass

