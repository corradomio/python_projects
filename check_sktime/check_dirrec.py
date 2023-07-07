# imports necessary for this chapter
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.utils.plotting import plot_series

# data loading for illustration (see section 1 for explanation)
y = load_airline()

# -- recursive

# y_train, y_test = temporal_train_test_split(y, test_size=36)
# fh = ForecastingHorizon(y_test.index, is_relative=False)

# regressor = KNeighborsRegressor(n_neighbors=1)
# forecaster = make_reduction(regressor, window_length=15, strategy="recursive")
#
# forecaster.fit(y_train)
# y_pred = forecaster.predict(fh)
# plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"], title="recursive")
# plt.show()
# mean_absolute_percentage_error(y_test, y_pred, symmetric=False)

# -- direct

# regressor = KNeighborsRegressor(n_neighbors=1)
# forecaster = make_reduction(regressor, window_length=15, strategy="direct")
#
# forecaster.fit(y=y_train, fh=fh)
# y_pred = forecaster.predict(fh=fh)
# plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"], title="direct")
# plt.show()
# mean_absolute_percentage_error(y_test, y_pred, symmetric=False)


# -- dirrec

# regressor = KNeighborsRegressor(n_neighbors=1)
# forecaster = make_reduction(regressor, window_length=15, strategy="dirrec")
#
# forecaster.fit(y=y_train, fh=fh)
# y_pred = forecaster.predict(fh=fh)
# plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"], title="dirrec")
# plt.show()
# mean_absolute_percentage_error(y_test, y_pred, symmetric=False)


# -- dirrec/2
from sktime.forecasting.compose._reduce import DirRecTabularRegressionForecaster

y_train, y_test_true = temporal_train_test_split(y, test_size=72)
y_test, y_true = temporal_train_test_split(y_test_true, test_size=36)
fh = ForecastingHorizon(y_test.index, is_relative=False)
fh2 = ForecastingHorizon(y_test_true.index, is_relative=False)

regressor = KNeighborsRegressor(n_neighbors=1)
forecaster: DirRecTabularRegressionForecaster = make_reduction(regressor, window_length=15, strategy="dirrec")

forecaster.fit(y=y_train, fh=fh)
y_pred = forecaster.predict(fh=fh2)
plot_series(y_train, y_true, y_pred, labels=["y_train", "y_test", "y_pred"], title="dirrec")
plt.show()
mean_absolute_percentage_error(y_test, y_pred, symmetric=False)



