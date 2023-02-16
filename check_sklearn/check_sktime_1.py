from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import make_reduction
# imports necessary for this chapter
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.utils.plotting import plot_series
import matplotlib.pyplot as plt

# data loading for illustration (see section 1 for explanation)
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=36)
fh = ForecastingHorizon(y_test.index, is_relative=False)


# regressor = KNeighborsRegressor(n_neighbors=1)
regressor = LinearRegression()
forecaster = make_reduction(regressor, window_length=64, strategy="recursive")

forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
plt.show()
mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
