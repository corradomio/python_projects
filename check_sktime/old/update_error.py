import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.compose._reduce import DirRecTabularRegressionForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils.plotting import plot_series

y = load_airline()

y_train, y_test_true = temporal_train_test_split(y, test_size=72)
y_test, y_true = temporal_train_test_split(y_test_true, test_size=36)
fh = ForecastingHorizon(list(range(1, 13)), is_relative=True)
# fh = ForecastingHorizon(y_test.index, is_relative=False)

fh_pred = ForecastingHorizon(y_test_true.index, is_relative=False)

regressor = KNeighborsRegressor(n_neighbors=1)
forecaster: DirRecTabularRegressionForecaster = make_reduction(regressor, window_length=15, strategy="direct")

forecaster.fit(y=y_train, fh=fh)
forecaster.update(y=y_test)
y_pred = forecaster.predict(fh=fh)
# forecaster.update(y=y_test, fh=fh)

plot_series(y_train, y_true, y_pred, labels=["y_train", "y_test", "y_pred"], title="dirrec")
plt.show()


