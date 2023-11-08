import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.utils.plotting import plot_series
from sktimex import ScikitForecastRegressor, LinearForecastRegressor


# data loading for illustration (see section 1 for explanation)
y = load_airline()

y_train, y_test_true = temporal_train_test_split(y, test_size=26)
y_test, y_true = temporal_train_test_split(y_test_true, train_size=12)
fh = ForecastingHorizon(y_test.index)
fh2 = ForecastingHorizon(y_test_true.index)

forecaster = ScikitForecastRegressor(window_length=15,
                                     #class_name='sklearn.neighbors.KNeighborsRegressor',
                                     # n_neighbors=1,
                                     strategy="dirrec")

# forecaster = LinearForecastRegressor(lags=15)


forecaster.fit(y=y_train, fh=fh)
# 1958/01 - 1960/12
y_pred = forecaster.predict(fh=fh2)
plot_series(y_train, y_test_true, y_pred, labels=["y_train", "y_test", "y_pred"], title="direct")
plt.show()





