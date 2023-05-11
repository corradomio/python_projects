from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from pandas import Series
from numpy import arange

from etime.linear_model import LinearForecastRegressor
from etime.skl_model import SklearnForecasterRegressor

# y = load_airline()
y = Series(arange(100))

y_train, y_test = temporal_train_test_split(y)
fh = ForecastingHorizon(y_test.index, is_relative=False)

#
# 1)
#

# forecaster = ThetaForecaster(sp=12)  # monthly seasonal periodicity
#
# forecaster.fit(y_train)
# y_pred_1 = forecaster.predict(fh)
# print(mean_absolute_percentage_error(y_test, y_pred_1))


#
# 3
#

forecaster = SklearnForecasterRegressor(
    class_name='sklearn.linear_model.LinearRegression',
    window_length=1)

forecaster.fit(y_train)
y_pred_2 = forecaster.predict(fh)
print(mean_absolute_percentage_error(y_test, y_pred_2))


#
# 2)
#

forecaster = LinearForecastRegressor(
    class_name='sklearn.linear_model.LinearRegression',
    lag=1)

forecaster.fit(y_train)
y_pred_3 = forecaster.predict(fh)
print(mean_absolute_percentage_error(y_test, y_pred_3))



