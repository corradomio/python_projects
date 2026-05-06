from matplotlib import pyplot as plt
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.fbprophet import Prophet
from sktime.split import temporal_train_test_split
from sktime.utils.plotting import plot_series


# Prophet requires to have data with a pandas.DatetimeIndex
y = load_airline().to_timestamp(freq='M')
y_train, y_test = temporal_train_test_split(y)

forecaster = Prophet(
    seasonality_mode='multiplicative',
    n_changepoints=int(len(y) / 12),
    add_country_holidays={'country_name': 'Germany'},
    yearly_seasonality=True)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))
# plot_series(y_pred, y_test.index.values, y_test.values,)


plot_series(y_train, y_test, y_pred, labels=["train", "test", "pred"])
plt.show()

