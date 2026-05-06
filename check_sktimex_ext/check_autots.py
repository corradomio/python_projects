# Example using 'amazon/chronos-t5-tiny' model
from matplotlib import pyplot as plt
from sktime.datasets import load_airline
from sktimex.forecasting.autots import AutoTS
from sktime.split import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series

y = load_airline()
y_train, y_test = temporal_train_test_split(y)
n = len(y_test)
fh = ForecastingHorizon(list(range(1,n+1)), is_relative=True)
forecaster = AutoTS(
    pred_len=6,
    # frequency='MS',
    verbose=-1
)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)

plot_series(y_train, y_test, y_pred, labels=["train", "test", "pred"])
plt.show()
