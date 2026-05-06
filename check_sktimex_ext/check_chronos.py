# Example using 'amazon/chronos-t5-tiny' model
from matplotlib import pyplot as plt
from sktime.datasets import load_airline
from sktime.forecasting.chronos import ChronosForecaster
from sktime.split import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series

y = load_airline()
y_train, y_test = temporal_train_test_split(y)
fh = ForecastingHorizon(y_test.index, is_relative=False)
forecaster = ChronosForecaster(
    model_path="amazon/chronos-t5-tiny",
    config={
        "device_map": "cuda"
    }
)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)

plot_series(y_train, y_test, y_pred, labels=["train", "test", "pred"])
plt.show()
