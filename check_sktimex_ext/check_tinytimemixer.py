
from matplotlib import pyplot as plt
from sktime.datasets import load_airline
from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.split import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series

#
# Doesn't work
#

y = load_airline()
y_train, y_test = temporal_train_test_split(y)
fh = ForecastingHorizon(y_test.index, is_relative=False)
forecaster = TinyTimeMixerForecaster(
    model_path="NX-AI/TiRex",
    fit_strategy="full",
    config={
        "context_length": 8,
        "prediction_length": 2
    },
    training_args = {
        "max_steps": 10,
        "output_dir": "test_output",
        "per_device_train_batch_size": 4,
        "report_to": "none",
    },
)
forecaster.fit(y_train, fh=fh)
y_pred = forecaster.predict(fh)

plot_series(y_train, y_test, y_pred, labels=["train", "test", "pred"])
plt.show()
