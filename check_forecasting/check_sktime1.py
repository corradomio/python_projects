import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# hide warnings
warnings.filterwarnings("ignore")

from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
from sktime.forecasting.base import ForecastingHorizon

y = load_airline()

# plotting for visualization
plot_series(y)

fh = np.arange(1, 37)
# fh = ForecastingHorizon(
#     pd.PeriodIndex(pd.date_range("1961-01", periods=36, freq="M")), is_relative=False
# )

from sktime.forecasting.naive import NaiveForecaster
forecaster = NaiveForecaster(strategy="last")

forecaster.fit(y)

y_pred = forecaster.predict(fh)

# plotting predictions and past data
plot_series(y, y_pred, labels=["y", "y_pred"])
plt.show()
