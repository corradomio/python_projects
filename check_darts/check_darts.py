import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.datasets import AirPassengersDataset

series = AirPassengersDataset().load()
series_df = series.pd_dataframe()
train, val = series.split_before(pd.Timestamp("19580101"))
pass
# series.plot()
# plt.show()

# series1, series2 = series.split_after(0.75)
# series1.plot()
# series2.plot()
#
# plt.show()
#
# series_noise = TimeSeries.from_times_and_values(
#     series.time_index, np.random.randn(len(series))
# )
# (series / 2 + 20 * series_noise - 10).plot()
# plt.show()

train.plot(label="training")
val.plot(label="validation")
plt.show()


# ---------------------------------------------------------------------------

from darts.models import NaiveSeasonal
from darts.models.forecasting.arima import ARIMA

seasonal_model = ARIMA()
seasonal_model.fit(train)
seasonal_forecast = seasonal_model.predict(36)


# ---------------------------------------------------------------------------

from darts.models import NaiveDrift

drift_model = NaiveDrift()
drift_model.fit(train)
drift_forecast = drift_model.predict(36)

combined_forecast = drift_forecast + seasonal_forecast - train.last_value()

series.plot()
combined_forecast.plot(label="combined")
# drift_forecast.plot(label="drift")
plt.show()

