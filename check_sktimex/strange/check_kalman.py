from darts.datasets import AirPassengersDataset
from darts.models import KalmanForecaster
from darts.utils.timeseries_generation import datetime_attribute_timeseries
series = AirPassengersDataset().load()
# optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series
future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)
# increasing the size of the state vector
model = KalmanForecaster(dim_x=12)
model.fit(series, future_covariates=future_cov)
pred = model.predict(6, future_covariates=future_cov)
print(pred.values())
