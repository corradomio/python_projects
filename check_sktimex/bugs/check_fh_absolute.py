from sktime.forecasting.base import ForecastingHorizon
from pandas import Series

ser = Series(data=list(range(100)))
print(ser.index)
fh = ForecastingHorizon(ser.index)
print(fh.is_relative)
assert not fh.is_relative, "fh must be absolute"


