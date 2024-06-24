import pandas as pd

from sktime.forecasting.base import ForecastingHorizon

dr = pd.date_range('2024-01-01', periods=2, freq='MS')
pt = pd.to_datetime('2023-12-12')

fh = ForecastingHorizon(dr)
print("isrelative:", fh.is_relative)
print(fh.is_all_out_of_sample(pt))
fh = ForecastingHorizon([1, 2])
print("isrelative:", fh.is_relative)
print(fh.is_all_out_of_sample(pt))
