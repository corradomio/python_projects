import datetime as dt
from arch import arch_model
from arch.univariate.base import ARCHModelForecast

import pandasx as pdx

start = dt.datetime(2000, 1, 1)
end = dt.datetime(2014, 1, 1)
# sp500 = web.DataReader('^GSPC', 'yahoo', start=start, end=end)
# returns = 100 * sp500['Adj Close'].pct_change().dropna()

sp500 = pdx.read_data(
    "htable://gspc.xml",
    datetime=("Date", "%b %d, %Y", "B"),
    index="Date",
    ignore=["Date"]
)
# sp500.to_csv("test.csv")
returns = 100 * sp500['Adj Close'].pct_change().dropna()


am = arch_model(returns)

# res = am.fit()
split_date = dt.datetime(2010,1,1)
res = am.fit(last_obs=split_date)


print(res.summary())

forecasts: ARCHModelForecast = res.forecast(horizon=5, start=end)
assert isinstance(forecasts, ARCHModelForecast)

print(forecasts.variance.tail())

pass