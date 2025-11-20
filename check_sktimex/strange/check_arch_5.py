import matplotlib.pyplot as plt
import arch
import arch.data.sp500
from arch import arch_model
import datetime as dt

from arch.univariate.base import ARCHModelForecast

start = dt.datetime(2000,1,1)
end = dt.datetime(2014,1,1)

sp500 = arch.data.sp500.load()
sp500.plot()
plt.show()


returns = 100 * sp500['Adj Close'].pct_change().dropna()
returns.plot()
plt.show()

am = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal')

split_date = dt.datetime(2010,1,1)
res = am.fit(last_obs=split_date)

forecasts: ARCHModelForecast = res.forecast(horizon=5, start=split_date)
forecasts.variance[split_date:].plot()
plt.show()

forecasts.mean[split_date:].plot()
plt.show()
