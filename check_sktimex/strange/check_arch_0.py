import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import arch.data.sp500


st = dt.datetime(1988, 1, 1)
en = dt.datetime(2018, 1, 1)
data = arch.data.sp500.load()

market = data["Adj Close"]

ax = market.plot()
# xlim = ax.set_xlim(market.index.min(), market.index.max())
plt.show()


# it starts the day after
returns = market.pct_change().dropna()

ax = returns.plot()
# xlim = ax.set_xlim(returns.index.min(), returns.index.max())
plt.show()

pass