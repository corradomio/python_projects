import pandasx as pdx
from pandasx.changes import fractional_change, fractional_update

P=3

sp500 = pdx.read_data(
    "htable://gspc.xml",
    datetime=("Date", "%b %d, %Y", "B"),
    index="Date",
    ignore=["Date"]
)
# sp500.to_csv("test.csv")
market = sp500[['Close', 'Adj Close']]
returns = fractional_change(market, periods=P).dropna()
orig = fractional_update(returns, market.iloc[0:P])

pass