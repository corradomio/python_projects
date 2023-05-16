# Python
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from pprint import pprint


# Python
df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
df.head()

# Python
m = Prophet()
m.fit(df)

# Python // forecasting horizon
future = m.make_future_dataframe(periods=365)
pprint(future.tail())

# Python
forecast = m.predict(future)
pprint(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Python
fig1 = m.plot(forecast)
plt.show()

# Python
fig2 = m.plot_components(forecast)
plt.show()

# Python
from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)
plt.show()

# Python
plot_components_plotly(m, forecast)
