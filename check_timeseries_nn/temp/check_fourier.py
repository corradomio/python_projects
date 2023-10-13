#
# https://medium.com/@aysuudemiir/mastering-time-series-forecasting-revealing-the-power-of-fourier-terms-in-arima-d34a762be1ce
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# Generate synthetic time series data with daily seasonality
np.random.seed(42)
n = 365  # Number of data points
t = np.arange(n)
seasonality = 7  # Weekly seasonality
data = 50 + 10 * np.sin(2 * np.pi * t / seasonality) + np.random.normal(0, 5, n)

# Convert the data to a pandas DataFrame
df = pd.DataFrame({'Date': pd.date_range(start='2020-01-01', periods=n), 'Value': data})
df.set_index('Date', inplace=True)

# Create Fourier terms for weekly seasonality
def create_fourier_terms(t, period, num_terms):
    terms = []
    for i in range(1, num_terms + 1):
        terms.append(np.sin(2 * np.pi * i * t / period))
        terms.append(np.cos(2 * np.pi * i * t / period))
    return np.column_stack(terms)

num_fourier_terms = 4
fourier_terms = create_fourier_terms(t, seasonality, num_fourier_terms)

# Fit the ARIMA model using pmdarima's auto_arima with Fourier terms as exogenous variables
model = auto_arima(df['Value'], exogenous=fourier_terms[:n], seasonal=True, suppress_warnings=True)
model.fit(df['Value'], exogenous=fourier_terms[:n])

# Forecast future values with the fitted model
forecast_steps =30
forecast_exog = create_fourier_terms(np.arange(n, n + forecast_steps), seasonality, num_fourier_terms)

# Get the forecast for the future steps with exogenous variables
forecast_df = pd.DataFrame(forecast_exog, columns=[f'Fourier_{i+1}' for i in range(num_fourier_terms * 2)])
#forecast_values = model.predict(n_periods=forecast_steps)
forecast_values = model.predict(n_periods=forecast_steps, exogenous=forecast_df)

# Retrieve the index for forecasting
forecast_index = pd.date_range(start='2021-01-01', periods=forecast_steps)

# Plot the original data and the forecasted values
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Value'], label='Original Data')
plt.plot(forecast_index, forecast_values, label='Forecasted Values', color='red')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('ARIMA with Fourier Terms Forecast')
plt.legend()
plt.show()