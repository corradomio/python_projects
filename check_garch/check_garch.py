# Import necessary libraries
from arch import arch_model
import yfinance as yf
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------

# Download data with yfinance
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2024-04-30'

data = yf.download(ticker, start=start_date, end=end_date)

# Print the first few rows of the data
print(data.head())

# ---------------------------------------------------------------------------

# Calculate log returns
# data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

# Data preprocessing
returns = data['log_return'].dropna()

# Fit GARCH Model
am = arch_model(returns, mean='Zero', vol='GARCH', p=1, q=1)
res = am.fit(disp='off')

# Forecast volatility
forecasts = res.forecast(horizon=5)

# Print the forecasted volatility
print(forecasts.mean.iloc[-1, :])

# ---------------------------------------------------------------------------

data = data.dropna()
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Define function for model evaluation
def evaluate_model(data):
    # Fit GARCH model
    am = arch_model(data['log_return'], mean='Zero', vol='GARCH', p=1, q=1, rescale=True)
    res = am.fit(disp='off')

    # Calculate AIC and BIC
    aic = res.aic
    bic = res.bic

    # Perform backtesting
    residuals = data['log_return'] - res.conditional_volatility
    res_t = residuals / res.conditional_volatility
    backtest = (res_t**2).sum()

    # Out-of-sample testing
    data_length = len(data)
    train_size = int(0.8 * data_length)
    train_data = data[:train_size]
    test_data = data[train_size:]

    res_oos = am.fit(last_obs=train_data.index[-1], disp='off')
    forecast = res_oos.forecast(start=train_data.index[-1], horizon=len(test_data))

    # Calculate out-of-sample forecast error
    forecast_vol = forecast.residual_variance.iloc[-1, :]
    error = (test_data['log_return'] - forecast_vol).dropna()

    return aic, bic, backtest, error

# Evaluate the GARCH model using log returns data
aic, bic, backtest, forecast_error = evaluate_model(data)

# Print the evaluation results
print(f'AIC: {aic}')
print(f'BIC: {bic}')
print(f'Backtesting Result: {backtest}')
