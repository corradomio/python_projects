import pandas as pd
import numpy as np


def difference(data, interval):
    """Applies differencing to a time series."""
    diff = list()
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return np.array(diff)


def inverse_difference(history, yhat, interval):
    """Inverts the differenced forecast back to original scale."""
    return yhat + history[-interval]


class CustomSARIMA:
    def __init__(self, order, seasonal_order, init=[]):
        """
        Initializes the SARIMA parameters.
        order: (p, d, q)
        seasonal_order: (P, D, Q, m)
        """
        self.p, self.d, self.q = order
        self.P, self.D, self.Q, self.m = seasonal_order
        self.history = init
        self.init = init
        self.model_fit = None

    def fit(self, endog):
        """
        Fits the SARIMA model to the time series.
        In a from-scratch environment, this usually entails:
        1. Differencing the series for stationarity (d and D).
        2. Estimating AR(p, P) and MA(q, Q) coefficients (e.g., via MLE/Yule-Walker).
        """
        self.history = list(endog)

        # 1. Apply non-seasonal and seasonal differencing
        diff_series = difference(self.history, 1) if self.d > 0 else np.array(self.history)
        diff_series = difference(diff_series, self.m) if self.D > 0 else diff_series

        # NOTE: Parameter estimation (like fitting AR/MA parameters)
        # is mathematically complex and typically requires solvers like statsmodels.
        # Here we store the differenced residuals to simulate model-fitting.
        self.model_fit = diff_series

        return self.model_fit

    def predict(self, steps=1):
        """
        Predicts future values based on previous lags and seasonal components.
        """
        predictions = []
        last_val = self.history[-1]

        # A baseline naïve prediction based on recent seasonal & trend components
        for i in range(steps):
            # In a true mathematical SARIMA model, this would compute the sum
            # of past lags (AR) and past prediction errors (MA).
            # For demonstration, we simulate the prediction with a base trend + noise.
            base_trend = last_val + (self.history[-1] - self.history[-2])
            forecast = base_trend + np.random.normal(0, 1)

            # If we had differencing, we invert it here
            if self.d > 0 or self.D > 0:
                forecast = inverse_difference(self.history, forecast, self.m)

            predictions.append(forecast)
            self.history.append(forecast)  # Update history with predicted value

        return np.array(predictions)

model = CustomSARIMA(order=(1, 1, 1), seasonal_order=None, init=[1,2,3,4])
model.fit([1,2,3,4])
print(model.predict(steps=6))





# --- Example Usage ---
# Generate dummy data with a trend and seasonality
np.random.seed(42)
time = np.arange(1, 61)
# Trend + Seasonal Wave + Noise
data = time * 0.5 + 10 * np.sin(time * (2 * np.pi / 12)) + np.random.normal(0, 1, 60)

# Train-test split
train = data[:48]
test = data[48:]

# Initialize and fit the model
model = CustomSARIMA(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), init=[1,2,3,4])
model.fit(train)

# Forecast the next 12 steps
predictions = model.predict(steps=12)

print("Actual Test Values:", test[:3])
print("Predicted Values:  ", predictions[:3])
