import numpy as np
import pandas as pd
import math as m
from sklearn.linear_model import LinearRegression
from sktime.forecasting.base import ForecastingHorizon
from etime.linear_model import LinearForecastRegressor
from etime.skl_model import SklearnForecasterRegressor
from stdlib import qualified_name

print(qualified_name(LinearRegression))

# ---------------------------------------------------------

x = np.arange(0., 10*m.pi, 0.1, dtype=float)
y = np.arange(-5*m.pi, +5*m.pi, 0.1, dtype=float)
z = np.sin(x+y)

d = pd.DataFrame({'x': x, 'y': y, 'z': z})
x = d[['x', 'y']]
y = d[['z']]

n = 100

x_train = x[:-n]
x_test = x[-n:]
y_train = y[:-n]
y_test = y[-n:]
# fh = ForecastingHorizon(list(range(90, 100)), is_relative=True)

# ---------------------------------------------------------
print("\n-- Test --\n")

print(y_test)

# ---------------------------------------------------------
# print("\n-- LinearRegression --\n")

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred = pd.Series(y_pred.reshape(-1), index=list(range(215, 315)))
print("y_pred_0\n", y_pred)

# ---------------------------------------------------------
# print("\n-- LinearForecastRegressor --\n")

lm = LinearForecastRegressor(
    class_name=qualified_name(LinearRegression),
    lag=0
)

lm.fit(X=x_train, y=y_train)
y_pred = lm.predict(X=x_test)
print("\ny_pred_1\n", y_pred)

# ---------------------------------------------------------
# print("\n-- SklearnForecasterRegressor (y) --\n")

skr = SklearnForecasterRegressor(
    class_name=qualified_name(LinearRegression),
    window_length=1
)
skr.fit(y=y_train)

# NO: fh is mandatory !!
# y_pred = skr.predict(X=x_test)
fh = ForecastingHorizon(list(range(215, 315)), is_relative=False)
y_pred = skr.predict(fh=fh)
print(y_pred)

print("\n--\n")

fh = ForecastingHorizon(list(range(1, 101)), is_relative=True)
y_pred = skr.predict(fh=fh)
print(y_pred)

# ---------------------------------------------------------
# print("\n-- LinearForecastRegressor (y) --\n")
lm = LinearForecastRegressor(
    class_name=qualified_name(LinearRegression),
    lag=1
)

lm.fit(y=y_train)
y_pred = lm.predict(fh=fh)
print(y_pred)

# ---------------------------------------------------------
print("\n-- SklearnForecasterRegressor (X,y) --\n")

skr = SklearnForecasterRegressor(
    class_name=qualified_name(LinearRegression),
    window_length=1
)
skr.fit(X=x_train, y=y_train)

# NO: fh is mandatory !!
# y_pred = skr.predict(X=x_test)
fh = ForecastingHorizon(list(range(215, 315)), is_relative=False)
y_pred = skr.predict(fh=fh, X=x_test)
print(y_pred)

print("\n--\n")

fh = ForecastingHorizon(list(range(1, 101)), is_relative=True)
y_pred = skr.predict(fh=fh, X=x_test)
print(y_pred)

# ---------------------------------------------------------
print("\n-- LinearForecastRegressor (X,y) --\n")
lm = LinearForecastRegressor(
    class_name=qualified_name(LinearRegression),
    lag=1
)

lm.fit(y=y_train, X=x_train)
y_pred = lm.predict(fh=fh, X=x_test)
print(y_pred)

print("\n--\n")

lm.fit(y=y_train, X=x_train)
y_pred = lm.predict(fh=fh, X=x, y=y_train)
print(y_pred)

# ---------------------------------------------------------
# ---------------------------------------------------------
