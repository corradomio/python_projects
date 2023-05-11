import numpy as np
import pandas as pd
import math as m
from sklearn.linear_model import LinearRegression
from sktime.forecasting.base import ForecastingHorizon
from etime.linear_model import LinearForecastRegressor
from etime.scikit_model import ScikitForecastRegressor
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
# print("\n-- SklearnForecasterRegressor (y) --\n")

skr = ScikitForecastRegressor(
    class_name=qualified_name(LinearRegression),
    window_length=1
)
skr.fit(y=y_train)

# NO: fh is mandatory !!
# y_pred = skr.predict(X=x_test)
fh = ForecastingHorizon(list(range(215, 315)), is_relative=False)
y_pred_1 = skr.predict(fh=fh)
print(y_pred_1)
print(y_pred_1.loc[215])
print(y_pred_1.loc[314])

fh = ForecastingHorizon([215, 314], is_relative=False)
y_pred_2 = skr.predict(fh=fh)
print(y_pred_2)


pass
