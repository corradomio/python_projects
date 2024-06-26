import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.compose._reduce import RecursiveTabularRegressionForecaster

x = np.arange(1, 101)+0.1
y = np.arange(1, 101)+0.2
z = np.arange(1, 101)+0.0

df = pd.DataFrame({'x': x, 'y': y, 'z': z})
trn = df[:80]
tst = df[80:]


f: RecursiveTabularRegressionForecaster = make_reduction(LinearRegression(), window_length=2)

f.fit(X=trn[['x', 'y']], y=trn['z'])
f.predict()

#
# content of Xt and yt in '_reduce.py' line 562
#
# Xt: 3 columns ONLY instead than 5
#
# -50.0,0.0,-50.0
# -48.0,1.0,-49.0
# -46.0,2.0,-48.0
# -44.0,3.0,-47.0
# ...
#
# yt:
# -48.0
# -46.0
# -44.0
# -42.0
# ...
# .

