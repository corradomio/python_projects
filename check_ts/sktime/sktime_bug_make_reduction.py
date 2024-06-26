import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import make_reduction

x = np.arange(1, 101)+0.1
y = np.arange(1, 101)+0.2
z = x + y

df = pd.DataFrame({'x': x, 'y': y, 'z': z})
tr = df[:80]
te = df[80:]

f = make_reduction(LinearRegression(), window_length=1, strategy='recursive')
f.fit(X=tr[['x', 'y']], y=tr['z'])

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

