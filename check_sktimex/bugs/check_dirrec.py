import numpy as np
from sklearn.linear_model import LinearRegression

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction

yt = np.random.randn(70, 1)

r = make_reduction(LinearRegression(), strategy="dirrec", window_length=10)

r.fit(y=yt, fh=ForecastingHorizon([1, 2, 3]))

yp = r.predict(fh=ForecastingHorizon(list(range(1, 31))))

