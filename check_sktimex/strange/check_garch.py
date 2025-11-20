import matplotlib.pyplot as plt
from sktime.datasets import load_airline
from sktime.forecasting.arch import StatsForecastGARCH
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series


y = load_airline()
n = int(len(y)*0.8)
y_train, y_test = y.iloc[:n], y.iloc[n:]
forecaster = StatsForecastGARCH(p=2,q=1)
forecaster.fit(y_train)
# y_pred = forecaster.predict(fh=ForecastingHorizon(y_test.index))
y_pred = forecaster.predict(fh=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])

plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
plt.show()

