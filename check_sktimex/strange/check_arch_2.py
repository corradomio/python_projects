from sktime.datasets import load_airline
from sktime.forecasting.arch import ARCH
from sktime.utils.plotting import plot_series
import matplotlib.pyplot as plt

y = load_airline()

forecaster = ARCH()

forecaster.fit(y)
# ARCH(...)
print(forecaster.get_tag("requires-fh-in-fit"))

y_pred = forecaster.predict(fh=list(range(12)))


plot_series(y, y_pred, labels=["y", "y_pred"])
plt.show()
