import warnings
import pandasx as pdx
import matplotlib.pyplot as plt
import numpy as np
from sktime.datasets import load_airline
from sktime.forecasting.arima import ARIMA
from sktime.utils.plotting import plot_series
from darts.datasets import AirPassengersDataset
from sktimex import predict_history

# hide warnings
warnings.filterwarnings("ignore")

y1 = load_airline()
y = AirPassengersDataset().load().pd_dataframe()

y, y_test = pdx.train_test_split(y, test_size=36)

plot_series(y, y_test, labels=['train', 'test'])
plt.show()

fh = np.arange(1, 37)

# forecaster = ARIMA()
# forecaster.fit(y)
# y_pred = forecaster.predict(fh)
# plot_series(y, y_pred, labels=["y", "y_pred"])
# plt.show()

# -----------------------------------------------------------------

from sktimex.darts.arima import ARIMA

forecaster = ARIMA()

# step 4: fitting the forecaster
forecaster.fit(y)

# step 5: querying predictions
y_pred = forecaster.predict_history(fh)


# optional: plotting predictions and past data
plot_series(y, y_pred, labels=["y", "y_pred"])
plt.show()


