import matplotlib.pyplot as plt
import numpy as np

from arimax import ar_process, ma_process
from sktime.forecasting.arima import ARIMA, AutoARIMA
# from sktime.utils.plotting import plot_series
from sktime.forecasting.base import ForecastingHorizon
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf


def plot_series(*ylist, labels=None, x0=0):
    i = 0
    for y in ylist:
        n = len(y)
        x = np.arange(x0, x0+n)
        plt.plot(x, y, label=labels[i])
        x0 += n
        i += 1
# end


def main():
    map = ma_process([.3, .6, .9], 100, tc=1)
    plot_series(map, labels=['y'])
    plt.show()

    adfr = adfuller(map)
    print(adfr[0])
    print(adfr[1])

    plot_acf(map, lags=10)
    plt.show()

    diff1 = np.diff(map, 1)
    plot_series(diff1, labels=['y'])
    plt.show()

    adfr = adfuller(diff1)
    print(adfr[0])
    print(adfr[1])

    plot_acf(diff1, lags=10)
    plt.show()

    pass


def main1():
    arp = ar_process(1, 1000)
    x = np.arange(len(arp))
    n = len(x)

    # plt.plot(arp)
    # plt.show()

    t = int(.9*n)
    y_train = arp[:t]
    x_train = x[:t]
    y_test = arp[t:]
    x_test = x[t:]

    # plot_series(y_train, y_test, labels=['train', 'test'])
    # plt.plot(x_train, y_train)
    plt.plot(x_test, y_test)
    plt.show()

    model = ARIMA(order=(1,0,0))
    model.fit(y_train)
    y_pred = model.predict(fh=ForecastingHorizon(x_test+1))

    plt.plot(x_test, y_test)
    plt.plot(x_test, y_pred)
    plt.show()
    pass


if __name__ == "__main__":
    main()
