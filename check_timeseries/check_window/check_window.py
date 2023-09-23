import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.utils.plotting import plot_series
import matplotlib.pyplot as plt

from sktimex import ScikitForecastRegressor, LinearForecastRegressor

N = 100
P = 10


def test1(y, fh):
    cutoff = y.index[-1]
    fh = ForecastingHorizon(fh, is_relative=True).to_absolute(cutoff)

    fc: BaseForecaster = make_reduction(estimator=LinearRegression(), strategy="recursive", window_length=10)

    fc.fit(y=y)
    y_pred = fc.predict(fh=fh)

    plot_series(y, y_pred, labels=["train", "pred"])
    plt.show()

    return fc


def test2(y, fc, fh):
    cutoff = y.index[-1]
    fh = ForecastingHorizon(fh, is_relative=True).to_absolute(cutoff)

    y_pred = fc.predict(fh=fh)

    plot_series(y, y_pred, labels=["train", "pred"])
    plt.show()


def test3(y, fh):
    cutoff = y.index[-1]
    fh = ForecastingHorizon(fh, is_relative=True).to_absolute(cutoff)

    fc = ScikitForecastRegressor(estimator=LinearRegression(), strategy="recursive", window_length=10)

    fc.fit(y=y)
    y_pred = fc.predict(fh=fh)

    plot_series(y, y_pred, labels=["train", "pred"])
    plt.show()

    return fc


def test4(y, fh):
    cutoff = y.index[-1]
    fh = ForecastingHorizon(fh, is_relative=True).to_absolute(cutoff)

    fc = LinearForecastRegressor(estimator=LinearRegression(), lags=[0, 10])

    fc.fit(y=y)
    y_pred = fc.predict(fh=fh)

    plot_series(y, y_pred, labels=["train", "pred"])
    plt.show()


def main():
    ar = np.arange(0, N, 1, dtype=int)

    ix = pd.period_range('1923-01-01', periods=N, freq='M')
    y = pd.DataFrame(data=ar, columns=['y'], index=ix)

    # cutoff: pd.Period = ix[-1]
    # fh = pd.date_range((cutoff + 1).to_timestamp(), periods=P)
    fh = list(range(1, P+1))

    # fc = test1(y, fh)
    # test2(y, fc, list(range(100, P+100)))
    # fc = test3(y, fh)

    fc = test4(y, fh)

    pass


if __name__ == "__main__":
    main()
