import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.utils.plotting import plot_series

N = 100
P = 10


def main():
    a0 = np.arange(0, N, 1, dtype=int)
    ix = pd.period_range('1923-01-01', periods=N, freq='M')
    y0 = pd.DataFrame(data=a0, columns=['y'], index=ix)

    fhi = list(range(1, P+1))

    cutoff = y0.index[-1]
    fh = ForecastingHorizon(fhi, is_relative=True).to_absolute(cutoff)

    fc: BaseForecaster = make_reduction(estimator=LinearRegression(), strategy="recursive", window_length=10)

    fc.fit(y=y0)
    y_pred = fc.predict(fh=fh)

    plot_series(y0, y_pred, labels=["train", "pred"])
    plt.show()

    # ---------------------------------------------------------------

    a1 = np.arange(1000, 1000+N, 1, dtype=int)
    ix = pd.period_range('2023-01-01', periods=N, freq='M')
    y1 = pd.DataFrame(data=a1, columns=['y'], index=ix)

    cutoff = y1.index[-1]
    fh = ForecastingHorizon(fhi, is_relative=True).to_absolute(cutoff)

    fc.update(y=y1, update_params=False)
    y_pred = fc.predict(fh=fh)

    plot_series(y1, y_pred, labels=["train", "pred"])
    plt.show()

    # ---------------------------------------------------------------

    cutoff = y0.index[-1]
    fh = ForecastingHorizon(fhi, is_relative=True).to_absolute(cutoff)

    fc.update(y=y0, update_params=False)
    y_pred = fc.predict(fh=fh)

    plot_series(y0, y_pred, labels=["train", "pred"])
    plt.show()

    # ---------------------------------------------------------------

    a2 = np.arange(500, 500+N, 1, dtype=int)
    ix = pd.period_range('2022-01-01', periods=N, freq='M')
    y2 = pd.DataFrame(data=a2, columns=['y'], index=ix)

    cutoff = y2.index[-1]
    fh = ForecastingHorizon(fhi, is_relative=True).to_absolute(cutoff)

    fc.update(y=y2, update_params=False)
    y_pred = fc.predict(fh=fh)

    plot_series(y2, y_pred, labels=["train", "pred"])
    plt.show()

    pass


if __name__ == "__main__":
    main()
