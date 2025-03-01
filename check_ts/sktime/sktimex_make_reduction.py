import pandas as pd
import numpy as np
from sktimex.forecasting.compose import make_reduction, TabularRegressorForecaster
from sktimex.utils import fh_range
from sklearn.linear_model import LinearRegression


def main():
    x = np.arange(1, 101) + 0.1
    y = np.arange(1, 101) + 0.2
    z = x + y
    w = 2*x + 0.5*y

    df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'w': w})
    tr = df[:80]
    te = df[80:]
    ye = te[['z', 'w']]

    # f = make_reduction(LinearRegression(), window_length=(4, 3, True), strategy='recursive')
    # f.fit(X=tr[['x', 'y']], y=tr['z'])
    # pred1 = f.predict(X=te[['x']], fh=fh_range(20))

    f = make_reduction(LinearRegression(), window_length=[4], strategy='direct')

    # f = TabularRegressorForecaster(
    #     LinearRegression(),
    #     window_length=4)

    f.fit(X=tr[['x', 'y']], y=tr[['z', 'w']], fh=[2, 3])
    p1 = f.predict(X=te[['x', 'y']], fh=fh_range(20))

    # f.fit(X=tr[['x', 'y']], y=tr['z'], fh=[1])
    # pred2 = f.predict(X=te[['x']], fh=fh_range(20))

    f.fit(X=tr[['x', 'y']], y=tr[['z', 'w']], fh=fh_range(4))
    p3 = f.predict(X=te[['x', 'y']], fh=fh_range(20))

    pass


if __name__ == "__main__":
    main()
