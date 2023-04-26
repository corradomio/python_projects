from etime.linear_model import LinearModel
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster


def test(df):

    lr = LinearRegression()
    Xd = df[['x']]
    yd = df['y']
    lr.fit(Xd, yd)

    Xa = Xd.to_numpy()
    ya = yd.to_numpy()
    lr.fit(Xa, ya)

    Xd = df[['x']]
    yd = df[['y']]
    lr.fit(Xd, yd)

    Xa = Xd.to_numpy()
    ya = yd.to_numpy()
    lr.fit(Xa, ya)

    Xd = df['x']
    yd = df['y']
    lr.fit(Xd, yd)

    Xa = Xd.to_numpy()
    ya = yd.to_numpy()
    lr.fit(Xa, ya)
# end


def test2():
    X = None
    y = None
    fh = None
    nf = NaiveForecaster()
    nf.fit(y, X, fh)


def main():
    df = DataFrame({'x': list(range(100)), 'y': [(1+i+i/100) for i in range(100)]})
    lm = LinearModel(class_name='sklearn.linear_model.LinearRegression',
                     lag=5)

    X = df[['x']]
    y = df[['y']]

    lm.fit(y=y, X=X)
    pass


if __name__ == "__main__":
    main()
