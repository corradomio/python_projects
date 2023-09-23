from sktimex.linear_model import LinearForecastRegressor
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster


def test1(df):

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


def test_y(df, lm):
    y = df[['y']]

    y_train = y[:90]
    y__test = y[90:]
    fh = list(range(10))

    lm.fit(y=y_train)

    yp1 = lm.predict(           fh=fh)
    yp2 = lm.predict(y=y_train, fh=fh)
    pass


def test_xy(df, lm):
    X = df[['x']]
    y = df[['y']]

    X_train = X[:90]
    X__test = X[90:]
    y_train = y[:90]
    y__test = y[90:]
    fh = list(range(10))

    lm.fit(X=X_train, y=y_train)

    yp3 = lm.predict(y=y_train,        X=X)
    yp4 = lm.predict(fh=fh, y=y_train, X=X)
    yp1 = lm.predict(            X=X__test)
    yp2 = lm.predict(fh=fh,      X=X__test)
    pass


def test_lag0(df):
    X = df[['x']]
    y = df[['y']]

    X_train = X[:90]
    X__test = X[90:]
    y_train = y[:90]
    y__test = y[90:]
    fh = list(range(10))

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    lm = LinearForecastRegressor(class_name='sklearn.linear_model.LinearRegression', lag=0)
    lm.fit(X_train, y_train)

    yp0 = lr.predict(X=X__test)
    yp1 = lm.predict(X=X__test)
    pass


def main():
    df = DataFrame({'x': list(range(100)), 'y': [(1+i+i/100) for i in range(100)]})
    lm = LinearForecastRegressor(class_name='sklearn.linear_model.LinearRegression', lag=5)

    # test_y(df, lm)
    # test_xy(df, lm)
    test_lag0(df)

    pass


if __name__ == "__main__":
    main()
