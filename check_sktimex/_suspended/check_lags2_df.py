from sktimex.transform.lags import yx_lags, tu_lags, dataframe
from sktimex.transform.lint import LinearTrainTransform
from sktime.forecasting.base import ForecastingHorizon


def main():
    X = dataframe(20, 2, name="x")
    y = dataframe(20, 0, name="y")
    X0 = X.iloc[:10]
    y0 = y.iloc[:10]
    X1 = X.iloc[10:]
    y1 = y.iloc[10:]

    # X0 = None
    # X1 = None

    ylags, xlags = yx_lags([2,2])
    tlags, ulags = tu_lags([[1,3],1])

    ltt = LinearTrainTransform(xlags=xlags, ylags=ylags, tlags=tlags, ulags=ulags)
    Xt, yt = ltt.fit_transform(y=y0, X=X0)

    lpt = ltt.predict_transform()
    Xt, yt = lpt.fit(y=y0, X=X0).transform(fh=ForecastingHorizon(y1.index), X=X1)

    Xf, yf = lpt.predict_steps(y1)

    pass


if __name__ == "__main__":
    main()

