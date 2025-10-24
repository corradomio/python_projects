from sktime.forecasting.base import ForecastingHorizon
from sktimex.forecasting import ReducerForecaster
from sktimex.transform.lags import yx_lags, t_lags, matrix
from sktimex.transform.lagt import LagsTrainTransform, LagsPredictTransform


def main():
    X0 = matrix(9, 9)
    y0 = matrix(9, 0)
    X1 = matrix(9, 9, 10)
    y1 = matrix(9, 1, 10)

    ylags, xlags = yx_lags([[], [0]])
    tlags = t_lags([0])

    ltt = LagsTrainTransform(xlags=xlags, ylags=ylags, tlags=tlags, flatten=True)
    lpt = ltt.predict_transform()

    # Xt, yt = ltt.fit_transform(y=y0, X=X0)
    lpt.fit(y=y0, X=X0).transform(X=X1)
    Xf, yf = lpt.predict_steps(y1)

    pass


if __name__ == "__main__":
    main()
