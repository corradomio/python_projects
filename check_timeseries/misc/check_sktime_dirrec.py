import pandas as pd
import pandasx as pdx
import matplotlib.pyplot as plt
from sktime.forecasting.base import ForecastingHorizon
# from sktime.forecasting.compose import make_reduction, DirRecTabularRegressionForecaster
from sktimex.forecasting.compose import make_reduction, DirRecTabularRegressionForecaster
from sklearn.linear_model import LinearRegression
from sktime.utils.plotting import plot_series


#
def prepare_data():
    periods = pd.period_range("2023-01-01", periods=365, freq='D')
    data = [
        [i]
        for i in range(1, 366)
    ]

    df = pd.DataFrame(data=data, columns=["data"], index=periods)
    return df


#
#   _Reducer
#       _DirRecReducer
#           DirRecTabularRegressionForecaster


def main():
    drtrf: DirRecTabularRegressionForecaster = None

    df = prepare_data()
    train_test, eval = pdx.train_test_split(df, train_size=.5)
    train, test = pdx.train_test_split(train_test, train_size=.8)

    fh7 = ForecastingHorizon(list(range(1, 8)))
    fht = ForecastingHorizon(list(range(1, len(test)+1)))
    fh8 = ForecastingHorizon(list(range(1, 80)))
    forecaster = make_reduction(LinearRegression(), strategy='dirrec', window_length=7)
    forecaster.fit(y=train, fh=fht)
    pred = forecaster.predict(fh=fht)

    plot_series(train, test, pred, labels=["train", "test", "pred"])
    plt.show()

    forecaster.update(y=test)

    fhp = ForecastingHorizon(list(range(1, len(eval)+1)))
    pred = forecaster.predict(fh=fhp)

    plot_series(train, test, pred, labels=["train", "test", "pred"])
    plt.show()

    pass


if __name__ == "__main__":
    main()
