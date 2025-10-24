import logging.config
import warnings

from sktime.forecasting.base import ForecastingHorizon

import pandasx as pdx
from sktimex.forecasting.reducer import ReducerForecaster
from sktimex.utils.plotting import plot_series, show

warnings.filterwarnings("ignore", category=FutureWarning)


TARGET = "Passengers"


def eval(train, test, r, fh=None):

    r.fit(y=train, fh=fh)
    # r.update(y=train)

    fh = ForecastingHorizon(test.index)
    pred = r.predict(fh=fh)

    plot_series(train, test, pred, labels=["train", "test", "pred"], title=str(r))
    show()


def main():
    df = pdx.read_data(
        "data/airline-passengers.csv",
        numeric="Passengers",
        datetime=("Month", "%Y-%m", 'M'),
        datetime_index="Month",
        ignore="Month"
    )[TARGET]

    start_date = pdx.to_datetime('19580101')

    train, test = pdx.train_test_split(df, datetime=start_date)

    # "recursive"       RecursiveTabularRegressionForecaster
    # "direct"          DirectTabularRegressionForecaster
    # "dirrec"          DirRecTabularRegressionForecaster
    # "multioutput"     MultioutputTabularRegressionForecaster
    # eval(train, test, NaiveForecaster(sp=36))

    # eval(train, test, ReducerForecaster(
    #     estimator="sklearn.linear_model.LinearRegression",
    #     window_length=36,
    #     prediction_length=1,
    #     strategy="recursive"
    # ))

    eval(train, test, ReducerForecaster(
        estimator="catboost.CatBoostRegressor",
        window_length=36,
        prediction_length=1,
        strategy="recursive"
    ))

    eval(train, test, ReducerForecaster(
        estimator="xgboost.XGBRegressor",
        window_length=36,
        prediction_length=1,
        strategy="recursive"
    ))

    eval(train, test, ReducerForecaster(
        estimator="lightgbm.LGBMRegressor",
        window_length=36,
        prediction_length=1,
        strategy="recursive"
    ))

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.info("Logging configured")
    main()
