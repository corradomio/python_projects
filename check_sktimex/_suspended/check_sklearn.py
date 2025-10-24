import logging.config
import warnings

import pandasx as pdx
from sktime.forecasting.base import ForecastingHorizon
# from sktimex.forecasting.cnn import CNNLinearForecaster
# from sktimex.forecasting.lnn import LNNLinearForecaster
# from sktimex.forecasting.rnn import RNNLinearForecaster
from sktimex.forecasting.sklearn import ScikitLearnForecaster
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

    # eval(train, test, ScikitLearnForecaster(
    #     estimator="sklearn.linear_model.LinearRegression",
    #     window_length=36,
    #     prediction_length=3,
    # ))

    # DOESN'T WORK
    # eval(train, test, ScikitLearnForecaster(
    #     estimator="catboost.CatBoostRegressor",
    #     window_length=36,
    #     prediction_length=3,
    #     loss_function='RMSE',
    # ))


    # eval(train, test, ScikitLearnForecaster(
    #     estimator="xgboost.XGBRegressor",
    #     window_length=36,
    #     prediction_length=3,
    #     n_estimators=100,
    # ))

    eval(train, test, ScikitLearnForecaster(
        estimator="lightgbm.LGBMRegressor",
        window_length=36,
        prediction_length=1,
        estimator_args=dict(
            boosting_type='gbdt'
        )
    ))
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.info("Logging configured")
    main()
