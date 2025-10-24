import logging.config

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster

import pandasx as pdx
from sktimexnn.forecasting.cnn import CNNLinearForecaster
from sktimex.forecasting import LinearForecaster
from sktimex.forecasting import ReducerForecaster
from sktimexnn.forecasting.lnn import LNNLinearForecaster
from sktimexnn.forecasting.rnn import RNNLinearForecaster
from sktimex.utils.plotting import plot_series, show

TARGET = "Passengers"


def eval(train, test, r):
    r.fit(y=train)
    r.update(y=train)

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

    eval(train, test, NaiveForecaster(sp=36))
    eval(train, test, LinearForecaster(
        lags=36,
        tlags=1,
        flatten=False
    ))

    eval(train, test, ReducerForecaster(
        window_length=36,
    ))

    eval(train, test, LNNLinearForecaster(
        lags=36,
        tlags=1,

        scaler=dict(
            method="minmax",
            outlier=3
        )
    ))

    eval(train, test, RNNLinearForecaster(
        flavour="rnn",
        lags=36,
        tlags=1,

        scaler=dict(
            method="minmax",
            outlier=3
        )

    ))

    eval(train, test, RNNLinearForecaster(
        flavour="gru",
        lags=36,
        tlags=1,

        scaler=dict(
            method="minmax",
            outlier=3
        )
    ))

    eval(train, test, RNNLinearForecaster(
        flavour="lstm",
        lags=36,
        tlags=1,

        scaler=dict(
            method="minmax",
            outlier=3
        )
    ))

    eval(train, test, CNNLinearForecaster(
        lags=36,
        tlags=1,

        scaler=dict(
            method="minmax",
            outlier=3
        )
    ))

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.info("Logging configured")
    main()
