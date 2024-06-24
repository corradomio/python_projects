import logging.config

import pandasx as pdx
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktimex.forecasting.linear import LinearForecaster as SktimexLinearForecaster
from sktimex.forecasting.darts.linear import DartsLinearForecaster
from sktimex.forecasting.darts.arima import DartsARIMAForecaster
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

    # eval(train, test, NaiveForecaster(sp=36))
    # eval(train, test, SktimexLinearForecaster(
    #     lags=36,
    #     tlags=1,
    #     flatten=False
    # ))
    eval(train, test, DartsLinearForecaster(
        lags=36,
        output_chunk_length=1
    ))
    eval(train, test, DartsLinearForecaster(
        lags=36,
        output_chunk_length=1
    ))
    eval(train, test, DartsARIMAForecaster(

    ))

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.info("Logging configured")
    main()
