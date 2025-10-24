import logging.config

from stdlib import lrange
from sktime.forecasting.neuralforecast import NeuralForecastRNN, NeuralForecastLSTM

import pandasx as pdx
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktimexnn.forecasting import CNNLinearForecaster
from sktimexnn.forecasting import LNNLinearForecaster
from sktimexnn.forecasting import RNNLinearForecaster

from sktimex.forecasting.reducer import ReducerForecaster
from sktimex.utils.plotting import plot_series, show

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

    # eval(train, test, NaiveForecaster(sp=36))

    eval(train, test, NeuralForecastRNN(),
         fh=ForecastingHorizon(lrange(1, 37))
    )

    eval(train, test, NeuralForecastLSTM(),
         fh=ForecastingHorizon(lrange(1, 37))
    )

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
