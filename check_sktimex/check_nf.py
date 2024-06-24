import logging.config

import pandasx as pdx
from sktime.forecasting.base import ForecastingHorizon
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

    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS
    from neuralforecast.utils import AirPassengersDF

    nf = NeuralForecast(
        models=[NBEATS(input_size=24, h=12, max_steps=100)],
        freq='M'
    )

    nf.fit(df=AirPassengersDF)
    nf.predict()

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.info("Logging configured")
    main()
