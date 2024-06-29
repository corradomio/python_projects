import logging.config

import pandasx as pdx
from stdlib.tprint import tprint
from sktime.forecasting.base import ForecastingHorizon
from sktimex.utils.plotting import plot_series, show

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.models import RNN
from neuralforecast.utils import AirPassengersDF


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

    tprint("nf")
    nf = NeuralForecast(
        models=[RNN(input_size=24, h=12, max_steps=100)],
        freq='M'
    )

    tprint("nf.fit")
    nf.fit(df=AirPassengersDF)
    tprint("nf.predict")
    nf.predict()
    tprint("end", force=True)

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.info("Logging configured")
    main()
