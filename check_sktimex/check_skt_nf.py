import logging.config
import warnings
import pandasx as pdx
from sktime.forecasting.base import ForecastingHorizon
from sktimex.forecasting.nf.mlp import MLP
from sktimex.forecasting.nf.rnn import RNN
from sktimex.utils.plotting import plot_series, show

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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

    # eval(train, test, MLP(
    #     window_length=36,
    #     prediction_length=1,
    #
    #     n_layers=2,
    #     hidden_size=10,
    #
    #     max_steps=20,
    #     trainer_kwargs={}
    # ))
    eval(train, test, RNN(
        window_length=36,
        prediction_length=21,

        max_steps=100,
        trainer_kwargs={}
    ))

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.info("Logging configured")
    main()
