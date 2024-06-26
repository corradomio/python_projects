import logging.config
import warnings

from neuralforecast.losses.pytorch import DistributionLoss
from pytorch_lightning.callbacks import ModelSummary

import pandasx as pdx
from sktime.forecasting.base import ForecastingHorizon
from sktimex.forecasting.nf.mlp import MLP
from sktimex.forecasting.nf.rnn import RNN
from sktimex.forecasting.nf.gru import GRU
from sktimex.forecasting.nf.lstm import LSTM
from sktimex.utils.plotting import plot_series, show

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


TARGET = "Passengers"


def eval(train, test, r, fh=None):
    # r.compile_model(train)
    print(r)

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

    eval(train, test, MLP(
        window_length=24,
        prediction_length=12,

        num_layers=2,
        hidden_size=1024,

        # loss=DistributionLoss(distribution='Normal', level=[80, 90]),
        loss="distributionloss",
        loss_kwargs=dict(distribution='Normal', level=[80, 90]),
        scaler_type='robust',
        learning_rate=1e-3,
        max_steps=200,
        val_check_steps=10,
        early_stop_patience_steps=2,
        val_size=12
    ))
    # eval(train, test, RNN(
    #     window_length=36,
    #     prediction_length=21,
    #
    #     max_steps=1,
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, GRU(
    #     window_length=36,
    #     prediction_length=21,
    #
    #     max_steps=1,
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, LSTM(
    #     window_length=36,
    #     prediction_length=21,
    #
    #     max_steps=1,
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.info("Logging configured")
    main()
