import logging.config

import pandasx as pdx
from sktime.forecasting.base import ForecastingHorizon
from sktimex.forecasting.darts.block_rnn_model import BlockRNNModel
from sktimex.forecasting.darts.dlinear import DLinearModel
from sktimex.forecasting.darts.linear_regression_model import LinearRegressionModel
from sktimex.forecasting.darts.nbeats import NBEATSModel
from sktimex.forecasting.darts.nhits import NHiTSModel
from sktimex.forecasting.darts.nlinear import NLinearModel
from sktimex.forecasting.darts.rnn_model import RNNModel
from sktimex.forecasting.darts.tcn_model import TCNModel
from sktimex.forecasting.darts.tft_model import TFTModel
from sktimex.forecasting.darts.tide_model import TiDEModel
from sktimex.forecasting.darts.transformer_model import TransformerModel
from sktimex.forecasting.darts.tsmixer_model import TSMixerModel

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

    # eval(train, test, BlockRNNModel(
    #     input_chunk_length=36,
    #     output_chunk_length=12
    # ))
    # eval(train, test, DLinearModel(
    #     input_chunk_length=36,
    #     output_chunk_length=12
    # ))
    # eval(train, test, LinearRegressionModel(
    #     lags=36,
    #     output_chunk_length=12
    # ))
    # eval(train, test, NBEATSModel(
    #     input_chunk_length=36,
    #     output_chunk_length=12
    # ))
    # eval(train, test, NHiTSModel(
    #     input_chunk_length=36,
    #     output_chunk_length=12
    # ))
    # eval(train, test, NLinearModel(
    #     input_chunk_length=36,
    #     output_chunk_length=12
    # ))
    # eval(train, test, RNNModel(
    #     input_chunk_length=36,
    #     output_chunk_length=12,
    #     training_length=36
    # ))
    # eval(train, test, TCNModel(
    #     input_chunk_length=36,
    #     output_chunk_length=12
    # ))
    eval(train, test, TiDEModel(
        input_chunk_length=36,
        output_chunk_length=12
    ))
    eval(train, test, TransformerModel(
        input_chunk_length=36,
        output_chunk_length=12
    ))
    eval(train, test, TSMixerModel(
        input_chunk_length=36,
        output_chunk_length=12
    ))

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.info("Logging configured")
    main()
