import logging.config
import warnings

from stdlib import lrange
from sktime.forecasting.neuralforecast import NeuralForecastRNN, NeuralForecastLSTM

import pandasx as pdx
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktimex.forecasting.cnn import CNNLinearForecaster
from sktimex.forecasting.lnn import LNNLinearForecaster
from sktimex.forecasting.rnn import RNNLinearForecaster

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

    # single model trained with
    #   window_length -> prediction_length
    # BUT NOT recursive
    eval(train, test, ReducerForecaster(
        window_length=36,
        # prediction_length=36,
        prediction_length=3,
        strategy="multioutput"
    ))

    # one model for each timeslot (prediction_length models)
    # to predict the timeslot i
    # windows_identical
    #   true    windows fixed
    #   false   the window advances [NO]
    eval(train, test, ReducerForecaster(
        window_length=36,
        # prediction_length=36,
        prediction_length=3,
        strategy="direct"
    ))

    # single model trained with
    #   window_length -> 1
    # recursive
    eval(train, test, ReducerForecaster(
        window_length=36,
        prediction_length=3,
        strategy="recursive"
    ))

    # eval(train, test, ReducerForecaster(
    #     window_length=36,
    #     prediction_length=36,
    #     strategy="direct",
    #     windows_identical=False
    # ))

    # one model for each timeslot (prediction_length models)
    # to predict the timeslot i
    # windows increases for each model from
    #   window_length  to  window_length+prediction_length
    # BOH! It has no sense
    eval(train, test, ReducerForecaster(
        window_length=36,
        prediction_length=3,
        strategy="dirrec"
    ))

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.info("Logging configured")
    main()
