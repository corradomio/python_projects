import logging.config
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpyx as npx
import pandas as pd
import pandasx as pdx
import darts as dts
import darts.models
import torch.nn as nn
import torchx.nn as nnx
from sklearnx.preprocessing import StandardScaler
from sktime.utils.plotting import plot_series


def main():
    df = pdx.read_data(
        "D:/Dropbox/Datasets/kaggle/airline-passengers.csv",
        # datetime=('Month', '%Y-%m', 'M'),
        datetime=('Month', '%Y-%m'),
        # ignore=['Month'],
        # index=['Month']
    )
    print(len(df))

    series = dts.TimeSeries.from_dataframe(df, "Month", "#Passengers")
    train, val = series[:-36], series[-36:]

    # ----------------------------------------------
    # OK

    # model = dts.models.ExponentialSmoothing()
    # model.fit(train)
    # prediction = model.predict(len(val), num_samples=1000)
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK

    # model = dts.models.AutoARIMA()
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK WORST

    # model = dts.models.StatsForecastAutoARIMA()
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # NO

    # model = dts.models.forecasting.StatsForecastAutoETS()
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK : WORST

    # model = dts.models.StatsForecastAutoCES()
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK

    # model = dts.models.BATS(seasonal_periods=[12])
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK

    # model = dts.models.TBATS(seasonal_periods=[12])
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK

    # model = dts.models.Theta(seasonality_period=12)
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK

    # model = dts.models.FourTheta(seasonality_period=12)
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK

    # model = dts.models.StatsForecastAutoTheta(season_length=12)
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    #

    # model = dts.models.Prophet(
    #     # {
    #     #     'name': 'yearly',
    #     #     'seasonal_periods': 12,
    #     #     'fourier_order': 1
    #     # }
    # )
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK: WORST

    # model = dts.models.FFT(nr_freqs_to_keep=24, trend='poly', trend_poly_degree=1)
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK

    # model = dts.models.KalmanForecaster()
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK

    # model = dts.models.Croston(version='optimized')
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK

    # model = dts.models.RandomForest(lags=12)
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK

    # model = dts.models.LinearRegressionModel(lags=12, multi_models=False)
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK

    # model = dts.models.LinearRegressionModel(lags=12, multi_models=True)
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK: flat

    # model = dts.models.LightGBMModel(lags=12)
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK: worst

    # model = dts.models.CatBoostModel(lags=12)
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    # OK: worst

    # model = dts.models.XGBModel(lags=12)
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    # plt.show()

    # ----------------------------------------------
    #
    EPOCHS = 300

    model = dts.models.RNNModel(model='RNN',
                                input_chunk_length=12,
                                training_length=24,
                                hidden_dim=32,
                                batch_size=16,
                                n_epochs=EPOCHS,
                                dropout=0.,
                                optimizer_kwargs={'lr': 1e-3},
                                log_tensorboard=False,
                                force_reset=True)
    model.fit(train)
    prediction = model.predict(len(val))

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.show()

    # ----------------------------------------------
    #

    model = dts.models.RNNModel(model='LSTM', input_chunk_length=24, hidden_dim=32)
    model.fit(train)
    prediction = model.predict(len(val))

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.show()

    # ----------------------------------------------
    #

    model = dts.models.RNNModel(model='GRU')
    model.fit(train)
    prediction = model.predict(len(val))

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.show()

    # ----------------------------------------------
    #

    model = dts.models.BlockRNNModel(model='RNN')
    model.fit(train)
    prediction = model.predict(len(val))

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.show()

    # ----------------------------------------------
    #

    model = dts.models.BlockRNNModel(model='LSTM')
    model.fit(train)
    prediction = model.predict(len(val))

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.show()

    # ----------------------------------------------
    #

    model = dts.models.BlockRNNModel(model='GRU')
    model.fit(train)
    prediction = model.predict(len(val))

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.show()

    # ----------------------------------------------
    #

    model = dts.models.NBEATSModel()
    model.fit(train)
    prediction = model.predict(len(val))

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.show()

    # ----------------------------------------------
    #

    model = dts.models.NHiTSModel()
    model.fit(train)
    prediction = model.predict(len(val))

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.show()

    # ----------------------------------------------
    #

    model = dts.models.TCNModel()
    model.fit(train)
    prediction = model.predict(len(val))

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.show()

    # ----------------------------------------------
    #

    model = dts.models.TransformerModel()
    model.fit(train)
    prediction = model.predict(len(val))

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.show()

    # ----------------------------------------------
    #

    model = dts.models.TFTModel()
    model.fit(train)
    prediction = model.predict(len(val))

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.show()

    # ----------------------------------------------
    #

    model = dts.models.DLinearModel()
    model.fit(train)
    prediction = model.predict(len(val))

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.show()

    # ----------------------------------------------
    #

    model = dts.models.NLinearModel()
    model.fit(train)
    prediction = model.predict(len(val))

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.show()

    # ----------------------------------------------
    # end
# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
