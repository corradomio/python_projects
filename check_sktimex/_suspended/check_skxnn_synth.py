import logging.config
import warnings
import os
import matplotlib.pyplot as plt
import pandas as pd

import pandasx as pdx
import sktimex as sktx
import sktimexnn.forecasting.darts
import sktimexnn.forecasting.nf
from synth import create_syntethic_data

# Suppress all UserWarning instances
warnings.simplefilter("ignore", UserWarning)

TARGET = "y"


def run_model_3(df, model_class):
    # df = create_syntethic_data()
    model_name = model_class.__name__
    module = "darts" if "darts" in model_class.__module__ else "nf"

    print("---", model_name, "---")

    dir = f"./preds/{module}/{model_name}"
    os.makedirs(dir, exist_ok=True)

    # print("loop")
    dfdict = pdx.groups_split(df, groups=["cat"])
    for g in dfdict:
        try:
            print("...", g)
            dfg = dfdict[g]

            X, y = pdx.xy_split(dfg, target=TARGET)

            X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=18)

            # DarTS
            # model = model_class(input_chunk_length=24, output_chunk_length=6)

            # NeuralForecast
            model = model_class(input_size=24, h=6, max_steps=100, enable_progress_bar=True)

            model.fit(y=y_train, X=X_train)

            # fh = ForecastingHorizon(y_test.index, is_relative=False)
            fh = y_test.index
            y_predict = model.predict(fh=fh, X=X_test)

            sktx.utils.plot_series(y_train, y_test, y_predict, labels=["train", "test", "predict"], title=f"{model_name}: {g[0]}")

            fname = f"{dir}/{model_name}-{g[0]}.png"
            plt.savefig(fname, dpi=300)

            # break
        except Exception as e:
            print("ERROR:", e)
    pass


def run_model_1(df, model_class):
    # df = create_syntethic_data()
    model_name = model_class.__name__
    module = "darts" if "darts" in model_class.__module__ else "nf"

    print("---", model_name, "---")

    dir = f"./preds/{module}/{model_name}"
    os.makedirs(dir, exist_ok=True)

    # print("loop")
    dfdict = pdx.groups_split(df, groups=["cat"])
    for g in dfdict:
        try:
            print("...", g)
            dfg = dfdict[g]

            X, y = pdx.xy_split(dfg, target=TARGET)

            X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=18)

            # DarTS
            model = model_class(input_chunk_length=24, output_chunk_length=6)

            # NeuralForecast
            # model = model_class(input_size=24, max_steps=100, enable_progress_bar=True)

            model.fit(y=y_train, X=X_train)

            # fh = ForecastingHorizon(y_test.index, is_relative=False)
            fh = y_test.index
            y_predict = model.predict(fh=fh, X=X_test)

            sktx.utils.plot_series(y_train, y_test, y_predict, labels=["train", "test", "predict"], title=f"{model_name}: {g[0]}")

            fname = f"{dir}/{model_name}-{g[0]}.png"
            plt.savefig(fname, dpi=300)
            plt.close()

            # break
        except Exception as e:
            print("ERROR:", e)
    pass


def run_model_2(df, model):
    # df = create_syntethic_data()
    model_name = model.__class__.__name__
    module = "darts" if "darts" in model.__class__.__module__ else "nf"

    print("---", model_name, "---")

    dir = f"./preds/{module}/{model_name}"
    os.makedirs(dir, exist_ok=True)

    # print("loop")
    dfdict = pdx.groups_split(df, groups=["cat"])
    for g in dfdict:
        try:
            print("...", g)
            dfg = dfdict[g]

            X, y = pdx.xy_split(dfg, target=TARGET)

            X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=18)

            model.fit(y=y_train, X=X_train)

            # fh = ForecastingHorizon(y_test.index, is_relative=False)
            fh = y_test.index
            y_predict = model.predict(fh=fh, X=X_test)

            sktx.utils.plot_series(y_train, y_test, y_predict, labels=["train", "test", "predict"], title=f"{model_name}: {g[0]}")

            fname = f"{dir}/{model_name}-{g[0]}.png"
            plt.savefig(fname, dpi=300)

            # break
        except Exception as e:
            print("ERROR:", e)
    pass


def check_darts(df: pd.DataFrame):
    run_model_1(df, sktimexnn.forecasting.darts.TSMixerModel)
    run_model_1(df, sktimexnn.forecasting.darts.RNNModel)
    run_model_1(df, sktimexnn.forecasting.darts.TCNModel)
    run_model_1(df, sktimexnn.forecasting.darts.NLinearModel)
    run_model_1(df, sktimexnn.forecasting.darts.NHiTSModel)
    run_model_1(df, sktimexnn.forecasting.darts.BlockRNNModel)
    run_model_1(df, sktimexnn.forecasting.darts.TransformerModel)
    run_model_1(df, sktimexnn.forecasting.darts.NBEATSModel)
    run_model_1(df, sktimexnn.forecasting.darts.DLinearModel)
    run_model_1(df, sktimexnn.forecasting.darts.TiDEModel)
    # run_model(df, sktimexnn.forecasting.darts.TFTModel)     # NO

    run_model_2(df, sktimexnn.forecasting.darts.ARIMA())
    run_model_2(df, sktimexnn.forecasting.darts.CatBoostModel(lags=24,lags_past_covariates=24,output_chunk_length=6))
    run_model_2(df, sktimexnn.forecasting.darts.ExponentialSmoothing(lags=24,lags_past_covariates=24,output_chunk_length=6))
    run_model_2(df, sktimexnn.forecasting.darts.FFT())
    run_model_2(df, sktimexnn.forecasting.darts.KalmanForecaster())
    run_model_2(df, sktimexnn.forecasting.darts.LightGBMModel(lags=24,lags_past_covariates=24,output_chunk_length=6))
    run_model_2(df, sktimexnn.forecasting.darts.LinearRegressionModel(lags=24,lags_past_covariates=24,output_chunk_length=6))
    run_model_2(df, sktimexnn.forecasting.darts.Prophet(force_col_wise=True))
    run_model_2(df, sktimexnn.forecasting.darts.RandomForestModel(lags=24,lags_past_covariates=24,output_chunk_length=6))
    run_model_2(df, sktimexnn.forecasting.darts.SKLearnModel(lags=24,lags_past_covariates=24,output_chunk_length=6))
    run_model_2(df, sktimexnn.forecasting.darts.Theta())


def check_nf(df: pd.DataFrame):
    run_model_3(df, sktimexnn.forecasting.nf.Autoformer)
    run_model_3(df, sktimexnn.forecasting.nf.BiTCN)
    run_model_3(df, sktimexnn.forecasting.nf.DeepAR)
    run_model_3(df, sktimexnn.forecasting.nf.DeepNPTS)
    run_model_3(df, sktimexnn.forecasting.nf.DilatedRNN)
    run_model_3(df, sktimexnn.forecasting.nf.DLinear)
    run_model_3(df, sktimexnn.forecasting.nf.FEDformer)
    run_model_3(df, sktimexnn.forecasting.nf.GRU)
    run_model_3(df, sktimexnn.forecasting.nf.Informer)
    run_model_3(df, sktimexnn.forecasting.nf.iTransformer)
    run_model_3(df, sktimexnn.forecasting.nf.KAN)
    run_model_3(df, sktimexnn.forecasting.nf.LSTM)
    run_model_3(df, sktimexnn.forecasting.nf.MLP)
    run_model_3(df, sktimexnn.forecasting.nf.MLPMultivariate)
    run_model_3(df, sktimexnn.forecasting.nf.NBEATS)
    run_model_3(df, sktimexnn.forecasting.nf.NBEATSx)
    run_model_3(df, sktimexnn.forecasting.nf.NHITS)
    run_model_3(df, sktimexnn.forecasting.nf.NLinear)
    run_model_3(df, sktimexnn.forecasting.nf.PatchTST)
    run_model_3(df, sktimexnn.forecasting.nf.RMoK)
    run_model_3(df, sktimexnn.forecasting.nf.RNN)
    run_model_3(df, sktimexnn.forecasting.nf.SOFTS)
    run_model_3(df, sktimexnn.forecasting.nf.StemGNN)
    run_model_3(df, sktimexnn.forecasting.nf.TCN)
    run_model_3(df, sktimexnn.forecasting.nf.TFT)
    run_model_3(df, sktimexnn.forecasting.nf.TiDE)
    # run_model_3(df, sktimexnn.forecasting.nf.TimeLLM)   # out of memory
    run_model_3(df, sktimexnn.forecasting.nf.TimeMixer)
    run_model_3(df, sktimexnn.forecasting.nf.TimesNet)
    run_model_3(df, sktimexnn.forecasting.nf.TimeXer)
    run_model_3(df, sktimexnn.forecasting.nf.TSMixer)
    run_model_3(df, sktimexnn.forecasting.nf.TSMixerx)
    run_model_3(df, sktimexnn.forecasting.nf.VanillaTransformer)
    run_model_3(df, sktimexnn.forecasting.nf.xLSTM)
    pass


def main():
    print("dataframe")
    df = create_syntethic_data(12*8, 0.0, 1, 0.33)

    # check_darts(df)
    check_nf(df)
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
