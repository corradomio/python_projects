import matplotlib.pyplot as plt
import logging.config

from darts import TimeSeries
from darts.models import *
from synth import create_syntethic_data

import pandasx as pdx
import sktimex as sktx
import sktimexnn.forecasting.darts
import warnings

# Suppress all UserWarning instances
warnings.simplefilter("ignore", UserWarning)

TARGET = "data"

def main():
    print("dataframe")
    df = create_syntethic_data()

    # print("loop")
    dfdict = pdx.groups_split(df, groups=["cat"])
    for g in dfdict:
        print("...", g)
        dfg = dfdict[g]

        X, y = pdx.xy_split(dfg, target=TARGET)

        # xscaler = pdx.StandardScaler(feature_range=(0,1), columns=NUMERICAL)
        # yscaler = pdx.StandardScaler(feature_range=(0,1), columns=TARGET)
        # xscaler = pdx.LinearMinMaxScaler(feature_range=(-1,1), columns=NUMERICAL)
        # yscaler = pdx.LinearMinMaxScaler(feature_range=(-1,1), columns=TARGET, clip=0)
        # xscaler = pdx.LinearMinMaxScaler(feature_range=(0,1), columns=NUMERICAL)
        # yscaler = pdx.LinearMinMaxScaler(feature_range=(0,1), columns=TARGET, clip=0)

        y_scaled = y
        # y_scaled = yscaler.fit_transform(y)

        y_train, y_test = pdx.train_test_split(y_scaled, train_size=0.8)

        # print("model")
        # model = sktime.forecasting.conditional_invertible_neural_network.CINNForecaster()
        # model = sktime.forecasting.neuralforecast.NeuralForecastRNN()   # ok
        # model = sktime.forecasting.neuralforecast.NeuralForecastLSTM()
        # model = sktimexnn.forecasting.darts.rnn_model.RNNModel(input_chunk_length=1)

        # model = NaiveSeasonal(K=7)
        # model = NaiveMovingAverage(input_chunk_length=7)
        # model = GlobalNaiveSeasonal(input_chunk_length=7, output_chunk_length=1)
        # model = RNNModel(input_chunk_length=7,  model='LSTM')
        #   only future_covariates
        model = NBEATSModel(input_chunk_length=14, output_chunk_length=1)
        #   past+future

        t_train = TimeSeries.from_series(y_train)

        # print("fit")
        model.fit(
            series=t_train,               # NOT 'target=t_train'
        )

        # print("predict")
        predict = model.predict(
            n=len(y_test.index),
        )

        y_predict = sktx.darts.to_series(predict)

        sktx.utils.plot_series(y_train, y_test, y_predict, labels=["train", "test", "predict"], title=g)
        plt.ylim((-1.2, 1.2))

        fname = f"preds/{model.__class__.__name__}-{g[0]}.png"
        plt.savefig(fname, dpi=300)

        pass

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
