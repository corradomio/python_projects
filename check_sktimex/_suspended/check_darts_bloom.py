import logging.config

import pandas as pd
import sqlalchemy as sa
from darts import TimeSeries
from darts.models import *

import pandasx as pdx
import sktimex as sktx


def main():
    print("dataframe")
    engine = sa.create_engine("postgresql://postgres:p0stgres@10.193.20.14:5432/bloom")
    with engine.connect() as conn:
        df = pd.read_sql("select * from vw_daily_energy_input_train", conn)

    # d_date', 'building', 'consumption'
    print(df.columns)
    print(df.head())

    print("loop")
    dfdict = pdx.groups_split(df, groups="building")
    for g in dfdict:
        dfg = dfdict[g]
        # pdx.set_index(dfg, "d_date", drop=True, inplace=True, as_datetime=True, freq="D")

        dfd = dfg.drop(columns=["d_date"])

        _, y = pdx.xy_split(dfd, target="consumption")
        y_train, y_test = pdx.xy_split(y, target="consumption")

        print("model")
        # model = sktime.forecasting.conditional_invertible_neural_network.CINNForecaster()
        # model = sktime.forecasting.neuralforecast.NeuralForecastRNN()   # ok
        # model = sktime.forecasting.neuralforecast.NeuralForecastLSTM()
        # model = sktimexnn.forecasting.darts.rnn_model.RNNModel(input_chunk_length=1)

        # model = NaiveSeasonal(K=7)
        # model = NaiveMovingAverage(input_chunk_length=7)
        # model = GlobalNaiveSeasonal(input_chunk_length=7, output_chunk_length=1)
        model = RNNModel(input_chunk_length=7,  model='LSTM')

        t_train = TimeSeries.from_series(y_train)
        # c_train = TimeSeries.from_dataframe(X_train)
        # c_predict = TimeSeries.from_dataframe(X_test)
        # c_future = TimeSeries.from_dataframe(X)

        print("fit")
        model.fit(
            target=t_train,
            # past_covariates=c_train,
            # future_covariates=c_future
        )

        print("predict")
        predict = model.predict(
            n=len(y_test.index),
            # future_covariates=c_predict
        )
        y_predict = predict.pd_series()

        sktx.utils.plot_series(y_train, y_test, y_predict, labels=["train", "test", "predict"], title=g)
        sktx.utils.show()

        pass

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
