import logging.config

import pandas as pd
import sqlalchemy as sa

import pandasx as pdx
import sktimex as sktx
import sktimexnn.forecasting.darts.rnn_model


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
        pdx.set_index(dfg, "d_date", drop=True, inplace=True, as_datetime=True, freq="D")

        train, test = pdx.train_test_split(dfg, train_size=0.8)
        X_train, y_train, X_test, y_test = pdx.xy_split(train, test, target="consumption")

        print("model")
        # model = sktime.forecasting.conditional_invertible_neural_network.CINNForecaster()
        # model = sktime.forecasting.neuralforecast.NeuralForecastRNN()   # ok
        # model = sktime.forecasting.neuralforecast.NeuralForecastLSTM()
        model = sktimexnn.forecasting.darts.rnn_model.RNNModel(input_chunk_length=1)

        print("fit")
        model.fit(y=y_train)

        print("predict")
        y_predict = model.predict(fh=y_test.index)

        sktx.utils.plot_series(y_train, y_test, y_predict, labels=["train", "test", "predict"], title=g)
        sktx.utils.show()

        pass

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
