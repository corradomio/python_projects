import logging.config

from darts.models import *

import pandasx as pdx
import sktimex as sktx
import sktimexnn.forecasting.darts.rnn_model

DATASET = "./data/vw_food_import_kg_train_test_area_skill_mini.csv"
TARGET = "import_kg"
NUMERICAL=["crude_oil_price","sandp_500_us","sandp_sensex_india","shenzhen_index_china","nikkei_225_japan",
           "max_temperature","mean_temperature","min_temperature","vap_pressure","evaporation","rainy_days"]

def main():
    print("dataframe")
    df = pdx.read_data(
        DATASET,
        ignore_unnamed=True,
        ignore=["prod_kg","avg_retail_price_src_country","producer_price_tonne_src_country"],
        datetime=("date", "%Y/%m/%d %H:%M:%S", "M"),
        onehot=["imp_month"]
    )

    print("loop")
    dfdict = pdx.groups_split(df, groups=["country", "item"])
    for g in dfdict:
        print("...", g)
        dfg = dfdict[g]
        # pdx.set_index(dfg, "d_date", drop=True, inplace=True, as_datetime=True, freq="D")

        dfd = dfg.drop(columns=["date"])

        X, y = pdx.xy_split(dfd, target=TARGET)
        y_train_orig, y_test_orig = pdx.train_test_split(y, train_size=0.8)

        # xscaler = pdx.StandardScaler(columns=NUMERICAL)
        # yscaler = pdx.StandardScaler(columns=TARGET)
        xscaler = pdx.MinMaxScaler(columns=NUMERICAL)
        yscaler = pdx.MinMaxScaler(columns=TARGET)

        X_scaled = xscaler.fit_transform(X)
        y_scaled = yscaler.fit_transform(y)

        X_train, X_test, y_train, y_test = pdx.train_test_split(X_scaled, y_scaled, train_size=0.8)

        print("model")
        # model = sktime.forecasting.conditional_invertible_neural_network.CINNForecaster()
        # model = sktime.forecasting.neuralforecast.NeuralForecastRNN()   # ok
        # model = sktime.forecasting.neuralforecast.NeuralForecastLSTM()
        # model = sktimexnn.forecasting.darts.rnn_model.RNNModel(input_chunk_length=1)

        # model = NaiveSeasonal(K=7)
        # model = NaiveMovingAverage(input_chunk_length=7)
        # model = GlobalNaiveSeasonal(input_chunk_length=7, output_chunk_length=1)
        # model = RNNModel(input_chunk_length=7,  model='LSTM')
        #   only future_covariates
        # model = NBEATSModel(input_chunk_length=7, output_chunk_length=1)
        #   only past_covariates
        model = TFTModel(input_chunk_length=7, output_chunk_length=1)
        #   past+future
        model = sktimexnn.forecasting.darts.TFTModel(input_chunk_length=1)

        # t_train = TimeSeries.from_series(y_train)
        # c_train = TimeSeries.from_dataframe(X_train)
        # c_test = TimeSeries.from_dataframe(X_test)
        # c_future = TimeSeries.from_dataframe(X)

        print("fit")
        # model.fit(series=t_train,               # NOT 'target=t_train'
        #           past_covariates=c_train,
        #           future_covariates=c_train
        #           )
        model.fit(y_train, X_train)

        print("predict")
        # predict = model.predict(
        #     n=len(y_test.index),
        #     past_covariates=c_future,
        #     future_covariates=c_future          # c_train + c_test
        # )
        # y_predict_scaled = predict.to_series()
        y_predict_scaled = model.predict(fh=y_test.index, X=X_test)

        y_predict = yscaler.inverse_transform(y_predict_scaled)

        sktx.utils.plot_series(y_train_orig, y_test_orig, y_predict, labels=["train", "test", "predict"], title=g)
        sktx.utils.show()

        pass

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
