#
# A LSTM-based seq2seq model for time series forecasting
# https://medium.com/@shouke.wei/a-lstm-based-seq2seq-model-for-time-series-forecasting-3730822301c5
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import LSTM, Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandasx as pdx
from sktimex.transform import RNNTrainTransform, RNNPredictTransform


def main():

    data = pdx.read_data(
        'https://raw.githubusercontent.com/NourozR/Stock-Price-Prediction-LSTM/master/apple_share_price.csv',
        datetime=('Date', '%d-%b-%y', 'D'),
        ignore=['Date'],
        index=['Date'],
        sort=True,
        reindex=False
    )

    prices = data[['Close']]
    scaler = pdx.MinMaxScaler(quantile=0.05)
    prices_scaled = scaler.fit_transform(prices)

    train_data, test_data = pdx.train_test_split(prices_scaled, train_size=0.8)

    tt = RNNTrainTransform(slots=10, tlags=1, flatten=False)
    Xt, yt = tt.fit_transform(X=None, y=train_data)



    pass
# end


if __name__ == "__main__":
    main()
