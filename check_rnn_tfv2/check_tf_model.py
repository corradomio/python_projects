import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

nodes=8

def main():
    # Initialising the RNN
    regressor = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=nodes, return_sequences=True, input_shape=(new.shape[1], new.shape[2])))
    regressor.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=nodes, return_sequences=True))
    regressor.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=nodes, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=nodes))
    regressor.add(Dropout(0.2))
    # Adding the output layer
    regressor.add(Dense(units=t))  # this is the output layer so this repersetns a single node with our output value
    # Compiling the RNN
    # regressor.compile(optimizer='adam', loss='mean_squared_error')
    # Fitting the RNN to the Training set
    # model = regressor.fit(new, y_train, epochs=50, batch_size=22)



if __name__ == "__main__":
    main()
