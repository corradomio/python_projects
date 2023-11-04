#
# A LSTM-based seq2seq model for time series forecasting
# https://medium.com/@shouke.wei/a-lstm-based-seq2seq-model-for-time-series-forecasting-3730822301c5
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Input
from keras.models import Model
from keras.src.utils import plot_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


def main1():
    #
    # 2)
    #
    # Load the CSV file
    data = pd.read_csv(
        'https://raw.githubusercontent.com/NourozR/Stock-Price-Prediction-LSTM/master/apple_share_price.csv')

    # Convert the 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Sort the dataset by date
    df = data.sort_values('Date').reset_index(drop=True)

    # Extract the 'Close' price column
    prices = df['Close'].values.reshape(-1, 1)

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    #
    # 3)
    #
    # Define the training and testing data sizes
    train_size = int(len(scaled_prices) * 0.8)
    test_size = len(scaled_prices) - train_size

    # Split the data into training and testing sets
    train_data = scaled_prices[:train_size]
    test_data = scaled_prices[train_size:]

    #
    # 4)
    #
    # Define the number of previous time steps to use for prediction
    n_steps = 10

    # Generate training sequences
    X_train, y_train = [], []
    for i in range(n_steps, len(train_data)):
        X_train.append(train_data[i - n_steps:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the lists to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape the input data to fit the LSTM input shape
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    #
    # 5)
    #
    # Define the encoder input
    encoder_input = Input(shape=(n_steps, 1))

    # Encoder LSTM
    # defaults:
    #   return_sequences=False
    #   return_state=False
    encoder_lstm = LSTM(128, return_sequences=True)(encoder_input)
    _, state_h, state_c = LSTM(128, return_state=True)(encoder_lstm)
    encoder_states = [state_h, state_c]

    # Define the decoder input (a che serve?)
    decoder_input = Input(shape=(1, 1))

    # Decoder LSTM
    # decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
    # decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
    decoder_lstm = LSTM(128, return_sequences=True, return_state=False)
    decoder_output = decoder_lstm(decoder_input, initial_state=encoder_states)

    # Dense layer
    decoder_output = Dense(1)(decoder_output)

    # Define the model
    model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    plot_model(model, show_shapes=True, show_layer_activations=True)
    plt.show()

    #
    # 6)
    #
    # Define callbacks for early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('lstm_seq2seq.h5', monitor='val_loss', save_best_only=True, verbose=1)

    # Train the model with validation.
    # 1) che cazzata! Converte y_train in un tensore [n, 1, 1]. OVVIO
    # 2) che cazzata! La rete richiede DUE vettori per l'input, una per l'encoder ed una per il decoder
    #    MA per il decoder usa un BANALE VETTORE DI ZERO!
    y_train = y_train.reshape(-1, 1, 1)
    z_train = np.zeros_like(y_train)
    history = model.fit([X_train, z_train], y_train, epochs=100, batch_size=32, validation_split=0.2,
                        callbacks=[early_stopping, model_checkpoint])

    #
    # 6.1)
    #
    # Plot the training and validation loss curves
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    #
    # 7)
    #
    # Generate testing sequences
    X_test, y_test = [], []
    for i in range(n_steps, len(test_data)):
        X_test.append(test_data[i - n_steps:i, 0])
        y_test.append(test_data[i, 0])

    # Convert the lists to numpy arrays
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Reshape the input data to fit the LSTM input shape
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Make predictions
    decoder_input = np.zeros((X_test.shape[0], 1, 1))
    predictions = model.predict([X_test, decoder_input])

    # Reshape the y_test and predictions
    y_test = y_test.reshape(-1, 1)
    predictions = predictions[:, :, 0]

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)

    # Scale back the predicted values to their original range
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    # Visualize the predicted values versus the actual values
    plt.plot(predictions, label='Predicted')
    plt.plot(y_test, label='Actual')
    plt.title('Predicted vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
# end


def main2():
    #
    # 2)
    #
    # Load the CSV file
    data = pd.read_csv(
        'https://raw.githubusercontent.com/NourozR/Stock-Price-Prediction-LSTM/master/apple_share_price.csv')

    # Convert the 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Sort the dataset by date
    df = data.sort_values('Date').reset_index(drop=True)

    # Extract the 'Close' price column
    prices = df['Close'].values.reshape(-1, 1)

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    #
    # 3)
    #
    # Define the training and testing data sizes
    train_size = int(len(scaled_prices) * 0.8)
    test_size = len(scaled_prices) - train_size

    # Split the data into training and testing sets
    train_data = scaled_prices[:train_size]
    test_data = scaled_prices[train_size:]

    #
    # 4)
    #
    # Define the number of previous time steps to use for prediction
    n_steps = 10

    # Generate training sequences
    X_train, y_train = [], []
    for i in range(n_steps, len(train_data)):
        X_train.append(train_data[i - n_steps:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the lists to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape the input data to fit the LSTM input shape
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    #
    # 5)
    #
    # Define the encoder input
    encoder_input = Input(shape=(n_steps, 1))

    # Encoder LSTM
    # defaults:
    #   return_sequences=False
    #   return_state=False
    encoder_lstm, state_h, state_c = LSTM(128, return_sequences=True, return_state=True)(encoder_input)
    encoder_states = [state_h, state_c]

    # Define the decoder input (a che serve?)
    decoder_input = Input(shape=(1, 1))

    # Decoder LSTM
    # decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
    # decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
    decoder_lstm = LSTM(128, return_sequences=True, return_state=False)
    decoder_output = decoder_lstm(decoder_input, initial_state=encoder_states)

    # Dense layer
    decoder_output = Dense(1)(decoder_output)

    # Define the model
    model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    plot_model(model, show_shapes=True, show_layer_activations=True)
    plt.show()

    #
    # 6)
    #
    # Define callbacks for early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('lstm_seq2seq.h5', monitor='val_loss', save_best_only=True, verbose=1)

    # Train the model with validation.
    # 1) che cazzata! Converte y_train in un tensore [n, 1, 1]. OVVIO
    # 2) che cazzata! La rete richiede DUE vettori per l'input, una per l'encoder ed una per il decoder
    #    MA per il decoder usa un BANALE VETTORE DI ZERO!
    y_train = y_train.reshape(-1, 1, 1)
    z_train = np.zeros_like(y_train)
    history = model.fit([X_train, z_train], y_train, epochs=100, batch_size=32, validation_split=0.2,
                        callbacks=[early_stopping, model_checkpoint])

    #
    # 6.1)
    #
    # Plot the training and validation loss curves
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    #
    # 7)
    #
    # Generate testing sequences
    X_test, y_test = [], []
    for i in range(n_steps, len(test_data)):
        X_test.append(test_data[i - n_steps:i, 0])
        y_test.append(test_data[i, 0])

    # Convert the lists to numpy arrays
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Reshape the input data to fit the LSTM input shape
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Make predictions
    decoder_input = np.zeros((X_test.shape[0], 1, 1))
    predictions = model.predict([X_test, decoder_input])

    # Reshape the y_test and predictions
    y_test = y_test.reshape(-1, 1)
    predictions = predictions[:, :, 0]

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)

    # Scale back the predicted values to their original range
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    # Visualize the predicted values versus the actual values
    plt.plot(predictions, label='Predicted')
    plt.plot(y_test, label='Actual')
    plt.title('Predicted vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
# end


def main3():
    #
    # 2)
    #
    # Load the CSV file
    data = pd.read_csv(
        'https://raw.githubusercontent.com/NourozR/Stock-Price-Prediction-LSTM/master/apple_share_price.csv')

    # Convert the 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Sort the dataset by date
    df = data.sort_values('Date').reset_index(drop=True)

    # Extract the 'Close' price column
    prices = df['Close'].values.reshape(-1, 1)

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    #
    # 3)
    #
    # Define the training and testing data sizes
    train_size = int(len(scaled_prices) * 0.8)
    test_size = len(scaled_prices) - train_size

    # Split the data into training and testing sets
    train_data = scaled_prices[:train_size]
    test_data = scaled_prices[train_size:]

    #
    # 4)
    #
    # Define the number of previous time steps to use for prediction
    n_steps = 10
    hidden_size = 32

    # Generate training sequences
    X_train, y_train = [], []
    for i in range(n_steps, len(train_data)):
        X_train.append(train_data[i - n_steps:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the lists to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape the input data to fit the LSTM input shape
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    #
    # 5)
    #
    # Define the encoder input
    encoder_input = Input(shape=(n_steps, 1))

    # Encoder LSTM
    # defaults:
    #   return_sequences=False
    #   return_state=False
    encoder_lstm, state_h, state_c = LSTM(hidden_size,
                                          return_sequences=True,
                                          return_state=True)(encoder_input)
    encoder_states = [state_h, state_c]

    # Define the decoder input (a che serve?)
    decoder_input = Input(shape=(1, 1))

    # Decoder LSTM
    # decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
    # decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
    decoder_lstm = LSTM(hidden_size,
                        return_sequences=True,
                        return_state=False)
    decoder_output = decoder_lstm(decoder_input, initial_state=encoder_states)

    # Dense layer
    decoder_output = Dense(1)(decoder_output)

    # Define the model
    model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    plot_model(model, show_shapes=True, show_layer_activations=True)
    plt.show()

    #
    # 6)
    #
    # Define callbacks for early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('lstm_seq2seq.h5', monitor='val_loss', save_best_only=True, verbose=1)

    # Train the model with validation.
    # 1) che cazzata! Converte y_train in un tensore [n, 1, 1]. OVVIO
    # 2) che cazzata! La rete richiede DUE vettori per l'input, una per l'encoder ed una per il decoder
    #    MA per il decoder usa un BANALE VETTORE DI ZERO!
    y_train = y_train.reshape(-1, 1, 1)
    z_train = np.zeros_like(y_train)
    history = model.fit([X_train, z_train], y_train, epochs=100, batch_size=32, validation_split=0.2,
                        callbacks=[early_stopping, model_checkpoint])

    #
    # 6.1)
    #
    # Plot the training and validation loss curves
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    #
    # 7)
    #
    # Generate testing sequences
    X_test, y_test = [], []
    for i in range(n_steps, len(test_data)):
        X_test.append(test_data[i - n_steps:i, 0])
        y_test.append(test_data[i, 0])

    # Convert the lists to numpy arrays
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Reshape the input data to fit the LSTM input shape
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Make predictions
    decoder_input = np.zeros((X_test.shape[0], 1, 1))
    predictions = model.predict([X_test, decoder_input])

    # Reshape the y_test and predictions
    y_test = y_test.reshape(-1, 1)
    predictions = predictions[:, :, 0]

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)

    # Scale back the predicted values to their original range
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    # Visualize the predicted values versus the actual values
    plt.plot(predictions, label='Predicted')
    plt.plot(y_test, label='Actual')
    plt.title('Predicted vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
# end

def main():
    #
    # 2)
    #
    # Load the CSV file
    data = pd.read_csv(
        'https://raw.githubusercontent.com/NourozR/Stock-Price-Prediction-LSTM/master/apple_share_price.csv')

    # Convert the 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Sort the dataset by date
    df = data.sort_values('Date').reset_index(drop=True)

    # Extract the 'Close' price column
    prices = df['Close'].values.reshape(-1, 1)

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    #
    # 3)
    #
    # Define the training and testing data sizes
    train_size = int(len(scaled_prices) * 0.8)
    test_size = len(scaled_prices) - train_size

    # Split the data into training and testing sets
    train_data = scaled_prices[:train_size]
    test_data = scaled_prices[train_size:]

    #
    # 4)
    #
    # Define the number of previous time steps to use for prediction
    n_steps = 10
    hidden_size = 32

    # Generate training sequences
    X_train, y_train = [], []
    for i in range(n_steps, len(train_data)):
        X_train.append(train_data[i - n_steps:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the lists to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape the input data to fit the LSTM input shape
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    #
    # 5)
    #
    # Define the encoder input
    encoder_input = Input(shape=(n_steps, 1))

    # Encoder LSTM
    # defaults:
    #   return_sequences=False
    #   return_state=False
    encoder_lstm, state_h, state_c = LSTM(hidden_size,
                                          return_sequences=True,
                                          return_state=True,
                                          activation=None)(encoder_input)
    encoder_states = [state_h, state_c]

    # Define the decoder input (a che serve?)
    decoder_input = Input(shape=(1, 1))

    # Decoder LSTM
    # decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
    # decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
    decoder_lstm = LSTM(hidden_size,
                        return_sequences=True,
                        return_state=False,
                        activation=None)
    decoder_output = decoder_lstm(decoder_input, initial_state=encoder_states)

    # Dense layer
    decoder_output = Dense(1)(decoder_output)

    # Define the model
    model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    plot_model(model, show_shapes=True, show_layer_activations=True)
    plt.show()

    #
    # 6)
    #
    # Define callbacks for early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('lstm_seq2seq.h5', monitor='val_loss', save_best_only=True, verbose=1)

    # Train the model with validation.
    # 1) che cazzata! Converte y_train in un tensore [n, 1, 1]. OVVIO
    # 2) che cazzata! La rete richiede DUE vettori per l'input, una per l'encoder ed una per il decoder
    #    MA per il decoder usa un BANALE VETTORE DI ZERO!
    y_train = y_train.reshape(-1, 1, 1)
    z_train = np.zeros_like(y_train)
    history = model.fit([X_train, z_train], y_train, epochs=100, batch_size=32, validation_split=0.2,
                        callbacks=[early_stopping, model_checkpoint])

    #
    # 6.1)
    #
    # Plot the training and validation loss curves
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    #
    # 7)
    #
    # Generate testing sequences
    X_test, y_test = [], []
    for i in range(n_steps, len(test_data)):
        X_test.append(test_data[i - n_steps:i, 0])
        y_test.append(test_data[i, 0])

    # Convert the lists to numpy arrays
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Reshape the input data to fit the LSTM input shape
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Make predictions
    decoder_input = np.zeros((X_test.shape[0], 1, 1))
    predictions = model.predict([X_test, decoder_input])

    # Reshape the y_test and predictions
    y_test = y_test.reshape(-1, 1)
    predictions = predictions[:, :, 0]

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)

    # Scale back the predicted values to their original range
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    # Visualize the predicted values versus the actual values
    plt.plot(predictions, label='Predicted')
    plt.plot(y_test, label='Actual')
    plt.title('Predicted vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
# end


if __name__ == "__main__":
    main()
