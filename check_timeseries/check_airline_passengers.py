import logging.config
import pandasx as pdx
import matplotlib.pyplot as plt
import numpy as np
import os
from sktime.utils.plotting import plot_series
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X), torch.tensor(y)


# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

        self.optimizer = None
        self.loss_fn = None

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        # x.shape = [8,1,1
        return x

    def compile(self, loss_fn, optimizer):
        self.optimizer = optim.Adam(self.parameters())
        self.loss_fn = nn.MSELoss()

    def train_(self, loader, X_train, y_train, X_test, y_test, n_epochs=2000):
        model = self
        loss_fn = self.loss_fn
        optimizer = self.optimizer

        for epoch in range(n_epochs):
            model.train()
            for X_batch, y_batch in loader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Validation
            if epoch % 25 != 0:
                continue
            model.eval()
            with torch.no_grad():
                y_pred = model(X_train)
                train_rmse = np.sqrt(loss_fn(y_pred, y_train))
                y_pred = model(X_test)
                test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    # end
# end


# https://towardsdatascience.com/time-series-forecasting-with-deep-learning-in-pytorch-lstm-rnn-1ba339885f0c
class LSTMForecaster(nn.Module):
    def __init__(self, n_features=1, n_hidden=50, n_outputs=1, sequence_len=1, n_lstm_layers=1, n_deep_layers=10,
                 dropout=0.2,
                 use_cuda=False):
        '''
        n_features: number of input features (1 for univariate forecasting)
        n_hidden: number of neurons in each hidden layer
        n_outputs: number of outputs to predict for each training example
        n_deep_layers: number of hidden dense layers after the lstm layer
        sequence_len: number of steps to look back at for prediction
        dropout: float (0 < dropout < 1) dropout ratio between dense layers
        '''
        super().__init__()

        self.n_lstm_layers = n_lstm_layers
        self.nhid = n_hidden
        self.use_cuda = use_cuda  # set option for device selection

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=n_hidden,
                            num_layers=n_lstm_layers,
                            batch_first=True)  # As we have transformed our data in this way

        # first dense after lstm
        self.fc1 = nn.Linear(in_features=n_hidden * sequence_len, out_features=n_hidden)
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Create fully connected layers (n_hidden x n_deep_layers)
        dnn_layers = []
        for i in range(n_deep_layers):
            # Last layer (n_hidden x n_outputs)
            if i == n_deep_layers - 1:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(in_features=self.nhid, out_features=n_outputs))
            # All other layers (n_hidden x n_hidden) with dropout option
            else:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(in_features=self.nhid, out_features=self.nhid))
                if dropout:
                    dnn_layers.append(nn.Dropout(p=dropout))
        # compile DNN layers
        self.dnn = nn.Sequential(*dnn_layers)

        self.optimizer = None
        self.criterion = None

    def forward(self, x):
        device = torch.device('cpu')

        # Initialize hidden state
        hidden_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)
        cell_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)

        # move hidden state to device
        if self.use_cuda:
            hidden_state = hidden_state.to(device)
            cell_state = cell_state.to(device)

        self.hidden = (hidden_state, cell_state)

        # Forward Pass
        x, h = self.lstm(x, self.hidden)  # LSTM
        x = self.dropout(x.contiguous().view(x.shape[0], -1))  # Flatten lstm out
        x = self.fc1(x)  # First Dense
        y = self.dnn(x)  # Pass forward through fully connected DNN.
        # y.shape = [8,1]
        return y

    def compile(self, lr=4e-4):
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def train_(self, trainloader, X_train, y_train, X_test, y_test, n_epochs=2000):
        model = self
        optimizer = self.optimizer
        criterion = self.criterion

        # Lists to store training and validation losses
        t_losses, v_losses = [], []
        # Loop over epochs
        for epoch in range(n_epochs):
            train_loss, valid_loss = 0.0, 0.0

            # train step
            model.train()
            # Loop over train dataset
            for x, y in trainloader:
                optimizer.zero_grad()
                # move inputs to device
                x = x
                y = y.squeeze()
                # Forward Pass
                preds = model(x).squeeze()
                loss = criterion(preds, y)  # compute batch loss
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            epoch_loss = train_loss / len(trainloader)
            t_losses.append(epoch_loss)

            # validation step
            model.eval()
            # Loop over validation dataset
            for x, y in testloader:
                with torch.no_grad():
                    x, y = x.to(device), y.squeeze().to(device)
                    preds = model(x).squeeze()
                    error = criterion(preds, y)
                valid_loss += error.item()
            valid_loss = valid_loss / len(testloader)
            v_losses.append(valid_loss)

            print(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')
        plot_losses(t_losses, v_losses)


def model_train(X_train, y_train, X_test, y_test, n_epochs=2000, lookback=1):

    model = AirModel()
    # model = LSTMForecaster(sequence_len=lookback)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

    # n_epochs = 2000
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 25 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    return model


def model_predict(model, timeseries, X_train, X_test, lookback=1):
    # lookback = 1
    train_size = len(X_train) + lookback
    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(timeseries) * np.nan
        # shape: [95,1, 1]
        y_pred = model(X_train)
        # shape: [95,1, 1]
        y_pred = torch.reshape(y_pred, (-1, lookback, 1))
        # shape: [95, 1]
        y_pred = y_pred[:, -1, :]
        # train_plot[lookback:train_size] = model(X_train)[:, -1, :]
        train_plot[lookback:train_size] = y_pred
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan

        y_test = model(X_test)
        y_test = torch.reshape(y_test, (-1, lookback, 1))
        test_plot[train_size + lookback:len(timeseries)] = y_test[:, -1, :]
    # plot
    plt.plot(timeseries, c='b')
    plt.plot(train_plot, c='r')
    plt.plot(test_plot, c='g')
    plt.title(f'lookbakc={lookback}')
    plt.show()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    df = pdx.read_data(
        "D:/Dropbox/Datasets/kaggle/airline-passengers.csv",
        datetime=('Month', '%Y-%m', 'M'),
        ignore=['Month'],
        index=['Month'])
    print(len(df))

    plot_series(df[["#Passengers"]], labels=['#Passengers'])
    plt.tight_layout()
    plt.show()

    timeseries = df[["#Passengers"]].values.astype('float32')

    df_train, df_test = pdx.train_test_split(df, train_size=0.67)
    np_train = df_train.to_numpy().astype('float32')
    np_test = df_test.to_numpy().astype('float32')

    for lookback in [3, 1, 3, 4, 12]:
        # lookback = 12
        X_train, y_train = create_dataset(np_train, lookback=lookback)
        X_test, y_test = create_dataset(np_test, lookback=lookback)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        model = model_train(X_train, y_train, X_test, y_test, n_epochs=20, lookback=lookback)
        model_predict(model, timeseries, X_train, X_test, lookback=lookback)


    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
