import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

df = pd.read_csv('airline-passengers.csv')
timeseries = df[["Passengers"]].values.astype('float32')

# train-test split for time series
train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]


def create_dataset(dataset, lookback, last=False):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        X.append(feature)
        if last:
            # target = dataset[i + lookback:i + lookback + 1]
            target = dataset[i + lookback]
        else:
            target = dataset[i + 1:i + lookback + 1]
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


class AirModel(nn.Module):
    def __init__(self, last=False):
        super().__init__()
        self.last = last
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        t, _ = self.lstm(x)
        if self.last:
            t = t[:, -1, :]
        # t = self.relu1(t)
        t = self.linear(t)
        t = self.relu1(t)
        return t


last = False
lookback = 4
X_train, y_train = create_dataset(train, lookback=lookback, last=last)
X_test, y_test = create_dataset(test, lookback=lookback, last=last)


model = AirModel(last=last)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 2000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    if not last:
        y_pred = y_pred[:, -1, :]
    # train_plot[lookback:train_size] = model(X_train)[:, -1, :]
    train_plot[lookback:train_size] = y_pred
    # shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    y_test = model(X_test)
    if not last:
        y_test = y_test[:, -1, :]
    # test_plot[train_size + lookback:len(timeseries)] = model(X_test)[:, -1, :]
    test_plot[train_size + lookback:len(timeseries)] = y_test
# plot
plt.plot(timeseries)
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()