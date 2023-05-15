# Dataset
# https://alfred.stlouisfed.org/series/downloaddata?seid=DCOILWTICO

import pandasx as pdx
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.nn as nn
from sktime.utils.plotting import plot_series
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler


# https://towardsdatascience.com/time-series-forecasting-with-deep-learning-in-pytorch-lstm-rnn-1ba339885f0c

# Defining a function that creates sequences and targets as shown above
def generate_sequences(df: pd.DataFrame, tw: int, pw: int, target_columns, drop_targets=False):
    '''
    df: Pandas DataFrame of the univariate time-series
    tw: Training Window - Integer defining how many steps to look back
    pw: Prediction Window - Integer defining how many steps forward to predict

    returns: dictionary of sequences and targets for all sequences
    '''
    data = dict()  # Store results into a dictionary
    L = len(df)
    for i in range(L - tw):
        # Option to drop target from dataframe
        if drop_targets:
            df.drop(target_columns, axis=1, inplace=True)

        # Get current sequence
        sequence = df[i:i + tw].values
        # Get values right after the current sequence
        target = df[i + tw:i + tw + pw][target_columns].values
        data[i] = {'sequence': sequence, 'target': target}
    return data


class SequenceDataset(Dataset):

    def __init__(self, df):
        self.data = df

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.Tensor(sample['sequence']), torch.Tensor(sample['target'])

    def __len__(self):
        return len(self.data)


class LSTMForecaster(nn.Module):

    def __init__(self, n_features, n_hidden, n_outputs, sequence_len, n_lstm_layers=1, n_deep_layers=10, use_cuda=False,
                 dropout=0.2):
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
        self.lstm = nn.LSTM(n_features,
                            n_hidden,
                            num_layers=n_lstm_layers,
                            batch_first=True)  # As we have transformed our data in this way

        # first dense after lstm
        self.fc1 = nn.Linear(n_hidden * sequence_len, n_hidden)
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Create fully connected layers (n_hidden x n_deep_layers)
        dnn_layers = []
        for i in range(n_deep_layers):
            # Last layer (n_hidden x n_outputs)
            if i == n_deep_layers - 1:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(self.nhid, n_outputs))
            # All other layers (n_hidden x n_hidden) with dropout option
            else:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(self.nhid, self.nhid))
                if dropout:
                    dnn_layers.append(nn.Dropout(p=dropout))
        # compile DNN layers
        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, x):
        device = 'cuda' if self.use_cuda else 'cpu'

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
        return self.dnn(x)  # Pass forward through fully connected DNN.


def make_predictions_from_dataloader(model, unshuffled_dataloader):
    model.eval()
    predictions, actuals = [], []
    for x, y in unshuffled_dataloader:
        with torch.no_grad():
            p = model(x)
            predictions.append(p)
            actuals.append(y.squeeze())
    predictions = torch.cat(predictions).numpy()
    actuals = torch.cat(actuals).numpy()
    return predictions.squeeze(), actuals


def plot_losses(t_losses, v_losses):
    plt.clf()
    plt.plot(t_losses, label="train loss")
    plt.plot(v_losses, label="validation loss")
    plt.legend()
    plt.show()

    pass


def main():
    df = pdx.read_data(
        "D:\\Dropbox\\Datasets\\kaggle\\cdoilwtico_valid.csv",
        datetime=("observation_date", "%Y-%m-%d", 'D'),
        index=["observation_date"],
        ignore=["observation_date"],
        reindex=True)

    plot_series(df, labels=["cdoilwtico"], markers=',')
    plt.tight_layout()
    plt.show()

    # Fit scalers
    scalers: dict[str, StandardScaler] = {}
    for c in df.columns:
        scalers[c] = StandardScaler().fit(df[c].values.reshape(-1, 1))

    # Transform data via scalers
    norm_df = df.copy()
    for i, key in enumerate(scalers.keys()):
        norm = scalers[key].transform(norm_df.iloc[:, i].values.reshape(-1, 1))
        norm_df.iloc[:, i] = norm

    # Here we are defining properties for our model

    BATCH_SIZE = 16  # Training batch size
    split = 0.8  # Train/Test Split ratio
    sequence_len = 7
    nout = 1
    ninp = 1

    sequences = generate_sequences(norm_df.dcoilwtico.to_frame(), sequence_len, nout, 'dcoilwtico')
    dataset = SequenceDataset(sequences)

    # Split the data according to our split ratio and load each subset into a
    # separate DataLoader object
    train_len = int(len(dataset) * split)
    lens = [train_len, len(dataset) - train_len]
    train_ds, test_ds = random_split(dataset, lens)
    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    nhid = 50  # Number of nodes in the hidden layer
    n_dnn_layers = 5  # Number of hidden fully connected layers
    # nout = 1  # Prediction Window
    # sequence_len = 180  # Training Window

    # Number of features (since this is a univariate timeseries we'll set
    # this to 1 -- multivariate analysis is coming in the future)
    # ninp = 1

    # Device selection (CPU | GPU)
    USE_CUDA = torch.cuda.is_available()
    device = 'cuda' if USE_CUDA else 'cpu'

    # Initialize the model
    model = LSTMForecaster(ninp, nhid, nout, sequence_len, n_deep_layers=n_dnn_layers, use_cuda=USE_CUDA).to(device)

    # Set learning rate and number of epochs to train over
    lr = 4e-4
    n_epochs = 20

    # Initialize the loss function and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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
            x = x.to(device)
            y = y.squeeze().to(device)
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

    pass


if __name__ == "__main__":
    main()
