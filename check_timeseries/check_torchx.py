import pandas as pd
import pandasx as pdx
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split

from torchx import ConfigurableModule


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


def prepare_data():
    df = pdx.read_data(
        "D:\\Dropbox\\Datasets\\kaggle\\cdoilwtico_valid.csv",
        datetime=("observation_date", "%Y-%m-%d", 'D'),
        index=["observation_date"],
        ignore=["observation_date"],
        reindex=True)

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


def main():

    input_size = 1
    output_size = 1
    n_hidden = 50
    sequence_len = 7
    num_layers = 1
    dropout = 0.2

    cm = ConfigurableModule(
        layers=[
            # LSTM
            ["nn.LSTM", {'input_size': 'input_size_',
                         'hidden_size': 'n_hidden_',
                         'num_layers': 'num_layers_',
                         'batch_first': True}],
            # first hidden layer
            {
                'layer': 'torch.nn.Linear',
                'in_features': 'n_hidden_ * sequence_len_',
                'out_features': 'n_hidden_'
            },
            ["nn.Dropout", {"p": 'dropout_'}],

            # sublayer 0
            "ReLU",
            ["Linear", {"in_features": 'n_hidden_', "out_features": 'n_hidden_'}],
            ["Dropout", {"p": 'dropout_'}],
            # sublayer 1
            "ReLU",
            ["Linear", {"in_features": 'n_hidden_', "out_features": 'n_hidden_'}],
            ["Dropout", {"p": 'dropout_'}],
            # sublayer 2
            "ReLU",
            ["Linear", {"in_features": 'n_hidden_', "out_features": 'n_hidden_'}],
            ["nn.Dropout", {"p": 'dropout_'}],
            # sublayer 3
            "ReLU",
            ["Linear", {"in_features": 'n_hidden_', "out_features": 'n_hidden_'}],
            ["Dropout", {"p": 'dropout_'}],
            # sublayer 4
            "ReLU",
            ["Linear", {"in_features": 'n_hidden_', "out_features": 'output_size_'}],
        ],
        input_size_=input_size,
        output_size_=output_size,
        n_hidden_=n_hidden,
        dropout_=dropout,
        num_layers_=num_layers,
        sequence_len_=sequence_len
    )

    cm.compile(loss="nn.MSELoss", optimizer={
        "optimizer": "torch.optim.AdamW",
        "lr": 4e-4
    })


    pass


if __name__ == "__main__":
    main()