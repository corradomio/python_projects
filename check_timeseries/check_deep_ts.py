import logging.config
import pandasx as pdx
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.nn as nn
import torchx
from sktime.utils.plotting import plot_series
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from torchx import Module, compose_data


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

    y = df.to_numpy(dtype=float)
    plt.plot(y)
    plt.show()

    input_size = 1
    output_size = 1
    n_hidden = 50
    sequence_len = 7
    n_lstm_layers = 1
    dropout = 0.2

    m = torchx.Module(
        model=[
            torchx.LSTM(
                input_size=input_size,
                hidden_size=n_hidden,
                num_layers=n_lstm_layers,
                batch_first=True),
            torchx.DropDimensions(dim=-1),
            # 0
            nn.Dropout(p=dropout),
            nn.Linear(in_features=n_hidden * sequence_len, out_features=n_hidden),
            # 1
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_hidden),
            nn.Dropout(p=dropout),
            # 2
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_hidden),
            nn.Dropout(p=dropout),
            # 3
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_hidden),
            nn.Dropout(p=dropout),
            # 4
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_hidden),
            nn.Dropout(p=dropout),
            # 5
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=output_size)
        ],
        batch_size=16,
        epochs=200
    )

    Xt, yt = compose_data(y=y, slots=7)

    m.fit(Xt, yt)
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
