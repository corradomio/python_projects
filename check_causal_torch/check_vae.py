import numpy as np
import matplotlib.pyplot as plt
import skorch
import torch

import causalx as cx
import netx as nxx
import numpyx as npx
import skorchx
import torch.nn as nn
import torchx.nn as nnx


def gen():
    G = nxx.random_dag(10, 20)

    # M = nx.adjacency_matrix(G).A
    # nxx.draw(G)
    # plt.show()

    X = cx.IIDSimulation(method='linear', sem_type='gauss').fit(G).generate(10000)

    scaler = npx.MinMaxScaler(globally=True)
    # scaler = npx.NormalScaler(globally=True)
    X = scaler.fit_transform(X)

    return X


def main():

    X = gen().astype(np.float32)

    tmodule = nnx.LinearVAE(input_size=10, hidden_size=5, latent_size=3)
    # tmodule = nnx.Autoencoder(input_size=10, latent_size=3)

    early_stop = skorch.callbacks.EarlyStopping(patience=10, threshold=1e-4, monitor="valid_loss")

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=1000,
        optimizer=torch.optim.Adam,
        criterion=nnx.LinearVAELoss,
        # criterion=nn.MSELoss,
        lr=1.e-3,
        callbacks=[early_stop],
        batch_size=32,
        device='cpu'
    )
    smodel.set_params(callbacks__print_log=skorchx.callbacks.PrintLog(delay=1))

    smodel.fit(X, X)

    L = tmodule.encode(X)
    Y = tmodule.decode(L)

    plt.imshow(X[:20, :])
    plt.show()
    print(X.min(), X.max(), X.mean())
    plt.imshow(Y[:20, :])
    plt.show()
    print(Y.min(), Y.max(), Y.mean())
    pass


if __name__ == "__main__":
    main()
