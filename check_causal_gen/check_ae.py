import matplotlib.pyplot as plt
import skorch
import skorchx
import torch
import torch.nn as nn

import causalx as cx
import netx as nxx
import numpyx as npx
import torchx.nn as nnx


def gen():
    G = nxx.random_dag(10, 20)

    # M = nx.adjacency_matrix(G).A
    # nxx.draw(G)
    # plt.show()

    X = cx.IIDSimulation(method='linear', sem_type='gauss').fit(G).generate(100)

    scaler = npx.MinMaxScaler()
    X = scaler.fit_transform(X)

    return X


def main():

    X = gen()

    tmodule = nnx.Autoencoder(input_size=10, latent_size=5)

    early_stop = skorch.callbacks.EarlyStopping(patience=10, threshold=0.0, monitor="valid_loss")
    print_log = skorchx.callbacks.PrintLog()

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=5000,
        optimizer=torch.optim.Adam,
        criterion=nn.MSELoss,
        lr=0.001,
        callbacks=[early_stop],
        batch_size=100
    )
    smodel.set_params(callbacks__print_log=print_log)

    smodel.fit(X, X)

    L = tmodule.encode(X)
    Y = tmodule.decode(L)

    plt.imshow(X[:20, :])
    plt.show()
    plt.imshow(Y[:20, :])
    plt.show()
    pass


if __name__ == "__main__":
    main()
