import matplotlib.pyplot as plt
import numpy as np
import skorch
import torch

import causalx as cx
import netx as nxx
import skorchx
import torchx.nn as nnx
import numpyx as npx
from mathx import sq


def gen():
    G = nxx.random_dag(10, 20)

    # M = nx.adjacency_matrix(G).A
    # nxx.draw(G)
    # plt.show()

    X = cx.IIDSimulation(method='linear', sem_type='gauss').fit(G).generate(1000)

    # scaler = npx.MinMaxScaler(globally=True)
    # scaler = npx.NormalScaler(globally=True)
    # X = scaler.fit_transform(X)

    return G, X.astype(np.float32)


def main():

    G, X = gen()
    mu = X.mean()
    sigma = X.std()

    tmodule = nnx.LinearVAE(input_size=10, hidden_size=10, latent_size=10)
    # tmodule = nnx.Autoencoder(input_size=10, latent_size=3)

    early_stop = skorch.callbacks.EarlyStopping(patience=10, threshold=0, monitor="valid_loss")

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=10000,
        optimizer=torch.optim.Adam,
        criterion=nnx.GaussianVAELoss,
        criterion__beta=1.,
        criterion__sigma=sigma,
        lr=1.e-3,
        callbacks=[early_stop],
        batch_size=32,
    )
    smodel.set_params(callbacks__print_log=skorchx.callbacks.PrintLog(delay=3))

    smodel.fit(X, X)

    L = tmodule.encode(X)
    Y = tmodule.decode(L)

    plt.imshow(X[:20, :])
    plt.show()
    plt.imshow(Y[:20, :])
    plt.show()

    mean = tmodule.mean.weight.data
    sigma = torch.exp(0.5*tmodule.log_var.weight.data)

    print(f"X:     {X.min():.3g}, {X.max():.3g}, {X.mean():.3g}")
    print(f"Y:     {Y.min():.3g}, {Y.max():.3g}, {Y.mean():.3g}")
    print(f"mean:  {mean.min():.3g}, {mean.max():.3g}, {mean.mean():.3g}")
    print(f"sigma: {sigma.min():.3g}, {sigma.max():.3g}, {sigma.mean():.3g}")
    pass


if __name__ == "__main__":
    main()
