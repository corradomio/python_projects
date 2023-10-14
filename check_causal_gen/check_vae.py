import skorch
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

    X = cx.IIDSimulation(method='linear', sem_type='gauss').fit(G).generate(2000)

    scaler = npx.MinMaxScaler()
    X = scaler.fit_transform(X)

    return X


def main():

    X = gen()

    tmodule = nnx.LinearVAE(10, 3)

    early_stop = skorch.callbacks.EarlyStopping(patience=12, threshold=0.0001, monitor="valid_loss")

    smodel = skorch.NeuralNetRegressor(
        module=tmodule,
        max_epochs=1000,
        optimizer=torch.optim.Adam,
        criterion=nn.KLDivLoss,
        lr=0.0005,
        callbacks=[early_stop],
        batch_size=6
    )

    smodel.fit(X, X)
# end


if __name__ == "__main__":
    main()
