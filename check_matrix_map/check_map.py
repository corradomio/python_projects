import numpy as np
import torch.nn as nn
import torch
import skorch
import skorchx


class MapModel(nn.Module):

    def __init__(self, size, L=1):
        super().__init__()
        list = []
        for i in range(L):
            list.append(nn.Linear(in_features=size, out_features=size))
            # list.append(nn.Tanh())

        self.lin = nn.Sequential(*list)

    def forward(self, x):
        return self.lin(x)


def main():
    S = 10
    N = 1000
    L = 4

    Xt = np.random.rand(N, S).astype(np.float32)
    yt = np.random.rand(N, S).astype(np.float32)

    mapm = MapModel(S, L)

    early_stop = skorchx.callbacks.EarlyStopping(warmup=50, patience=10, threshold=0, monitor="valid_loss")

    model = skorch.NeuralNetRegressor(
        module=mapm,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.MSELoss,
        batch_size=32,
        max_epochs=1000,
        lr=0.001,
        callbacks=[early_stop],
    )

    model.fit(Xt, yt)

    yp = model.predict(Xt)

    print(np.linalg.norm(yt-yp, 'fro')/N)

    print(yt[:10, :5])
    print('---')
    print(yp[:10, :5])

    pass



if __name__ == "__main__":
    main()
