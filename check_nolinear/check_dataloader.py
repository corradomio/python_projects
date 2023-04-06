import lightning
import numpy as np


def main():
    x = np.arange(-6.28, +6.28, 0.01, dtype=float)
    y = np.sin(x, dtype=float)

    dl = lightning.NumpyDataloader(x, y, batch_size=32)

    for X, y in dl:
        print(X.shape, y.shape)
    pass


if __name__ == "__main__":
    main()
