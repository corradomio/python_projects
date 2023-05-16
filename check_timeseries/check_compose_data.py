import numpy as np
from torchx import compose_data


def main():
    X = np.array([[i*100 + j for j in range(1, 4)] for i in range(1, 100)])
    y = np.array([i for i in range(1001, 1100)])

    Xt, yt = compose_data(y, None, slots=[1, 2, 3])

    pass


if __name__ == "__main__":
    main()


