import numpy as np
import numpy.random
from numpyx.scalers import MinMaxScaler


def main():
    X = 5+np.random.randn(100, 10)*2

    s = MinMaxScaler(globally=True, outlier=3)

    S = s.fit_transform(X)
    O = s.inverse_transform(S)

    print((X-O).sum())

    pass


if __name__ == "__main__":
    main()

