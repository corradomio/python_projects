import numpy as np
import numpyx as npx
from numpyx.scalers import NormalScaler, MinMaxScaler


def main():
    data = 100*np.random.random((100, 3))

    scaler = NormalScaler(params=(.5, .2), clip=[0, 1], globally=False)

    scaled = scaler.fit_transform(data)
    back = scaler.inverse_transform(scaled)

    print(np.linalg.norm(data-back))

    pass



if __name__ == "__main__":
    main()
