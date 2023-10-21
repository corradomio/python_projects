import numpy as np
import numpyx as npx


def main():
    d0 = np.random.uniform(0, 1, size=(100, 10))

    sc1 = npx.scalers.MinMaxScaler()
    d1 = sc1.fit_transform(d0)
    d2 = sc1.inverse_transform(d1)
    print("minmax by column", np.linalg.norm(d0-d2))

    sc1 = npx.scalers.MinMaxScaler(globally=True)
    d1 = sc1.fit_transform(d0)
    d2 = sc1.inverse_transform(d1)
    print("minmax globally ", np.linalg.norm(d0-d2))

    sc1 = npx.scalers.NormalScaler()
    d1 = sc1.fit_transform(d0)
    d2 = sc1.inverse_transform(d1)
    print("normal by column", np.linalg.norm(d0-d2))

    sc1 = npx.scalers.NormalScaler(globally=True)
    d1 = sc1.fit_transform(d0)
    d2 = sc1.inverse_transform(d1)
    print("normal globally ", np.linalg.norm(d0-d2))

    pass



if __name__ == "__main__":
    main()