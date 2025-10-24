import numpy as np
import pandas as pd


def main():
    m = np.zeros((4, 3))
    print(id(m))

    df = pd.DataFrame(m)

    v = df.values
    print(id(v))

    v[1,1] = 11
    v[2,2] = 22

    pass


if __name__ == "__main__":
    main()
