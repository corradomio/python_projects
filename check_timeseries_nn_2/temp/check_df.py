import numpy as np
import pandas as pd


def main():
    N = 100
    M = 10
    data = np.random.random(size=(N, M))
    columns = [f"c{c:02}" for c in range(M)]
    index = pd.period_range("2023-10-04", periods=N)

    df = pd.DataFrame(data=data, columns=columns, index=index)
    ar = df.values

    for i in range(N):
        for j in range (M):
            df.iloc[i, j] = i*M + j
    pass


if __name__ == "__main__":
    main()
