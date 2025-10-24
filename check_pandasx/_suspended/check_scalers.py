import pandas as pd
import pandasx as pdx
from pandasx.preprocessing import MinMaxScaler


def main():
    df0 = pdx.read_data("Xy-100x2x3.csv", numeric=['y', 'x0', 'x1'])
    df0['y' ] = df0["y" ].apply(lambda x: 1.+x*2)
    df0['x0'] = df0["x0"].apply(lambda x: 1.+x*2)
    df0['x1'] = df0["x1"].apply(lambda x: 1.+x*2)

    s = MinMaxScaler(feature_range=(-1, 1))
    df1 = s.fit_transform(df0)
    df2 = s.inverse_transform(df1)

    pass


if __name__ == "__main__":
    main()
