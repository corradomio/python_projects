import pandas as pd
import numpy as np
import pandasx as pdx
import matplotlib.pyplot as plt


DATA_DIR = "../data"


def analyze(df: pd.DataFrame):
    columns = df.columns
    for c in columns:
        data: np.ndarray = df[c].to_numpy()
        u = np.unique(data)
        if len(u) == 2:
            continue
        fft = np.absolute(np.fft.fft(data))
        N = 21
        x = np.arange(0, N)
        plt.scatter(x, fft[0:N])
        plt.title(c)
        plt.show()
        pass
    pass


def main():
    df = pdx.read_data(f"{DATA_DIR}/vw_food_import_train_test_newfeatures.csv",
                       datetime=['imp_date', "[%Y/%m/%d %H:%M:%S]", "M"],
                       ignore=['item_country', 'imp_date'] + [
                           "imp_month",
                           "prod_kg",
                           "avg_retail_price_src_country",
                           "producer_price_tonne_src_country",
                           "min_temperature",
                           "max_temperature",

                           # "crude_oil_price",
                           # "sandp_500_us",
                           # "sandp_sensex_india",
                           # "shenzhen_index_china",
                           # "nikkei_225_japan",

                           # "mean_temperature",
                           "vap_pressure",
                           "evaporation",
                           "rainy_days",
                       ],
                       onehot=["imp_month"],
                       dropna=True,
                       na_values=['(null)'],
                       index=['item_country', 'imp_date']
                       )

    dfdict = pdx.groups_split(df)

    for g in dfdict:
        analyze(dfdict[g])


if __name__ == "__main__":
    main()
