import os
import pandas as pd
import numpy as np
import pandasx as pdx
import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_series


DATA_DIR = "../data"
PLOTS_DIR = "./plots/import_kg"

def name_of(n):
    return n.replace('/', '-').replace(':', '-')


def analyze(df: pd.DataFrame, g):
    name = g[0]
    fname = f"{PLOTS_DIR}/{name_of(name)}.png"
    if os.path.exists(fname):
        return

    print("...", name)
    target = df['import_kg']
    plot_series(target, labels=["import_kg"], title=name)
    plt.savefig(fname, dpi=300)
    plt.close()


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

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
        analyze(dfdict[g], g)


if __name__ == "__main__":
    main()
