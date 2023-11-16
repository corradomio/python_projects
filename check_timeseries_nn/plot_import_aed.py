import logging
import os.path
import warnings
import pandasx as pdx
import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_series

# hide warnings
warnings.filterwarnings("ignore")

LOGGER = logging.getLogger("root")

DATA_DIR = "./data"
PLOTS_DIR = "D:/Projects.github/python_projects/check_timeseries_nn/plots"

DATETIME = ["imp_date", "[%Y/%m/%d %H:%M:%S]", "M"]
TARGET = "import_aed"
GROUPS = ['item_country']


def main():
    df_all = pdx.read_data(f"{DATA_DIR}/vw_food_import_aed_train_test.csv",
                       datetime=DATETIME,
                       ignore=GROUPS + DATETIME[0:1] + [
                           "imp_month",
                       ],
                       onehot=["imp_month"],
                       dropna=True,
                       na_values=['(null)'],
                       index=GROUPS + DATETIME[0:1]
                       )

    dfg = pdx.groups_split(df_all)

    i = 0
    n = len(dfg)
    for g in dfg:
        i += 1
        print(f"[{i:4}/{n}]", g)
        df = dfg[g]

        item = g[0].replace('/', '_')
        fname = f"{PLOTS_DIR}/import_aed/{item}.png"
        if os.path.exists(fname):
            continue

        plot_series(df[TARGET], labels=[TARGET], title=str(g))

        plt.savefig(fname, dpi=300)
        plt.tight_layout()
        plt.close()
    pass


if __name__ == "__main__":
    main()
