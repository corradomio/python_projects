import os
import logging.config
import warnings
import matplotlib.pyplot as plt
import pandasx as pdx

from sktimex.utils import plot_series
from pandasx.preprocessing import MinMaxScaler, DetrendTransformer

# hide warnings
warnings.filterwarnings("ignore")

LOGGER = logging.getLogger("root")

DATA_DIR = "./data"
PLOTS_DIR = "D:/Projects.github/python_projects/check_timeseries_nn/plots"

DATETIME = ["imp_date", "[%Y/%m/%d %H:%M:%S]", "M"]
TARGET = "import_kg"
GROUPS = ['item_country']


MINMAX_METHODS = ['identity', 'linear', 'piecewise', 'stepwise', 'global', 'poly1', 'power', 0.8]
TREND_METHODS  = ['identity', 'linear', 'piecewise', 'stepwise', 'global', 'poly1', 'power']


def plot_anonymized_ts(df_all):
    os.makedirs(f'./plots/import_kg/anonts', exist_ok=True)

    tsid = 0

    dfg = pdx.groups_split(df_all)
    for g in sorted(dfg.keys()):
        print(g)

        tsid += 1
        tsname = g[0].replace('/', '-')
        fname = f'./plots/import_kg/anonts/{tsname}.png'
        if os.path.exists(fname):
            continue

        df = dfg[g]
        y = df[TARGET]
        y.name = 'target'

        plot_series(y, labels=['target'], title=f"TS-{tsid}")
        plt.savefig(fname)
        plt.close()


def main():
    os.makedirs('./plots/import_kg/minmax', exist_ok=True)
    os.makedirs('./plots/import_kg/detrend', exist_ok=True)

    df_all = pdx.read_data(
        f"{DATA_DIR}/vw_food_import_kg_train_test.csv",
        datetime=DATETIME,
        ignore=GROUPS + DATETIME[0:1] + [
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
        index=GROUPS + DATETIME[0:1]
    )

    plot_anonymized_ts(df_all)

    pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
