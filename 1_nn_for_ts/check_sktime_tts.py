import logging.config
import os
import warnings

import matplotlib.pyplot as plt

import pandasx as pdx
import sktimex.forecasting.tts as skts
import sktimex.forecasting.tts.tide
from sktimex.forecasting.base import ForecastingHorizon
from sktimex.utils.plotting import plot_series

DATA_DIR = "./data"
DATETIME = ["imp_date", "[%Y/%m/%d %H:%M:%S]", "M"]
TARGET = "import_kg"
GROUPS = ['item_country']


def load_data():
    os.makedirs('./plots/import_kg/trend', exist_ok=True)

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

    return df_all
# end


def analyze(g, df):
    print(g)
    gname = g[0].replace('/', '-')

    X, y = pdx.xy_split(df, target=TARGET)

    X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=12)
    fh = ForecastingHorizon(y_test.index)

    # -- scale the data

    xscaler = pdx.preprocessing.MinMaxScaler()
    X_train_s = xscaler.fit_transform(X_train)
    X_test_s = xscaler.transform(X_test)

    yscaler = pdx.preprocessing.MinMaxScaler()
    y_train_s = yscaler.fit_transform(y_train)
    y_test_s = yscaler.transform(y_test)

    # -- create the model, train, predict
    MODEL = 'sktimex.tide'

    model = skts.tide.TiDE(lags=24, tlags=2)

    model.fit(y=y_train_s, X=X_train_s)

    y_pred_s = model.predict(fh=fh, X=X_test_s)
    # --

    # invert the scaling in y predicted
    y_pred = yscaler.inverse_transform(y_pred_s)

    plot_series(y_train, y_test, y_pred, labels=['train', 'test', 'pred'])

    # -- plot the prediction
    os.makedirs(f"./plots/{MODEL}/", exist_ok=True)
    fname = f"./plots/{MODEL}/{gname}.png"
    plt.savefig(fname, dpi=300)

    pass
# end


def main():
    df_all = load_data()

    dfg = pdx.groups_split(df_all)
    n = 0
    for g in sorted(dfg.keys()):
        analyze(g, dfg[g])
        n += 1
        if n >= 3: break
    pass
# end


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()


