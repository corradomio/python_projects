import logging.config
import os
import warnings

import matplotlib.pyplot as plt
import skorch
import torch

import pandasx as pdx
import skorchx
import sktimex
import sktimex.transform
from sktimex.utils.plotting import plot_series
from torchx.nn.timeseries import *

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
    name = g[0].replace('/', '-')

    xlags = range(1, 24)
    ylags = range(1, 24)
    tlags = range(2)

    X, y = pdx.xy_split(df, target=TARGET)

    X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=12)
    fh = len(y_test)

    # plot_series(y_train, y_test, labels=['train', 'test'], title=g[0])
    # plt.show()

    # xscaler = pdx.preprocessing.MinMaxScaler(method='poly1', sp=12, tau=2)
    xscaler = pdx.preprocessing.MinMaxScaler()
    X_train_s = xscaler.fit_transform(X_train)
    X_test_s = xscaler.transform(X_test)

    # yscaler = pdx.preprocessing.MinMaxScaler(method='poly1', sp=12, tau=2)
    yscaler = pdx.preprocessing.MinMaxScaler()
    y_train_s = yscaler.fit_transform(y_train)
    y_test_s = yscaler.transform(y_test)

    # plot_series(y_train_s, y_test_s, labels=['train', 'test'], title=g[0])
    # plt.show()

    # compute the input/output shapes based on [X|y] train and [x|y|t] lags
    input_shape, output_shape = sktimex.transform.compute_input_output_shapes(X_train, y_train, xlags, ylags, tlags)

    # prepare the data

    # compose Xy_past [*, xylags, |X|+|y|], y_past, X_future, y_future
    tt = sktimex.LagsTrainTransform(xlags=xlags, ylags=ylags, tlags=tlags, concat=True)
    all = tt.fit_transform(y=y_train_s, X=X_train_s)
    # Xt, _, Xf, yt = tt.fit_transform(y=y_train_s, X=X_train_s)
    pass

    # compose y_pred
    pt = sktimex.LagsPredictTransform(xlags=xlags, ylags=ylags, tlags=tlags, concat=True)
    y_pred_s = pt.fit(y=y_train_s, X=X_train_s).transform(fh=fh, X=X_test_s)

    #
    # Model
    #
    # MODEL = 'attn1'
    # MODEL = "tide1"
    MODEL = "tide1"

    os.makedirs(f"./plots/{MODEL}/", exist_ok=True)
    fname = f"./plots/{MODEL}/{name}.png"
    # if os.path.exists(fname):
    #     return

    tsmodel = create_model(
        MODEL, input_shape, output_shape,
        num_encoder_layers=1,
        num_decoder_layers=1,
        # use_layer_norm=False,
        use_future_features=False
    )

    #
    # End
    #

    early_stop = skorchx.callbacks.EarlyStopping(warmup=50, patience=25, threshold=0, monitor="valid_loss")

    model = skorch.NeuralNetRegressor(
        module=tsmodel,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.MSELoss,
        batch_size=12,
        max_epochs=5000,
        lr=0.005,
        callbacks=[early_stop],
        # callbacks__print_log=PrintLog
    )


    # tide1
    model.fit((Xt, Xf), yt)

    # tide2
    # model.fit(Xt, yt)

    i = 0
    while i < fh:
        # tide1
        # Xp, _, Xf = pt.step(i)
        # y_pred = model.predict((Xp, Xf))

        # tide2
        Xp, _, _ = pt.step(i)
        y_pred = model.predict(Xp)

        i = pt.update(i, y_pred)
    # end

    y_pred = yscaler.inverse_transform(y_pred_s)

    plot_series(y_train, y_test, y_pred, labels=['train', 'test', 'pred'])

    # name = g[0].replace('/', '-')
    # os.makedirs(f"./plots/{MODEL}/", exist_ok=True)
    # fname = f"./plots/{MODEL}/{name}.png"
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


