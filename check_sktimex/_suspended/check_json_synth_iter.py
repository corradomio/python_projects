import logging.config
import os
import sys
import traceback
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pandasx as pdx
import sktimex as sktx
import sktimex.utils
from stdlib import jsonx
from stdlib.qname import create_from
from sktime.forecasting.arch import ARCH
from synth import create_syntethic_data

# Suppress all UserWarning instances
warnings.simplefilter("ignore", UserWarning)

TARGET = "y"


def create_fdir(jmodel: dict) -> str:
    model = create_from(jmodel)
    s = model.__class__.__module__
    p1 = s.rfind(".")
    p2 = s.rfind(".", 0, p1-1)
    module = s[p2 + 1:].replace(".", "/")

    fdir = f"plots/{module}"
    os.makedirs(fdir, exist_ok=True)
    return fdir


def check_models(df: pd.DataFrame, jmodels: dict[str, dict]):
    dfdict = pdx.groups_split(df, groups=["cat"])

    for name in jmodels:
        if name.startswith("#"):
            continue

        jmodel = jmodels[name]

        print("---", name, "---")
        fdir = create_fdir(jmodel)

        for g in dfdict:
            print("...", g)
            try:
                dfg = dfdict[g]

                X, y = pdx.xy_split(dfg, target=TARGET)
                X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=18)

                # print("... create")
                model = create_from(jmodel)

                # print("... fit")
                model.fit(y=y_train, X=X_train)

                # print("... predict")
                fh = y_test.index
                y_predict = model.predict(fh=fh, X=X_test)

                # print("... plot")
                sktx.utils.plot_series(y_train, y_test, y_predict,
                                       labels=["train", "test", "predict"],
                                       title=f"{name}: {g[0]}")

                # save plot
                fname = f"{fdir}/{name}-{g[0]}.png"
                plt.savefig(fname, dpi=300)
                plt.close()

                # break
            except Exception as e:
                print("ERROR:", e)
                traceback.print_exception(*sys.exc_info())
        # end
    pass
# end


def StatsForecastGARCH_iter(df):

    jmodels = {}

    for p in [1,2,3,4,6,12]:
        for q in [1,2,3,4,6,12]:
            name = f"StatsForecastGARCH_{p}_{q}"
            jmodels[name] = {
                "class": "sktime.forecasting.arch.StatsForecastGARCH",
                "p": p,
                "q": q
            }

    check_models(df, jmodels)


def ARCH_iter(df):

    jmodels = {}

    for mean in [ 'LS', 'AR', 'ARX', 'HAR', 'HARX']:                            # 5
        for lags in [3,6,12]:                                                   # 3
            for vol in ['GARCH', 'ARCH', 'EGARCH', 'FIARCH' 'HARCH']:           # 5
                for p in [3,6,12]:                                              # 3
                    for o in [3,6,12]:                                          # 3
                        for q in [3,6,12]:                                      # 3
                            for dist in ['normal', 't', 'skewt', 'ged']:        # 4
                                name = f"ARCH_{mean}_{vol}_{p}_{o}_{q}_{dist}"
                                jmodels[name] = {
                                    # "class": "sktimext.forecasting.arch.ARCH",
                                    "class": "sktime.forecasting.arch.ARCH",
                                    "mean": mean,
                                    "lags": lags,
                                    "vol": vol,
                                    "p": p,
                                    "o": o,
                                    "q": q,
                                    "dist": dist
                                }

    check_models(df, jmodels)
# end


def main():
    print("dataframe")
    df = create_syntethic_data(12*8, 0.0, 1, 0.33)

    # StatsForecastGARCH_iter(df)
    ARCH_iter(df)
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
