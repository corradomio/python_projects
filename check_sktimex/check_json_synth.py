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
from stdlib import jsonx, create_from
from synth import create_syntethic_data
from joblib import Parallel, delayed

# Suppress all UserWarning instances
warnings.simplefilter("ignore", UserWarning)

TARGET = "y"


def replaces(s: str, tlist: list[str], r: str) -> str:
    for t in tlist:
        s = s.replace(t, r)
    return s



def create_fdir(name:str, jmodel: dict) -> str:
    # model = create_from(jmodel)
    # s = model.__class__.__module__
    # p1 = s.rfind(".")
    # p2 = s.rfind(".", 0, p1-1)

    # module = s[p2 + 1:].replace(".", "/")
    module = replaces(name, ["_", "-", "."], "/")

    fdir = f"plots/{module}/"
    os.makedirs(fdir, exist_ok=True)
    return fdir


def has_files(dirpath) -> bool:
    return len(os.listdir(dirpath)) > 0



def check_model(name, dfdict: dict[tuple, pd.DataFrame], jmodel: dict):
    if name.startswith("#"):
        return

    print("---", name, "---")
    fdir = create_fdir(name, jmodel)
    # if has_files(fdir):
    #     return

    for g in dfdict:
        print("...", g)
        try:
            dfg = dfdict[g]

            fname = f"{fdir}/{name}-{g[0]}.png"
            if os.path.exists(fname):
                continue

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
            # fname = f"{fdir}/{name}-{g[0]}.png"
            plt.savefig(fname, dpi=300)
            plt.close()

            # break
        except Exception as e:
            print("ERROR:", e)
            traceback.print_exception(*sys.exc_info())


def check_models(df: pd.DataFrame, jmodels: dict[str, dict]):
    dfdict = pdx.groups_split(df, groups=["cat"])

    for name in jmodels:
        check_model(name, dfdict, jmodels[name])

    # Parallel(n_jobs=6)(
    #     delayed(check_model)(name, dfdict, jmodels[name])
    #     for name in jmodels
    # )

    pass
# end




def main():
    print("dataframe")
    df = create_syntethic_data(12*8, 0.0, 1, 0.33)

    jmodels = jsonx.load("darts_models.json")
    check_models(df, jmodels)

    jmodels = jsonx.load("nf_models.json")
    check_models(df, jmodels)

    jmodels = jsonx.load("skx_models.json")
    check_models(df, jmodels)

    jmodels = jsonx.load("skt_models.json")
    check_models(df, jmodels)

    jmodels = jsonx.load("skl_models.json")
    check_models(df, jmodels)

    # jmodels = jsonx.load("ext_models.json")
    # check_models(df, jmodels)

    pass


if __name__ == "__main__":
    # logging.config.fileConfig('logging_config.ini')
    # logging.getLogger('root').info('Logging initialized')
    main()
