import os
import sys
import traceback
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

import pandasx as pdx
import sktimex as sktx
import sktimex.utils
from sktimex.forecasting import create_forecaster
from stdlib import jsonx
from stdlib.tprint import tprint
from stdlib.qname import create_from
from synth import create_syntethic_data

# Suppress all UserWarning instances
warnings.simplefilter("ignore", UserWarning)

TARGET = "y"


def replaces(s: str, tlist: list[str], r: str) -> str:
    for t in tlist:
        s = s.replace(t, r)
    return s
# end


def create_fdir(name:str, cat: str) -> str:
    module = replaces(name, ["_", "-", "."], "/")

    if cat.endswith("-t"):
        fdir = f"plots_trends/{module}/"
    else:
        fdir = f"plots_plain/{module}/"

    os.makedirs(fdir, exist_ok=True)
    return fdir
# end


def selected(name, included: list[str], excluded: list[str]) -> bool:
    if len(included) == 0 and len(excluded) == 0:
        return True

    for m in included:
        if m in name:
            return True
    for m in excluded:
        if m in name:
            return False
    if len(included) == 0 and len(excluded) == 0:
        return True
    elif len(included) > 0:
        return False
    else:
        return True
# end


def check_model(name, dfdict: dict[tuple, pd.DataFrame], jmodel: dict, override=False):
    if name.startswith("#"):
        return

    print("---", name, "---")

    for g in dfdict:
        try:
            dfg = dfdict[g]

            # ---------------------------------------------------------------

            cat = g[0]
            fdir = create_fdir(name, cat)

            fname = f"{fdir}/{name}-{cat}.png"
            if os.path.exists(fname) and not override:
                continue
            else:
                print("...", g)

            # ---------------------------------------------------------------

            X, y = pdx.xy_split(dfg, target=TARGET)
            X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=18)
            fh = y_test.index

            # print("... create")
            # model = create_from(jmodel)
            model = create_forecaster(name, jmodel)

            # print("... fit")
            model.fit(y=y_train, X=X_train)

            # print("... predict")
            y_predict = model.predict(fh=fh, X=X_test)
            # y_predict = y_predict + 0.01

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
            print(f"ERROR[{name}]:", e)
            traceback.print_exception(*sys.exc_info())
# end


def check_models(df: pd.DataFrame, jmodels: dict[str, dict], override=False, includes=[], excludes=[]):
    dfdict = pdx.groups_split(df, groups=["cat"])

    # -- sequential
    for name in jmodels:
        if selected(name, includes, excludes):
            check_model(name, dfdict, jmodels[name], override)

    # -- parallel
    # Parallel(n_jobs=14)(
    #     delayed(check_model)(name, dfdict, jmodels[name])
    #     for name in jmodels if selected(name, includes, excludes)
    # )

    pass
# end


def main():
    tprint("dataframe")
    df = create_syntethic_data(12*8, 0.0, 1, 0.33)

    SELECTED = []
    EXCLUDED = []

    # tprint("config/darts_models.json")
    # jmodels = jsonx.load("config/darts_models.json")
    # check_models(df, jmodels, includes=SELECTED, excludes=EXCLUDED)
    #
    # tprint("config/nf_models.json")
    # jmodels = jsonx.load("config/nf_models.json")
    # check_models(df, jmodels, includes=SELECTED, excludes=EXCLUDED)
    #
    # tprint("config/skt_models.json")
    # jmodels = jsonx.load("config/skt_models.json")
    # check_models(df, jmodels, includes=SELECTED, excludes=EXCLUDED)
    #
    # tprint("config/skl_models.json")
    # jmodels = jsonx.load("config/skl_models.json")
    # check_models(df, jmodels, includes=SELECTED, excludes=EXCLUDED)
    #
    # tprint("config/skx_models.json")
    # jmodels = jsonx.load("config/skx_models.json")
    # check_models(df, jmodels, includes=SELECTED, excludes=EXCLUDED)
    #
    # tprint("config/ext_models.json")
    # jmodels = jsonx.load("config/ext_models.json")
    # check_models(df, jmodels, override=True, includes=SELECTED, excludes=EXCLUDED)

    tprint("config/auto_models.json")
    jmodels = jsonx.load("config/auto_models.json")
    check_models(df, jmodels, override=True, includes=SELECTED, excludes=EXCLUDED)

    pass
# end


if __name__ == "__main__":
    # logging.config.fileConfig('logging_config.ini')
    # logging.getLogger('root').info('Logging initialized')
    main()
# end
