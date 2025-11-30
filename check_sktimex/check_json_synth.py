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
from stdlib import jsonx, create_from
from synth import create_syntethic_data

# Suppress all UserWarning instances
warnings.simplefilter("ignore", UserWarning)

TARGET = "y"


def replaces(s: str, tlist: list[str], r: str) -> str:
    for t in tlist:
        s = s.replace(t, r)
    return s
# end


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
    fdir = create_fdir(name, jmodel)

    for g in dfdict:
        try:
            dfg = dfdict[g]

            fname = f"{fdir}/{name}-{g[0]}.png"
            if os.path.exists(fname) and not override:
                continue
            else:
                print("...", g)

            X, y = pdx.xy_split(dfg, target=TARGET)
            X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=18)

            # print("... create")
            model = create_from(jmodel)

            # print("... fit")
            model.fit(y=y_train, X=X_train)

            # print("... predict")
            fh = y_test.index
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
            print("ERROR:", e)
            traceback.print_exception(*sys.exc_info())
# end


def check_models(df: pd.DataFrame, jmodels: dict[str, dict], override=False, includes=None, excludes=None):
    dfdict = pdx.groups_split(df, groups=["cat"])

    # for name in jmodels:
    #     if selected(name, includes, excludes):
    #         check_model(name, dfdict, jmodels[name], override)

    Parallel(n_jobs=14)(
        delayed(check_model)(name, dfdict, jmodels[name])
        for name in jmodels if selected(name, includes, excludes)
    )

    pass
# end


def main():
    print("dataframe")
    df = create_syntethic_data(12*8, 0.0, 1, 0.33)

    SELECTED = []
    EXCLUDED = ["ESRNN", "FED"]

    jmodels = jsonx.load("darts_models.json")
    check_models(df, jmodels, includes=SELECTED, excludes=EXCLUDED)

    jmodels = jsonx.load("nf_models.json")
    check_models(df, jmodels, includes=SELECTED, excludes=EXCLUDED)

    jmodels = jsonx.load("skx_models.json")
    check_models(df, jmodels, includes=SELECTED, excludes=EXCLUDED)

    jmodels = jsonx.load("skt_models.json")
    check_models(df, jmodels, includes=SELECTED, excludes=EXCLUDED)

    jmodels = jsonx.load("skl_models.json")
    check_models(df, jmodels, includes=SELECTED, excludes=EXCLUDED)

    # jmodels = jsonx.load("ext_models.json")
    # check_models(df, jmodels, override=True)

    pass
# end


if __name__ == "__main__":
    # logging.config.fileConfig('logging_config.ini')
    # logging.getLogger('root').info('Logging initialized')
    main()
# end
