import logging
import os
import sys
import traceback
import warnings

import matplotlib.pyplot as plt
import pandas as pd

import pandasx as pdx
import sktimex as sktx
import sktimex.utils
from sktimex.forecasting import create_forecaster
from stdlib import jsonx
from stdlib.tprint import tprint
from synth import create_syntethic_data

# Suppress all UserWarning instances
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

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


def included(name, includes: list[str], excludes: list[str]) -> bool:
    assert isinstance(name, str)
    if includes is not None and len(includes) > 0:
        return name in includes
    if excludes is not None and len(excludes) > 0:
        return name not in excludes
    else:
        return True
# end


def save_params(name, cat, model):
    try:
        best_params = model.best_params_

        module = replaces(name, ["_", "-", "."], "/")
        if cat.endswith("-t"):
            fdir = f"best_params/{module}/"
        else:
            fdir = f"best_params/{module}/"

        os.makedirs(fdir, exist_ok=True)

        fname = f"{fdir}/{name}-{cat}.json"
        jsonx.dump(best_params, fname)
    except Exception as e:
        pass



def check_model(name, dfdict: dict[tuple, pd.DataFrame], jmodel: dict, override=False,
                data_includes=None, data_excludes=None):
    if name.startswith("#"):
        return

    print("---", name, "---")

    for g in dfdict:
        cat = g[0]
        if not included(cat, data_includes, data_excludes):
            continue

        try:
            dfg = dfdict[g]

            # ---------------------------------------------------------------

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

            # save params
            save_params(name, cat, model)

            # break
        except Exception as e:
            print(f"ERROR[{name}]:", e)
            traceback.print_exception(*sys.exc_info())
# end


def check_models(df: pd.DataFrame, jmodels: dict[str, dict], override=False,
                 model_includes=None, model_excludes=None,
                 data_includes=None, data_excludes=None,):
    dfdict = pdx.groups_split(df, groups=["cat"])

    # -- sequential
    for name in jmodels:
        if included(name, model_includes, model_excludes):
            check_model(name, dfdict, jmodels[name], override, data_includes, data_excludes)

    # -- parallel
    # Parallel(n_jobs=14)(
    #     delayed(check_model)(name, dfdict, jmodels[name])
    #     for name in jmodels if INCLUDED(name, includes, excludes)
    # )

    pass
# end


def main():
    tprint("dataframe")
    df = create_syntethic_data(12*8, 0.0, 1, 0.33)

    MODEL_INCLUDES = []
    MODEL_EXCLUDES = []
    DATA_INCLUDES = ["sin1","sin2","sin3","sin4","sin12"]
    DATA_EXCLUDES = []

    # tprint("config/darts_models.json")
    # jmodels = jsonx.load("config/darts_models.json")
    # check_models(df, jmodels, includes=INCLUDES, excludes=EXCLUDES)
    #
    # tprint("config/nf_models.json")
    # jmodels = jsonx.load("config/nf_models.json")
    # check_models(df, jmodels, includes=INCLUDES, excludes=EXCLUDES)
    #
    # tprint("config/skt_models.json")
    # jmodels = jsonx.load("config/skt_models.json")
    # check_models(df, jmodels, includes=INCLUDES, excludes=EXCLUDES)
    #
    # tprint("config/skl_models.json")
    # jmodels = jsonx.load("config/skl_models.json")
    # check_models(df, jmodels, includes=INCLUDES, excludes=EXCLUDES)
    #
    # tprint("config/skx_models.json")
    # jmodels = jsonx.load("config/skx_models.json")
    # check_models(df, jmodels, includes=INCLUDES, excludes=EXCLUDES)
    #
    # tprint("config/ext_models.json")
    # jmodels = jsonx.load("config/ext_models.json")
    # check_models(df, jmodels, override=True, includes=INCLUDES, excludes=EXCLUDES)

    tprint("config/auto_models.json")
    jmodels = jsonx.load("config/auto_models.json")
    check_models(df, jmodels,
                 model_includes=MODEL_INCLUDES, model_excludes=MODEL_EXCLUDES,
                 data_includes=DATA_INCLUDES, data_excludes=DATA_EXCLUDES,
                 override=False,
    )

    pass
# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
# end
