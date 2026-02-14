import logging
import os
import sys
import traceback
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from filelock import FileLock
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError, MeanSquaredScaledError, \
    MeanAbsoluteScaledError

import pandasx as pdx
import sktimex as sktx
import sktimex.utils
from joblibx import Parallel, delayed
from sktimex.forecasting import create_forecaster
from sklearn.metrics import r2_score
from stdlib import jsonx
from stdlib.tprint import tprint
from synth import create_syntethic_data
from load_config import load_model_selection_config

# Suppress all UserWarning instances
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

TARGET = "y"
N_JOBS = 12

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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
    best_params = model.best_params_

    module = replaces(name, ["_", "-", "."], "/")
    if cat.endswith("-t"):
        fdir = f"best_params/{module}/"
    else:
        fdir = f"best_params/{module}/"

    os.makedirs(fdir, exist_ok=True)

    fname = f"{fdir}/{name}-{cat}.json"
    jsonx.dump(best_params, fname)
# end


def save_scores(name, cat, scores):
    scores_file = "models_scores.csv"
    lock_file = scores_file + ".lock"
    lock = FileLock(lock_file)
    with lock:
        if not os.path.exists(scores_file):
            with open(scores_file, "w") as f:
                meas_names = ",".join(scores.keys())
                f.writelines("model,cat," + meas_names + "\n")
        with open(scores_file, "a") as f:
            values = ",".join(map(str, scores.values()))
            f.writelines(f"{name},{cat},{values}\n")
# end


# ---------------------------------------------------------------------------
# check_model
# ---------------------------------------------------------------------------

def check_model_par(*args, **kwargs):

    # it is necessary to configure the logging system inside each
    # python process, when it is used the joblib
    logging.config.fileConfig('logging_config.ini')

    check_model_cat(*args, **kwargs)
# end


def check_model_cat(
        name: str,
        cat: str,
        jmodel: dict
):
    # 1) chech if to analize the timeseries (cat is excluded or not included)
    cats_excluded = []
    cats_included = []

    jforecaster = jmodel["forecaster"]
    if "data_includes" in jforecaster:
        cats_included += jforecaster["data_includes"]
        del jforecaster["data_includes"]
    if "data_excludes" in jforecaster:
        cats_excluded += jforecaster["data_excludes"]
        del jforecaster["data_excludes"]

    if not included(cat, cats_included, cats_excluded):
        return

    # 2) check if the time series is already analyzed (the plot is present)
    fdir = create_fdir(name, cat)
    fname = f"{fdir}/{name}-{cat}.png"
    if os.path.exists(fname):
        return

    # 3) create the dataset (not very efficient, but is it not a big problem)
    df = create_syntethic_data(12 * 8, 0.0, 1, 0.33)
    dfdict = pdx.groups_split(df, groups=["cat"])
    dfg = dfdict[(cat,)]

    # 4) evaluate the model
    print("---", name, "/", cat, "---")
    try:
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

        # save params
        save_params(name, cat, model)

        # save scores
        save_scores(name, cat, {
            "mae": MeanAbsoluteError()(y_test, y_predict),
            "mase": MeanAbsoluteScaledError()(y_test, y_predict),
            "mse": MeanSquaredError()(y_test, y_predict),
            "r2": r2_score(y_test, y_predict),
        })

        # print("... plot")
        sktx.utils.plot_series(y_train, y_test, y_predict,
                               labels=["train", "test", "predict"],
                               title=f"{name}: {cat}")

        # save plot
        plt.savefig(fname, dpi=300)
        plt.close()

        # break
    except Exception as e:
        print(f"ERROR[{name}]:", e)
        traceback.print_exception(*sys.exc_info())
# end


def check_models(cats: list[str], jmodels: dict[str, dict]):

    # -- sequential
    # for name in jmodels:
    #     for cat in cats:
    #         check_model_cat(name, cat, jmodels[name])

    # -- parallel
    Parallel(n_jobs=N_JOBS)(
        delayed(check_model_par)(name, cat, jmodels[name])
        for name in jmodels for cat in cats
    )

    pass
# end


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():

    tprint("dataframe")
    df = create_syntethic_data(12 * 8, 0.0, 1, 0.33)
    cats = df["cat"].unique().tolist()

    tprint("auto_dartsx_models")
    jmodels = load_model_selection_config("config/auto_dartsx_models.json")
    jsonx.dump(jmodels, f"config/resolved_auto_dartsx_models.json")

    # tprint("auto_nfx_models")
    # jmodels = load_model_selection_config("config/auto_nfx_models.json")
    # jsonx.dump(jmodels, f"config/resolved_auto_nfx_models.json")

    check_models(cats, jmodels)
    pass
# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
# end
